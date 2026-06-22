"""Sequential VLM-plan, Falcon-segment, VLM-reason pipeline."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from visualcue.harness.systems._falcon import DEFAULT_DEVICE, DEFAULT_MODEL_ID, FalconSegmenter
from visualcue.harness.systems._vlm import VLMClient
from visualcue.harness.types import Instance, SystemOutput

PLAN_SYSTEM_PROMPT = (
    "You receive an image and a user request. Decide whether the user wants to COUNT objects or "
    "to LOCATE a specific object. Produce a concise segmentation prompt naming the object class "
    "to segment. Respond ONLY as JSON: "
    '{"intent": "count"|"locate", "segmentation_prompt": "<noun phrase>"}.'
)
COUNT_REASON_SYSTEM_PROMPT = (
    "You receive an image with marked segmentation candidates. Answer the user's counting request. "
    "Respond ONLY as JSON: {\"count\": <int>, \"answer\": \"<text>\"}."
)
LOCATE_REASON_SYSTEM_PROMPT = (
    "You receive an image with numbered candidate masks and bounding boxes. Select the candidate "
    "indices that satisfy the user's request. Respond ONLY as JSON: "
    '{"selected": [<indices>], "answer": "<text>"}.'
)
OVERLAY_COLORS = np.asarray(
    [
        [255, 0, 0],
        [0, 180, 255],
        [0, 220, 120],
        [255, 190, 0],
        [190, 0, 255],
        [255, 80, 160],
        [120, 255, 0],
        [0, 120, 255],
    ],
    dtype=np.float32,
)
OVERLAY_ALPHA = 0.45


class SequentialPipeline:
    """One-pass plan, segment, reason pipeline."""

    name = "sequential_pipeline"

    def __init__(
        self,
        vlm_base_url: str = "http://localhost:1234/v1",
        vlm_model: str = "gemma",
        vlm_api_key: str = "lm-studio",
        falcon_model_id: str = DEFAULT_MODEL_ID,
        falcon_device: str = DEFAULT_DEVICE,
        enable_reasoning: bool = True,
        free_falcon_between_calls: bool = False,
        plan_system_prompt: str = PLAN_SYSTEM_PROMPT,
        count_reason_system_prompt: str = COUNT_REASON_SYSTEM_PROMPT,
        locate_reason_system_prompt: str = LOCATE_REASON_SYSTEM_PROMPT,
        vlm: VLMClient | None = None,
        segmenter: FalconSegmenter | None = None,
    ) -> None:
        self.vlm_model = vlm_model
        self.falcon_model_id = falcon_model_id
        self.vlm = vlm or VLMClient(base_url=vlm_base_url, model=vlm_model, api_key=vlm_api_key)
        self.segmenter = segmenter or FalconSegmenter(model_id=falcon_model_id, device=falcon_device)
        self.enable_reasoning = enable_reasoning
        self.free_falcon_between_calls = free_falcon_between_calls
        self.plan_system_prompt = plan_system_prompt
        self.count_reason_system_prompt = count_reason_system_prompt
        self.locate_reason_system_prompt = locate_reason_system_prompt

    def config(self) -> dict[str, Any]:
        return {
            "vlm_model": self.vlm_model,
            "enable_reasoning": self.enable_reasoning,
            "plan_system_prompt": self.plan_system_prompt,
            "count_reason_system_prompt": self.count_reason_system_prompt,
            "locate_reason_system_prompt": self.locate_reason_system_prompt,
            "falcon_model_id": self.falcon_model_id,
        }

    def run(
        self,
        image: Image.Image,
        query: str,
        gold_targets: list[str] | None = None,
    ) -> SystemOutput:
        """Run exactly one plan-segment-reason pass; runner overwrites latency."""

        rgb_image = image.convert("RGB") if image.mode != "RGB" else image
        intermediate: dict[str, Any] = {}
        if gold_targets is not None:
            intent = "locate"
            segmentation_prompt = ", ".join(gold_targets) if gold_targets else query
            intermediate["plan_skipped_gold_targets"] = True
        else:
            intent, segmentation_prompt = self._plan(rgb_image, query, intermediate)

        if self.free_falcon_between_calls:
            self.segmenter.to_device()
        instances = self.segmenter.segment(rgb_image, segmentation_prompt)
        if self.free_falcon_between_calls:
            self.segmenter.to_cpu()

        intermediate.update(
            {
                "intent": intent,
                "segmentation_prompt": segmentation_prompt,
                "n_candidates": len(instances),
            }
        )
        final_instances, count, answer = self._reason_or_fallback(rgb_image, query, intent, segmentation_prompt, instances, intermediate)
        return SystemOutput(
            instances=final_instances,
            answer=answer,
            count=count,
            latency_ms=0.0,
            intermediate=intermediate,
        )

    def _plan(self, image: Image.Image, query: str, intermediate: dict[str, Any]) -> tuple[str, str]:
        raw = self.vlm.complete(self.plan_system_prompt, query, image=image)
        intermediate["plan_raw"] = raw
        try:
            parsed = _parse_json_object(raw)
            intent = str(parsed.get("intent", "locate")).lower()
            if intent not in {"count", "locate"}:
                raise ValueError(f"invalid intent: {intent}")
            segmentation_prompt = str(parsed["segmentation_prompt"])
            return intent, segmentation_prompt
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            intermediate["plan_parse_fallback"] = True
            intermediate["plan_parse_error"] = str(exc)
            return "locate", query

    def _reason_or_fallback(
        self,
        image: Image.Image,
        query: str,
        intent: str,
        segmentation_prompt: str,
        instances: list[Instance],
        intermediate: dict[str, Any],
    ) -> tuple[list[Instance], int, str]:
        if not self.enable_reasoning:
            count = len(instances)
            return instances, count, str(count)
        if intent == "count":
            return self._reason_count(image, query, segmentation_prompt, instances, intermediate)
        return self._reason_locate(image, query, segmentation_prompt, instances, intermediate)

    def _reason_count(
        self,
        image: Image.Image,
        query: str,
        segmentation_prompt: str,
        instances: list[Instance],
        intermediate: dict[str, Any],
    ) -> tuple[list[Instance], int, str]:
        overlay = render_overlay(image, instances)
        user_text = (
            f"Original request: {query}\n"
            f"The segmenter marked {len(instances)} region(s) matching '{segmentation_prompt}'."
        )
        raw = self.vlm.complete(self.count_reason_system_prompt, user_text, image=overlay)
        intermediate["reason_raw"] = raw
        try:
            parsed = _parse_json_object(raw)
            count = int(parsed["count"])
            answer = str(parsed.get("answer", count))
            return instances, count, answer
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            intermediate["reason_parse_fallback"] = True
            intermediate["reason_parse_error"] = str(exc)
            count = len(instances)
            return instances, count, str(count)

    def _reason_locate(
        self,
        image: Image.Image,
        query: str,
        segmentation_prompt: str,
        instances: list[Instance],
        intermediate: dict[str, Any],
    ) -> tuple[list[Instance], int, str]:
        overlay = render_overlay(image, instances)
        user_text = (
            f"Original request: {query}\n"
            f"Segmentation prompt: {segmentation_prompt}\n"
            "Candidates:\n"
            + "\n".join(_candidate_text(index, instance) for index, instance in enumerate(instances))
        )
        raw = self.vlm.complete(self.locate_reason_system_prompt, user_text, image=overlay)
        intermediate["reason_raw"] = raw
        try:
            parsed = _parse_json_object(raw)
            selected = [int(index) for index in parsed["selected"]]
            final_instances = [instances[index] for index in selected if 0 <= index < len(instances)]
            answer = str(parsed.get("answer", len(final_instances)))
            return final_instances, len(final_instances), answer
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            intermediate["reason_parse_fallback"] = True
            intermediate["reason_parse_error"] = str(exc)
            count = len(instances)
            return instances, count, str(count)


def render_overlay(image: Image.Image, instances: list[Instance]) -> Image.Image:
    """Render numbered mask candidates without pixel-wise Python loops."""

    base = np.asarray(image.convert("RGB")).astype(np.float32)
    for index, instance in enumerate(instances):
        if instance.mask is None:
            continue
        mask = instance.mask.astype(bool, copy=False)
        color = OVERLAY_COLORS[index % len(OVERLAY_COLORS)]
        base[mask] = (1.0 - OVERLAY_ALPHA) * base[mask] + OVERLAY_ALPHA * color
    overlay = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    for index, instance in enumerate(instances):
        anchor = _label_anchor(instance)
        if anchor is not None:
            draw.text(anchor, str(index), fill=(255, 255, 255))
    return overlay


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = _strip_code_fence(raw.strip())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise TypeError("expected JSON object")
    return parsed


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _candidate_text(index: int, instance: Instance) -> str:
    return f"{index}: bbox={instance.bbox} label={instance.label}"


def _label_anchor(instance: Instance) -> tuple[int, int] | None:
    if instance.bbox is not None:
        return (int(instance.bbox[0]), int(instance.bbox[1]))
    if instance.mask is None:
        return None
    ys, xs = np.where(instance.mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()))
