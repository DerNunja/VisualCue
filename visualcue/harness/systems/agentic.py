"""Agentic VLM-plan, Falcon-segment/refine-loop, VLM-reason pipeline."""

from __future__ import annotations

import json
from typing import Any

from PIL import Image

from visualcue.harness.systems._falcon import DEFAULT_DEVICE, DEFAULT_MODEL_ID, FalconSegmenter
from visualcue.harness.systems._pipeline import (
    COUNT_REASON_SYSTEM_PROMPT,
    LOCATE_REASON_SYSTEM_PROMPT,
    VlmSegPipeline,
    _candidate_text,
    _parse_json_object,
    render_overlay,
)
from visualcue.harness.systems._vlm import VLMClient
from visualcue.harness.types import Instance, SystemOutput

EVALUATE_SYSTEM_PROMPT = (
    "You are refining an open-vocabulary segmentation. You see the original request, the current "
    "segmentation prompt, and an image with the current candidate masks drawn and numbered. If the "
    "current segmentation already lets you answer the request, choose finalize. Otherwise choose "
    "refine and give an improved segmentation_prompt that preserves or adjusts distinguishing "
    "detail (attributes, ordinal position, spatial relations). Respond ONLY as JSON: "
    '{"action": "finalize"|"refine", "segmentation_prompt": "<required if refine>", '
    '"reason": "<short>"}.'
)


class AgenticPipeline(VlmSegPipeline):
    """Plan once, iteratively refine segmentation prompts, then reason."""

    name = "agentic_pipeline"

    def __init__(
        self,
        vlm_base_url: str = "http://localhost:1234/v1",
        vlm_model: str = "gemma",
        vlm_api_key: str = "lm-studio",
        falcon_model_id: str = DEFAULT_MODEL_ID,
        falcon_device: str = DEFAULT_DEVICE,
        enable_reasoning: bool = True,
        free_falcon_between_calls: bool = False,
        plan_system_prompt: str | None = None,
        count_reason_system_prompt: str = COUNT_REASON_SYSTEM_PROMPT,
        locate_reason_system_prompt: str = LOCATE_REASON_SYSTEM_PROMPT,
        segmentation_prompt_style: str = "short",
        max_steps: int = 8,
        include_prompt_history: bool = False,
        evaluate_system_prompt: str = EVALUATE_SYSTEM_PROMPT,
        vlm: VLMClient | None = None,
        segmenter: FalconSegmenter | None = None,
    ) -> None:
        super().__init__(
            vlm_base_url=vlm_base_url,
            vlm_model=vlm_model,
            vlm_api_key=vlm_api_key,
            falcon_model_id=falcon_model_id,
            falcon_device=falcon_device,
            enable_reasoning=enable_reasoning,
            free_falcon_between_calls=free_falcon_between_calls,
            plan_system_prompt=plan_system_prompt,
            count_reason_system_prompt=count_reason_system_prompt,
            locate_reason_system_prompt=locate_reason_system_prompt,
            segmentation_prompt_style=segmentation_prompt_style,
            vlm=vlm,
            segmenter=segmenter,
        )
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.include_prompt_history = include_prompt_history
        self.evaluate_system_prompt = evaluate_system_prompt

    def config(self) -> dict[str, Any]:
        config = super().config()
        config.update(
            {
                "max_steps": self.max_steps,
                "include_prompt_history": self.include_prompt_history,
                "evaluate_system_prompt": self.evaluate_system_prompt,
            }
        )
        return config

    def run(
        self,
        image: Image.Image,
        query: str,
        gold_targets: list[str] | None = None,
    ) -> SystemOutput:
        """Run plan, bounded segment/evaluate/refine loop, then shared reasoning."""

        rgb_image = image.convert("RGB") if image.mode != "RGB" else image
        intermediate: dict[str, Any] = {"steps": []}
        if gold_targets is not None:
            intent = "locate"
            segmentation_prompt = ", ".join(gold_targets) if gold_targets else query
            intermediate["plan_skipped_gold_targets"] = True
            instances = self._segment(rgb_image, segmentation_prompt)
            stop_reason = "finalize"
            intermediate["steps"].append(_step_record(0, segmentation_prompt, instances, "finalize", None))
        else:
            intent, segmentation_prompt = self._plan(rgb_image, query, intermediate)
            instances, segmentation_prompt, stop_reason = self._run_loop(
                rgb_image,
                query,
                segmentation_prompt,
                intermediate,
            )

        intermediate.update(
            {
                "intent": intent,
                "segmentation_prompt": segmentation_prompt,
                "segmentation_prompt_style": self.segmentation_prompt_style,
                "include_prompt_history": self.include_prompt_history,
                "n_candidates": len(instances),
                "n_steps": len(intermediate["steps"]),
                "stop_reason": stop_reason,
            }
        )
        final_instances, count, answer = self._reason_or_fallback(
            rgb_image,
            query,
            intent,
            segmentation_prompt,
            instances,
            intermediate,
        )
        return SystemOutput(
            instances=final_instances,
            answer=answer,
            count=count,
            latency_ms=0.0,
            intermediate=intermediate,
        )

    def _run_loop(
        self,
        image: Image.Image,
        query: str,
        segmentation_prompt: str,
        intermediate: dict[str, Any],
    ) -> tuple[list[Instance], str, str]:
        instances: list[Instance] = []
        stop_reason = "max_steps"
        for step in range(self.max_steps):
            instances = self._segment(image, segmentation_prompt)
            record = _step_record(step, segmentation_prompt, instances, "max_steps", None)
            intermediate["steps"].append(record)
            if step == self.max_steps - 1:
                stop_reason = "max_steps"
                break

            action, next_prompt, evaluate_raw = self._evaluate(
                image,
                query,
                segmentation_prompt,
                instances,
                intermediate,
            )
            record["action"] = action
            record["evaluate_raw"] = evaluate_raw
            if action == "finalize":
                stop_reason = "finalize"
                break

            if next_prompt.strip().lower() == segmentation_prompt.strip().lower():
                stop_reason = "no_change"
                break
            segmentation_prompt = next_prompt
        return instances, segmentation_prompt, stop_reason

    def _evaluate(
        self,
        image: Image.Image,
        query: str,
        segmentation_prompt: str,
        instances: list[Instance],
        intermediate: dict[str, Any],
    ) -> tuple[str, str, str]:
        overlay = render_overlay(image, instances, fill_masks=False, draw_mask_outlines=True)
        parts = [
            f"Original request: {query}\n"
            f"Current segmentation prompt: {segmentation_prompt}\n"
            f"The segmenter returned {len(instances)} candidate(s)."
        ]
        if self.include_prompt_history:
            parts.append(_prompt_history_text(intermediate))
        parts.append(
            "Candidates:\n"
            + "\n".join(_candidate_text(index, instance) for index, instance in enumerate(instances))
        )
        user_text = "\n".join(parts)
        raw = self.vlm.complete(self.evaluate_system_prompt, user_text, image=overlay)
        try:
            parsed = _parse_json_object(raw)
            action = str(parsed.get("action", "finalize")).lower()
            if action not in {"finalize", "refine"}:
                raise ValueError(f"invalid action: {action}")
            if action == "finalize":
                return "finalize", segmentation_prompt, raw
            next_prompt = str(parsed["segmentation_prompt"])
            if not next_prompt.strip():
                raise ValueError("empty segmentation_prompt")
            return "refine", next_prompt, raw
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            intermediate["evaluate_parse_fallback"] = True
            intermediate["evaluate_parse_error"] = str(exc)
            return "finalize", segmentation_prompt, raw


def _step_record(
    step: int,
    segmentation_prompt: str,
    instances: list[Instance],
    action: str,
    evaluate_raw: str | None,
) -> dict[str, Any]:
    return {
        "step": step,
        "segmentation_prompt": segmentation_prompt,
        "n_candidates": len(instances),
        "action": action,
        "evaluate_raw": evaluate_raw,
    }


def _prompt_history_text(intermediate: dict[str, Any]) -> str:
    steps = intermediate.get("steps", [])
    if not isinstance(steps, list):
        return "Previous segmentation prompts: none"

    previous_prompts = [
        str(step.get("segmentation_prompt"))
        for step in steps[:-1]
        if isinstance(step, dict) and step.get("segmentation_prompt")
    ]
    if not previous_prompts:
        return "Previous segmentation prompts: none"
    lines = ["Previous segmentation prompts:"]
    lines.extend(f"{index}: {prompt}" for index, prompt in enumerate(previous_prompts))
    return "\n".join(lines)
