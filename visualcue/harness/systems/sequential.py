"""Sequential VLM-plan, Falcon-segment, VLM-reason pipeline."""

from __future__ import annotations

from typing import Any

from PIL import Image

from visualcue.harness.systems._pipeline import (
    COUNT_REASON_SYSTEM_PROMPT,
    LOCATE_REASON_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT_DETAILED,
    PLAN_SYSTEM_PROMPT_SHORT,
    VlmSegPipeline,
    render_overlay,
)
from visualcue.harness.types import SystemOutput


class SequentialPipeline(VlmSegPipeline):
    """One-pass plan, segment, reason pipeline."""

    name = "sequential_pipeline"

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

        instances = self._segment(rgb_image, segmentation_prompt)
        intermediate.update(
            {
                "intent": intent,
                "segmentation_prompt": segmentation_prompt,
                "segmentation_prompt_style": self.segmentation_prompt_style,
                "n_candidates": len(instances),
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


__all__ = [
    "COUNT_REASON_SYSTEM_PROMPT",
    "LOCATE_REASON_SYSTEM_PROMPT",
    "PLAN_SYSTEM_PROMPT_DETAILED",
    "PLAN_SYSTEM_PROMPT_SHORT",
    "SequentialPipeline",
    "render_overlay",
]
