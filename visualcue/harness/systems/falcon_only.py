"""Falcon Perception single-pass benchmark system."""

from __future__ import annotations

from PIL import Image

from visualcue.harness.systems._falcon import DEFAULT_DEVICE, DEFAULT_MODEL_ID, FalconSegmenter
from visualcue.harness.types import SystemOutput


class FalconOnly:
    """Single-pass Falcon Perception baseline without query rewriting."""

    name = "falcon_only"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = DEFAULT_DEVICE,
        segmenter: FalconSegmenter | None = None,
    ) -> None:
        """Load the Falcon Perception model once for repeated harness calls."""

        self.model_id = model_id
        self.device = device
        self.segmenter = segmenter or FalconSegmenter(model_id=model_id, device=device)

    def run(
        self,
        image: Image.Image,
        query: str,
        gold_targets: list[str] | None = None,
    ) -> SystemOutput:
        """Run Falcon on the verbatim query; gold_targets are accepted and ignored."""

        del gold_targets

        instances = self.segmenter.segment(image, query)
        count = len(instances)
        return SystemOutput(
            instances=instances,
            answer=str(count),
            count=count,
            latency_ms=0.0,
            intermediate={"prompt": query, "n_preds": count},
        )
