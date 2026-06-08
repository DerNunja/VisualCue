"""Falcon-only benchmark stub pending model integration."""

from __future__ import annotations

from PIL import Image

from visualcue.harness.types import SystemOutput


class FalconOnly:
    """Single-pass Falcon system placeholder; inference is intentionally absent."""

    name = "falcon_only"

    def run(self, image: Image.Image, query: str, gold_targets: list[str] | None = None) -> SystemOutput:
        """Raise until model weights and inference code are integrated."""

        raise NotImplementedError("integration pending")
