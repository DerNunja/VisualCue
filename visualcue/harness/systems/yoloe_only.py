"""YOLOE-only benchmark stub pending model integration."""

from __future__ import annotations

from PIL import Image

from visualcue.harness.types import SystemOutput

# AGPL-3.0 — nur Benchmark, nicht in ausgeliefertem Pipeline-Code.


class YoloeOnly:
    """Single-pass YOLOE system placeholder; inference is intentionally absent."""

    name = "yoloe_only"

    def run(self, image: Image.Image, query: str, gold_targets: list[str] | None = None) -> SystemOutput:
        """Raise until model weights and inference code are integrated."""

        raise NotImplementedError("integration pending")
