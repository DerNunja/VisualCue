"""VisionSystem protocol shared by all benchmarked systems."""

from __future__ import annotations

from typing import Protocol

from PIL import Image

from visualcue.harness.types import SystemOutput


class VisionSystem(Protocol):
    """Model-agnostic visual question system contract."""

    name: str

    def run(
        self,
        image: Image.Image,
        query: str,
        gold_targets: list[str] | None = None,
    ) -> SystemOutput:
        """Return deterministic output; runner overwrites latency_ms."""
        ...
