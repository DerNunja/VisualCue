"""Deterministic mock system for offline end-to-end harness tests."""

from __future__ import annotations

import hashlib
import random

import numpy as np
from PIL import Image

from visualcue.harness.types import Instance, SystemOutput

DEFAULT_MIN_INSTANCES = 1
DEFAULT_MAX_INSTANCES = 3
MIN_RECT_FRACTION = 0.15
MAX_RECT_FRACTION = 0.35


class MockSystem:
    """Create reproducible rectangular pseudo-instances; zero gold targets yields no masks."""

    name = "mock"

    def __init__(
        self,
        min_instances: int = DEFAULT_MIN_INSTANCES,
        max_instances: int = DEFAULT_MAX_INSTANCES,
    ) -> None:
        self.min_instances = min_instances
        self.max_instances = max_instances

    def run(
        self,
        image: Image.Image,
        query: str,
        gold_targets: list[str] | None = None,
    ) -> SystemOutput:
        """Return deterministic rectangles; latency is left at 0.0 for the runner."""

        sample_id = str(getattr(image, "sample_id", query))
        seed_material = f"{sample_id}|{query}|{gold_targets or []}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        rng = random.Random(seed)

        if gold_targets is not None:
            count = len(gold_targets)
            labels = gold_targets
        else:
            count = rng.randint(self.min_instances, self.max_instances)
            labels = ["mock" for _ in range(count)]

        width, height = image.size
        instances: list[Instance] = []
        for index in range(count):
            rect_width = max(1, int(width * rng.uniform(MIN_RECT_FRACTION, MAX_RECT_FRACTION)))
            rect_height = max(1, int(height * rng.uniform(MIN_RECT_FRACTION, MAX_RECT_FRACTION)))
            max_x = max(0, width - rect_width)
            max_y = max(0, height - rect_height)
            x = rng.randint(0, max_x) if max_x else 0
            y = rng.randint(0, max_y) if max_y else 0
            mask = np.zeros((height, width), dtype=bool)
            mask[y : y + rect_height, x : x + rect_width] = True
            instances.append(
                Instance(
                    mask=mask,
                    bbox=(float(x), float(y), float(rect_width), float(rect_height)),
                    label=labels[index] if index < len(labels) else "mock",
                    score=1.0,
                )
            )

        return SystemOutput(
            instances=instances,
            answer=str(count),
            count=count,
            latency_ms=0.0,
            intermediate={"mock_seed": seed},
        )
