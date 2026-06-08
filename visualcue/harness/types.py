"""Shared dataclasses for VisualCue evaluation records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from PIL import Image


@dataclass
class Instance:
    """One object instance; masks must be 2D bool arrays when present."""

    mask: np.ndarray | None
    bbox: tuple[float, float, float, float] | None
    label: str | None
    score: float | None

    def __post_init__(self) -> None:
        if self.mask is None:
            return
        if self.mask.ndim != 2:
            raise ValueError("Instance.mask must have shape (H, W)")
        if self.mask.dtype != np.bool_:
            self.mask = self.mask.astype(bool)


@dataclass
class SystemOutput:
    """System prediction; instances are shown, count/answer are said."""

    instances: list[Instance]
    answer: str | None
    count: int | None
    latency_ms: float
    intermediate: dict[str, Any]


@dataclass
class GTSample:
    """Ground-truth sample with query type controlling metric applicability."""

    image: Image.Image
    query: str
    query_type: Literal["referring", "counting", "open_ended"]
    gt_instances: list[Instance]
    gt_count: int | None
    sample_id: str


@dataclass
class ResultRecord:
    """Serialized evaluation summary for one system/dataset run."""

    system_name: str
    dataset_name: str
    n_samples: int
    metrics: dict[str, Any]
    latency: dict[str, float]
    config_hash: str
    timestamp: str
