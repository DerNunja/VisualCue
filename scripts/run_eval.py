"""CLI entry point for VisualCue evaluation runs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualcue.harness.datasets.base import DatasetAdapter
from visualcue.harness.datasets.custom import CustomAdapter
from visualcue.harness.datasets.fsc147 import FSC147Adapter
from visualcue.harness.datasets.refcocog import RefCOCOgAdapter
from visualcue.harness.runner import evaluate
from visualcue.harness.systems.agentic import AgenticPipeline
from visualcue.harness.systems.falcon_only import FalconOnly
from visualcue.harness.systems.mock import MockSystem
from visualcue.harness.systems.sequential import SequentialPipeline
from visualcue.harness.systems.yoloe_only import YoloeOnly
from visualcue.harness.types import GTSample

SYSTEM_REGISTRY = {
    "mock": MockSystem,
    "falcon_only": FalconOnly,
    "yoloe_only": YoloeOnly,
    "sequential_pipeline": SequentialPipeline,
    "agentic_pipeline": AgenticPipeline,
}
DATASET_REGISTRY = {
    "custom": CustomAdapter,
    "refcocog": RefCOCOgAdapter,
    "fsc147": FSC147Adapter,
}


class LimitedDataset:
    """Cap dataset iteration for smoke runs; None leaves data uncapped."""

    def __init__(self, dataset: DatasetAdapter, limit: int | None) -> None:
        self.dataset = dataset
        self.limit = limit
        self.name = dataset.name

    def __iter__(self) -> Iterator[GTSample]:
        """Yield at most `limit` samples while preserving original order."""

        for index, sample in enumerate(self.dataset):
            if self.limit is not None and index >= self.limit:
                break
            yield sample

    def __len__(self) -> int:
        """Return capped length when a limit is configured."""

        if self.limit is None:
            return len(self.dataset)
        return min(len(self.dataset), self.limit)


def main() -> None:
    """Parse config and run every configured system/dataset combination."""

    parser = argparse.ArgumentParser(description="Run VisualCue eval harness")
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes) or {}
    systems = [_make_system(entry) for entry in config.get("systems", [])]
    datasets = [_make_dataset(entry) for entry in config.get("datasets", [])]
    sample_limit = config.get("sample_limit")
    if sample_limit is not None:
        sample_limit = int(sample_limit)
    out_dir = Path(config.get("out_dir", "results"))
    iou_threshold = float(config.get("iou_threshold", 0.5))
    attribution = bool(config.get("attribution", False))
    seed = config.get("seed")
    seed = int(seed) if seed is not None else None

    for system in systems:
        for dataset in datasets:
            evaluate(
                system=system,
                dataset=LimitedDataset(dataset, sample_limit),
                out_dir=out_dir,
                iou_threshold=iou_threshold,
                attribution=attribution,
                config_bytes=config_bytes,
                seed=seed,
            )


def _make_system(entry: str | dict[str, Any]) -> Any:
    name = entry if isinstance(entry, str) else entry["name"]
    kwargs = {} if isinstance(entry, str) else {key: value for key, value in entry.items() if key != "name"}
    return SYSTEM_REGISTRY[name](**kwargs)


def _make_dataset(entry: str | dict[str, Any]) -> DatasetAdapter:
    if isinstance(entry, str):
        return DATASET_REGISTRY[entry]()
    name = entry["name"]
    kwargs = {key: value for key, value in entry.items() if key != "name"}
    return DATASET_REGISTRY[name](**kwargs)


if __name__ == "__main__":
    main()
