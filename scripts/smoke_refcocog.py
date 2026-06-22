"""Smoke-check local RefCOCOg files and mask shapes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualcue.harness.datasets.refcocog import RefCOCOgAdapter


def main() -> None:
    """Iterate RefCOCOg and verify every decoded mask matches its image shape."""

    parser = argparse.ArgumentParser(description="Smoke-test local RefCOCOg annotations")
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--split-by", default="umd")
    args = parser.parse_args()

    dataset = RefCOCOgAdapter(args.root, split=args.split, split_by=args.split_by)
    checked = 0
    for sample in dataset:
        expected_shape = (sample.image.height, sample.image.width)
        for instance in sample.gt_instances:
            if instance.mask is None:
                raise RuntimeError(f"sample {sample.sample_id} has no mask")
            if instance.mask.shape != expected_shape:
                raise RuntimeError(
                    f"sample {sample.sample_id} mask shape {instance.mask.shape} != image shape {expected_shape}"
                )
        checked += 1

    print(f"checked samples: {checked}")
    print("mask shapes match image shapes")


if __name__ == "__main__":
    main()
