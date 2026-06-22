"""Manual smoke check for Falcon Perception on one image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualcue.harness.systems.falcon_only import FalconOnly
from visualcue.harness.types import SystemOutput

MASK_COLOR = np.array([255, 0, 0], dtype=np.float32)
MASK_ALPHA = 0.45


def main() -> None:
    """Run FalconOnly and write a qualitative overlay to results/smoke_falcon.png."""

    parser = argparse.ArgumentParser(description="Smoke-test Falcon Perception on one image")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("query")
    args = parser.parse_args()

    image = Image.open(args.image_path).convert("RGB")
    output = FalconOnly().run(image, args.query)
    out_path = ROOT / "results" / "smoke_falcon.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_overlay(image, output, out_path)

    first_bbox = output.instances[0].bbox if output.instances else None
    print(f"instances: {len(output.instances)}")
    print(f"first_bbox: {first_bbox}")
    print(f"overlay: {out_path}")


def _write_overlay(image: Image.Image, output: SystemOutput, path: Path) -> None:
    base = np.asarray(image).astype(np.float32)
    combined_mask = np.zeros((image.height, image.width), dtype=bool)
    for instance in output.instances:
        if instance.mask is not None:
            combined_mask |= instance.mask
    base[combined_mask] = (1.0 - MASK_ALPHA) * base[combined_mask] + MASK_ALPHA * MASK_COLOR
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(path)


if __name__ == "__main__":
    main()
