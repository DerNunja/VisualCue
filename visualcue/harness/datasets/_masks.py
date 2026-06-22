"""Shared COCO mask decoding helpers for dataset adapters."""

from __future__ import annotations

from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


def decode_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray | None:
    """COCO polygon list or RLE dict to a (H, W) bool mask; None if absent."""

    if segmentation is None:
        return None
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        rle.setdefault("size", [height, width])
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        return mask_utils.decode(rle).astype(bool)

    if isinstance(segmentation, list):
        polygons = segmentation if segmentation and isinstance(segmentation[0], list) else [segmentation]
        rles = mask_utils.frPyObjects(polygons, height, width)
        return mask_utils.decode(mask_utils.merge(rles)).astype(bool)
    raise ValueError("segmentation must be a COCO RLE dict, polygon list, or None")
