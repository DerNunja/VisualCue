from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

from visualcue.harness.types import SystemOutput


def test_prediction_bbox_falls_back_to_mask_for_malformed_xy() -> None:
    from visualcue.harness.systems.falcon_only import _prediction_to_instance

    mask = np.zeros((5, 6), dtype=bool)
    mask[1:3, 2:5] = True
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    prediction = {
        "xy": {"h": 0.1, "w": 0.2},
        "hw": {"h": 0.4, "w": 0.5},
        "mask_rle": {"size": rle["size"], "counts": rle["counts"].decode("ascii")},
    }

    instance = _prediction_to_instance(prediction, "object", (6, 5))

    assert instance.bbox == (2.0, 1.0, 3.0, 2.0)


def test_falcon_only_runs_with_local_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("Falcon Perception requires CUDA")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    from visualcue.harness.systems.falcon_only import FalconOnly

    image_path = os.path.join(os.path.dirname(__file__), "fixtures", "custom_dataset", "images", "ref.jpg")
    image = Image.open(image_path).convert("RGB")

    try:
        output = FalconOnly().run(image, "square")
    except OSError as exc:
        pytest.skip(f"Falcon Perception weights are not available locally: {exc}")

    assert isinstance(output, SystemOutput)
    for instance in output.instances:
        assert isinstance(instance.mask, np.ndarray)
        assert instance.mask.dtype == np.bool_
        assert instance.mask.shape == (image.height, image.width)
    assert output.count == len(output.instances)
