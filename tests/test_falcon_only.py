from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from visualcue.harness.types import SystemOutput


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
