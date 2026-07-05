"""Reusable Falcon Perception segmenter."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from visualcue.harness.types import Instance

DEFAULT_MODEL_ID = "tiiuae/falcon-perception"
DEFAULT_DEVICE = "cuda:0"


class FalconSegmenter:
    """Load Falcon once and return harness Instances for a prompt."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = DEFAULT_DEVICE,
        clear_cuda_cache_after_segment: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self.model_id = model_id
        self.device = device
        self.clear_cuda_cache_after_segment = clear_cuda_cache_after_segment
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map={"": device},
            dtype=torch.bfloat16,
        )
        self._disable_sampling_defaults()

    def segment(self, image: Image.Image, prompt: str) -> list[Instance]:
        """Run Falcon on prompt; return Instances with label=prompt."""

        rgb_image = image.convert("RGB") if image.mode != "RGB" else image
        try:
            predictions = self.model.generate(rgb_image, prompt)[0]
            return [_prediction_to_instance(prediction, prompt, rgb_image.size) for prediction in predictions]
        finally:
            if self.clear_cuda_cache_after_segment:
                _clear_cuda_cache(self.device)

    def to_cpu(self) -> None:
        """Move Falcon weights to CPU and clear CUDA cache when available."""

        import torch

        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_device(self) -> None:
        """Move Falcon weights back to the configured device."""

        self.model.to(self.device)

    def _disable_sampling_defaults(self) -> None:
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "do_sample"):
            generation_config.do_sample = False


def _prediction_to_instance(
    prediction: dict[str, Any],
    query: str,
    image_size: tuple[int, int],
) -> Instance:
    width, height = image_size
    mask = _decode_mask(prediction["mask_rle"])
    bbox = _bbox_from_prediction(prediction, width, height, mask)
    return Instance(mask=mask, bbox=bbox, label=query, score=None)


def _decode_mask(rle: dict[str, Any]) -> np.ndarray:
    encoded = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
    return mask_utils.decode(encoded).astype(bool)


def _bbox_from_normalized_center(
    prediction: dict[str, Any],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    w_abs = prediction["hw"]["w"] * width
    h_abs = prediction["hw"]["h"] * height
    x = prediction["xy"]["x"] * width - w_abs / 2
    y = prediction["xy"]["y"] * height - h_abs / 2
    return (x, y, w_abs, h_abs)


def _bbox_from_prediction(
    prediction: dict[str, Any],
    width: int,
    height: int,
    mask: np.ndarray,
) -> tuple[float, float, float, float] | None:
    xy = prediction.get("xy")
    hw = prediction.get("hw")
    if isinstance(xy, dict) and isinstance(hw, dict) and {"x", "y"} <= xy.keys() and {"w", "h"} <= hw.keys():
        return _bbox_from_normalized_center(prediction, width, height)
    return _bbox_from_mask(mask)


def _bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max() + 1)
    y1 = float(ys.max() + 1)
    return (x0, y0, x1 - x0, y1 - y0)


def _clear_cuda_cache(device: str) -> None:
    if not device.startswith("cuda"):
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    ipc_collect = getattr(torch.cuda, "ipc_collect", None)
    if callable(ipc_collect):
        ipc_collect()
