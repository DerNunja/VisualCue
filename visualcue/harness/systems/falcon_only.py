"""Falcon Perception single-pass benchmark system."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from visualcue.harness.types import Instance, SystemOutput

DEFAULT_MODEL_ID = "tiiuae/falcon-perception"
DEFAULT_DEVICE = "cuda:0"


class FalconOnly:
    """Single-pass Falcon Perception baseline without query rewriting."""

    name = "falcon_only"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = DEFAULT_DEVICE) -> None:
        """Load the Falcon Perception model once for repeated harness calls."""

        from transformers import AutoModelForCausalLM

        self.model_id = model_id
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map={"": device},
        )
        self._disable_sampling_defaults()

    def run(
        self,
        image: Image.Image,
        query: str,
        gold_targets: list[str] | None = None,
    ) -> SystemOutput:
        """Run Falcon on the verbatim query; gold_targets are accepted and ignored."""

        del gold_targets

        rgb_image = image.convert("RGB") if image.mode != "RGB" else image
        predictions = self.model.generate(rgb_image, query)[0]
        instances = [_prediction_to_instance(prediction, query, rgb_image.size) for prediction in predictions]
        count = len(instances)
        return SystemOutput(
            instances=instances,
            answer=str(count),
            count=count,
            latency_ms=0.0,
            intermediate={"prompt": query, "n_preds": count},
        )

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
    bbox = _bbox_from_normalized_center(prediction, width, height)
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
