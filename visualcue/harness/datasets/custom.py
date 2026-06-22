"""Custom annotated dataset adapter with simple-list and COCO-style JSON support."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from visualcue.harness.datasets._masks import decode_segmentation
from visualcue.harness.types import GTSample, Instance

VALID_QUERY_TYPES = {"referring", "counting", "open_ended"}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class CustomAdapter:
    """Load `images/{id}.jpg` and annotations with COCO-RLE or polygon masks."""

    name = "custom"

    def __init__(self, root: str | Path, name: str | None = None) -> None:
        self.root = Path(root)
        self.name = name or self.name
        self.annotation_path = self.root / "annotations.json"
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"missing annotations file: {self.annotation_path}")
        self._samples = self._load_samples()

    def __iter__(self) -> Iterator[GTSample]:
        """Yield loaded samples; images are loaded eagerly to avoid closed file handles."""

        return iter(self._samples)

    def __len__(self) -> int:
        """Return sample count, including open-ended samples."""

        return len(self._samples)

    def _load_samples(self) -> list[GTSample]:
        data = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [self._sample_from_simple(entry) for entry in data]
        if isinstance(data, dict):
            return self._samples_from_coco(data)
        raise ValueError("annotations.json must be a list or object")

    def _sample_from_simple(self, entry: dict[str, Any]) -> GTSample:
        sample_id = str(entry["sample_id"])
        image = self._load_image(entry.get("file_name"), sample_id)
        width, height = image.size
        query_type = _query_type(entry["query_type"])
        instances = [self._instance_from_entry(instance, height, width, {}) for instance in entry.get("instances", [])]
        return GTSample(
            image=image,
            query=str(entry["query"]),
            query_type=query_type,
            gt_instances=instances,
            gt_count=entry.get("gt_count"),
            sample_id=sample_id,
        )

    def _samples_from_coco(self, data: dict[str, Any]) -> list[GTSample]:
        categories = {category.get("id"): category.get("name") for category in data.get("categories", [])}
        images = {image.get("id"): image for image in data.get("images", [])}
        annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for annotation in data.get("annotations", []):
            annotations_by_image[annotation.get("image_id")].append(annotation)

        sample_entries = data.get("samples") or list(images.values())
        samples: list[GTSample] = []
        for entry in sample_entries:
            image_id = entry.get("image_id", entry.get("id"))
            image_info = images.get(image_id, entry)
            sample_id = str(entry.get("sample_id", image_id))
            image = self._load_image(entry.get("file_name") or image_info.get("file_name"), sample_id)
            width, height = image.size
            query_type = _query_type(entry["query_type"])
            raw_instances = [*annotations_by_image.get(image_id, []), *entry.get("instances", [])]
            instances = [self._instance_from_entry(instance, height, width, categories) for instance in raw_instances]
            samples.append(
                GTSample(
                    image=image,
                    query=str(entry["query"]),
                    query_type=query_type,
                    gt_instances=instances,
                    gt_count=entry.get("gt_count"),
                    sample_id=sample_id,
                )
            )
        return samples

    def _load_image(self, file_name: str | None, sample_id: str) -> Image.Image:
        if file_name is not None:
            image_path = self.root / "images" / file_name
            if not image_path.exists():
                image_path = self.root / file_name
            if image_path.exists():
                return _open_rgb(image_path)
        for extension in IMAGE_EXTENSIONS:
            candidate = self.root / "images" / f"{sample_id}{extension}"
            if candidate.exists():
                return _open_rgb(candidate)
        raise FileNotFoundError(f"missing image for sample {sample_id}")

    def _instance_from_entry(
        self,
        entry: dict[str, Any],
        height: int,
        width: int,
        categories: dict[Any, str | None],
    ) -> Instance:
        mask = _decode_mask(entry, height, width)
        bbox = tuple(float(value) for value in entry["bbox"]) if "bbox" in entry else None
        label = entry.get("label") or categories.get(entry.get("category_id"))
        score = float(entry["score"]) if "score" in entry else None
        return Instance(mask=mask, bbox=bbox, label=label, score=score)


def _query_type(value: str) -> Literal["referring", "counting", "open_ended"]:
    if value not in VALID_QUERY_TYPES:
        raise ValueError(f"invalid query_type: {value}")
    return value  # type: ignore[return-value]


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _decode_mask(entry: dict[str, Any], height: int, width: int) -> np.ndarray | None:
    rle = entry.get("rle")
    segmentation = entry.get("segmentation")
    polygon = entry.get("polygon")
    if rle is not None:
        return decode_segmentation(rle, height, width)
    if isinstance(segmentation, dict):
        return decode_segmentation(segmentation, height, width)
    if polygon is not None:
        return decode_segmentation(polygon, height, width)
    if isinstance(segmentation, list):
        return decode_segmentation(segmentation, height, width)
    return None
