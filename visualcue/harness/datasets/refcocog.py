"""Offline RefCOCOg adapter for refer-format annotations and COCO images."""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image

from visualcue.harness.datasets._masks import decode_segmentation
from visualcue.harness.types import GTSample, Instance

DEFAULT_SPLIT = "val"
DEFAULT_SPLIT_BY = "umd"


class RefCOCOgAdapter:
    """Read RefCOCOg from refs(split_by).p, instances.json, and local images/."""

    name = "refcocog"

    def __init__(self, root: str | Path, split: str = DEFAULT_SPLIT, split_by: str = DEFAULT_SPLIT_BY) -> None:
        self.root = Path(root)
        self.split = split
        self.split_by = split_by
        self.refs_path = self.root / f"refs({split_by}).p"
        self.instances_path = self.root / "instances.json"
        if not self.refs_path.exists():
            raise FileNotFoundError(f"missing refs file: {self.refs_path}")
        if not self.instances_path.exists():
            raise FileNotFoundError(f"missing instances file: {self.instances_path}")

        refs = _load_refs(self.refs_path)
        instances = json.loads(self.instances_path.read_text(encoding="utf-8"))
        self._images = {image["id"]: image for image in instances.get("images", [])}
        self._annotations = {annotation["id"]: annotation for annotation in instances.get("annotations", [])}
        self._categories = {category["id"]: category.get("name") for category in instances.get("categories", [])}
        self._entries = _expression_entries(refs, split)

    def __iter__(self) -> Iterator[GTSample]:
        """Yield one referring-expression sample per sentence; no network access."""

        for entry in self._entries:
            ref = entry["ref"]
            sentence = entry["sentence"]
            annotation = self._annotations[ref["ann_id"]]
            image_info = self._images[ref["image_id"]]
            image = _open_rgb(self.root / "images" / image_info["file_name"])
            width, height = image.size
            mask = decode_segmentation(annotation["segmentation"], height, width)
            label = self._categories.get(annotation.get("category_id"))
            bbox = tuple(float(value) for value in annotation["bbox"]) if "bbox" in annotation else None
            sample_id = f"{ref['ref_id']}__{sentence['sent_id']}"
            query = str(sentence.get("raw") or sentence.get("sent"))
            yield GTSample(
                image=image,
                query=query,
                query_type="referring",
                gt_instances=[Instance(mask=mask, bbox=bbox, label=label, score=None)],
                gt_count=None,
                sample_id=sample_id,
            )

    def __len__(self) -> int:
        """Return the number of referring expressions in the selected split."""

        return len(self._entries)


def required_image_files(root: str | Path, split: str = DEFAULT_SPLIT, split_by: str = DEFAULT_SPLIT_BY) -> list[str]:
    """Return sorted COCO file names needed by a RefCOCOg split."""

    root_path = Path(root)
    refs = _load_refs(root_path / f"refs({split_by}).p")
    instances = json.loads((root_path / "instances.json").read_text(encoding="utf-8"))
    image_ids = {ref["image_id"] for ref in refs if ref.get("split") == split}
    images = {image["id"]: image["file_name"] for image in instances.get("images", [])}
    missing_ids = sorted(image_id for image_id in image_ids if image_id not in images)
    if missing_ids:
        raise KeyError(f"instances.json has no image entries for ids: {missing_ids[:10]}")
    return sorted(images[image_id] for image_id in image_ids)


def _load_refs(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _expression_entries(refs: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for ref in refs:
        if ref.get("split") != split:
            continue
        for sentence in ref.get("sentences", []):
            entries.append({"ref": ref, "sentence": sentence})
    return entries


def _open_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"missing RefCOCOg image: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")
