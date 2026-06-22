"""Offline FSC-147 counting dataset adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image

from visualcue.harness.types import GTSample

DEFAULT_SPLIT = "test"
DEFAULT_IMAGE_DIR = "images_384_VarV2"
ANNOTATION_FILE = "annotation_FSC147_384.json"
SPLIT_FILE = "Train_Test_Val_FSC_147.json"
CLASSES_FILE = "ImageClasses_FSC147.txt"


class FSC147Adapter:
    """Read FSC-147 point-count annotations and images from local disk."""

    name = "fsc147"

    def __init__(
        self,
        root: str | Path,
        split: str = DEFAULT_SPLIT,
        image_dir: str | Path | None = None,
        name: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.name = name or self.name
        self.image_dir = Path(image_dir) if image_dir is not None else self.root / DEFAULT_IMAGE_DIR

        annotation_path = self.root / ANNOTATION_FILE
        split_path = self.root / SPLIT_FILE
        classes_path = self.root / CLASSES_FILE
        for path in (annotation_path, split_path, classes_path):
            if not path.exists():
                raise FileNotFoundError(f"missing FSC-147 file: {path}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"missing FSC-147 image directory: {self.image_dir}")

        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        classes = _load_classes(classes_path)
        file_names = splits[split]
        self._samples = [_sample_from_file_name(file_name, annotations, classes, self.image_dir) for file_name in file_names]

    def __iter__(self) -> Iterator[GTSample]:
        """Yield preloaded counting samples; no network or disk re-indexing."""

        return iter(self._samples)

    def __len__(self) -> int:
        """Return the number of samples in the configured split."""

        return len(self._samples)


def _sample_from_file_name(
    file_name: str,
    annotations: dict[str, Any],
    classes: dict[str, str],
    image_dir: Path,
) -> GTSample:
    annotation = annotations[file_name]
    gt_count = len(annotation["points"])
    class_name = classes[file_name]
    image = _open_rgb(image_dir / file_name)
    return GTSample(
        image=image,
        query=f"How many {class_name} are there?",
        query_type="counting",
        gt_instances=[],
        gt_count=gt_count,
        sample_id=file_name,
    )


def _load_classes(path: Path) -> dict[str, str]:
    classes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        file_name, class_name = line.split(maxsplit=1)
        classes[file_name] = class_name.strip()
    return classes


def _open_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"missing FSC-147 image: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")
