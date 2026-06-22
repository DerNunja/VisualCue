from __future__ import annotations

import json

from PIL import Image

from visualcue.harness.datasets.fsc147 import FSC147Adapter


def test_fsc147_adapter_reads_counting_split(tmp_path) -> None:
    images_dir = tmp_path / "images_384_VarV2"
    images_dir.mkdir()
    Image.new("RGB", (8, 8), "white").save(images_dir / "1.jpg")
    (tmp_path / "annotation_FSC147_384.json").write_text(
        json.dumps({"1.jpg": {"points": [[1, 1], [2, 2], [3, 3]]}}),
        encoding="utf-8",
    )
    (tmp_path / "Train_Test_Val_FSC_147.json").write_text(
        json.dumps({"train": [], "val": [], "test": ["1.jpg"]}),
        encoding="utf-8",
    )
    (tmp_path / "ImageClasses_FSC147.txt").write_text("1.jpg\tapples\n", encoding="utf-8")

    dataset = FSC147Adapter(tmp_path, split="test")
    samples = list(dataset)

    assert len(dataset) == 1
    assert len(samples) == 1
    sample = samples[0]
    assert sample.query_type == "counting"
    assert sample.gt_count == 3
    assert sample.gt_instances == []
    assert "apples" in sample.query.lower()
    assert sample.sample_id == "1.jpg"
    assert len(FSC147Adapter(tmp_path, split="train")) == 0
