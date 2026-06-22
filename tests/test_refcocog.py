from __future__ import annotations

import json
import pickle

import numpy as np
from PIL import Image

from visualcue.harness.datasets.refcocog import RefCOCOgAdapter, required_image_files


def test_refcocog_adapter_reads_local_refer_files(tmp_path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (6, 5), "white").save(images_dir / "COCO_train2014_000000000001.jpg")

    refs = [
        {
            "ref_id": 10,
            "ann_id": 20,
            "image_id": 1,
            "split": "val",
            "sentences": [{"sent_id": 30, "sent": "white square"}],
        },
        {
            "ref_id": 11,
            "ann_id": 21,
            "image_id": 2,
            "split": "test",
            "sentences": [{"sent_id": 31, "sent": "ignored"}],
        },
    ]
    with (tmp_path / "refs(umd).p").open("wb") as handle:
        pickle.dump(refs, handle)

    instances = {
        "images": [{"id": 1, "file_name": "COCO_train2014_000000000001.jpg", "height": 5, "width": 6}],
        "annotations": [
            {
                "id": 20,
                "image_id": 1,
                "category_id": 3,
                "bbox": [1, 1, 3, 2],
                "segmentation": [[1, 1, 4, 1, 4, 3, 1, 3]],
            }
        ],
        "categories": [{"id": 3, "name": "square"}],
    }
    (tmp_path / "instances.json").write_text(json.dumps(instances), encoding="utf-8")

    assert required_image_files(tmp_path, split="val") == ["COCO_train2014_000000000001.jpg"]

    dataset = RefCOCOgAdapter(tmp_path, split="val")
    samples = list(dataset)

    assert len(dataset) == 1
    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == "10__30"
    assert sample.query == "white square"
    assert sample.query_type == "referring"
    assert sample.gt_instances[0].label == "square"
    assert isinstance(sample.gt_instances[0].mask, np.ndarray)
    assert sample.gt_instances[0].mask.dtype == np.bool_
    assert sample.gt_instances[0].mask.shape == (5, 6)
