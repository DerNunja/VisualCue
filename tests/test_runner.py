from __future__ import annotations

import json

from PIL import Image

from visualcue.harness.datasets.custom import CustomAdapter
from visualcue.harness.runner import evaluate
from visualcue.harness.systems.mock import MockSystem


def test_runner_end_to_end_with_custom_adapter(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images"
    images_dir.mkdir(parents=True)
    for sample_id in ["ref", "count", "open"]:
        Image.new("RGB", (10, 10), "white").save(images_dir / f"{sample_id}.jpg")
    annotations = [
        {
            "sample_id": "ref",
            "query": "mark the square",
            "query_type": "referring",
            "instances": [{"label": "square", "polygon": [[1, 1, 5, 1, 5, 5, 1, 5]]}],
        },
        {"sample_id": "count", "query": "count squares", "query_type": "counting", "gt_count": 1, "instances": []},
        {"sample_id": "open", "query": "describe image", "query_type": "open_ended", "instances": []},
    ]
    (dataset_root / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")

    record = evaluate(MockSystem(), CustomAdapter(dataset_root), tmp_path / "results", config_bytes=b"test", seed=7)

    assert record.n_samples == 3
    result_path = tmp_path / "results" / "mock__custom.json"
    assert result_path.exists()
    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data["n_samples"] == 3
    assert set(data["metrics"]["end_to_end"]) == {"referring", "counting", "open_ended"}
    assert "mean_iou" in data["metrics"]["end_to_end"]["referring"]
    assert "mae" in data["metrics"]["end_to_end"]["counting"]
    assert data["metrics"]["end_to_end"]["open_ended"]["skipped"] is True
