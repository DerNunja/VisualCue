from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from visualcue.harness.datasets.custom import CustomAdapter
from visualcue.harness.runner import evaluate
from visualcue.harness.systems._vlm import VLMTokenLimitExceeded
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
    samples_path = tmp_path / "results" / "mock__custom__samples.jsonl"
    assert result_path.exists()
    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data["n_samples"] == 3
    assert set(data["metrics"]["end_to_end"]) == {"referring", "counting", "open_ended"}
    assert "mean_iou" in data["metrics"]["end_to_end"]["referring"]
    assert "mae" in data["metrics"]["end_to_end"]["counting"]
    assert data["metrics"]["end_to_end"]["open_ended"]["skipped"] is True
    rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
    assert all("intermediate" in row for row in rows)


def test_runner_skips_vlm_token_limit_samples(tmp_path) -> None:
    dataset_root = _write_custom_dataset(tmp_path)

    record = evaluate(_TokenLimitOnceSystem(), CustomAdapter(dataset_root), tmp_path / "results", config_bytes=b"test", seed=7)

    assert record.n_samples == 2
    result_path = tmp_path / "results" / "token_limit_once__custom.json"
    samples_path = tmp_path / "results" / "token_limit_once__custom__samples.jsonl"
    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data["n_samples"] == 2
    assert data["metrics"]["metadata"]["skipped_samples"] == 1
    rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
    skipped = [row for row in rows if row.get("skipped")]
    assert len(skipped) == 1
    assert skipped[0]["sample_id"] == "count"
    assert skipped[0]["skip_reason"] == "vlm_token_limit_exceeded"
    assert skipped[0]["vlm_usage"] == {"prompt_tokens": 1, "completion_tokens": 16000, "total_tokens": 16001}


class _TokenLimitOnceSystem:
    name = "token_limit_once"

    def __init__(self) -> None:
        self.mock = MockSystem()

    def run(self, image: Image.Image, query: str, gold_targets: list[str] | None = None):
        if getattr(image, "sample_id", None) == "count":
            raise VLMTokenLimitExceeded(
                "VLM response exceeded max_tokens=16000",
                usage={"prompt_tokens": 1, "completion_tokens": 16000, "total_tokens": 16001},
            )
        return self.mock.run(image, query, gold_targets=gold_targets)


def _write_custom_dataset(tmp_path: Path) -> Path:
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
    return dataset_root
