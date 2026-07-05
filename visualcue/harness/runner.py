"""Evaluation runner, metric aggregation, and result serialization."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm

from visualcue.harness.datasets.base import DatasetAdapter
from visualcue.harness.metrics import (
    answer_marking_consistency,
    counting_errors,
    cumulative_iou,
    latency_stats,
    match_instances,
    mean_iou,
    precision_recall_f1,
)
from visualcue.harness.systems.base import VisionSystem
from visualcue.harness.systems._vlm import VLMRequestError, VLMTokenLimitExceeded
from visualcue.harness.types import GTSample, Instance, ResultRecord, SystemOutput
DEFAULT_IOU_THRESHOLD = 0.5
QUALITATIVE_LIMIT = 8
PRED_OVERLAY_COLOR = (255, 0, 0, 96)
GT_OVERLAY_COLOR = (0, 255, 0, 96)


def evaluate(
    system: VisionSystem,
    dataset: DatasetAdapter,
    out_dir: Path,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    attribution: bool = False,
    config_bytes: bytes | None = None,
    seed: int | None = None,
    maintenance_interval: int | None = None,
    maintenance_command: str | list[str] | None = None,
) -> ResultRecord:
    """Evaluate one system/dataset pair; writes unique JSON, JSONL, and overlays."""

    _set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _unique_stem(out_dir, system.name, dataset.name)
    samples_path = out_dir / f"{stem}__samples.jsonl"
    qualitative_dir = out_dir / "qualitative"
    qualitative_dir.mkdir(exist_ok=True)

    per_type: dict[str, list[tuple[SystemOutput, GTSample]]] = defaultdict(list)
    gold_per_type: dict[str, list[tuple[SystemOutput, GTSample]]] = defaultdict(list)
    raw_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    skipped_samples = 0

    progress_desc = f"{system.name} / {dataset.name}"
    total_samples = _safe_len(dataset)
    processed_samples = 0
    with tqdm(dataset, total=total_samples, desc=progress_desc, unit="sample") as progress:
        for index, sample in enumerate(progress):
            setattr(sample.image, "sample_id", sample.sample_id)
            start = time.perf_counter()
            try:
                out = system.run(sample.image, sample.query)
            except (VLMRequestError, VLMTokenLimitExceeded) as exc:
                skipped_samples += 1
                raw_rows.append(_skip_row(sample, exc, (time.perf_counter() - start) * 1000.0))
                progress.write(f"skipped {sample.sample_id}: {exc}")
                out = None
            if out is not None:
                out.latency_ms = (time.perf_counter() - start) * 1000.0
                latencies.append(out.latency_ms)
                per_type[sample.query_type].append((out, sample))
                raw_rows.append(_raw_row(sample, out))
                if index < QUALITATIVE_LIMIT:
                    _write_overlay(sample, out, qualitative_dir / f"{stem}__{sample.sample_id}.png")

            if attribution and out is not None:
                gold_targets = [instance.label for instance in sample.gt_instances if instance.label]
                setattr(sample.image, "sample_id", sample.sample_id)
                gold_start = time.perf_counter()
                try:
                    gold_out = system.run(sample.image, sample.query, gold_targets=gold_targets)
                except (VLMRequestError, VLMTokenLimitExceeded) as exc:
                    progress.write(f"skipped gold-target attribution for {sample.sample_id}: {exc}")
                    gold_out = None
                if gold_out is not None:
                    gold_out.latency_ms = (time.perf_counter() - gold_start) * 1000.0
                    gold_per_type[sample.query_type].append((gold_out, sample))

            processed_samples += 1
            _maybe_run_maintenance(
                processed_samples,
                total_samples,
                maintenance_interval,
                maintenance_command,
                progress.write,
            )

    with samples_path.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    metrics = {
        "end_to_end": _metrics_by_type(per_type, iou_threshold),
        "metadata": _metadata(seed),
    }
    config_fn = getattr(system, "config", None)
    if callable(config_fn):
        metrics["metadata"]["system_config"] = config_fn()
    metrics["metadata"]["skipped_samples"] = skipped_samples
    if maintenance_interval and maintenance_command:
        metrics["metadata"]["maintenance"] = {
            "interval": maintenance_interval,
            "command": maintenance_command,
        }
    if attribution:
        metrics["gold_targets"] = _segmentation_metrics_by_type(gold_per_type, iou_threshold)

    record = ResultRecord(
        system_name=system.name,
        dataset_name=dataset.name,
        n_samples=len(raw_rows) - skipped_samples,
        metrics=metrics,
        latency=latency_stats(latencies),
        config_hash=hashlib.sha256(config_bytes or b"").hexdigest(),
        timestamp=datetime.now(UTC).isoformat(),
    )
    record_path = out_dir / f"{stem}.json"
    record_path.write_text(json.dumps(dataclasses.asdict(record), indent=2, sort_keys=True), encoding="utf-8")
    return record


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def _safe_len(dataset: DatasetAdapter) -> int | None:
    try:
        return len(dataset)
    except TypeError:
        return None


def _maybe_run_maintenance(
    processed_samples: int,
    total_samples: int | None,
    maintenance_interval: int | None,
    maintenance_command: str | list[str] | None,
    write: Any,
) -> None:
    if not maintenance_interval or not maintenance_command:
        return
    if maintenance_interval < 1:
        raise ValueError("maintenance_interval must be >= 1")
    if processed_samples % maintenance_interval != 0:
        return
    if total_samples is not None and processed_samples >= total_samples:
        return

    write(f"running maintenance command after {processed_samples} samples")
    subprocess.run(
        maintenance_command,
        check=True,
        shell=isinstance(maintenance_command, str),
    )
    write("maintenance command finished")


def _unique_stem(out_dir: Path, system_name: str, dataset_name: str) -> str:
    base = f"{system_name}__{dataset_name}"
    stem = base
    suffix = 2
    while (out_dir / f"{stem}.json").exists() or (out_dir / f"{stem}__samples.jsonl").exists():
        stem = f"{base}_{suffix}"
        suffix += 1
    return stem


def _metrics_by_type(
    per_type: dict[str, list[tuple[SystemOutput, GTSample]]],
    iou_threshold: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if referring := per_type.get("referring"):
        metrics["referring"] = _segmentation_metrics(referring, iou_threshold)
    if counting := per_type.get("counting"):
        pred_counts = [out.count if out.count is not None else len(out.instances) for out, _ in counting]
        gt_counts = [sample.gt_count if sample.gt_count is not None else len(sample.gt_instances) for _, sample in counting]
        consistency_values = [answer_marking_consistency(out) for out, _ in counting]
        metrics["counting"] = {
            **counting_errors(pred_counts, gt_counts),
            "answer_marking_consistency": float(np.mean(consistency_values)) if consistency_values else 0.0,
            "n_samples": len(counting),
        }
    if open_ended := per_type.get("open_ended"):
        metrics["open_ended"] = {"n_samples": len(open_ended), "skipped": True}
    return metrics


def _segmentation_metrics_by_type(
    per_type: dict[str, list[tuple[SystemOutput, GTSample]]],
    iou_threshold: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if referring := per_type.get("referring"):
        metrics["referring"] = _segmentation_metrics(referring, iou_threshold)
    return metrics


def _segmentation_metrics(
    samples: list[tuple[SystemOutput, GTSample]],
    iou_threshold: float,
) -> dict[str, float | int]:
    tp = fp = fn = 0
    for out, sample in samples:
        sample_tp, sample_fp, sample_fn = match_instances(out.instances, sample.gt_instances, iou_threshold)
        tp += sample_tp
        fp += sample_fp
        fn += sample_fn
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "mean_iou": mean_iou(samples),
        "cumulative_iou": cumulative_iou(samples),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_samples": len(samples),
    }


def _raw_row(sample: GTSample, out: SystemOutput) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "query_type": sample.query_type,
        "predictions": [_instance_to_json(instance) for instance in out.instances],
        "count": out.count,
        "answer": out.answer,
        "latency_ms": out.latency_ms,
        "intermediate": out.intermediate,
    }


def _skip_row(sample: GTSample, exc: VLMRequestError | VLMTokenLimitExceeded, latency_ms: float) -> dict[str, Any]:
    skip_reason = _skip_reason(exc)
    return {
        "sample_id": sample.sample_id,
        "query_type": sample.query_type,
        "skipped": True,
        "skip_reason": skip_reason,
        "error": str(exc),
        "latency_ms": latency_ms,
        "vlm_usage": exc.usage,
        "predictions": [],
        "count": None,
        "answer": None,
        "intermediate": {
            "skip_reason": skip_reason,
            "vlm_usage": [{"stage": "unknown", **exc.usage}] if exc.usage else [],
        },
    }


def _skip_reason(exc: VLMRequestError | VLMTokenLimitExceeded) -> str:
    if isinstance(exc, VLMTokenLimitExceeded):
        return "vlm_token_limit_exceeded"
    return "vlm_request_error"


def _instance_to_json(instance: Instance) -> dict[str, Any]:
    return {
        "mask": _mask_to_rle_json(instance.mask),
        "bbox": instance.bbox,
        "label": instance.label,
        "score": instance.score,
    }


def _mask_to_rle_json(mask: np.ndarray | None) -> dict[str, Any] | None:
    if mask is None:
        return None
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(value) for value in rle["size"]], "counts": rle["counts"].decode("ascii")}


def _write_overlay(sample: GTSample, out: SystemOutput, path: Path) -> None:
    image = sample.image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for instance in sample.gt_instances:
        _draw_mask(draw, instance.mask, GT_OVERLAY_COLOR)
    for instance in out.instances:
        _draw_mask(draw, instance.mask, PRED_OVERLAY_COLOR)
    Image.alpha_composite(image, overlay).convert("RGB").save(path)


def _draw_mask(draw: ImageDraw.ImageDraw, mask: np.ndarray | None, color: tuple[int, int, int, int]) -> None:
    if mask is None:
        return
    ys, xs = np.where(mask)
    for x, y in zip(xs.tolist(), ys.tolist()):
        draw.point((x, y), fill=color)


def _metadata(seed: int | None) -> dict[str, Any]:
    packages = ["numpy", "Pillow", "pycocotools", "scipy"]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    try:
        import torch
    except ImportError:
        torch_version = None
    else:
        torch_version = torch.__version__
    versions["torch"] = torch_version
    return {"seed": seed, "versions": versions}
