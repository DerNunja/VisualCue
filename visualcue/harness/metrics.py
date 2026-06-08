"""Pure metric functions for mask, count, consistency, and latency evaluation."""

from __future__ import annotations

import math

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

from visualcue.harness.types import GTSample, Instance, SystemOutput

DEFAULT_IOU_THRESHOLD = 0.5


def _bool_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must have shape (H, W)")
    return mask.astype(bool, copy=False)


def _rle(mask: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(_bool_mask(mask).astype(np.uint8)))


def _area(mask: np.ndarray) -> float:
    return float(mask_utils.area(_rle(mask)))


def _union_mask(instances: list[Instance], shape: tuple[int, int]) -> np.ndarray:
    masks = [_bool_mask(instance.mask) for instance in instances if instance.mask is not None]
    if not masks:
        return np.zeros(shape, dtype=bool)
    return np.logical_or.reduce(masks)


def _sample_masks(out: SystemOutput, sample: GTSample) -> tuple[np.ndarray, np.ndarray]:
    width, height = sample.image.size
    shape = (height, width)
    return _union_mask(out.instances, shape), _union_mask(sample.gt_instances, shape)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Return mask IoU; both empty is 1.0 and exactly one empty is 0.0."""

    pred_mask = _bool_mask(pred)
    gt_mask = _bool_mask(gt)
    if pred_mask.shape != gt_mask.shape:
        raise ValueError("pred and gt masks must have the same shape")

    pred_area = _area(pred_mask)
    gt_area = _area(gt_mask)
    if pred_area == 0.0 and gt_area == 0.0:
        return 1.0
    if pred_area == 0.0 or gt_area == 0.0:
        return 0.0
    return float(mask_utils.iou([_rle(pred_mask)], [_rle(gt_mask)], [0])[0, 0])


def mean_iou(per_sample: list[tuple[SystemOutput, GTSample]]) -> float:
    """Return mean union-mask IoU across samples; empty input returns 0.0."""

    if not per_sample:
        return 0.0
    return float(np.mean([iou(*_sample_masks(out, sample)) for out, sample in per_sample]))


def cumulative_iou(per_sample: list[tuple[SystemOutput, GTSample]]) -> float:
    """Return summed intersection over summed union; all empty samples return 1.0."""

    if not per_sample:
        return 0.0

    intersection_total = 0.0
    union_total = 0.0
    for out, sample in per_sample:
        pred_mask, gt_mask = _sample_masks(out, sample)
        pred_rle = _rle(pred_mask)
        gt_rle = _rle(gt_mask)
        intersection = float(mask_utils.area(mask_utils.merge([pred_rle, gt_rle], intersect=True)))
        union = float(mask_utils.area(mask_utils.merge([pred_rle, gt_rle], intersect=False)))
        intersection_total += intersection
        union_total += union
    if union_total == 0.0:
        return 1.0
    return intersection_total / union_total


def match_instances(
    pred: list[Instance],
    gt: list[Instance],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> tuple[int, int, int]:
    """Return TP/FP/FN via Hungarian IoU matching; empty sides follow detection conventions."""

    if not pred and not gt:
        return (0, 0, 0)
    if not pred:
        return (0, 0, len(gt))
    if not gt:
        return (0, len(pred), 0)

    valid_pred = [instance for instance in pred if instance.mask is not None]
    valid_gt = [instance for instance in gt if instance.mask is not None]
    if not valid_pred or not valid_gt:
        return (0, len(pred), len(gt))

    pred_rles = [_rle(instance.mask) for instance in valid_pred if instance.mask is not None]
    gt_rles = [_rle(instance.mask) for instance in valid_gt if instance.mask is not None]
    ious = mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles))

    row_indices, col_indices = linear_sum_assignment(1.0 - ious)
    tp = int(sum(ious[row, col] >= iou_threshold for row, col in zip(row_indices, col_indices)))
    fp = len(pred) - tp
    fn = len(gt) - tp
    return (tp, fp, fn)


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return precision, recall, and F1; zero denominators produce 0.0."""

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (precision, recall, f1)


def counting_errors(pred_counts: list[int], gt_counts: list[int]) -> dict[str, float]:
    """Return MAE and RMSE for counts; empty aligned inputs return zeros."""

    if len(pred_counts) != len(gt_counts):
        raise ValueError("pred_counts and gt_counts must have equal length")
    if not pred_counts:
        return {"mae": 0.0, "rmse": 0.0}
    pred_array = np.asarray(pred_counts, dtype=float)
    gt_array = np.asarray(gt_counts, dtype=float)
    errors = pred_array - gt_array
    return {"mae": float(np.mean(np.abs(errors))), "rmse": float(math.sqrt(np.mean(errors**2)))}


def answer_marking_consistency(out: SystemOutput) -> bool:
    """Return whether the stated count equals shown instances; None count is always false."""

    return out.count is not None and out.count == len(out.instances)


def latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    """Return mean, p50, and p95 latency; empty input returns zeros."""

    if not latencies_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    latency_array = np.asarray(latencies_ms, dtype=float)
    return {
        "mean": float(np.mean(latency_array)),
        "p50": float(np.percentile(latency_array, 50)),
        "p95": float(np.percentile(latency_array, 95)),
    }
