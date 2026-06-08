from __future__ import annotations

import math

import numpy as np

from visualcue.harness.metrics import (
    answer_marking_consistency,
    counting_errors,
    iou,
    match_instances,
    precision_recall_f1,
)
from visualcue.harness.types import Instance, SystemOutput


def _mask(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((10, 10), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _instance(mask: np.ndarray | None = None) -> Instance:
    return Instance(mask=mask if mask is not None else np.ones((10, 10), dtype=bool), bbox=None, label=None, score=None)


def test_iou_identical_full_masks() -> None:
    assert iou(np.ones((10, 10), dtype=bool), np.ones((10, 10), dtype=bool)) == 1.0


def test_iou_disjoint_halves() -> None:
    assert iou(_mask(0, 0, 5, 10), _mask(5, 0, 10, 10)) == 0.0


def test_iou_partial_overlap() -> None:
    assert iou(_mask(0, 0, 10, 5), np.ones((10, 10), dtype=bool)) == 0.5


def test_iou_both_empty() -> None:
    assert iou(np.zeros((10, 10), dtype=bool), np.zeros((10, 10), dtype=bool)) == 1.0


def test_match_perfect() -> None:
    gt = [_instance(_mask(0, 0, 3, 3)), _instance(_mask(5, 5, 8, 8))]
    pred = [_instance(_mask(0, 0, 3, 3)), _instance(_mask(5, 5, 8, 8))]
    assert match_instances(pred, gt) == (2, 0, 0)


def test_match_with_fp() -> None:
    gt = [_instance(_mask(0, 0, 3, 3)), _instance(_mask(5, 5, 8, 8))]
    pred = [_instance(_mask(0, 0, 3, 3)), _instance(_mask(5, 5, 8, 8)), _instance(_mask(8, 0, 10, 2))]
    assert match_instances(pred, gt) == (2, 1, 0)


def test_match_with_fn() -> None:
    gt = [_instance(_mask(0, 0, 3, 3)), _instance(_mask(5, 5, 8, 8))]
    pred = [_instance(_mask(0, 0, 3, 3))]
    assert match_instances(pred, gt) == (1, 0, 1)


def test_precision_recall_f1_values() -> None:
    precision, recall, _ = precision_recall_f1(tp=2, fp=1, fn=0)
    assert precision == 2 / 3
    assert recall == 1.0


def test_counting_errors() -> None:
    errors = counting_errors([3, 5], [3, 4])
    assert errors["mae"] == 0.5
    assert math.isclose(errors["rmse"], 0.707, rel_tol=1e-3)


def test_consistency_true() -> None:
    out = SystemOutput(instances=[_instance(), _instance(), _instance()], answer=None, count=3, latency_ms=0.0, intermediate={})
    assert answer_marking_consistency(out) is True


def test_consistency_none() -> None:
    out = SystemOutput(instances=[_instance(), _instance(), _instance()], answer=None, count=None, latency_ms=0.0, intermediate={})
    assert answer_marking_consistency(out) is False
