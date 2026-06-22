from __future__ import annotations

import pytest

from scripts.run_eval import LimitedDataset


class FakeDataset:
    name = "fake"

    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


def test_limited_dataset_head_strategy_keeps_first_items() -> None:
    dataset = LimitedDataset(FakeDataset(list(range(10))), limit=3, strategy="head")

    assert list(dataset) == [0, 1, 2]
    assert len(dataset) == 3


def test_limited_dataset_random_strategy_is_reproducible_and_ordered() -> None:
    values = list(range(20))
    first = LimitedDataset(FakeDataset(values), limit=3, strategy="random", seed=123)
    second = LimitedDataset(FakeDataset(values), limit=3, strategy="random", seed=123)

    first_values = list(first)
    assert len(first_values) == 3
    assert first_values == list(second)
    assert first_values == sorted(first_values)
    assert len(first) == 3


def test_limited_dataset_random_strategy_uses_seed() -> None:
    values = list(range(100))
    first = LimitedDataset(FakeDataset(values), limit=10, strategy="random", seed=1)
    second = LimitedDataset(FakeDataset(values), limit=10, strategy="random", seed=2)

    assert list(first) != list(second)


def test_limited_dataset_none_limit_yields_all_items() -> None:
    values = list(range(10))
    dataset = LimitedDataset(FakeDataset(values), limit=None, strategy="random", seed=123)

    assert list(dataset) == values
    assert len(dataset) == 10


def test_limited_dataset_invalid_strategy_raises() -> None:
    with pytest.raises(ValueError):
        LimitedDataset(FakeDataset(list(range(10))), limit=3, strategy="middle")
