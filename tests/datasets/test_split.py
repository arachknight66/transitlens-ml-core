"""Tests for deterministic dataset splitting."""

import pytest
from torch.utils.data import Dataset

from transitlens_ml_core.datasets import split_dataset


class IntegerDataset(Dataset[int]):
    """Small index-valued dataset fixture."""

    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> int:
        return index


def test_split_dataset_is_complete_non_overlapping_and_deterministic() -> None:
    dataset = IntegerDataset(20)

    first = split_dataset(dataset, 0.7, 0.15, 0.15, seed=42)
    second = split_dataset(dataset, 0.7, 0.15, 0.15, seed=42)

    assert (len(first.train), len(first.validation), len(first.test)) == (14, 3, 3)
    assert first.train.indices == second.train.indices
    groups = [
        set(first.train.indices),
        set(first.validation.indices),
        set(first.test.indices),
    ]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert set.union(*groups) == set(range(20))


def test_split_dataset_seed_changes_assignment() -> None:
    dataset = IntegerDataset(10)

    first = split_dataset(dataset, 0.6, 0.2, 0.2, seed=1)
    second = split_dataset(dataset, 0.6, 0.2, 0.2, seed=2)

    assert first.train.indices != second.train.indices


def test_split_dataset_keeps_small_splits_non_empty() -> None:
    splits = split_dataset(IntegerDataset(3), 0.7, 0.15, 0.15, seed=0)

    assert (len(splits.train), len(splits.validation), len(splits.test)) == (1, 1, 1)


@pytest.mark.parametrize(
    ("fractions", "seed", "size", "message"),
    [
        ((0.0, 0.5, 0.5), 1, 10, "between zero and one"),
        ((0.6, 0.3, 0.2), 1, 10, "sum to 1.0"),
        ((0.7, 0.15, 0.15), -1, 10, "non-negative"),
        ((0.7, 0.15, 0.15), 1, 2, "at least three"),
    ],
)
def test_split_dataset_rejects_invalid_inputs(
    fractions: tuple[float, float, float], seed: int, size: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        split_dataset(IntegerDataset(size), *fractions, seed=seed)
