"""Deterministic train, validation, and test dataset splitting."""

from dataclasses import dataclass
from typing import TypeVar

import torch
from torch.utils.data import Dataset, Subset

Sample = TypeVar("Sample")


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """Non-overlapping train, validation, and test subsets."""

    train: Subset[object]
    validation: Subset[object]
    test: Subset[object]


def split_dataset(
    dataset: Dataset[Sample],
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> DatasetSplits:
    """Split a dataset reproducibly into three non-empty subsets.

    Args:
        dataset: Dataset to partition.
        train_fraction: Desired training proportion.
        validation_fraction: Desired validation proportion.
        test_fraction: Desired test proportion.
        seed: Non-negative seed controlling index shuffling.

    Returns:
        Deterministically shuffled, non-overlapping dataset subsets.

    Raises:
        ValueError: If fractions, seed, or dataset size are invalid.

    """
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(fraction <= 0.0 or fraction >= 1.0 for fraction in fractions):
        raise ValueError("split fractions must each be between zero and one")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")
    if seed < 0:
        raise ValueError("split seed must be non-negative")
    if len(dataset) < 3:
        raise ValueError("dataset must contain at least three samples")

    lengths = _split_lengths(len(dataset), fractions)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    train_end = lengths[0]
    validation_end = train_end + lengths[1]
    return DatasetSplits(
        train=Subset(dataset, indices[:train_end]),
        validation=Subset(dataset, indices[train_end:validation_end]),
        test=Subset(dataset, indices[validation_end:]),
    )


def _split_lengths(
    size: int, fractions: tuple[float, float, float]
) -> tuple[int, int, int]:
    """Allocate rounded split sizes while keeping every subset non-empty."""
    exact = [size * fraction for fraction in fractions]
    lengths = [int(value) for value in exact]
    remainder = size - sum(lengths)
    priorities = sorted(
        range(3),
        key=lambda index: (exact[index] - lengths[index], -index),
        reverse=True,
    )
    for index in priorities[:remainder]:
        lengths[index] += 1

    for empty_index, length in enumerate(lengths):
        if length == 0:
            donor_index = max(range(3), key=lengths.__getitem__)
            lengths[donor_index] -= 1
            lengths[empty_index] += 1
    return lengths[0], lengths[1], lengths[2]
