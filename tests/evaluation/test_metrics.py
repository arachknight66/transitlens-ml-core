"""Tests for binary evaluation metrics."""

import numpy as np
import pytest

from transitlens_ml_core.evaluation import calculate_binary_metrics


def test_calculate_binary_metrics_returns_exact_expected_values() -> None:
    metrics = calculate_binary_metrics(
        targets=[0, 0, 1, 1],
        probabilities=[0.1, 0.6, 0.4, 0.9],
        threshold=0.5,
    )

    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1_score == 0.5
    assert metrics.roc_auc == 0.75
    assert metrics.confusion_matrix == ((1, 1), (1, 1))


def test_threshold_is_inclusive_and_confusion_matrix_is_always_two_by_two() -> None:
    metrics = calculate_binary_metrics([0, 0], [0.1, 0.5], threshold=0.5)

    assert metrics.confusion_matrix == ((1, 1), (0, 0))
    assert metrics.roc_auc is None
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.1])
def test_metrics_reject_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        calculate_binary_metrics([0, 1], [0.1, 0.9], threshold)


@pytest.mark.parametrize(
    ("targets", "probabilities", "message"),
    [
        ([], [], "must not be empty"),
        ([[0, 1]], [0.1, 0.9], "one-dimensional"),
        ([0, 1], [[0.1, 0.9]], "one-dimensional"),
        ([0, 1], [0.1], "equal lengths"),
        ([0, 2], [0.1, 0.9], "binary labels"),
        ([0, 1], [-0.1, 0.9], r"range \[0, 1\]"),
        ([0, 1], [0.1, 1.1], r"range \[0, 1\]"),
        ([0, 1], [0.1, np.nan], "finite"),
        ([0, 1], [0.1, np.inf], "finite"),
        ([0, 1], ["bad", "data"], "numeric"),
    ],
)
def test_metrics_reject_invalid_vectors(
    targets: object, probabilities: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        calculate_binary_metrics(targets, probabilities, 0.5)  # type: ignore[arg-type]
