"""Tests for decision confidence estimation."""

import pytest

from transitlens_ml_core.inference import estimate_confidence


@pytest.mark.parametrize(
    ("probability", "threshold", "expected"),
    [
        (0.0, 0.5, 1.0),
        (0.25, 0.5, 0.5),
        (0.5, 0.5, 0.0),
        (0.75, 0.5, 0.5),
        (1.0, 0.5, 1.0),
        (0.2, 0.2, 0.0),
        (0.6, 0.2, 0.5),
    ],
)
def test_estimate_confidence_is_normalized_threshold_distance(
    probability: float, threshold: float, expected: float
) -> None:
    assert estimate_confidence(probability, threshold) == pytest.approx(expected)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_estimate_confidence_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        estimate_confidence(probability, 0.5)


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")])
def test_estimate_confidence_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        estimate_confidence(0.5, threshold)
