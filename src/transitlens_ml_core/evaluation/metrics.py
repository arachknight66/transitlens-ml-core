"""Validated binary classification metric calculation."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Stable binary performance metric collection."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float | None
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]


def calculate_binary_metrics(
    targets: Sequence[int | float] | NDArray[np.number],
    probabilities: Sequence[float] | NDArray[np.number],
    threshold: float,
) -> EvaluationMetrics:
    """Calculate all prototype binary classification metrics.

    Args:
        targets: Ground-truth binary labels.
        probabilities: Transit probabilities aligned with ``targets``.
        threshold: Probability at or above which a sample is positive.

    Returns:
        Accuracy, precision, recall, F1, ROC-AUC, and a 2×2 confusion matrix.
        ROC-AUC is ``None`` when only one target class is present.

    Raises:
        ValueError: If arrays, labels, probabilities, or threshold are invalid.

    """
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("classification threshold must be between zero and one")
    target_array = _validated_vector(targets, "targets")
    probability_array = _validated_vector(probabilities, "probabilities")
    if target_array.size != probability_array.size:
        raise ValueError("targets and probabilities must have equal lengths")
    if not np.all(np.isin(target_array, (0.0, 1.0))):
        raise ValueError("targets must contain only binary labels")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("probabilities must be in the range [0, 1]")

    binary_targets = target_array.astype(np.int64)
    predictions = (probability_array >= threshold).astype(np.int64)
    matrix = confusion_matrix(binary_targets, predictions, labels=[0, 1])
    roc_auc = (
        float(roc_auc_score(binary_targets, probability_array))
        if np.unique(binary_targets).size == 2
        else None
    )
    return EvaluationMetrics(
        accuracy=float(accuracy_score(binary_targets, predictions)),
        precision=float(precision_score(binary_targets, predictions, zero_division=0)),
        recall=float(recall_score(binary_targets, predictions, zero_division=0)),
        f1_score=float(f1_score(binary_targets, predictions, zero_division=0)),
        roc_auc=roc_auc,
        confusion_matrix=(
            (int(matrix[0, 0]), int(matrix[0, 1])),
            (int(matrix[1, 0]), int(matrix[1, 1])),
        ),
    )


def _validated_vector(
    values: Sequence[int | float] | NDArray[np.number], name: str
) -> NDArray[np.float64]:
    """Convert a finite, numeric, non-empty one-dimensional vector."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")
    result = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result
