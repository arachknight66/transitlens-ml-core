"""
eval/metrics.py
---------------
Metric computation for evaluating the TransitLens ML Core pipeline.

Functions:
    period_recovery_rate  — fraction of targets with detected period within tolerance
    classification_report — per-class precision, recall, F1 and overall accuracy
    confidence_calibration — mean confidence for correct vs. incorrect predictions

Used by: eval/evaluate.py
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import NamedTuple

logger = logging.getLogger(__name__)

CLASSES = ("exoplanet_like", "eclipsing_binary_like", "noise_or_other")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class ClassMetrics(NamedTuple):
    """Per-class precision, recall, F1."""
    label: str
    precision: float
    recall: float
    f1: float
    support: int  # number of true instances


class EvalSummary(NamedTuple):
    """Overall evaluation summary."""
    accuracy: float
    per_class: list[ClassMetrics]
    period_recovery_rate: float
    mean_confidence_correct: float
    mean_confidence_incorrect: float


# ---------------------------------------------------------------------------
# Period recovery
# ---------------------------------------------------------------------------

def period_recovery_rate(
    results: list[dict],
    tolerance_pct: float = 1.0,
) -> float:
    """
    Fraction of targets with a known true_period where the detected period
    is within tolerance_pct% of the true period.

    Parameters
    ----------
    results : list[dict]
        Each dict must have 'metadata' (with optional 'true_period') and
        'period_days' from the pipeline result.
    tolerance_pct : float
        Acceptable relative error in percent (default: 1%).

    Returns
    -------
    float
        Recovery rate in [0.0, 1.0]. Returns 1.0 if no targets have true_period.
    """
    eligible = 0
    recovered = 0

    for r in results:
        meta = r.get("metadata", {}) or {}
        true_period = meta.get("true_period")
        detected_period = r.get("period_days")

        if true_period is None or true_period <= 0:
            continue

        eligible += 1

        if detected_period is not None and detected_period > 0:
            rel_error = abs(detected_period - true_period) / true_period * 100
            if rel_error <= tolerance_pct:
                recovered += 1

    if eligible == 0:
        return 1.0

    rate = recovered / eligible
    logger.info(
        "period_recovery: %d/%d recovered within %.1f%% (rate=%.3f)",
        recovered, eligible, tolerance_pct, rate,
    )
    return rate


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_report(
    true_labels: list[str],
    pred_labels: list[str],
) -> tuple[float, list[ClassMetrics]]:
    """
    Compute per-class precision, recall, F1 and overall accuracy.

    Parameters
    ----------
    true_labels, pred_labels : list[str]
        Paired lists of ground-truth and predicted class labels.

    Returns
    -------
    (accuracy, per_class_metrics)
        accuracy : float in [0, 1]
        per_class_metrics : list of ClassMetrics (one per class in CLASSES)
    """
    n = len(true_labels)
    if n == 0:
        return 0.0, []

    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    accuracy = correct / n

    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)

    per_class = []
    for cls in CLASSES:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = true_counts.get(cls, 0)
        per_class.append(ClassMetrics(cls, precision, recall, f1, support))

    return accuracy, per_class


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------

def confidence_calibration(
    true_labels: list[str],
    pred_labels: list[str],
    confidences: list[float],
) -> tuple[float, float]:
    """
    Compute mean confidence for correct vs. incorrect predictions.

    Returns
    -------
    (mean_correct, mean_incorrect)
        Mean confidence for correct predictions and incorrect predictions.
        If no correct/incorrect predictions exist, returns (0.0, 0.0).
    """
    correct_confs = [c for t, p, c in zip(true_labels, pred_labels, confidences) if t == p]
    incorrect_confs = [c for t, p, c in zip(true_labels, pred_labels, confidences) if t != p]

    mean_correct = sum(correct_confs) / len(correct_confs) if correct_confs else 0.0
    mean_incorrect = sum(incorrect_confs) / len(incorrect_confs) if incorrect_confs else 0.0

    return mean_correct, mean_incorrect


# ---------------------------------------------------------------------------
# Text report formatter
# ---------------------------------------------------------------------------

def format_report(
    accuracy: float,
    per_class: list[ClassMetrics],
    period_rate: float,
    mean_correct: float,
    mean_incorrect: float,
) -> str:
    """Format a human-readable classification report string."""
    lines = [
        "=" * 72,
        "TransitLens ML Core — Classification Report",
        "=" * 72,
        "",
        f"Overall Accuracy: {accuracy:.1%}",
        f"Period Recovery Rate: {period_rate:.1%} (within 1% tolerance)",
        "",
        f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
        "-" * 72,
    ]

    for m in per_class:
        lines.append(
            f"{m.label:<25} {m.precision:>10.3f} {m.recall:>10.3f} "
            f"{m.f1:>10.3f} {m.support:>10d}"
        )

    lines.extend([
        "-" * 72,
        "",
        f"Mean confidence (correct predictions):   {mean_correct:.3f}",
        f"Mean confidence (incorrect predictions): {mean_incorrect:.3f}",
        "",
        "Note: evaluated on synthetic data only. Real TESS performance may vary.",
        "=" * 72,
    ])

    return "\n".join(lines)
