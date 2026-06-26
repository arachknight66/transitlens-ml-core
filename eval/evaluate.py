"""
eval/evaluate.py
-----------------
Evaluation runner for the TransitLens ML Core pipeline.

Runs analyze_light_curve() on the labeled dataset from data-pipeline (or
built-in synthetic data) and produces:
    1. Classification report (text)  → eval/results/classification_report.txt
    2. Confusion matrix (PNG)        → eval/results/confusion_matrix.png
    3. Per-target results (CSV)      → eval/results/per_target_results.csv

Usage:
    python -m eval.evaluate          # runs from repo root
    python eval/evaluate.py          # direct execution

Used by: standalone script, CI
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on the path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline import analyze_light_curve
from eval.metrics import (
    classification_report,
    confidence_calibration,
    format_report,
    period_recovery_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = _REPO_ROOT / "eval" / "results"


# ---------------------------------------------------------------------------
# Synthetic test cases (built-in fallback)
# ---------------------------------------------------------------------------

def _generate_synthetic_cases() -> list[dict]:
    """
    Generate the three standard synthetic test cases.

    Returns a list of dicts, each with keys: target_id, time, flux, metadata.
    """
    rng = np.random.default_rng(42)
    n_points = 18000
    time = np.linspace(0.0, 27.0, n_points)
    noise_level = 0.001

    cases = []

    # Candidate A — exoplanet (P=3.42d, depth=1.3%)
    flux_a = 1.0 + rng.normal(0, noise_level, n_points)
    period_a, depth_a, duration_a = 3.42, 0.013, 0.12
    phase_a = ((time - 1.5) / period_a) % 1.0
    in_transit_a = (phase_a < duration_a / period_a) | (phase_a > 1.0 - duration_a / period_a)
    flux_a[in_transit_a] -= depth_a
    cases.append({
        "target_id": "candidate_a",
        "time": time.copy(),
        "flux": flux_a,
        "metadata": {
            "target_id": "candidate_a",
            "true_period": period_a,
            "true_depth": depth_a,
            "true_label": "exoplanet_like",
        },
    })

    # Candidate B — eclipsing binary (P=1.87d, depth=18%)
    flux_b = 1.0 + rng.normal(0, noise_level, n_points)
    period_b, depth_b, duration_b = 1.87, 0.18, 0.15
    half_phase_b = (duration_b / period_b) / 2.0
    phase_b = ((time - 0.8) / period_b) % 1.0
    for i, ph in enumerate(phase_b):
        if ph < half_phase_b:
            flux_b[i] -= depth_b * (1.0 - ph / half_phase_b)
        elif ph > 1.0 - half_phase_b:
            flux_b[i] -= depth_b * (1.0 - (1.0 - ph) / half_phase_b)
        elif abs(ph - 0.5) < half_phase_b:
            flux_b[i] -= 0.08 * (1.0 - abs(ph - 0.5) / half_phase_b)
    cases.append({
        "target_id": "candidate_b",
        "time": time.copy(),
        "flux": flux_b,
        "metadata": {
            "target_id": "candidate_b",
            "true_period": period_b,
            "true_depth": depth_b,
            "true_label": "eclipsing_binary_like",
        },
    })

    # Candidate C — noise
    flux_c = 1.0 + rng.normal(0, noise_level, n_points)
    cases.append({
        "target_id": "candidate_c",
        "time": time.copy(),
        "flux": flux_c,
        "metadata": {
            "target_id": "candidate_c",
            "true_label": "noise_or_other",
        },
    })

    return cases


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    """
    Run the full evaluation pipeline and write results.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load or generate test cases
    cases = _generate_synthetic_cases()
    logger.info("evaluate: running pipeline on %d cases", len(cases))

    # Run pipeline on each case
    results = []
    true_labels = []
    pred_labels = []
    confidences = []

    for case in cases:
        logger.info("evaluate: processing %s", case["target_id"])
        result = analyze_light_curve(
            time=case["time"],
            flux=case["flux"],
            metadata=case["metadata"],
        )

        aliases = {
            "exoplanet_like": "exoplanet_transit",
            "eclipsing_binary_like": "eclipsing_binary",
            "noise_or_other": "stellar_variability_or_other",
        }
        true_label = aliases.get(case["metadata"].get("true_label", "unknown"), "stellar_variability_or_other")
        pred_label = aliases.get(result["predicted_class"], result["predicted_class"])

        results.append({**result, "metadata": case["metadata"]})
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        confidences.append(result["confidence"])

        logger.info(
            "  → true=%s pred=%s conf=%.3f period=%s",
            true_label, pred_label, result["confidence"],
            f"{result['period_days']:.4f}" if result["period_days"] else "N/A",
        )

    # Compute metrics
    accuracy, per_class = classification_report(true_labels, pred_labels)
    period_rate = period_recovery_rate(results, tolerance_pct=1.0)
    mean_correct, mean_incorrect = confidence_calibration(
        true_labels, pred_labels, confidences
    )

    # Format and save classification report
    report_text = format_report(
        accuracy, per_class, period_rate, mean_correct, mean_incorrect
    )
    report_path = RESULTS_DIR / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("evaluate: classification report saved to %s", report_path)
    print("\n" + report_text)

    # Save confusion matrix as PNG
    _save_confusion_matrix(true_labels, pred_labels, RESULTS_DIR / "confusion_matrix.png")

    # Save per-target results CSV
    _save_per_target_csv(results, true_labels, RESULTS_DIR / "per_target_results.csv")

    logger.info("evaluate: all results saved to %s", RESULTS_DIR)


def _save_confusion_matrix(
    true_labels: list[str],
    pred_labels: list[str],
    output_path: Path,
) -> None:
    """Generate and save a confusion matrix as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        classes = ["exoplanet_transit", "eclipsing_binary", "blend_contamination", "stellar_variability_or_other"]
        short = ["Planet", "EB", "Blend", "Noise"]
        n = len(classes)
        matrix = np.zeros((n, n), dtype=int)

        for t, p in zip(true_labels, pred_labels):
            if t in classes and p in classes:
                matrix[classes.index(t), classes.index(p)] += 1

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="Blues", vmin=0)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short, fontsize=10)
        ax.set_yticklabels(short, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

        for i in range(n):
            for j in range(n):
                color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        fontsize=16, fontweight="bold", color=color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("evaluate: confusion matrix saved to %s", output_path)

    except Exception as exc:
        logger.warning("evaluate: failed to save confusion matrix: %s", exc)


def _save_per_target_csv(
    results: list[dict],
    true_labels: list[str],
    output_path: Path,
) -> None:
    """Save per-target results to CSV."""
    fieldnames = [
        "target_id", "true_label", "predicted_label",
        "true_period", "detected_period", "period_error_pct",
        "confidence", "processing_time_ms",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result, true_label in zip(results, true_labels):
            meta = result.get("metadata", {}) or {}
            true_period = meta.get("true_period")
            detected_period = result.get("period_days")

            period_error = None
            if true_period and detected_period:
                period_error = abs(detected_period - true_period) / true_period * 100

            writer.writerow({
                "target_id": result.get("target_id", "unknown"),
                "true_label": true_label,
                "predicted_label": result.get("predicted_class", ""),
                "true_period": f"{true_period:.6f}" if true_period else "",
                "detected_period": f"{detected_period:.6f}" if detected_period else "",
                "period_error_pct": f"{period_error:.4f}" if period_error is not None else "",
                "confidence": f"{result.get('confidence', 0):.6f}",
                "processing_time_ms": f"{result.get('processing_time_ms', 0):.2f}",
            })

    logger.info("evaluate: per-target CSV saved to %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
