"""
eval/benchmark.py
-----------------
Speed benchmarks for the TransitLens ML Core pipeline.

Runs analyze_light_curve() on each synthetic case multiple times and reports:
    - Mean, std, and 95th percentile processing time
    - Per-stage time breakdown (preprocessing, BLS, features, classification, plotting)

Output:
    eval/results/benchmark_summary.csv

Usage:
    python -m eval.benchmark
    python eval/benchmark.py

Used by: standalone script, CI
"""

from __future__ import annotations

import csv
import logging
import sys
import time as _time
from pathlib import Path

import numpy as np

# Ensure repo root is on the path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.evaluate import _generate_synthetic_cases

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = _REPO_ROOT / "eval" / "results"
N_ITERATIONS = 5  # number of timed runs per case


def run_benchmark() -> None:
    """
    Run speed benchmarks on the three synthetic cases.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = _generate_synthetic_cases()
    rows: list[dict] = []

    print("\n" + "=" * 72)
    print("TransitLens ML Core — Speed Benchmark")
    print("=" * 72)

    for case in cases:
        target_id = case["target_id"]
        n_points = len(case["time"])
        times_ms = []

        logger.info("benchmark: running %d iterations for %s (%d points)",
                     N_ITERATIONS, target_id, n_points)

        for i in range(N_ITERATIONS):
            # Instrument per-stage timing
            stage_times = _run_instrumented(case)
            total_ms = sum(stage_times.values())
            times_ms.append(total_ms)

            if i == 0:
                first_stages = stage_times  # save first run for breakdown

        times_arr = np.array(times_ms)
        mean_ms = float(np.mean(times_arr))
        std_ms = float(np.std(times_arr))
        p95_ms = float(np.percentile(times_arr, 95))

        row = {
            "target_id": target_id,
            "n_points": n_points,
            "n_iterations": N_ITERATIONS,
            "mean_ms": f"{mean_ms:.1f}",
            "std_ms": f"{std_ms:.1f}",
            "p95_ms": f"{p95_ms:.1f}",
            "preprocess_ms": f"{first_stages.get('preprocess', 0):.1f}",
            "bls_ms": f"{first_stages.get('bls', 0):.1f}",
            "features_ms": f"{first_stages.get('features', 0):.1f}",
            "classification_ms": f"{first_stages.get('classification', 0):.1f}",
            "plotting_ms": f"{first_stages.get('plotting', 0):.1f}",
        }
        rows.append(row)

        print(f"\n  {target_id} ({n_points} points):")
        print(f"    Mean: {mean_ms:.0f} ms  |  Std: {std_ms:.0f} ms  |  P95: {p95_ms:.0f} ms")
        print(f"    Breakdown: preprocess={first_stages.get('preprocess', 0):.0f}ms"
              f"  BLS={first_stages.get('bls', 0):.0f}ms"
              f"  features={first_stages.get('features', 0):.0f}ms"
              f"  classification={first_stages.get('classification', 0):.0f}ms"
              f"  plotting={first_stages.get('plotting', 0):.0f}ms")

    # Save CSV
    csv_path = RESULTS_DIR / "benchmark_summary.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Results saved to {csv_path}")
    print("=" * 72)
    logger.info("benchmark: complete — results saved to %s", csv_path)


def _run_instrumented(case: dict) -> dict[str, float]:
    """
    Run the pipeline with per-stage timing instrumentation.

    Returns a dict mapping stage name → elapsed milliseconds.
    """
    from core.preprocess import clean
    from core.bls_detector import detect
    from core.feature_extractor import extract
    from core.classifier import classify
    from core.confidence import score
    from pipeline import _load_config, _generate_plots

    time_arr = np.asarray(case["time"], dtype=float)
    flux_arr = np.asarray(case["flux"], dtype=float)
    cfg = _load_config()

    stages: dict[str, float] = {}

    # Preprocessing
    t0 = _time.perf_counter()
    result = clean(time_arr, flux_arr, config=cfg.get("preprocessing", {}))
    stages["preprocess"] = (_time.perf_counter() - t0) * 1000
    time_clean, flux_clean = result.time, result.flux

    # BLS
    t0 = _time.perf_counter()
    bls_result = detect(time_clean, flux_clean, config=cfg.get("bls", {}))
    stages["bls"] = (_time.perf_counter() - t0) * 1000

    # Features
    t0 = _time.perf_counter()
    feature_result = extract(time_clean, flux_clean, bls_result, config=cfg.get("features", {}))
    stages["features"] = (_time.perf_counter() - t0) * 1000

    # Classification + confidence
    t0 = _time.perf_counter()
    clf_result = classify(feature_result.features, config={"classification": cfg.get("classification", {})})
    _ = score(feature_result.features, clf_result.predicted_class)
    stages["classification"] = (_time.perf_counter() - t0) * 1000

    # Plotting
    t0 = _time.perf_counter()
    _ = _generate_plots(
        time=time_arr, flux=flux_arr,
        time_clean=time_clean, flux_clean=flux_clean,
        bls_result=bls_result,
        target_id=case["target_id"],
        cfg=cfg,
    )
    stages["plotting"] = (_time.perf_counter() - t0) * 1000

    return stages


if __name__ == "__main__":
    run_benchmark()
