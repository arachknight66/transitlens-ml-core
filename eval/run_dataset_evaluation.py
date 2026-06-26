"""
eval/run_dataset_evaluation.py
------------------------------
Runs evaluation specifically on a processed dataset manifest.
Supports loading time/flux arrays from target .npz files.

Usage:
    python -m eval.run_dataset_evaluation --manifest ../transitlens-data-pipeline/datasets/processed/lightcurves/splits/test_manifest.csv
"""

from __future__ import annotations

import os
import argparse
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure repo root is on python path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Sibling path
_DP_PATH = _REPO_ROOT.parent / "transitlens-data-pipeline"
if str(_DP_PATH) not in sys.path:
    sys.path.insert(0, str(_DP_PATH))

from pipeline import analyze_light_curve
from eval.metrics import classification_report, period_recovery_rate
from eval.run_full_evaluation import load_npz_targets, evaluate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate TransitLens model on a processed manifest.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV file.")
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(f"Manifest file not found: {manifest_path}")
        sys.exit(1)
        
    logger.info(f"Loading targets from manifest: {manifest_path}")
    targets = load_npz_targets(manifest_path)
    
    if not targets:
        logger.error("No valid targets loaded from manifest.")
        sys.exit(1)
        
    results, metrics = evaluate_dataset(manifest_path.name, targets)
    
    # Print results summary
    print("\n" + "=" * 60)
    print(f"  Evaluation Results for: {manifest_path.name}")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Period Recovery Rate: {metrics['period_recovery_rate'] * 100:.2f}%")
    print(f"Average Runtime: {metrics['average_runtime_ms']:.2f} ms/target")
    print("-" * 60)
    print("Per-class Metrics:")
    for cls, dist in metrics["per_class"].items():
        print(f"  {cls:<30} | F1: {dist['f1'] * 100:.1f}% | P: {dist['precision'] * 100:.1f}% | R: {dist['recall'] * 100:.1f}% | Support: {dist['support']}")
    
    print("-" * 60)
    print("Parameter Error Summary:")
    print(f"  Mean Period Error: {metrics['mean_period_error_pct']:.4f}%")
    print(f"  Mean Depth Error: {metrics['mean_depth_error_pct']:.4f}%")
    print(f"  Mean Duration Error: {metrics['mean_duration_error_pct']:.4f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
