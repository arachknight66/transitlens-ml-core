# cli.py
# ------
# Orchestration CLI for Phase 2 diagnostic evaluations.

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd

from eval.phase2.evaluate_eb import evaluate_eb_diagnostics
from eval.phase2.evaluate_blends import evaluate_blend_diagnostics
from eval.phase2.evaluate_missingness import evaluate_missingness_rates
from eval.phase2.evaluate_thresholds import evaluate_threshold_sensitivity
from eval.phase2.plots import generate_evaluation_plots

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("eval_phase2")

def main():
    parser = argparse.ArgumentParser(description="TransitLens Phase 2 Scientific Evaluation CLI")
    parser.add_argument("--features-dir", default="../data/manifests/phase1", help="Path to materialized feature parquets")
    parser.add_argument("--output-dir", default="runs/phase2/evaluation", help="Directory to save evaluation reports and plots")
    
    args = parser.parse_args()
    features_dir = Path(args.features_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting Phase 2 diagnostic evaluation using features from: {features_dir}")
    
    # Check if validation features table exists
    val_features_path = features_dir / "phase2_features_validation.parquet"
    if not val_features_path.exists():
        logger.error(f"Validation feature table missing: {val_features_path}")
        sys.exit(1)
        
    df_val = pd.read_parquet(val_features_path)
    
    # Run evaluations
    eb_report = evaluate_eb_diagnostics(df_val)
    blend_report = evaluate_blend_diagnostics(df_val)
    missingness_report = evaluate_missingness_rates(df_val)
    sensitivity_report = evaluate_threshold_sensitivity(df_val)
    
    # Generate plots
    generate_evaluation_plots(df_val, out_dir)
    
    # Save reports
    metrics = {
        "eb_diagnostics": eb_report,
        "blend_diagnostics": blend_report,
        "missingness": missingness_report,
        "sensitivity": sensitivity_report,
    }
    
    metrics_path = out_dir / "phase2_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Write confusion matrices and reports
    pd.DataFrame([eb_report]).to_csv(out_dir / "eb_classification_report.csv", index=False)
    pd.DataFrame([blend_report]).to_csv(out_dir / "blend_classification_report.csv", index=False)
    
    logger.info(f"Phase 2 evaluation completed successfully. Reports saved in {out_dir}")

if __name__ == "__main__":
    main()
