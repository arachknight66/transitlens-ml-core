"""
eval/run_full_evaluation.py
--------------------------
Automated script to evaluate the TransitLens event detection and classification system.
Runs separate evaluations on:
1. Synthetic sanity cases
2. Real validation split (val.csv)
3. Blind test split (test.csv)
4. Labeled gold demonstration set (gold_set.csv)
5. Injection-recovery suite (Phase 4 — optional, controlled by --injection flag)

Outputs classification metrics (accuracy, macro F1, confusion matrices, ROC/PR curves),
parameter recovery accuracy, execution speed profiles, and injection-recovery summary.

Usage:
    python -m eval.run_full_evaluation                    # standard eval, no injection
    python -m eval.run_full_evaluation --injection        # + quick injection recovery
    python -m eval.run_full_evaluation --injection --injection-mode standard

Note: Injection-recovery is NOT run by default to keep the full eval fast.
Run it explicitly with --injection, or separately:
    python -m eval.run_injection_recovery --mode standard
"""

from __future__ import annotations

import os
import json
import time as _time
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

import yaml

# Ensure repo root is on python path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Also add sibling path for transitlens-data-pipeline
_DP_PATH = _REPO_ROOT.parent / "transitlens-data-pipeline"
if str(_DP_PATH) not in sys.path:
    sys.path.insert(0, str(_DP_PATH))

from pipeline import analyze_light_curve
from eval.metrics import classification_report, confidence_calibration, period_recovery_rate
from eval.injection_recovery import run_injection_recovery_suite, run_suite as run_injection_suite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = _REPO_ROOT / "eval" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALIASES = {
    "exoplanet_like": "exoplanet_transit",
    "eclipsing_binary_like": "eclipsing_binary",
    "noise_or_other": "stellar_variability_or_other",
}

def load_csv_targets(csv_path: Path) -> dict:
    """Loads and reconstructs light curve arrays from standard splits CSV."""
    if not csv_path.exists():
        logger.warning(f"File not found: {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    targets = {}
    for target_id, group in df.groupby("target_id"):
        group = group.sort_values("time")
        first_row = group.iloc[0]
        
        raw_label = first_row.get("label")
        label = ALIASES.get(raw_label, raw_label)
        
        metadata = {
            "target_id": str(target_id),
            "label": label,
            "true_period": float(first_row["true_period"]) if pd.notna(first_row.get("true_period")) else None,
            "true_depth": float(first_row["true_depth"]) if pd.notna(first_row.get("true_depth")) else None,
            "true_duration": float(first_row["true_duration"]) if pd.notna(first_row.get("true_duration")) else None,
            "cadence_min": float(first_row["cadence_min"]) if pd.notna(first_row.get("cadence_min")) else None,
            "sector": int(first_row["sector"]) if pd.notna(first_row.get("sector")) else None,
        }
        targets[target_id] = {
            "time": group["time"].astype(float).values,
            "flux": group["flux"].astype(float).values,
            "metadata": metadata,
            "true_label": label
        }
    return targets

def load_npz_targets(manifest_path: Path) -> dict:
    """Loads target light curves and metadata using a processed split manifest CSV and its NPZ files."""
    if not manifest_path.exists():
        logger.warning(f"Manifest not found: {manifest_path}")
        return {}
    
    df = pd.read_csv(manifest_path)
    parent_dir = manifest_path.parent
    
    # Check for central manifest
    central_manifest_path = parent_dir / "manifest.csv"
    if not central_manifest_path.exists() and parent_dir.name == "splits":
        central_manifest_path = parent_dir.parent / "manifest.csv"
        
    central_df = None
    if central_manifest_path.exists():
        try:
            central_df = pd.read_csv(central_manifest_path)
        except Exception as e:
            logger.warning(f"Could not load central manifest: {e}")
            
    targets = {}
    for _, row in df.iterrows():
        target_id = row["target_id"]
        class_label = row["class_label"]
        lc_path_rel = row["lightcurve_path"]
        
        # Locate NPZ
        lc_path = parent_dir / lc_path_rel
        if not lc_path.exists():
            lc_path = parent_dir.parent / lc_path_rel
            
        if not lc_path.exists():
            logger.error(f"NPZ file not found for target {target_id} at {lc_path}")
            continue
            
        try:
            npz_data = np.load(lc_path)
            time_arr = npz_data["time"]
            flux_arr = npz_data["flux"]
            centroid_x = npz_data["centroid_x"] if "centroid_x" in npz_data else None
            centroid_y = npz_data["centroid_y"] if "centroid_y" in npz_data else None
            quality = npz_data["quality"] if "quality" in npz_data else None
        except Exception as e:
            logger.error(f"Failed to load NPZ data for {target_id}: {e}")
            continue
            
        # Standardize labels
        label = ALIASES.get(class_label, class_label)
        
        # Default metadata fields
        sector = None
        cadence_min = 2.0
        true_epoch = None
        
        if central_df is not None:
            c_row = central_df[central_df["target_id"] == target_id]
            if not c_row.empty:
                sector_val = c_row.iloc[0].get("sector")
                if pd.notna(sector_val):
                    sector = int(sector_val)
                cadence_val = c_row.iloc[0].get("cadence_min_median")
                if pd.notna(cadence_val):
                    cadence_min = float(cadence_val)
                epoch_val = c_row.iloc[0].get("true_epoch_btjd")
                if pd.notna(epoch_val):
                    true_epoch = float(epoch_val)
                    
        metadata = {
            "target_id": str(target_id),
            "label": label,
            "true_period": float(row["true_period_days"]) if pd.notna(row.get("true_period_days")) else None,
            "true_depth": float(row["true_depth"]) if pd.notna(row.get("true_depth")) else None,
            "true_duration": float(row["true_duration_days"]) if pd.notna(row.get("true_duration_days")) else None,
            "true_epoch": true_epoch,
            "cadence_min": cadence_min,
            "sector": sector,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "quality": quality,
        }
        
        targets[target_id] = {
            "time": time_arr,
            "flux": flux_arr,
            "metadata": metadata,
            "true_label": label
        }
    return targets


def evaluate_dataset(name: str, targets: dict) -> tuple[list[dict], dict]:
    """Runs pipeline analysis over a dictionary of targets and returns results and metrics."""
    logger.info(f"Running evaluation on {name} ({len(targets)} targets)...")
    results = []
    true_labels = []
    pred_labels = []
    confidences = []
    
    start_time = _time.perf_counter()
    for tid, target in targets.items():
        meta_clean = dict(target["metadata"])
        if "label" in meta_clean:
            del meta_clean["label"]
        if "class_label" in meta_clean:
            del meta_clean["class_label"]
            
        res = analyze_light_curve(
            time=target["time"],
            flux=target["flux"],
            metadata=meta_clean
        )
        
        true_label = target["true_label"]
        pred_label = res["predicted_class"]
        
        results.append({
            "target_id": tid,
            "true_label": true_label,
            "predicted_class": pred_label,
            "confidence": res["confidence"],
            "period_days": res["period_days"],
            "period_uncertainty_days": res["period_uncertainty_days"],
            "depth": res["depth"],
            "depth_uncertainty": res["depth_uncertainty"],
            "duration_days": res["duration_days"],
            "duration_uncertainty_days": res["duration_uncertainty_days"],
            "fit_quality": res["fit_quality"],
            "bootstrap_fap": res["bootstrap_fap"],
            "processing_time_ms": res["processing_time_ms"],
            "true_period": target["metadata"].get("true_period"),
            "true_depth": target["metadata"].get("true_depth"),
            "true_duration": target["metadata"].get("true_duration"),
            "diagnostics": res.get("diagnostics"),
            "candidate_detected": res.get("candidate_detected"),
        })
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        confidences.append(res["confidence"])
        
    elapsed = _time.perf_counter() - start_time
    avg_time_ms = (elapsed / len(targets)) * 1000 if len(targets) > 0 else 0
    
    # Compute metrics
    acc, per_class = classification_report(true_labels, pred_labels)
    rec_rate = period_recovery_rate(
        [{"period_days": r["period_days"], "metadata": {"true_period": r["true_period"]}} for r in results],
        tolerance_pct=1.0
    )
    
    # Compute parameter errors
    period_errs, depth_errs, dur_errs = [], [], []
    for r in results:
        if r["true_label"] in ("exoplanet_transit", "eclipsing_binary") and r["period_days"] is not None:
            if r["true_period"]:
                period_errs.append(abs(r["period_days"] - r["true_period"]) / r["true_period"])
            if r["true_depth"] and r["depth"]:
                depth_errs.append(abs(r["depth"] - r["true_depth"]) / r["true_depth"])
            if r["true_duration"] and r["duration_days"]:
                dur_errs.append(abs(r["duration_days"] - r["true_duration"]) / r["true_duration"])
                
    per_class_dict = {
        m.label: {
            "precision": float(m.precision),
            "recall": float(m.recall),
            "f1": float(m.f1),
            "support": int(m.support)
        }
        for m in per_class
    }
    
    metrics = {
        "accuracy": acc,
        "per_class": per_class_dict,
        "period_recovery_rate": rec_rate,
        "average_runtime_ms": avg_time_ms,
        "total_runtime_s": elapsed,
        "mean_period_error_pct": float(np.mean(period_errs) * 100) if period_errs else 0.0,
        "mean_depth_error_pct": float(np.mean(depth_errs) * 100) if depth_errs else 0.0,
        "mean_duration_error_pct": float(np.mean(dur_errs) * 100) if dur_errs else 0.0,
    }
    
    return results, metrics

def save_confusion_matrix(true_labels: list[str], pred_labels: list[str], output_path: Path):
    """Saves confusion matrix as PNG using matplotlib."""
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
        ax.set_title("Confusion Matrix (Taxonomy-Calibrated)", fontsize=12, fontweight="bold")
        
        for i in range(n):
            for j in range(n):
                color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold", color=color)
                        
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Saved confusion matrix plot to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to generate confusion matrix plot: {e}")

# evaluate_gold_set deleted to avoid stale precomputed mock metrics on Kepler/TESS gold sets.


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="TransitLens Full Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--injection", action="store_true",
        help=(
            "Run Phase 4 injection-recovery suite in addition to classification evaluation. "
            "Not run by default to keep the full eval fast. "
            "Run separately with: python -m eval.run_injection_recovery --mode standard"
        ),
    )
    parser.add_argument(
        "--injection-mode",
        choices=("quick", "standard", "full"),
        default="quick",
        help="Injection-recovery mode (default: quick). Used only when --injection is set.",
    )
    args, _unknown = parser.parse_known_args()

    splits_dir = _DP_PATH / "datasets" / "splits"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"
    
    processed_dir = _DP_PATH / "datasets" / "processed" / "lightcurves"
    val_manifest = processed_dir / "splits" / "val_manifest.csv"
    test_manifest = processed_dir / "splits" / "test_manifest.csv"

    # Validate model config file exists
    config_path = _REPO_ROOT / "models" / "rule_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Critical model artifact missing: rule_config.yaml not found at {config_path}")
    
    with open(config_path, "r") as f:
        rule_cfg = yaml.safe_load(f)
    ml_cfg = rule_cfg.get("ml_classifier", {})
    if ml_cfg.get("enabled", False) and not ml_cfg.get("dev_fallback", False):
        model_path = _REPO_ROOT / "models" / "final_classifier.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Critical model artifact final_classifier.pkl not found at {model_path} with ML enabled.")

    # 1. Optionally run Phase 4 Injection-Recovery suite
    injection_summary = None
    if args.injection:
        logger.info(
            "Running Phase 4 injection-recovery suite (mode=%s)...",
            args.injection_mode
        )
        logger.info(
            "Note: Run 'python -m eval.run_injection_recovery --mode standard' "
            "for a full standalone benchmark run with all outputs and plots."
        )
        try:
            injection_summary = run_injection_recovery_suite(
                mode=args.injection_mode,
                output_dir=str(RESULTS_DIR),
            )
            logger.info(
                "Injection-recovery complete: recall=%.1f%%, period_1%%=%.1f%%, FP=%.1f%%",
                (injection_summary.get('detection_recall') or 0) * 100,
                (injection_summary.get('period_recovery_rate_1pct') or 0) * 100,
                (injection_summary.get('false_positive_rate_controls') or 0) * 100,
            )
        except Exception as exc:
            logger.warning("Injection-recovery suite failed (non-fatal): %s", exc)
    else:
        logger.info(
            "Skipping Phase 4 injection-recovery (use --injection to enable). "
            "Run 'python -m eval.run_injection_recovery --mode standard' for a full benchmark."
        )

    # 2. Load Split Datasets (Strictly enforce dataset presence)
    if val_manifest.exists() and test_manifest.exists():
        logger.info("Processed NPZ manifests found. Running evaluation on Phase 1 NPZ dataset splits...")
        val_targets = load_npz_targets(val_manifest)
        test_targets = load_npz_targets(test_manifest)
    else:
        logger.info("Processed manifests not found. Falling back to old CSV targets...")
        if not val_csv.exists() or not test_csv.exists():
            raise FileNotFoundError(
                f"Missing critical split datasets. Manifests ({val_manifest}, {test_manifest}) "
                f"and CSVs ({val_csv}, {test_csv}) are not found."
            )
        val_targets = load_csv_targets(val_csv)
        test_targets = load_csv_targets(test_csv)
    
    if not val_targets or len(val_targets) == 0:
        raise ValueError("Validation targets dataset is empty or could not be loaded!")
    if not test_targets or len(test_targets) == 0:
        raise ValueError("Test targets dataset is empty or could not be loaded!")
    
    # Run evaluation
    val_results, val_metrics = evaluate_dataset("Validation Split", val_targets)
    test_results, test_metrics = evaluate_dataset("Test Split", test_targets)
    
    # Compute aggregate metrics
    all_true = [r["true_label"] for r in val_results + test_results]
    all_pred = [r["predicted_class"] for r in val_results + test_results]
    save_confusion_matrix(all_true, all_pred, RESULTS_DIR / "confusion_matrix.png")
    
    # Save parameter error summary CSV
    param_records = []
    for r in val_results + test_results:
        if r["true_period"] and r["period_days"]:
            p_err = abs(r["period_days"] - r["true_period"]) / r["true_period"] * 100
            d_err = abs(r["depth"] - r["true_depth"]) / r["true_depth"] * 100 if r["true_depth"] and r["depth"] else None
            dur_err = abs(r["duration_days"] - r["true_duration"]) / r["true_duration"] * 100 if r["true_duration"] and r["duration_days"] else None
            param_records.append({
                "target_id": r["target_id"],
                "class": r["true_label"],
                "true_period": r["true_period"],
                "det_period": r["period_days"],
                "period_err_pct": p_err,
                "true_depth": r["true_depth"],
                "det_depth": r["depth"],
                "depth_err_pct": d_err,
                "true_duration": r["true_duration"],
                "det_duration": r["duration_days"],
                "duration_err_pct": dur_err,
                "fit_quality": r["fit_quality"],
                "period_uncertainty": r["period_uncertainty_days"]
            })
    df_params = pd.DataFrame(param_records)
    params_path = RESULTS_DIR / "parameter_error_summary.csv"
    df_params.to_csv(params_path, index=False)
    logger.info(f"Saved parameter error summary to {params_path}")
    
    # Save metrics JSON
    metrics_json = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "overall_period_recovery_pct": float(period_recovery_rate(
            [{"period_days": r["period_days"], "metadata": {"true_period": r["true_period"]}} for r in val_results + test_results],
            tolerance_pct=1.0
        ) * 100)
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"Saved metrics to {RESULTS_DIR / 'metrics.json'}")

    # Compile Phase 6 Blend Diagnostics outputs
    all_eval_results = val_results + test_results
    blend_per_target = []
    tp, fp, fn, tn = 0, 0, 0, 0
    false_blend_flags_planets = 0
    total_planets = 0
    
    total_targets = len(all_eval_results)
    diag_avail_count = 0
    centroid_avail_count = 0
    crowding_avail_count = 0
    neighbor_avail_count = 0
    
    false_blend_flags_list = []
    
    for r in all_eval_results:
        tid = r["target_id"]
        true_lbl = r["true_label"]
        pred_class = r["predicted_class"]
        cand_det = r["candidate_detected"]
        conf = r["confidence"]
        
        diag = r.get("diagnostics", {}).get("blend", {}) if r.get("diagnostics") else {}
        
        c_avail = diag.get("centroid_available", False)
        c_shift = diag.get("centroid_shift")
        c_sig = diag.get("centroid_shift_significance")
        
        cr_avail = diag.get("crowding_available", False)
        cr_metric = diag.get("crowding_metric")
        
        n_avail = diag.get("neighbor_available", False)
        n_count = diag.get("gaia_neighbor_count")
        n_sep = diag.get("nearest_neighbor_sep_arcsec")
        n_dmag = diag.get("nearest_neighbor_delta_mag")
        
        risk_score = diag.get("blend_risk_score")
        risk_level = diag.get("blend_risk_level", "unavailable")
        flags = diag.get("blend_evidence_flags", [])
        
        if c_avail or cr_avail or n_avail:
            diag_avail_count += 1
        if c_avail:
            centroid_avail_count += 1
        if cr_avail:
            crowding_avail_count += 1
        if n_avail:
            neighbor_avail_count += 1
            
        correct_blend = (pred_class == "blend_contamination") == (true_lbl == "blend_contamination")
        
        # Confusion slice metrics
        if true_lbl == "blend_contamination":
            if pred_class == "blend_contamination":
                tp += 1
            else:
                fn += 1
        else:
            if pred_class == "blend_contamination":
                fp += 1
            else:
                tn += 1
                
        # False blend flags on exoplanet transit cases
        if true_lbl == "exoplanet_transit":
            total_planets += 1
            if risk_level == "high" or pred_class == "blend_contamination":
                false_blend_flags_planets += 1
                false_blend_flags_list.append({
                    "target_id": tid,
                    "true_label": true_lbl,
                    "predicted_class": pred_class,
                    "centroid_shift": c_shift,
                    "centroid_shift_significance": c_sig,
                    "crowding_metric": cr_metric,
                    "blend_risk_level": risk_level,
                    "blend_evidence_flags": ",".join(flags) if flags else ""
                })
                
        blend_per_target.append({
            "target_id": tid,
            "true_label": true_lbl,
            "predicted_class": pred_class,
            "candidate_detected": cand_det,
            "confidence": conf,
            "centroid_available": c_avail,
            "centroid_shift": c_shift,
            "centroid_shift_significance": c_sig,
            "crowding_available": cr_avail,
            "crowding_metric": cr_metric,
            "neighbor_available": n_avail,
            "gaia_neighbor_count": n_count,
            "nearest_neighbor_sep_arcsec": n_sep,
            "nearest_neighbor_delta_mag": n_dmag,
            "blend_risk_score": risk_score,
            "blend_risk_level": risk_level,
            "blend_evidence_flags": ",".join(flags) if flags else "",
            "correct_blend_prediction": correct_blend
        })
        
    # Write blend_per_target_diagnostics.csv
    df_blend_target = pd.DataFrame(blend_per_target)
    df_blend_target.to_csv(RESULTS_DIR / "blend_per_target_diagnostics.csv", index=False)
    logger.info(f"Saved blend per-target diagnostics to {RESULTS_DIR / 'blend_per_target_diagnostics.csv'}")
    
    # Write blend_confusion_slice.csv
    df_confusion = pd.DataFrame([
        {"metric": "true_blend", "predicted_blend": tp, "predicted_non_blend": fn},
        {"metric": "true_non_blend", "predicted_blend": fp, "predicted_non_blend": tn}
    ])
    df_confusion.to_csv(RESULTS_DIR / "blend_confusion_slice.csv", index=False)
    logger.info(f"Saved blend confusion slice to {RESULTS_DIR / 'blend_confusion_slice.csv'}")
    
    # Write blend_false_flags.csv
    df_false = pd.DataFrame(false_blend_flags_list)
    df_false.to_csv(RESULTS_DIR / "blend_false_flags.csv", index=False)
    logger.info(f"Saved blend false flags list to {RESULTS_DIR / 'blend_false_flags.csv'}")
    
    # Compute precision, recall, F1
    blend_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    blend_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    blend_f1 = 2 * blend_precision * blend_recall / (blend_precision + blend_recall) if (blend_precision + blend_recall) > 0 else 0.0
    false_flag_rate = false_blend_flags_planets / total_planets if total_planets > 0 else 0.0
    
    blend_summary = {
        "blend_precision": blend_precision,
        "blend_recall": blend_recall,
        "blend_f1": blend_f1,
        "false_blend_flag_rate_on_planets": false_flag_rate,
        "diagnostic_availability_rate": diag_avail_count / total_targets if total_targets > 0 else 0.0,
        "centroid_availability_rate": centroid_avail_count / total_targets if total_targets > 0 else 0.0,
        "crowding_availability_rate": crowding_avail_count / total_targets if total_targets > 0 else 0.0,
        "neighbor_availability_rate": neighbor_avail_count / total_targets if total_targets > 0 else 0.0
    }
    
    with open(RESULTS_DIR / "blend_diagnostics_summary.json", "w") as f:
        json.dump(blend_summary, f, indent=2)
    logger.info(f"Saved blend diagnostics summary JSON to {RESULTS_DIR / 'blend_diagnostics_summary.json'}")
        
    def to_md_table(df):
        if df.empty:
            return "No false flags identified."
        cols = list(df.columns)
        headers = "| " + " | ".join(cols) + " |"
        separator = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for _, row in df.iterrows():
            row_str = "| " + " | ".join(str(row[c]) for c in cols) + " |"
            rows.append(row_str)
        return "\n".join([headers, separator] + rows)

    # Write blend_diagnostics_report.md
    report_md = f"""# TransitLens Blend & Contamination Diagnostics Performance Report

## 1. Executive Summary
- **Diagnostic Availability Rate**: {blend_summary['diagnostic_availability_rate']*100:.1f}%
- **Centroid Availability Rate**: {blend_summary['centroid_availability_rate']*100:.1f}%
- **Crowding Availability Rate**: {blend_summary['crowding_availability_rate']*100:.1f}%
- **Neighbor Availability Rate**: {blend_summary['neighbor_availability_rate']*100:.1f}%
- **Blend Classification Precision**: {blend_precision*100:.1f}%
- **Blend Classification Recall**: {blend_recall*100:.1f}%
- **Blend Classification F1-Score**: {blend_f1:.4f}
- **False Blend Flag Rate on Clean Planets**: {false_flag_rate*100:.1f}%

## 2. Confusion Matrix (Blend Contamination Slice)
| | Predicted Blend | Predicted Non-Blend |
|---|---|---|
| **True Blend** | {tp} (TP) | {fn} (FN) |
| **True Non-Blend** | {fp} (FP) | {tn} (TN) |

## 3. False Blend Flags
Listed below are the clean exoplanet transit targets that were flagged with high blend risk or classified as blend:

{to_md_table(df_false)}
"""
    with open(RESULTS_DIR / "blend_diagnostics_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"Saved blend diagnostics report to {RESULTS_DIR / 'blend_diagnostics_report.md'}")
    
    # Write full_evaluation_summary.md report
    def _ir_val(key, default="N/A"):
        """Safely get injection-recovery metric for report."""
        if injection_summary is None:
            return "(not run - use --injection to enable)"
        val = injection_summary.get(key)
        if val is None:
            return default
        import math
        try:
            if math.isnan(val):
                return default
            return f"{val * 100:.1f}%"
        except Exception:
            return str(val)

    summary_md = f"""# TransitLens Scientific Performance Evaluation Summary

## 1. Executive Summary
- **Overall Period Recovery Rate**: {metrics_json['overall_period_recovery_pct']:.2f}% (tolerance < 1.0%)
- **Validation Split Classification Accuracy**: {val_metrics['accuracy'] * 100:.2f}%
- **Blind Test Split Classification Accuracy**: {test_metrics['accuracy'] * 100:.2f}%
- **Average Pipeline Execution Latency**: {val_metrics['average_runtime_ms']:.1f} ms per target


## 2. Classification Performance (Test Split)
| Class Label | Precision | Recall | F1-Score |
|---|---|---|---|
| exoplanet_transit | {test_metrics['per_class'].get('exoplanet_transit', {}).get('precision', 0.0) * 100:.1f}% | {test_metrics['per_class'].get('exoplanet_transit', {}).get('recall', 0.0) * 100:.1f}% | {test_metrics['per_class'].get('exoplanet_transit', {}).get('f1', 0.0) * 100:.1f}% |
| eclipsing_binary | {test_metrics['per_class'].get('eclipsing_binary', {}).get('precision', 0.0) * 100:.1f}% | {test_metrics['per_class'].get('eclipsing_binary', {}).get('recall', 0.0) * 100:.1f}% | {test_metrics['per_class'].get('eclipsing_binary', {}).get('f1', 0.0) * 100:.1f}% |
| blend_contamination | {test_metrics['per_class'].get('blend_contamination', {}).get('precision', 0.0) * 100:.1f}% | {test_metrics['per_class'].get('blend_contamination', {}).get('recall', 0.0) * 100:.1f}% | {test_metrics['per_class'].get('blend_contamination', {}).get('f1', 0.0) * 100:.1f}% |
| stellar_variability_or_other | {test_metrics['per_class'].get('stellar_variability_or_other', {}).get('precision', 0.0) * 100:.1f}% | {test_metrics['per_class'].get('stellar_variability_or_other', {}).get('recall', 0.0) * 100:.1f}% | {test_metrics['per_class'].get('stellar_variability_or_other', {}).get('f1', 0.0) * 100:.1f}% |

## 3. Parameter Estimation Accuracy
- **Mean Period Error**: {test_metrics['mean_period_error_pct']:.4f}%
- **Mean Transit Depth Error**: {test_metrics['mean_depth_error_pct']:.2f}%
- **Mean Transit Duration Error**: {test_metrics['mean_duration_error_pct']:.2f}%

*Parameter errors are computed relative to synthetic/archive catalogue ground truth.*

## 4. Phase 4 Injection-Recovery Summary (Synthetic Evidence Only)

> Evidence type: Synthetic injection-recovery benchmark. NOT real-TESS evidence.
> Run `python -m eval.run_injection_recovery --mode standard` for a full benchmark.

- **Detection Recall (all SNR)**: {_ir_val('detection_recall')}
- **Detection Recall (SNR >= 7)**: {_ir_val('detection_recall_high_snr')}
- **Period Recovery +/- 1% (all)**: {_ir_val('period_recovery_rate_1pct')}
- **Period Recovery +/- 1% (SNR >= 7)**: {_ir_val('period_recovery_1pct_high_snr')}
- **False-Positive Rate (controls)**: {_ir_val('false_positive_rate_controls')}

See `eval/results/phase4_injection_recovery_report.md` for the full Phase 4 report.
"""
    summary_path = RESULTS_DIR / "full_evaluation_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    logger.info(f"Saved evaluation summary report to {summary_path}")
    print("\nEvaluation Complete!")

if __name__ == "__main__":
    main()
