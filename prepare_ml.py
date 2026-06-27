"""
prepare_ml.py
-------------
Phase 5 Feature Matrix Generator.

Loads target-disjoint splits, generates features using pipeline extraction (with stochastically 
simulated light curves for catalog-only targets by default, or strictly real-only targets), 
and outputs train, validation, and test feature matrices for classifier training.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# Add current directory and repo root to system path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "transitlens-ml-core"))

from core.preprocess import clean
from core.bls_detector import detect
from core.feature_extractor import extract, FEATURE_NAMES
from core.classifier import classify
from core.exceptions import InvalidInputError, InsufficientDataError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
CLASSES = ["exoplanet_transit", "eclipsing_binary", "blend_contamination", "stellar_variability_or_other"]
DATA_PIPELINE_DIR = REPO_ROOT / "transitlens-data-pipeline"
PROCESSED_LC_DIR = DATA_PIPELINE_DIR / "datasets" / "processed" / "lightcurves"

def process_single_target(
    target_id: str,
    label: str,
    time_arr: np.ndarray,
    flux_arr: np.ndarray,
    metadata: dict
) -> dict:
    """Processes a single target through clean, detect, and extract, returning a feature row dict."""
    row = {
        "target_id": target_id,
        "class_label": label,
        "processing_status": "success",
        "failure_reason": ""
    }
    
    try:
        preprocess_result = clean(time_arr, flux_arr)
        time_clean = preprocess_result.time
        flux_clean = preprocess_result.flux
    except (InvalidInputError, InsufficientDataError) as exc:
        row["processing_status"] = "failed"
        row["failure_reason"] = f"Preprocessing: {exc}"
        row["candidate_detected"] = 0
        for k in FEATURE_NAMES:
            row[k] = 0.0
        return row
        
    try:
        bls_result = detect(time_clean, flux_clean)
        row["candidate_detected"] = int(bls_result.candidate_detected)
    except Exception as exc:
        row["processing_status"] = "failed"
        row["failure_reason"] = f"BLS: {exc}"
        row["candidate_detected"] = 0
        for k in FEATURE_NAMES:
            row[k] = 0.0
        return row
        
    try:
        # Pass only observational metadata needed for features
        meta_extract = {
            "sector": int(metadata.get("sector", 78)),
            "tic_id": int(metadata.get("tic_id", 0)),
            "target_id": target_id,
            "class_label": label,
            "label": label
        }
        if "quality" in metadata:
            meta_extract["quality"] = metadata["quality"]
        if "centroid_x" in metadata:
            meta_extract["centroid_x"] = metadata["centroid_x"]
        if "centroid_y" in metadata:
            meta_extract["centroid_y"] = metadata["centroid_y"]
        if "flux_err" in metadata:
            meta_extract["flux_err"] = metadata["flux_err"]
            
        feature_result = extract(time_clean, flux_clean, bls_result, metadata=meta_extract)
        features = feature_result.features
        for k in FEATURE_NAMES:
            row[k] = float(features[k])
            
        # Run rule based classifier to get fallback class
        rule_res = classify(features, config={"ml_classifier": {"enabled": False}})
        row["predicted_class_rule"] = rule_res.predicted_class
    except Exception as exc:
        row["processing_status"] = "failed"
        row["failure_reason"] = f"Features: {exc}"
        for k in FEATURE_NAMES:
            row[k] = 0.0
        row["predicted_class_rule"] = "stellar_variability_or_other"
        
    return row

def generate_split_real_only(
    split_name: str,
    manifest_df: pd.DataFrame,
    resume: bool,
    output_dir: Path
) -> list[dict]:
    """Generates the feature list for a specific split using strictly real data from manifest."""
    logger.info("--- Generating real-only split: %s ---", split_name)
    split_rows = []
    processed_targets = set()
    
    # In resume mode, load existing files if available
    output_file = output_dir / f"{split_name}_features.csv"
    if resume and output_file.exists():
        logger.info("Resume mode: Loading existing feature file %s", output_file.name)
        df_existing = pd.read_csv(output_file)
        split_rows = df_existing.to_dict(orient="records")
        processed_targets = set(df_existing["target_id"].tolist())
        logger.info("Loaded %d already processed targets", len(processed_targets))
        
    split_targets_df = manifest_df[manifest_df["split"] == split_name]
    logger.info("Found %d targets in manifest for split %s", len(split_targets_df), split_name)
    
    tasks = []
    
    for idx, target in split_targets_df.iterrows():
        tid = target["target_id"]
        class_label = target["class_label"]
        manifest_status = target.get("processing_status", "success")
        
        if tid in processed_targets:
            continue
            
        if manifest_status == "failed":
            split_rows.append({
                "target_id": tid,
                "tic_id": int(target["tic_id"]),
                "sector": int(target["sector"]) if pd.notnull(target.get("sector")) else 78,
                "class_label": class_label,
                "split": split_name,
                "processing_status": "failed",
                "failure_reason": target.get("failure_reason", "Photometry extraction failed"),
                "true_period_days": target.get("period_days", 0.0),
                "true_duration_days": target.get("duration_days", 0.0),
                "true_depth": target.get("depth_ppm", 0.0) / 1e6,
                **{k: 0.0 for k in FEATURE_NAMES}
            })
            continue
            
        processed_path = target.get("processed_path")
        if pd.isnull(processed_path) or not processed_path or not os.path.exists(processed_path):
            split_rows.append({
                "target_id": tid,
                "tic_id": int(target["tic_id"]),
                "sector": int(target["sector"]) if pd.notnull(target.get("sector")) else 78,
                "class_label": class_label,
                "split": split_name,
                "processing_status": "failed",
                "failure_reason": "Processed light curve (NPZ) missing",
                "true_period_days": target.get("period_days", 0.0),
                "true_duration_days": target.get("duration_days", 0.0),
                "true_depth": target.get("depth_ppm", 0.0) / 1e6,
                **{k: 0.0 for k in FEATURE_NAMES}
            })
            continue
            
        try:
            data = np.load(processed_path)
            time_arr = data["time"]
            flux_arr = data["flux"]
            
            meta = {
                "sector": int(target.get("sector", 78)),
                "tic_id": int(target.get("tic_id", 0)),
                "target_id": tid
            }
            if "quality" in data:
                meta["quality"] = data["quality"]
            if "centroid_x" in data:
                meta["centroid_x"] = data["centroid_x"]
            if "centroid_y" in data:
                meta["centroid_y"] = data["centroid_y"]
            if "flux_err" in data:
                meta["flux_err"] = data["flux_err"]
                
            tasks.append((tid, class_label, time_arr, flux_arr, meta, target, manifest_status))
            processed_targets.add(tid)
        except Exception as exc:
            logger.error("Failed to load NPZ for target %s: %s", tid, exc)
            split_rows.append({
                "target_id": tid,
                "tic_id": int(target["tic_id"]),
                "sector": int(target["sector"]) if pd.notnull(target.get("sector")) else 78,
                "class_label": class_label,
                "split": split_name,
                "processing_status": "failed",
                "failure_reason": f"NPZ load failed: {exc}",
                "true_period_days": target.get("period_days", 0.0),
                "true_duration_days": target.get("duration_days", 0.0),
                "true_depth": target.get("depth_ppm", 0.0) / 1e6,
                **{k: 0.0 for k in FEATURE_NAMES}
            })
            
    # Process tasks in parallel
    if tasks:
        logger.info("Processing %d real targets in parallel...", len(tasks))
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_single_target, tid, lbl, t_arr, f_arr, meta): (tid, lbl, target, status)
                for tid, lbl, t_arr, f_arr, meta, target, status in tasks
            }
            completed_count = 0
            for fut in concurrent.futures.as_completed(futures):
                tid, lbl, target, status = futures[fut]
                try:
                    row_feat = fut.result()
                    row_feat["split"] = split_name
                    row_feat["tic_id"] = int(target["tic_id"])
                    row_feat["sector"] = int(target["sector"])
                    
                    row_feat["true_period_days"] = target.get("period_days", 0.0)
                    row_feat["true_duration_days"] = target.get("duration_days", 0.0)
                    row_feat["true_depth"] = target.get("depth_ppm", 0.0) / 1e6
                    
                    if status == "suspicious" and row_feat["processing_status"] == "success":
                        row_feat["processing_status"] = "suspicious"
                        row_feat["failure_reason"] = target.get("failure_reason", "Photometry marked suspicious")
                        
                    for k in FEATURE_NAMES:
                        if not np.isfinite(row_feat[k]):
                            row_feat[k] = 0.0
                            row_feat["processing_status"] = "failed"
                            row_feat["failure_reason"] = f"Non-finite feature {k}"
                            
                    split_rows.append(row_feat)
                except Exception as exc:
                    logger.error("Task failed for target %s: %s", tid, exc)
                completed_count += 1
                if completed_count % 50 == 0:
                    logger.info("Progress: %d/%d completed for split %s", completed_count, len(tasks), split_name)
                    
    return split_rows

def main():
    parser = argparse.ArgumentParser(description="TransitLens Phase 5 Feature Matrix Generator")
    parser.add_argument("--manifest", required=True, help="Path to manifest Parquet file")
    parser.add_argument("--output-dir", default="transitlens-ml-core/data/real_ml_features", help="Directory to save output feature CSVs")
    parser.add_argument("--include-suspicious", action="store_true", help="Include suspicious targets in training features")
    parser.add_argument("--resume", action="store_true", help="Resume processing from existing CSVs")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running in --real-only mode. Loading manifest: {args.manifest}")
    manifest_df = pd.read_parquet(args.manifest)
    
    downloaded_manifest = manifest_df[manifest_df["download_status"].isin(["downloaded", "cached"])]
    logger.info(f"Loaded {len(downloaded_manifest)} successfully acquired targets from manifest.")
    
    splits_to_process = ["train", "val", "test"]
    report_data = []
    
    all_failed_rows = []
    all_suspicious_rows = []
    
    for split in splits_to_process:
        rows = generate_split_real_only(split, downloaded_manifest, args.resume, output_dir)
        df_all = pd.DataFrame(rows)
        
        if len(df_all) == 0:
            df_success = pd.DataFrame()
            df_failed = pd.DataFrame()
            df_susp = pd.DataFrame()
        else:
            df_success = df_all[df_all["processing_status"] == "success"]
            df_failed = df_all[df_all["processing_status"] == "failed"]
            df_susp = df_all[df_all["processing_status"] == "suspicious"]
            
        all_failed_rows.extend(df_failed.to_dict(orient="records"))
        all_suspicious_rows.extend(df_susp.to_dict(orient="records"))
        
        if args.include_suspicious:
            df_to_save = pd.concat([df_success, df_susp], ignore_index=True) if len(df_susp) > 0 else df_success
        else:
            df_to_save = df_success
            
        cols = ["target_id", "tic_id", "sector", "class_label", "split", "processing_status", "failure_reason",
                "true_period_days", "true_duration_days", "true_depth"] + list(FEATURE_NAMES)
                
        if len(df_to_save) > 0:
            for c in cols:
                if c not in df_to_save.columns:
                    df_to_save[c] = ""
            df_to_save = df_to_save[cols]
        else:
            df_to_save = pd.DataFrame(columns=cols)
            
        out_csv = output_dir / f"{split}_features.csv"
        df_to_save.to_csv(out_csv, index=False)
        logger.info("Saved feature matrix with %d entries to %s", len(df_to_save), out_csv)
        
        status_counts = df_all["processing_status"].value_counts().to_dict() if len(df_all) > 0 else {}
        class_distribution = df_to_save["class_label"].value_counts().to_dict() if len(df_to_save) > 0 else {}
        
        report_data.append({
            "split": split,
            "total": len(df_all),
            "success": len(df_to_save),
            "failed": status_counts.get("failed", 0),
            "suspicious": status_counts.get("suspicious", 0),
            "classes": class_distribution
        })
        
    pd.DataFrame(all_failed_rows).to_csv(output_dir / "failed_targets.csv", index=False)
    pd.DataFrame(all_suspicious_rows).to_csv(output_dir / "suspicious_targets.csv", index=False)
    
    train_df = pd.read_csv(output_dir / "train_features.csv")
    val_df = pd.read_csv(output_dir / "val_features.csv")
    test_df = pd.read_csv(output_dir / "test_features.csv")
    
    train_tics = set(train_df["tic_id"].dropna().astype(int))
    val_tics = set(val_df["tic_id"].dropna().astype(int))
    test_tics = set(test_df["tic_id"].dropna().astype(int))
    
    train_val_overlap = list(train_tics.intersection(val_tics))
    train_test_overlap = list(train_tics.intersection(test_tics))
    val_test_overlap = list(val_tics.intersection(test_tics))
    
    assert len(train_val_overlap) == 0, f"Target leakage detected between train & val splits! Common TICs: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Target leakage detected between train & test splits! Common TICs: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Target leakage detected between val & test splits! Common TICs: {val_test_overlap}"
    
    integrity_report = {
        "unique_tics": {
            "train": len(train_tics),
            "val": len(val_tics),
            "test": len(test_tics)
        },
        "overlaps": {
            "train_val": train_val_overlap,
            "train_test": train_test_overlap,
            "val_test": val_test_overlap
        },
        "integrity_passed": True
    }
    with open(output_dir / "split_integrity_report.json", "w") as f:
        json.dump(integrity_report, f, indent=2)
    logger.info("Saved split integrity report.")
    
    report_path = output_dir / "feature_generation_report.md"
    report_lines = [
        "# Feature Generation Report",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Split | Total Targets | Success Features | Failed | Suspicious | Exoplanet Transit | Stellar Var/Other |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for r in report_data:
        cls_dist = r["classes"]
        report_lines.append(
            f"| {r['split']} | {r['total']} | {r['success']} | {r['failed']} | {r['suspicious']} | "
            f"{cls_dist.get('exoplanet_transit', 0)} | {cls_dist.get('stellar_variability_or_other', 0)} |"
        )
    report_lines.extend([
        "",
        "## Configuration",
        "- Real-only mode: True",
        f"- Resume: {args.resume}",
        f"- Include Suspicious: {args.include_suspicious}",
        f"- Feature count: {len(FEATURE_NAMES)} features",
        "",
        "## Checked Features List",
        ""
    ])
    for fname in FEATURE_NAMES:
        report_lines.append(f"- `{fname}`")
        
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Report written to %s", report_path)
    print("\nFeature matrix generation complete!\n")

if __name__ == "__main__":
    main()