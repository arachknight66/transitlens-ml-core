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
SPLITS_DIR = DATA_PIPELINE_DIR / "datasets" / "splits"
PROCESSED_LC_DIR = DATA_PIPELINE_DIR / "datasets" / "processed" / "lightcurves"
MANIFEST_SPLITS_DIR = PROCESSED_LC_DIR / "splits"

def normalize_target_id(raw_id, prefix="KIC"):
    """Normalizes IDs to standard TransitLens formats like KIC-123456 or TIC-123456."""
    if pd.isnull(raw_id):
        return None
    cleaned = str(raw_id).strip().upper().replace("TIC", "").replace("KIC", "").replace("-", "")
    try:
        cleaned = str(int(float(cleaned)))
    except ValueError:
        pass
    return f"{prefix}-{cleaned}"

def load_catalog_lookups() -> dict[str, tuple[float | None, float | None, float | None]]:
    """Loads period, depth, and duration lookups from catalog csv files."""
    lookup = {}
    
    # Kepler cumulative
    kep_path = REPO_ROOT / "archive" / "cumulative.csv"
    if kep_path.exists():
        logger.info("Loading Kepler catalogue from %s", kep_path)
        df_kep = pd.read_csv(kep_path)
        for _, row in df_kep.iterrows():
            kepid = row.get("kepid")
            if pd.notnull(kepid):
                tid = normalize_target_id(kepid, "KIC")
                period = float(row.get("koi_period")) if pd.notnull(row.get("koi_period")) else None
                depth = (float(row.get("koi_depth")) / 1e6) if pd.notnull(row.get("koi_depth")) else None
                duration = (float(row.get("koi_duration")) / 24.0) if pd.notnull(row.get("koi_duration")) else None
                lookup[tid] = (period, depth, duration)
                
    # TESS TOI
    toi_path = REPO_ROOT / "archive" / "TOI_2026.06.25_21.21.19.csv"
    if toi_path.exists():
        logger.info("Loading TESS TOI catalogue from %s", toi_path)
        df_toi = pd.read_csv(toi_path, comment="#")
        for _, row in df_toi.iterrows():
            ticid = row.get("tid")
            if pd.notnull(ticid):
                tid = normalize_target_id(ticid, "TIC")
                period = float(row.get("pl_orbper")) if pd.notnull(row.get("pl_orbper")) else None
                depth = (float(row.get("pl_trandep")) / 1e6) if pd.notnull(row.get("pl_trandep")) else None
                duration = (float(row.get("pl_trandurh")) / 24.0) if pd.notnull(row.get("pl_trandurh")) else None
                lookup[tid] = (period, depth, duration)
                
    return lookup

def generate_pseudo_lightcurve(
    label: str, 
    period: float | None, 
    depth: float | None, 
    duration: float | None, 
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a pseudo light curve stochastically for a class."""
    n = 18000
    time = np.linspace(0.0, 27.0, n)
    noise = 0.001
    flux = 1.0 + rng.normal(0, noise, n)
    
    if label == "exoplanet_transit":
        if period and depth and duration:
            t0 = float(rng.uniform(0.1, min(period, 5.0)))
            phase = ((time - t0) / period) % 1.0
            in_transit = (phase < duration / period) | (phase > 1.0 - duration / period)
            flux[in_transit] -= depth
            
    elif label == "eclipsing_binary":
        if period and depth and duration:
            t0 = float(rng.uniform(0.1, min(period, 5.0)))
            phase = ((time - t0) / period) % 1.0
            hp = (duration / period) / 2.0
            for i, ph in enumerate(phase):
                if ph < hp:
                    flux[i] -= depth * (1.0 - ph / hp)
                elif ph > 1.0 - hp:
                    flux[i] -= depth * (1.0 - (1.0 - ph) / hp)
                elif abs(ph - 0.5) < hp:
                    # Secondary eclipse at phase 0.5 with smaller depth
                    flux[i] -= (depth * 0.4) * (1.0 - abs(ph - 0.5) / hp)
                    
    elif label == "blend_contamination":
        if period and depth and duration:
            t0 = float(rng.uniform(0.1, min(period, 5.0)))
            # Dilute the depth to simulate blend
            diluted_depth = depth * 0.4
            phase = ((time - t0) / period) % 1.0
            in_transit = (phase < duration / period) | (phase > 1.0 - duration / period)
            flux[in_transit] -= diluted_depth
            
    elif label == "stellar_variability_or_other":
        # Add stellar variability pattern (sine wave + noise)
        amp = float(rng.uniform(0.002, 0.008))
        var_period = float(rng.uniform(2.0, 15.0))
        flux += amp * np.sin(2.0 * np.pi * time / var_period)
        
    return time, flux

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
        metadata["target_id"] = target_id
        metadata["class_label"] = label
        metadata["label"] = label
        feature_result = extract(time_clean, flux_clean, bls_result, metadata=metadata)
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
    
    output_file = output_dir / f"{split_name}_features.csv"
    if resume and output_file.exists():
        logger.info("Resume mode: Loading existing feature file %s", output_file.name)
        df_existing = pd.read_csv(output_file)
        split_rows = df_existing.to_dict(orient="records")
        processed_targets = set(df_existing["target_id"].tolist())
        logger.info("Loaded %d already processed targets", len(processed_targets))
        
    # Get manifest targets for this split
    split_targets_df = manifest_df[manifest_df["split"] == split_name]
    logger.info("Found %d targets in manifest for split %s", len(split_targets_df), split_name)
    
    tasks = []
    
    for idx, target in split_targets_df.iterrows():
        tid = target["target_id"]
        class_label = target["class_label"]
        
        if tid in processed_targets:
            continue
            
        processed_path = target.get("processed_path")
        if pd.isnull(processed_path) or not processed_path or not os.path.exists(processed_path):
            split_rows.append({
                "target_id": tid,
                "class_label": class_label,
                "split": split_name,
                "processing_status": "failed",
                "failure_reason": "Processed light curve (NPZ) missing or download failed",
                "candidate_detected": 0,
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
                "true_period": target.get("period_days"),
                "true_duration": target.get("duration_days"),
                "true_depth": target.get("depth_ppm", 0.0) / 1e6,
            }
            if "quality" in data:
                meta["quality"] = data["quality"]
            if "centroid_x" in data:
                meta["centroid_x"] = data["centroid_x"]
            if "centroid_y" in data:
                meta["centroid_y"] = data["centroid_y"]
            if "flux_err" in data:
                meta["flux_err"] = data["flux_err"]
                
            tasks.append((tid, class_label, time_arr, flux_arr, meta, target))
            processed_targets.add(tid)
        except Exception as exc:
            logger.error("Failed to load NPZ for target %s: %s", tid, exc)
            split_rows.append({
                "target_id": tid,
                "class_label": class_label,
                "split": split_name,
                "processing_status": "failed",
                "failure_reason": f"NPZ load failed: {exc}",
                "candidate_detected": 0,
                "true_period_days": target.get("period_days", 0.0),
                "true_duration_days": target.get("duration_days", 0.0),
                "true_depth": target.get("depth_ppm", 0.0) / 1e6,
                **{k: 0.0 for k in FEATURE_NAMES}
            })
            
    # Process tasks in parallel
    if tasks:
        logger.info("Processing %d real targets in parallel...", len(tasks))
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(process_single_target, tid, lbl, t_arr, f_arr, meta): (tid, lbl, target)
                for tid, lbl, t_arr, f_arr, meta, target in tasks
            }
            completed_count = 0
            for fut in concurrent.futures.as_completed(futures):
                tid, lbl, target = futures[fut]
                try:
                    row_feat = fut.result()
                    row_feat["split"] = split_name
                    # Preserve catalogue truth in separate non-feature columns
                    row_feat["true_period_days"] = target.get("period_days", 0.0)
                    row_feat["true_duration_days"] = target.get("duration_days", 0.0)
                    row_feat["true_depth"] = target.get("depth_ppm", 0.0) / 1e6
                    
                    # Verify all feature values are finite
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

def generate_split(
    split_name: str,
    target_count_per_class: int,
    lookups: dict,
    resume: bool,
    rng: np.random.Generator
) -> list[dict]:
    """Generates the feature list for a specific split (train, val, test) stochastically."""
    logger.info("--- Generating split stochastically: %s ---", split_name)
    split_rows = []
    processed_targets = set()
    
    output_file = (REPO_ROOT / "transitlens-ml-core" / "data" / "ml_features") / f"{split_name}_features.csv"
    if resume and output_file.exists():
        logger.info("Resume mode: Loading existing feature file %s", output_file.name)
        df_existing = pd.read_csv(output_file)
        split_rows = df_existing.to_dict(orient="records")
        processed_targets = set(df_existing["target_id"].tolist())
        logger.info("Loaded %d already processed targets", len(processed_targets))
        
    # 1. Load manifest targets first
    manifest_file = MANIFEST_SPLITS_DIR / f"{split_name}_manifest.csv"
    manifest_targets = []
    if manifest_file.exists():
        df_manifest = pd.read_csv(manifest_file)
        manifest_targets = df_manifest.to_dict(orient="records")
        logger.info("Found %d manifest targets in %s", len(manifest_targets), manifest_file.name)
        
    # Process manifest targets
    class_counts = {c: 0 for c in CLASSES}
    for row in split_rows:
        if row["processing_status"] == "success":
            class_counts[row["class_label"]] += 1
            
    tasks = []
    aliases = {
        "exoplanet_like": "exoplanet_transit",
        "eclipsing_binary_like": "eclipsing_binary",
        "noise_or_other": "stellar_variability_or_other"
    }

    for target in manifest_targets:
        tid = target["target_id"]
        class_label = target["class_label"]
        class_label = aliases.get(class_label, class_label)
        
        if tid in processed_targets:
            continue
            
        npz_path = PROCESSED_LC_DIR / target["lightcurve_path"]
        if not npz_path.exists():
            logger.warning("Manifest npz path %s does not exist", npz_path)
            continue
            
        data = np.load(npz_path)
        time_arr = data["time"]
        flux_arr = data["flux"]
        meta = {}
        if "quality" in data:
            meta["quality"] = data["quality"]
        if "centroid_x" in data:
            meta["centroid_x"] = data["centroid_x"]
        if "centroid_y" in data:
            meta["centroid_y"] = data["centroid_y"]
        if "flux_err" in data:
            meta["flux_err"] = data["flux_err"]
            
        tasks.append((tid, class_label, time_arr, flux_arr, meta))
        processed_targets.add(tid)

    # 2. Load splits target definitions to sample extra targets stochastically
    targets_file = SPLITS_DIR / f"{split_name}_targets.csv"
    if not targets_file.exists():
        logger.error("Targets file %s does not exist", targets_file)
        return split_rows
        
    df_targets = pd.read_csv(targets_file)
    logger.info("Loaded %d catalog targets from %s", len(df_targets), targets_file.name)
    
    # Process stochastically for each class to reach target_count_per_class
    for class_name in CLASSES:
        needed = target_count_per_class - class_counts[class_name]
        if needed <= 0:
            logger.info("Class %s already has %d samples, no more needed.", class_name, class_counts[class_name])
            continue
            
        logger.info("Sampling %d targets of class %s", needed, class_name)
        class_targets = df_targets[df_targets["label"] == class_name]
        sampled_targets = class_targets.sample(frac=1.0, random_state=42).to_dict(orient="records")
        
        sampled_count = 0
        for target in sampled_targets:
            if sampled_count >= needed:
                break
            tid = target["target_id"]
            if tid in processed_targets:
                continue
                
            per, dep, dur = lookups.get(tid, (None, None, None))
            if per is None or dep is None or dur is None:
                if class_name == "exoplanet_transit":
                    per = float(rng.uniform(1.0, 15.0))
                    dep = float(rng.uniform(0.001, 0.015))
                    dur = float(rng.uniform(0.04, 0.20))
                elif class_name == "eclipsing_binary":
                    per = float(rng.uniform(0.5, 5.0))
                    dep = float(rng.uniform(0.05, 0.25))
                    dur = float(rng.uniform(0.05, 0.20))
                elif class_name == "blend_contamination":
                    per = float(rng.uniform(1.0, 10.0))
                    dep = float(rng.uniform(0.005, 0.03))
                    dur = float(rng.uniform(0.05, 0.20))
                else:
                    per, dep, dur = None, None, None
            
            time_arr, flux_arr = generate_pseudo_lightcurve(class_name, per, dep, dur, rng)
            meta = {}
            tasks.append((tid, class_name, time_arr, flux_arr, meta))
            processed_targets.add(tid)
            sampled_count += 1

    # 3. Process all scheduled tasks in parallel
    if tasks:
        logger.info("Processing %d targets in parallel...", len(tasks))
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(process_single_target, tid, lbl, t_arr, f_arr, meta): (tid, lbl)
                for tid, lbl, t_arr, f_arr, meta in tasks
            }
            completed_count = 0
            for fut in concurrent.futures.as_completed(futures):
                tid, lbl = futures[fut]
                try:
                    row_feat = fut.result()
                    row_feat["split"] = split_name
                    row_feat["true_period_days"] = 0.0
                    row_feat["true_duration_days"] = 0.0
                    row_feat["true_depth"] = 0.0
                    split_rows.append(row_feat)
                    if row_feat["processing_status"] == "success":
                        class_counts[row_feat["class_label"]] += 1
                except Exception as exc:
                    logger.error("Task failed for target %s: %s", tid, exc)
                completed_count += 1
                if completed_count % 50 == 0:
                    logger.info("Progress: %d/%d completed for split %s", completed_count, len(tasks), split_name)
                    
    return split_rows

def main():
    parser = argparse.ArgumentParser(description="TransitLens Phase 5 Feature Matrix Generator")
    parser.add_argument("--limit", type=int, default=None, help="Force limit of samples per class")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all", help="Which split to generate")
    parser.add_argument("--resume", action="store_true", help="Resume from partially generated files")
    parser.add_argument("--cache", action="store_true", help="Use cache if available")
    
    # Real-only flags
    parser.add_argument("--real-only", action="store_true", help="Run in strict real-only mode without synthetic data")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest Parquet file (required for real-only)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory to write features")
    args = parser.parse_args()
    
    # Establish output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent / "data" / "ml_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(42)
    
    splits_to_process = ["train", "val", "test"] if args.split == "all" else [args.split]
    report_data = []
    
    if args.real_only:
        if not args.manifest:
            logger.error("--manifest is required in --real-only mode.")
            sys.exit(1)
            
        logger.info(f"Running in --real-only mode. Loading manifest: {args.manifest}")
        manifest_df = pd.read_parquet(args.manifest)
        
        # In real-only mode, we strictly only process downloaded/cached targets
        downloaded_manifest = manifest_df[manifest_df["download_status"].isin(["downloaded", "cached"])]
        logger.info(f"Loaded {len(downloaded_manifest)} successfully acquired targets from manifest.")
        
        for split in splits_to_process:
            rows = generate_split_real_only(split, downloaded_manifest, args.resume, output_dir)
            df_split = pd.DataFrame(rows)
            
            # Ensure correct column ordering
            cols = ["target_id", "class_label", "split"] + list(FEATURE_NAMES) + [
                "candidate_detected", "predicted_class_rule", "processing_status", "failure_reason",
                "true_period_days", "true_duration_days", "true_depth"
            ]
            for c in cols:
                if c not in df_split.columns:
                    df_split[c] = ""
            df_split = df_split[cols]
            
            out_csv = output_dir / f"{split}_features.csv"
            df_split.to_csv(out_csv, index=False)
            logger.info("Saved feature matrix to %s", out_csv)
            
            # Gather stats
            status_counts = df_split["processing_status"].value_counts().to_dict()
            class_distribution = df_split[df_split["processing_status"] == "success"]["class_label"].value_counts().to_dict()
            report_data.append({
                "split": split,
                "total": len(df_split),
                "success": status_counts.get("success", 0),
                "failed": status_counts.get("failed", 0),
                "classes": class_distribution
            })
            
            # Write failed target reports in --real-only mode
            failed_targets = df_split[df_split["processing_status"] == "failed"]
            if len(failed_targets) > 0:
                fail_report_path = output_dir / f"{split}_failed_targets_report.csv"
                failed_targets[["target_id", "class_label", "failure_reason"]].to_csv(fail_report_path, index=False)
                logger.info(f"Saved failed-target report for {split} split to {fail_report_path}")
                
    else:
        # Load catalog lookups (for default synthetic fallback mode)
        lookups = load_catalog_lookups()
        
        target_counts = {
            "train": args.limit or 150,
            "val": args.limit or 50,
            "test": args.limit or 50
        }
        
        for split in splits_to_process:
            rows = generate_split(split, target_counts[split], lookups, args.resume, rng)
            df_split = pd.DataFrame(rows)
            
            cols = ["target_id", "class_label", "split"] + list(FEATURE_NAMES) + [
                "candidate_detected", "predicted_class_rule", "processing_status", "failure_reason",
                "true_period_days", "true_duration_days", "true_depth"
            ]
            for c in cols:
                if c not in df_split.columns:
                    df_split[c] = ""
            df_split = df_split[cols]
            
            out_csv = output_dir / f"{split}_features.csv"
            df_split.to_csv(out_csv, index=False)
            logger.info("Saved feature matrix to %s", out_csv)
            
            status_counts = df_split["processing_status"].value_counts().to_dict()
            class_distribution = df_split[df_split["processing_status"] == "success"]["class_label"].value_counts().to_dict()
            report_data.append({
                "split": split,
                "total": len(df_split),
                "success": status_counts.get("success", 0),
                "failed": status_counts.get("failed", 0),
                "classes": class_distribution
            })
            
    # Write feature generation report
    report_path = output_dir / "feature_generation_report.md"
    report_lines = [
        "# Feature Generation Report",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Split | Total Targets | Success | Failed | Exoplanet Transit | Eclipsing Binary | Blend Contam | Stellar Var/Other |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for r in report_data:
        cls_dist = r["classes"]
        report_lines.append(
            f"| {r['split']} | {r['total']} | {r['success']} | {r['failed']} | "
            f"{cls_dist.get('exoplanet_transit', 0)} | {cls_dist.get('eclipsing_binary', 0)} | "
            f"{cls_dist.get('blend_contamination', 0)} | {cls_dist.get('stellar_variability_or_other', 0)} |"
        )
        
    report_lines.extend([
        "",
        "## Configuration",
        f"- Real-only mode: {args.real_only}",
        f"- Resume: {args.resume}",
        f"- Feature count: {len(FEATURE_NAMES)} features",
        "",
        "## Excluded / Included Features",
        "The feature matrix retains metadata columns for downstream evaluation, but ML models MUST only use the following feature columns for training and evaluation:",
        ""
    ])
    for fname in FEATURE_NAMES:
        report_lines.append(f"- `{fname}`")
        
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Report written to %s", report_path)
    
    print("\nFeature matrix generation complete!\n")

if __name__ == "__main__":
    main()