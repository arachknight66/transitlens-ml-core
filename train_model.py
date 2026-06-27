"""
train_model.py
--------------
Unified training script for TransitLens Phase 5.

Loads train/val/test feature matrices, performs dataset sufficiency check,
trains majority, rule-based, and ML models, runs probability calibration,
saves model artifacts, and writes model_card.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
import hashlib
import os
import yaml
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.metrics import classification_report, confusion_matrix

# Ensure workspace on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "transitlens-ml-core"))

from core.classifier import TransitLensClassifier, CLASSES
from core.feature_extractor import FEATURE_NAMES
from core.preprocess import DEFAULT_CONFIG as PREPROCESS_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def get_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    if not os.path.exists(filepath):
        return "missing"
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_package_versions():
    """Retrieve versions of key scientific dependencies."""
    versions = {}
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "astropy", "astroquery", "lightkurve", "joblib", "tenacity"]:
        try:
            name = "scikit-learn" if pkg == "sklearn" else pkg
            import importlib.metadata
            versions[pkg] = importlib.metadata.version(name)
        except Exception:
            try:
                mod = __import__(pkg)
                versions[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                versions[pkg] = "not_installed"
    return versions

# Helper function to compute ECE
def compute_ece(y_true_indices, y_prob, n_bins=10):
    if len(y_true_indices) == 0:
        return 0.0
    ece = 0.0
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true_indices)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)

# Helper function to compute Brier Score
def compute_brier_score(y_true_onehot, y_prob):
    if len(y_true_onehot) == 0:
        return 0.0
    return float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))

def check_sufficiency(train_df, val_df, test_df, expected_classes) -> tuple[bool, str, str]:
    """Strict gate: fails if train < 20, val < 5, or test < 10 for any expected class."""
    train_counts = train_df["class_label"].value_counts().to_dict() if len(train_df) > 0 else {}
    val_counts = val_df["class_label"].value_counts().to_dict() if len(val_df) > 0 else {}
    test_counts = test_df["class_label"].value_counts().to_dict() if len(test_df) > 0 else {}
    
    sufficient = True
    reasons = []
    
    for cls in expected_classes:
        tr_c = train_counts.get(cls, 0)
        va_c = val_counts.get(cls, 0)
        te_c = test_counts.get(cls, 0)
        
        if tr_c < 20:
            sufficient = False
            reasons.append(f"Class '{cls}' train count {tr_c} < 20")
        if va_c < 5:
            sufficient = False
            reasons.append(f"Class '{cls}' val count {va_c} < 5")
        if te_c < 10:
            sufficient = False
            reasons.append(f"Class '{cls}' test count {te_c} < 10")
            
    evidence_level = "sufficient" if sufficient else "restricted"
    reason_str = "; ".join(reasons) if reasons else "All expected classes satisfy strict target sizes."
    return sufficient, evidence_level, reason_str

def main():
    parser = argparse.ArgumentParser(description="TransitLens Unified Model Trainer")
    parser.add_argument("--feature-dir", type=str, default=None, help="Directory containing feature CSVs")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output model artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--model", type=str, choices=["random_forest"], default="random_forest", help="Model type to train")
    parser.add_argument("--strict-real-data", action="store_true", help="Enable strict real data training pipeline")
    parser.add_argument("--label-mode", type=str, choices=["binary", "four_class"], default="four_class", help="Label classification mode")
    parser.add_argument("--bypass-sufficiency", action="store_true", help="Bypass sufficiency gate failure for small development runs")
    args = parser.parse_args()
    
    # Define expected classes based on mode
    if args.label_mode == "binary":
        expected_classes = ["exoplanet_transit", "stellar_variability_or_other"]
    else:
        expected_classes = CLASSES
        
    # Paths Setup
    if args.feature_dir:
        feature_dir = Path(args.feature_dir)
    else:
        feature_dir = Path(__file__).resolve().parent / "data" / "ml_features"
        
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent / "models" / "staging"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Strict validation of bypass write destination
    is_staging_or_pilot = "staging" in str(output_dir).lower() or "pilot" in str(output_dir).lower()
    if args.bypass_sufficiency and not is_staging_or_pilot:
        logger.error("Bypass sufficiency is enabled, but output directory is not in staging or pilot! Bypassed training must NOT write directly to production.")
        sys.exit(1)
        
    train_csv = feature_dir / "train_features.csv"
    val_csv = feature_dir / "val_features.csv"
    test_csv = feature_dir / "test_features.csv"
    
    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        logger.error(f"Feature CSVs not found in {feature_dir}! Run prepare_ml.py first.")
        sys.exit(1)
        
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    train_df = train_df[train_df["processing_status"] == "success"]
    val_df = val_df[val_df["processing_status"] == "success"]
    test_df = test_df[test_df["processing_status"] == "success"]
    
    if len(train_df) == 0:
        logger.error("No successful targets in training split! Cannot train model.")
        sys.exit(1)
        
    # Run strict sufficiency check
    suff_ok, evidence_level, suff_reasons = check_sufficiency(train_df, val_df, test_df, expected_classes)
    
    production_eligible = False
    if not suff_ok:
        if not args.bypass_sufficiency:
            logger.error(f"Sufficiency gate failed: {suff_reasons}. Model cannot be trained.")
            sys.exit(1)
        else:
            logger.warning(f"WARNING: Sufficiency gate failed ({suff_reasons}), but bypass is enabled. Saving as STAGING ONLY.")
            production_eligible = False
    else:
        # If sufficiency check is fully satisfied, mark as production eligible
        production_eligible = True
        logger.info("All strict sufficiency checks satisfied. Model is production eligible.")
        
    # Check no target leakage across splits
    train_tics = set(train_df["tic_id"].dropna().unique())
    val_tics = set(val_df["tic_id"].dropna().unique())
    test_tics = set(test_df["tic_id"].dropna().unique())
    assert train_tics.isdisjoint(val_tics), "Leakage detected: train & val share TICs!"
    assert train_tics.isdisjoint(test_tics), "Leakage detected: train & test share TICs!"
    assert val_tics.isdisjoint(test_tics), "Leakage detected: val & test share TICs!"
    
    # Extract features and targets
    X_train = train_df[list(FEATURE_NAMES)].values
    y_train = train_df["class_label"].values
    
    X_val = val_df[list(FEATURE_NAMES)].values if len(val_df) > 0 else np.empty((0, len(FEATURE_NAMES)))
    y_val = val_df["class_label"].values if len(val_df) > 0 else np.array([])
    
    X_test = test_df[list(FEATURE_NAMES)].values if len(test_df) > 0 else np.empty((0, len(FEATURE_NAMES)))
    y_test = test_df["class_label"].values if len(test_df) > 0 else np.array([])
    
    # Standard scale: fit ONLY on training split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else np.empty((0, len(FEATURE_NAMES)))
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.empty((0, len(FEATURE_NAMES)))
    
    # Label mapping setup
    class_to_idx = {cls: idx for idx, cls in enumerate(expected_classes)}
    y_train_idx = np.array([class_to_idx[y] for y in y_train])
    y_val_idx = np.array([class_to_idx[y] for y in y_val]) if len(y_val) > 0 else np.array([])
    y_test_idx = np.array([class_to_idx[y] for y in y_test]) if len(y_test) > 0 else np.array([])
    
    # Train class-weighted RandomForest baseline
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    # Platt probability calibration
    is_calibrated = False
    can_calibrate = len(X_val_scaled) > 0 and len(np.unique(y_val)) == len(expected_classes)
    if can_calibrate:
        logger.info("Fitting Platt probability calibrator on validation split using FrozenEstimator...")
        from sklearn.frozen import FrozenEstimator
        frozen_model = FrozenEstimator(rf)
        custom_cv = [(np.arange(len(X_val_scaled)), np.arange(len(X_val_scaled)))]
        calibrator = CalibratedClassifierCV(estimator=frozen_model, method="sigmoid", cv=custom_cv)
        calibrator.fit(X_val_scaled, y_val)
        model_to_save = calibrator
        is_calibrated = True
    else:
        logger.warning("Bypassing calibration due to insufficient validation samples or classes. Using raw classifier.")
        model_to_save = rf
        is_calibrated = False
        
    # Save final model wrapper
    wrapper = TransitLensClassifier(model=model_to_save, scaler=scaler, classes=expected_classes, is_xgboost=False)
    
    # Save files
    with open(output_dir / "final_classifier.pkl", "wb") as f:
        pickle.dump(wrapper, f)
    with open(output_dir / "final_feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(output_dir / "final_label_mapping.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)
    with open(output_dir / "final_feature_order.json", "w") as f:
        json.dump(list(FEATURE_NAMES), f, indent=2)
        
    # Evaluate on Test Split
    test_probs = np.zeros((len(y_test), len(expected_classes)))
    test_preds = []
    
    if len(X_test) > 0:
        for i, row_features in enumerate(X_test):
            row_probs_dict = wrapper.predict_proba(row_features.reshape(1, -1))
            row_probs = [row_probs_dict[c] for c in expected_classes]
            test_probs[i] = row_probs
            test_preds.append(wrapper.predict(row_features.reshape(1, -1)))
            
        test_report = classification_report(y_test, test_preds, labels=expected_classes, output_dict=True, zero_division=0)
        y_test_onehot = np.zeros((len(y_test), len(expected_classes)))
        for i, val in enumerate(y_test_idx):
            y_test_onehot[i, val] = 1.0
        ece = compute_ece(y_test_idx, test_probs)
        brier = compute_brier_score(y_test_onehot, test_probs)
        cm = confusion_matrix(y_test, test_preds, labels=expected_classes).tolist()
    else:
        test_report = {}
        ece = 0.0
        brier = 0.0
        cm = []
        
    # Dynamic sectors tracking
    all_sectors = []
    for df_split in [train_df, val_df, test_df]:
        if "sector" in df_split.columns:
            all_sectors.extend(df_split["sector"].dropna().unique().astype(int).tolist())
    sectors_list = sorted(list(set(all_sectors)))
    
    # Metadata
    archive_toi = REPO_ROOT / "archive" / "TOI_2026.06.25_21.21.19.csv"
    archive_tce = REPO_ROOT / "archive" / "tess s0078-s0078_tcestats.csv"
    
    label_policy_path = REPO_ROOT / "config" / "toi_label_policy.yaml"
    policy_version = "unknown"
    if label_policy_path.exists():
        with open(label_policy_path, "r") as f:
            policy = yaml.safe_load(f)
            policy_version = policy.get("version", "unknown")
            
    train_tic_ids = [int(tid.split("-")[-1]) for tid in train_df["target_id"].dropna().tolist() if "-" in tid]
    
    metadata = {
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": args.seed,
        "model_type": "CalibratedRandomForest" if is_calibrated else "RandomForest",
        "label_mode": args.label_mode,
        "trained_classes": expected_classes,
        "production_eligible": production_eligible,
        "is_calibrated": is_calibrated,
        "sufficiency_notes": suff_reasons,
        "archive_hashes": {
            "TOI_catalog": get_file_hash(archive_toi),
            "TCE_stats": get_file_hash(archive_tce)
        },
        "label_policy_version": policy_version,
        "preprocessing_config": PREPROCESS_CONFIG,
        "aperture_photometry_version": "connected_threshold_v1.0",
        "cutout_size_pixels": 15,
        "train_tic_ids": train_tic_ids,
        "sectors_trained": sectors_list,
        "package_versions": get_package_versions(),
        "dataset_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "metrics": {
            "test_accuracy": test_report.get("accuracy", 0.0),
            "test_macro_f1": test_report.get("macro avg", {}).get("f1-score", 0.0),
            "ece": ece,
            "brier_score": brier
        }
    }
    
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / "confusion_matrix.json", "w") as f:
        json.dump(cm, f, indent=2)
    with open(output_dir / "per_sector_metrics.json", "w") as f:
        json.dump({}, f, indent=2)
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(test_report, f, indent=2)
        
    eval_summary = {
        "production_eligible": production_eligible,
        "label_mode": args.label_mode,
        "is_calibrated": is_calibrated,
        "test_metrics": metadata["metrics"],
        "dataset_sizes": metadata["dataset_sizes"]
    }
    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)
        
    write_model_card(output_dir, metadata, test_report, ece, brier, evidence_level, suff_reasons)
    
    logger.info("--- Model Retraining Complete ---")
    logger.info(f"Production Eligible: {production_eligible} | Test Accuracy: {metadata['metrics']['test_accuracy']:.4f} | Test Macro F1: {metadata['metrics']['test_macro_f1']:.4f}")

def write_model_card(output_dir, meta, test_report, ece, brier, evidence_level, suff_reasons):
    card_path = output_dir / "model_card.md"
    
    card_lines = [
        "# Model Card: Retrained TransitLens Classifier",
        "",
        "## Model Details",
        f"- **Model Type**: {meta['model_type']}",
        f"- **Training Date**: {meta['training_date']}",
        f"- **Evidence Level**: `{evidence_level}`",
        f"- **Sufficiency Notes**: {suff_reasons}",
        f"- **Production Eligible**: `{meta['production_eligible']}`",
        "",
        "## Parameters & Training Metadata",
        f"- **Label Mode**: `{meta['label_mode']}`",
        f"- **Trained Classes**: {meta['trained_classes']}",
        f"- **Random Seed**: {meta['random_seed']}",
        f"- **Label Policy Version**: `{meta['label_policy_version']}`",
        f"- **Aperture Photometry Version**: `{meta['aperture_photometry_version']}`",
        f"- **Cutout Size**: {meta['cutout_size_pixels']}x{meta['cutout_size_pixels']} pixels",
        f"- **Archive Hashes (TOI)**: `{meta['archive_hashes']['TOI_catalog']}`",
        f"- **Archive Hashes (TCE)**: `{meta['archive_hashes']['TCE_stats']}`",
        "",
        "## Performance Metrics (Test Split)",
        f"- **Accuracy**: {meta['metrics']['test_accuracy']:.4%}",
        f"- **Macro F1 Score**: {meta['metrics']['test_macro_f1']:.4f}",
        f"- **Expected Calibration Error (ECE)**: {ece:.4f}",
        f"- **Brier Score**: {brier:.4f}",
        "",
        "### Per-Class Performance Summary",
        "",
        "| Class Label | Precision | Recall | F1-Score | Support |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for cls in meta["trained_classes"]:
        cls_metrics = test_report.get(cls, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        card_lines.append(
            f"| {cls} | {cls_metrics['precision']:.4f} | {cls_metrics['recall']:.4f} | "
            f"{cls_metrics['f1-score']:.4f} | {int(cls_metrics['support'])} |"
        )
        
    card_lines.extend([
        "",
        "## Dataset Partition Sizes",
        f"- **Train Samples**: {meta['dataset_sizes']['train']}",
        f"- **Validation Samples**: {meta['dataset_sizes']['val']}",
        f"- **Test Samples**: {meta['dataset_sizes']['test']}",
        "",
        "## Scientific Warnings & Limitations",
        "- **Offline Vetted Retraining**: This model has been explicitly retrained offline with vetted archives.",
        "- **No Continual Self-Training**: Model predictions are never automatically injected back into the training catalog.",
        f"- **Trained Sectors**: {meta['sectors_trained']}"
    ])
    
    card_path.write_text("\n".join(card_lines), encoding="utf-8")
    logger.info("Saved model card to %s", card_path)

if __name__ == "__main__":
    main()
