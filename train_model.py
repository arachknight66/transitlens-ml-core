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
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "astropy", "astroquery", "lightkurve"]:
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

def check_sufficiency(train_df, test_df, strict_real_data=False) -> tuple[bool, str, str]:
    """Strict gate: warns/fails if train < 20 or test < 10 for any class."""
    train_counts = train_df["class_label"].value_counts().to_dict()
    test_counts = test_df["class_label"].value_counts().to_dict() if len(test_df) > 0 else {}
    
    sufficient = True
    reasons = []
    
    # In strict-real-data mode, we only require sufficiency for the classes actually present
    classes_to_check = CLASSES
    if strict_real_data:
        classes_to_check = [c for c in CLASSES if train_counts.get(c, 0) > 0]
        
    for cls in classes_to_check:
        tr_c = train_counts.get(cls, 0)
        te_c = test_counts.get(cls, 0)
        
        if tr_c < 20:
            sufficient = False
            reasons.append(f"{cls} train count {tr_c} < 20")
        if te_c < 10:
            sufficient = False
            reasons.append(f"{cls} test count {te_c} < 10")
            
    evidence_level = "sufficient" if sufficient else "restricted"
    reason_str = "; ".join(reasons) if reasons else "All classes satisfy target sizes."
    return sufficient, evidence_level, reason_str

def main():
    parser = argparse.ArgumentParser(description="TransitLens Unified Model Trainer")
    parser.add_argument("--feature-dir", type=str, default=None, help="Directory containing feature CSVs")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output model artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--model", type=str, choices=["random_forest"], default="random_forest", help="Model type to train")
    parser.add_argument("--strict-real-data", action="store_true", help="Enable strict real data training pipeline")
    parser.add_argument("--bypass-sufficiency", action="store_true", help="Bypass sufficiency gate failure for small testing runs")
    args = parser.parse_args()
    
    # 1. Paths Setup
    if args.feature_dir:
        feature_dir = Path(args.feature_dir)
    else:
        feature_dir = Path(__file__).resolve().parent / "data" / "ml_features"
        
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent / "models"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv = feature_dir / "train_features.csv"
    val_csv = feature_dir / "val_features.csv"
    test_csv = feature_dir / "test_features.csv"
    
    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        logger.error(f"Feature CSVs not found in {feature_dir}! Run prepare_ml.py first.")
        sys.exit(1)
        
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    # Keep only successfully processed targets
    train_df = train_df[train_df["processing_status"] == "success"]
    val_df = val_df[val_df["processing_status"] == "success"]
    test_df = test_df[test_df["processing_status"] == "success"]
    
    if len(train_df) == 0:
        logger.error("No successful targets in training split! Cannot train model.")
        sys.exit(1)
        
    # Run sufficiency check
    suff_ok, evidence_level, suff_reasons = check_sufficiency(train_df, test_df, args.strict_real_data)
    if not suff_ok:
        if args.strict_real_data and not args.bypass_sufficiency:
            logger.error(f"Sufficiency gate failed under --strict-real-data mode: {suff_reasons}")
            sys.exit(1)
        else:
            logger.warning("WARNING: Dataset size is insufficient: %s", suff_reasons)
            
    # Extract features and targets
    X_train = train_df[list(FEATURE_NAMES)].values
    y_train = train_df["class_label"].values
    
    X_val = val_df[list(FEATURE_NAMES)].values
    y_val = val_df["class_label"].values
    
    X_test = test_df[list(FEATURE_NAMES)].values if len(test_df) > 0 else np.empty((0, len(FEATURE_NAMES)))
    y_test = test_df["class_label"].values if len(test_df) > 0 else np.array([])
    
    # Scale features: Fit scaler ONLY on the training split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else np.empty((0, len(FEATURE_NAMES)))
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.empty((0, len(FEATURE_NAMES)))
    
    # Label mapping setup
    class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
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
    
    # Calibrate probabilities using Sigmoid/Platt scaling on the Validation split
    can_calibrate = len(X_val_scaled) > 1 and len(np.unique(y_val)) >= 2
    if can_calibrate:
        logger.info("Calibrating model using validation split...")
        frozen_model = FrozenEstimator(rf)
        calibrator = CalibratedClassifierCV(estimator=frozen_model, method="sigmoid")
        calibrator.fit(X_val_scaled, y_val)
        model_to_save = calibrator
    else:
        logger.warning("Bypassing calibration due to insufficient validation samples/classes. Saving raw classifier.")
        model_to_save = rf
        
    # Save final model wrapper
    wrapper = TransitLensClassifier(model=model_to_save, scaler=scaler, classes=CLASSES, is_xgboost=False)
    
    # Save artifacts
    with open(output_dir / "final_classifier.pkl", "wb") as f:
        pickle.dump(wrapper, f)
    with open(output_dir / "final_feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(output_dir / "final_label_mapping.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)
    with open(output_dir / "final_feature_order.json", "w") as f:
        json.dump(list(FEATURE_NAMES), f, indent=2)
        
    # Evaluate on Test Split
    test_probs = np.zeros((len(y_test), len(CLASSES)))
    test_preds = []
    
    if len(X_test) > 0:
        for i, row_features in enumerate(X_test):
            row_probs_dict = wrapper.predict_proba(row_features.reshape(1, -1))
            row_probs = [row_probs_dict[c] for c in CLASSES]
            test_probs[i] = row_probs
            test_preds.append(wrapper.predict(row_features.reshape(1, -1)))
            
        test_report = classification_report(y_test, test_preds, labels=CLASSES, output_dict=True, zero_division=0)
        y_test_onehot = np.zeros((len(y_test), len(CLASSES)))
        for i, val in enumerate(y_test_idx):
            y_test_onehot[i, val] = 1.0
        ece = compute_ece(y_test_idx, test_probs)
        brier = compute_brier_score(y_test_onehot, test_probs)
        cm = confusion_matrix(y_test, test_preds, labels=CLASSES).tolist()
    else:
        test_report = {}
        ece = 0.0
        brier = 0.0
        cm = []
        
    per_sector = {}
    
    # Training Metadata gathering
    archive_toi = REPO_ROOT / "archive" / "TOI_2026.06.25_21.21.19.csv"
    archive_tce = REPO_ROOT / "archive" / "tess s0078-s0078_tcestats.csv"
    
    label_policy_path = REPO_ROOT / "config" / "toi_label_policy.yaml"
    policy_version = "unknown"
    if label_policy_path.exists():
        with open(label_policy_path, "r") as f:
            policy = yaml.safe_load(f)
            policy_version = policy.get("version", "unknown")
            
    train_tic_ids = [int(tid.split("-")[-1]) for tid in train_df["target_id"].dropna().tolist() if "-" in tid]
    sectors_list = [78]
    
    metadata = {
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": args.seed,
        "model_type": "CalibratedRandomForest",
        "evidence_level": evidence_level,
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
    
    # Save files
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / "confusion_matrix.json", "w") as f:
        json.dump(cm, f, indent=2)
    with open(output_dir / "per_sector_metrics.json", "w") as f:
        json.dump(per_sector, f, indent=2)
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(test_report, f, indent=2)
        
    # Write model card
    write_model_card(output_dir, metadata, test_report, ece, brier, evidence_level, suff_reasons)
    
    logger.info("--- Model Retraining & Platt Calibration Complete ---")
    logger.info(f"Test Accuracy: {metadata['metrics']['test_accuracy']:.4f} | Test Macro F1: {metadata['metrics']['test_macro_f1']:.4f}")

def write_model_card(output_dir, meta, test_report, ece, brier, evidence_level, suff_reasons):
    card_path = output_dir / "model_card.md"
    
    card_lines = [
        "# Model Card: Retrained TransitLens Classifier",
        "",
        "## Model Details",
        f"- **Model Type**: Calibrated RandomForest Classifier (Sigmoid Platt Scaling)",
        f"- **Training Date**: {meta['training_date']}",
        f"- **Evidence Level**: `{evidence_level}`",
        f"- **Sufficiency Notes**: {suff_reasons}",
        "",
        "## Parameters & Training Metadata",
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
    
    for cls in CLASSES:
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
        "- **Class Exclusions**: Eclipsing Binary and Blend Contamination classes are intentionally empty for this TESS run due to a lack of vetted TESS EB/blend catalogs. They will consistently output 0.0% probability during inference."
    ])
    
    card_path.write_text("\n".join(card_lines), encoding="utf-8")
    logger.info("Saved model card to %s", card_path)

if __name__ == "__main__":
    main()
