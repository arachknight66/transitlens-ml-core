"""
train_model.py
--------------
unified training script for TransitLens Phase 5.

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Output directories
MODELS_DIR = REPO_ROOT / "transitlens-ml-core" / "models"
EVAL_RESULTS_DIR = REPO_ROOT / "transitlens-ml-core" / "eval" / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Helper function to compute ECE
def compute_ece(y_true_indices, y_prob, n_bins=10):
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
    return float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))

def check_sufficiency(train_df, test_df) -> tuple[bool, str]:
    """Strict gate: warns if train < 20 or test < 10 for any class."""
    train_counts = train_df["class_label"].value_counts().to_dict()
    test_counts = test_df["class_label"].value_counts().to_dict()
    
    sufficient = True
    reasons = []
    
    for cls in CLASSES:
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
    parser = argparse.ArgumentParser(description="TransitLens Phase 5 Classifier Trainer")
    args = parser.parse_args()
    
    feature_dir = Path(__file__).resolve().parent / "data" / "ml_features"
    train_csv = feature_dir / "train_features.csv"
    val_csv = feature_dir / "val_features.csv"
    test_csv = feature_dir / "test_features.csv"
    
    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        logger.error("Feature CSVs not found! Run prepare_ml.py first.")
        sys.exit(1)
        
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    # Run sufficiency check
    suff_ok, evidence_level, suff_reasons = check_sufficiency(train_df, test_df)
    if not suff_ok:
        logger.warning("WARNING: Dataset size is insufficient: %s", suff_reasons)
        
    # Extract features and targets
    X_train = train_df[list(FEATURE_NAMES)].values
    y_train = train_df["class_label"].values
    
    X_val = val_df[list(FEATURE_NAMES)].values
    y_val = val_df["class_label"].values
    
    X_test = test_df[list(FEATURE_NAMES)].values
    y_test = test_df["class_label"].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
    y_train_idx = np.array([class_to_idx[y] for y in y_train])
    y_val_idx = np.array([class_to_idx[y] for y in y_val])
    y_test_idx = np.array([class_to_idx[y] for y in y_test])
    
    # ── 1. Majority Classifier Baseline ─────────────────────────────────────
    majority_class = Counter(y_train).most_common(1)[0][0]
    y_pred_maj = [majority_class] * len(y_test)
    maj_report = classification_report(y_test, y_pred_maj, labels=CLASSES, output_dict=True, zero_division=0)
    
    # ── 2. Rule-based Baseline ──────────────────────────────────────────────
    # Uses predictions generated in test_features.csv by prepare_ml.py
    y_pred_rule = test_df["predicted_class_rule"].values
    rule_report = classification_report(y_test, y_pred_rule, labels=CLASSES, output_dict=True, zero_division=0)
    
    # ── 3. Random Forest Model ──────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    val_preds_rf = rf.predict(X_val_scaled)
    rf_val_report = classification_report(y_val, val_preds_rf, labels=CLASSES, output_dict=True, zero_division=0)
    rf_val_f1 = rf_val_report["macro avg"]["f1-score"]
    
    # ── 4. XGBoost Model (if xgboost available) ─────────────────────────────
    xgb_available = False
    xgb_val_f1 = -1.0
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss"
        )
        xgb.fit(X_train_scaled, y_train_idx)
        val_preds_xgb = xgb.predict(X_val_scaled)
        # map indexes back to class labels
        val_preds_xgb_labels = [CLASSES[idx] for idx in val_preds_xgb]
        xgb_val_report = classification_report(y_val, val_preds_xgb_labels, labels=CLASSES, output_dict=True, zero_division=0)
        xgb_val_f1 = xgb_val_report["macro avg"]["f1-score"]
        xgb_available = True
    except ImportError:
        logger.info("XGBoost not available, training only Random Forest.")
        
    # Choose best model based on validation macro F1
    if xgb_available and xgb_val_f1 > rf_val_f1:
        logger.info("Choosing XGBoost (Val Macro F1 = %.4f) over RF (Val Macro F1 = %.4f)", xgb_val_f1, rf_val_f1)
        best_model = xgb
        best_is_xgb = True
        best_model_name = "XGBoost"
    else:
        logger.info("Choosing Random Forest (Val Macro F1 = %.4f)", rf_val_f1)
        best_model = rf
        best_is_xgb = False
        best_model_name = "RandomForest"
        
    # ── 5. Probability Calibration ─────────────────────────────────────────
    # Since cv='prefit', it calibrates best_model on validation set
    frozen_model = FrozenEstimator(best_model)
    if best_is_xgb:
        calibrator = CalibratedClassifierCV(estimator=frozen_model, method="sigmoid")
        calibrator.fit(X_val_scaled, y_val_idx)
    else:
        calibrator = CalibratedClassifierCV(estimator=frozen_model, method="sigmoid")
        calibrator.fit(X_val_scaled, y_val)
        
    # Save final model wrapper
    wrapper = TransitLensClassifier(model=calibrator, scaler=scaler, classes=CLASSES, is_xgboost=best_is_xgb)
    
    # Save artifacts
    with open(MODELS_DIR / "final_classifier.pkl", "wb") as f:
        pickle.dump(wrapper, f)
    with open(MODELS_DIR / "final_feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    with open(MODELS_DIR / "final_label_mapping.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)
    with open(MODELS_DIR / "final_feature_order.json", "w") as f:
        json.dump(list(FEATURE_NAMES), f, indent=2)
        
    # ── 6. Final Evaluation on Test Split ──────────────────────────────────
    test_probs = np.zeros((len(y_test), len(CLASSES)))
    test_preds = []
    
    for i, row in enumerate(X_test):
        row_probs_dict = wrapper.predict_proba(row.reshape(1, -1))
        # align probabilities with canonical CLASSES order
        row_probs = [row_probs_dict[c] for c in CLASSES]
        test_probs[i] = row_probs
        test_preds.append(wrapper.predict(row.reshape(1, -1)))
        
    # Compute test classification metrics
    test_report = classification_report(y_test, test_preds, labels=CLASSES, output_dict=True, zero_division=0)
    test_acc = test_report["accuracy"]
    test_macro_f1 = test_report["macro avg"]["f1-score"]
    
    # Compute Calibration Metrics
    y_test_onehot = np.zeros((len(y_test), len(CLASSES)))
    for i, val in enumerate(y_test_idx):
        y_test_onehot[i, val] = 1.0
        
    ece = compute_ece(y_test_idx, test_probs)
    brier = compute_brier_score(y_test_onehot, test_probs)
    
    # Save test predictions details to csv
    predictions_df = pd.DataFrame({
        "target_id": test_df["target_id"],
        "true_label": y_test,
        "predicted_label": test_preds,
        "confidence": np.max(test_probs, axis=1),
        "probability_exoplanet_transit": test_probs[:, 0],
        "probability_eclipsing_binary": test_probs[:, 1],
        "probability_blend_contamination": test_probs[:, 2],
        "probability_stellar_variability_or_other": test_probs[:, 3],
        "correct": (np.array(test_preds) == y_test).astype(int),
        "candidate_detected": test_df["candidate_detected"]
    })
    predictions_df.to_csv(EVAL_RESULTS_DIR / "predictions_test.csv", index=False)
    logger.info("Saved test predictions to eval/results/predictions_test.csv")
    
    # Generate training metadata JSON
    training_meta = {
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evidence_level": evidence_level,
        "sufficiency_reasons": suff_reasons,
        "model_type": best_model_name,
        "features_used": list(FEATURE_NAMES),
        "dataset_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "metrics": {
            "test_accuracy": test_acc,
            "test_macro_f1": test_macro_f1,
            "ece": ece,
            "brier_score": brier
        }
    }
    with open(MODELS_DIR / "training_metadata.json", "w") as f:
        json.dump(training_meta, f, indent=2)
        
    # Write model card
    write_model_card(training_meta, test_report, ece, brier, evidence_level, suff_reasons)
    
    logger.info("--- Model Training & Calibration Complete ---")
    logger.info("Test Accuracy: %.4f | Test Macro F1: %.4f | ECE: %.4f", test_acc, test_macro_f1, ece)

def write_model_card(meta, test_report, ece, brier, evidence_level, suff_reasons):
    card_path = MODELS_DIR / "model_card.md"
    
    card_lines = [
        "# Model Card: TransitLens ML Classifier",
        "",
        "## Model Details",
        f"- **Model Type**: Sigmoid-Calibrated {meta['model_type']} Classifier",
        f"- **Training Date**: {meta['training_date']}",
        f"- **Evidence Level**: `{evidence_level}`",
        f"- **Sufficiency Notes**: {suff_reasons}",
        "",
        "## Intended Use",
        "The model is designed to classify light curve transiting events into one of four categories: `exoplanet_transit`, `eclipsing_binary`, `blend_contamination`, and `stellar_variability_or_other`.",
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
        "## Training & Split Distribution",
        f"- **Train Samples**: {meta['dataset_sizes']['train']}",
        f"- **Validation Samples**: {meta['dataset_sizes']['val']}",
        f"- **Test Samples**: {meta['dataset_sizes']['test']}",
        "",
        "## Warnings & Limitations"
    ])
    
    if meta['dataset_sizes']['test'] < 100:
        card_lines.append("- **⚠ Limited Evidence Warning**: Classifier evidence is limited; metrics should not be treated as production-grade.")
        
    if evidence_level == "restricted":
        card_lines.append("- **⚠ Class Support Instability Warning**: Per-class metrics for some classes are unstable due to insufficient samples (<10 test samples per class).")
        
    card_lines.append("")
    card_path.write_text("\n".join(card_lines), encoding="utf-8")
    logger.info("Model card saved to %s", card_path)

if __name__ == "__main__":
    main()
