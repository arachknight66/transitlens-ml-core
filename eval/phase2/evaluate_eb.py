# evaluate_eb.py
# --------------
# Evaluates eclipsing binary (EB) diagnostics performance on splits.

from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def evaluate_eb_diagnostics(
    df_features: pd.DataFrame,
) -> dict:
    """
    Computes precision, recall, and specificity for EB diagnostics against ground truth labels.
    """
    # EB Ground Truth label = "eclipsing_binary"
    # Diagnostics flag = "eb_risk_level" in ("medium", "high", "critical")
    if df_features.empty or "label" not in df_features.columns:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
        
    y_true = df_features["label"] == "eclipsing_binary"
    
    # We predict EB if recommended route is eclipsing_binary or risk is high
    y_pred = df_features["eb_risk_score"] >= 0.50 if "eb_risk_score" in df_features.columns else np.zeros(len(df_features), dtype=bool)
    
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    # Simple bootstrap confidence intervals grouped by TIC ID
    unique_tics = df_features["tic_id"].unique()
    n_resamples = 200
    rng = np.random.default_rng(42)
    
    boot_recalls = []
    for _ in range(n_resamples):
        sample_tics = rng.choice(unique_tics, size=len(unique_tics), replace=True)
        # Resample rows matching selected TICs
        sample_rows = pd.concat([df_features[df_features["tic_id"] == t] for t in sample_tics])
        if len(sample_rows) > 0:
            yt = sample_rows["label"] == "eclipsing_binary"
            yp = sample_rows["eb_risk_score"] >= 0.50 if "eb_risk_score" in sample_rows.columns else np.zeros(len(sample_rows), dtype=bool)
            boot_tp = np.sum(yt & yp)
            boot_fn = np.sum(yt & ~yp)
            if (boot_tp + boot_fn) > 0:
                boot_recalls.append(boot_tp / (boot_tp + boot_fn))
                
    rec_ci_lower = float(np.percentile(boot_recalls, 2.5)) if boot_recalls else recall
    rec_ci_upper = float(np.percentile(boot_recalls, 97.5)) if boot_recalls else recall
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "recall_ci_lower": round(rec_ci_lower, 4),
        "recall_ci_upper": round(rec_ci_upper, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "support_positives": int(y_true.sum()),
        "support_negatives": int((~y_true).sum()),
    }
