# evaluate_thresholds.py
# --------------------
# Audits threshold sensitivity curves for classification routing.

from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def evaluate_threshold_sensitivity(
    df_features: pd.DataFrame,
) -> dict:
    """
    Computes precision/recall curves over a grid of eb/blend risk thresholds.
    """
    if df_features.empty:
        return {}
        
    thresholds = np.linspace(0.10, 0.90, 9)
    eb_results = []
    blend_results = []
    
    y_true_eb = df_features["label"] == "eclipsing_binary" if "label" in df_features.columns else np.zeros(len(df_features), dtype=bool)
    y_true_bl = df_features["label"] == "blend_contamination" if "label" in df_features.columns else np.zeros(len(df_features), dtype=bool)
    
    for t in thresholds:
        # EB re-evaluate
        if "eb_risk_score" in df_features.columns:
            yp = df_features["eb_risk_score"] >= t
            tp = np.sum(y_true_eb & yp)
            fp = np.sum(~y_true_eb & yp)
            fn = np.sum(y_true_eb & ~yp)
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            eb_results.append({"threshold": float(t), "precision": prec, "recall": rec})
            
        # Blend re-evaluate
        if "blend_risk_score" in df_features.columns:
            yp = df_features["blend_risk_score"] >= t
            tp = np.sum(y_true_bl & yp)
            fp = np.sum(~y_true_bl & yp)
            fn = np.sum(y_true_bl & ~yp)
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            blend_results.append({"threshold": float(t), "precision": prec, "recall": rec})
            
    return {
        "eb_threshold_sweep": eb_results,
        "blend_threshold_sweep": blend_results,
    }
