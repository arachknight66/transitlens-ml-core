"""Leakage-safe late fusion using out-of-fold base predictions."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from .contracts import PHYSICAL_CLASSES

def fit_stacker(tabular_oof, timeseries_oof, y, seed=42):
    a, b = np.asarray(tabular_oof), np.asarray(timeseries_oof)
    if a.shape != b.shape or a.shape[1] != len(PHYSICAL_CLASSES):
        raise ValueError("base OOF predictions must have matching four-class shapes")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    return model.fit(np.hstack([a, b]), y)

def predict_stacker(model, tabular_probs, timeseries_probs):
    return model.predict_proba(np.hstack([tabular_probs, timeseries_probs]))
