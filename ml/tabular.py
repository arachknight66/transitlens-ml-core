"""Interpretable target-grouped tabular baselines."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from .contracts import ContractError, PHYSICAL_CLASSES

def require_all_classes(y) -> None:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=len(PHYSICAL_CLASSES))
    missing = [PHYSICAL_CLASSES[i] for i, n in enumerate(counts) if n == 0]
    if missing:
        raise ContractError(f"training requires real targets from every class; absent: {missing}")

def model_families(seed: int = 42):
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=250, l2_regularization=1.0, random_state=seed),
    }

def grouped_oof(model, X, y, groups, folds: int = 5, seed: int = 42) -> np.ndarray:
    y, groups = np.asarray(y, dtype=int), np.asarray(groups)
    require_all_classes(y)
    output = np.zeros((len(y), len(PHYSICAL_CLASSES)), dtype=float)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, holdout_idx in splitter.split(X, y, groups):
        if len(np.unique(y[train_idx])) != len(PHYSICAL_CLASSES):
            raise ContractError("a grouped CV fold lacks a physical class; reduce folds or add real targets")
        model.fit(X[train_idx], y[train_idx])
        fold_probs = model.predict_proba(X[holdout_idx])
        output[holdout_idx[:, None], np.asarray(model.classes_)[None, :]] = fold_probs
    return output

def compare_models(X, y, groups, folds: int = 5, seed: int = 42):
    rows, predictions = [], {}
    for name, model in model_families(seed).items():
        oof = grouped_oof(model, X, y, groups, folds, seed)
        predictions[name] = oof
        rows.append({"model": name, "grouped_oof_macro_f1": f1_score(y, oof.argmax(1), average="macro")})
    return sorted(rows, key=lambda row: row["grouped_oof_macro_f1"], reverse=True), predictions
