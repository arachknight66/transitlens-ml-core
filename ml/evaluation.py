from __future__ import annotations
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from .calibration import expected_calibration_error, multiclass_brier
from .contracts import PHYSICAL_CLASSES, PromotionGates

def evaluate(y_true, probabilities, review_reasons=None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = probabilities.argmax(axis=1)
    report = classification_report(y_true, y_pred, labels=range(4), target_names=PHYSICAL_CLASSES,
                                   output_dict=True, zero_division=0)
    review_reasons = review_reasons or [[] for _ in y_true]
    return {
        "n_targets": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "ece": expected_calibration_error(y_true, probabilities),
        "brier_score": multiclass_brier(y_true, probabilities),
        "review_rate": float(np.mean([bool(x) for x in review_reasons])),
        "per_class": {name: report[name] for name in PHYSICAL_CLASSES},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(4)).tolist(),
    }

def promotion_gate(metrics: dict, integrity: dict, gates: PromotionGates = PromotionGates()) -> dict:
    pc = metrics.get("per_class", {})
    checks = {
        "macro_f1": metrics.get("macro_f1", -1) >= gates.macro_f1,
        "planet_precision": pc.get(PHYSICAL_CLASSES[0], {}).get("precision", -1) >= gates.planet_precision,
        "planet_recall": pc.get(PHYSICAL_CLASSES[0], {}).get("recall", -1) >= gates.planet_recall,
        "eb_recall": pc.get(PHYSICAL_CLASSES[1], {}).get("recall", -1) >= gates.eb_recall,
        "blend_recall": pc.get(PHYSICAL_CLASSES[2], {}).get("recall", -1) >= gates.blend_recall,
        "blend_precision": pc.get(PHYSICAL_CLASSES[2], {}).get("precision", -1) >= gates.blend_precision,
        "ece": metrics.get("ece", 1) <= gates.ece,
        "brier_reported": "brier_score" in metrics,
        "all_class_support": all(pc.get(c, {}).get("support", 0) >= gates.minimum_class_support for c in PHYSICAL_CLASSES),
        "input_integrity": integrity.get("status") == "PASS",
        "artifact_integrity": bool(integrity.get("artifact_integrity", False)),
        "model_card_agrees": bool(integrity.get("model_card_agrees", False)),
        "smoke_test": bool(integrity.get("smoke_test", False)),
        "rollback_target": bool(integrity.get("rollback_target", False)),
    }
    return {"production_eligible": all(checks.values()), "checks": checks}
