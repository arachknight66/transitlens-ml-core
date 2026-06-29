"""Fail-closed inference from the hash-verified active registry entry."""
from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from .checksums import sha256_file
from .contracts import PHYSICAL_CLASSES, validate_probability_vector
from .registry import validate_candidate
from .uncertainty import ReviewPolicy, route_review

class MLUnavailable(RuntimeError): pass

class ActiveModel:
    def __init__(self, registry: Path):
        pointer_path = registry / "active_model.json"
        if not pointer_path.exists(): raise MLUnavailable("no active promoted model")
        self.pointer = json.loads(pointer_path.read_text())
        self.path = registry / "models" / self.pointer["model_id"]
        self.record = validate_candidate(self.path)
        self.model = joblib.load(self.path / "model.joblib")
        self.preprocessor = joblib.load(self.path / "preprocessor.joblib")
        self.calibrator = joblib.load(self.path / "calibration.joblib")

    def predict(self, frame: pd.DataFrame, *, missing_critical=None, rule_contradiction=None) -> list[dict]:
        X = self.preprocessor.transform(frame)
        raw = self.model.predict_proba(X)
        probabilities = self.calibrator.transform(np.log(np.clip(raw, 1e-15, 1)))
        ood = self.preprocessor.out_of_range(frame)
        reasons = route_review(probabilities, ReviewPolicy(), ood=ood,
                               missing_critical=missing_critical, rule_contradiction=rule_contradiction)
        results = []
        for probs, why in zip(probabilities, reasons):
            vector = {name: float(value) for name, value in zip(PHYSICAL_CLASSES, probs)}
            validate_probability_vector(vector)
            physical = PHYSICAL_CLASSES[int(np.argmax(probs))]
            results.append({"predicted_astrophysical_class": physical,
                            "routing_outcome": "review_required" if why else physical,
                            "calibrated_class_probabilities": vector, "review_required": bool(why),
                            "review_reasons": why, "ood": "out_of_distribution" in why,
                            "model_id": self.pointer["model_id"], "calibration_status": "calibrated",
                            "confirmation_claim": False})
        return results
