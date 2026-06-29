from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar

class TemperatureScaler:
    """Multiclass temperature scaling fitted only on validation logits."""
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=float)
        y = np.asarray(y, dtype=int)
        if logits.ndim != 2 or len(logits) != len(y):
            raise ValueError("invalid calibration arrays")
        def loss(log_t):
            probs = _softmax(logits / np.exp(log_t))
            return -np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1)))
        result = minimize_scalar(loss, bounds=(-4, 4), method="bounded")
        self.temperature = float(np.exp(result.x))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return _softmax(np.asarray(logits, dtype=float) / self.temperature)

def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

def expected_calibration_error(y: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    y, probs = np.asarray(y), np.asarray(probs)
    confidence, predicted = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs((predicted[mask] == y[mask]).mean() - confidence[mask].mean())
    return float(ece)

def multiclass_brier(y: np.ndarray, probs: np.ndarray) -> float:
    targets = np.eye(probs.shape[1])[np.asarray(y, dtype=int)]
    return float(np.mean(np.sum((np.asarray(probs) - targets) ** 2, axis=1)))
