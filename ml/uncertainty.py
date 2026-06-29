from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ReviewPolicy:
    minimum_probability: float = 0.65
    maximum_normalized_entropy: float = 0.75
    maximum_disagreement: float = 0.12
    review_on_ood: bool = True
    review_on_missing_critical: bool = True
    review_on_rule_contradiction: bool = True

def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1)
    return -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])

def route_review(probabilities: np.ndarray, policy: ReviewPolicy, *, disagreement=None,
                 ood=None, missing_critical=None, rule_contradiction=None) -> list[list[str]]:
    probabilities = np.asarray(probabilities)
    n = len(probabilities)
    entropy = normalized_entropy(probabilities)
    disagreement = np.zeros(n) if disagreement is None else np.asarray(disagreement)
    ood = np.zeros(n, bool) if ood is None else np.asarray(ood, bool)
    missing_critical = np.zeros(n, bool) if missing_critical is None else np.asarray(missing_critical, bool)
    rule_contradiction = np.zeros(n, bool) if rule_contradiction is None else np.asarray(rule_contradiction, bool)
    result = []
    for i in range(n):
        reasons = []
        if probabilities[i].max() < policy.minimum_probability: reasons.append("low_calibrated_probability")
        if entropy[i] > policy.maximum_normalized_entropy: reasons.append("high_predictive_entropy")
        if disagreement[i] > policy.maximum_disagreement: reasons.append("high_ensemble_disagreement")
        if policy.review_on_ood and ood[i]: reasons.append("out_of_distribution")
        if policy.review_on_missing_critical and missing_critical[i]: reasons.append("missing_critical_diagnostics")
        if policy.review_on_rule_contradiction and rule_contradiction[i]: reasons.append("ml_rule_contradiction")
        result.append(reasons)
    return result
