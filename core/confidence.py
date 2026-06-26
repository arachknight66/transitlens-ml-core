"""
core/confidence.py
------------------
Calibrated confidence score for a transit classification.

The confidence score is NOT simply the ML model's probability output.
It is a composite score built from physically-meaningful feature components,
each weighted by how strongly it supports the predicted class.

Score range: [0.0, 1.0]
    0.0 → no evidence at all for the predicted class
    1.0 → every component perfectly satisfies its threshold

Partial scoring: components that do not fully satisfy their threshold
receive a fractional score proportional to how close they are.

This makes confidence interpretable: "88% confident" means roughly 88% of
the weighted evidence supports the predicted class.

The component definitions, weights, and full-score thresholds are all read
from models/rule_config.yaml (confidence section) — nothing is hardcoded.

Used by: pipeline.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from core.classifier import _load_rule_config
from core.exceptions import ClassificationError

logger = logging.getLogger(__name__)

# Default path — same as classifier.py
_DEFAULT_RULE_CONFIG_PATH = Path(__file__).parent.parent / "models" / "rule_config.yaml"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score(
    features: dict[str, float],
    predicted_class: str,
    config: dict | None = None,
    rule_config_path: str | None = None,
) -> float:
    """
    Compute a calibrated confidence score for the predicted class.

    Parameters
    ----------
    features : dict[str, float]
        The 11-feature dict from feature_extractor.extract().
    predicted_class : str
        One of "exoplanet_like", "eclipsing_binary_like", "noise_or_other".
    config : dict or None
        Optional runtime overrides for the confidence section of rule_config.
    rule_config_path : str or None
        Path to rule_config.yaml. Uses default models/ location if None.

    Returns
    -------
    float
        Confidence score in [0.0, 1.0]. Always finite.
    """
    rule_cfg = _load_rule_config(rule_config_path)
    confidence_cfg = dict(rule_cfg.get("confidence", {}))
    if config and "confidence" in config:
        confidence_cfg.update(config["confidence"])

    aliases = {
        "exoplanet_like": "exoplanet_transit",
        "eclipsing_binary_like": "eclipsing_binary",
        "noise_or_other": "stellar_variability_or_other",
    }
    predicted_class = aliases.get(predicted_class, predicted_class)

    if predicted_class not in confidence_cfg:
        logger.warning(
            "confidence: no config for class '%s' — returning 0.5 as default",
            predicted_class,
        )
        return 0.5

    components = confidence_cfg[predicted_class].get("components", [])
    if not components:
        logger.warning("confidence: no components defined for '%s'", predicted_class)
        return 0.5

    total_weight = 0.0
    weighted_score = 0.0

    for comp in components:
        name      = comp.get("name", "unknown")
        weight    = float(comp.get("weight", 0.0))
        feature   = comp.get("feature", "")
        direction = comp.get("direction", "above")
        threshold = float(comp.get("full_score_threshold", 1.0))

        if weight <= 0:
            continue

        feature_val = float(features.get(feature, 0.0))
        component_score = _compute_component_score(
            feature_val, direction, threshold, name
        )

        weighted_score += weight * component_score
        total_weight   += weight

        logger.debug(
            "confidence[%s] component '%s': feature=%s=%.4f direction=%s "
            "threshold=%.4f → score=%.3f (weight=%.2f)",
            predicted_class, name, feature, feature_val,
            direction, threshold, component_score, weight,
        )

    if total_weight <= 0:
        return 0.5

    raw = weighted_score / total_weight
    confidence = float(np.clip(raw, 0.0, 1.0))

    logger.info(
        "confidence: class='%s' score=%.4f (weighted_score=%.4f, total_weight=%.4f)",
        predicted_class, confidence, weighted_score, total_weight,
    )

    return confidence


# ---------------------------------------------------------------------------
# Component scoring
# ---------------------------------------------------------------------------

def _compute_component_score(
    feature_val: float,
    direction: str,
    threshold: float,
    name: str = "",
) -> float:
    """
    Compute a partial score [0, 1] for a single confidence component.

    direction="above": full score when feature_val >= threshold.
                       Partial score proportional to feature_val / threshold
                       when feature_val < threshold.

    direction="below": full score when feature_val <= threshold.
                       Partial score proportional to threshold / feature_val
                       when feature_val > threshold (inverted).

    Parameters
    ----------
    feature_val : float
        Observed feature value.
    direction : str
        "above" or "below".
    threshold : float
        Value at which component receives full score (1.0).
    name : str
        Component name for logging only.

    Returns
    -------
    float
        Component score in [0.0, 1.0].
    """
    if not np.isfinite(feature_val):
        logger.debug("confidence component '%s': non-finite feature_val → score 0", name)
        return 0.0

    if direction == "above":
        if threshold <= 0:
            return 1.0 if feature_val > 0 else 0.0
        if feature_val >= threshold:
            return 1.0
        # Partial: linear from 0 at feature_val=0 up to 1 at feature_val=threshold
        return float(np.clip(feature_val / threshold, 0.0, 1.0))

    elif direction == "below":
        if threshold <= 0:
            return 0.0
        if feature_val <= threshold:
            return 1.0
        # Partial: linear from 1 at feature_val=0 down to 0 at feature_val=2*threshold
        # Gives smooth falloff without a hard cutoff
        decay = float(np.clip(1.0 - (feature_val - threshold) / threshold, 0.0, 1.0))
        return decay

    else:
        logger.warning("confidence: unknown direction '%s' for component '%s'", direction, name)
        return 0.5


# ---------------------------------------------------------------------------
# Explanation helper
# ---------------------------------------------------------------------------

def score_with_breakdown(
    features: dict[str, float],
    predicted_class: str,
    config: dict | None = None,
    rule_config_path: str | None = None,
) -> tuple[float, list[dict]]:
    """
    Compute confidence score and return a detailed breakdown of each component.

    Returns
    -------
    (confidence, breakdown)
        confidence : float in [0, 1]
        breakdown  : list of dicts, each with keys:
            name, feature, feature_val, direction, threshold,
            component_score, weight, weighted_contribution
    """
    rule_cfg = _load_rule_config(rule_config_path)
    confidence_cfg = dict(rule_cfg.get("confidence", {}))
    if config and "confidence" in config:
        confidence_cfg.update(config["confidence"])

    aliases = {
        "exoplanet_like": "exoplanet_transit",
        "eclipsing_binary_like": "eclipsing_binary",
        "noise_or_other": "stellar_variability_or_other",
    }
    predicted_class = aliases.get(predicted_class, predicted_class)

    if predicted_class not in confidence_cfg:
        return 0.5, []

    components = confidence_cfg[predicted_class].get("components", [])
    breakdown = []
    total_weight = 0.0
    weighted_score = 0.0

    for comp in components:
        name      = comp.get("name", "unknown")
        weight    = float(comp.get("weight", 0.0))
        feature   = comp.get("feature", "")
        direction = comp.get("direction", "above")
        threshold = float(comp.get("full_score_threshold", 1.0))

        if weight <= 0:
            continue

        feature_val = float(features.get(feature, 0.0))
        cs = _compute_component_score(feature_val, direction, threshold, name)

        breakdown.append({
            "name": name,
            "feature": feature,
            "feature_val": feature_val,
            "direction": direction,
            "threshold": threshold,
            "component_score": cs,
            "weight": weight,
            "weighted_contribution": weight * cs,
        })

        weighted_score += weight * cs
        total_weight   += weight

    confidence = float(np.clip(weighted_score / total_weight, 0.0, 1.0)) \
        if total_weight > 0 else 0.5

    return confidence, breakdown