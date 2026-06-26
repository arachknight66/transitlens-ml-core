"""
core/classifier.py
------------------
Rule-based classification of transit candidates into one of three classes:

    "exoplanet_like"        — signal consistent with a transiting exoplanet
    "eclipsing_binary_like" — signal consistent with an eclipsing binary star
    "noise_or_other"        — no significant signal detected

Primary path: rule-based decision tree (always active).
All thresholds are loaded from models/rule_config.yaml — none are hardcoded.

Secondary path (optional): scikit-learn Random Forest or XGBoost classifier.
Active only when trained model files exist in models/ and
ml_classifier.enabled = true in rule_config.yaml.

When both paths produce a prediction:
    - Agreement  → use that class; use ML confidence.
    - Disagreement → use rule-based class; note disagreement in explanation.

Used by: pipeline.py
"""

from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from core.exceptions import ClassificationError

logger = logging.getLogger(__name__)

# Canonical allowed class labels
CLASSES = ("exoplanet_like", "eclipsing_binary_like", "noise_or_other")

# Default path to rule config — resolved relative to this file's location
_DEFAULT_RULE_CONFIG_PATH = Path(__file__).parent.parent / "models" / "rule_config.yaml"

# Module-level cache: config loaded once at first call, not per-classification
_rule_config_cache: dict | None = None
_rule_config_path_cache: str | None = None


# ---------------------------------------------------------------------------
# Public result container
# ---------------------------------------------------------------------------

class ClassificationResult:
    """
    Output of the classification stage.

    Attributes
    ----------
    predicted_class : str
        One of "exoplanet_like", "eclipsing_binary_like", "noise_or_other".
    rule_path : list[str]
        Ordered list of rule conditions evaluated to reach the decision.
        Each entry is a human-readable condition string, e.g.
        "depth=0.0127 <= depth_threshold_eb=0.050 → continue".
    ml_class : str or None
        The ML model's prediction (if active), or None.
    ml_agreement : bool
        True if rule-based and ML predictions agree (or ML not active).
    thresholds : dict
        The threshold values used in this classification (for explanation).
    """

    def __init__(
        self,
        predicted_class: str,
        rule_path: list[str],
        ml_class: Optional[str] = None,
        ml_agreement: bool = True,
        thresholds: dict | None = None,
    ):
        if predicted_class not in CLASSES:
            raise ClassificationError(
                f"predicted_class must be one of {CLASSES}, got '{predicted_class}'"
            )
        self.predicted_class = predicted_class
        self.rule_path = rule_path
        self.ml_class = ml_class
        self.ml_agreement = ml_agreement
        self.thresholds = thresholds or {}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_rule_config(rule_config_path: str | None = None) -> dict:
    """
    Load and cache the rule_config.yaml file.

    The config is loaded once per process (module-level cache) unless
    the path changes. Thread-safety is not a concern for the hackathon.
    """
    global _rule_config_cache, _rule_config_path_cache

    path = str(rule_config_path or _DEFAULT_RULE_CONFIG_PATH)

    if _rule_config_cache is not None and _rule_config_path_cache == path:
        return _rule_config_cache

    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise ClassificationError(
            f"rule_config.yaml not found at '{path}'. "
            "Ensure models/rule_config.yaml exists."
        )
    except yaml.YAMLError as exc:
        raise ClassificationError(
            f"Failed to parse rule_config.yaml: {exc}"
        )

    # Validate required keys exist
    required_sections = ["detection", "classification", "confidence", "ml_classifier"]
    for section in required_sections:
        if section not in cfg:
            raise ClassificationError(
                f"rule_config.yaml missing required section '{section}'"
            )

    _rule_config_cache = cfg
    _rule_config_path_cache = path
    logger.debug("classifier: rule_config loaded from '%s'", path)
    return cfg


def reload_rule_config(rule_config_path: str | None = None) -> dict:
    """Force reload the rule config (useful in tests that modify thresholds)."""
    global _rule_config_cache, _rule_config_path_cache
    _rule_config_cache = None
    _rule_config_path_cache = None
    return _load_rule_config(rule_config_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify(
    features: dict[str, float],
    config: dict | None = None,
    rule_config_path: str | None = None,
) -> ClassificationResult:
    """
    Classify a feature vector into one of the three transit signal classes.

    Parameters
    ----------
    features : dict[str, float]
        The 11-feature dict from feature_extractor.extract().
    config : dict or None
        Optional runtime overrides. Recognised key:
            "classification" — override classification thresholds dict.
    rule_config_path : str or None
        Path to rule_config.yaml. Uses default models/ location if None.

    Returns
    -------
    ClassificationResult
        Contains predicted_class, rule_path (decision trace), and optional
        ML model prediction.

    Raises
    ------
    ClassificationError
        If rule_config.yaml is missing, malformed, or produces an invalid class.
    """
    rule_cfg = _load_rule_config(rule_config_path)

    # Merge any runtime overrides into the classification thresholds
    clf_thresholds = dict(rule_cfg["classification"])
    if config and "classification" in config:
        clf_thresholds.update(config["classification"])

    det_thresholds = dict(rule_cfg["detection"])
    if config and "detection" in config:
        det_thresholds.update(config["detection"])

    # Load ML classifier config
    ml_cfg = rule_cfg.get("ml_classifier", {})
    ml_enabled = ml_cfg.get("enabled", True)
    dev_fallback = ml_cfg.get("dev_fallback", False)
    
    # Check runtime overrides for ml_classifier
    if config and "ml_classifier" in config:
        ml_override = config["ml_classifier"]
        ml_enabled = ml_override.get("enabled", ml_enabled)
        dev_fallback = ml_override.get("dev_fallback", dev_fallback)

    # Run rule-based classification for diagnostics/explanation
    rule_result = _apply_rules(features, clf_thresholds, det_thresholds)

    # Stage 1 Detection gate bypass: if it is noise because of Stage 1 fail, bypass ML
    if rule_result.predicted_class == "noise_or_other" and any("Stage 1 FAIL" in s for s in rule_result.rule_path):
        logger.info("classifier: candidate failed Stage 1 detection gate — forcing noise_or_other")
        return rule_result

    ml_class = None
    ml_agreement = True

    if ml_enabled:
        try:
            ml_class = _run_ml_classifier(features, rule_cfg, rule_config_path)
            ml_agreement = (ml_class == rule_result.predicted_class)
        except Exception as exc:
            if dev_fallback:
                logger.warning(
                    "ML classifier failed: %s — falling back to rule-based because dev_fallback=true",
                    exc
                )
                ml_class = None
                ml_agreement = True
            else:
                raise ClassificationError(
                    f"ML classifier execution failed (and dev_fallback=false): {exc}. "
                    "Please ensure the ML models are trained by running: python train_model.py"
                ) from exc
    else:
        if dev_fallback:
            logger.info("ML classifier disabled — using rule-based diagnostic because dev_fallback=true")
            ml_class = None
            ml_agreement = True
        else:
            raise ClassificationError(
                "ML classifier is disabled in configuration, but dev_fallback is false. "
                "The trained ML model is required for production inference."
            )

    predicted_class = ml_class if ml_class is not None else rule_result.predicted_class

    logger.info(
        "classifier: predicted_class='%s' (ml_class=%s, rule_class=%s, agreement=%s)",
        predicted_class, ml_class, rule_result.predicted_class, ml_agreement,
    )

    return ClassificationResult(
        predicted_class=predicted_class,
        rule_path=rule_result.rule_path,
        ml_class=ml_class,
        ml_agreement=ml_agreement,
        thresholds=rule_result.thresholds,
    )


# ---------------------------------------------------------------------------
# Rule-based decision tree
# ---------------------------------------------------------------------------

def _apply_rules(
    features: dict[str, float],
    clf_thresholds: dict,
    det_thresholds: dict,
) -> ClassificationResult:
    """
    Apply the rule-based decision tree and return a ClassificationResult.

    Decision tree (evaluated in order — first matching rule wins):

        Stage 1 (Detection gate):
            SNR < snr_threshold OR BLS power < bls_power_threshold
            → noise_or_other

        Stage 2 (Primary depth discriminator):
            depth > depth_threshold_eb
            → eclipsing_binary_like

        Stage 3 (Secondary discriminators, within planet-like depth range):
            odd_even_depth_delta > odd_even_threshold  → eclipsing_binary_like
            v_shape_score > v_shape_threshold          → eclipsing_binary_like
            depth_to_noise_ratio < depth_snr_threshold → noise_or_other
            all pass                                   → exoplanet_like
    """
    rule_path: list[str] = []

    # Extract feature values with safe defaults
    snr         = float(features.get("snr", 0.0))
    bls_power   = float(features.get("bls_power", 0.0))
    depth       = float(features.get("depth", 0.0))
    odd_even    = float(features.get("odd_even_depth_delta", 0.0))
    v_shape     = float(features.get("v_shape_score", 0.0))
    dtnr        = float(features.get("depth_to_noise_ratio", 0.0))

    # Thresholds from config
    power_thresh  = float(det_thresholds.get("bls_power_threshold", 0.15))
    snr_thresh    = float(det_thresholds.get("snr_threshold", 5.0))
    depth_eb      = float(clf_thresholds.get("depth_threshold_eb", 0.050))
    odd_even_thr  = float(clf_thresholds.get("odd_even_threshold", 0.020))
    v_shape_thr   = float(clf_thresholds.get("v_shape_threshold", 0.40))
    dtnr_thr      = float(clf_thresholds.get("depth_snr_threshold", 6.0))

    thresholds_used = {
        "bls_power_threshold": power_thresh,
        "snr_threshold": snr_thresh,
        "depth_threshold_eb": depth_eb,
        "odd_even_threshold": odd_even_thr,
        "v_shape_threshold": v_shape_thr,
        "depth_snr_threshold": dtnr_thr,
    }

    # ── Stage 1: Detection gate ───────────────────────────────────────────
    power_fail = bls_power < power_thresh
    snr_fail   = snr < snr_thresh

    if power_fail or snr_fail:
        reason_parts = []
        if power_fail:
            reason_parts.append(
                f"bls_power={bls_power:.4f} < threshold={power_thresh:.3f}"
            )
        if snr_fail:
            reason_parts.append(
                f"snr={snr:.2f} < threshold={snr_thresh:.1f}"
            )
        rule_path.append(
            f"Stage 1 FAIL [{'; '.join(reason_parts)}] → noise_or_other"
        )
        return ClassificationResult("noise_or_other", rule_path, thresholds=thresholds_used)

    rule_path.append(
        f"Stage 1 PASS [bls_power={bls_power:.4f} >= {power_thresh:.3f}; "
        f"snr={snr:.2f} >= {snr_thresh:.1f}]"
    )

    # ── Stage 2: Primary depth discriminator ─────────────────────────────
    if depth > depth_eb:
        rule_path.append(
            f"Stage 2 MATCH [depth={depth:.4f} > depth_threshold_eb={depth_eb:.3f}] "
            f"→ eclipsing_binary_like"
        )
        return ClassificationResult(
            "eclipsing_binary_like", rule_path, thresholds=thresholds_used
        )

    rule_path.append(
        f"Stage 2 PASS [depth={depth:.4f} <= depth_threshold_eb={depth_eb:.3f}]"
    )

    # ── Stage 3: Secondary discriminators ────────────────────────────────

    # 3a: Odd/even depth asymmetry
    if odd_even > odd_even_thr:
        rule_path.append(
            f"Stage 3a MATCH [odd_even_delta={odd_even:.4f} > threshold={odd_even_thr:.3f}] "
            f"→ eclipsing_binary_like"
        )
        return ClassificationResult(
            "eclipsing_binary_like", rule_path, thresholds=thresholds_used
        )
    rule_path.append(
        f"Stage 3a PASS [odd_even_delta={odd_even:.4f} <= {odd_even_thr:.3f}]"
    )

    # 3b: V-shape profile
    if v_shape > v_shape_thr:
        rule_path.append(
            f"Stage 3b MATCH [v_shape_score={v_shape:.4f} > threshold={v_shape_thr:.3f}] "
            f"→ eclipsing_binary_like"
        )
        return ClassificationResult(
            "eclipsing_binary_like", rule_path, thresholds=thresholds_used
        )
    rule_path.append(
        f"Stage 3b PASS [v_shape_score={v_shape:.4f} <= {v_shape_thr:.3f}]"
    )

    # 3c: Depth-to-noise ratio (sub-threshold signal after passing detection gate)
    if dtnr < dtnr_thr:
        rule_path.append(
            f"Stage 3c MATCH [depth_to_noise_ratio={dtnr:.2f} < threshold={dtnr_thr:.1f}] "
            f"→ noise_or_other"
        )
        return ClassificationResult("noise_or_other", rule_path, thresholds=thresholds_used)
    rule_path.append(
        f"Stage 3c PASS [depth_to_noise_ratio={dtnr:.2f} >= {dtnr_thr:.1f}]"
    )

    # All checks passed → exoplanet-like
    rule_path.append("All stages passed → exoplanet_like")
    return ClassificationResult("exoplanet_like", rule_path, thresholds=thresholds_used)


# ---------------------------------------------------------------------------
# Optional ML classifier
# ---------------------------------------------------------------------------

def _run_ml_classifier(
    features: dict[str, float],
    rule_cfg: dict,
    rule_config_path: str | None,
) -> str:
    """
    Run the ML classifier (RF or XGBoost) on the feature vector.

    Returns the predicted class string. Raises an exception if the model
    files are absent or prediction fails.
    """
    from core.feature_extractor import FEATURE_NAMES

    ml_cfg = rule_cfg.get("ml_classifier", {})
    model_type = ml_cfg.get("model_type", "rf")

    # Locate model files relative to rule_config
    if rule_config_path:
        models_dir = Path(rule_config_path).parent
    else:
        models_dir = _DEFAULT_RULE_CONFIG_PATH.parent

    # Validate feature order against FEATURE_NAMES
    feature_order_file = models_dir / "feature_order.json"
    if not feature_order_file.exists():
        raise FileNotFoundError(
            f"feature_order.json not found in {models_dir}. "
            "Ensure the model has been trained."
        )
    with open(feature_order_file, "r") as f:
        saved_features = json.load(f)
    if list(saved_features) != list(FEATURE_NAMES):
        raise ClassificationError(
            f"Feature schema mismatch. Models trained with: {saved_features}, "
            f"but code expects: {list(FEATURE_NAMES)}"
        )

    model_file    = models_dir / f"{'rf' if model_type == 'rf' else 'xgb'}_model.pkl"
    scaler_file   = models_dir / "feature_scaler.pkl"

    if not model_file.exists():
        raise FileNotFoundError(f"ML model not found: {model_file}")
    if not scaler_file.exists():
        raise FileNotFoundError(f"Feature scaler not found: {scaler_file}")

    import pickle
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)

    # Build feature array in canonical order
    feature_vec = np.array([features.get(k, 0.0) for k in FEATURE_NAMES]).reshape(1, -1)
    scaled = scaler.transform(feature_vec)

    probas = model.predict_proba(scaled)[0]
    class_idx = int(np.argmax(probas))

    # Map index back to class label (assumes model was trained with CLASSES order)
    model_classes = list(model.classes_)
    if class_idx < len(model_classes):
        return str(model_classes[class_idx])

    raise ClassificationError(f"ML model returned unexpected class index {class_idx}")