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
CLASSES = ("exoplanet_transit", "eclipsing_binary", "blend_contamination", "stellar_variability_or_other")

class TransitLensClassifier:
    """Wrapper class for the final calibrated ML classifier."""
    def __init__(self, model, scaler, classes, is_xgboost=False):
        self.model = model
        self.scaler = scaler
        self.classes = list(classes)
        self.is_xgboost = is_xgboost
        
    def predict(self, feature_array: np.ndarray, calibrated: bool = True) -> str:
        scaled = self.scaler.transform(feature_array)
        estimator = self.model
        if not calibrated:
            if hasattr(self.model, "estimator"):
                base_est = self.model.estimator
                if hasattr(base_est, "estimator"):
                    estimator = base_est.estimator
                else:
                    estimator = base_est
        
        if self.is_xgboost:
            pred_idx = int(estimator.predict(scaled)[0])
            try:
                return self.classes[pred_idx]
            except (ValueError, IndexError):
                return str(pred_idx)
        else:
            pred = estimator.predict(scaled)[0]
            return str(pred)
            
    def predict_proba(self, feature_array: np.ndarray, calibrated: bool = True) -> dict[str, float]:
        scaled = self.scaler.transform(feature_array)
        estimator = self.model
        if not calibrated:
            if hasattr(self.model, "estimator"):
                base_est = self.model.estimator
                if hasattr(base_est, "estimator"):
                    estimator = base_est.estimator
                else:
                    estimator = base_est
                    
        probs = estimator.predict_proba(scaled)[0]
        if self.is_xgboost:
            return {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        else:
            model_classes = list(estimator.classes_)
            prob_dict = {}
            for cls_val, prob in zip(model_classes, probs):
                if isinstance(cls_val, (int, np.integer)):
                    cls_name = self.classes[cls_val]
                else:
                    cls_name = str(cls_val)
                prob_dict[cls_name] = float(prob)
                
            for cls in self.classes:
                if cls not in prob_dict:
                    prob_dict[cls] = 0.0
            return prob_dict

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
        One of "exoplanet_transit", "eclipsing_binary", "blend_contamination", "stellar_variability_or_other".
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
    class_probabilities : dict[str, float]
        Calibrated probabilities for each class.
    """

    def __init__(
        self,
        predicted_class: str,
        rule_path: list[str],
        ml_class: Optional[str] = None,
        ml_agreement: bool = True,
        thresholds: dict | None = None,
        class_probabilities: dict[str, float] | None = None,
    ):
        aliases = {
            "exoplanet_like": "exoplanet_transit",
            "eclipsing_binary_like": "eclipsing_binary",
            "noise_or_other": "stellar_variability_or_other"
        }
        predicted_class = aliases.get(predicted_class, predicted_class)
        if ml_class is not None:
            ml_class = aliases.get(ml_class, ml_class)
            
        if predicted_class not in CLASSES:
            raise ClassificationError(
                f"predicted_class must be one of {CLASSES}, got '{predicted_class}'"
            )
        self.predicted_class = predicted_class
        self.rule_path = rule_path
        self.ml_class = ml_class
        self.ml_agreement = ml_agreement
        self.thresholds = thresholds or {}
        self.class_probabilities = class_probabilities or {}


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
    Classify a feature vector into one of the four transit signal classes.
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
    ml_calibrated = ml_cfg.get("calibrated", True)
    use_rule_fallback_on_disagreement = ml_cfg.get("use_rule_fallback_on_disagreement", False)
    
    # Check runtime overrides for ml_classifier
    if config and "ml_classifier" in config:
        ml_override = config["ml_classifier"]
        ml_enabled = ml_override.get("enabled", ml_enabled)
        dev_fallback = ml_override.get("dev_fallback", dev_fallback)
        ml_calibrated = ml_override.get("calibrated", ml_calibrated)
        use_rule_fallback_on_disagreement = ml_override.get("use_rule_fallback_on_disagreement", use_rule_fallback_on_disagreement)

    # Run rule-based classification for diagnostics/explanation
    rule_result = _apply_rules(features, clf_thresholds, det_thresholds)

    # Stage 1 Detection gate bypass: if it is noise because of Stage 1 fail, bypass ML
    if rule_result.predicted_class == "stellar_variability_or_other" and any("Stage 1 FAIL" in s for s in rule_result.rule_path):
        logger.info("classifier: candidate failed Stage 1 detection gate — forcing stellar_variability_or_other")
        # Build default probabilities for noise
        class_probabilities = {cls: 0.0 for cls in CLASSES}
        class_probabilities["stellar_variability_or_other"] = 1.0
        return ClassificationResult(
            predicted_class="stellar_variability_or_other",
            rule_path=rule_result.rule_path,
            thresholds=rule_result.thresholds,
            class_probabilities=class_probabilities
        )

    ml_class = None
    ml_agreement = True
    class_probabilities = None

    if ml_enabled:
        try:
            ml_class, class_probabilities = _run_ml_classifier(features, rule_cfg, rule_config_path, ml_calibrated)
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
        if dev_fallback or ml_cfg.get("production_state") == "rule_only_restricted":
            logger.info("ML classifier disabled — using rule-based diagnostic because dev_fallback=true or production_state is rule_only_restricted")
            ml_class = None
            ml_agreement = True
        else:
            raise ClassificationError(
                "ML classifier is disabled in configuration, but dev_fallback is false. "
                "The trained ML model is required for production inference."
            )

    predicted_class = ml_class if ml_class is not None else rule_result.predicted_class
    
    if ml_enabled and ml_class is not None and not ml_agreement:
        logger.info(
            "classifier: ML class '%s' and rule class '%s' disagree.",
            ml_class, rule_result.predicted_class
        )
        if use_rule_fallback_on_disagreement:
            logger.info("classifier: falling back to rule-based class '%s' due to disagreement fallback config.", rule_result.predicted_class)
            predicted_class = rule_result.predicted_class



    if class_probabilities is None:
        # Build simulated probabilities from rule confidence
        from core.confidence import score
        conf = score(features, predicted_class, rule_config_path=rule_config_path)
        class_probabilities = {}
        for cls in CLASSES:
            if cls == predicted_class:
                class_probabilities[cls] = float(conf)
            else:
                class_probabilities[cls] = float((1.0 - conf) / 3.0)

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
        class_probabilities=class_probabilities,
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
            → stellar_variability_or_other

        Stage 2 (Primary depth discriminator):
            depth > depth_threshold_eb
            → eclipsing_binary

        Stage 3 (Secondary discriminators, within planet-like depth range):
            odd_even_depth_delta > odd_even_threshold   → eclipsing_binary
            v_shape_score > v_shape_threshold           → eclipsing_binary
            depth_to_noise_ratio < depth_snr_threshold  → stellar_variability_or_other
            crowding_metric < crowding_threshold        → blend_contamination
            centroid_shift > centroid_shift_threshold   → blend_contamination
            all pass                                    → exoplanet_transit
    """
    rule_path: list[str] = []

    # Extract feature values with safe defaults
    snr         = float(features.get("snr", 0.0))
    bls_power   = float(features.get("bls_power", 0.0))
    depth       = float(features.get("depth", 0.0))
    odd_even    = float(features.get("odd_even_depth_delta", 0.0))
    v_shape     = float(features.get("v_shape_score", 0.0))
    dtnr        = float(features.get("depth_to_noise_ratio", 0.0))
    crowding    = float(features.get("crowding_metric", 1.0))
    centroid_sh = float(features.get("centroid_shift", 0.0))

    # Thresholds from config
    power_thresh  = float(det_thresholds.get("bls_power_threshold", 0.15))
    snr_thresh    = float(det_thresholds.get("snr_threshold", 5.0))
    depth_eb      = float(clf_thresholds.get("depth_threshold_eb", 0.050))
    odd_even_thr  = float(clf_thresholds.get("odd_even_threshold", 0.020))
    v_shape_thr   = float(clf_thresholds.get("v_shape_threshold", 0.40))
    dtnr_thr      = float(clf_thresholds.get("depth_snr_threshold", 6.0))
    crowding_thr  = float(clf_thresholds.get("crowding_threshold", 0.80))
    shift_thr     = float(clf_thresholds.get("centroid_shift_threshold", 0.015))

    thresholds_used = {
        "bls_power_threshold": power_thresh,
        "snr_threshold": snr_thresh,
        "depth_threshold_eb": depth_eb,
        "odd_even_threshold": odd_even_thr,
        "v_shape_threshold": v_shape_thr,
        "depth_snr_threshold": dtnr_thr,
        "crowding_threshold": crowding_thr,
        "centroid_shift_threshold": shift_thr,
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
            f"Stage 1 FAIL [{'; '.join(reason_parts)}] → stellar_variability_or_other"
        )
        return ClassificationResult("stellar_variability_or_other", rule_path, thresholds=thresholds_used)

    rule_path.append(
        f"Stage 1 PASS [bls_power={bls_power:.4f} >= {power_thresh:.3f}; "
        f"snr={snr:.2f} >= {snr_thresh:.1f}]"
    )

    # ── Stage 2: Primary depth discriminator ─────────────────────────────
    if depth > depth_eb:
        rule_path.append(
            f"Stage 2 MATCH [depth={depth:.4f} > depth_threshold_eb={depth_eb:.3f}] "
            f"→ eclipsing_binary"
        )
        return ClassificationResult(
            "eclipsing_binary", rule_path, thresholds=thresholds_used
        )

    rule_path.append(
        f"Stage 2 PASS [depth={depth:.4f} <= depth_threshold_eb={depth_eb:.3f}]"
    )

    # ── Stage 3: Secondary discriminators ────────────────────────────────

    # 3a: Odd/even depth asymmetry
    if odd_even > odd_even_thr:
        rule_path.append(
            f"Stage 3a MATCH [odd_even_delta={odd_even:.4f} > threshold={odd_even_thr:.3f}] "
            f"→ eclipsing_binary"
        )
        return ClassificationResult(
            "eclipsing_binary", rule_path, thresholds=thresholds_used
        )
    rule_path.append(
        f"Stage 3a PASS [odd_even_delta={odd_even:.4f} <= {odd_even_thr:.3f}]"
    )

    # 3b: V-shape profile
    if v_shape > v_shape_thr:
        rule_path.append(
            f"Stage 3b MATCH [v_shape_score={v_shape:.4f} > threshold={v_shape_thr:.3f}] "
            f"→ eclipsing_binary"
        )
        return ClassificationResult(
            "eclipsing_binary", rule_path, thresholds=thresholds_used
        )
    rule_path.append(
        f"Stage 3b PASS [v_shape_score={v_shape:.4f} <= {v_shape_thr:.3f}]"
    )

    # 3c: Depth-to-noise ratio (sub-threshold signal after passing detection gate)
    if dtnr < dtnr_thr:
        rule_path.append(
            f"Stage 3c MATCH [depth_to_noise_ratio={dtnr:.2f} < threshold={dtnr_thr:.1f}] "
            f"→ stellar_variability_or_other"
        )
        return ClassificationResult("stellar_variability_or_other", rule_path, thresholds=thresholds_used)
    rule_path.append(
        f"Stage 3c PASS [depth_to_noise_ratio={dtnr:.2f} >= {dtnr_thr:.1f}]"
    )

    # 3d: Blend and Crowding Diagnostics
    if crowding < crowding_thr or centroid_sh > shift_thr:
        reason_parts = []
        if crowding < crowding_thr:
            reason_parts.append(f"crowding_metric={crowding:.2f} < threshold={crowding_thr:.2f}")
        if centroid_sh > shift_thr:
            reason_parts.append(f"centroid_shift={centroid_sh:.4f} > threshold={shift_thr:.3f}")
        rule_path.append(
            f"Stage 3d MATCH [{'; '.join(reason_parts)}] → blend_contamination"
        )
        return ClassificationResult("blend_contamination", rule_path, thresholds=thresholds_used)
    rule_path.append(
        f"Stage 3d PASS [crowding_metric={crowding:.2f} >= {crowding_thr:.2f}; "
        f"centroid_shift={centroid_sh:.4f} <= {shift_thr:.3f}]"
    )

    # All checks passed → exoplanet_transit
    rule_path.append("All stages passed → exoplanet_transit")
    return ClassificationResult("exoplanet_transit", rule_path, thresholds=thresholds_used)


# ---------------------------------------------------------------------------
# Optional ML classifier
# ---------------------------------------------------------------------------

def _run_ml_classifier(
    features: dict[str, float],
    rule_cfg: dict,
    rule_config_path: str | None,
    calibrated: bool = True,
) -> tuple[str, dict[str, float]]:
    """
    Run the ML classifier on the feature vector.
    Loads final_classifier.pkl (which contains the TransitLensClassifier wrapper)
    """
    from core.feature_extractor import FEATURE_NAMES

    ml_cfg = rule_cfg.get("ml_classifier", {})
    model_type = ml_cfg.get("model_type", "rf")
    model_path_str = ml_cfg.get("model_path")

    # Locate models directory
    if rule_config_path:
        models_dir = Path(rule_config_path).parent
    else:
        models_dir = _DEFAULT_RULE_CONFIG_PATH.parent

    # Determine model file to load
    if model_path_str:
        model_file = Path(model_path_str)
        if not model_file.is_absolute():
            model_file = models_dir / model_file
    else:
        model_file = models_dir / "final_classifier.pkl"

    # Validate feature order against final_feature_order.json
    feature_order_file = models_dir / "final_feature_order.json"
    if not feature_order_file.exists() and not model_file.exists():
        # Fallback to legacy file order if old model is used
        feature_order_file = models_dir / "feature_order.json"
        
    if feature_order_file.exists():
        with open(feature_order_file, "r") as f:
            saved_features = json.load(f)
        if list(saved_features) != list(FEATURE_NAMES):
            raise ClassificationError(
                f"Feature schema mismatch. Models trained with: {saved_features}, "
                f"but code expects: {list(FEATURE_NAMES)}"
            )
    else:
        if model_file.exists():
            raise ClassificationError(
                f"final_feature_order.json missing for the trained final_classifier.pkl model in {models_dir}!"
            )

    # Validate production_eligible flag in training_metadata.json
    metadata_file = models_dir / "training_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                meta = json.load(f)
            if not meta.get("production_eligible", False):
                # Only raise error if we are running in strict production mode
                # i.e., enabled=True in rule_config and dev_fallback=False
                dev_fallback = ml_cfg.get("dev_fallback", False)
                if not dev_fallback:
                    raise ClassificationError(
                        f"Model at {model_file} is not marked as production eligible in training_metadata.json!"
                    )
        except ClassificationError:
            raise
        except Exception as e:
            logger.warning(f"Could not validate metadata: {e}")

    # Fallback to old model files if final_classifier.pkl doesn't exist
    if not model_file.exists() and not model_path_str:
        old_model_file = models_dir / f"{'rf' if model_type == 'rf' else 'xgb'}_model.pkl"
        scaler_file   = models_dir / "feature_scaler.pkl"
        
        if not old_model_file.exists() or not scaler_file.exists():
            raise FileNotFoundError(f"Neither final_classifier.pkl nor old model files found in {models_dir}")
            
        import pickle
        with open(old_model_file, "rb") as f:
            model = pickle.load(f)
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)
            
        # Verify dimension matches
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(FEATURE_NAMES):
            raise ClassificationError(f"Scaler feature count mismatch: expected {len(FEATURE_NAMES)}, got {scaler.n_features_in_}")
            
        feature_vec = np.array([features.get(k, 0.0) for k in FEATURE_NAMES]).reshape(1, -1)
        scaled = scaler.transform(feature_vec)
        
        probas = model.predict_proba(scaled)[0]
        model_classes = list(model.classes_)
        class_probabilities = {str(cls): float(prob) for cls, prob in zip(model_classes, probas)}
        for cls in CLASSES:
            if cls not in class_probabilities:
                class_probabilities[cls] = 0.0
                
        class_idx = int(np.argmax(probas))
        predicted_class = str(model_classes[class_idx])
        return predicted_class, class_probabilities

    # Load final_classifier.pkl
    import pickle
    with open(model_file, "rb") as f:
        wrapper = pickle.load(f)
        
    # Verify dimensions and class schema consistency
    if hasattr(wrapper, "scaler") and hasattr(wrapper.scaler, "n_features_in_"):
        if wrapper.scaler.n_features_in_ != len(FEATURE_NAMES):
            raise ClassificationError(f"Trained model scaler expects {wrapper.scaler.n_features_in_} features, but code has {len(FEATURE_NAMES)} features.")
            
    feature_vec = np.array([features.get(k, 0.0) for k in FEATURE_NAMES]).reshape(1, -1)
    
    predicted_class = wrapper.predict(feature_vec, calibrated=calibrated)
    class_probabilities = wrapper.predict_proba(feature_vec, calibrated=calibrated)
    
    return predicted_class, class_probabilities