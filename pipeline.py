"""
pipeline.py
-----------
Public entry point for transitlens-ml-core.

Single exported function:

    analyze_light_curve(time, flux, metadata=None, config=None) → dict

This is the face of the repo. All other modules are internal. When
transitlens-platform or an API caller wants an analysis, it calls this
function and receives a fully self-contained result dict.

Execution order (Phases 1–4 in sequence):
    1. Load config (file + optional override)
    2. preprocess.clean()         → time_clean, flux_clean
    3. bls_detector.detect()      → bls_result
    4. feature_extractor.extract()→ feature_result
    5. classifier.classify()      → classification_result
    6. confidence.score()         → confidence_float
    7. plotter.generate_all()     → plots dict   [optional, skipped if matplotlib absent]
    8. Build explanation string
    9. Assemble and return full result dict

Error handling:
    - InvalidInputError propagates to the caller (broken input arrays).
    - All other internal exceptions are caught and converted to a graceful
      result dict with candidate_detected=False and an explanation describing
      the failure.

Used by: api/routes.py, eval/evaluate.py, tests/test_pipeline.py
"""

from __future__ import annotations

import logging
import time as _time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core.bls_detector import detect
from core.classifier import classify, reload_rule_config
from core.confidence import score_with_breakdown
from core.exceptions import (
    InvalidInputError,
    InsufficientDataError,
    MLCoreError,
)
from core.feature_extractor import extract
from core.preprocess import clean

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading — module-level cache (not per-call)
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_global_config: dict | None = None


def _load_config(override: dict | None = None) -> dict:
    """
    Load config.yaml once, then merge any per-call override recursively.

    The base config is module-level cached. The override is applied fresh
    each call (it may differ per request).
    """
    global _global_config
    if _global_config is None:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "r") as f:
                _global_config = yaml.safe_load(f) or {}
            logger.debug("pipeline: config loaded from %s", _CONFIG_PATH)
        else:
            logger.warning(
                "pipeline: config.yaml not found at %s — using empty config", _CONFIG_PATH
            )
            _global_config = {}

    if not override:
        return _global_config

    return _deep_merge(_global_config, override)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_light_curve(
    time: Any,
    flux: Any,
    metadata: dict | None = None,
    config: dict | None = None,
) -> dict:
    """
    Analyse a raw normalised light curve and return a complete result dict.

    Parameters
    ----------
    time : array-like of float
        BTJD timestamps, monotonically increasing.
    flux : array-like of float
        Normalised flux values (median ≈ 1.0).
    metadata : dict or None
        Optional metadata from data-pipeline's load_light_curve().
        Used for target_id and optional ground-truth fields.
    config : dict or None
        Optional override for any pipeline parameter. Keys mirror config.yaml.
        Nested dicts are merged recursively; top-level keys override defaults.

    Returns
    -------
    dict
        Complete result dict. Always contains all keys defined in the
        interface contract (Section 3 of the build plan), including:
            candidate_detected, predicted_class, confidence, period_days,
            duration_days, depth, snr, transit_count, features, explanation,
            plots, processing_time_ms, pipeline_version.

    Raises
    ------
    InvalidInputError
        If time/flux have mismatched lengths, contain infinities, or are
        fundamentally broken. All other exceptions are caught internally.
    """
    wall_start = _time.perf_counter()

    # ── Convert to numpy arrays early so validation is consistent ─────────
    try:
        time = np.asarray(time, dtype=float)
        flux = np.asarray(flux, dtype=float)
    except Exception as exc:
        raise InvalidInputError(
            f"Could not convert time/flux to float arrays: {exc}"
        ) from exc

    # ── Load config ───────────────────────────────────────────────────────
    cfg = _load_config(config)
    version = str(cfg.get("version", "0.1.0"))
    target_id = str((metadata or {}).get("target_id", "unknown"))

    logger.info(
        "pipeline: starting analysis for target_id='%s' "
        "(n=%d points, version=%s)",
        target_id, len(time), version,
    )

    # ── Stage 1: Preprocessing ────────────────────────────────────────────
    try:
        preprocess_cfg = cfg.get("preprocessing", {})
        preprocess_result = clean(time, flux, config=preprocess_cfg)
        time_clean = preprocess_result.time
        flux_clean = preprocess_result.flux
    except InvalidInputError:
        # Propagate to caller — broken input is caller's problem
        raise
    except Exception as exc:
        logger.error("pipeline: preprocessing failed: %s", exc)
        return _error_result(
            target_id=target_id,
            version=version,
            wall_start=wall_start,
            explanation=f"Preprocessing failed: {exc}",
        )

    # ── Stage 2: BLS Detection ────────────────────────────────────────────
    try:
        bls_cfg = cfg.get("bls", {})
        bls_result = detect(time_clean, flux_clean, config=bls_cfg)
    except Exception as exc:
        logger.error("pipeline: BLS detection failed: %s", exc)
        return _error_result(
            target_id=target_id,
            version=version,
            wall_start=wall_start,
            explanation=f"BLS detection failed: {exc}",
        )

    # ── Stage 3: Feature Extraction ───────────────────────────────────────
    try:
        feature_cfg = cfg.get("features", {})
        feature_result = extract(time_clean, flux_clean, bls_result, config=feature_cfg)
        features = feature_result.features
    except Exception as exc:
        logger.error("pipeline: feature extraction failed: %s", exc)
        return _error_result(
            target_id=target_id,
            version=version,
            wall_start=wall_start,
            explanation=f"Feature extraction failed: {exc}",
        )

    # ── Stage 4: Classification ───────────────────────────────────────────
    try:
        clf_cfg = {"classification": cfg.get("classification", {})}
        classification_result = classify(features, config=clf_cfg)
        predicted_class = classification_result.predicted_class
    except Exception as exc:
        logger.error("pipeline: classification failed: %s", exc)
        return _error_result(
            target_id=target_id,
            version=version,
            wall_start=wall_start,
            explanation=f"Classification failed: {exc}",
        )

    # ── Stage 5: Confidence Scoring ───────────────────────────────────────
    try:
        confidence_float, confidence_breakdown = score_with_breakdown(
            features, predicted_class
        )
    except Exception as exc:
        logger.warning("pipeline: confidence scoring failed: %s — defaulting to 0.5", exc)
        confidence_float = 0.5
        confidence_breakdown = []

    # ── Stage 6: Plotting ─────────────────────────────────────────────────
    plots = _generate_plots(
        time=time,
        flux=flux,
        time_clean=time_clean,
        flux_clean=flux_clean,
        bls_result=bls_result,
        target_id=target_id,
        cfg=cfg,
    )

    # ── Stage 7: Build explanation string ─────────────────────────────────
    explanation = _build_explanation(
        target_id=target_id,
        predicted_class=predicted_class,
        confidence=confidence_float,
        features=features,
        bls_result=bls_result,
        classification_result=classification_result,
        confidence_breakdown=confidence_breakdown,
        feature_result=feature_result,
    )

    # ── Stage 8: Assemble result dict ─────────────────────────────────────
    processing_time_ms = (_time.perf_counter() - wall_start) * 1000

    candidate_detected = bls_result.candidate_detected

    result = {
        "target_id": target_id,
        "candidate_detected": candidate_detected,
        "predicted_class": predicted_class,
        "confidence": round(confidence_float, 6),
        # Top-level detection parameters (null when no candidate)
        "period_days":    round(bls_result.best_period, 6)   if candidate_detected else None,
        "duration_days":  round(bls_result.best_duration, 6) if candidate_detected else None,
        "depth":          round(bls_result.best_depth, 6)    if candidate_detected else None,
        "snr":            round(bls_result.snr, 4)           if candidate_detected else None,
        "transit_count":  features["transit_count"]          if candidate_detected else None,
        # Full 11-feature vector
        "features": {k: round(float(v), 8) for k, v in features.items()},
        # Human-readable explanation
        "explanation": explanation,
        # Diagnostic plots (base64 PNG strings)
        "plots": plots,
        # Provenance
        "processing_time_ms": round(processing_time_ms, 2),
        "pipeline_version": version,
    }

    # ── Invariant check ───────────────────────────────────────────────────
    _check_invariants(result)

    logger.info(
        "pipeline: complete — target='%s' class='%s' conf=%.3f "
        "detected=%s period=%s ms=%.0f",
        target_id, predicted_class, confidence_float,
        candidate_detected,
        f"{bls_result.best_period:.4f}d" if candidate_detected else "N/A",
        processing_time_ms,
    )

    return result


# ---------------------------------------------------------------------------
# Plotting (optional — graceful if matplotlib absent)
# ---------------------------------------------------------------------------

def _generate_plots(
    time: np.ndarray,
    flux: np.ndarray,
    time_clean: np.ndarray,
    flux_clean: np.ndarray,
    bls_result: Any,
    target_id: str,
    cfg: dict,
) -> dict:
    """
    Generate all four diagnostic plots as base64 PNG strings.

    Returns a dict with all four keys always present. If plotting fails
    or matplotlib is unavailable, all values are empty strings.
    """
    empty_plots = {
        "raw_lightcurve": "",
        "cleaned_lightcurve": "",
        "periodogram": "",
        "phase_folded": "",
    }

    try:
        from core.plotter import generate_all
        plot_cfg = cfg.get("plotting", {})
        return generate_all(
            time=time,
            flux=flux,
            time_clean=time_clean,
            flux_clean=flux_clean,
            bls_result=bls_result,
            target_id=target_id,
            config=plot_cfg,
        )
    except ImportError:
        logger.info("pipeline: matplotlib not available — skipping plots")
        return empty_plots
    except Exception as exc:
        logger.warning("pipeline: plotting failed: %s — returning empty plots", exc)
        return empty_plots


# ---------------------------------------------------------------------------
# Explanation string builder
# ---------------------------------------------------------------------------

def _build_explanation(
    target_id: str,
    predicted_class: str,
    confidence: float,
    features: dict,
    bls_result: Any,
    classification_result: Any,
    confidence_breakdown: list,
    feature_result: Any,
) -> str:
    """
    Build a specific, human-readable explanation of the classification.

    Format (one paragraph):
        - Predicted class and confidence.
        - Detection parameters (period, depth, SNR) if detected.
        - Top 2–3 driving features with plain-language interpretation.
        - Any caveats (alias warning, low transit count, ML disagreement).
    """
    pct = int(round(confidence * 100))
    lines: list[str] = []

    depth    = features.get("depth", 0.0)
    snr      = features.get("snr", 0.0)
    period   = features.get("period_days", 0.0)
    dtnr     = features.get("depth_to_noise_ratio", 0.0)
    odd_even = features.get("odd_even_depth_delta", 0.0)
    v_shape  = features.get("v_shape_score", 0.0)
    n_tr     = int(features.get("transit_count", 0))
    duration = features.get("duration_days", 0.0)

    if predicted_class == "exoplanet_like":
        lines.append(
            f"Classified as exoplanet_like with {pct}% confidence. "
            f"A periodic transit signal was detected at {period:.4f} days "
            f"with a depth of {depth*100:.2f}% ({snr:.1f}\u03c3 above noise). "
            f"The depth is consistent with a sub-Jupiter-sized planet transiting a Sun-like star."
        )
        lines.append(
            f"Odd/even transit depths are nearly identical "
            f"(\u0394depth = {odd_even:.4f}), ruling out a double-lined eclipsing binary. "
            f"The transit profile is flat-bottomed (V-shape score = {v_shape:.3f}), "
            f"consistent with a planetary disc crossing a stellar surface."
        )
        lines.append(
            f"{n_tr} transit event(s) detected across the observation window "
            f"(transit duration \u2248 {duration*24:.1f} hours)."
        )

    elif predicted_class == "eclipsing_binary_like":
        lines.append(
            f"Classified as eclipsing_binary_like with {pct}% confidence. "
            f"A strong periodic signal was detected at {period:.4f} days "
            f"with depth {depth*100:.2f}% (SNR = {snr:.1f})."
        )
        if depth > 0.050:
            lines.append(
                f"The primary discriminator is transit depth: {depth*100:.2f}% far exceeds "
                f"the planetary threshold of 5.0%. No planet can block more than ~3% "
                f"of a Sun-like star's flux; a deeper signal indicates a stellar companion."
            )
        if odd_even > 0.020:
            lines.append(
                f"Odd/even transit depth asymmetry (\u0394 = {odd_even:.4f}) further "
                f"supports an eclipsing binary interpretation: alternating primary and "
                f"secondary eclipses produce different depths."
            )
        if v_shape > 0.40:
            lines.append(
                f"The V-shaped transit profile (score = {v_shape:.3f}) is consistent "
                f"with a grazing or total stellar eclipse rather than a planetary transit."
            )

    else:  # noise_or_other
        lines.append(
            f"Classified as noise_or_other with {pct}% confidence. "
            f"No significant periodic transit signal was detected in this light curve."
        )
        lines.append(
            f"The BLS power spectrum shows no dominant peak above the detection threshold "
            f"(SNR = {snr:.2f}, depth-to-noise ratio = {dtnr:.2f}). "
            f"The signal is consistent with photon noise or unresolved stellar variability."
        )

    # Caveats
    caveats: list[str] = []
    if bls_result.alias_warning:
        caveats.append(
            f"Alias warning: a harmonic period has comparable BLS power — "
            f"the detected period of {period:.4f} days may be a 2:1 alias."
        )
    if n_tr < 3 and bls_result.candidate_detected:
        caveats.append(
            f"Caution: only {n_tr} transit event(s) detected. "
            f"Period estimate is less reliable with fewer than 3 transits."
        )
    if not classification_result.ml_agreement and classification_result.ml_class:
        caveats.append(
            f"Note: the ML classifier suggested '{classification_result.ml_class}' "
            f"(rule-based prediction used as it is more interpretable)."
        )

    if caveats:
        lines.append(" ".join(caveats))

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Error result — graceful failure dict
# ---------------------------------------------------------------------------

def _error_result(
    target_id: str,
    version: str,
    wall_start: float,
    explanation: str,
) -> dict:
    """
    Return a well-formed result dict when the pipeline fails internally.

    All invariants are satisfied: candidate_detected=False, all nullable
    top-level fields are None, all four plot keys present as empty strings.
    """
    processing_time_ms = (_time.perf_counter() - wall_start) * 1000
    from core.feature_extractor import FEATURE_NAMES, _FALLBACK

    return {
        "target_id": target_id,
        "candidate_detected": False,
        "predicted_class": "noise_or_other",
        "confidence": 0.0,
        "period_days": None,
        "duration_days": None,
        "depth": None,
        "snr": None,
        "transit_count": None,
        "features": {k: float(_FALLBACK[k]) for k in FEATURE_NAMES},
        "explanation": explanation,
        "plots": {
            "raw_lightcurve": "",
            "cleaned_lightcurve": "",
            "periodogram": "",
            "phase_folded": "",
        },
        "processing_time_ms": round(processing_time_ms, 2),
        "pipeline_version": version,
    }


# ---------------------------------------------------------------------------
# Invariant checker
# ---------------------------------------------------------------------------

def _check_invariants(result: dict) -> None:
    """
    Assert that the result dict satisfies all interface contract invariants.

    Raises ValueError if any invariant is violated (indicates a pipeline bug).
    """
    detected = result["candidate_detected"]

    # Invariant 1: nullable fields are consistent with detection status
    nullable = ["period_days", "duration_days", "depth", "snr", "transit_count"]
    if not detected:
        for key in nullable:
            if result[key] is not None:
                logger.warning(
                    "invariant violation: candidate_detected=False but %s=%s (setting to None)",
                    key, result[key],
                )
                result[key] = None
    else:
        for key in nullable:
            if result[key] is None:
                logger.warning(
                    "invariant violation: candidate_detected=True but %s is None",
                    key,
                )

    # Invariant 2: confidence is always a float in [0, 1]
    conf = result["confidence"]
    if not (isinstance(conf, (float, int)) and 0.0 <= float(conf) <= 1.0):
        logger.warning(
            "invariant violation: confidence=%s out of [0,1] — clamping", conf
        )
        result["confidence"] = float(np.clip(float(conf), 0.0, 1.0))

    # Invariant 3: predicted_class is always one of the three allowed strings
    from core.classifier import CLASSES
    if result["predicted_class"] not in CLASSES:
        logger.warning(
            "invariant violation: predicted_class='%s' not in CLASSES — "
            "setting to noise_or_other", result["predicted_class"]
        )
        result["predicted_class"] = "noise_or_other"

    # Invariant 4: all four plot keys are always present
    required_plots = {"raw_lightcurve", "cleaned_lightcurve", "periodogram", "phase_folded"}
    plots = result.get("plots", {})
    for key in required_plots:
        if key not in plots:
            logger.warning("invariant: missing plot key '%s' — inserting empty string", key)
            plots[key] = ""
    result["plots"] = plots

    # Invariant 5: explanation is a non-empty string
    if not result.get("explanation"):
        result["explanation"] = "Analysis complete."

    # Invariant 6: features dict has exactly 11 keys
    from core.feature_extractor import FEATURE_NAMES
    if set(result["features"].keys()) != set(FEATURE_NAMES):
        logger.warning("invariant: features dict has wrong keys")