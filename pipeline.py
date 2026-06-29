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
# Empty blend diagnostics (all unavailable)
# ---------------------------------------------------------------------------

def _empty_blend_diagnostics() -> dict:
    """Return a blend diagnostics dict with all fields set to unavailable/None."""
    return {
        "centroid_available": False,
        "centroid_shift": None,
        "centroid_shift_significance": None,
        "centroid_in_transit_x": None,
        "centroid_in_transit_y": None,
        "centroid_out_transit_x": None,
        "centroid_out_transit_y": None,
        "centroid_shift_points_used": None,
        "crowding_available": False,
        "crowding_metric": None,
        "flux_fraction": None,
        "dilution_factor": None,
        "dilution_corrected_depth": None,
        "contamination_ratio": None,
        "neighbor_available": False,
        "gaia_neighbor_count": None,
        "nearest_neighbor_sep_arcsec": None,
        "nearest_neighbor_delta_mag": None,
        "neighbor_flux_ratio_sum": None,
        "aperture_neighbor_risk": None,
        "blend_risk_score": None,
        "blend_risk_level": "unavailable",
        "blend_evidence_flags": [],
    }


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
        feature_result = extract(time_clean, flux_clean, bls_result, config=feature_cfg, metadata=metadata)
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
        clf_cfg = {
            "classification": cfg.get("classification", {}),
            "ml_classifier": cfg.get("ml_classifier", {}),
        }
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

    # ── Stage 6: Plotting (Moved after transit fitting) ───────────────────
    fit_res = None
    plots = {}

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

    # ── Stage 7.1: Blend diagnostics explanation ──────────────────────────
    blend_diagnostics = getattr(feature_result, "blend_diagnostics", None)
    if blend_diagnostics:
        from core.blend_features import get_blend_explanation
        blend_expl = get_blend_explanation(blend_diagnostics, predicted_class)
        if blend_expl:
            explanation = explanation + " " + blend_expl

    # ── Stage 7.5: Scientific Fitting and Uncertainties ─────────────────────
    candidate_detected = bls_result.candidate_detected
    bootstrap_fap = None
    period_uncertainty_days = None
    duration_uncertainty_days = None
    depth_uncertainty = None
    epoch_btjd = None
    fit_quality = None
    
    # Initialize all Phase 7 scientific parameters to None or default
    fit_status = "SUCCESS" if not candidate_detected else "FAILED"
    quality_flags = []
    rp_rstar = None
    rp_rstar_err_lower = None
    rp_rstar_err_upper = None
    a_rstar = None
    a_rstar_err_lower = None
    a_rstar_err_upper = None
    b = None
    b_err_lower = None
    b_err_upper = None
    u1 = None
    u2 = None
    baseline_offset = None
    baseline_slope = None
    jitter = None
    chi2 = None
    reduced_chi2 = None
    bic = None
    aic = None
    residual_rms = None
    durbin_watson = None
    beta_factor = None
    autocorr_lag1 = None
    mcmc_passed = None
    mcmc_rhat = None
    mcmc_ess = None
    observed_depth = None
    observed_depth_uncertainty = None
    corrected_depth = None
    corrected_depth_uncertainty = None
    planet_radius_earth = None
    planet_radius_earth_err_lower = None
    planet_radius_earth_err_upper = None
    inferred_density = None
    inclination_deg = None
    observed_transits = None
    in_transit_cadences = None
    phase_coverage_fraction = None
    alias_warning_fit = False
    alias_type_fit = "none"
    alias_reason_fit = ""
    odd_even_delta_fit = None
    secondary_depth_fit = None
    uncertainty_method = None

    if candidate_detected:
        try:
            from core.transit_fitter import fit_transit
            from core.uncertainty import estimate_uncertainties
            from core.detection_significance import calculate_bootstrap_fap

            init_period = bls_result.best_period
            init_t0 = bls_result.best_t0
            init_duration = bls_result.best_duration
            init_depth = bls_result.best_depth

            fit_cfg = cfg.get("fitting", {})
            fit_res = fit_transit(
                time_clean,
                flux_clean,
                init_period=init_period,
                init_t0=init_t0,
                init_duration=init_duration,
                init_depth=init_depth,
                config=fit_cfg,
                metadata=metadata,
            )

            epoch_btjd = fit_res.get("epoch_btjd")
            fit_quality = fit_res.get("fit_quality")
            fit_status = fit_res.get("fit_status", "FAILED")
            quality_flags = fit_res.get("quality_flags", [])
            rp_rstar = fit_res.get("rp_rstar")
            rp_rstar_err_lower = fit_res.get("rp_rstar_err_lower")
            rp_rstar_err_upper = fit_res.get("rp_rstar_err_upper")
            a_rstar = fit_res.get("a_rstar")
            a_rstar_err_lower = fit_res.get("a_rstar_err_lower")
            a_rstar_err_upper = fit_res.get("a_rstar_err_upper")
            b = fit_res.get("b")
            b_err_lower = fit_res.get("b_err_lower")
            b_err_upper = fit_res.get("b_err_upper")
            u1 = fit_res.get("u1")
            u2 = fit_res.get("u2")
            baseline_offset = fit_res.get("baseline")
            baseline_slope = fit_res.get("slope")
            jitter = fit_res.get("jitter")
            chi2 = fit_res.get("chi2")
            reduced_chi2 = fit_res.get("reduced_chi2")
            bic = fit_res.get("bic")
            aic = fit_res.get("aic")
            residual_rms = fit_res.get("residual_rms")
            durbin_watson = fit_res.get("durbin_watson")
            beta_factor = fit_res.get("beta_factor")
            autocorr_lag1 = fit_res.get("autocorr_lag1")
            mcmc_passed = fit_res.get("mcmc_passed")
            mcmc_rhat = fit_res.get("mcmc_rhat")
            mcmc_ess = fit_res.get("mcmc_ess")
            observed_depth = fit_res.get("observed_depth")
            observed_depth_uncertainty = fit_res.get("observed_depth_uncertainty")
            corrected_depth = fit_res.get("corrected_depth")
            corrected_depth_uncertainty = fit_res.get("corrected_depth_uncertainty")
            planet_radius_earth = fit_res.get("planet_radius_earth")
            planet_radius_earth_err_lower = fit_res.get("planet_radius_earth_err_lower")
            planet_radius_earth_err_upper = fit_res.get("planet_radius_earth_err_upper")
            inferred_density = fit_res.get("inferred_density")
            inclination_deg = fit_res.get("inclination_deg")
            observed_transits = fit_res.get("observed_transits")
            in_transit_cadences = fit_res.get("in_transit_cadences")
            phase_coverage_fraction = fit_res.get("phase_coverage_fraction")
            alias_warning_fit = fit_res.get("alias_warning", False)
            alias_type_fit = fit_res.get("alias_type", "none")
            alias_reason_fit = fit_res.get("alias_reason", "")
            odd_even_delta_fit = fit_res.get("odd_even_delta")
            secondary_depth_fit = fit_res.get("secondary_depth")
            uncertainty_method = fit_res.get("uncertainty_method")

            time_span = float(np.max(time_clean) - np.min(time_clean)) if len(time_clean) > 0 else 0.0
            unc = estimate_uncertainties(fit_res, time_span, bls_result.snr)

            period_uncertainty_days = unc.get("period_uncertainty_days")
            duration_uncertainty_days = unc.get("duration_uncertainty_days")
            depth_uncertainty = unc.get("depth_uncertainty")

            fap_cfg = cfg.get("significance", {})
            fap_iter = fap_cfg.get("bootstrap_iterations", 50)
            bootstrap_fap = calculate_bootstrap_fap(
                time_clean,
                flux_clean,
                period=init_period,
                t0=epoch_btjd if epoch_btjd is not None else init_t0,
                duration=fit_res.get("duration_days", init_duration),
                depth=fit_res.get("depth", init_depth),
                n_iter=fap_iter,
            )
        except Exception as exc:
            logger.error("pipeline: transit fitting or uncertainty calculation failed: %s", exc)

    # ── Run Phase 2 Diagnostics ──
    try:
        from diagnostics import run_diagnostics
        diag_res = run_diagnostics(
            time_clean,
            flux_clean,
            period=bls_result.best_period,
            epoch_btjd=epoch_btjd if epoch_btjd is not None else bls_result.best_t0,
            duration_days=fit_res.get("duration_days") if (fit_res and fit_res.get("duration_days")) else bls_result.best_duration,
            depth=fit_res.get("depth") if (fit_res and fit_res.get("depth")) else bls_result.best_depth,
            centroid_x=metadata.get("centroid_x") if metadata else None,
            centroid_y=metadata.get("centroid_y") if metadata else None,
            quality=metadata.get("quality") if metadata else None,
            metadata=metadata,
            config=cfg
        )
    except Exception as exc:
        logger.error("pipeline: Phase 2 diagnostics execution failed: %s", exc)
        from diagnostics.contracts import get_default_diagnostics_dict
        diag_res = get_default_diagnostics_dict()

    if diag_res:
        # Add legacy keys for backward compatibility
        diag_res["neighbor_available"] = diag_res.get("gaia_available", False)
        diag_res["neighbor_flux_ratio_sum"] = diag_res.get("summed_neighbor_flux_ratio")
        diag_res["nearest_neighbor_delta_mag"] = diag_res.get("nearest_neighbor_delta_gmag")
        diag_res["contamination_ratio"] = diag_res.get("contamination_fraction")
        diag_res["flux_fraction"] = diag_res.get("flfrcsap")
        diag_res["dilution_factor"] = diag_res.get("correction_factor")
        diag_res["centroid_shift"] = diag_res.get("centroid_shift_pixels")
        diag_res["centroid_shift_significance"] = diag_res.get("centroid_shift_significance")
        diag_res["centroid_shift_points_used"] = diag_res.get("centroid_points_in", 0) + diag_res.get("centroid_points_out", 0)
        
        # Merge legacy list representation of flags
        diag_res["blend_evidence_flags"] = diag_res.get("blend_evidence_flags", [])
        
        if diag_res.get("recommendation_reason"):
            explanation = explanation + " Phase 2 Vetting: " + diag_res["recommendation_reason"]

    from core.classifier import CLASSES
    class_probabilities = getattr(classification_result, "class_probabilities", None)
    if class_probabilities is None:
        class_probabilities = {}

    # ── Stage 8: Assemble result dict ─────────────────────────────────────
    # Generate all plots (enhanced with fitting results)
    plots = _generate_plots(
        time=time,
        flux=flux,
        time_clean=time_clean,
        flux_clean=flux_clean,
        bls_result=bls_result,
        target_id=target_id,
        cfg=cfg,
        fit_result=fit_res,
    )
    processing_time_ms = (_time.perf_counter() - wall_start) * 1000

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
        # Scientific uncertainties and significance
        "bootstrap_fap":             round(bootstrap_fap, 6) if (candidate_detected and bootstrap_fap is not None) else None,
        "class_probabilities":        {cls: round(float(prob), 6) for cls, prob in class_probabilities.items()},
        "class_probability_status":   "calibrated" if class_probabilities else "unavailable_rule_only",
        "ml_inference_status":        "available" if class_probabilities else "restricted",
        "period_uncertainty_days":   round(period_uncertainty_days, 8) if (candidate_detected and period_uncertainty_days is not None) else None,
        "duration_uncertainty_days": round(duration_uncertainty_days, 8) if (candidate_detected and duration_uncertainty_days is not None) else None,
        "depth_uncertainty":         round(depth_uncertainty, 8) if (candidate_detected and depth_uncertainty is not None) else None,
        "epoch_btjd":                round(epoch_btjd, 6) if (candidate_detected and epoch_btjd is not None) else None,
        "fit_quality":                round(fit_quality, 6) if (candidate_detected and fit_quality is not None) else None,
        
        # New Phase 7 scientific parameters
        "fit_status":                fit_status,
        "quality_flags":             quality_flags,
        "rp_rstar":                  round(rp_rstar, 6) if rp_rstar is not None else None,
        "rp_rstar_err_lower":        round(rp_rstar_err_lower, 6) if rp_rstar_err_lower is not None else None,
        "rp_rstar_err_upper":        round(rp_rstar_err_upper, 6) if rp_rstar_err_upper is not None else None,
        "a_rstar":                   round(a_rstar, 4) if a_rstar is not None else None,
        "a_rstar_err_lower":         round(a_rstar_err_lower, 4) if a_rstar_err_lower is not None else None,
        "a_rstar_err_upper":         round(a_rstar_err_upper, 4) if a_rstar_err_upper is not None else None,
        "b":                         round(b, 4) if b is not None else None,
        "b_err_lower":               round(b_err_lower, 4) if b_err_lower is not None else None,
        "b_err_upper":               round(b_err_upper, 4) if b_err_upper is not None else None,
        "u1":                        round(u1, 4) if u1 is not None else None,
        "u2":                        round(u2, 4) if u2 is not None else None,
        "baseline_offset":           round(baseline_offset, 6) if baseline_offset is not None else None,
        "baseline_slope":            round(baseline_slope, 8) if baseline_slope is not None else None,
        "jitter":                    round(jitter, 6) if jitter is not None else None,
        "chi2":                      round(chi2, 2) if chi2 is not None else None,
        "reduced_chi2":              round(reduced_chi2, 4) if reduced_chi2 is not None else None,
        "bic":                       round(bic, 2) if bic is not None else None,
        "aic":                       round(aic, 2) if aic is not None else None,
        "residual_rms":              round(residual_rms, 6) if residual_rms is not None else None,
        "durbin_watson":             round(durbin_watson, 4) if durbin_watson is not None else None,
        "beta_factor":               round(beta_factor, 4) if beta_factor is not None else None,
        "autocorr_lag1":             round(autocorr_lag1, 4) if autocorr_lag1 is not None else None,
        "mcmc_passed":               mcmc_passed,
        "mcmc_rhat":                 round(mcmc_rhat, 4) if mcmc_rhat is not None else None,
        "mcmc_ess":                  mcmc_ess,
        "observed_depth":            round(observed_depth, 6) if observed_depth is not None else None,
        "observed_depth_uncertainty": round(observed_depth_uncertainty, 6) if observed_depth_uncertainty is not None else None,
        "corrected_depth":           round(corrected_depth, 6) if corrected_depth is not None else None,
        "corrected_depth_uncertainty": round(corrected_depth_uncertainty, 6) if corrected_depth_uncertainty is not None else None,
        "planet_radius_earth":       round(planet_radius_earth, 4) if planet_radius_earth is not None else None,
        "planet_radius_earth_err_lower": round(planet_radius_earth_err_lower, 4) if planet_radius_earth_err_lower is not None else None,
        "planet_radius_earth_err_upper": round(planet_radius_earth_err_upper, 4) if planet_radius_earth_err_upper is not None else None,
        "inferred_density":          round(inferred_density, 4) if inferred_density is not None else None,
        "inclination_deg":           round(inclination_deg, 4) if inclination_deg is not None else None,
        "observed_transits":         observed_transits,
        "in_transit_cadences":       in_transit_cadences,
        "phase_coverage_fraction":   round(phase_coverage_fraction, 4) if phase_coverage_fraction is not None else None,
        "alias_warning_fitter":      alias_warning_fit,
        "alias_type_fitter":         alias_type_fit,
        "alias_reason_fitter":       alias_reason_fit,
        "odd_even_delta_fitter":     round(odd_even_delta_fit, 6) if odd_even_delta_fit is not None else None,
        "secondary_depth_fitter":    round(secondary_depth_fit, 6) if secondary_depth_fit is not None else None,
        "uncertainty_method":        uncertainty_method,
        
        # Full feature vector
        "features": {k: round(float(v), 8) for k, v in features.items()},
        # Blend/contamination diagnostics
        "diagnostics": {
            **diag_res,
            "blend": diag_res,
        },
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
    fit_result: dict | None = None,
) -> dict:
    """
    Generate diagnostic plots as base64 PNG strings.
    """
    empty_plots = {
        "raw_lightcurve": "",
        "cleaned_lightcurve": "",
        "periodogram": "",
        "phase_folded": "",
        "transit_stack": "",
        "posterior_corner": "",
        "alias_comparison": "",
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
            fit_result=fit_result,
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
    """
    aliases = {
        "exoplanet_like": "exoplanet_transit",
        "eclipsing_binary_like": "eclipsing_binary",
        "noise_or_other": "stellar_variability_or_other",
    }
    predicted_class = aliases.get(predicted_class, predicted_class)
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

    if predicted_class == "exoplanet_transit":
        lines.append(
            f"Classified as exoplanet_transit with {pct}% confidence. "
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

    elif predicted_class == "eclipsing_binary":
        lines.append(
            f"Classified as eclipsing_binary with {pct}% confidence. "
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

    elif predicted_class == "blend_contamination":
        lines.append(
            f"Classified as blend_contamination with {pct}% confidence. "
            f"A periodic transit-like signal was detected at {period:.4f} days, "
            f"but diagnostics indicate nearby stellar companion or centroid displacement. "
            f"The observed depth of {depth*100:.2f}% may be diluted by crowding."
        )
    else:  # stellar_variability_or_other
        lines.append(
            f"Classified as stellar_variability_or_other with {pct}% confidence. "
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
    """
    processing_time_ms = (_time.perf_counter() - wall_start) * 1000
    from core.feature_extractor import FEATURE_NAMES, _FALLBACK

    return {
        "target_id": target_id,
        "candidate_detected": False,
        "predicted_class": "stellar_variability_or_other",
        "confidence": 0.0,
        "period_days": None,
        "duration_days": None,
        "depth": None,
        "snr": None,
        "transit_count": None,
        "bootstrap_fap": None,
        "class_probabilities": {},
        "class_probability_status": "unavailable_rule_only",
        "ml_inference_status": "restricted",
        "period_uncertainty_days": None,
        "duration_uncertainty_days": None,
        "depth_uncertainty": None,
        "epoch_btjd": None,
        "fit_quality": None,
        "features": {k: float(_FALLBACK[k]) for k in FEATURE_NAMES},
        "diagnostics": {
            "blend": _empty_blend_diagnostics(),
        },
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
    nullable = [
        "period_days", "duration_days", "depth", "snr", "transit_count",
        "bootstrap_fap", "period_uncertainty_days", "duration_uncertainty_days",
        "depth_uncertainty", "epoch_btjd", "fit_quality",
        "rp_rstar", "rp_rstar_err_lower", "rp_rstar_err_upper",
        "a_rstar", "a_rstar_err_lower", "a_rstar_err_upper",
        "b", "b_err_lower", "b_err_upper", "u1", "u2",
        "baseline_offset", "baseline_slope", "jitter", "chi2", "reduced_chi2",
        "bic", "aic", "residual_rms", "durbin_watson", "beta_factor", "autocorr_lag1",
        "mcmc_passed", "mcmc_rhat", "mcmc_ess", "observed_depth", "observed_depth_uncertainty",
        "corrected_depth", "corrected_depth_uncertainty", "planet_radius_earth",
        "planet_radius_earth_err_lower", "planet_radius_earth_err_upper",
        "inferred_density", "inclination_deg", "observed_transits", "in_transit_cadences",
        "phase_coverage_fraction", "odd_even_delta_fitter", "secondary_depth_fitter",
        "uncertainty_method"
    ]
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
            if result[key] is None and key not in ("planet_radius_earth", "planet_radius_earth_err_lower", "planet_radius_earth_err_upper"):
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

    # Invariant 3: predicted_class is always one of the allowed strings
    from core.classifier import CLASSES
    if result["predicted_class"] not in CLASSES:
        logger.warning(
            "invariant violation: predicted_class='%s' not in CLASSES — "
            "setting to stellar_variability_or_other", result["predicted_class"]
        )
        result["predicted_class"] = "stellar_variability_or_other"

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

    # Invariant 6: features dict has exactly 16 keys
    from core.feature_extractor import FEATURE_NAMES
    if set(result["features"].keys()) != set(FEATURE_NAMES):
        logger.warning("invariant: features dict has wrong keys")


def analyze_tess_multi_sector(
    tic_id: str,
    sectors: list[int] | None = None,
    metadata: dict | None = None,
    config: dict | None = None,
) -> dict:
    """
    Retrieves and analyzes multiple sectors for a TESS target.
    Analyzes each sector separately first, validates them, and optionally
    combines them after normalization and offset alignment.
    """
    import os
    import sys
    from pathlib import Path
    logger = logging.getLogger(__name__)

    # Ensure data-pipeline is in path
    repo_root = Path(__file__).resolve().parent
    dp_path = repo_root.parent / "transitlens-data-pipeline"
    if dp_path.exists() and str(dp_path) not in sys.path:
        sys.path.insert(0, str(dp_path))

    from interface import load_light_curve
    from real_tess.mast_loader import TessDataUnavailableError

    # Ensure input metadata has target_id set
    clean_id = tic_id.upper().replace("TIC", "").replace("-", "").strip()
    clean_id = "".join(clean_id.split())
    
    merged_metadata = metadata.copy() if metadata else {}
    if "target_id" not in merged_metadata:
        merged_metadata["target_id"] = f"TIC {clean_id}"

    # Resolve cache directory
    cache_dir = config.get("cache_dir") if config else None
    if not cache_dir:
        cache_dir = str(dp_path / "real_tess" / "cache")

    # 1. If sectors not specified, query MAST to discover them
    if not sectors:
        try:
            try:
                from astroquery.mast import Observations
                Observations.TIMEOUT = 10
            except Exception:
                pass
            import lightkurve as lk
            search_result = lk.search_lightcurve(f"TIC {clean_id}", mission="TESS")
            all_sectors = []
            if len(search_result) > 0:
                if hasattr(search_result, "sequence_number"):
                    all_sectors = sorted(list(set(int(s) for s in search_result.sequence_number)))
                elif hasattr(search_result, "sector"):
                    all_sectors = sorted(list(set(int(s) for s in search_result.sector)))
                else:
                    table = getattr(search_result, "table", None)
                    if table is not None:
                        col = "sequence_number" if "sequence_number" in table.colnames else "sector"
                        all_sectors = sorted(list(set(int(s) for s in table[col])))
            
            # Prioritize cached sectors
            cached_secs = []
            non_cached_secs = []
            for sec in all_sectors:
                p3 = os.path.join(cache_dir, f"TIC{clean_id}_sector{sec:03d}.fits")
                p2 = os.path.join(cache_dir, f"TIC{clean_id}_sector{sec:02d}.fits")
                if os.path.exists(p3) or os.path.exists(p2):
                    cached_secs.append(sec)
                else:
                    non_cached_secs.append(sec)
            
            # If cached sectors exist, only use cached sectors. Otherwise, download the single newest sector.
            if cached_secs:
                sectors = sorted(cached_secs)
            else:
                non_cached_secs.sort(reverse=True)
                sectors = sorted(non_cached_secs[:1])
        except Exception as e:
            logger.warning(f"Failed to query available sectors on MAST: {e}. Falling back to default best sector.")
            sectors = []

    # If still no sectors, fall back to best single sector via load_light_curve
    if not sectors:
        lc_data = load_light_curve(source="tess", target_id=tic_id, config=config)
        sector_metadata = {**merged_metadata, **lc_data.get("metadata", {})}
        sector_metadata["target_id"] = f"TIC {clean_id}"
        return analyze_light_curve(lc_data["time"], lc_data["flux"], sector_metadata, config)

    logger.info(f"Multi-sector analysis triggered for TIC {tic_id} over sectors: {sectors}")

    sector_results = []
    times_all = []
    fluxes_all = []

    for sec in sectors:
        try:
            # Load sector
            lc_data = load_light_curve(source="tess", target_id=tic_id, config={"sector": sec})
            time_sec = np.array(lc_data["time"])
            flux_sec = np.array(lc_data["flux"])
            
            # Analyze separately first
            sec_metadata = {**merged_metadata, **lc_data.get("metadata", {})}
            sec_metadata["target_id"] = f"TIC {clean_id}"
            res_sec = analyze_light_curve(time_sec, flux_sec, sec_metadata, config)
            sector_results.append((sec, res_sec))
            
            # Normalize and offset checks
            median_sec = np.median(flux_sec)
            if abs(median_sec - 1.0) > 0.05:
                # Re-align to 1.0
                flux_sec = flux_sec / median_sec
                
            times_all.append(time_sec)
            fluxes_all.append(flux_sec)
        except Exception as e:
            logger.warning(f"Failed to load or analyze sector {sec}: {e}")

    if not sector_results:
        raise TessDataUnavailableError(f"No sectors could be successfully loaded or analyzed for TIC {tic_id}")

    # If only one sector succeeded, return its result
    if len(sector_results) == 1:
        res = sector_results[0][1]
        if "metadata" in res:
            res["metadata"]["target_id"] = f"TIC {clean_id}"
        return res

    # 2. Combine sectors after sorting by time
    combined_time = np.concatenate(times_all)
    combined_flux = np.concatenate(fluxes_all)
    
    sort_idx = np.argsort(combined_time)
    combined_time = combined_time[sort_idx]
    combined_flux = combined_flux[sort_idx]

    # Deduplicate timestamps just in case of overlaps
    _, unique_idx = np.unique(combined_time, return_index=True)
    combined_time = combined_time[unique_idx]
    combined_flux = combined_flux[unique_idx]

    # Combine metadata
    combined_metadata = merged_metadata.copy()
    combined_metadata["target_id"] = f"TIC {clean_id}"
    combined_metadata["sector"] = sectors  # list of sectors
    combined_metadata["cadence_min"] = sector_results[0][1].get("metadata", {}).get("cadence_min", 2.0)
    combined_metadata["time_span_days"] = float(combined_time[-1] - combined_time[0])
    combined_metadata["multi_sector_details"] = {
        "analyzed_sectors": sectors,
        "single_sector_predictions": {
            str(sec): {
                "predicted_class": res.get("predicted_class"),
                "confidence": res.get("confidence"),
                "period_days": res.get("period_days"),
                "candidate_detected": res.get("candidate_detected")
            }
            for sec, res in sector_results
        }
    }

    # Run full analysis on combined light curve
    combined_result = analyze_light_curve(combined_time, combined_flux, combined_metadata, config)
    combined_result["metadata"] = combined_metadata
    
    return combined_result
