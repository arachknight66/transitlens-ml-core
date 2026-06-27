"""
core/feature_extractor.py
--------------------------
Feature extraction from BLS detection results and the phase-folded light curve.

Computes exactly 11 physically-interpretable numerical features that serve as
inputs to the classifier. Every feature corresponds to something a human
astronomer would examine when deciding whether a signal is a planet, an
eclipsing binary, or noise.

Features (in order):
    1.  bls_power             — normalised BLS peak power
    2.  snr                   — signal-to-noise ratio of detection
    3.  period_days           — best-fit orbital period
    4.  duration_days         — best-fit transit duration
    5.  depth                 — fractional flux drop at transit centre
    6.  transit_count         — number of transit events in time series
    7.  odd_even_depth_delta  — |depth_odd - depth_even| (EB discriminator)
    8.  v_shape_score         — 0 = flat-bottomed, 1 = fully V-shaped
    9.  local_noise           — RMS of out-of-transit flux (phase-folded)
    10. depth_to_noise_ratio  — depth / local_noise (local SNR)
    11. phase_shape_kurtosis  — excess kurtosis of in-transit profile

Used by: pipeline.py, classifier.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from core.bls_detector import BLSResult
from core.exceptions import FeatureExtractionError
from core.utils import bin_phase_folded, phase_fold

logger = logging.getLogger(__name__)

# Feature names in canonical order — used for validation
FEATURE_NAMES = (
    "bls_power",
    "snr",
    "period_days",
    "duration_days",
    "depth",
    "transit_count",
    "odd_even_depth_delta",
    "v_shape_score",
    "local_noise",
    "depth_to_noise_ratio",
    "phase_shape_kurtosis",
    "bls_sde",
    "secondary_eclipse_depth",
    "centroid_shift",
    "crowding_metric",
    "gaia_neighbor_count",
)

# Default fallback values used when a feature cannot be computed reliably
_FALLBACK = {
    "bls_power": 0.0,
    "snr": 0.0,
    "period_days": 0.0,
    "duration_days": 0.0,
    "depth": 0.0,
    "transit_count": 0,
    "odd_even_depth_delta": 0.0,
    "v_shape_score": 0.0,
    "local_noise": 1.0,
    "depth_to_noise_ratio": 0.0,
    "phase_shape_kurtosis": 0.0,
    "bls_sde": 0.0,
    "secondary_eclipse_depth": 0.0,
    "centroid_shift": 0.0,
    "crowding_metric": 1.0,
    "gaia_neighbor_count": 0,
}

# Minimum in-transit points needed to compute shape features reliably
_MIN_TRANSIT_POINTS = 5
# Minimum transits needed for odd/even comparison
_MIN_TRANSITS_ODD_EVEN = 4
# Phase exclusion buffer around transit for out-of-transit noise (factor × half-duration)
_NOISE_EXCLUSION_FACTOR = 1.5
# Number of bins for phase-folded light curve
_N_PHASE_BINS = 100


# ---------------------------------------------------------------------------
# Public result container
# ---------------------------------------------------------------------------

@dataclass
class FeatureResult:
    """
    Container for the 16 extracted features plus reliability metadata.

    Attributes
    ----------
    features : dict[str, float]
        Exactly 16 keys — the canonical FEATURE_NAMES.
    reliable : dict[str, bool]
        Per-feature reliability flag. False means the feature was set to
        its fallback value because computation was not possible (e.g.
        too few transit events for odd_even_depth_delta).
    candidate_detected : bool
        Forwarded from BLSResult.candidate_detected.
    blend_diagnostics : dict or None
        Full blend/contamination diagnostics from blend_features module.
        Contains availability flags, centroid/crowding/neighbor results,
        and blend risk score. None if blend diagnostics were not computed.
    """
    features: dict[str, float]
    reliable: dict[str, bool]
    candidate_detected: bool
    blend_diagnostics: dict | None = None

    def as_array(self) -> np.ndarray:
        """Return features as a numpy array in FEATURE_NAMES order."""
        return np.array([self.features[k] for k in FEATURE_NAMES], dtype=float)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract(
    time: np.ndarray,
    flux: np.ndarray,
    bls_result: BLSResult,
    config: dict | None = None,
    metadata: dict | None = None,
) -> FeatureResult:
    """
    Compute all 11 features from the cleaned light curve and BLS result.

    Parameters
    ----------
    time : np.ndarray
        Cleaned BTJD timestamps (output of preprocess.clean).
    flux : np.ndarray
        Cleaned normalised flux (output of preprocess.clean).
    bls_result : BLSResult
        Output of bls_detector.detect — used for period, depth, duration, t0.
    config : dict or None
        Optional overrides. Recognised keys:
            phase_bins (int)         — number of phase bins (default 100)
            noise_exclusion_factor   — buffer around transit for noise (default 1.5)
            odd_even_min_transits    — minimum transits for odd/even (default 4)

    Returns
    -------
    FeatureResult
        Container with features dict, reliability flags, and detection status.

    Raises
    ------
    FeatureExtractionError
        If a catastrophic failure prevents any features from being computed.
        Individual feature failures are handled gracefully via fallback values.
    """
    cfg = {
        "phase_bins": _N_PHASE_BINS,
        "noise_exclusion_factor": _NOISE_EXCLUSION_FACTOR,
        "odd_even_min_transits": _MIN_TRANSITS_ODD_EVEN,
        **(config or {}),
    }

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    # ── Core BLS parameters ───────────────────────────────────────────────
    period   = bls_result.best_period   if bls_result.best_period   is not None else 0.0
    t0       = bls_result.best_t0       if bls_result.best_t0       is not None else float(time[0])
    duration = bls_result.best_duration if bls_result.best_duration is not None else 0.0
    depth    = bls_result.best_depth    if bls_result.best_depth    is not None else 0.0

    features  = dict(_FALLBACK)
    reliable  = {k: True for k in FEATURE_NAMES}

    # ── Feature 1: bls_power ─────────────────────────────────────────────
    features["bls_power"] = float(np.clip(bls_result.bls_power_peak, 0.0, 1.0))

    # ── Feature 2: snr ───────────────────────────────────────────────────
    features["snr"] = float(max(0.0, bls_result.snr))

    # ── Resolve aliases (double-period/half-period check) ─────────────────
    from core.alias_resolver import resolve_aliases
    alias_res = resolve_aliases(time, flux, period, t0, duration, depth)
    period = alias_res["resolved_period"]
    t0 = alias_res["resolved_t0"]

    # ── Feature 3: period_days ───────────────────────────────────────────
    features["period_days"] = float(max(0.0, period))

    # ── Feature 4: duration_days ─────────────────────────────────────────
    features["duration_days"] = float(max(0.0, duration))

    # ── Feature 5: depth ─────────────────────────────────────────────────
    features["depth"] = float(max(0.0, depth))

    # ── Feature 6: transit_count ─────────────────────────────────────────
    time_span = float(time[-1] - time[0])
    if period > 0:
        features["transit_count"] = int(np.floor(time_span / period))
    else:
        features["transit_count"] = 0
        reliable["transit_count"] = False

    # ── Phase-fold for shape features (7–11) ─────────────────────────────
    if period > 0 and duration > 0:
        try:
            phase = phase_fold(time, period=period, t0=t0)
            dur_phase = duration / period   # transit duration as fraction of period
            half_dur  = dur_phase / 2.0

            # In-transit mask (phase-folded)
            in_transit_mask  = np.abs(phase) <= half_dur
            # Out-of-transit mask with exclusion buffer
            excl = cfg["noise_exclusion_factor"] * half_dur
            out_transit_mask = np.abs(phase) > excl

            n_in  = int(in_transit_mask.sum())
            n_out = int(out_transit_mask.sum())

        except Exception as exc:
            logger.warning("phase_fold failed: %s — using fallback for features 7-11", exc)
            phase = None
            in_transit_mask = out_transit_mask = np.zeros(len(time), dtype=bool)
            n_in = n_out = 0
    else:
        phase = None
        in_transit_mask = out_transit_mask = np.zeros(len(time), dtype=bool)
        n_in = n_out = 0

    # ── Feature 7: odd_even_depth_delta ──────────────────────────────────
    features["odd_even_depth_delta"], reliable["odd_even_depth_delta"] = \
        _compute_odd_even_delta(time, flux, period, t0, duration,
                                int(features["transit_count"]),
                                int(cfg["odd_even_min_transits"]))

    # ── Feature 8: v_shape_score ─────────────────────────────────────────
    if n_in >= _MIN_TRANSIT_POINTS and phase is not None:
        features["v_shape_score"] = _compute_v_shape_score(
            phase[in_transit_mask], flux[in_transit_mask], depth, half_dur
        )
    else:
        features["v_shape_score"] = 0.0
        reliable["v_shape_score"] = False
        logger.debug("v_shape_score: too few in-transit points (%d)", n_in)

    # ── Feature 9: local_noise ───────────────────────────────────────────
    if n_out >= 10:
        oot_flux = flux[out_transit_mask]
        features["local_noise"] = float(np.std(oot_flux - 1.0, ddof=1))
        if features["local_noise"] <= 0:
            features["local_noise"] = float(np.std(flux, ddof=1))
            reliable["local_noise"] = False
    else:
        features["local_noise"] = float(np.std(flux, ddof=1))
        reliable["local_noise"] = False
        logger.debug("local_noise: too few out-of-transit points (%d)", n_out)

    # ── Feature 10: depth_to_noise_ratio ─────────────────────────────────
    if features["local_noise"] > 0 and depth > 0:
        features["depth_to_noise_ratio"] = float(depth / features["local_noise"])
    else:
        features["depth_to_noise_ratio"] = 0.0
        reliable["depth_to_noise_ratio"] = False

    # ── Feature 11: phase_shape_kurtosis ─────────────────────────────────
    if n_in >= _MIN_TRANSIT_POINTS and phase is not None:
        features["phase_shape_kurtosis"] = _compute_phase_kurtosis(
            phase, flux, in_transit_mask, int(cfg["phase_bins"])
        )
    else:
        features["phase_shape_kurtosis"] = 0.0
        reliable["phase_shape_kurtosis"] = False

    # ── Feature 12: bls_sde ──────────────────────────────────────────────
    from core.detection_significance import calculate_sde
    if hasattr(bls_result, "power") and bls_result.power is not None:
        features["bls_sde"] = calculate_sde(bls_result.power)
    else:
        features["bls_sde"] = 0.0

    # ── Feature 13: secondary_eclipse_depth ──────────────────────────────
    features["secondary_eclipse_depth"] = alias_res["secondary_eclipse_depth"]

    # Update odd_even_depth_delta if alias resolver has a better one
    if reliable["odd_even_depth_delta"] and alias_res["odd_even_delta"] > 0:
        features["odd_even_depth_delta"] = alias_res["odd_even_delta"]

    # ── Features 14-16: blend features ───────────────────────────────────
    from core.blend_features import extract_blend_diagnostics
    meta = metadata or {}
    # Extract centroid arrays from metadata if provided (real TESS data)
    centroid_x = meta.get("centroid_x")
    centroid_y = meta.get("centroid_y")
    quality_arr = meta.get("quality")

    blend_result = extract_blend_diagnostics(
        time=time,
        flux=flux,
        period=period,
        t0=t0,
        duration=duration,
        observed_depth=depth,
        metadata=meta,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        quality=quality_arr,
        transit_features=features,
    )
    blend_diagnostics = blend_result["diagnostics"]
    clf_blend = blend_result["classifier_features"]
    features["centroid_shift"] = clf_blend["centroid_shift"]
    features["crowding_metric"] = clf_blend["crowding_metric"]
    features["gaia_neighbor_count"] = clf_blend["gaia_neighbor_count"]

    if not bls_result.candidate_detected:
        for k in ["odd_even_depth_delta", "v_shape_score", "phase_shape_kurtosis", "secondary_eclipse_depth"]:
            features[k] = _FALLBACK[k]
            reliable[k] = False

    # ── Validate all features are finite ─────────────────────────────────
    features = _ensure_finite(features, reliable)

    logger.info(
        "feature_extractor: depth=%.4f snr=%.1f power=%.4f period=%.4fd "
        "odd_even=%.4f v_shape=%.3f sde=%.2f secondary_depth=%.4f crowding=%.2f",
        features["depth"], features["snr"], features["bls_power"],
        features["period_days"], features["odd_even_depth_delta"],
        features["v_shape_score"], features["bls_sde"],
        features["secondary_eclipse_depth"], features["crowding_metric"],
    )

    return FeatureResult(
        features=features,
        reliable=reliable,
        candidate_detected=bls_result.candidate_detected,
        blend_diagnostics=blend_diagnostics,
    )


# ---------------------------------------------------------------------------
# Feature 7: Odd/even depth delta
# ---------------------------------------------------------------------------

def _compute_odd_even_delta(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    transit_count: int,
    min_transits: int,
) -> tuple[float, bool]:
    """
    Compute |depth_odd - depth_even| by measuring individual transit depths.

    Returns (delta, reliable). reliable=False when transit_count < min_transits.

    The physical interpretation:
        - Planet: all transits identical → delta ≈ 0
        - Eclipsing binary: primary eclipse (odd) and secondary eclipse (even)
          have different depths → delta > 0
    """
    if period <= 0 or duration <= 0 or transit_count < min_transits:
        logger.debug(
            "odd_even_delta: skipped (period=%.3f, count=%d, min=%d)",
            period, transit_count, min_transits,
        )
        return 0.0, False

    # Half-duration in time units
    half_dur_time = duration / 2.0

    odd_depths  = []
    even_depths = []

    for i in range(1, transit_count + 1):
        # Centre of transit i
        tc_i = t0 + (i - 1) * period

        # In-transit points for this event
        dt = np.abs(time - tc_i)
        in_win = dt <= half_dur_time

        if in_win.sum() < 3:
            continue

        transit_flux = flux[in_win]
        # Depth = 1 - median(in-transit flux)
        transit_depth = 1.0 - float(np.median(transit_flux))

        if i % 2 == 1:
            odd_depths.append(transit_depth)
        else:
            even_depths.append(transit_depth)

    if len(odd_depths) < 1 or len(even_depths) < 1:
        return 0.0, False

    delta = abs(float(np.mean(odd_depths)) - float(np.mean(even_depths)))
    logger.debug(
        "odd_even_delta: odd_mean=%.4f even_mean=%.4f delta=%.4f "
        "(n_odd=%d n_even=%d)",
        np.mean(odd_depths), np.mean(even_depths), delta,
        len(odd_depths), len(even_depths),
    )
    return delta, True


# ---------------------------------------------------------------------------
# Feature 8: V-shape score
# ---------------------------------------------------------------------------

def _compute_v_shape_score(
    phase_in: np.ndarray,
    flux_in: np.ndarray,
    depth: float,
    half_dur_phase: float,
) -> float:
    """
    Compute V-shape score in [0, 1].

    Compares fit quality of two models on the in-transit flux:
        Model A (flat box): constant flux = 1.0 - depth
        Model B (V-shape):  flux = 1.0 - depth * (1 - |phase| / half_dur_phase)

    Score = fraction of variance explained by B beyond A, clamped to [0, 1].

    Interpretation:
        0.0 → perfectly flat-bottomed (consistent with planetary transit)
        1.0 → perfectly V-shaped (consistent with grazing EB or stellar companion)
    """
    if len(phase_in) < _MIN_TRANSIT_POINTS or half_dur_phase <= 0 or depth <= 0:
        return 0.0

    # Model A: flat box at transit depth
    model_a = np.full(len(flux_in), 1.0 - depth)

    # Model B: linear V-shape, deepest at phase=0
    abs_phase = np.abs(phase_in)
    model_b = 1.0 - depth * np.clip(1.0 - abs_phase / half_dur_phase, 0.0, 1.0)

    residuals_a = flux_in - model_a
    residuals_b = flux_in - model_b

    ss_a = float(np.sum(residuals_a ** 2))
    ss_b = float(np.sum(residuals_b ** 2))

    if ss_a <= 0:
        return 0.0

    # V-shape score: how much better does model B fit relative to model A?
    # Positive → B fits better (V-shaped); negative → A fits better (flat)
    score = (ss_a - ss_b) / ss_a
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Feature 11: Phase shape kurtosis
# ---------------------------------------------------------------------------

def _compute_phase_kurtosis(
    phase: np.ndarray,
    flux: np.ndarray,
    in_transit_mask: np.ndarray,
    n_bins: int,
) -> float:
    """
    Compute excess kurtosis of the in-transit flux distribution.

    Uses the binned phase-folded profile to reduce noise impact.

    Interpretation:
        Low kurtosis  → flat distribution (flat-bottomed transit, planet-like)
        High kurtosis → peaked/spiky distribution (V-shaped, EB-like)
        ≈ 0           → Gaussian-like (noise)
    """
    if in_transit_mask.sum() < _MIN_TRANSIT_POINTS:
        return 0.0

    try:
        bins = bin_phase_folded(phase, flux, n_bins=n_bins)
        # Identify in-transit bins
        # half_dur is encoded in the mask — re-derive bin centre threshold
        in_tr_phase = phase[in_transit_mask]
        if len(in_tr_phase) == 0:
            return 0.0
        half_dur_approx = float(np.abs(in_tr_phase).max()) * 1.1
        in_bin_mask = np.abs(bins.bin_centres) <= half_dur_approx
        in_bin_flux = bins.bin_means[in_bin_mask & ~np.isnan(bins.bin_means)]

        if len(in_bin_flux) < 4:
            # Fall back to raw in-transit flux
            in_bin_flux = flux[in_transit_mask]
            if len(in_bin_flux) < 4:
                return 0.0

        kurtosis = float(scipy_stats.kurtosis(in_bin_flux, fisher=True, bias=False))
        # Clamp to reasonable range to avoid extreme outliers
        return float(np.clip(kurtosis, -3.0, 10.0))

    except Exception as exc:
        logger.debug("phase_kurtosis failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _ensure_finite(
    features: dict[str, float],
    reliable: dict[str, bool],
) -> dict[str, float]:
    """
    Replace any NaN or inf feature with its fallback value and mark unreliable.

    Guarantees the feature dict never contains NaN or inf.
    """
    cleaned = {}
    for key in FEATURE_NAMES:
        val = features.get(key, _FALLBACK[key])
        if not np.isfinite(float(val)):
            logger.warning(
                "feature '%s' is non-finite (%s) — replacing with fallback %s",
                key, val, _FALLBACK[key],
            )
            cleaned[key] = _FALLBACK[key]
            reliable[key] = False
        else:
            cleaned[key] = val
    return cleaned