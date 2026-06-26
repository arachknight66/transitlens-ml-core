"""
core/bls_detector.py
--------------------
Box Least Squares (BLS) transit detection for cleaned light curves.

Searches for the most significant periodic box-shaped dip in a cleaned
(time, flux) pair and returns the best-fit period, duration, depth,
phase, power spectrum, and detection significance.

Detection requires BOTH conditions:
    - BLS power peak > bls_power_threshold
    - SNR (depth / local_noise) > snr_threshold

Primary implementation: astropy.timeseries.BoxLeastSquares (when available)
Fallback implementation: vectorised NumPy/SciPy BLS (always available)

Used by: pipeline.py
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.exceptions import BLSDetectionError, NoCandidateFoundError
from core.utils import phase_fold, bin_phase_folded, estimate_cadence

logger = logging.getLogger(__name__)

# ── Try to import astropy BLS ───────────────────────────────────────────────
try:
    from astropy.timeseries import BoxLeastSquares as _AstropyBLS
    _ASTROPY_AVAILABLE = True
    logger.debug("bls_detector: astropy BoxLeastSquares available")
except ImportError:
    _AstropyBLS = None
    _ASTROPY_AVAILABLE = False
    logger.info(
        "bls_detector: astropy not available — using scipy fallback BLS. "
        "Install astropy for faster, more accurate period recovery."
    )


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "period_min_days": 0.5,
    "period_max_days": None,        # None → time_span / 2
    "n_oversample": 10,             # frequency grid oversampling factor
    "n_durations": 5,               # number of trial durations per period
    "duration_min_days": 0.01,      # ~15 minutes
    "duration_max_fraction": 0.5,   # max duration = fraction × period
    "bls_power_threshold": 0.15,    # minimum normalised BLS power for detection
    "snr_threshold": 7.0,           # minimum multi-point SNR for detection
    "min_depth_snr": 0.5,           # minimum depth/global_flux_rms guard (FP suppressor)
    "alias_check_tolerance": 0.20,  # power within this fraction of peak → flag alias
    "alias_promote_threshold": 0.90, # promote 2xP if its power >= this fraction of peak (alias correction)
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BLSResult:
    """
    Complete output of the BLS detection stage.

    Attributes
    ----------
    candidate_detected : bool
        True only if bls_power_peak > threshold AND snr > threshold.
    best_period : float or None
        Period at the BLS power peak (days). None if not detected.
    best_t0 : float or None
        Transit centre epoch at best period (BTJD). None if not detected.
    best_duration : float or None
        Transit duration at best period (days). None if not detected.
    best_depth : float or None
        Transit depth at best period (fractional flux drop). None if not detected.
    bls_power_peak : float
        BLS power at the best period (always set, even for sub-threshold).
    snr : float
        Signal-to-noise ratio depth / local_noise (always set).
    periods : np.ndarray
        Period grid used in the search (days).
    power : np.ndarray
        BLS power at each period in the grid.
    alias_warning : bool
        True if a harmonic alias (×2 or ÷2) has comparable power.
    backend : str
        'astropy' or 'scipy' — which implementation was used.
    detection_reason : str
        Human-readable reason for detection/non-detection outcome.
    """
    candidate_detected: bool
    best_period: Optional[float]
    best_t0: Optional[float]
    best_duration: Optional[float]
    best_depth: Optional[float]
    bls_power_peak: float
    snr: float
    periods: np.ndarray
    power: np.ndarray
    alias_warning: bool
    backend: str
    detection_reason: str
    top_periods: np.ndarray = field(default_factory=lambda: np.array([]))
    top_powers: np.ndarray = field(default_factory=lambda: np.array([]))
    top_t0s: np.ndarray = field(default_factory=lambda: np.array([]))
    top_durations: np.ndarray = field(default_factory=lambda: np.array([]))
    top_depths: np.ndarray = field(default_factory=lambda: np.array([]))
    selected_period_before_alias_correction: Optional[float] = None
    selected_period_after_alias_correction: Optional[float] = None
    alias_type: str = "none"
    alias_corrected: bool = False
    detection_sde: float = 0.0
    local_noise: float = 0.0
    false_alarm_proxy: float = 1.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect(
    time: np.ndarray,
    flux: np.ndarray,
    config: dict | None = None,
) -> BLSResult:
    """
    Run BLS transit search on a cleaned, normalised light curve.

    Parameters
    ----------
    time : np.ndarray
        Cleaned BTJD timestamps (monotonically increasing, no NaN).
    flux : np.ndarray
        Cleaned normalised flux (median ≈ 1.0, no NaN/inf).
    config : dict or None
        Optional overrides for BLS parameters. Keys correspond to
        DEFAULT_CONFIG entries.

    Returns
    -------
    BLSResult
        Detection result with power spectrum and best-fit parameters.
        candidate_detected=False is a valid, expected result for noise cases.

    Raises
    ------
    BLSDetectionError
        If the BLS algorithm fails due to a numerical issue. Does NOT
        raise NoCandidateFoundError — that is returned as candidate_detected=False.
    """
    t0_wall = _time.perf_counter()
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    # ── Build period grid ─────────────────────────────────────────────────
    time_span = float(time[-1] - time[0])
    period_min = float(cfg["period_min_days"])
    period_max = float(cfg["period_max_days"] or time_span / 2.0)
    period_max = min(period_max, time_span / 2.0)   # enforce physical limit

    if period_min >= period_max:
        raise BLSDetectionError(
            f"Invalid period range: period_min={period_min:.3f} >= period_max={period_max:.3f}",
            details={"period_min": period_min, "period_max": period_max},
        )

    logger.debug(
        "bls_detector: time_span=%.2f days, period range [%.2f, %.2f] days",
        time_span, period_min, period_max,
    )

    # ── Dispatch to backend ───────────────────────────────────────────────
    try:
        if _ASTROPY_AVAILABLE:
            periods, power, best_params = _run_astropy_bls(time, flux, cfg, period_min, period_max, normalize=False)
            backend = "astropy"
        else:
            periods, power, best_params = _run_scipy_bls(time, flux, cfg, period_min, period_max, normalize=False)
            backend = "scipy"
    except BLSDetectionError:
        raise
    except Exception as exc:
        raise BLSDetectionError(
            f"BLS computation failed: {exc}",
            details={"backend": "astropy" if _ASTROPY_AVAILABLE else "scipy"},
        ) from exc

    # -- Extract peak and run local fine search --------------------------------
    peak_idx = int(np.argmax(power))
    coarse_best_period = float(periods[peak_idx])

    # Run local fine-resolution BLS around coarse_best_period
    fine_periods1 = np.linspace(coarse_best_period * 0.98, coarse_best_period * 1.02, 100)
    if _ASTROPY_AVAILABLE:
        _, fine_power1, fine_params1 = _run_astropy_bls(time, flux, cfg, period_min, period_max, periods_grid=fine_periods1, normalize=False)
    else:
        _, fine_power1, fine_params1 = _run_scipy_bls(time, flux, cfg, period_min, period_max, periods_grid=fine_periods1, normalize=False)

    idx_f1 = int(np.argmax(fine_power1))
    best_period = float(fine_periods1[idx_f1])
    bls_power_peak_raw = float(fine_power1[idx_f1])
    best_t0 = float(fine_params1["t0"][idx_f1])
    best_duration = float(fine_params1["duration"][idx_f1])
    best_depth = float(fine_params1["depth"][idx_f1])

    # -- Half-period alias promotion -------------------------------------------
    # BLS preferentially recovers P/2 when only a few (<=3) transits are visible
    # (e.g. P=7d or P=12d in a 27-day TESS sector). Check whether doubling the
    # period gives comparable power, and if so promote 2×P as the true period.
    time_span = float(time[-1] - time[0]) if len(time) > 1 else 27.0
    n_transits_at_best = time_span / best_period if best_period > 0 else 999.0
    double_period = best_period * 2.0
    if (n_transits_at_best <= 10.0) and (double_period <= time_span / 1.5):
        # Run local fine-resolution BLS around double_period
        fine_periods2 = np.linspace(double_period * 0.98, double_period * 1.02, 100)
        if _ASTROPY_AVAILABLE:
            _, fine_power2, fine_params2 = _run_astropy_bls(time, flux, cfg, period_min, period_max, periods_grid=fine_periods2, normalize=False)
        else:
            _, fine_power2, fine_params2 = _run_scipy_bls(time, flux, cfg, period_min, period_max, periods_grid=fine_periods2, normalize=False)

        idx_f2 = int(np.argmax(fine_power2))
        power_2p_raw = float(fine_power2[idx_f2])
        alias_promote_thresh = cfg.get("alias_promote_threshold", 0.40)
        if power_2p_raw >= bls_power_peak_raw * alias_promote_thresh:
            best_period = float(fine_periods2[idx_f2])
            best_t0 = float(fine_params2["t0"][idx_f2])
            best_duration = float(fine_params2["duration"][idx_f2])
            best_depth = float(fine_params2["depth"][idx_f2])
            bls_power_peak_raw = power_2p_raw

    # -- Normalize power peak and power spectrum ------------------------------
    global_max_power = float(np.max(power)) if len(power) > 0 else 1.0
    if global_max_power > 0:
        power = power / global_max_power
        bls_power_peak = bls_power_peak_raw / global_max_power
    else:
        bls_power_peak = 0.0

    # -- Compute SNR and local_noise -------------------------------------------
    snr, local_noise = _compute_snr(time, flux, best_period, best_t0, best_duration, best_depth)

    # -- Alias check -----------------------------------------------------------
    alias_warning = _check_aliases(periods, power, best_period, bls_power_peak, cfg)

    # -- Detection decision ----------------------------------------------------
    # Two independent gates must ALL pass:
    #   1. BLS power peak above threshold (spectral significance)
    #   2. Multi-point SNR above threshold (stacked transit depth)
    # Additionally compute depth_snr using the GLOBAL flux RMS (period-independent)
    # as a diagnostic and optional third gate. Using local_noise at the BLS period
    # would be unreliable when BLS recovers the wrong period.
    global_noise = float(np.std(flux)) if len(flux) > 1 else 1.0
    power_ok = bls_power_peak >= cfg["bls_power_threshold"]
    snr_ok = snr >= cfg["snr_threshold"]
    depth_snr = (best_depth / global_noise) if global_noise > 0 else 0.0
    dur_fraction = (best_duration / best_period) if best_period > 0 else 1.0
    depth_ok = (depth_snr >= cfg.get("min_depth_snr", 0.5)) or (
        snr >= 11.0 and dur_fraction <= 0.35 and depth_snr >= 0.15
    )
    candidate_detected = power_ok and snr_ok and depth_ok

    if candidate_detected:
        detection_reason = (
            f"Detected: BLS power={bls_power_peak:.4f} (threshold={cfg['bls_power_threshold']:.3f}), "
            f"SNR={snr:.2f} (threshold={cfg['snr_threshold']:.1f}), "
            f"depth_snr={depth_snr:.2f} (threshold={cfg.get('min_single_point_depth_snr', 1.5):.1f})"
        )
        logger.info(
            "bls_detector: candidate detected -- period=%.4f days, depth=%.4f, "
            "duration=%.4f days, power=%.4f, SNR=%.2f, depth_snr=%.2f",
            best_period, best_depth, best_duration, bls_power_peak, snr, depth_snr,
        )
    elif not power_ok:
        detection_reason = (
            f"Not detected: BLS power={bls_power_peak:.4f} < threshold={cfg['bls_power_threshold']:.3f} "
            f"(SNR={snr:.2f}, but power spectrum shows no dominant peak -- consistent with noise)."
        )
        logger.info("bls_detector: sub-threshold power=%.4f, SNR=%.2f", bls_power_peak, snr)
    elif not snr_ok:
        detection_reason = (
            f"Not detected: SNR={snr:.2f} < threshold={cfg['snr_threshold']:.1f} "
            f"(BLS power={bls_power_peak:.4f} would pass but stacked SNR is sub-threshold)."
        )
        logger.info("bls_detector: sub-threshold SNR=%.2f, power=%.4f", snr, bls_power_peak)
    else:
        detection_reason = (
            f"Not detected: depth_snr={depth_snr:.2f} < {cfg.get('min_depth_snr', 1.5):.1f} "
            f"(BLS power={bls_power_peak:.4f}, SNR={snr:.2f} pass, but single-point depth is sub-noise)."
        )
        logger.info(
            "bls_detector: sub-noise depth -- depth_snr=%.2f, SNR=%.2f, power=%.4f",
            depth_snr, snr, bls_power_peak,
        )

    elapsed_ms = (_time.perf_counter() - t0_wall) * 1000
    logger.debug("bls_detector: completed in %.0f ms (backend=%s)", elapsed_ms, backend)

    return BLSResult(
        candidate_detected=candidate_detected,
        best_period=best_period if candidate_detected else best_period,   # always set for sub-threshold plotting
        best_t0=best_t0 if candidate_detected else best_t0,
        best_duration=best_duration if candidate_detected else best_duration,
        best_depth=best_depth if candidate_detected else best_depth,
        bls_power_peak=bls_power_peak,
        snr=snr,
        periods=periods,
        power=power,
        alias_warning=alias_warning,
        backend=backend,
        detection_reason=detection_reason,
    )


# ---------------------------------------------------------------------------
# Astropy BLS backend
# ---------------------------------------------------------------------------

def _run_astropy_bls(
    time: np.ndarray,
    flux: np.ndarray,
    cfg: dict,
    period_min: float,
    period_max: float,
    periods_grid: np.ndarray | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Run BLS using astropy.timeseries.BoxLeastSquares.

    Returns (periods, power, best_params) where best_params is a dict of
    arrays keyed by 't0', 'duration', 'depth', one value per period grid point.
    """
    time_span = float(time[-1] - time[0])
    n_oversample = int(cfg["n_oversample"])

    # Build frequency grid
    if periods_grid is not None:
        periods = np.asarray(periods_grid, dtype=float)
    else:
        freq_min = 1.0 / period_max
        freq_max = 1.0 / period_min
        df = 1.0 / (n_oversample * time_span)
        n_freqs = max(1, int(np.ceil((freq_max - freq_min) / df)))
        freqs = np.linspace(freq_min, freq_max, n_freqs)
        periods = 1.0 / freqs

    # Build duration grid: log-spaced between duration_min and duration_max_fraction × period_min
    n_dur = int(cfg["n_durations"])
    dur_min = float(cfg["duration_min_days"])
    dur_max = float(cfg["duration_max_fraction"]) * period_min
    durations = np.geomspace(dur_min, max(dur_max, dur_min * 2), n_dur)

    bls = _AstropyBLS(time, y=flux)
    result = bls.power(periods, durations)

    # result.power is the BLS power array (one value per period)
    power = np.asarray(result.power, dtype=float)

    # Normalise power to [0, 1] range — astropy returns unnormalised SR statistic
    if normalize:
        if power.max() > 0:
            power_norm = power / power.max()
        else:
            power_norm = power
    else:
        power_norm = power

    # Extract best parameters at each period
    t0_arr = np.asarray(result.transit_time, dtype=float)
    duration_arr = np.asarray(result.duration, dtype=float)
    depth_arr = np.asarray(result.depth, dtype=float)
    depth_arr = np.clip(depth_arr, 0.0, None)   # depth must be non-negative

    best_params = {
        "t0": t0_arr,
        "duration": duration_arr,
        "depth": depth_arr,
    }

    return periods, power_norm, best_params


# ---------------------------------------------------------------------------
# SciPy fallback BLS backend
# ---------------------------------------------------------------------------

def _run_scipy_bls(
    time: np.ndarray,
    flux: np.ndarray,
    cfg: dict,
    period_min: float,
    period_max: float,
    periods_grid: np.ndarray | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Pure NumPy/SciPy BLS implementation — used when astropy is unavailable.

    Algorithm:
        For each period P in the grid:
            1. Phase-fold the light curve to get phase array φ
            2. For each trial duration q (as fraction of period):
                a. Identify in-transit points: |φ| < q/2
                b. Compute s = Σ(flux_in - 1) × w_in  (weighted depth sum)
                c. Compute r = Σ(w_in)                 (total in-transit weight)
                d. BLS statistic: SR = s² / (r × (1 - r))   [Kovacs et al. 2002]
            3. Record the best SR and its parameters over all durations

    Power is then normalised by dividing by the maximum across the entire grid.

    This is O(N × M_periods × N_durations) where N is the number of data points.
    Vectorised over data points using NumPy broadcasting; loop only over periods.
    """
    n_points = len(time)
    time_span = float(time[-1] - time[0])
    n_oversample = int(cfg["n_oversample"])

    # ── Period grid (frequency-uniform) ───────────────────────────────────
    if periods_grid is not None:
        periods_grid = np.asarray(periods_grid, dtype=float)
        n_freqs = len(periods_grid)
    else:
        freq_min = 1.0 / period_max
        freq_max = 1.0 / period_min
        df = 1.0 / (n_oversample * time_span)
        n_freqs = max(1, int(np.ceil((freq_max - freq_min) / df)))
        # Limit to a reasonable size for compute time
        n_freqs = min(n_freqs, 20_000)
        freqs = np.linspace(freq_min, freq_max, n_freqs)
        periods_grid = 1.0 / freqs

    # ── Duration grid ─────────────────────────────────────────────────────
    n_dur = int(cfg["n_durations"])
    dur_min_frac = float(cfg["duration_min_days"]) / period_min
    dur_max_frac = float(cfg["duration_max_fraction"])
    # Duration as fraction of period (q = duration / period)
    q_grid = np.geomspace(
        max(dur_min_frac, 0.001),
        min(dur_max_frac, 0.45),
        n_dur,
    )

    # ── Pre-compute weights ───────────────────────────────────────────────
    # Unit weights (equal cadence). Using inverse-variance weights would
    # require per-point noise estimates; uniform weights are correct for
    # TESS 2-min cadence data with approximately uniform noise.
    weights = np.ones(n_points, dtype=float) / n_points

    # Flux deviations from baseline (1.0)
    delta = 1.0 - flux   # positive for transit dips (flux < 1)

    # ── Power spectrum arrays ─────────────────────────────────────────────
    power_arr = np.zeros(n_freqs, dtype=float)
    t0_arr = np.zeros(n_freqs, dtype=float)
    duration_arr = np.zeros(n_freqs, dtype=float)
    depth_arr = np.zeros(n_freqs, dtype=float)

    t0_ref = float(time[0])   # reference epoch for phase folding

    for i, period in enumerate(periods_grid):
        # Phase-fold: φ ∈ [0, 1)
        phase = ((time - t0_ref) / period) % 1.0

        best_sr = -np.inf
        best_q = q_grid[0]
        best_phi0 = 0.0

        for q in q_grid:
            half_q = q / 2.0

            # Slide the transit window across all phases using vectorised ops.
            # For each possible transit centre φ₀, in-transit condition is:
            #   (phase - φ₀) mod 1 < q  (wrapping)
            # We test a grid of φ₀ values equal to the data phases (optimum sampling).
            # This is a standard BLS approach: test all N data phases as centres.

            # For each data point as potential transit centre:
            #   shifted_phase[j,k] = (phase[k] - phase[j]) % 1.0
            # But full N×N is expensive. Instead use vectorised approach over
            # a coarser φ₀ grid of n_phi0 = max(50, n_points//100) values.
            n_phi0 = max(50, min(200, n_points // 50))
            phi0_grid = np.linspace(0, 1.0 - q, n_phi0, endpoint=False)

            # Vectorised: shape (n_phi0, n_points)
            shifted = (phase[np.newaxis, :] - phi0_grid[:, np.newaxis]) % 1.0
            in_transit = shifted < q   # shape (n_phi0, n_points)

            # Weighted sums: s = sum(w*delta) for in-transit; r = sum(w) for in-transit
            s = (in_transit * weights[np.newaxis, :] * delta[np.newaxis, :]).sum(axis=1)
            r = (in_transit * weights[np.newaxis, :]).sum(axis=1)

            # BLS statistic: SR = s^2 / (r * (1-r)), valid when 0 < r < 1
            valid = (r > 0.001) & (r < 0.999)
            sr = np.where(valid, s ** 2 / (r * (1.0 - r)), 0.0)

            best_idx_q = int(np.argmax(sr))
            if sr[best_idx_q] > best_sr:
                best_sr = float(sr[best_idx_q])
                best_q = q
                best_phi0 = float(phi0_grid[best_idx_q])

        power_arr[i] = best_sr
        # Convert best_phi0 (phase fraction) back to time
        t0_arr[i] = t0_ref + (best_phi0 + best_q / 2.0) * period
        duration_arr[i] = best_q * period

        # Compute depth at best parameters: mean in-transit delta
        phase_at_best = (phase - best_phi0) % 1.0
        in_transit_best = phase_at_best < best_q
        if in_transit_best.sum() > 0:
            depth_arr[i] = max(0.0, float(np.mean(delta[in_transit_best])))
        else:
            depth_arr[i] = 0.0

    # -- Normalise power -------------------------------------------------------
    if normalize:
        max_power = power_arr.max()
        if max_power > 0:
            power_norm = power_arr / max_power
        else:
            power_norm = power_arr
    else:
        power_norm = power_arr

    best_params = {
        "t0": t0_arr,
        "duration": duration_arr,
        "depth": depth_arr,
    }

    return periods_grid, power_norm, best_params


# ---------------------------------------------------------------------------
# SNR computation
# ---------------------------------------------------------------------------

def _compute_snr(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    depth: float,
) -> tuple[float, float]:
    """
    Compute signal-to-noise ratio of the transit detection.
    SNR = depth / (local_noise / sqrt(n_in_transit))

    local_noise is the RMS of the out-of-transit flux in the phase-folded
    light curve. Using out-of-transit noise (rather than global noise)
    avoids the transit signal inflating the noise estimate.

    Parameters
    ----------
    time, flux : np.ndarray
        Cleaned light curve.
    period, t0, duration, depth : float
        BLS best-fit parameters.

    Returns
    -------
    tuple[float, float]
        (snr, local_noise) where snr = depth/local_noise * sqrt(n_in_transit).
        Returns (0.0, 0.0) if computation fails or out-of-transit points are insufficient.
    """
    if period <= 0 or duration <= 0 or depth <= 0:
        return 0.0, 0.0

    try:
        phase = phase_fold(time, period=period, t0=t0)
        duration_phase = duration / period
        # Exclude transit and a buffer zone (1.5x duration) from noise estimate
        exclusion = 1.5 * duration_phase / 2.0
        out_of_transit = np.abs(phase) > exclusion

        if out_of_transit.sum() < 10:
            logger.warning("_compute_snr: too few out-of-transit points (%d)", out_of_transit.sum())
            return 0.0, 0.0

        local_noise = float(np.std(flux[out_of_transit], ddof=1))

        if local_noise <= 0:
            return 0.0, 0.0

        # Calculate number of points in transit
        in_transit = np.abs(phase) <= (duration_phase / 2.0)
        n_in_transit = int(np.sum(in_transit))
        if n_in_transit <= 0:
            n_in_transit = 1

        snr = float((depth / local_noise) * np.sqrt(n_in_transit))
        return snr, local_noise

    except Exception as exc:
        logger.warning("_compute_snr failed: %s", exc)
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Alias check
# ---------------------------------------------------------------------------

def _check_aliases(
    periods: np.ndarray,
    power: np.ndarray,
    best_period: float,
    peak_power: float,
    cfg: dict,
) -> bool:
    """
    Check whether harmonic aliases (×2 or ÷2 of best_period) have
    comparable power to the primary peak, indicating possible aliasing.

    Returns True if an alias is detected (warning should be noted).
    """
    if peak_power <= 0:
        return False

    tolerance = float(cfg["alias_check_tolerance"])
    threshold = peak_power * (1.0 - tolerance)

    alias_periods = [best_period * 2.0, best_period / 2.0]

    for alias_p in alias_periods:
        if alias_p < periods[0] or alias_p > periods[-1]:
            continue

        # Find the power at the alias period (nearest grid point)
        idx = int(np.argmin(np.abs(periods - alias_p)))
        alias_power = float(power[idx])

        # Check in a small window (±5 grid points) for a local peak
        window = slice(max(0, idx - 5), min(len(power), idx + 5))
        local_max = float(power[window].max())

        if local_max >= threshold:
            logger.info(
                "_check_aliases: alias at %.4f days has power %.4f (primary %.4f, threshold %.4f)",
                alias_p, local_max, peak_power, threshold,
            )
            return True

    return False