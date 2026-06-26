"""
core/preprocess.py
------------------
Preprocessing and signal cleaning for raw normalised light curves.

Transforms a raw (time, flux) pair into an analysis-ready pair by
applying, in strict order:

    1. Input validation
    2. NaN removal
    3. Outlier removal (iterative sigma clipping)
    4. Trend removal (detrending)
    5. Re-normalisation
    6. Gap detection
    7. Minimum data quality gates

The output flux array has:
    - No NaN or infinite values
    - Median ≈ 1.0
    - Outliers removed
    - Low-frequency instrumental trends divided out
    - Metadata about detected gaps

Used by: pipeline.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from core.exceptions import (
    InvalidInputError,
    InsufficientDataError,
    PreprocessingError,
)
from core.utils import (
    detect_gaps,
    estimate_cadence,
    running_median,
    sigma_clip,
    validate_equal_length,
    validate_finite,
    validate_monotonic,
    Gap,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    """
    Container for the output of the preprocessing stage.

    Attributes
    ----------
    time : np.ndarray
        Cleaned, monotonically increasing time array (BTJD).
    flux : np.ndarray
        Cleaned, normalised flux array (median ≈ 1.0).
    n_original : int
        Number of points in the raw input.
    n_after_nan : int
        Points remaining after NaN removal.
    n_after_clip : int
        Points remaining after sigma clipping.
    fraction_retained : float
        Fraction of original points in the cleaned output.
    gaps : list[Gap]
        Detected gaps in the cleaned time series.
    cadence_days : float
        Estimated observing cadence in days.
    time_span_days : float
        Total time span of the cleaned light curve.
    detrend_method : str
        Method used for detrending ("running_median" or "polynomial").
    """
    time: np.ndarray
    flux: np.ndarray
    n_original: int
    n_after_nan: int
    n_after_clip: int
    fraction_retained: float
    gaps: list[Gap]
    cadence_days: float
    time_span_days: float
    detrend_method: str

    @property
    def n_points(self) -> int:
        return len(self.time)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Upper sigma is tight to remove stellar flares and cosmic rays (upward spikes).
    # Lower sigma is loose so that transit dips (which are legitimate downward signals)
    # are NOT removed. A 1.3% transit is 13σ below the median for noise level 0.001;
    # setting sigma_lower=50 ensures no real transit is clipped by this stage.
    # Genuine downward outliers (single-point cosmic-ray dips) are far deeper (>50%)
    # and will still be clipped.
    "sigma_upper": 5.0,
    "sigma_lower": 50.0,
    "max_sigma_iter": 3,
    "detrend_method": "running_median",      # "running_median" | "polynomial"
    "detrend_window_days": 1.5,
    "detrend_poly_degree": 2,
    "gap_threshold_factor": 5.0,
    "min_points": 500,
    "min_time_span_days": 5.0,
    "min_fraction_retained": 0.80,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def clean(
    time: np.ndarray,
    flux: np.ndarray,
    config: dict | None = None,
) -> PreprocessResult:
    """
    Full preprocessing pipeline: validate → clean → detrend → normalise.

    Parameters
    ----------
    time : np.ndarray
        Raw BTJD timestamps, expected to be monotonically increasing.
    flux : np.ndarray
        Raw normalised flux values (median ≈ 1.0 expected).
    config : dict or None
        Optional override for preprocessing parameters. Any key present
        overrides the corresponding entry in DEFAULT_CONFIG.

    Returns
    -------
    PreprocessResult
        Cleaned arrays plus provenance metadata.

    Raises
    ------
    InvalidInputError
        If time/flux have mismatched length, time is non-monotonic,
        or flux contains infinities.
    InsufficientDataError
        If the data does not pass minimum quality gates.
    PreprocessingError
        If a preprocessing step fails for an unexpected reason.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    n_original = len(time)

    # ------------------------------------------------------------------
    # Step 1: Input validation
    # ------------------------------------------------------------------
    _validate_inputs(time, flux)

    # ------------------------------------------------------------------
    # Step 2: NaN removal
    # ------------------------------------------------------------------
    time, flux = _remove_nans(time, flux)
    n_after_nan = len(time)

    logger.debug(
        "preprocess: NaN removal — %d → %d points (%d NaN removed)",
        n_original, n_after_nan, n_original - n_after_nan,
    )

    # ------------------------------------------------------------------
    # Step 3: Outlier removal (sigma clipping)
    # ------------------------------------------------------------------
    time, flux = _sigma_clip_flux(
        time, flux,
        sigma_upper=cfg["sigma_upper"],
        sigma_lower=cfg["sigma_lower"],
        max_iter=cfg["max_sigma_iter"],
    )
    n_after_clip = len(time)

    logger.debug(
        "preprocess: sigma clipping — %d → %d points (%d outliers removed)",
        n_after_nan, n_after_clip, n_after_nan - n_after_clip,
    )

    # ------------------------------------------------------------------
    # Step 4: Trend removal (detrending)
    # ------------------------------------------------------------------
    try:
        flux = _detrend(
            time, flux,
            method=cfg["detrend_method"],
            window_days=cfg["detrend_window_days"],
            poly_degree=cfg["detrend_poly_degree"],
        )
    except Exception as exc:
        raise PreprocessingError(
            f"Detrending failed ({cfg['detrend_method']}): {exc}",
            details={"method": cfg["detrend_method"]},
        ) from exc

    # ------------------------------------------------------------------
    # Step 5: Re-normalisation
    # ------------------------------------------------------------------
    flux = _renormalise(flux)

    # ------------------------------------------------------------------
    # Step 6: Gap detection
    # ------------------------------------------------------------------
    cadence_days = estimate_cadence(time)
    gaps = detect_gaps(
        time,
        cadence_days=cadence_days,
        threshold_factor=cfg["gap_threshold_factor"],
    )

    time_span_days = float(time[-1] - time[0]) if len(time) >= 2 else 0.0
    fraction_retained = n_after_clip / n_original if n_original > 0 else 0.0

    logger.info(
        "preprocess: %d points, span=%.1f days, cadence=%.4f days, gaps=%d, "
        "fraction_retained=%.3f",
        len(time), time_span_days, cadence_days, len(gaps), fraction_retained,
    )

    # ------------------------------------------------------------------
    # Step 7: Minimum data quality gates
    # ------------------------------------------------------------------
    _check_quality_gates(
        n_points=len(time),
        time_span_days=time_span_days,
        fraction_retained=fraction_retained,
        cfg=cfg,
    )

    return PreprocessResult(
        time=time,
        flux=flux,
        n_original=n_original,
        n_after_nan=n_after_nan,
        n_after_clip=n_after_clip,
        fraction_retained=fraction_retained,
        gaps=gaps,
        cadence_days=cadence_days,
        time_span_days=time_span_days,
        detrend_method=cfg["detrend_method"],
    )


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def _validate_inputs(time: np.ndarray, flux: np.ndarray) -> None:
    """Raise InvalidInputError for any fundamental array problem."""
    if len(time) == 0 or len(flux) == 0:
        raise InvalidInputError(
            "Input arrays are empty.",
            details={"len_time": len(time), "len_flux": len(flux)},
        )

    try:
        validate_equal_length(time, flux)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc

    try:
        validate_monotonic(time)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc

    try:
        validate_finite(flux, name="flux")
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc

    # time may contain NaN but not inf
    n_time_inf = int(np.sum(np.isinf(time)))
    if n_time_inf > 0:
        raise InvalidInputError(
            f"time array contains {n_time_inf} infinite value(s)."
        )


def _remove_nans(
    time: np.ndarray,
    flux: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove all entries where time or flux is NaN (keeps arrays synchronised)."""
    valid = np.isfinite(time) & np.isfinite(flux)
    return time[valid], flux[valid]


def _sigma_clip_flux(
    time: np.ndarray,
    flux: np.ndarray,
    sigma_upper: float,
    sigma_lower: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply iterative sigma clipping to the flux array."""
    mask = sigma_clip(
        flux,
        sigma_upper=sigma_upper,
        sigma_lower=sigma_lower,
        max_iter=max_iter,
    )
    return time[mask], flux[mask]


def _detrend(
    time: np.ndarray,
    flux: np.ndarray,
    method: str,
    window_days: float,
    poly_degree: int,
) -> np.ndarray:
    """
    Remove low-frequency trends from the flux array.

    Parameters
    ----------
    time : np.ndarray
        Cleaned time array.
    flux : np.ndarray
        Cleaned flux array (after sigma clipping).
    method : str
        "running_median" — divide by running median (default, more robust).
        "polynomial"     — divide by best-fit polynomial.
    window_days : float
        Window size for running median detrending in days.
    poly_degree : int
        Polynomial degree for polynomial detrending.

    Returns
    -------
    np.ndarray
        Detrended flux array.
    """
    if method == "running_median":
        return _detrend_running_median(time, flux, window_days)
    elif method == "polynomial":
        return _detrend_polynomial(time, flux, poly_degree)
    elif method == "savgol":
        return _detrend_savgol(time, flux, window_days)
    elif method == "spline":
        return _detrend_spline(time, flux, window_days)
    else:
        raise ValueError(
            f"Unknown detrend method '{method}'. "
            "Expected 'running_median', 'polynomial', 'savgol', or 'spline'."
        )


def _detrend_savgol(
    time: np.ndarray,
    flux: np.ndarray,
    window_days: float,
) -> np.ndarray:
    """Divide flux by a Savitzky-Golay filtered baseline."""
    from scipy.signal import savgol_filter
    cadence = estimate_cadence(time)
    window_points = max(3, int(round(window_days / cadence)))
    if window_points % 2 == 0:
        window_points += 1
    if window_points >= len(flux):
        window_points = len(flux) - 1
        if window_points % 2 == 0:
            window_points -= 1
        if window_points < 3:
            return flux.copy()
    baseline = savgol_filter(flux, window_length=window_points, polyorder=2)
    baseline = np.where(np.abs(baseline) < 1e-10, 1.0, baseline)
    return flux / baseline


def _detrend_spline(
    time: np.ndarray,
    flux: np.ndarray,
    window_days: float,
) -> np.ndarray:
    """Divide flux by a UnivariateSpline baseline fit."""
    from scipy.interpolate import UnivariateSpline
    # Determine degree and smoothing based on window_days
    spl = UnivariateSpline(time, flux, k=3)
    # smoothing factor s balances fitting vs smoothness
    spl.set_smoothing_factor(len(time) * 0.0001)
    baseline = spl(time)
    baseline = np.where(np.abs(baseline) < 1e-10, 1.0, baseline)
    return flux / baseline


def _detrend_running_median(
    time: np.ndarray,
    flux: np.ndarray,
    window_days: float,
) -> np.ndarray:
    """
    Divide flux by a running median baseline to remove slow trends.

    The window size in number of points is determined from window_days
    and the estimated cadence. A minimum window of 3 is enforced.
    """
    cadence = estimate_cadence(time)
    window_points = max(3, int(round(window_days / cadence)))
    # Ensure odd window
    if window_points % 2 == 0:
        window_points += 1

    logger.debug(
        "detrend_running_median: window=%.2f days → %d points",
        window_days, window_points,
    )

    baseline = running_median(flux, window_size=window_points)

    # Guard against zero or near-zero baseline
    baseline = np.where(np.abs(baseline) < 1e-10, 1.0, baseline)

    detrended = flux / baseline

    # Verify detrending didn't corrupt the array
    if not np.all(np.isfinite(detrended)):
        n_bad = int(np.sum(~np.isfinite(detrended)))
        logger.warning(
            "detrend_running_median: %d non-finite values after detrending; "
            "replacing with 1.0",
            n_bad,
        )
        detrended = np.where(np.isfinite(detrended), detrended, 1.0)

    return detrended


def _detrend_polynomial(
    time: np.ndarray,
    flux: np.ndarray,
    poly_degree: int,
) -> np.ndarray:
    """
    Divide flux by a polynomial baseline fit to remove slow trends.

    Time is normalised to [-1, 1] before fitting to improve numerical
    conditioning of the polynomial fit.
    """
    # Normalise time to [-1, 1] for numerical stability
    t_min, t_max = time[0], time[-1]
    if t_max == t_min:
        logger.warning("detrend_polynomial: zero time span; skipping detrending")
        return flux.copy()

    t_norm = 2.0 * (time - t_min) / (t_max - t_min) - 1.0

    # Fit polynomial to the full flux (not just out-of-transit,
    # since we don't know transit locations yet)
    coeffs = np.polyfit(t_norm, flux, deg=poly_degree)
    baseline = np.polyval(coeffs, t_norm)

    # Guard against zero or near-zero baseline
    baseline = np.where(np.abs(baseline) < 1e-10, 1.0, baseline)

    detrended = flux / baseline

    logger.debug(
        "detrend_polynomial: degree=%d, baseline range=[%.4f, %.4f]",
        poly_degree, baseline.min(), baseline.max(),
    )

    return detrended


def _renormalise(flux: np.ndarray) -> np.ndarray:
    """
    Divide flux by its median so the baseline is exactly 1.0.

    This corrects for any small baseline offset introduced by detrending.
    """
    median = np.median(flux)
    if median == 0.0:
        logger.warning(
            "renormalise: median flux is exactly 0.0; skipping normalisation"
        )
        return flux.copy()

    normalised = flux / median

    achieved_median = np.median(normalised)
    if abs(achieved_median - 1.0) > 1e-3:
        logger.warning(
            "renormalise: post-normalisation median is %.6f (expected 1.0 ± 0.001)",
            achieved_median,
        )

    return normalised


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

def _check_quality_gates(
    n_points: int,
    time_span_days: float,
    fraction_retained: float,
    cfg: dict,
) -> None:
    """
    Raise InsufficientDataError if any minimum quality criterion is not met.

    Checks (in order):
        1. Minimum number of data points
        2. Minimum time span
        3. Minimum fraction of points retained after sigma clipping
    """
    min_points = cfg["min_points"]
    if n_points < min_points:
        raise InsufficientDataError(
            f"Insufficient data: only {n_points} points after cleaning "
            f"(minimum required: {min_points}). "
            "Cannot run BLS on so few points.",
            details={"n_points": n_points, "min_points": min_points},
        )

    min_span = cfg["min_time_span_days"]
    if time_span_days < min_span:
        raise InsufficientDataError(
            f"Insufficient time baseline: light curve spans only {time_span_days:.2f} days "
            f"(minimum required: {min_span} days). "
            "Need at least 2 transit events to measure a period.",
            details={"time_span_days": time_span_days, "min_time_span_days": min_span},
        )

    min_fraction = cfg["min_fraction_retained"]
    if fraction_retained < min_fraction:
        raise InsufficientDataError(
            f"Too many outliers: only {fraction_retained:.1%} of points retained "
            f"after sigma clipping (minimum required: {min_fraction:.1%}). "
            "The light curve is too noisy to analyse reliably.",
            details={
                "fraction_retained": fraction_retained,
                "min_fraction_retained": min_fraction,
            },
        )

    logger.debug(
        "quality gates passed: n_points=%d, time_span=%.1f days, fraction_retained=%.3f",
        n_points, time_span_days, fraction_retained,
    )