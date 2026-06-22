"""
core/utils.py
-------------
Pure mathematical utility functions shared across ml-core modules.

All functions here are stateless and side-effect-free. They operate on
NumPy arrays and return NumPy arrays or plain Python types.

Used by: preprocess, bls_detector, feature_extractor, plotter
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase folding
# ---------------------------------------------------------------------------

def phase_fold(
    time: np.ndarray,
    period: float,
    t0: float,
) -> np.ndarray:
    """
    Fold a time array onto the range [-0.5, 0.5) centred on transit.

    Parameters
    ----------
    time : np.ndarray
        Monotonically increasing BTJD timestamps.
    period : float
        Orbital period in days. Must be > 0.
    t0 : float
        Reference epoch (transit centre time) in BTJD.

    Returns
    -------
    np.ndarray
        Phase array with values in [-0.5, 0.5).
        Phase 0 corresponds to the transit centre.

    Raises
    ------
    ValueError
        If period <= 0.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    phase = ((time - t0) / period) % 1.0
    # Shift so transit centre is at 0, range becomes [-0.5, 0.5)
    phase[phase >= 0.5] -= 1.0
    return phase


# ---------------------------------------------------------------------------
# Sigma clipping
# ---------------------------------------------------------------------------

def sigma_clip(
    values: np.ndarray,
    sigma_upper: float = 5.0,
    sigma_lower: float = 5.0,
    max_iter: int = 3,
) -> np.ndarray:
    """
    Iterative sigma clipping. Returns a boolean mask of kept points.

    In each iteration, the median and standard deviation are recomputed
    from the currently unclipped points. Points beyond sigma bounds are
    excluded from subsequent iterations.

    Parameters
    ----------
    values : np.ndarray
        1D array of values to clip.
    sigma_upper : float
        Upper bound in units of standard deviation above the median.
    sigma_lower : float
        Lower bound in units of standard deviation below the median.
    max_iter : int
        Maximum number of clipping iterations.

    Returns
    -------
    np.ndarray
        Boolean mask. True = point is kept (within sigma bounds).
    """
    mask = np.ones(len(values), dtype=bool)

    for iteration in range(max_iter):
        clipped = values[mask]
        if len(clipped) == 0:
            logger.warning("sigma_clip: all points clipped — returning original mask")
            break

        median = np.median(clipped)
        std = np.std(clipped, ddof=1) if len(clipped) > 1 else 0.0

        if std == 0.0:
            # All values identical; nothing to clip
            break

        new_mask = (
            (values >= median - sigma_lower * std) &
            (values <= median + sigma_upper * std)
        )

        if np.array_equal(new_mask, mask):
            # Converged — no new points removed
            break

        mask = new_mask
        logger.debug(
            "sigma_clip iteration %d/%d: removed %d points, %d remaining",
            iteration + 1, max_iter,
            np.sum(~mask), np.sum(mask),
        )

    return mask


# ---------------------------------------------------------------------------
# Running median
# ---------------------------------------------------------------------------

def running_median(values: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute a centred running median with edge reflection padding.

    Parameters
    ----------
    values : np.ndarray
        1D array of values.
    window_size : int
        Number of points in the sliding window. If even, rounded up to odd
        so the window is symmetric.

    Returns
    -------
    np.ndarray
        Running median array of the same length as ``values``.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    # Ensure odd window for symmetric centring
    if window_size % 2 == 0:
        window_size += 1

    half = window_size // 2
    n = len(values)

    # Reflect-pad to handle boundaries
    padded = np.pad(values, pad_width=half, mode="reflect")
    result = np.empty(n, dtype=float)

    for i in range(n):
        result[i] = np.median(padded[i: i + window_size])

    return result


# ---------------------------------------------------------------------------
# Phase-folded binning
# ---------------------------------------------------------------------------

class PhaseBins(NamedTuple):
    """Container for binned phase-folded light curve."""
    bin_centres: np.ndarray   # shape (n_bins,)
    bin_means: np.ndarray     # shape (n_bins,)
    bin_stds: np.ndarray      # shape (n_bins,)
    bin_counts: np.ndarray    # shape (n_bins,), int


def bin_phase_folded(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int = 100,
) -> PhaseBins:
    """
    Bin a phase-folded light curve into evenly spaced phase bins.

    Parameters
    ----------
    phase : np.ndarray
        Phase values in [-0.5, 0.5).
    flux : np.ndarray
        Corresponding flux values.
    n_bins : int
        Number of phase bins. Default 100.

    Returns
    -------
    PhaseBins
        Named tuple with bin_centres, bin_means, bin_stds, bin_counts.
        Empty bins (no data) are represented as NaN in bin_means and bin_stds.
    """
    if len(phase) != len(flux):
        raise ValueError(
            f"phase and flux must have equal length, got {len(phase)} vs {len(flux)}"
        )

    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_means = np.full(n_bins, np.nan)
    bin_stds = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    indices = np.digitize(phase, bin_edges) - 1
    # Clip to valid range (edge cases at ±0.5)
    indices = np.clip(indices, 0, n_bins - 1)

    for i in range(n_bins):
        in_bin = flux[indices == i]
        bin_counts[i] = len(in_bin)
        if len(in_bin) > 0:
            bin_means[i] = np.mean(in_bin)
            bin_stds[i] = np.std(in_bin, ddof=1) if len(in_bin) > 1 else 0.0

    return PhaseBins(
        bin_centres=bin_centres,
        bin_means=bin_means,
        bin_stds=bin_stds,
        bin_counts=bin_counts,
    )


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

class Gap(NamedTuple):
    """A detected gap in a time series."""
    start_idx: int    # Index of the last point before the gap
    end_idx: int      # Index of the first point after the gap
    gap_days: float   # Duration of the gap in days


def detect_gaps(
    time: np.ndarray,
    cadence_days: float | None = None,
    threshold_factor: float = 5.0,
) -> list[Gap]:
    """
    Detect gaps in a time series where spacing exceeds threshold_factor × cadence.

    Parameters
    ----------
    time : np.ndarray
        Monotonically increasing time array in days (BTJD).
    cadence_days : float or None
        Expected cadence in days. If None, estimated as the median of
        consecutive differences.
    threshold_factor : float
        A gap is flagged when spacing > threshold_factor × cadence.
        Default 5.0 (gaps larger than 5 cadences).

    Returns
    -------
    list[Gap]
        Sorted list of Gap named tuples. Empty list if no gaps detected.
    """
    if len(time) < 2:
        return []

    diffs = np.diff(time)

    if cadence_days is None:
        cadence_days = float(np.median(diffs))

    if cadence_days <= 0:
        raise ValueError(f"cadence_days must be positive, got {cadence_days}")

    threshold = threshold_factor * cadence_days
    gap_indices = np.where(diffs > threshold)[0]

    gaps = []
    for idx in gap_indices:
        gaps.append(Gap(
            start_idx=int(idx),
            end_idx=int(idx + 1),
            gap_days=float(diffs[idx]),
        ))

    logger.debug("detect_gaps: found %d gap(s) with threshold %.3f days", len(gaps), threshold)
    return gaps


# ---------------------------------------------------------------------------
# Cadence estimation
# ---------------------------------------------------------------------------

def estimate_cadence(time: np.ndarray) -> float:
    """
    Estimate the typical observing cadence from a time array.

    Returns the median of consecutive time differences, ignoring large gaps.

    Parameters
    ----------
    time : np.ndarray
        Monotonically increasing time array.

    Returns
    -------
    float
        Estimated cadence in the same time units as ``time``.
    """
    if len(time) < 2:
        raise ValueError("Need at least 2 points to estimate cadence")

    diffs = np.diff(time)
    # Use only the smallest 90% of differences to exclude gap effects
    threshold = np.percentile(diffs, 90)
    typical_diffs = diffs[diffs <= threshold]

    if len(typical_diffs) == 0:
        return float(np.median(diffs))

    return float(np.median(typical_diffs))


# ---------------------------------------------------------------------------
# Array validation helpers
# ---------------------------------------------------------------------------

def validate_equal_length(time: np.ndarray, flux: np.ndarray) -> None:
    """Raise ValueError if time and flux have different lengths."""
    if len(time) != len(flux):
        raise ValueError(
            f"time and flux must have equal length, got time={len(time)}, flux={len(flux)}"
        )


def validate_monotonic(time: np.ndarray) -> None:
    """Raise ValueError if time is not strictly monotonically increasing."""
    if len(time) < 2:
        return
    diffs = np.diff(time)
    if np.any(diffs <= 0):
        n_bad = int(np.sum(diffs <= 0))
        raise ValueError(
            f"time array must be strictly monotonically increasing; "
            f"found {n_bad} non-increasing step(s)"
        )


def validate_finite(arr: np.ndarray, name: str = "array") -> None:
    """Raise ValueError if array contains any infinite values."""
    n_inf = int(np.sum(np.isinf(arr)))
    if n_inf > 0:
        raise ValueError(f"{name} contains {n_inf} infinite value(s)")