"""
core/plotter.py
---------------
Generate all four diagnostic plots as base64-encoded PNG strings.

Plots produced:
    1. Raw light curve          — unprocessed flux vs. time
    2. Cleaned light curve      — preprocessed flux with transit windows shaded
    3. BLS Periodogram          — BLS power vs. period (log scale)
    4. Phase-folded light curve — flux folded at best period, binned model overlay

All plots use the Matplotlib Agg backend (headless, no GUI required).
Output is base64-encoded PNG suitable for embedding in HTML or JSON APIs.

Used by: pipeline.py
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np

# Force Agg backend before any other matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from core.bls_detector import BLSResult
from core.exceptions import PlottingError
from core.utils import phase_fold, bin_phase_folded

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "dpi": 100,
    "figure_width": 10,
    "figure_height": 4,
    "downsample_points": 2000,
    "phase_bins": 100,
    "transit_shade_alpha": 0.15,
    "style": "seaborn-v0_8-whitegrid",
}

# Colour palette — modern, professional, accessible
_C_PRIMARY   = "#4A90D9"   # blue — data points
_C_SECONDARY = "#2ECC71"   # green — transit windows
_C_ACCENT    = "#E74C3C"   # red — best period marker
_C_ALIAS     = "#F39C12"   # orange — alias markers
_C_BINNED    = "#1A1A2E"   # dark navy — binned model line
_C_SCATTER   = "#A0AEC0"   # light grey — phase-folded scatter
_C_SHADE     = "#9B59B6"   # purple — transit window shading
_C_GRID      = "#E2E8F0"   # very light grey — grid


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all(
    time: np.ndarray,
    flux: np.ndarray,
    time_clean: np.ndarray,
    flux_clean: np.ndarray,
    bls_result: BLSResult,
    target_id: str = "unknown",
    config: dict | None = None,
) -> dict[str, str]:
    """
    Generate all four diagnostic plots and return as base64-encoded PNGs.

    Parameters
    ----------
    time, flux : np.ndarray
        Raw (unprocessed) light curve arrays.
    time_clean, flux_clean : np.ndarray
        Preprocessed light curve arrays.
    bls_result : BLSResult
        Complete BLS detection output.
    target_id : str
        Identifier for plot titles.
    config : dict or None
        Override plotting parameters (see DEFAULT_CONFIG).

    Returns
    -------
    dict[str, str]
        Keys: raw_lightcurve, cleaned_lightcurve, periodogram, phase_folded.
        Values: base64-encoded PNG strings (empty string on failure).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    plots = {
        "raw_lightcurve": "",
        "cleaned_lightcurve": "",
        "periodogram": "",
        "phase_folded": "",
    }

    # Apply style safely
    _apply_style(cfg.get("style", ""))

    # Generate each plot independently so one failure doesn't block others
    plot_funcs = [
        ("raw_lightcurve",     _plot_raw,           (time, flux, target_id, cfg)),
        ("cleaned_lightcurve", _plot_cleaned,        (time_clean, flux_clean, bls_result, target_id, cfg)),
        ("periodogram",        _plot_periodogram,    (bls_result, target_id, cfg)),
        ("phase_folded",       _plot_phase_folded,   (time_clean, flux_clean, bls_result, target_id, cfg)),
    ]

    for key, func, args in plot_funcs:
        try:
            plots[key] = func(*args)
        except Exception as exc:
            logger.warning("plotter: %s failed: %s", key, exc)
            plots[key] = ""

    non_empty = sum(1 for v in plots.values() if v)
    logger.info("plotter: generated %d/4 plots for target '%s'", non_empty, target_id)
    return plots


# ---------------------------------------------------------------------------
# Plot 1: Raw light curve
# ---------------------------------------------------------------------------

def _plot_raw(
    time: np.ndarray,
    flux: np.ndarray,
    target_id: str,
    cfg: dict,
) -> str:
    """Raw flux vs. time — thin blue line, downsampled for speed."""
    fig, ax = plt.subplots(
        figsize=(cfg["figure_width"], cfg["figure_height"]),
    )

    t_ds, f_ds = _downsample(time, flux, cfg["downsample_points"])

    ax.plot(t_ds, f_ds, color=_C_PRIMARY, linewidth=0.5, alpha=0.8, rasterized=True)
    ax.set_xlabel("Time (BTJD)", fontsize=11)
    ax.set_ylabel("Normalised Flux", fontsize=11)
    ax.set_title(f"Raw Light Curve — {target_id}", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=9)
    ax.grid(True, color=_C_GRID, linewidth=0.5)

    return _fig_to_base64(fig, cfg["dpi"])


# ---------------------------------------------------------------------------
# Plot 2: Cleaned light curve with transit windows
# ---------------------------------------------------------------------------

def _plot_cleaned(
    time_clean: np.ndarray,
    flux_clean: np.ndarray,
    bls_result: BLSResult,
    target_id: str,
    cfg: dict,
) -> str:
    """Cleaned flux with shaded transit windows at the detected period."""
    fig, ax = plt.subplots(
        figsize=(cfg["figure_width"], cfg["figure_height"]),
    )

    t_ds, f_ds = _downsample(time_clean, flux_clean, cfg["downsample_points"])
    ax.plot(t_ds, f_ds, color=_C_PRIMARY, linewidth=0.5, alpha=0.8, rasterized=True)

    # Shade in-transit windows if a candidate was detected
    if bls_result.candidate_detected and bls_result.best_period and bls_result.best_t0:
        period   = bls_result.best_period
        t0       = bls_result.best_t0
        duration = bls_result.best_duration or 0.1
        half_dur = duration / 2.0

        # Compute transit centre times within the observation window
        t_min, t_max = time_clean.min(), time_clean.max()
        n_start = int(np.floor((t_min - t0) / period))
        n_end   = int(np.ceil((t_max - t0) / period))

        for n in range(n_start, n_end + 1):
            tc = t0 + n * period
            left  = tc - half_dur
            right = tc + half_dur
            if right >= t_min and left <= t_max:
                ax.axvspan(
                    left, right,
                    color=_C_SECONDARY, alpha=cfg["transit_shade_alpha"],
                    label="Transit window" if n == n_start else None,
                )

    ax.set_xlabel("Time (BTJD)", fontsize=11)
    ax.set_ylabel("Normalised Flux", fontsize=11)
    ax.set_title(f"Cleaned Light Curve — {target_id}", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=9)
    ax.grid(True, color=_C_GRID, linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9, loc="lower right", framealpha=0.7)

    return _fig_to_base64(fig, cfg["dpi"])


# ---------------------------------------------------------------------------
# Plot 3: BLS Periodogram
# ---------------------------------------------------------------------------

def _plot_periodogram(
    bls_result: BLSResult,
    target_id: str,
    cfg: dict,
) -> str:
    """BLS power spectrum with best-period marker and optional alias lines."""
    fig, ax = plt.subplots(
        figsize=(cfg["figure_width"], cfg["figure_height"]),
    )

    periods = bls_result.periods
    power   = bls_result.power

    # Full power spectrum — grey line
    ax.plot(periods, power, color="#718096", linewidth=0.8, alpha=0.9, rasterized=True)

    # Best period — vertical red dashed line
    if bls_result.best_period is not None:
        bp = bls_result.best_period
        ax.axvline(
            bp, color=_C_ACCENT, linestyle="--", linewidth=1.5, alpha=0.9,
            label=f"Best period: {bp:.4f} d",
        )

        # Alias harmonics if flagged
        if bls_result.alias_warning:
            for factor, lbl in [(2.0, "×2 harmonic"), (0.5, "÷2 harmonic")]:
                alias_p = bp * factor
                if periods.min() <= alias_p <= periods.max():
                    ax.axvline(
                        alias_p, color=_C_ALIAS, linestyle=":", linewidth=1.2, alpha=0.8,
                        label=f"{lbl}: {alias_p:.4f} d",
                    )

    ax.set_xscale("log")
    ax.set_xlabel("Period (days)", fontsize=11)
    ax.set_ylabel("BLS Power", fontsize=11)

    title_suffix = ""
    if bls_result.best_period is not None:
        title_suffix = f" — Best Period: {bls_result.best_period:.4f} days"
    ax.set_title(
        f"BLS Periodogram{title_suffix}",
        fontsize=13, fontweight="bold",
    )
    ax.tick_params(labelsize=9)
    ax.grid(True, color=_C_GRID, linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9, loc="upper right", framealpha=0.7)

    return _fig_to_base64(fig, cfg["dpi"])


# ---------------------------------------------------------------------------
# Plot 4: Phase-folded light curve
# ---------------------------------------------------------------------------

def _plot_phase_folded(
    time_clean: np.ndarray,
    flux_clean: np.ndarray,
    bls_result: BLSResult,
    target_id: str,
    cfg: dict,
) -> str:
    """
    Phase-folded flux at the best period.

    Shows raw scatter (grey dots) and a binned average model (dark line).
    Transit window is shaded in purple.
    """
    fig, ax = plt.subplots(
        figsize=(cfg["figure_width"], cfg["figure_height"]),
    )

    # Use best period, or sub-threshold peak for noise case
    period = bls_result.best_period
    t0     = bls_result.best_t0

    if period is None or period <= 0:
        # For noise case with no valid period, use the period at max power
        if len(bls_result.periods) > 0 and len(bls_result.power) > 0:
            idx = int(np.argmax(bls_result.power))
            period = float(bls_result.periods[idx])
            t0 = float(time_clean[0])
        else:
            # Cannot phase-fold without a period
            ax.text(
                0.5, 0.5, "No period available for phase folding",
                transform=ax.transAxes, ha="center", va="center", fontsize=12,
            )
            ax.set_title("Phase-Folded Light Curve", fontsize=13, fontweight="bold")
            return _fig_to_base64(fig, cfg["dpi"])

    if t0 is None:
        t0 = float(time_clean[0])

    # Phase-fold
    phase = phase_fold(time_clean, period, t0)

    # Raw scatter — light grey
    ax.scatter(
        phase, flux_clean,
        s=1, c=_C_SCATTER, alpha=0.4, rasterized=True, zorder=1,
    )

    # Binned model — dark navy line
    n_bins = cfg.get("phase_bins", 100)
    try:
        bin_centres, bin_means, bin_stds = bin_phase_folded(phase, flux_clean, n_bins)
        ax.plot(
            bin_centres, bin_means,
            color=_C_BINNED, linewidth=2.0, zorder=3, label="Binned average",
        )
    except Exception:
        logger.debug("plotter: phase binning failed — showing scatter only")

    # Transit window shading
    duration = bls_result.best_duration
    if duration and duration > 0:
        half_phase = (duration / period) / 2.0
        ax.axvspan(
            -half_phase, half_phase,
            color=_C_SHADE, alpha=cfg["transit_shade_alpha"],
            zorder=0, label="Transit window",
        )

    depth = bls_result.best_depth
    title_parts = [f"Phase-folded at P = {period:.4f} days"]
    if depth is not None:
        title_parts.append(f"Depth = {depth:.4f}")
    ax.set_title(", ".join(title_parts), fontsize=13, fontweight="bold")

    ax.set_xlabel("Orbital Phase", fontsize=11)
    ax.set_ylabel("Normalised Flux", fontsize=11)
    ax.set_xlim(-0.5, 0.5)
    ax.tick_params(labelsize=9)
    ax.grid(True, color=_C_GRID, linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9, loc="lower right", framealpha=0.7)

    return _fig_to_base64(fig, cfg["dpi"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _downsample(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample arrays to at most max_points for fast plotting."""
    n = len(x)
    if n <= max_points:
        return x, y
    step = max(1, n // max_points)
    return x[::step], y[::step]


def _fig_to_base64(fig: plt.Figure, dpi: int = 100) -> str:
    """Render a matplotlib Figure to a base64-encoded PNG string."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return encoded
    finally:
        plt.close(fig)


def _apply_style(style_name: str) -> None:
    """Apply a matplotlib style, falling back to default if unavailable."""
    if not style_name:
        return
    try:
        plt.style.use(style_name)
    except OSError:
        logger.debug("plotter: style '%s' not available — using default", style_name)
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            pass
