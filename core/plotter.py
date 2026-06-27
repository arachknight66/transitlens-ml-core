"""
core/plotter.py
---------------
Generate all seven diagnostic plots as base64-encoded PNG strings.

Plots produced:
    1. Raw light curve          — unprocessed flux vs. time
    2. Cleaned light curve      — preprocessed flux with transit windows shaded
    3. BLS Periodogram          — BLS power vs. period (log scale)
    4. Phase-folded light curve — flux folded at best period, binned average model overlay,
                                  plus residuals subplot if fit_result is provided.
    5. Transit stack            — cutouts of each transit stacked vertically.
    6. Posterior corner plot    — emcee posterior parameter distributions and correlations.
    7. Period-alias comparison  — folding comparison at P/2, P, and 2P.

All plots use the Matplotlib Agg backend.
Output is base64-encoded PNG suitable for embedding in HTML or JSON APIs.
"""

from __future__ import annotations
import base64
import io
import logging
from typing import Any
import numpy as np

# Force Agg backend before Matplotlib imports
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Colour palette
_C_PRIMARY   = "#4A90D9"   # blue — data points
_C_SECONDARY = "#2ECC71"   # green — transit windows
_C_ACCENT    = "#E74C3C"   # red — best period marker
_C_ALIAS     = "#F39C12"   # orange — alias markers
_C_BINNED    = "#1A1A2E"   # dark navy — binned model line
_C_SCATTER   = "#A0AEC0"   # light grey — phase-folded scatter
_C_SHADE     = "#9B59B6"   # purple — transit window shading
_C_GRID      = "#E2E8F0"   # very light grey — grid
_C_MODEL     = "#1A1A2E"   # dark navy — physical model overlay


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
    fit_result: dict | None = None,
) -> dict[str, str]:
    """
    Generate all diagnostic plots and return as base64-encoded PNGs.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    
    plots = {
        "raw_lightcurve": "",
        "cleaned_lightcurve": "",
        "periodogram": "",
        "phase_folded": "",
        "transit_stack": "",
        "posterior_corner": "",
        "alias_comparison": "",
    }
    
    _apply_style(cfg.get("style", ""))
    
    # Generate standard plots
    plots["raw_lightcurve"] = _safe_run(_plot_raw, time, flux, target_id, cfg)
    plots["cleaned_lightcurve"] = _safe_run(_plot_cleaned, time_clean, flux_clean, bls_result, target_id, cfg)
    plots["periodogram"] = _safe_run(_plot_periodogram, bls_result, target_id, cfg)
    plots["phase_folded"] = _safe_run(_plot_phase_folded, time_clean, flux_clean, bls_result, fit_result, target_id, cfg)
    
    # Generate Phase 7 specific plots
    if fit_result and fit_result.get("fit_status") != "FAILED":
        plots["transit_stack"] = _safe_run(_plot_transit_stack, time_clean, flux_clean, fit_result, target_id, cfg)
        plots["posterior_corner"] = _safe_run(_plot_posterior_corner, fit_result, target_id, cfg)
        plots["alias_comparison"] = _safe_run(_plot_alias_comparison, time_clean, flux_clean, fit_result, target_id, cfg)
        
    return plots


def _safe_run(func, *args) -> str:
    try:
        return func(*args)
    except Exception as exc:
        logger.warning("plotter: %s failed: %s", func.__name__, exc)
        return ""


# ---------------------------------------------------------------------------
# Plot 1: Raw light curve
# ---------------------------------------------------------------------------

def _plot_raw(time: np.ndarray, flux: np.ndarray, target_id: str, cfg: dict) -> str:
    fig, ax = plt.subplots(figsize=(cfg["figure_width"], cfg["figure_height"]))
    t_ds, f_ds = _downsample(time, flux, cfg["downsample_points"])
    
    ax.plot(t_ds, f_ds, color=_C_PRIMARY, linewidth=0.5, alpha=0.8, rasterized=True)
    ax.set_xlabel("Time (BTJD)", fontsize=11)
    ax.set_ylabel("Normalised Flux", fontsize=11)
    ax.set_title(f"Raw Light Curve — {target_id}", fontsize=13, fontweight="bold")
    ax.grid(True, color=_C_GRID, linewidth=0.5)
    return _fig_to_base64(fig, cfg["dpi"])


# ---------------------------------------------------------------------------
# Plot 2: Cleaned light curve
# ---------------------------------------------------------------------------

def _plot_cleaned(time: np.ndarray, flux: np.ndarray, bls_result: BLSResult, target_id: str, cfg: dict) -> str:
    fig, ax = plt.subplots(figsize=(cfg["figure_width"], cfg["figure_height"]))
    ax.plot(time, flux, color=_C_PRIMARY, linewidth=0.5, alpha=0.7, label="Cleaned flux")
    
    period = bls_result.best_period
    t0 = bls_result.best_t0
    duration = bls_result.best_duration
    
    if period and t0 and duration:
        # Mark predicted transit centers
        t_span = time[-1] - time[0]
        n_transits = int(t_span / period) + 2
        start_cycle = int((time[0] - t0) / period) - 1
        
        for i in range(start_cycle, start_cycle + n_transits):
            t_center = t0 + i * period
            if time[0] - duration <= t_center <= time[-1] + duration:
                # Plot dashed marker lines
                ax.axvline(t_center, color=_C_SECONDARY, linestyle="--", linewidth=1.0, alpha=0.6)
                
        # Label once
        ax.axvline(np.nan, color=_C_SECONDARY, linestyle="--", linewidth=1.0, label="Transit centers")
        
    ax.set_xlabel("Time (BTJD)", fontsize=11)
    ax.set_ylabel("Normalised Flux", fontsize=11)
    ax.set_title(f"Cleaned Light Curve — {target_id}", fontsize=13, fontweight="bold")
    ax.grid(True, color=_C_GRID, linewidth=0.5)
    ax.legend(fontsize=9, loc="upper right")
    return _fig_to_base64(fig, cfg["dpi"])


# ---------------------------------------------------------------------------
# Plot 3: BLS Periodogram
# ---------------------------------------------------------------------------

def _plot_periodogram(bls_result: BLSResult, target_id: str, cfg: dict) -> str:
    fig, ax = plt.subplots(figsize=(cfg["figure_width"], cfg["figure_height"]))
    
    ax.plot(bls_result.periods, bls_result.power, color=_C_PRIMARY, linewidth=1.0)
    
    if bls_result.best_period:
        ax.axvline(
            bls_result.best_period,
            color=_C_ACCENT, linestyle="--", linewidth=1.5,
            label=f"Peak Period: {bls_result.best_period:.4f} days",
        )
        
    ax.set_xscale("log")
    ax.set_xlabel("Period (days)", fontsize=11)
    ax.set_ylabel("BLS Power", fontsize=11)
    
    title_suffix = f" — Peak: {bls_result.best_period:.4f} days" if bls_result.best_period else ""
    ax.set_title(f"BLS Periodogram{title_suffix}", fontsize=13, fontweight="bold")
    ax.grid(True, color=_C_GRID, linewidth=0.5)
    ax.legend(fontsize=9, loc="upper right")
    return _fig_to_base64(fig, cfg["dpi"])


# ---------------------------------------------------------------------------
# Plot 4: Phase-folded light curve (Enhanced with residuals panel)
# ---------------------------------------------------------------------------

def _plot_phase_folded(
    time_clean: np.ndarray,
    flux_clean: np.ndarray,
    bls_result: BLSResult,
    fit_result: dict | None,
    target_id: str,
    cfg: dict,
) -> str:
    """
    Phase-folded light curve. If fit_result is available, it includes a
    residual panel at the bottom and best-fit physical model overlay.
    """
    period = fit_result.get("period_days", bls_result.best_period) if fit_result else bls_result.best_period
    t0 = fit_result.get("epoch_btjd", bls_result.best_t0) if fit_result else bls_result.best_t0
    duration = fit_result.get("duration_days", bls_result.best_duration) if fit_result else bls_result.best_duration
    depth = fit_result.get("depth", bls_result.best_depth) if fit_result else bls_result.best_depth
    
    if period is None or period <= 0:
        fig, ax = plt.subplots(figsize=(cfg["figure_width"], cfg["figure_height"]))
        ax.text(0.5, 0.5, "No period available", transform=ax.transAxes, ha="center")
        return _fig_to_base64(fig, cfg["dpi"])
        
    phase = phase_fold(time_clean, period, t0)
    
    # Check if we should draw the residuals panel
    has_fit = fit_result is not None and "best_model" in fit_result and len(fit_result["best_model"]) == len(flux_clean)
    
    if has_fit:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            figsize=(cfg["figure_width"], cfg["figure_height"] + 2)
        )
    else:
        fig, ax_top = plt.subplots(figsize=(cfg["figure_width"], cfg["figure_height"]))
        ax_bot = None
        
    # ── 1. Top Panel: Scatter + Binned Average + Physical Model ──
    # Raw scatter points
    ax_top.scatter(phase, flux_clean, s=2, c=_C_SCATTER, alpha=0.3, rasterized=True, zorder=1, label="Raw points")
    
    # Binned average
    try:
        bin_centres, bin_means, _ = bin_phase_folded(phase, flux_clean, cfg.get("phase_bins", 100))
        ax_top.plot(bin_centres, bin_means, color=_C_BINNED, linewidth=2.0, zorder=3, label="Binned average")
    except Exception:
        pass
        
    # Physical model overlay
    if has_fit:
        sort_idx = np.argsort(phase)
        phase_sorted = phase[sort_idx]
        model_sorted = fit_result["best_model"][sort_idx]
        ax_top.plot(phase_sorted, model_sorted, color=_C_ACCENT, linewidth=2.0, zorder=4, label="Best physical fit")
        
    # Shade transit window
    if duration:
        half_phase = (duration / period) / 2.0
        ax_top.axvspan(-half_phase, half_phase, color=_C_SHADE, alpha=cfg["transit_shade_alpha"], zorder=0, label="Transit window")
        
    # Header titles
    err_str = f" \u00b1 {fit_result.get('period_uncertainty_days', 0.0):.6f}" if fit_result and fit_result.get('period_uncertainty_days') else ""
    title_str = f"Phase-folded at P = {period:.5f}{err_str} days — Depth: {depth*100:.3f}%"
    ax_top.set_title(title_str, fontsize=12, fontweight="bold")
    ax_top.set_ylabel("Normalised Flux", fontsize=11)
    ax_top.legend(fontsize=9, loc="lower right", framealpha=0.7)
    ax_top.grid(True, color=_C_GRID, linewidth=0.5)
    ax_top.set_xlim(-0.5, 0.5)
    
    # ── 2. Bottom Panel: Residuals ──
    if has_fit and ax_bot is not None:
        residuals = fit_result["residuals"]
        ax_bot.scatter(phase, residuals, s=1, c=_C_SCATTER, alpha=0.3, rasterized=True)
        # Residuals binned average
        try:
            bin_c, bin_m, _ = bin_phase_folded(phase, residuals, cfg.get("phase_bins", 100))
            ax_bot.plot(bin_c, bin_m, color=_C_BINNED, linewidth=1.5)
        except Exception:
            pass
            
        ax_bot.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
        ax_bot.set_ylabel("Residuals", fontsize=11)
        ax_bot.set_xlabel("Orbital Phase", fontsize=11)
        ax_bot.grid(True, color=_C_GRID, linewidth=0.5)
        
    else:
        ax_top.set_xlabel("Orbital Phase", fontsize=11)
        
    return _fig_to_base64(fig, cfg["dpi"])


# ---------------------------------------------------------------------------
# Plot 5: Individual Transit Stack (Phase 7 Panel 3)
# ---------------------------------------------------------------------------

def _plot_transit_stack(time_clean: np.ndarray, flux_clean: np.ndarray, fit_result: dict, target_id: str, cfg: dict) -> str:
    """
    Displays individual transit events vertically stacked to inspect cycle-to-cycle variation.
    """
    period = fit_result["period_days"]
    t0 = fit_result["epoch_btjd"]
    duration = fit_result["duration_days"]
    
    cycles = np.round((time_clean - t0) / period)
    unique_cycles = np.unique(cycles)
    
    fig, ax = plt.subplots(figsize=(cfg["figure_width"], cfg["figure_height"] + 1))
    
    plotted = 0
    # Stack up to 6 events maximum
    for cycle in unique_cycles:
        if plotted >= 6:
            break
            
        t_center = t0 + cycle * period
        mask = np.abs(time_clean - t_center) <= (2.0 * duration)
        if np.sum(mask) < 6:
            continue
            
        t_sub = (time_clean[mask] - t_center) * 24.0 # in hours
        f_sub = flux_clean[mask]
        
        # Apply vertical shift offset to stack transits
        offset = -0.006 * plotted
        ax.plot(t_sub, f_sub + offset, label=f"Cycle {int(cycle)}", alpha=0.8, linewidth=1.2)
        plotted += 1
        
    ax.set_xlabel("Time from mid-transit (hours)", fontsize=11)
    ax.set_ylabel("Normalized Offset Flux", fontsize=11)
    ax.set_title(f"Individual Transit Event Stack — {target_id}", fontsize=12, fontweight="bold")
    ax.grid(True, color=_C_GRID, linewidth=0.5)
    if plotted > 0:
        ax.legend(fontsize=9, loc="lower right", framealpha=0.7)
        
    return _fig_to_base64(fig, cfg["dpi"])


# ---------------------------------------------------------------------------
# Plot 6: Posterior Parameter Corner Plot (Phase 7 Panel 4)
# ---------------------------------------------------------------------------

def _plot_posterior_corner(fit_result: dict, target_id: str, cfg: dict) -> str:
    """
    Draws a parameter covariance/marginal correlation plot from MCMC samples using 'corner'.
    """
    import corner
    
    # Retrieve samples from fit result (check standard format)
    # samples shape expected: (N_samples, 5) or similar
    samples = fit_result.get("flat_samples")
    if samples is None:
        return ""
        
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[0] < 20:
        return ""
        
    # Slice the first 5 physical parameters: period, t0, rp_rstar, a_rstar, b
    plot_samples = samples[:, :5]
    labels = ["P (d)", "T0 (BTJD)", "Rp/R*", "a/R*", "b"]
    
    fig = corner.corner(
        plot_samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".5f" if np.max(plot_samples[:, 0]) < 1.0 else ".4f",
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 9},
    )
    fig.suptitle(f"MCMC Posterior Distribution — {target_id}", fontsize=12, fontweight="bold", y=1.02)
    return _fig_to_base64(fig, cfg["dpi"])


# ---------------------------------------------------------------------------
# Plot 7: Period-Alias fold comparisons (Phase 7 Panel 5)
# ---------------------------------------------------------------------------

def _plot_alias_comparison(time_clean: np.ndarray, flux_clean: np.ndarray, fit_result: dict, target_id: str, cfg: dict) -> str:
    """
    Folded comparisons at P/2, P, and 2P to diagnose period grid selection error modes.
    """
    period = fit_result["period_days"]
    t0 = fit_result["epoch_btjd"]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    
    periods = [period / 2.0, period, period * 2.0]
    titles = [
        f"Half Period\n(P/2 = {period/2.0:.4f} d)",
        f"Preferred Period\n(P = {period:.4f} d)",
        f"Double Period\n(2P = {period*2.0:.4f} d)"
    ]
    
    for idx, (p, title) in enumerate(zip(periods, titles)):
        ax = axes[idx]
        phase = phase_fold(time_clean, p, t0)
        ax.scatter(phase, flux_clean, s=1, c=_C_SCATTER, alpha=0.3, rasterized=True)
        
        # Overlay binned average line
        try:
            bin_centres, bin_means, _ = bin_phase_folded(phase, flux_clean, 60)
            ax.plot(bin_centres, bin_means, color=_C_BINNED, linewidth=1.5, zorder=3)
        except Exception:
            pass
            
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Phase", fontsize=9)
        ax.set_xlim(-0.5, 0.5)
        ax.grid(True, color=_C_GRID, linewidth=0.5)
        
    axes[0].set_ylabel("Normalised Flux", fontsize=11)
    fig.suptitle(f"Period-Alias Folding Diagnostic Grid — {target_id}", fontsize=12, fontweight="bold", y=1.02)
    return _fig_to_base64(fig, cfg["dpi"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n <= max_points:
        return x, y
    step = max(1, n // max_points)
    return x[::step], y[::step]


def _fig_to_base64(fig: plt.Figure, dpi: int = 100) -> str:
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return encoded
    finally:
        plt.close(fig)


def _apply_style(style_name: str) -> None:
    if not style_name:
        return
    try:
        plt.style.use(style_name)
    except OSError:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            pass
