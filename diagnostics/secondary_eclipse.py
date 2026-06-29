# secondary_eclipse.py
# --------------------
# Secondary-eclipse detection: circular phase-0.5 check and eccentric scans.

from __future__ import annotations
import logging
import numpy as np
from diagnostics.phase_windows import fold_phase, get_event_mask
from diagnostics.robust_statistics import robust_median, estimate_median_error

logger = logging.getLogger(__name__)

def run_secondary_eclipse_search(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch_btjd: float,
    duration_days: float,
    primary_depth: float,
    config: dict,
) -> dict:
    """
    Performs secondary eclipse search at phase 0.5 and across a grid of eccentric phases.
    """
    sec_config = config.get("SecondaryEclipse", {})
    min_points = sec_config.get("minimum_secondary_points", 5)
    excl_mult = sec_config.get("excluded_primary_transit_window_multiplier", 1.5)
    sig_threshold = sec_config.get("minimum_depth_significance_sigma", 3.0)
    max_planetary_ratio = sec_config.get("maximum_planetary_consistent_secondary_depth_ratio", 0.10)
    
    unavailable = {
        "secondary_available": False,
        "secondary_phase": None,
        "secondary_epoch_btjd": None,
        "secondary_depth": None,
        "secondary_depth_uncertainty": None,
        "secondary_significance": None,
        "secondary_duration_days": None,
        "secondary_primary_depth_ratio": None,
        "secondary_delta_bic": None,
        "secondary_global_p_value": None,
        "secondary_evidence_flag": False,
        "secondary_quality": "unavailable",
    }
    
    if len(time) == 0 or period <= 0 or duration_days <= 0:
        return unavailable
        
    # Phase fold
    phase = fold_phase(time, period, epoch_btjd)
    
    # Define primary transit exclusion window
    half_dur_phase = (duration_days / period) / 2.0
    primary_excl_window = half_dur_phase * excl_mult
    
    # Baseline mask: completely outside primary exclusion window
    oot_mask = np.abs(phase) > primary_excl_window
    if oot_mask.sum() < min_points * 3:
        return unavailable
        
    median_baseline = robust_median(flux[oot_mask])
    baseline_scatter = estimate_median_error(flux[oot_mask]) # Standard error of the median baseline
    
    # ── Search A: Phase 0.5 Check ──
    # Check if a secondary eclipse exists at phase ~0.5
    secondary_05_mask = np.abs(np.abs(phase) - 0.5) <= half_dur_phase
    
    sec_05_flux = flux[secondary_05_mask]
    n_sec_05 = len(sec_05_flux)
    
    best_phase = 0.5
    best_depth = 0.0
    best_err = 0.001
    best_sig = 0.0
    best_dur = duration_days
    
    if n_sec_05 >= min_points:
        sec_05_med = robust_median(sec_05_flux)
        sec_05_depth = float(median_baseline - sec_05_med)
        sec_05_err = estimate_median_error(sec_05_flux)
        
        if sec_05_err > 0:
            sec_05_sig = sec_05_depth / sec_05_err
        else:
            sec_05_sig = 0.0
            
        if sec_05_sig > best_sig:
            best_phase = 0.5
            best_depth = sec_05_depth
            best_err = sec_05_err
            best_sig = sec_05_sig
            
    # ── Search B: Full eccentric phase scan ──
    # Scan from -0.45 to 0.45, avoiding primary exclusion window
    trial_phases = np.linspace(-0.5, 0.5, 50)
    trial_phases = trial_phases[np.abs(trial_phases) > primary_excl_window]
    
    for tp in trial_phases:
        sec_mask = np.abs(phase - tp) <= half_dur_phase
        sec_flux = flux[sec_mask]
        if len(sec_flux) >= min_points:
            sec_med = robust_median(sec_flux)
            sec_depth = float(median_baseline - sec_med)
            sec_err = estimate_median_error(sec_flux)
            sec_sig = sec_depth / sec_err if sec_err > 0 else 0.0
            
            # Record best eccentric secondary
            if sec_sig > best_sig:
                best_phase = float(tp)
                best_depth = sec_depth
                best_err = sec_err
                best_sig = sec_sig
                
    if best_sig <= 0.0:
        return {
            **unavailable,
            "secondary_available": True,
            "secondary_depth": 0.0,
            "secondary_significance": 0.0,
            "secondary_quality": "no_eclipse_detected",
        }
        
    # Calculate delta BIC
    # Model 1 (flat baseline): N * ln(Residual Variance)
    # Model 2 (with secondary): N * ln(Residual Variance with secondary) + 2 parameters
    # Simplify using standard chi-square reduction: delta_bic = chi2_flat - chi2_secondary - ln(N) * delta_params
    # delta_params = 1 (depth of secondary)
    # chi2 difference ≈ best_sig^2 (for single point parameter)
    delta_bic = best_sig**2 - np.log(len(time))
    
    primary_depth = max(1e-6, primary_depth)
    depth_ratio = best_depth / primary_depth
    
    # Evidence flags: secondary detected if significance >= sig_threshold AND delta_bic > 0
    evidence_flag = (best_sig >= sig_threshold) and (delta_bic > 0)
    
    
    # Verify if secondary depth is too large for a planet (depth ratio > max_planetary_ratio)
    quality = "planetary_consistent"
    if evidence_flag:
        if depth_ratio > max_planetary_ratio:
            quality = "eb_like"
        else:
            quality = "planet_candidate_like"
            
    best_epoch = epoch_btjd + best_phase * period
    
    return {
        "secondary_available": True,
        "secondary_phase": round(best_phase, 4),
        "secondary_epoch_btjd": round(best_epoch, 6),
        "secondary_depth": round(max(0.0, best_depth), 6),
        "secondary_depth_uncertainty": round(best_err, 6),
        "secondary_significance": round(best_sig, 4),
        "secondary_duration_days": round(best_dur, 6),
        "secondary_primary_depth_ratio": round(depth_ratio, 4),
        "secondary_delta_bic": round(delta_bic, 2),
        "secondary_global_p_value": round(float(np.exp(-0.5 * max(0.0, delta_bic))), 6),
        "secondary_evidence_flag": bool(evidence_flag),
        "secondary_quality": quality,
    }
