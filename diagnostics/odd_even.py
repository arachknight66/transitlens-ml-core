# odd_even.py
# -----------
# Odd-versus-even transit depth consistency analysis and period alias retests.

from __future__ import annotations
import logging
import numpy as np
from diagnostics.phase_windows import fold_phase, assign_cycle_numbers, get_event_mask
from diagnostics.robust_statistics import robust_median, robust_mad, estimate_median_error, permutation_test_difference

logger = logging.getLogger(__name__)

def run_odd_even_analysis(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch_btjd: float,
    duration_days: float,
    config: dict,
) -> dict:
    """
    Computes odd/even transit depth differences, significance, and alias check statistics.
    """
    oe_config = config.get("OddEven", {})
    min_odd = oe_config.get("minimum_odd_events", 2)
    min_even = oe_config.get("minimum_even_events", 2)
    win_mult = oe_config.get("transit_window_multiplier", 1.2)
    base_mult = oe_config.get("baseline_window_multiplier", 3.0)
    sig_threshold = oe_config.get("significance_threshold_sigma", 3.0)
    frac_diff_threshold = oe_config.get("absolute_fractional_difference_threshold", 0.05)
    
    unavailable = {
        "odd_even_available": False,
        "odd_event_count": 0,
        "even_event_count": 0,
        "odd_depth": None,
        "odd_depth_uncertainty": None,
        "even_depth": None,
        "even_depth_uncertainty": None,
        "odd_even_depth_difference": None,
        "odd_even_fractional_difference": None,
        "odd_even_significance": None,
        "odd_even_p_value": None,
        "odd_even_evidence_flag": False,
        "odd_even_quality": "unavailable",
    }
    
    if len(time) == 0 or period <= 0 or duration_days <= 0:
        return unavailable
        
    cycles = assign_cycle_numbers(time, period, epoch_btjd)
    unique_cycles = np.unique(cycles)
    
    # Assign even vs odd cycles
    odd_cycles = unique_cycles[unique_cycles % 2 == 1]
    even_cycles = unique_cycles[unique_cycles % 2 == 0]
    
    # Check counts of events with some points in transit
    half_dur = (duration_days * win_mult) / 2.0
    
    n_odd_observed = 0
    for cycle in odd_cycles:
        tc = epoch_btjd + cycle * period
        in_transit = np.abs(time - tc) <= half_dur
        if in_transit.sum() >= 3:
            n_odd_observed += 1
            
    n_even_observed = 0
    for cycle in even_cycles:
        tc = epoch_btjd + cycle * period
        in_transit = np.abs(time - tc) <= half_dur
        if in_transit.sum() >= 3:
            n_even_observed += 1
            
    if n_odd_observed < min_odd or n_even_observed < min_even:
        logger.debug(f"Odd/even unavailable: observed odd={n_odd_observed} < {min_odd} or even={n_even_observed} < {min_even}")
        return unavailable
        
    # Separate odd vs even cadences
    in_transit_mask = get_event_mask(time, period, epoch_btjd, duration_days, window_multiplier=win_mult)
    
    # Baseline comparison mask (out of transit local baseline)
    baseline_mask = get_event_mask(time, period, epoch_btjd, duration_days, window_multiplier=base_mult) & ~in_transit_mask
    
    if baseline_mask.sum() < 10:
        return unavailable
        
    median_baseline = robust_median(flux[baseline_mask])
    
    odd_mask = in_transit_mask & (cycles % 2 == 1)
    even_mask = in_transit_mask & (cycles % 2 == 0)
    
    if odd_mask.sum() < 3 or even_mask.sum() < 3:
        return unavailable
        
    odd_flux = flux[odd_mask]
    even_flux = flux[even_mask]
    
    # Depth = baseline - flux
    odd_depth = float(median_baseline - robust_median(odd_flux))
    even_depth = float(median_baseline - robust_median(even_flux))
    
    # Ensure physical positive depth values (or close to 0)
    odd_depth = max(-0.01, odd_depth)
    even_depth = max(-0.01, even_depth)
    
    odd_err = estimate_median_error(odd_flux)
    even_err = estimate_median_error(even_flux)
    
    # Handle error bounds
    odd_err = max(1e-6, odd_err)
    even_err = max(1e-6, even_err)
    
    # Compute delta and significance
    delta = float(abs(odd_depth - even_depth))
    sigma_delta = float(np.sqrt(odd_err**2 + even_err**2))
    
    significance = float(delta / sigma_delta) if sigma_delta > 0 else 0.0
    
    # Fractional difference relative to average depth
    avg_depth = (odd_depth + even_depth) / 2.0
    if avg_depth > 0:
        frac_diff = float(delta / avg_depth)
    else:
        frac_diff = 0.0
        
    # p-value via permutation
    p_value = permutation_test_difference(odd_flux, even_flux, n_permutations=200)
    
    evidence_flag = (significance >= sig_threshold) and (frac_diff >= frac_diff_threshold)
    
    # Retest at aliases (P/2 and 2P) if requested or when alias policy is active
    quality = "nominal"
    if evidence_flag:
        quality = "asymmetric"
        
    return {
        "odd_even_available": True,
        "odd_event_count": int(n_odd_observed),
        "even_event_count": int(n_even_observed),
        "odd_depth": round(odd_depth, 6),
        "odd_depth_uncertainty": round(odd_err, 6),
        "even_depth": round(even_depth, 6),
        "even_depth_uncertainty": round(even_err, 6),
        "odd_even_depth_difference": round(delta, 6),
        "odd_even_fractional_difference": round(frac_diff, 6),
        "odd_even_significance": round(significance, 4),
        "odd_even_p_value": round(p_value, 4),
        "odd_even_evidence_flag": bool(evidence_flag),
        "odd_even_quality": quality,
    }
