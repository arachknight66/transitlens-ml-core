# morphology.py
# --------------
# Transit morphology: trapezoid profile fitting, ingress/egress times, and V-shape scores.

from __future__ import annotations
import logging
import numpy as np
from scipy.optimize import minimize
from diagnostics.phase_windows import fold_phase

logger = logging.getLogger(__name__)

def trapezoid_model(phase: np.ndarray, depth: float, total_dur_phase: float, ingress_phase: float) -> np.ndarray:
    """
    Returns flux for a symmetric trapezoid transit model.
    Parameters:
        depth: fractional transit depth
        total_dur_phase: total transit width (phase units)
        ingress_phase: ingress width (phase units), egress is assumed equal.
    """
    flux = np.ones_like(phase)
    abs_phase = np.abs(phase)
    
    t1 = total_dur_phase / 2.0
    t2 = t1 - ingress_phase
    
    # Ingress region
    ingress_mask = (abs_phase < t1) & (abs_phase >= t2)
    if ingress_phase > 0:
        flux[ingress_mask] = 1.0 - depth * (t1 - abs_phase[ingress_mask]) / ingress_phase
        
    # Flat bottom region
    flat_mask = abs_phase < t2
    flux[flat_mask] = 1.0 - depth
    
    return flux

def run_morphology_analysis(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch_btjd: float,
    duration_days: float,
    depth: float,
    config: dict,
) -> dict:
    """
    Fits a symmetric trapezoid model to the folded light curve to measure profile shapes.
    """
    morph_config = config.get("Morphology", {})
    min_points = config.get("General", {}).get("minimum_in_transit_points", 5)
    
    unavailable = {
        "morphology_available": False,
        "trapezoid_depth": None,
        "trapezoid_duration": None,
        "ingress_duration": None,
        "egress_duration": None,
        "ingress_fraction": None,
        "egress_fraction": None,
        "ingress_egress_asymmetry": None,
        "flat_bottom_duration": None,
        "v_shape_score": None,
        "grazing_probability_proxy": None,
        "morphology_fit_quality": None,
        "morphology_evidence_flag": False,
        "morphology_quality": "unavailable",
    }
    
    if len(time) == 0 or period <= 0 or duration_days <= 0 or depth <= 0:
        return unavailable
        
    phase = fold_phase(time, period, epoch_btjd)
    
    # Select in-transit and baseline points (exclude everything > 1.5 * duration)
    half_dur_phase = (duration_days / period) / 2.0
    fit_mask = np.abs(phase) <= half_dur_phase * 2.0
    
    if fit_mask.sum() < min_points * 2:
        return unavailable
        
    fit_phase = phase[fit_mask]
    fit_flux = flux[fit_mask]
    
    # Grid search / minimize over depth, total_dur, and ingress
    # We construct objective function as sum of squared residuals
    def objective(params):
        d, t_dur, t_ing = params
        if d < 0 or t_dur < 0 or t_ing < 0 or t_ing > t_dur / 2.0:
            return 1e9
        model = trapezoid_model(fit_phase, d, t_dur, t_ing)
        return float(np.sum((fit_flux - model)**2))
        
    # Initial guesses
    init_depth = depth
    init_total_dur = duration_days / period
    init_ingress = 0.1 * init_total_dur
    
    bounds = [(1e-6, 0.99), (0.1 * init_total_dur, 2.0 * init_total_dur), (0.0, 0.5 * init_total_dur)]
    
    res = minimize(
        objective,
        x0=[init_depth, init_total_dur, init_ingress],
        method="Nelder-Mead",
        options={"maxiter": 200},
    )
    
    if not res.success:
        return unavailable
        
    fit_depth, fit_tot_dur_phase, fit_ing_phase = res.x
    
    # Check boundaries
    fit_depth = float(np.clip(fit_depth, 0.0, 1.0))
    fit_tot_dur_phase = float(np.clip(fit_tot_dur_phase, 0.0, 1.0))
    fit_ing_phase = float(np.clip(fit_ing_phase, 0.0, 0.5 * fit_tot_dur_phase))
    
    # Calculate durations in days
    total_dur_days = fit_tot_dur_phase * period
    ingress_days = fit_ing_phase * period
    egress_days = ingress_days  # symmetric model
    flat_bottom_days = total_dur_days - 2.0 * ingress_days
    
    ingress_frac = ingress_days / total_dur_days if total_dur_days > 0 else 0.0
    
    # V-shape score: fraction of total duration occupied by ingress and egress
    # For a perfect V-shape, flat_bottom is 0, so v_shape_score is 1.0
    v_shape_score = (2.0 * ingress_days) / total_dur_days if total_dur_days > 0 else 0.0
    v_shape_score = float(np.clip(v_shape_score, 0.0, 1.0))
    
    # Fit quality R2
    mean_flux = np.mean(fit_flux)
    ss_tot = np.sum((fit_flux - mean_flux)**2)
    ss_res = res.fun
    fit_quality = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    # Evidence flag: V-shape score crosses configured limit
    v_shape_threshold = config.get("Morphology", {}).get("grazing_event_warning_thresholds", {}).get("v_shape_limit", 0.80)
    evidence_flag = v_shape_score >= v_shape_threshold
    
    quality = "trapezoidal"
    if evidence_flag:
        quality = "v_shaped"
        
    return {
        "morphology_available": True,
        "trapezoid_depth": round(fit_depth, 6),
        "trapezoid_duration": round(total_dur_days, 6),
        "ingress_duration": round(ingress_days, 6),
        "egress_duration": round(egress_days, 6),
        "ingress_fraction": round(ingress_frac, 4),
        "egress_fraction": round(ingress_frac, 4),
        "ingress_egress_asymmetry": 0.0, # symmetric fit
        "flat_bottom_duration": round(flat_bottom_days, 6),
        "v_shape_score": round(v_shape_score, 4),
        "grazing_probability_proxy": round(v_shape_score, 4),
        "morphology_fit_quality": round(fit_quality, 4),
        "morphology_evidence_flag": bool(evidence_flag),
        "morphology_quality": quality,
    }
