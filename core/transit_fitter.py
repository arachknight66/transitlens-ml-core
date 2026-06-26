import logging
import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

def trapezoid_transit_model(time, t0, duration, depth, ingress_ratio, period):
    """
    Analytical trapezoidal transit model.
    ingress_ratio is the ratio of ingress duration to total duration (0.0 to 0.5).
    0.0 represents a flat box (flat bottom), 0.5 represents a fully V-shaped profile.
    """
    flux = np.ones_like(time, dtype=float)
    
    if period <= 0.0 or duration <= 0.0 or depth <= 0.0:
        return flux
        
    # Phase fold around 0
    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
    dt = np.abs(phase * period)
    
    ingress_duration = duration * ingress_ratio
    half_dur = duration / 2.0
    
    # Out of transit
    out_mask = dt >= half_dur
    # Full transit flat bottom
    flat_mask = dt <= (half_dur - ingress_duration)
    # Ingress/egress
    slope_mask = (~out_mask) & (~flat_mask)
    
    flux[flat_mask] = 1.0 - depth
    
    # Linear interpolation for ingress/egress
    if ingress_duration > 1e-6:
        fraction = (half_dur - dt[slope_mask]) / ingress_duration
        flux[slope_mask] = 1.0 - depth * fraction
        
    return flux

def fit_transit(time, flux, init_period, init_t0, init_duration, init_depth):
    """
    Fits the trapezoidal transit model to the (time, flux) data.
    Fixes period, fits t0, duration, depth, and ingress_ratio.
    """
    # Restrict data to phase window around transit to speed up fitting
    phase = ((time - init_t0) / init_period + 0.5) % 1.0 - 0.5
    # Keep points within 3 * duration of transit
    keep = np.abs(phase * init_period) <= max(0.5, 3.0 * init_duration)
    t_fit = time[keep]
    f_fit = flux[keep]
    
    if len(t_fit) < 10:
        logger.warning("Too few data points in transit window to fit.")
        return {
            "period_days": init_period,
            "epoch_btjd": init_t0,
            "duration_days": init_duration,
            "depth": init_depth,
            "ingress_ratio": 0.1,
            "fit_quality": 0.0,
            "pcov": None
        }
        
    # Initial guesses: t0, duration, depth, ingress_ratio
    p0 = [init_t0, init_duration, init_depth, 0.1]
    
    # Parameter bounds
    bounds = (
        [init_t0 - 0.2, 0.005, 0.0, 0.0],  # lower
        [init_t0 + 0.2, 1.0, 0.5, 0.5]     # upper
    )
    
    try:
        # Define wrapper fitting function that keeps period fixed
        def fit_func(t, t0, duration, depth, ingress_ratio):
            return trapezoid_transit_model(t, t0, duration, depth, ingress_ratio, init_period)
            
        popt, pcov = curve_fit(fit_func, t_fit, f_fit, p0=p0, bounds=bounds, method="trf")
        
        t0_fit, dur_fit, depth_fit, ingress_ratio_fit = popt
        
        # Calculate fit quality (R-squared score)
        f_pred = fit_func(t_fit, *popt)
        ss_res = np.sum((f_fit - f_pred) ** 2)
        ss_tot = np.sum((f_fit - np.mean(f_fit)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-10 else 0.0
        
        return {
            "period_days": init_period,
            "epoch_btjd": float(t0_fit),
            "duration_days": float(dur_fit),
            "depth": float(depth_fit),
            "ingress_ratio": float(ingress_ratio_fit),
            "fit_quality": max(0.0, r2),
            "pcov": pcov
        }
    except Exception as e:
        logger.warning("Transit fitting failed: %s. Returning BLS estimates.", e)
        return {
            "period_days": init_period,
            "epoch_btjd": init_t0,
            "duration_days": init_duration,
            "depth": init_depth,
            "ingress_ratio": 0.1,
            "fit_quality": 0.0,
            "pcov": None
        }
