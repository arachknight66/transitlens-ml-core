"""
core/uncertainty.py
------------------
Extracts parameter uncertainties for TransitLens fitting outputs.
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

def estimate_uncertainties(fit_result: dict, time_span: float, snr: float) -> dict:
    """
    Extracts uncertainties for orbital period, epoch, duration, and depth from fit results.
    Maintains compatibility with legacy pipeline.py signatures.
    """
    # Simply extract values from the detailed fit results or apply fallbacks
    period = fit_result.get("period_days", 0.0)
    epoch = fit_result.get("epoch_btjd", 0.0)
    duration = fit_result.get("duration_days", 0.0)
    depth = fit_result.get("depth", 0.0)
    
    period_err = fit_result.get("period_uncertainty_days")
    epoch_err = fit_result.get("epoch_uncertainty_days")
    duration_err = fit_result.get("duration_uncertainty_days")
    depth_err = fit_result.get("depth_uncertainty")
    
    # Fallback to Kovacs or defaults if not populated
    if period_err is None or period_err == 0.0:
        if period > 0.0 and time_span > 0.0 and snr > 0.0:
            period_err = float((period ** 2) / (time_span * snr))
            period_err = min(period_err, 0.1 * period)
        else:
            period_err = 0.05 * period if period > 0.0 else 0.0
            
    if epoch_err is None or epoch_err == 0.0:
        epoch_err = 0.05 * duration if duration > 0.0 else 0.01
        
    if duration_err is None or duration_err == 0.0:
        duration_err = 0.05 * duration if duration > 0.0 else 0.01
        
    if depth_err is None or depth_err == 0.0:
        depth_err = 0.05 * depth if depth > 0.0 else 0.0001
        
    return {
        "period_uncertainty_days": float(period_err),
        "epoch_uncertainty_days": float(epoch_err),
        "duration_uncertainty_days": float(duration_err),
        "depth_uncertainty": float(depth_err)
    }
