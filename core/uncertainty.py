import logging
import numpy as np

logger = logging.getLogger(__name__)

def estimate_uncertainties(fit_result, time_span, snr):
    """
    Extracts uncertainties for epoch, duration, and depth from the curve_fit covariance matrix.
    Approximates period uncertainty using the Kovacs scaling relation: P_err = P^2 / (T_span * SNR).
    """
    period = fit_result.get("period_days", 0.0)
    epoch = fit_result.get("epoch_btjd", 0.0)
    duration = fit_result.get("duration_days", 0.0)
    depth = fit_result.get("depth", 0.0)
    pcov = fit_result.get("pcov")
    
    # Defaults/fallbacks (e.g. 5% relative error)
    period_err = 0.05 * period if period > 0.0 else 0.0
    epoch_err = 0.05 * duration if duration > 0.0 else 0.01
    duration_err = 0.05 * duration if duration > 0.0 else 0.01
    depth_err = 0.05 * depth if depth > 0.0 else 0.0001
    
    # Calculate period uncertainty from SDE/SNR scaling
    if period > 0.0 and time_span > 0.0 and snr > 0.0:
        period_err = float((period ** 2) / (time_span * snr))
        # Ensure it doesn't exceed a reasonable threshold
        period_err = min(period_err, 0.1 * period)
        
    if pcov is not None:
        try:
            perr = np.sqrt(np.diag(pcov))
            # p0 index map: 0: t0, 1: duration, 2: depth, 3: ingress_ratio
            if len(perr) >= 3:
                if np.isfinite(perr[0]) and perr[0] > 0.0:
                    epoch_err = float(perr[0])
                if np.isfinite(perr[1]) and perr[1] > 0.0:
                    duration_err = float(perr[1])
                if np.isfinite(perr[2]) and perr[2] > 0.0:
                    depth_err = float(perr[2])
        except Exception as e:
            logger.warning("Failed to extract errors from covariance: %s. Using analytical fallbacks.", e)
            
    return {
        "period_uncertainty_days": float(period_err),
        "epoch_uncertainty_days": float(epoch_err),
        "duration_uncertainty_days": float(duration_err),
        "depth_uncertainty": float(depth_err)
    }
