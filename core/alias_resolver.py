import logging
import numpy as np
from core.utils import phase_fold

logger = logging.getLogger(__name__)

def resolve_aliases(time, flux, detected_period, detected_t0, detected_duration, detected_depth):
    """
    Checks for double-period or half-period aliases in the detected signal.
    
    Returns:
        dict: {
            "resolved_period": float,
            "resolved_t0": float,
            "alias_warning": bool,
            "alias_type": str (e.g., "double_period", "half_period", "none"),
            "odd_even_delta": float,
            "secondary_eclipse_detected": bool,
            "secondary_eclipse_depth": float
        }
    """
    resolved_period = detected_period
    resolved_t0 = detected_t0
    alias_warning = False
    alias_type = "none"
    secondary_eclipse_detected = False
    secondary_eclipse_depth = 0.0
    
    if detected_period <= 0.0:
        return {
            "resolved_period": resolved_period,
            "resolved_t0": resolved_t0,
            "alias_warning": alias_warning,
            "alias_type": alias_type,
            "odd_even_delta": 0.0,
            "secondary_eclipse_detected": False,
            "secondary_eclipse_depth": 0.0
        }
        
    # 1. Check for Secondary Eclipse at phase = 0.5 when folded at 2 * detected_period
    # (If the true period is 2P, folding at 2P puts the primary eclipse at phase 0 and the secondary at phase 0.5)
    double_period = 2.0 * detected_period
    phase_double = phase_fold(time, double_period, detected_t0)
    
    # Measure average out-of-transit flux level
    half_dur_primary = (detected_duration / double_period) / 2.0
    in_transit_primary = np.abs(phase_double) < half_dur_primary
    in_transit_secondary = np.abs(np.abs(phase_double) - 0.5) < half_dur_primary
    out_of_transit = ~(in_transit_primary | in_transit_secondary)
    
    out_flux = flux[out_of_transit]
    local_noise = np.std(out_flux) if len(out_flux) > 0 else 0.001
    
    # Secondary eclipse depth measurement
    secondary_flux = flux[in_transit_secondary]
    primary_flux = flux[in_transit_primary]
    
    if len(secondary_flux) > 5 and len(out_flux) > 0:
        med_out = np.median(out_flux)
        med_sec = np.median(secondary_flux)
        sec_depth = float(med_out - med_sec)
        
        # Check if secondary eclipse is significant (e.g. > 3.0 * noise)
        if sec_depth > 3.0 * local_noise and sec_depth > 0.0002:
            # Check if primary and secondary depths are different
            prim_depth = float(med_out - np.median(primary_flux))
            if abs(prim_depth - sec_depth) > 2.0 * local_noise:
                logger.info("Found secondary eclipse with depth %.5f. True period is likely double: %.4fd", sec_depth, double_period)
                resolved_period = double_period
                alias_warning = True
                alias_type = "double_period"
                secondary_eclipse_detected = True
                secondary_eclipse_depth = sec_depth
                
    # 2. Check for Odd/Even Depth Asymmetry
    # Fold at resolved_period, label transits as odd vs even, and compare their depths
    phase = phase_fold(time, resolved_period, resolved_t0)
    # Transit epoch cycles
    cycle = np.round((time - resolved_t0) / resolved_period)
    
    half_dur_resolved = (detected_duration / resolved_period) / 2.0
    in_transit = np.abs(phase) < half_dur_resolved
    odd_mask = (cycle % 2 == 1) & in_transit
    even_mask = (cycle % 2 == 0) & in_transit
    
    odd_flux = flux[odd_mask]
    even_flux = flux[even_mask]
    
    odd_even_delta = 0.0
    if len(odd_flux) > 5 and len(even_flux) > 5:
        med_odd = np.median(odd_flux)
        med_even = np.median(even_flux)
        odd_even_delta = float(abs(med_odd - med_even))
        
        # If odd and even depth delta is significant, this suggests an eclipsing binary
        # and the period is likely double the detected period (meaning we missed the secondary eclipse)
        if odd_even_delta > 3.0 * local_noise and odd_even_delta > 0.0005:
            logger.info("Significant odd/even depth delta: %.5f. True period is likely double: %.4fd", odd_even_delta, double_period)
            resolved_period = double_period
            alias_warning = True
            alias_type = "odd_even_asymmetry"
            
    return {
        "resolved_period": resolved_period,
        "resolved_t0": resolved_t0,
        "alias_warning": alias_warning,
        "alias_type": alias_type,
        "odd_even_delta": odd_even_delta,
        "secondary_eclipse_detected": secondary_eclipse_detected,
        "secondary_eclipse_depth": secondary_eclipse_depth
    }
