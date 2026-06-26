import logging
import numpy as np
from core.utils import phase_fold

logger = logging.getLogger(__name__)

def calculate_sde(power):
    """
    Computes Signal Detection Efficiency (SDE) from the BLS power spectrum.
    SDE = (peak_power - mean_power) / std_power
    """
    if len(power) == 0:
        return 0.0
    mean_p = np.mean(power)
    std_p = np.std(power)
    if std_p < 1e-10:
        return 0.0
    peak_p = np.max(power)
    return float((peak_p - mean_p) / std_p)

def calculate_bootstrap_fap(time, flux, period, t0, duration, depth, n_iter=100):
    """
    Computes the False Alarm Probability (FAP) of the transit depth at the best period
    by shuffling the out-of-transit residuals and calculating random depths at the transit phase.
    This runs in milliseconds since it folds at the fixed period rather than running the full BLS.
    """
    if period <= 0.0 or depth <= 0.0 or len(time) < 50:
        return 1.0
        
    phase = phase_fold(time, period, t0)
    in_transit = (phase < (duration / period)) | (phase > 1.0 - (duration / period))
    
    out_flux = flux[~in_transit]
    if len(out_flux) < 10:
        return 1.0
        
    med_out = np.median(out_flux)
    actual_depth = med_out - np.median(flux[in_transit])
    
    # We shuffle the out-of-transit residuals and inject them back
    residuals = out_flux - med_out
    
    exceed_count = 0
    rng = np.random.default_rng(42)
    
    for _ in range(n_iter):
        shuffled_residuals = rng.choice(residuals, size=len(flux), replace=True)
        shuffled_flux = med_out + shuffled_residuals
        
        # Calculate depth of shuffled light curve at the same transit phase
        shuffled_depth = med_out - np.median(shuffled_flux[in_transit])
        if shuffled_depth >= actual_depth:
            exceed_count += 1
            
    fap = float(exceed_count) / n_iter
    return fap

def calculate_transit_snr(flux, in_transit_mask):
    """
    Calculates the Signal-to-Noise Ratio (SNR) of the transit detection.
    SNR = depth / (out-of-transit std / sqrt(n_in_transit))
    """
    out_transit_mask = ~in_transit_mask
    out_flux = flux[out_transit_mask]
    in_flux = flux[in_transit_mask]
    
    if len(out_flux) < 5 or len(in_flux) < 3:
        return 0.0
        
    med_out = np.median(out_flux)
    med_in = np.median(in_flux)
    depth = med_out - med_in
    
    std_out = np.std(out_flux)
    if std_out < 1e-10:
        return 0.0
        
    snr = depth / (std_out / np.sqrt(len(in_flux)))
    return float(max(0.0, snr))
