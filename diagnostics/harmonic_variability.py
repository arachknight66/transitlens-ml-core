# harmonic_variability.py
# ---------------------
# Out-of-eclipse harmonic variability analysis: ellipsoidal, reflection, and beaming modulations.

from __future__ import annotations
import logging
import numpy as np
from diagnostics.phase_windows import fold_phase, get_event_mask

logger = logging.getLogger(__name__)

def run_harmonic_analysis(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch_btjd: float,
    duration_days: float,
    config: dict,
) -> dict:
    """
    Fits sin/cos terms at 1/P and 2/P on the out-of-eclipse light curve.
    """
    hv_config = config.get("HarmonicVariability", {})
    sig_threshold = hv_config.get("amplitude_significance_thresholds", 3.0)
    
    unavailable = {
        "harmonic_available": False,
        "orbital_amplitude": None,
        "orbital_amplitude_uncertainty": None,
        "first_harmonic_amplitude": None,
        "first_harmonic_uncertainty": None,
        "ellipsoidal_amplitude": None,
        "ellipsoidal_significance": None,
        "reflection_amplitude": None,
        "reflection_significance": None,
        "beaming_amplitude": None,
        "beaming_significance": None,
        "harmonic_delta_bic": None,
        "harmonic_evidence_flag": False,
        "harmonic_quality": "unavailable",
    }
    
    if len(time) == 0 or period <= 0 or duration_days <= 0:
        return unavailable
        
    # Mask primary and secondary (phase ~0.5) eclipses
    phase = fold_phase(time, period, epoch_btjd)
    half_dur_phase = (duration_days / period) / 2.0
    
    in_primary = np.abs(phase) <= half_dur_phase * 1.5
    in_secondary = np.abs(np.abs(phase) - 0.5) <= half_dur_phase * 1.5
    
    out_of_eclipse = ~(in_primary | in_secondary)
    
    if out_of_eclipse.sum() < 30:
        return unavailable
        
    t_oe = time[out_of_eclipse]
    f_oe = flux[out_of_eclipse]
    
    # Standardize time to avoid numerical issues
    t_ref = t_oe - epoch_btjd
    
    # Construct design matrix: [1, sin(2pi t / P), cos(2pi t / P), sin(4pi t / P), cos(4pi t / P)]
    omega = 2.0 * np.pi / period
    x_matrix = np.column_stack([
        np.ones_like(t_ref),
        np.sin(omega * t_ref),
        np.cos(omega * t_ref),
        np.sin(2.0 * omega * t_ref),
        np.cos(2.0 * omega * t_ref),
    ])
    
    try:
        # Solve least squares: beta = (X^T X)^-1 X^T y
        beta, residuals, rank, s = np.linalg.lstsq(x_matrix, f_oe, rcond=None)
        
        # Calculate parameter covariance matrix
        n = len(t_oe)
        p = x_matrix.shape[1]
        
        if n > p:
            res_var = residuals[0] / (n - p) if len(residuals) > 0 else np.var(f_oe - np.dot(x_matrix, beta))
            covariance = np.linalg.inv(np.dot(x_matrix.T, x_matrix)) * res_var
            stderr = np.sqrt(np.diag(covariance))
        else:
            res_var = 1e-6
            stderr = np.zeros(p)
            
        c_const, c_sin1, c_cos1, c_sin2, c_cos2 = beta
        e_const, e_sin1, e_cos1, e_sin2, e_cos2 = stderr
        
        # Reflection/beaming: modulation at fundamental frequency 1/P
        amp_1 = float(np.sqrt(c_sin1**2 + c_cos1**2))
        err_1 = float(np.sqrt((c_sin1*e_sin1)**2 + (c_cos1*e_cos1)**2) / amp_1) if amp_1 > 0 else 0.0
        
        # Ellipsoidal variation: modulation at first harmonic 2/P
        amp_2 = float(np.sqrt(c_sin2**2 + c_cos2**2))
        err_2 = float(np.sqrt((c_sin2*e_sin2)**2 + (c_cos2*e_cos2)**2) / amp_2) if amp_2 > 0 else 0.0
        
        # Significances
        sig_1 = amp_1 / err_1 if err_1 > 0 else 0.0
        sig_2 = amp_2 / err_2 if err_2 > 0 else 0.0
        
        # Separate reflection vs beaming components
        # reflection typically cos(omega*t), beaming sin(omega*t)
        reflection_amp = abs(c_cos1)
        reflection_sig = reflection_amp / e_cos1 if e_cos1 > 0 else 0.0
        
        beaming_amp = abs(c_sin1)
        beaming_sig = beaming_amp / e_sin1 if e_sin1 > 0 else 0.0
        
        # Delta BIC
        # Flat model: design matrix is just column of ones
        # residual variance flat vs harmonic
        flat_res = max(1e-12, float(np.sum((f_oe - np.mean(f_oe))**2)))
        harm_res = max(1e-12, float(np.sum((f_oe - np.dot(x_matrix, beta))**2)))
        
        # BIC = k * ln(n) - 2 * ln(L) ≈ k * ln(n) + n * ln(res/n)
        bic_flat = 1.0 * np.log(n) + n * np.log(flat_res / n)
        bic_harm = p * np.log(n) + n * np.log(harm_res / n)
        delta_bic = float(bic_flat - bic_harm)
        
        if not np.isfinite(delta_bic):
            delta_bic = 0.0
        else:
            delta_bic = float(np.clip(delta_bic, -9999.0, 9999.0))
        
        evidence_flag = (sig_2 >= sig_threshold or sig_1 >= sig_threshold) and (delta_bic > 10)
        
        quality = "stable"
        if evidence_flag:
            quality = "variable"
            
        return {
            "harmonic_available": True,
            "orbital_amplitude": round(amp_1, 6),
            "orbital_amplitude_uncertainty": round(err_1, 6),
            "first_harmonic_amplitude": round(amp_2, 6),
            "first_harmonic_uncertainty": round(err_2, 6),
            "ellipsoidal_amplitude": round(amp_2, 6),
            "ellipsoidal_significance": round(sig_2, 4),
            "reflection_amplitude": round(reflection_amp, 6),
            "reflection_significance": round(reflection_sig, 4),
            "beaming_amplitude": round(beaming_amp, 6),
            "beaming_significance": round(beaming_sig, 4),
            "harmonic_delta_bic": round(delta_bic, 2),
            "harmonic_evidence_flag": bool(evidence_flag),
            "harmonic_quality": quality,
        }
        
    except Exception as exc:
        logger.warning(f"Harmonic analysis failed: {exc}")
        return unavailable
