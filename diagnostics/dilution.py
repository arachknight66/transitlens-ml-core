# dilution.py
# -----------
# Dilution correction: correcting observed transit depths for aperture third-light blending.

from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)

def run_dilution_correction(
    observed_depth: float,
    observed_depth_uncertainty: float | None,
    crowding_res: dict,
    config: dict,
) -> dict:
    """
    Applies dilution correction: intrinsic_depth = observed_depth / CROWDSAP.
    Propagates uncertainties.
    """
    unavailable = {
        "observed_depth": observed_depth,
        "observed_depth_uncertainty": observed_depth_uncertainty,
        "dilution_corrected_depth": None,
        "dilution_corrected_depth_uncertainty": None,
        "correction_factor": None,
        "correction_quality": "unavailable",
    }
    
    if not crowding_res.get("crowding_available"):
        return unavailable
        
    crowdsap = crowding_res["crowdsap"]
    if crowdsap <= 0:
        return unavailable
        
    cd_config = config.get("CrowdingDilution", {})
    extreme_threshold = cd_config.get("extreme_correction_warning_threshold", 0.30)
    
    # Correction factor: 1 / CROWDSAP
    correction_factor = 1.0 / crowdsap
    
    # Corrected depth
    corrected_depth = observed_depth * correction_factor
    
    # Uncertainty propagation:
    # (sigma_corr / d_corr) ^ 2 = (sigma_obs / d_obs) ^ 2 + (sigma_crowd / crowdsap) ^ 2
    # Assume crowdsap uncertainty is ~0.02 (from config)
    crowdsap_err = cd_config.get("uncertainty_assumptions", {}).get("crowdsap_sigma", 0.02)
    
    if observed_depth_uncertainty is not None and observed_depth > 0:
        rel_obs = observed_depth_uncertainty / observed_depth
        rel_crowd = crowdsap_err / crowdsap
        corrected_depth_err = corrected_depth * np.sqrt(rel_obs**2 + rel_crowd**2)
    else:
        corrected_depth_err = None
        
    quality = "nominal"
    if crowdsap < extreme_threshold:
        quality = "extreme_correction"
        
    return {
        "observed_depth": round(observed_depth, 6),
        "observed_depth_uncertainty": round(observed_depth_uncertainty, 6) if observed_depth_uncertainty is not None else None,
        "dilution_corrected_depth": round(corrected_depth, 8),
        "dilution_corrected_depth_uncertainty": round(corrected_depth_err, 8) if corrected_depth_err is not None else None,
        "correction_factor": round(correction_factor, 6),
        "correction_quality": quality,
    }
