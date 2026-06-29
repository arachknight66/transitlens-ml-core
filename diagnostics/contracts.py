# contracts.py
# ------------
# Formal schema definitions and validation for TransitLens Phase 2 vetting outputs.

from __future__ import annotations
import math
from typing import Any

SCHEMA_KEYS = {
    "Identity": {
        "target_id": str,
        "tic_id": int,
        "sector": int,
        "observation_id": str,
        "source_product": str,
        "source_checksum": str,
        "diagnostics_version": str,
        "ephemeris_mode": str,
    },
    "Ephemeris": {
        "period_days": (float, type(None)),
        "epoch_btjd": (float, type(None)),
        "duration_days": (float, type(None)),
        "ephemeris_source": str,
        "period_uncertainty_days": (float, type(None)),
        "epoch_uncertainty_days": (float, type(None)),
        "alias_warning": bool,
        "alias_type": str,
        "ephemeris_quality": str,
    },
    "OddEven": {
        "odd_even_available": bool,
        "odd_event_count": int,
        "even_event_count": int,
        "odd_depth": (float, type(None)),
        "odd_depth_uncertainty": (float, type(None)),
        "even_depth": (float, type(None)),
        "even_depth_uncertainty": (float, type(None)),
        "odd_even_depth_difference": (float, type(None)),
        "odd_even_fractional_difference": (float, type(None)),
        "odd_even_significance": (float, type(None)),
        "odd_even_p_value": (float, type(None)),
        "odd_even_evidence_flag": bool,
        "odd_even_quality": str,
    },
    "SecondaryEclipse": {
        "secondary_available": bool,
        "secondary_phase": (float, type(None)),
        "secondary_epoch_btjd": (float, type(None)),
        "secondary_depth": (float, type(None)),
        "secondary_depth_uncertainty": (float, type(None)),
        "secondary_significance": (float, type(None)),
        "secondary_duration_days": (float, type(None)),
        "secondary_primary_depth_ratio": (float, type(None)),
        "secondary_delta_bic": (float, type(None)),
        "secondary_global_p_value": (float, type(None)),
        "secondary_evidence_flag": bool,
        "secondary_quality": str,
    },
    "Morphology": {
        "morphology_available": bool,
        "trapezoid_depth": (float, type(None)),
        "trapezoid_duration": (float, type(None)),
        "ingress_duration": (float, type(None)),
        "egress_duration": (float, type(None)),
        "ingress_fraction": (float, type(None)),
        "egress_fraction": (float, type(None)),
        "ingress_egress_asymmetry": (float, type(None)),
        "flat_bottom_duration": (float, type(None)),
        "v_shape_score": (float, type(None)),
        "grazing_probability_proxy": (float, type(None)),
        "morphology_fit_quality": (float, type(None)),
        "morphology_evidence_flag": bool,
        "morphology_quality": str,
    },
    "Harmonics": {
        "harmonic_available": bool,
        "orbital_amplitude": (float, type(None)),
        "orbital_amplitude_uncertainty": (float, type(None)),
        "first_harmonic_amplitude": (float, type(None)),
        "first_harmonic_uncertainty": (float, type(None)),
        "ellipsoidal_amplitude": (float, type(None)),
        "ellipsoidal_significance": (float, type(None)),
        "reflection_amplitude": (float, type(None)),
        "reflection_significance": (float, type(None)),
        "beaming_amplitude": (float, type(None)),
        "beaming_significance": (float, type(None)),
        "harmonic_delta_bic": (float, type(None)),
        "harmonic_evidence_flag": bool,
        "harmonic_quality": str,
    },
    "Centroids": {
        "centroid_available": bool,
        "centroid_column_out": (float, type(None)),
        "centroid_row_out": (float, type(None)),
        "centroid_column_in": (float, type(None)),
        "centroid_row_in": (float, type(None)),
        "centroid_shift_column_pixels": (float, type(None)),
        "centroid_shift_row_pixels": (float, type(None)),
        "centroid_shift_pixels": (float, type(None)),
        "centroid_shift_arcsec": (float, type(None)),
        "centroid_shift_uncertainty_pixels": (float, type(None)),
        "centroid_shift_significance": (float, type(None)),
        "centroid_mahalanobis_distance": (float, type(None)),
        "centroid_permutation_p_value": (float, type(None)),
        "centroid_points_in": int,
        "centroid_points_out": int,
        "centroid_evidence_flag": bool,
        "centroid_quality": str,
    },
    "DifferenceImaging": {
        "difference_image_available": bool,
        "difference_image_snr": (float, type(None)),
        "difference_source_column": (float, type(None)),
        "difference_source_row": (float, type(None)),
        "target_column": (float, type(None)),
        "target_row": (float, type(None)),
        "source_target_offset_pixels": (float, type(None)),
        "source_target_offset_arcsec": (float, type(None)),
        "source_target_offset_uncertainty_pixels": (float, type(None)),
        "source_target_offset_significance": (float, type(None)),
        "difference_flux": (float, type(None)),
        "difference_flux_uncertainty": (float, type(None)),
        "difference_image_evidence_flag": bool,
        "difference_image_quality": str,
        "difference_image_path": str,
        "difference_image_plot_path": str,
    },
    "Gaia": {
        "gaia_available": bool,
        "gaia_release": str,
        "gaia_target_source_id": (int, type(None)),
        "gaia_target_match_sep_arcsec": (float, type(None)),
        "gaia_neighbor_count": (int, type(None)),
        "nearest_neighbor_source_id": (int, type(None)),
        "nearest_neighbor_sep_arcsec": (float, type(None)),
        "nearest_neighbor_delta_gmag": (float, type(None)),
        "nearest_neighbor_delta_tmag": (float, type(None)),
        "summed_neighbor_flux_ratio": (float, type(None)),
        "aperture_weighted_neighbor_flux_ratio": (float, type(None)),
        "brightest_neighbor_position_angle": (float, type(None)),
        "gaia_query_timestamp": str,
        "gaia_cache_key": str,
        "gaia_evidence_flag": bool,
        "gaia_quality": str,
    },
    "Crowding": {
        "crowding_available": bool,
        "crowdsap": (float, type(None)),
        "flfrcsap": (float, type(None)),
        "contamination_fraction": (float, type(None)),
        "observed_depth": (float, type(None)),
        "observed_depth_uncertainty": (float, type(None)),
        "dilution_corrected_depth": (float, type(None)),
        "dilution_corrected_depth_uncertainty": (float, type(None)),
        "correction_factor": (float, type(None)),
        "correction_quality": str,
        "crowding_evidence_flag": bool,
    },
    "MultiAperture": {
        "multi_aperture_available": bool,
        "aperture_count": int,
        "aperture_pixel_counts": list,
        "aperture_depths": list,
        "aperture_depth_uncertainties": list,
        "aperture_depth_slope": (float, type(None)),
        "aperture_depth_slope_uncertainty": (float, type(None)),
        "aperture_depth_consistency_chi2": (float, type(None)),
        "aperture_depth_consistency_p_value": (float, type(None)),
        "multi_aperture_evidence_flag": bool,
        "multi_aperture_quality": str,
        "aperture_diagnostic_plot_path": str,
    },
    "Aggregation": {
        "eb_risk_score": (float, type(None)),
        "blend_risk_score": (float, type(None)),
        "eb_risk_level": str,
        "blend_risk_level": str,
        "eb_evidence_flags": list,
        "blend_evidence_flags": list,
        "independent_eb_evidence_count": int,
        "independent_blend_evidence_count": int,
        "contradictory_evidence_flags": list,
        "recommended_route": str,
        "recommendation_reason": str,
        "review_required": bool,
        "missing_diagnostics": list,
        "quality_flags": list,
        "threshold_policy_version": str,
    }
}

def get_default_diagnostics_dict() -> dict:
    """Creates a diagnostics dict populated with standard default/unavailable values."""
    res = {}
    for section, fields in SCHEMA_KEYS.items():
        for field, ftype in fields.items():
            if ftype == bool:
                res[field] = False
            elif ftype == int:
                res[field] = 0
            elif ftype == str:
                res[field] = ""
            elif ftype == list:
                res[field] = []
            elif isinstance(ftype, tuple) and type(None) in ftype:
                res[field] = None
            else:
                res[field] = None
    # Adjust specific available flags
    res["diagnostics_version"] = "2.0.0"
    res["ephemeris_source"] = "none"
    res["ephemeris_quality"] = "unknown"
    res["odd_even_quality"] = "untested"
    res["secondary_quality"] = "untested"
    res["morphology_quality"] = "untested"
    res["harmonic_quality"] = "untested"
    res["centroid_quality"] = "untested"
    res["difference_image_quality"] = "untested"
    res["gaia_quality"] = "untested"
    res["correction_quality"] = "untested"
    res["multi_aperture_quality"] = "untested"
    res["eb_risk_level"] = "unavailable"
    res["blend_risk_level"] = "unavailable"
    res["recommended_route"] = "review_required"
    res["recommendation_reason"] = "no diagnostics run"
    res["review_required"] = True
    return res

def validate_schema(data: dict) -> bool:
    """
    Validates a diagnostics dictionary against the contract schema.
    Raises ValueError on validation failures.
    """
    for section, fields in SCHEMA_KEYS.items():
        for field, ftype in fields.items():
            if field not in data:
                raise ValueError(f"Schema violation: missing field '{field}' in section '{section}'")
            val = data[field]
            if isinstance(ftype, tuple):
                if not any(isinstance(val, t) for t in ftype):
                    raise ValueError(f"Schema violation: field '{field}' expected types {ftype}, got {type(val)}")
            else:
                if not isinstance(val, ftype):
                    raise ValueError(f"Schema violation: field '{field}' expected type {ftype}, got {type(val)}")
                    
            # Ensure NaN values are not in fields that should be floats or None (JSON serialization safety)
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                raise ValueError(f"Schema violation: field '{field}' contains non-finite float {val}. Use None instead.")
    return True
