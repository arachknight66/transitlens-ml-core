# __init__.py
# -----------
# Orchestration interface for TransitLens Phase 2 vetting diagnostics.

from __future__ import annotations
import logging
from pathlib import Path
import yaml

from diagnostics.contracts import get_default_diagnostics_dict, validate_schema
from diagnostics.phase_windows import fold_phase, get_event_mask, assign_cycle_numbers, analyze_sector_coverage
from diagnostics.odd_even import run_odd_even_analysis
from diagnostics.secondary_eclipse import run_secondary_eclipse_search
from diagnostics.morphology import run_morphology_analysis
from diagnostics.harmonic_variability import run_harmonic_analysis
from diagnostics.centroid import run_centroid_analysis
from diagnostics.difference_imaging import run_difference_imaging
from diagnostics.source_localization import run_source_localization
from diagnostics.gaia_neighbors import run_gaia_neighbor_query
from diagnostics.crowding import run_crowding_analysis
from diagnostics.dilution import run_dilution_correction
from diagnostics.multi_aperture import run_multi_aperture_analysis
from diagnostics.ephemeris_matching import run_ephemeris_matching
from diagnostics.evidence_aggregation import run_evidence_aggregation
from diagnostics.quality import audit_observation_quality

logger = logging.getLogger(__name__)

# Default config path
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "phase2_diagnostics.yaml"

def _load_diagnostics_config(config_override: dict | None = None) -> dict:
    """Loads and parses the default phase2 diagnostics configuration."""
    cfg = {}
    if _DEFAULT_CONFIG_PATH.exists():
        try:
            with open(_DEFAULT_CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to read diagnostics config at {_DEFAULT_CONFIG_PATH}: {e}")
            
    if config_override:
        # Deep merge override
        for k, v in config_override.items():
            if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg

def run_diagnostics(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch_btjd: float,
    duration_days: float,
    depth: float,
    centroid_x: np.ndarray | None = None,
    centroid_y: np.ndarray | None = None,
    quality: np.ndarray | None = None,
    metadata: dict | None = None,
    config: dict | None = None,
) -> dict:
    """
    Unified entry point executing all Phase 2 scientific vetting diagnostics.
    """
    metadata = metadata or {}
    cfg = _load_diagnostics_config(config)
    
    # 0. Set default response structure
    res = get_default_diagnostics_dict()
    
    # Identity and metadata parameters
    res["target_id"] = str(metadata.get("target_id", "unknown"))
    res["tic_id"] = int(metadata.get("tic_id", 0))
    res["sector"] = int(metadata.get("sector", 0))
    res["observation_id"] = str(metadata.get("observation_id", "unknown"))
    res["source_product"] = str(metadata.get("fits_filename", ""))
    res["source_checksum"] = str(metadata.get("source_checksum", ""))
    res["ephemeris_mode"] = str(metadata.get("ephemeris_mode", "detected"))
    
    # Ephemeris params
    res["period_days"] = float(period) if period > 0 else None
    res["epoch_btjd"] = float(epoch_btjd) if epoch_btjd > 0 else None
    res["duration_days"] = float(duration_days) if duration_days > 0 else None
    
    if period <= 0 or duration_days <= 0 or len(time) == 0:
        validate_schema(res)
        return res
        
    # Run individual sub-modules
    try:
        # 1. Odd/Even depth analysis
        oe_res = run_odd_even_analysis(time, flux, period, epoch_btjd, duration_days, cfg)
        res.update(oe_res)
        
        # 2. Secondary eclipse search
        sec_res = run_secondary_eclipse_search(time, flux, period, epoch_btjd, duration_days, depth, cfg)
        res.update(sec_res)
        
        # 3. Morphology profile fit
        morph_res = run_morphology_analysis(time, flux, period, epoch_btjd, duration_days, depth, cfg)
        res.update(morph_res)
        
        # 4. Harmonic orbital variability
        harm_res = run_harmonic_analysis(time, flux, period, epoch_btjd, duration_days, cfg)
        res.update(harm_res)
        
        # 5. Centroid shifts
        cent_res = run_centroid_analysis(time, flux, period, epoch_btjd, duration_days, centroid_x, centroid_y, quality, cfg)
        res.update(cent_res)
        
        # 6 & 7 & 10. Difference imaging, source localization, and multi-aperture consistency (via TPF path if available)
        tpf_path = metadata.get("tpf_path")
        diff_res = run_difference_imaging(tpf_path, period, epoch_btjd, duration_days, cfg)
        res["difference_image_available"] = diff_res["difference_image_available"]
        res["difference_image_snr"] = diff_res["difference_image_snr"]
        res["target_column"] = diff_res["target_column"]
        res["target_row"] = diff_res["target_row"]
        res["difference_image_quality"] = diff_res["difference_image_quality"]
        
        loc_res = run_source_localization(diff_res, cfg)
        res.update({k: v for k, v in loc_res.items() if k in res})
        
        ma_res = run_multi_aperture_analysis(tpf_path, period, epoch_btjd, duration_days, cfg)
        res.update(ma_res)
        
        # 8. Gaia neighbor check
        gaia_res = run_gaia_neighbor_query(res["target_id"], metadata.get("ra"), metadata.get("dec"), cfg)
        res.update(gaia_res)
        
        # 9 & 11. Crowding and dilution correction
        crowd_res = run_crowding_analysis(metadata, cfg)
        res["crowding_available"] = crowd_res["crowding_available"]
        res["crowdsap"] = crowd_res["crowdsap"]
        res["flfrcsap"] = crowd_res["flfrcsap"]
        res["contamination_fraction"] = crowd_res["contamination_fraction"]
        res["crowding_evidence_flag"] = crowd_res["crowding_evidence_flag"]
        
        dil_res = run_dilution_correction(depth, metadata.get("depth_uncertainty"), crowd_res, cfg)
        res["observed_depth"] = dil_res["observed_depth"]
        res["observed_depth_uncertainty"] = dil_res["observed_depth_uncertainty"]
        res["dilution_corrected_depth"] = dil_res["dilution_corrected_depth"]
        res["dilution_corrected_depth_uncertainty"] = dil_res["dilution_corrected_depth_uncertainty"]
        res["correction_factor"] = dil_res["correction_factor"]
        res["correction_quality"] = dil_res["correction_quality"]
        
        # 12. Ephemeris cross-match check
        ephem_res = run_ephemeris_matching(res["target_id"], period, epoch_btjd, metadata.get("ra"), metadata.get("dec"), cfg)
        res.update(ephem_res)
        
        # 13. Quality controls
        res["quality_flags"] = audit_observation_quality(time, flux, centroid_x, centroid_y, quality, metadata)
        
        # 14. Evidence Aggregation and Vetting Routing
        agg_res = run_evidence_aggregation(
            oe_res, sec_res, morph_res, harm_res, cent_res, diff_res, loc_res, gaia_res, crowd_res, ma_res, ephem_res, cfg
        )
        res.update(agg_res)
        
    except Exception as e:
        logger.error(f"Error executing diagnostic sub-modules: {e}", exc_info=True)
        res["recommended_route"] = "review_required"
        res["recommendation_reason"] = f"unhandled execution failure: {str(e)[:100]}"
        res["review_required"] = True
        res["quality_flags"].append("diagnostic_execution_failure")
        
    # Schema validation verification
    validate_schema(res)
    
    return res
