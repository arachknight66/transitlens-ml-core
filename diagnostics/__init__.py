# __init__.py
# -----------
# Orchestration interface for TransitLens Phase 2 vetting diagnostics.

from __future__ import annotations
import logging
from pathlib import Path
import yaml
import numpy as np

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
    res["ephemeris_source"] = str(metadata.get("ephemeris_source", "transitlens_bls"))
    res["ephemeris_quality"] = "detected" if res["ephemeris_mode"] == "detected" else "debug_only"
    res["diagnostics_version"] = str(cfg.get("General", {}).get("diagnostics_version", "2.0.0"))
    
    if period <= 0 or duration_days <= 0 or len(time) == 0:
        validate_schema(res)
        return res
        
    # Each component fails independently: unavailable evidence must not erase
    # successful measurements from other diagnostic families.
    failures = []
    def safe(name, function):
        try:
            return function()
        except Exception as exc:
            logger.exception("Phase 2 diagnostic '%s' failed", name)
            failures.append(f"{name}:{type(exc).__name__}")
            return {}

    oe_res = safe("odd_even", lambda: run_odd_even_analysis(time, flux, period, epoch_btjd, duration_days, cfg)); res.update(oe_res)
    sec_res = safe("secondary", lambda: run_secondary_eclipse_search(time, flux, period, epoch_btjd, duration_days, depth, cfg)); res.update(sec_res)
    morph_res = safe("morphology", lambda: run_morphology_analysis(time, flux, period, epoch_btjd, duration_days, depth, cfg)); res.update(morph_res)
    harm_res = safe("harmonics", lambda: run_harmonic_analysis(time, flux, period, epoch_btjd, duration_days, cfg)); res.update(harm_res)
    cent_res = safe("centroid", lambda: run_centroid_analysis(time, flux, period, epoch_btjd, duration_days, centroid_x, centroid_y, quality, cfg)); res.update(cent_res)

    tpf_path = metadata.get("tpf_path")
    diff_res = safe("difference_image", lambda: run_difference_imaging(tpf_path, period, epoch_btjd, duration_days, cfg))
    for key in ("difference_image_available", "difference_image_snr", "target_column", "target_row", "difference_image_quality"):
        if key in diff_res: res[key] = diff_res[key]
    loc_res = safe("source_localization", lambda: run_source_localization(diff_res, cfg)) if diff_res else {}
    res.update({k: v for k, v in loc_res.items() if k in res})
    ma_res = safe("multi_aperture", lambda: run_multi_aperture_analysis(tpf_path, period, epoch_btjd, duration_days, cfg)); res.update(ma_res)
    gaia_res = safe("gaia", lambda: run_gaia_neighbor_query(res["target_id"], metadata.get("ra"), metadata.get("dec"), cfg)); res.update(gaia_res)
    crowd_res = safe("crowding", lambda: run_crowding_analysis(metadata, cfg)); res.update({k:v for k,v in crowd_res.items() if k in res})
    dil_res = safe("dilution", lambda: run_dilution_correction(depth, metadata.get("depth_uncertainty"), crowd_res, cfg)); res.update({k:v for k,v in dil_res.items() if k in res})
    ephem_res = safe("ephemeris_matching", lambda: run_ephemeris_matching(res["target_id"], period, epoch_btjd, metadata.get("ra"), metadata.get("dec"), cfg)); res.update({k:v for k,v in ephem_res.items() if k in res})
    res["quality_flags"] = safe("quality", lambda: audit_observation_quality(time, flux, centroid_x, centroid_y, quality, metadata)) or []
    agg_res = safe("aggregation", lambda: run_evidence_aggregation(
        oe_res, sec_res, morph_res, harm_res, cent_res, diff_res, loc_res, gaia_res, crowd_res, ma_res, ephem_res, cfg))
    res.update(agg_res)
    if failures:
        res["quality_flags"] = list(res.get("quality_flags", [])) + [f"diagnostic_failure:{item}" for item in failures]
        res["review_required"] = True
        if not res.get("recommendation_reason"):
            res["recommendation_reason"] = "one or more diagnostics unavailable"
        
    # Convert non-finite floats (NaN/inf) to None for schema compliance
    def clean_nan_to_none(d: dict) -> dict:
        import math
        cleaned = {}
        for k, v in d.items():
            if isinstance(v, dict):
                cleaned[k] = clean_nan_to_none(v)
            elif isinstance(v, list):
                cleaned[k] = [None if (isinstance(x, (float, np.floating)) and not np.isfinite(x)) else x for x in v]
            elif isinstance(v, (float, np.floating)) and not np.isfinite(v):
                cleaned[k] = None
            else:
                cleaned[k] = v
        return cleaned

    res = clean_nan_to_none(res)

    # Schema validation verification
    validate_schema(res)
    
    return res
