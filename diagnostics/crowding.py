# crowding.py
# -----------
# Crowding diagnostics: parsing CROWDSAP / FLFRCSAP and checking aperture contamination.

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

def run_crowding_analysis(
    metadata: dict | None,
    config: dict,
) -> dict:
    """
    Extracts CROWDSAP and FLFRCSAP aperture parameters from observation metadata.
    """
    cd_config = config.get("CrowdingDilution", {})
    valid_range = cd_config.get("crowdsap_valid_range", [0.01, 1.0])
    
    unavailable = {
        "crowding_available": False,
        "crowdsap": None,
        "flfrcsap": None,
        "contamination_fraction": None,
        "crowding_evidence_flag": False,
    }
    
    if metadata is None:
        return unavailable
        
    # Extract values from metadata (look for both uppercase SPOC standard and QLP names)
    crowdsap = metadata.get("crowding_metric") or metadata.get("CROWDSAP")
    flfrcsap = metadata.get("flux_fraction") or metadata.get("FLFRCSAP")
    
    if crowdsap is None:
        return unavailable
        
    try:
        crowdsap = float(crowdsap)
        if flfrcsap is not None:
            flfrcsap = float(flfrcsap)
    except Exception:
        return unavailable
        
    # Validate range
    if not (valid_range[0] <= crowdsap <= valid_range[1]):
        logger.warning(f"CROWDSAP value {crowdsap} outside valid range {valid_range}")
        return unavailable
        
    contamination = 1.0 - crowdsap
    
    # Evidence flag: CROWDSAP < 0.80 (means > 20% contamination in aperture)
    evidence_flag = crowdsap < 0.80
    
    return {
        "crowding_available": True,
        "crowdsap": round(crowdsap, 6),
        "flfrcsap": round(flfrcsap, 6) if flfrcsap is not None else None,
        "contamination_fraction": round(contamination, 6),
        "crowding_evidence_flag": bool(evidence_flag),
    }
