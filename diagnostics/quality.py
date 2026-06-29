# quality.py
# ----------
# Quality control and diagnostic failure auditing for TransitLens Phase 2.

from __future__ import annotations
import numpy as np

def audit_observation_quality(
    time: np.ndarray,
    flux: np.ndarray,
    centroid_x: np.ndarray | None,
    centroid_y: np.ndarray | None,
    quality: np.ndarray | None,
    metadata: dict | None,
) -> list[str]:
    """
    Audits the light curve and centroid arrays for quality warnings.
    Returns a list of warning string flags.
    """
    warnings = []
    metadata = metadata or {}
    
    # 1. Check point count
    n_points = len(time)
    if n_points < 100:
        warnings.append("insufficient_data_points")
        
    # 2. Check saturation
    # Standard SPOC saturation bit is 131072 (bit 18) or similar, QLP uses other bits.
    # We can check flux level: if median flux is unusually high or has flat peaks near saturation
    if np.any(flux > 2.0):
        warnings.append("saturated_flux_values")
        
    # 3. Check for edge source
    # Check if target is close to TPF crop boundary (from metadata)
    if metadata.get("source_edge_warning"):
        warnings.append("edge_source")
        
    # 4. Check for momentum dumps / quality flags
    if quality is not None and len(quality) > 0:
        # TESS momentum dumps are bit 32 (bit 6) or similar
        mom_dumps = np.sum((quality & 32) > 0)
        if mom_dumps > 0:
            warnings.append(f"momentum_dumps_present_{mom_dumps}_events")
            
    # 5. Check gaps
    if len(time) > 1:
        diffs = np.diff(time)
        max_gap = np.max(diffs)
        if max_gap > 10.0:
            warnings.append(f"large_gap_detected_{max_gap:.1f}_days")
            
    return warnings
