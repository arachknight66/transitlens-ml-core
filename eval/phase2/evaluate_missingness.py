# evaluate_missingness.py
# ---------------------
# Audits availability rates and missingness statistics across diagnostics.

from __future__ import annotations
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def evaluate_missingness_rates(
    df_features: pd.DataFrame,
) -> dict:
    """
    Computes diagnostic availability rates across the evaluation dataset.
    """
    if df_features.empty:
        return {}
        
    n_total = len(df_features)
    
    # Calculate availability fraction for columns
    # In features, missing values are NaNs
    availability = {}
    
    check_cols = {
        "centroid_available": "centroid_shift_pixels",
        "difference_image_available": "source_target_offset_pixels",
        "gaia_available": "gaia_neighbor_count",
        "crowding_available": "crowdsap",
        "multi_aperture_available": "aperture_depth_slope"
    }
    
    for flag_name, col_name in check_cols.items():
        if col_name in df_features.columns:
            # count non-nulls
            n_avail = int(df_features[col_name].notnull().sum())
            availability[flag_name] = {
                "count": n_avail,
                "fraction": round(float(n_avail / n_total), 4),
            }
        else:
            availability[flag_name] = {
                "count": 0,
                "fraction": 0.0,
            }
            
    return {
        "total_targets": n_total,
        "availability": availability,
    }
