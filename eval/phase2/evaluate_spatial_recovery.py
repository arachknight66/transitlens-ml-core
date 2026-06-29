# evaluate_spatial_recovery.py
# ---------------------------
# Evaluates spatial offset recovery and localization errors using pixel injection metrics.

from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def evaluate_spatial_recovery_rates(
    df_injections: pd.DataFrame,
) -> dict:
    """
    Computes localization error statistics (mean, median offset error) for synthetic pixel injections.
    """
    # Expected columns in df_injections:
    # true_offset_pixels, localized_offset_pixels, diff_image_snr, depth, target_magnitude
    if df_injections.empty or "true_offset_pixels" not in df_injections.columns:
        return {"mean_localization_error_pixels": 0.0, "median_localization_error_pixels": 0.0, "support": 0}
        
    # Error = abs(true_offset_pixels - localized_offset_pixels)
    err = np.abs(df_injections["true_offset_pixels"] - df_injections["localized_offset_pixels"])
    
    mean_err = float(np.mean(err))
    med_err = float(np.median(err))
    
    # Stratify by SNR
    snr_bins = [0, 5, 10, 20, 50, 1000]
    snr_recovery = {}
    for i in range(len(snr_bins)-1):
        low, high = snr_bins[i], snr_bins[i+1]
        bin_df = df_injections[(df_injections["diff_image_snr"] >= low) & (df_injections["diff_image_snr"] < high)]
        if len(bin_df) > 0:
            bin_err = np.abs(bin_df["true_offset_pixels"] - bin_df["localized_offset_pixels"])
            snr_recovery[f"SNR_{low}_to_{high}"] = {
                "mean_error_pixels": round(float(np.mean(bin_err)), 4),
                "median_error_pixels": round(float(np.median(bin_err)), 4),
                "count": int(len(bin_df)),
            }
            
    return {
        "mean_localization_error_pixels": round(mean_err, 4),
        "median_localization_error_pixels": round(med_err, 4),
        "snr_recovery_stratified": snr_recovery,
        "support": int(len(df_injections)),
    }
