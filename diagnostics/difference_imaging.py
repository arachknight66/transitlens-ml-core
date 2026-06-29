# difference_imaging.py
# --------------------
# Event difference imaging: pixel-level stacking, difference maps, and SNR calculation.

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from astropy.io import fits
from diagnostics.phase_windows import get_event_mask

logger = logging.getLogger(__name__)

def run_difference_imaging(
    tpf_path: str | Path | None,
    period: float,
    epoch_btjd: float,
    duration_days: float,
    config: dict,
) -> dict:
    """
    Computes in-transit and out-of-transit stacked pixel images from a TPF, and returns their difference.
    """
    di_config = config.get("DifferenceImaging", {})
    min_snr = di_config.get("minimum_difference_image_snr", 4.0)
    bg_strategy = di_config.get("background_strategy", "local_median")
    stack_method = di_config.get("image_stacking_method", "robust_mean")
    
    unavailable = {
        "difference_image_available": False,
        "difference_image_snr": None,
        "in_transit_stack": None,
        "out_of_transit_stack": None,
        "difference_image": None,
        "difference_image_uncertainty": None,
        "difference_image_quality": "unavailable",
        "aperture_mask": None,
        "target_column": None,
        "target_row": None,
    }
    
    if tpf_path is None:
        return unavailable
        
    tpf_path = Path(tpf_path)
    if not tpf_path.exists():
        logger.debug(f"TPF file not found at {tpf_path}")
        return unavailable
        
    try:
        with fits.open(tpf_path, memmap=False) as hdul:
            if len(hdul) < 3:
                return unavailable
                
            primary_hdr = hdul[0].header
            tpf_data = hdul[1].data
            aperture_mask = hdul[2].data # 2 = aperture mask extension
            
            # Extract cubes and vectors
            time_btjd = np.array(tpf_data["TIME"], dtype=np.float64)
            flux_cube = np.array(tpf_data["FLUX"], dtype=np.float64)
            flux_err_cube = np.array(tpf_data["FLUX_ERR"], dtype=np.float64)
            quality = np.array(tpf_data["QUALITY"], dtype=np.int64)
            
            # Target nominal coordinates
            target_col = float(primary_hdr.get("1CRPX4", 0.0)) # WCS col
            target_row = float(primary_hdr.get("2CRPX4", 0.0)) # WCS row
            if target_col == 0.0:
                # Try fallback column/row center of aperture
                target_col = float(flux_cube.shape[2] / 2.0)
                target_row = float(flux_cube.shape[1] / 2.0)
                
            # Edge pixels in SPOC TPF cutouts are commonly NaN. A frame is
            # usable when the science/aperture pixels are mostly finite; do not
            # reject the whole cadence because an irrelevant edge pixel is NaN.
            valid_frames = np.isfinite(time_btjd) & (quality == 0)
            science_mask = aperture_mask > 0
            if not science_mask.any():
                science_mask = np.any(np.isfinite(flux_cube), axis=0)
            valid_cube_mask = np.mean(np.isfinite(flux_cube[:, science_mask]), axis=1) >= 0.8
            valid_frames = valid_frames & valid_cube_mask
            
            if valid_frames.sum() < 10:
                return unavailable
                
            # Event masks
            in_transit_mask = valid_frames & get_event_mask(time_btjd, period, epoch_btjd, duration_days, window_multiplier=1.0)
            out_transit_mask = valid_frames & (get_event_mask(time_btjd, period, epoch_btjd, duration_days, window_multiplier=1.5) == False)
            
            if in_transit_mask.sum() < 3 or out_transit_mask.sum() < 5:
                return unavailable
                
            in_stack_cube = flux_cube[in_transit_mask]
            out_stack_cube = flux_cube[out_transit_mask]
            
            # Perform Stacking (robust mean or median)
            if stack_method == "median":
                in_stack = np.nanmedian(in_stack_cube, axis=0)
                out_stack = np.nanmedian(out_stack_cube, axis=0)
            else: # robust mean (default)
                in_stack = np.nanmean(in_stack_cube, axis=0)
                out_stack = np.nanmean(out_stack_cube, axis=0)
                
            # Signed Difference: Out-of-Transit - In-Transit (representing the flux deficit)
            diff_img = out_stack - in_stack
            
            # Propagate uncertainties
            in_err_cube = flux_err_cube[in_transit_mask]
            out_err_cube = flux_err_cube[out_transit_mask]
            
            # Combined variance: var = sum(var_i) / N^2
            in_n = np.maximum(np.sum(np.isfinite(in_err_cube), axis=0), 1)
            out_n = np.maximum(np.sum(np.isfinite(out_err_cube), axis=0), 1)
            in_var = np.nansum(in_err_cube**2, axis=0) / (in_n**2)
            out_var = np.nansum(out_err_cube**2, axis=0) / (out_n**2)
            diff_err = np.sqrt(in_var + out_var)
            
            # Regularize error to prevent div by zero
            diff_err = np.clip(diff_err, 1e-5, None)
            
            # SNR image
            snr_img = diff_img / diff_err
            max_snr = float(np.nanmax(snr_img[science_mask]))
            
            quality_flag = "nominal"
            if max_snr < min_snr:
                quality_flag = "low_snr"
                
            return {
                "difference_image_available": True,
                "difference_image_snr": round(max_snr, 4),
                "in_transit_stack": in_stack,
                "out_of_transit_stack": out_stack,
                "difference_image": diff_img,
                "difference_image_uncertainty": diff_err,
                "difference_image_quality": quality_flag,
                "aperture_mask": aperture_mask,
                "target_column": target_col,
                "target_row": target_row,
            }
            
    except Exception as exc:
        logger.warning(f"Difference imaging processing failed: {exc}")
        return unavailable
