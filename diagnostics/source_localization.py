# source_localization.py
# -------------------
# Source localization: pinpointing transit source from difference image, target offset.

from __future__ import annotations
import logging
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """2D Gaussian function for profile fitting."""
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def run_source_localization(
    diff_imaging_res: dict,
    config: dict,
) -> dict:
    """
    Fits a 2D Gaussian/profile model to the difference image to estimate the source position and target offset.
    """
    unavailable = {
        "difference_source_column": None,
        "difference_source_row": None,
        "source_target_offset_pixels": None,
        "source_target_offset_arcsec": None,
        "source_target_offset_uncertainty_pixels": None,
        "source_target_offset_significance": None,
        "difference_flux": None,
        "difference_flux_uncertainty": None,
        "difference_image_evidence_flag": False,
        "difference_image_path": "",
        "difference_image_plot_path": "",
    }
    
    if not diff_imaging_res.get("difference_image_available"):
        return unavailable
        
    diff_image = diff_imaging_res["difference_image"]
    diff_err = diff_imaging_res["difference_image_uncertainty"]
    target_col = diff_imaging_res["target_column"]
    target_row = diff_imaging_res["target_row"]
    
    di_config = config.get("DifferenceImaging", {})
    pixel_scale = config.get("Centroid", {}).get("pixel_scale_arcsec_per_pixel", 21.0)
    sig_threshold = di_config.get("acceptable_target_offset_pixels", 1.0) # threshold for flag
    
    rows, cols = diff_image.shape
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    
    # ── Method 1: Flux-Weighted Centroid ──
    # Focus only on positive peaks (flux deficit is positive in our diff_image)
    pos_diff = np.clip(diff_image, 0.0, None)
    total_flux = np.sum(pos_diff)
    
    if total_flux <= 0:
        return unavailable
        
    centroid_col = float(np.sum(pos_diff * x_grid) / total_flux)
    centroid_row = float(np.sum(pos_diff * y_grid) / total_flux)
    
    # ── Method 2: 2D Gaussian Fit ──
    # Let's perform a simple 2D Gaussian fit to refine the coordinates
    try:
        # Initial guesses: max peak position
        peak_idx = np.argmax(diff_image)
        peak_row, peak_col = np.unravel_index(peak_idx, diff_image.shape)
        
        init_amp = float(diff_image[peak_row, peak_col])
        init_offset = float(np.median(diff_image))
        
        # Fit coordinates
        xy_grid = (x_grid, y_grid)
        
        def objective(params):
            amp, xo, yo, sig_x, sig_y, theta, offset = params
            if sig_x <= 0.1 or sig_y <= 0.1 or sig_x > cols or sig_y > rows:
                return 1e9
            model = gaussian_2d(xy_grid, amp, xo, yo, sig_x, sig_y, theta, offset)
            return float(np.sum((diff_image.ravel() - model)**2))
            
        res = minimize(
            objective,
            x0=[init_amp, float(peak_col), float(peak_row), 1.0, 1.0, 0.0, init_offset],
            method="Nelder-Mead",
            options={"maxiter": 200},
        )
        
        if res.success:
            fit_col = float(res.x[1])
            fit_row = float(res.x[2])
            fit_amp = float(res.x[0])
        else:
            fit_col = centroid_col
            fit_row = centroid_row
            fit_amp = total_flux
    except Exception:
        fit_col = centroid_col
        fit_row = centroid_row
        fit_amp = total_flux
        
    # Offset from target
    # target_col/row are in absolute coordinate space. The difference image grids are local pixel index spaces.
    # We must match the absolute pixel coordinates. TESS FITS TPFs usually specify absolute coordinates
    # in columns like RAWX/RAWY or in WCS keywords CRVAL/CRPIX.
    # For simplification, within the cutout region: offset_col = localized_col - target_local_col
    # We define target coordinates in local pixel grid as target_col, target_row.
    # Note: TPF header 1CRPX4, 2CRPX4 can be used if they map to the cutout center.
    offset_col = fit_col - target_col
    offset_row = fit_row - target_row
    
    offset_pixels = float(np.sqrt(offset_col**2 + offset_row**2))
    offset_arcsec = offset_pixels * pixel_scale
    
    # Calculate uncertainty from noise map
    # uncertainty ≈ PSF_width / SNR
    snr = diff_imaging_res["difference_image_snr"]
    psf_width = 1.0 # typical pixel width
    uncertainty_pixels = float(psf_width / snr) if snr > 0 else 0.5
    
    significance = offset_pixels / uncertainty_pixels if uncertainty_pixels > 0 else 0.0
    
    # Evidence flag: offset exceeds target limit in pixels (e.g. 1.0 pixel)
    evidence_flag = offset_pixels >= sig_threshold
    
    return {
        "difference_source_column": round(fit_col, 4),
        "difference_source_row": round(fit_row, 4),
        "source_target_offset_pixels": round(offset_pixels, 4),
        "source_target_offset_arcsec": round(offset_arcsec, 4),
        "source_target_offset_uncertainty_pixels": round(uncertainty_pixels, 4),
        "source_target_offset_significance": round(significance, 4),
        "difference_flux": round(fit_amp, 6),
        "difference_flux_uncertainty": round(float(np.mean(diff_err)), 6),
        "difference_image_evidence_flag": bool(evidence_flag),
        "difference_image_path": "",
        "difference_image_plot_path": "",
    }
