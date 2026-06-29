# multi_aperture.py
# -----------------
# Multi-aperture depth-consistency: nesting rules, nested photometry, and depth trends.

from __future__ import annotations
import logging
import numpy as np
from scipy.stats import chi2
from diagnostics.phase_windows import get_event_mask
from diagnostics.robust_statistics import robust_median, estimate_median_error

logger = logging.getLogger(__name__)

def run_multi_aperture_analysis(
    tpf_path: str | Path | None,
    period: float,
    epoch_btjd: float,
    duration_days: float,
    config: dict,
) -> dict:
    """
    Measures transit depth consistency across nested apertures to detect off-target events.
    """
    unavailable = {
        "multi_aperture_available": False,
        "aperture_count": 0,
        "aperture_pixel_counts": [],
        "aperture_depths": [],
        "aperture_depth_uncertainties": [],
        "aperture_depth_slope": None,
        "aperture_depth_slope_uncertainty": None,
        "aperture_depth_consistency_chi2": None,
        "aperture_depth_consistency_p_value": None,
        "multi_aperture_evidence_flag": False,
        "multi_aperture_quality": "unavailable",
        "aperture_diagnostic_plot_path": "",
    }
    
    if tpf_path is None:
        return unavailable
        
    from pathlib import Path
    tpf_path = Path(tpf_path)
    if not tpf_path.exists():
        return unavailable
        
    try:
        from astropy.io import fits
        with fits.open(tpf_path, memmap=False) as hdul:
            if len(hdul) < 3:
                return unavailable
                
            tpf_data = hdul[1].data
            aperture_mask = hdul[2].data
            
            time = np.array(tpf_data["TIME"], dtype=np.float64)
            flux_cube = np.array(tpf_data["FLUX"], dtype=np.float64)
            flux_err_cube = np.array(tpf_data["FLUX_ERR"], dtype=np.float64)
            quality = np.array(tpf_data["QUALITY"], dtype=np.int64)
            
            # Mask valid points
            valid = np.isfinite(time) & (quality == 0)
            for i in range(len(time)):
                if not np.all(np.isfinite(flux_cube[i, :, :])):
                    valid[i] = False
                    
            if valid.sum() < 20:
                return unavailable
                
            # Define 3 apertures: Small, Nominal, Expanded
            # 1. Nominal = pixels in TPF aperture mask with bit value >= 2 (which means used in photometry)
            nominal_mask = (aperture_mask & 2) > 0
            if nominal_mask.sum() == 0:
                # Fallback: any bit > 0
                nominal_mask = aperture_mask > 0
                
            # 2. Small = a subset of nominal mask (e.g. center pixels)
            # Find center of nominal aperture mask
            y_indices, x_indices = np.where(nominal_mask)
            if len(x_indices) == 0:
                return unavailable
            center_x = int(np.median(x_indices))
            center_y = int(np.median(y_indices))
            
            # Small is a 1-pixel radius around center
            rows, cols = nominal_mask.shape
            y_grid, x_grid = np.mgrid[0:rows, 0:cols]
            small_mask = nominal_mask & (np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2) <= 1.0)
            if small_mask.sum() == 0:
                small_mask = nominal_mask.copy()
                
            # 3. Expanded = nominal mask expanded by 1 pixel boundary
            expanded_mask = nominal_mask.copy()
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    expanded_mask = expanded_mask | np.roll(np.roll(nominal_mask, dy, axis=0), dx, axis=1)
            # Ensure it fits within crop
            expanded_mask = expanded_mask & (aperture_mask >= 0)
            
            aperture_masks = [small_mask, nominal_mask, expanded_mask]
            aperture_names = ["small", "nominal", "expanded"]
            pixel_counts = [int(mask.sum()) for mask in aperture_masks]
            
            in_transit = get_event_mask(time, period, epoch_btjd, duration_days, window_multiplier=1.0)
            out_transit = get_event_mask(time, period, epoch_btjd, duration_days, window_multiplier=1.5) == False
            
            in_transit_mask = valid & in_transit
            out_transit_mask = valid & out_transit
            
            if in_transit_mask.sum() < 3 or out_transit_mask.sum() < 5:
                return unavailable
                
            depths = []
            depth_errs = []
            
            # Perform photometry on each aperture
            for mask in aperture_masks:
                # Sum pixels within mask for each cadence
                lightcurve = np.sum(flux_cube[:, mask], axis=1)
                
                # Normalize
                med_oot = robust_median(lightcurve[out_transit_mask])
                if med_oot <= 0:
                    depths.append(0.0)
                    depth_errs.append(0.001)
                    continue
                    
                lightcurve_norm = lightcurve / med_oot
                
                # Fit depth: oot_median - in_transit_median
                in_med = robust_median(lightcurve_norm[in_transit_mask])
                depth = float(1.0 - in_med)
                
                err = estimate_median_error(lightcurve_norm[in_transit_mask])
                
                depths.append(max(0.0, depth))
                depth_errs.append(max(1e-6, err))
                
            # Fit depth slope vs pixel count
            # depth = a * pixel_count + b
            x = np.array(pixel_counts, dtype=float)
            y = np.array(depths, dtype=float)
            w = 1.0 / np.array(depth_errs, dtype=float)**2
            
            # Weighted linear regression
            poly = np.polyfit(x, y, 1, w=w, cov=True)
            slope = float(poly[0][0])
            slope_err = float(np.sqrt(poly[1][0][0]))
            
            # Chi-square consistency: check if all depths are equal
            # H0: depths are equal (weighted mean depth)
            w_mean = np.sum(y * w) / np.sum(w)
            chi2_stat = float(np.sum(((y - w_mean) / np.array(depth_errs))**2))
            
            # Degrees of freedom = n - 1 = 2
            dof = len(depths) - 1
            p_val = float(1.0 - chi2.cdf(chi2_stat, dof))
            
            # Evidence flag: p-value < 0.05 (significant difference)
            p_threshold = config.get("MultiAperture", {}).get("consistency_significance_p_value", 0.05)
            evidence_flag = p_val < p_threshold
            
            quality = "consistent"
            if evidence_flag:
                quality = "inconsistent_apertures"
                
            return {
                "multi_aperture_available": True,
                "aperture_count": 3,
                "aperture_pixel_counts": pixel_counts,
                "aperture_depths": [round(d, 6) for d in depths],
                "aperture_depth_uncertainties": [round(de, 6) for de in depth_errs],
                "aperture_depth_slope": round(slope, 8),
                "aperture_depth_slope_uncertainty": round(slope_err, 8),
                "aperture_depth_consistency_chi2": round(chi2_stat, 4),
                "aperture_depth_consistency_p_value": round(p_val, 4),
                "multi_aperture_evidence_flag": bool(evidence_flag),
                "multi_aperture_quality": quality,
                "aperture_diagnostic_plot_path": "",
            }
            
    except Exception as exc:
        logger.warning(f"Multi-aperture analysis failed: {exc}")
        return unavailable
