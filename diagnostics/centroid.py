# centroid.py
# -----------
# Centroid shift diagnostics: robust 2D in/out locations, covariance, and significance.

from __future__ import annotations
import logging
import numpy as np
from diagnostics.phase_windows import fold_phase, get_event_mask
from diagnostics.robust_statistics import robust_median, robust_mad, estimate_median_error, permutation_test_difference

logger = logging.getLogger(__name__)

def run_centroid_analysis(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch_btjd: float,
    duration_days: float,
    centroid_x: np.ndarray | None,
    centroid_y: np.ndarray | None,
    quality: np.ndarray | None,
    config: dict,
) -> dict:
    """
    Computes centroid shifts, covariances, and significances (Euclidean, Mahalanobis, Permutation).
    """
    c_config = config.get("Centroid", {})
    pixel_scale = c_config.get("pixel_scale_arcsec_per_pixel", 21.0)
    sig_threshold = c_config.get("shift_significance_threshold_sigma", 3.0)
    abs_shift_threshold = c_config.get("absolute_shift_threshold_arcsec", 2.0)
    min_points = c_config.get("minimum_in_out_centroid_points", 5)
    cov_reg = float(c_config.get("covariance_regularization", 1e-6))
    
    unavailable = {
        "centroid_available": False,
        "centroid_column_out": None,
        "centroid_row_out": None,
        "centroid_column_in": None,
        "centroid_row_in": None,
        "centroid_shift_column_pixels": None,
        "centroid_shift_row_pixels": None,
        "centroid_shift_pixels": None,
        "centroid_shift_arcsec": None,
        "centroid_shift_uncertainty_pixels": None,
        "centroid_shift_significance": None,
        "centroid_mahalanobis_distance": None,
        "centroid_permutation_p_value": None,
        "centroid_points_in": 0,
        "centroid_points_out": 0,
        "centroid_evidence_flag": False,
        "centroid_quality": "unavailable",
    }
    
    if centroid_x is None or centroid_y is None:
        return unavailable
        
    centroid_x = np.asarray(centroid_x, dtype=float)
    centroid_y = np.asarray(centroid_y, dtype=float)
    
    if len(centroid_x) != len(time) or len(centroid_y) != len(time):
        return unavailable
        
    if period <= 0 or duration_days <= 0:
        return unavailable
        
    # Build mask: finite arrays and quality == 0
    valid = np.isfinite(centroid_x) & np.isfinite(centroid_y) & np.isfinite(flux)
    if quality is not None:
        valid = valid & (quality == 0)
        
    if valid.sum() < min_points * 2:
        return unavailable
        
    # Phase fold
    phase = fold_phase(time, period, epoch_btjd)
    
    # Event masks
    in_transit_mask = valid & get_event_mask(time, period, epoch_btjd, duration_days, window_multiplier=1.0)
    out_transit_mask = valid & (get_event_mask(time, period, epoch_btjd, duration_days, window_multiplier=1.5) == False)
    
    n_in = int(in_transit_mask.sum())
    n_out = int(out_transit_mask.sum())
    
    if n_in < min_points or n_out < min_points * 2:
        return unavailable
        
    cx_in = centroid_x[in_transit_mask]
    cy_in = centroid_y[in_transit_mask]
    cx_out = centroid_x[out_transit_mask]
    cy_out = centroid_y[out_transit_mask]
    
    # Robust 2D centroids
    med_x_in = robust_median(cx_in)
    med_y_in = robust_median(cy_in)
    med_x_out = robust_median(cx_out)
    med_y_out = robust_median(cy_out)
    
    # Shifts
    dx = med_x_in - med_x_out
    dy = med_y_in - med_y_out
    shift_pixels = float(np.sqrt(dx**2 + dy**2))
    shift_arcsec = shift_pixels * pixel_scale
    
    # Uncertainties (standard error of median)
    err_x_out = estimate_median_error(cx_out)
    err_y_out = estimate_median_error(cy_out)
    
    # Combined pixel error
    err_combined = float(np.sqrt(err_x_out**2 + err_y_out**2))
    err_combined = max(1e-6, err_combined)
    
    # Euclidean significance
    significance = shift_pixels / err_combined
    
    # Out-of-transit Covariance matrix for Mahalanobis distance
    cov_x = np.var(cx_out, ddof=1) if len(cx_out) > 1 else cov_reg
    cov_y = np.var(cy_out, ddof=1) if len(cy_out) > 1 else cov_reg
    cov_xy = np.cov(cx_out, cy_out)[0, 1] if len(cx_out) > 1 else 0.0
    
    cov_matrix = np.array([[cov_x, cov_xy], [cov_xy, cov_y]])
    # Regularize to prevent singular matrix
    cov_matrix += np.eye(2) * cov_reg
    
    # Mahalanobis distance: d = sqrt( delta^T Cov^-1 delta )
    delta = np.array([dx, dy])
    try:
        inv_cov = np.linalg.inv(cov_matrix)
        mahalanobis_dist = float(np.sqrt(np.dot(delta.T, np.dot(inv_cov, delta))))
    except Exception:
        mahalanobis_dist = 0.0
        
    # Permutation test p-values (average of columns & row permutation)
    p_x = permutation_test_difference(cx_in, cx_out, n_permutations=200)
    p_y = permutation_test_difference(cy_in, cy_out, n_permutations=200)
    p_val = float(min(p_x, p_y))
    
    # Evidence flag: both statistical significance and absolute shift threshold exceeded
    evidence_flag = (significance >= sig_threshold) and (shift_arcsec >= abs_shift_threshold)
    
    quality = "on_target"
    if evidence_flag:
        quality = "off_target_shift"
        
    return {
        "centroid_available": True,
        "centroid_column_out": round(med_x_out, 6),
        "centroid_row_out": round(med_y_out, 6),
        "centroid_column_in": round(med_x_in, 6),
        "centroid_row_in": round(med_y_in, 6),
        "centroid_shift_column_pixels": round(dx, 6),
        "centroid_shift_row_pixels": round(dy, 6),
        "centroid_shift_pixels": round(shift_pixels, 6),
        "centroid_shift_arcsec": round(shift_arcsec, 4),
        "centroid_shift_uncertainty_pixels": round(err_combined, 6),
        "centroid_shift_significance": round(significance, 4),
        "centroid_mahalanobis_distance": round(mahalanobis_dist, 4),
        "centroid_permutation_p_value": round(p_val, 4),
        "centroid_points_in": int(n_in),
        "centroid_points_out": int(n_out),
        "centroid_evidence_flag": bool(evidence_flag),
        "centroid_quality": quality,
    }
