# robust_statistics.py
# ------------------
# Robust statistical estimators, standard errors, bootstraps, and permutations.

from __future__ import annotations
import numpy as np

def robust_median(x: np.ndarray) -> float:
    """Computes the median of an array, ignoring NaNs."""
    if len(x) == 0:
        return 0.0
    return float(np.nanmedian(x))

def robust_mad(x: np.ndarray) -> float:
    """Computes Median Absolute Deviation (MAD), normalized to match Gaussian scale."""
    if len(x) <= 1:
        return 0.0
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return float(1.4826 * mad)

def robust_mean(x: np.ndarray) -> float:
    """Computes median-clipped mean to reject extreme outliers."""
    if len(x) == 0:
        return 0.0
    med = np.nanmedian(x)
    mad = robust_mad(x)
    if mad == 0:
        return float(np.nanmean(x))
    # Clip at 5 sigma
    clipped = x[np.abs(x - med) <= 5.0 * mad]
    if len(clipped) == 0:
        return float(med)
    return float(np.nanmean(clipped))

def estimate_median_error(x: np.ndarray) -> float:
    """
    Standard error of median: 1.2533 * std / sqrt(N).
    Uses robust MAD as the std estimate.
    """
    n = len(x)
    if n <= 1:
        return 0.0
    mad = robust_mad(x)
    return float(1.2533 * mad / np.sqrt(n))

def bootstrap_median_ci(
    x: np.ndarray,
    n_resamples: int = 200,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Computes the confidence interval of the median using bootstrap resampling."""
    n = len(x)
    if n <= 2:
        return (float(x.min()) if n > 0 else 0.0, float(x.max()) if n > 0 else 0.0)
    
    medians = []
    # Seed generator for reproducibility
    rng = np.random.default_rng(42)
    for _ in range(n_resamples):
        sample = rng.choice(x, size=n, replace=True)
        medians.append(np.median(sample))
        
    alpha = 1.0 - confidence_level
    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)
    
    return float(np.percentile(medians, lower_pct)), float(np.percentile(medians, upper_pct))

def permutation_test_difference(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_permutations: int = 500,
) -> float:
    """
    Calculates the two-tailed p-value for the difference in medians between two groups
    using a permutation test.
    """
    n_a = len(group_a)
    n_b = len(group_b)
    if n_a == 0 or n_b == 0:
        return 1.0
        
    obs_diff = abs(np.median(group_a) - np.median(group_b))
    combined = np.concatenate([group_a, group_b])
    
    rng = np.random.default_rng(42)
    extreme_count = 0
    
    for _ in range(n_permutations):
        shuffled = rng.permutation(combined)
        perm_a = shuffled[:n_a]
        perm_b = shuffled[n_a:]
        perm_diff = abs(np.median(perm_a) - np.median(perm_b))
        if perm_diff >= obs_diff:
            extreme_count += 1
            
    return float(extreme_count / n_permutations)
