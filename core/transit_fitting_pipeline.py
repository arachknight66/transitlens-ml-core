"""
core/transit_fitting_pipeline.py
--------------------------------
Phase 7 Scientific Transit Fitting, Parameter Estimation, and Uncertainty Quantification.

Implements:
1. Pure-Python physical circular-orbit transit model with quadratic limb darkening (vectorized Gauss-Legendre).
2. Stage A deterministic fitting (trapezoid and physical model).
3. Period alias resolution (P/2, P, 2P, odd-even depths, secondary eclipse check).
4. Stage B MCMC sampling (emcee) with priors and convergence metrics.
5. Red-noise diagnostics and uncertainty inflation.
"""

from __future__ import annotations
import logging
import numpy as np
from scipy.optimize import minimize
from core.utils import phase_fold, bin_phase_folded
from core.exceptions import MLCoreError

logger = logging.getLogger(__name__)

# Gauss-Legendre quadrature points and weights for 15-point integration in [-1, 1]
_X_GL, _W_GL = np.polynomial.legendre.leggauss(15)

# ---------------------------------------------------------------------------
# 1. Physics Occultation Equations (Gauss-Legendre 1D Integration)
# ---------------------------------------------------------------------------

def physical_limb_darkened_flux(z_arr: np.ndarray, p: float, u1: float, u2: float) -> np.ndarray:
    """
    Computes normalized transit flux for projected center-to-center distance z_arr.
    Handles any radius ratio p (0 to 1), impact parameter, and quadratic limb darkening u1, u2.
    Uses 1D Gauss-Legendre quadrature to integrate intensity over the occulted stellar area.
    """
    # Ensure inputs are physical
    p = float(p)
    u1 = float(u1)
    u2 = float(u2)
    
    flux_ratio = np.ones_like(z_arr, dtype=float)
    
    # Mask for points where planet overlaps the stellar disk
    mask = z_arr < (1.0 + p)
    if not np.any(mask):
        return flux_ratio
        
    z_sub = z_arr[mask]
    
    # Integration limits for each time point: [r_min, r_max]
    # where r is the radial distance from the stellar center
    r_min = np.maximum(0.0, z_sub - p)
    r_max = np.minimum(1.0, z_sub + p)
    
    # Set up Gauss-Legendre evaluation grid
    # Mapped from [-1, 1] to [r_min, r_max]
    rmin_col = r_min[:, np.newaxis]
    rmax_col = r_max[:, np.newaxis]
    
    # Shape: (len(z_sub), 15)
    r = 0.5 * (rmax_col - rmin_col) * _X_GL + 0.5 * (rmax_col + rmin_col)
    
    # Compute dtheta (width of occulted arc at stellar radius r)
    z_col = z_sub[:, np.newaxis]
    
    # If r < p - z, the entire ring at radius r is occulted (dtheta = 2*pi)
    full_occ = r < (p - z_col)
    
    # Avoid division by zero when z is exactly 0
    denom = 2.0 * r * z_col
    denom = np.where(denom < 1e-12, 1e-12, denom)
    
    arg = (r**2 + z_col**2 - p**2) / denom
    arg = np.clip(arg, -1.0, 1.0)
    
    dtheta = np.where(full_occ, 2.0 * np.pi, 2.0 * np.arccos(arg))
    
    # Quadratic limb darkening profile
    mu = np.sqrt(np.clip(1.0 - r**2, 0.0, 1.0))
    I = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu)**2
    
    # Integrand = Intensity * radius * dtheta
    integrand = I * r * dtheta
    
    # Evaluate integral: sum along GL axis with weights
    integral = np.sum(integrand * _W_GL, axis=1) * 0.5 * (r_max - r_min)
    
    # Total stellar flux normalization
    F0 = np.pi * (1.0 - u1 / 3.0 - u2 / 6.0)
    F0 = max(1e-12, F0)
    
    flux_ratio[mask] = 1.0 - integral / F0
    return flux_ratio


def physical_transit_model(
    time: np.ndarray,
    period: float,
    t0: float,
    rp_rstar: float,
    a_rstar: float,
    b: float,
    u1: float,
    u2: float,
    baseline: float = 1.0,
    slope: float = 0.0,
) -> np.ndarray:
    """
    Computes a circular-orbit physical transit light curve.
    
    Parameters
    ----------
    time : np.ndarray
        Time array (e.g. BTJD).
    period, t0 : float
        Orbital period (days) and mid-transit epoch.
    rp_rstar : float
        Radius ratio Rp/Rstar.
    a_rstar : float
        Scaled semi-major axis a/Rstar.
    b : float
        Impact parameter (b = a/Rstar * cos(i)).
    u1, u2 : float
        Quadratic limb darkening coefficients.
    baseline : float
        Out-of-transit flux level (typically ~1.0).
    slope : float
        Local linear trend coefficient.
    """
    # Enforce physical constraints on parameters
    period = max(1e-4, period)
    rp_rstar = np.clip(rp_rstar, 0.0, 1.0)
    a_rstar = max(1.01, a_rstar)
    b = np.clip(b, 0.0, a_rstar)
    
    # Phase fold around mid-transit epoch t0
    phase = phase_fold(time, period, t0) # Output is centered on 0, range [-0.5, 0.5]
    theta = phase * 2.0 * np.pi
    
    z_sq = (a_rstar * np.sin(theta))**2 + (b * np.cos(theta))**2
    z = np.sqrt(np.maximum(0.0, z_sq))
    z = np.where(np.cos(theta) > 0.0, z, 99.0)
    
    # Occulted flux
    flux_ratio = physical_limb_darkened_flux(z, rp_rstar, u1, u2)
    
    # Apply baseline and trend (slope relative to average time)
    t_mid = np.mean(time) if len(time) > 0 else 0.0
    trend = baseline + slope * (time - t_mid)
    
    return flux_ratio * trend


def trapezoid_transit_model(
    time: np.ndarray,
    period: float,
    t0: float,
    depth: float,
    duration: float,
    ingress_ratio: float,
    baseline: float = 1.0,
    slope: float = 0.0,
) -> np.ndarray:
    """
    Analytical trapezoidal transit model.
    ingress_ratio is the ratio of ingress duration to total duration (0.0 to 0.5).
    """
    if period <= 0.0 or duration <= 0.0 or depth <= 0.0:
        return np.ones_like(time) * baseline
        
    phase = phase_fold(time, period, t0)
    dt = np.abs(phase * period)
    
    ingress_duration = duration * ingress_ratio
    half_dur = duration / 2.0
    
    flux = np.ones_like(time, dtype=float)
    
    # Out of transit
    out_mask = dt >= half_dur
    # Full transit flat bottom
    flat_mask = dt <= (half_dur - ingress_duration)
    # Ingress/egress
    slope_mask = (~out_mask) & (~flat_mask)
    
    flux[flat_mask] = 1.0 - depth
    
    if ingress_duration > 1e-6:
        fraction = (half_dur - dt[slope_mask]) / ingress_duration
        flux[slope_mask] = 1.0 - depth * fraction
        
    t_mid = np.mean(time) if len(time) > 0 else 0.0
    trend = baseline + slope * (time - t_mid)
    return flux * trend


# ---------------------------------------------------------------------------
# 2. Stage A Bounded Deterministic Optimizer
# ---------------------------------------------------------------------------

def fit_trapezoid_deterministic(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    init_period: float,
    init_t0: float,
    init_depth: float,
    init_duration: float,
) -> dict:
    """
    Refines parameters using the trapezoidal model via SciPy minimize.
    Fits: period, t0, depth, duration, ingress_ratio, baseline, slope.
    """
    # ── 1. Restrict data to phase window around transit to speed up fitting ──
    phase = phase_fold(time, init_period, init_t0)
    keep = np.abs(phase * init_period) <= max(0.5, 3.0 * init_duration)
    
    # Ensure enough points are in the window
    if np.sum(keep) < 15:
        keep = np.ones_like(time, dtype=bool)
        
    t_fit = time[keep]
    f_fit = flux[keep]
    fe_fit = flux_err[keep]
    
    # Bounds: [period, t0, depth, duration, ingress_ratio, baseline, slope]
    # Restrict period refinement to prevent jumping to aliases at this stage
    p_tol = min(0.05 * init_period, 0.1)
    t0_tol = min(0.15 * init_period, 0.2)
    
    bounds = [
        (init_period - p_tol, init_period + p_tol),
        (init_t0 - t0_tol, init_t0 + t0_tol),
        (1e-5, 0.70), # depth max 70%
        (0.002, min(0.4 * init_period, 1.5)), # duration days
        (0.0, 0.5), # ingress ratio
        (0.95, 1.05), # baseline
        (-0.02, 0.02), # slope
    ]
    
    # Ingress default to 0.1
    p0 = [init_period, init_t0, init_depth, init_duration, 0.1, 1.0, 0.0]
    
    # Weighted chi-square objective
    def objective(params):
        p, t0, dep, dur, ing, base, slp = params
        model = trapezoid_transit_model(t_fit, p, t0, dep, dur, ing, base, slp)
        return np.sum(((f_fit - model) / fe_fit) ** 2)
        
    res = minimize(objective, x0=p0, bounds=bounds, method="L-BFGS-B")
    
    if res.success:
        p_fit, t0_fit, dep_fit, dur_fit, ing_fit, base_fit, slp_fit = res.x
        return {
            "period": float(p_fit),
            "t0": float(t0_fit),
            "depth": float(dep_fit),
            "duration": float(dur_fit),
            "ingress_ratio": float(ing_fit),
            "baseline": float(base_fit),
            "slope": float(slp_fit),
            "success": True,
            "fun": float(res.fun),
            "nfev": int(res.nfev),
        }
    else:
        return {
            "period": init_period,
            "t0": init_t0,
            "depth": init_depth,
            "duration": init_duration,
            "ingress_ratio": 0.1,
            "baseline": 1.0,
            "slope": 0.0,
            "success": False,
            "fun": float(objective(p0)),
            "nfev": 0,
        }


def fit_physical_deterministic(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    init_period: float,
    init_t0: float,
    init_depth: float,
    init_duration: float,
    stellar_density: float | None = None,
    u1_init: float = 0.4,
    u2_init: float = 0.3,
) -> dict:
    """
    Fits the physical circular-orbit model using SciPy minimize.
    Converts (depth, duration) guesses to (rp_rstar, a_rstar, b).
    """
    # ── 1. Restrict data to phase window ──
    phase = phase_fold(time, init_period, init_t0)
    keep = np.abs(phase * init_period) <= max(0.5, 3.0 * init_duration)
    if np.sum(keep) < 15:
        keep = np.ones_like(time, dtype=bool)
        
    t_fit = time[keep]
    f_fit = flux[keep]
    fe_fit = flux_err[keep]
    
    # ── 2. Parameter conversions ──
    # p ≈ sqrt(depth)
    p_guess = np.clip(np.sqrt(max(1e-5, init_depth)), 0.001, 0.6)
    
    # Initialize impact parameter b = 0.3 (common moderately central value)
    b_guess = 0.3
    
    # a/Rstar ≈ P / (pi * T14) * sqrt((1+p)^2 - b^2)
    # T14 duration limit checks
    dur = max(0.005, init_duration)
    a_guess = (init_period / (np.pi * dur)) * np.sqrt((1.0 + p_guess)**2 - b_guess**2)
    a_guess = max(1.1, a_guess)
    
    # Bounds: [period, t0, rp_rstar, a_rstar, b, u1, u2, baseline, slope]
    p_tol = min(0.05 * init_period, 0.1)
    t0_tol = min(0.15 * init_period, 0.2)
    
    bounds = [
        (init_period - p_tol, init_period + p_tol),
        (init_t0 - t0_tol, init_t0 + t0_tol),
        (0.0001, 0.8), # rp_rstar
        (1.05, 150.0), # a_rstar
        (0.0, 1.2), # b (can be grazing, but not too far out)
        (0.0, 1.0), # u1
        (-1.0, 1.0), # u2
        (0.95, 1.05), # baseline
        (-0.02, 0.02), # slope
    ]
    
    p0 = [init_period, init_t0, p_guess, a_guess, b_guess, u1_init, u2_init, 1.0, 0.0]
    
    # Objective function
    def objective(params):
        p, t0, rp, a_rs, b, u1, u2, base, slp = params
        # Geometry checks
        if b > (1.0 + rp):
            return 1e10 # unphysical geometry for a transit
        if b > a_rs:
            return 1e10
        # Physical constraints on quadratic limb darkening (Kipping 2013 boundary checks)
        if u1 + u2 > 1.0 or u1 < 0.0 or u1 + 2.0*u2 < 0.0:
            # We add a soft penalty or hard boundary
            return 1e10
            
        model = physical_transit_model(t_fit, p, t0, rp, a_rs, b, u1, u2, base, slp)
        return np.sum(((f_fit - model) / fe_fit) ** 2)
        
    res = minimize(objective, x0=p0, bounds=bounds, method="L-BFGS-B")
    
    if res.success:
        p_fit, t0_fit, rp_fit, a_fit, b_fit, u1_fit, u2_fit, base_fit, slp_fit = res.x
        return {
            "period": float(p_fit),
            "t0": float(t0_fit),
            "rp_rstar": float(rp_fit),
            "a_rstar": float(a_fit),
            "b": float(b_fit),
            "u1": float(u1_fit),
            "u2": float(u2_fit),
            "baseline": float(base_fit),
            "slope": float(slp_fit),
            "success": True,
            "fun": float(res.fun),
            "nfev": int(res.nfev),
        }
    else:
        return {
            "period": init_period,
            "t0": init_t0,
            "rp_rstar": p_guess,
            "a_rstar": a_guess,
            "b": b_guess,
            "u1": u1_init,
            "u2": u2_init,
            "baseline": 1.0,
            "slope": 0.0,
            "success": False,
            "fun": float(objective(p0)),
            "nfev": 0,
        }


# ---------------------------------------------------------------------------
# 3. Period Alias and Ephemeris Refinement
# ---------------------------------------------------------------------------

def resolve_period_alias_hypotheses(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    bls_period: float,
    bls_t0: float,
    bls_duration: float,
    bls_depth: float,
) -> dict:
    """
    Evaluates P, P/2, and 2P hypotheses to check for period alias failure modes.
    Refits parameters for each, computes BIC, and compares metrics.
    """
    hypotheses = {
        "P": bls_period,
        "half_P": bls_period / 2.0,
        "double_P": bls_period * 2.0,
    }
    
    results = {}
    
    # Pre-calculated local out-of-transit noise using robust successive differences
    local_noise = np.std(np.diff(flux)) / np.sqrt(2.0) if len(flux) > 1 else 0.001
    
    for name, p_val in hypotheses.items():
        # Fit trapezoid first for this period
        fit_t = fit_trapezoid_deterministic(
            time, flux, flux_err,
            init_period=p_val,
            init_t0=bls_t0,
            init_depth=bls_depth,
            init_duration=bls_duration,
        )
        
        # Calculate BIC
        # BIC = chi^2 + k * ln(N)
        chi2 = fit_t["fun"]
        k = 7 # fitted parameters
        n_points = len(time)
        bic = chi2 + k * np.log(n_points)
        
        # Count observed transits and phase coverage
        phase = phase_fold(time, fit_t["period"], fit_t["t0"])
        in_transit_mask = np.abs(phase * fit_t["period"]) <= (fit_t["duration"] / 2.0)
        in_transit_points = np.sum(in_transit_mask)
        
        # Estimate number of cycles covered
        cycles = np.round((time - fit_t["t0"]) / fit_t["period"])
        unique_cycles = np.unique(cycles[in_transit_mask])
        observed_events = len(unique_cycles)
        
        # Measure phase coverage fraction (fraction of phase bins near transit containing data)
        # Using 30 bins within +/- 1.5 * duration
        window_phase = (1.5 * fit_t["duration"]) / fit_t["period"]
        bins = np.linspace(-window_phase, window_phase, 30)
        hist, _ = np.histogram(phase, bins=bins)
        phase_cov = np.sum(hist > 0) / (len(bins) - 1) if len(bins) > 1 else 0.0
        
        results[name] = {
            "period": fit_t["period"],
            "t0": fit_t["t0"],
            "depth": fit_t["depth"],
            "duration": fit_t["duration"],
            "ingress_ratio": fit_t["ingress_ratio"],
            "chi2": chi2,
            "bic": bic,
            "observed_transits": observed_events,
            "in_transit_cadences": in_transit_points,
            "phase_coverage": phase_cov,
        }
        
    # Check odd/even depth differences at candidate period P
    p_cand = results["P"]["period"]
    t0_cand = results["P"]["t0"]
    dur_cand = results["P"]["duration"]
    
    phase_c = phase_fold(time, p_cand, t0_cand)
    cycles_c = np.round((time - t0_cand) / p_cand)
    half_dur = (dur_cand / p_cand) / 2.0
    
    in_transit_c = np.abs(phase_c) < half_dur
    odd_mask = (cycles_c % 2 == 1) & in_transit_c
    even_mask = (cycles_c % 2 == 0) & in_transit_c
    
    odd_flux = flux[odd_mask]
    even_flux = flux[even_mask]
    
    odd_even_delta = 0.0
    if len(odd_flux) > 5 and len(even_flux) > 5:
        med_odd = np.median(odd_flux)
        med_even = np.median(even_flux)
        odd_even_delta = float(abs(med_odd - med_even))
        
    # Check for secondary eclipse at phase = 0.5 of double_P
    p_double = results["double_P"]["period"]
    t0_double = results["double_P"]["t0"]
    dur_double = results["double_P"]["duration"]
    
    phase_d = phase_fold(time, p_double, t0_double)
    half_dur_d = (dur_double / p_double) / 2.0
    in_transit_d = np.abs(phase_d) < half_dur_d
    in_transit_sec = np.abs(np.abs(phase_d) - 0.5) < half_dur_d
    
    sec_flux = flux[in_transit_sec]
    out_mask = ~(in_transit_d | in_transit_sec)
    out_flux = flux[out_mask]
    
    sec_depth = 0.0
    if len(sec_flux) > 5 and len(out_flux) > 0:
        sec_depth = float(np.median(out_flux) - np.median(sec_flux))
        
    # Selection logic:
    # Default to BLS period P unless double_P or half_P are significantly better supported.
    preferred_key = "P"
    reason = "Standard BLS period has best fit power and coverage."
    alias_warning = False
    alias_type = "none"
    
    # 1. Significant odd-even delta at candidate period P -> double the period
    if odd_even_delta > 3.5 * local_noise and odd_even_delta > 0.0005:
        preferred_key = "double_P"
        reason = "Significant odd-even depth difference found; true period is double."
        alias_warning = True
        alias_type = "double_period_odd_even"
    # 2. Significant secondary eclipse at phase 0.5 of double_P, but only if it is
    # significantly shallower than the primary
    elif sec_depth > 3.5 * local_noise and sec_depth > 0.0005 and sec_depth < 0.85 * results["double_P"]["depth"]:
        preferred_key = "double_P"
        reason = "Significant secondary eclipse detected at phase 0.5; true period is double."
        alias_warning = True
        alias_type = "double_period_secondary"
    # 3. If half_P has a similar or better BIC AND has at least 2 observed transits, check if it's preferred
    elif results["half_P"]["bic"] < (results["P"]["bic"] - 10.0) and results["half_P"]["observed_transits"] >= 2:
        # Check if depth at half_P is still substantial
        if results["half_P"]["depth"] > 3.0 * local_noise:
            preferred_key = "half_P"
            reason = "Half period shows significantly lower BIC score with valid transits."
            alias_warning = True
            alias_type = "half_period_alias"
            
    # Verify that the preferred period has at least 2 observed transit events
    # If the candidate has only 1 transit, we cannot refine its period.
    pref = results[preferred_key]
    if pref["observed_transits"] < 2:
        # Flag this constraint, but keep the preferred key
        logger.warning("Preferred period %.4fd has fewer than 2 transits covered.", pref["period"])
        
    return {
        "preferred_period": pref["period"],
        "preferred_t0": pref["t0"],
        "preferred_depth": pref["depth"],
        "preferred_duration": pref["duration"],
        "preferred_ingress_ratio": pref["ingress_ratio"],
        "alias_warning": alias_warning,
        "alias_type": alias_type,
        "reason": reason,
        "odd_even_delta": odd_even_delta,
        "secondary_depth": sec_depth,
        "hypotheses": results,
    }


# ---------------------------------------------------------------------------
# 4. Stage B MCMC Probabilistic Sampler (emcee)
# ---------------------------------------------------------------------------

def run_mcmc_sampler(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    init_params: dict,
    mcmc_steps: int = 500,
    burn_in: int = 100,
    n_walkers: int = 16,
    seed: int = 42,
    density_prior: tuple[float, float] | None = None, # (density, density_err)
) -> dict:
    """
    Runs emcee to sample the posterior distribution of the physical transit model.
    Samples: [period, t0, rp_rstar, a_rstar, b, q1, q2, baseline, slope, log_jitter].
    q1, q2 are triangular limb-darkening parameters.
    """
    import emcee
    
    np.random.seed(seed)
    
    # ── 1. Restrict data to phase window to make MCMC highly efficient ──
    # Keep within 3.5 * duration
    dur = init_params.get("duration", 0.1)
    per = init_params.get("period", 3.0)
    t0_ref = init_params.get("t0", 0.0)
    
    phase = phase_fold(time, per, t0_ref)
    keep = np.abs(phase * per) <= max(0.5, 3.5 * dur)
    if np.sum(keep) < 15:
        keep = np.ones_like(time, dtype=bool)
        
    t_mcmc = time[keep]
    f_mcmc = flux[keep]
    fe_mcmc = flux_err[keep]
    n_points = len(t_mcmc)
    
    # ── 2. Parameter setup and initial guess ──
    # Setup initial guesses from Stage A optimum
    p_init = init_params.get("period", per)
    t0_init = init_params.get("t0", t0_ref)
    rp_init = init_params.get("rp_rstar", np.sqrt(init_params.get("depth", 0.01)))
    a_init = init_params.get("a_rstar", 8.0)
    b_init = init_params.get("b", 0.3)
    u1_init = init_params.get("u1", 0.4)
    u2_init = init_params.get("u2", 0.3)
    base_init = init_params.get("baseline", 1.0)
    slope_init = init_params.get("slope", 0.0)
    
    # Inverse limb darkening: map u1, u2 to q1, q2 in [0,1]
    u1_val = np.clip(u1_init, 0.01, 0.99)
    u2_val = np.clip(u2_init, -0.99, 0.99)
    # Ensure they are in physical domain
    if u1_val + u2_val > 1.0:
        u1_val, u2_val = 0.4, 0.3
    if u1_val < 0.0 or u1_val + 2.0 * u2_val < 0.0:
        u1_val, u2_val = 0.4, 0.3
        
    q1_init = (u1_val + u2_val)**2
    q2_init = u1_val / (2.0 * (u1_val + u2_val))
    q1_init = np.clip(q1_init, 1e-4, 1.0 - 1e-4)
    q2_init = np.clip(q2_init, 1e-4, 1.0 - 1e-4)
    
    # Initial log jitter
    residual_rms = np.std(f_mcmc)
    log_jit_init = np.log(max(1e-6, 0.1 * residual_rms))
    
    # Parameter vector theta
    theta_guess = np.array([
        p_init, t0_init, rp_init, a_init, b_init,
        q1_init, q2_init, base_init, slope_init, log_jit_init
    ])
    
    n_dim = len(theta_guess)
    
    # ── 3. Prior and Likelihood definitions ──
    p_tol = min(0.02 * p_init, 0.05)
    t0_tol = min(0.1 * dur, 0.1)
    
    # Flat prior boundaries
    p_bounds = (p_init - p_tol, p_init + p_tol)
    t0_bounds = (t0_init - t0_tol, t0_init + t0_tol)
    rp_bounds = (0.0005, 0.6)
    a_bounds = (1.02, 120.0)
    b_bounds = (0.0, 1.1)
    q_bounds = (0.0, 1.0)
    base_bounds = (0.98, 1.02)
    slope_bounds = (-0.01, 0.01)
    log_jit_bounds = (-12.0, -2.0)
    
    def get_inferred_density(a_rs, p_days):
        return 1.408 * (a_rs ** 3) / (p_days ** 2)
        
    def log_prior(theta):
        p, t0, rp, a_rs, b, q1, q2, base, slp, log_jit = theta
        
        # Check bounds
        if not (p_bounds[0] < p < p_bounds[1]): return -np.inf
        if not (t0_bounds[0] < t0 < t0_bounds[1]): return -np.inf
        if not (rp_bounds[0] < rp < rp_bounds[1]): return -np.inf
        if not (a_bounds[0] < a_rs < a_bounds[1]): return -np.inf
        if not (b_bounds[0] < b < b_bounds[1]): return -np.inf
        if not (q_bounds[0] < q1 < q_bounds[1]): return -np.inf
        if not (q_bounds[0] < q2 < q_bounds[1]): return -np.inf
        if not (base_bounds[0] < base < base_bounds[1]): return -np.inf
        if not (slope_bounds[0] < slp < slope_bounds[1]): return -np.inf
        if not (log_jit_bounds[0] < log_jit < log_jit_bounds[1]): return -np.inf
        
        # Geometry checks
        if b > a_rs: return -np.inf
        if b > (1.0 + rp): return -np.inf
        
        lp = 0.0
        
        # a/Rstar must be physically larger than 1 + Rp/Rstar
        if a_rs < (1.0 + rp): return -np.inf
        
        # Optional stellar density prior
        if density_prior is not None:
            rho_star_cat, rho_star_err = density_prior
            if rho_star_cat > 0.0 and rho_star_err > 0.0:
                rho_inf = get_inferred_density(a_rs, p)
                lp += -0.5 * ((rho_inf - rho_star_cat) / rho_star_err) ** 2
                
        return lp

    def log_likelihood(theta):
        p, t0, rp, a_rs, b, q1, q2, base, slp, log_jit = theta
        
        # Convert q1, q2 back to u1, u2
        u1 = 2.0 * np.sqrt(q1) * q2
        u2 = np.sqrt(q1) * (1.0 - 2.0 * q2)
        
        # Generate model
        model = physical_transit_model(t_mcmc, p, t0, rp, a_rs, b, u1, u2, base, slp)
        
        # Variance
        jit_var = np.exp(2.0 * log_jit)
        variance = fe_mcmc**2 + jit_var
        
        return -0.5 * np.sum(((f_mcmc - model) ** 2) / variance + np.log(2.0 * np.pi * variance))

    def log_probability(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    # ── 4. Initialize walkers ──
    pos = []
    attempts = 0
    while len(pos) < n_walkers and attempts < 1000:
        attempts += 1
        noise = np.array([
            1e-6 * np.random.randn(),
            1e-5 * np.random.randn(),
            0.002 * np.random.randn(),
            0.1 * np.random.randn(),
            0.05 * np.random.randn(),
            0.01 * np.random.randn(),
            0.01 * np.random.randn(),
            0.0005 * np.random.randn(),
            1e-5 * np.random.randn(),
            0.1 * np.random.randn(),
        ])
        candidate = theta_guess + noise
        if np.isfinite(log_prior(candidate)) and np.isfinite(log_likelihood(candidate)):
            pos.append(candidate)
            
    if len(pos) < n_walkers:
        logger.warning("Could not initialize all walkers within strict prior bounds. Forcing initialization.")
        pos = theta_guess + 1e-4 * np.random.randn(n_walkers, n_dim)
        
    pos = np.array(pos)
    
    # ── 5. Run MCMC ──
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability)
    
    # Warmup / Burn-in
    if burn_in > 0:
        pos, _, _ = sampler.run_mcmc(pos, burn_in, progress=False)
        sampler.reset()
        
    sampler.run_mcmc(pos, mcmc_steps, progress=False)
    
    # ── 6. Convergence diagnostics ──
    chain = sampler.get_chain()
    
    rhats = []
    autocorr_taus = []
    
    for i in range(n_dim):
        param_chain = chain[:, :, i].T # (walkers, steps)
        rhat = compute_rhat(param_chain)
        rhats.append(rhat)
        tau = compute_autocorr_time(param_chain)
        autocorr_taus.append(tau)
        
    max_rhat = float(np.max(rhats))
    min_ess = int(n_walkers * mcmc_steps / np.max(autocorr_taus))
    
    # Extract flat samples
    flat_samples = sampler.get_chain(flat=True)
    
    q1_samples = flat_samples[:, 5]
    q2_samples = flat_samples[:, 6]
    u1_samples = 2.0 * np.sqrt(q1_samples) * q2_samples
    u2_samples = np.sqrt(q1_samples) * (1.0 - 2.0 * q2_samples)
    
    physical_samples = np.zeros((flat_samples.shape[0], n_dim))
    physical_samples[:, 0:5] = flat_samples[:, 0:5]
    physical_samples[:, 5] = u1_samples
    physical_samples[:, 6] = u2_samples
    physical_samples[:, 7:10] = flat_samples[:, 7:10]
    
    # Calculate percentiles
    posteriors = {}
    labels = ["period", "t0", "rp_rstar", "a_rstar", "b", "u1", "u2", "baseline", "slope", "jitter"]
    
    for idx, label in enumerate(labels):
        samples_col = physical_samples[:, idx] if label != "jitter" else np.exp(physical_samples[:, idx])
        p16, p50, p84 = np.percentile(samples_col, [16, 50, 84])
        posteriors[label] = {
            "median": float(p50),
            "lower_err": float(p50 - p16),
            "upper_err": float(p84 - p50),
            "samples": samples_col.tolist(),
        }
        
    chains_stuck = False
    for i in range(n_dim):
        walker_means = np.mean(chain[:, :, i], axis=0)
        if np.var(walker_means) < 1e-10:
            chains_stuck = True
            
    passed_conv = (max_rhat <= 1.05) and (min_ess >= 100) and not chains_stuck
    
    return {
        "posteriors": posteriors,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "passed_convergence": bool(passed_conv),
        "chains_stuck": chains_stuck,
        "n_walkers": n_walkers,
        "n_steps": mcmc_steps,
        "flat_samples": physical_samples,
    }


def compute_rhat(chain: np.ndarray) -> float:
    """
    Computes Gelman-Rubin R-hat for shape (M, N).
    """
    M, N = chain.shape
    if M <= 1 or N <= 5:
        return 1.0
        
    means = np.mean(chain, axis=1)
    overall_mean = np.mean(means)
    
    B = (N / (M - 1)) * np.sum((means - overall_mean) ** 2)
    W = np.mean(np.var(chain, axis=1, ddof=1))
    
    if W < 1e-12:
        return 1.0
        
    var_x = ((N - 1) / N) * W + (1.0 / N) * B
    rhat = np.sqrt(var_x / W)
    return float(rhat)


def compute_autocorr_time(chain: np.ndarray) -> float:
    """
    Estimates autocorrelation time.
    """
    M, N = chain.shape
    taus = []
    
    for m in range(M):
        x = chain[m, :]
        n = len(x)
        if n < 10:
            taus.append(1.0)
            continue
            
        x_mean = np.mean(x)
        x_var = np.var(x)
        if x_var < 1e-12:
            taus.append(1.0)
            continue
            
        max_lag = min(n // 2, 200)
        acf = np.zeros(max_lag)
        for lag in range(max_lag):
            acf[lag] = np.mean((x[:n-lag] - x_mean) * (x[lag:] - x_mean)) / x_var
            
        neg_idx = np.where(acf < 0)[0]
        cut = neg_idx[0] if len(neg_idx) > 0 else max_lag
        
        tau = 1.0 + 2.0 * np.sum(acf[1:cut])
        taus.append(max(1.0, tau))
        
    return float(np.mean(taus))


# ---------------------------------------------------------------------------
# 5. Red-Noise and Correlated-Noise Diagnostics
# ---------------------------------------------------------------------------

def calculate_red_noise_diagnostics(residuals: np.ndarray) -> dict:
    """
    Calculates correlated-noise diagnostics from fit residuals.
    """
    n = len(residuals)
    if n < 10:
        return {
            "durbin_watson": 2.0,
            "beta_factor": 1.0,
            "autocorr_lag1": 0.0,
            "warning": False,
        }
        
    # Durbin-Watson
    diff = np.diff(residuals)
    sum_diff_sq = np.sum(diff ** 2)
    sum_sq = np.sum(residuals ** 2)
    dw = float(sum_diff_sq / sum_sq) if sum_sq > 1e-12 else 2.0
    
    # Autocorrelation
    res_mean = np.mean(residuals)
    res_var = np.var(residuals)
    ac1 = 0.0
    if res_var > 1e-12:
        ac1 = float(np.mean((residuals[:-1] - res_mean) * (residuals[1:] - res_mean)) / res_var)
        
    # Time-averaging beta factor
    sigma_1 = np.std(residuals)
    beta_factors = []
    
    bin_sizes = [2, 4, 8, 16, 32]
    for M in bin_sizes:
        if n < 2 * M:
            continue
        n_bins = n // M
        binned = np.mean(residuals[:n_bins * M].reshape(n_bins, M), axis=1)
        sigma_M = np.std(binned)
        sigma_exp = sigma_1 / np.sqrt(M)
        if sigma_exp > 0.0:
            beta_factors.append(sigma_M / sigma_exp)
            
    max_beta = float(np.max(beta_factors)) if len(beta_factors) > 0 else 1.0
    max_beta = max(1.0, max_beta)
    
    warning = (dw < 1.5) or (ac1 > 0.25) or (max_beta > 1.25)
    
    return {
        "durbin_watson": dw,
        "beta_factor": max_beta,
        "autocorr_lag1": ac1,
        "warning": warning,
    }
