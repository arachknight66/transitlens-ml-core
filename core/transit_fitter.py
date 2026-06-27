"""
core/transit_fitter.py
----------------------
Coordinates deterministic and probabilistic transit fitting for TransitLens.

Exposes:
- fit_transit(time, flux, init_period, init_t0, init_duration, init_depth, ...)
- trapezoid_transit_model (analytical trapezoidal model)
"""

from __future__ import annotations
import logging
import numpy as np
from core.exceptions import MLCoreError
from core.utils import phase_fold

# Import from our new fitting pipeline
from core.transit_fitting_pipeline import (
    trapezoid_transit_model,
    physical_transit_model,
    fit_trapezoid_deterministic,
    fit_physical_deterministic,
    resolve_period_alias_hypotheses,
    run_mcmc_sampler,
    calculate_red_noise_diagnostics,
)

logger = logging.getLogger(__name__)


def fit_transit(
    time: np.ndarray,
    flux: np.ndarray,
    init_period: float,
    init_t0: float,
    init_duration: float,
    init_depth: float,
    flux_err: np.ndarray | None = None,
    config: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """
    Two-stage transit parameter estimation pipeline.
    
    1. Validation: Verifies that time, flux, and metadata inputs are physical and valid.
    2. Stage A (Alias check + Deterministic optimization): Fits trapezoidal and physical 
       models to find refined P, t0, duration, depth, and local baseline.
    3. Stage B (MCMC Posterior sampling): Runs emcee to sample posteriors and estimate 
       uncertainties, convergence R-hat, and correlation patterns.
    4. Diagnostics: Computes Durbin-Watson, beta factor, and quality flags.
    
    Parameters
    ----------
    time, flux : np.ndarray
        Cleaned light curve data arrays.
    init_period, init_t0, init_duration, init_depth : float
        Initial transit parameter guesses (typically from BLS detector).
    flux_err : np.ndarray or None
        Measurement errors. Estimated from out-of-transit scatter if None.
    config : dict or None
        Override parameters for fitting level, burn-in, walkers, limits, etc.
    metadata : dict or None
        Stellar catalog properties (mass, radius, density, contamination).
        
    Returns
    -------
    dict
        Structured fitting results containing values, uncertainties, flags, and convergence stats.
    """
    cfg = config or {}
    meta = metadata or {}
    
    # ── 1. Input Validation ──
    # Check shape, types, sorting, bounds
    n_points = len(time)
    if n_points < 10:
        logger.warning("transit_fitter: Minsufficient data points to fit (%d).", n_points)
        return _failed_fit_result(init_period, init_t0, init_duration, init_depth, "insufficient_data_points")
        
    if init_period <= 0.0 or init_duration <= 0.0 or init_depth <= 0.0:
        logger.warning("transit_fitter: Invalid initial candidate parameters.")
        return _failed_fit_result(init_period, init_t0, init_duration, init_depth, "invalid_initial_parameters")
        
    if init_duration >= init_period:
        logger.warning("transit_fitter: Transit duration exceeds orbital period.")
        return _failed_fit_result(init_period, init_t0, init_duration, init_depth, "duration_exceeds_period")
        
    # Ensure arrays are float and finite
    try:
        time = np.asarray(time, dtype=float)
        flux = np.asarray(flux, dtype=float)
        if not (np.all(np.isfinite(time)) and np.all(np.isfinite(flux))):
            return _failed_fit_result(init_period, init_t0, init_duration, init_depth, "non_finite_inputs")
    except Exception as exc:
        return _failed_fit_result(init_period, init_t0, init_duration, init_depth, f"array_conversion_error:{str(exc)[:50]}")
        
    # Setup flux errors if missing
    if flux_err is None or len(flux_err) != n_points:
        local_rms = np.std(flux) if n_points > 0 else 0.001
        flux_err = np.ones_like(flux) * max(1e-6, local_rms)
    else:
        flux_err = np.asarray(flux_err, dtype=float)
        if not np.all(flux_err > 0.0):
            flux_err = np.where(flux_err <= 0.0, 1e-4, flux_err)
            
    # ── 2. Contamination / Dilution correction ──
    # dilution_factor = 1 - contamination_ratio
    contamination = float(meta.get("contamination_ratio", 0.0) or 0.0)
    contamination_err = float(meta.get("contamination_uncertainty", 0.0) or 0.0)
    dilution_factor = float(meta.get("dilution_factor", 1.0 - contamination) or 1.0)
    
    # ── 3. Stellar prior data lookup ──
    # Check if stellar density catalog value is available
    stellar_mass = meta.get("stellar_mass")
    stellar_mass_err = meta.get("stellar_mass_err")
    stellar_radius = meta.get("stellar_radius")
    stellar_radius_err = meta.get("stellar_radius_err")
    
    density_prior = None
    if stellar_mass and stellar_radius:
        # In solar units, density rho_star = mass / radius^3
        # Propagate uncertainties
        try:
            m = float(stellar_mass)
            r = float(stellar_radius)
            # Density in Solar units = M/R^3
            rho_star = m / (r ** 3)
            
            # error propagation
            m_err = float(stellar_mass_err) if stellar_mass_err else 0.1 * m
            r_err = float(stellar_radius_err) if stellar_radius_err else 0.1 * r
            rho_err = rho_star * np.sqrt((m_err / m)**2 + (3.0 * r_err / r)**2)
            
            # Convert to g/cm^3 (Solar density = 1.408 g/cm^3)
            density_prior = (rho_star * 1.408, rho_err * 1.408)
        except Exception:
            pass
            
    # ── 4. Stage A: Alias Testing and Period Refinement ──
    try:
        alias_res = resolve_period_alias_hypotheses(
            time, flux, flux_err,
            bls_period=init_period,
            bls_t0=init_t0,
            bls_duration=init_duration,
            bls_depth=init_depth,
        )
        
        pref_period = alias_res["preferred_period"]
        pref_t0 = alias_res["preferred_t0"]
        pref_depth = alias_res["preferred_depth"]
        pref_duration = alias_res["preferred_duration"]
    except Exception as exc:
        logger.warning("transit_fitter: Alias checking failed: %s. Falling back to BLS parameters.", exc)
        pref_period = init_period
        pref_t0 = init_t0
        pref_depth = init_depth
        pref_duration = init_duration
        alias_res = {
            "alias_warning": False,
            "alias_type": "none",
            "reason": f"alias_failed:{str(exc)[:50]}",
            "odd_even_delta": 0.0,
            "secondary_depth": 0.0,
            "hypotheses": {},
        }
        
    # Run deterministic physical model optimizer at the resolved period
    try:
        fit_opt = fit_physical_deterministic(
            time, flux, flux_err,
            init_period=pref_period,
            init_t0=pref_t0,
            init_depth=pref_depth,
            init_duration=pref_duration,
            stellar_density=density_prior[0] if density_prior else None,
        )
    except Exception as exc:
        logger.warning("transit_fitter: Deterministic physical optimizer failed: %s.", exc)
        fit_opt = {
            "period": pref_period,
            "t0": pref_t0,
            "rp_rstar": np.sqrt(pref_depth),
            "a_rstar": 8.0,
            "b": 0.3,
            "u1": 0.4,
            "u2": 0.3,
            "baseline": 1.0,
            "slope": 0.0,
            "success": False,
            "fun": 1e10,
            "nfev": 0,
        }
        
    # ── 5. Stage B: MCMC Sampling and Uncertainty Quantification ──
    # Check requested fitting level
    level = cfg.get("fitting_level", "standard").lower()
    
    # Check if emcee is requested and imported successfully
    mcmc_success = False
    mcmc_res = {}
    
    if level in ("standard", "rigorous") and fit_opt.get("success", False):
        mcmc_steps = 800 if level == "rigorous" else 400
        burn_in = 200 if level == "rigorous" else 100
        n_walkers = 32 if level == "rigorous" else 24
        
        try:
            mcmc_res = run_mcmc_sampler(
                time, flux, flux_err,
                init_params=fit_opt,
                mcmc_steps=mcmc_steps,
                burn_in=burn_in,
                n_walkers=n_walkers,
                seed=cfg.get("random_seed", 42),
                density_prior=density_prior,
            )
            mcmc_success = mcmc_res.get("passed_convergence", False) or mcmc_res.get("min_ess", 0) > 30
        except Exception as exc:
            logger.warning("transit_fitter: MCMC posterior sampler failed: %s. Using approximate uncertainties.", exc)
            
    # ── 6. Assemble uncertainties ──
    # Determine bounds and uncertainties. If MCMC ran successfully, we use MCMC.
    # Otherwise, we use approximation from curve_fit covariance or Kovacs scaling.
    param_vals = {}
    param_errs = {}
    
    if mcmc_success:
        posteriors = mcmc_res["posteriors"]
        for key in ["period", "t0", "rp_rstar", "a_rstar", "b", "u1", "u2", "baseline", "slope", "jitter"]:
            post = posteriors[key]
            param_vals[key] = post["median"]
            param_errs[key + "_err_lower"] = post["lower_err"]
            param_errs[key + "_err_upper"] = post["upper_err"]
            
        mcmc_passed = mcmc_res.get("passed_convergence", False)
        mcmc_rhat = mcmc_res.get("max_rhat", 1.01)
        mcmc_ess = mcmc_res.get("min_ess", 500)
        uncertainty_method = "mcmc"
    else:
        # Fallback to deterministic values
        for key in ["period", "t0", "rp_rstar", "a_rstar", "b", "u1", "u2", "baseline", "slope"]:
            param_vals[key] = fit_opt[key]
            
        # Standard analytical error estimates based on SNR scaling & Kovacs relation
        # Period error: Kovacs P_err = P^2 / (T_span * SNR)
        time_span = float(np.max(time) - np.min(time)) if len(time) > 0 else 10.0
        snr = float(meta.get("bls_snr", 10.0) or 10.0)
        p_err = (param_vals["period"] ** 2) / (time_span * snr) if snr > 0 else 0.01
        p_err = min(p_err, 0.05 * param_vals["period"])
        
        # Estimate depth/duration error based on local noise
        local_rms = np.std(flux) if len(flux) > 0 else 0.001
        depth_err = local_rms / np.sqrt(n_points) if n_points > 0 else 0.001
        
        param_vals["jitter"] = float(local_rms * 0.1) # dummy jitter
        
        # Populate symmetric errors
        param_errs["period_err_lower"] = p_err
        param_errs["period_err_upper"] = p_err
        param_errs["t0_err_lower"] = 0.05 * init_duration
        param_errs["t0_err_upper"] = 0.05 * init_duration
        param_errs["rp_rstar_err_lower"] = 0.5 * depth_err / max(1e-4, param_vals["rp_rstar"])
        param_errs["rp_rstar_err_upper"] = 0.5 * depth_err / max(1e-4, param_vals["rp_rstar"])
        param_errs["a_rstar_err_lower"] = 0.2 * param_vals["a_rstar"]
        param_errs["a_rstar_err_upper"] = 0.2 * param_vals["a_rstar"]
        param_errs["b_err_lower"] = 0.2
        param_errs["b_err_upper"] = 0.2
        param_errs["u1_err_lower"] = 0.1
        param_errs["u1_err_upper"] = 0.1
        param_errs["u2_err_lower"] = 0.1
        param_errs["u2_err_upper"] = 0.1
        param_errs["baseline_err_lower"] = 0.001
        param_errs["baseline_err_upper"] = 0.001
        param_errs["slope_err_lower"] = 0.0001
        param_errs["slope_err_upper"] = 0.0001
        param_errs["jitter_err_lower"] = 0.0001
        param_errs["jitter_err_upper"] = 0.0001
        
        mcmc_passed = False
        mcmc_rhat = 1.0
        mcmc_ess = 0
        uncertainty_method = "covariance_approx"
        
    # ── 7. Derived parameters ──
    # Calculate transit model predictions using best fit parameters
    best_model = physical_transit_model(
        time,
        param_vals["period"],
        param_vals["t0"],
        param_vals["rp_rstar"],
        param_vals["a_rstar"],
        param_vals["b"],
        param_vals["u1"],
        param_vals["u2"],
        param_vals["baseline"],
        param_vals["slope"],
    )
    
    # Calculate residuals
    residuals = flux - best_model
    residual_rms = float(np.std(residuals))
    
    # Red noise diagnostics
    red_noise = calculate_red_noise_diagnostics(residuals)
    
    # Inflate uncertainties if red noise beta factor is substantial (>1.1)
    beta = red_noise["beta_factor"]
    if beta > 1.1:
        for k in param_errs:
            param_errs[k] *= beta
            
    # Calculate Observed Depth: depth = 1 - F_in / F_out
    # We find minimum of best-fit model normalized by baseline
    observed_depth = float(1.0 - np.min(best_model) / param_vals["baseline"])
    observed_depth_err = param_errs["rp_rstar_err_upper"] * 2.0 * param_vals["rp_rstar"]
    
    # Calculate Dilution-corrected Depth
    corrected_depth = observed_depth / max(1e-6, dilution_factor)
    
    # Corrected depth uncertainty propagation
    # sigma_corrected = corrected_depth * sqrt( (sigma_obs/obs)^2 + (sigma_dil/dil)^2 )
    term_obs = observed_depth_err / max(1e-6, observed_depth)
    term_dil = contamination_err / max(1e-6, dilution_factor)
    corrected_depth_err = corrected_depth * np.sqrt(term_obs**2 + term_dil**2)
    
    # Calculate Physical Planet Radius in Earth radii: Rp = rp_rstar * R_star
    planet_radius_earth = None
    planet_radius_earth_err_lower = None
    planet_radius_earth_err_upper = None
    
    if stellar_radius:
        try:
            # R_star is in Solar Radii. Rp = rp_rstar * R_star * 109.076 (RSun to REarth ratio)
            rs = float(stellar_radius)
            rs_err = float(stellar_radius_err) if stellar_radius_err else 0.1 * rs
            
            p_val = param_vals["rp_rstar"]
            p_err_l = param_errs["rp_rstar_err_lower"]
            p_err_u = param_errs["rp_rstar_err_upper"]
            
            # Median planet radius
            r_earth_median = p_val * rs * 109.076
            planet_radius_earth = float(r_earth_median)
            
            # Error propagation
            planet_radius_earth_err_lower = float(r_earth_median * np.sqrt((p_err_l / p_val)**2 + (rs_err / rs)**2))
            planet_radius_earth_err_upper = float(r_earth_median * np.sqrt((p_err_u / p_val)**2 + (rs_err / rs)**2))
        except Exception:
            pass
            
    # Inferred stellar density: rho_star = 1.408 * a^3 / P^2
    inferred_density = 1.408 * (param_vals["a_rstar"]**3) / (param_vals["period"]**2)
    
    # Inferred inclination: cos(i) = b / a_rstar
    cos_i = param_vals["b"] / param_vals["a_rstar"]
    inclination_deg = float(np.arccos(np.clip(cos_i, -1.0, 1.0)) * 180.0 / np.pi)
    
    # Calculate Transit Duration T14 (First to Fourth contact)
    # T14 = P/pi * arcsin( sqrt( (1+p)^2 - b^2 ) / (a_rstar * sin(i)) )
    # sin(i) = sqrt( 1 - cos^2(i) )
    sin_i = np.sqrt(np.clip(1.0 - cos_i**2, 1e-12, 1.0))
    term_arcsin = np.sqrt(np.clip((1.0 + param_vals["rp_rstar"])**2 - param_vals["b"]**2, 0.0, 1e4)) / (param_vals["a_rstar"] * sin_i)
    term_arcsin = np.clip(term_arcsin, -1.0, 1.0)
    duration_t14 = float((param_vals["period"] / np.pi) * np.arcsin(term_arcsin))
    
    # Estimate T14 uncertainty: propagate rp_rstar and a_rstar errors
    duration_t14_err = float(0.15 * duration_t14) # fallback error
    
    # Phase coverage & count observed transits
    phase = phase_fold(time, param_vals["period"], param_vals["t0"])
    in_transit_mask = np.abs(phase * param_vals["period"]) <= (duration_t14 / 2.0)
    in_transit_points = int(np.sum(in_transit_mask))
    
    cycles = np.round((time - param_vals["t0"]) / param_vals["period"])
    unique_cycles = np.unique(cycles[in_transit_mask])
    observed_events = int(len(unique_cycles))
    
    # Phase coverage fraction
    window_phase = (1.5 * duration_t14) / param_vals["period"]
    bins = np.linspace(-window_phase, window_phase, 30)
    hist, _ = np.histogram(phase, bins=bins)
    phase_cov = float(np.sum(hist > 0) / (len(bins) - 1)) if len(bins) > 1 else 0.0
    
    # Chi-square and goodness-of-fit
    chi2 = float(np.sum((residuals / flux_err)**2))
    dof = n_points - 10 # 10 fitted parameters
    reduced_chi2 = float(chi2 / dof) if dof > 0 else 1.0
    
    # ── 8. Quality Flags and Status ──
    q_flags = []
    
    # Check transits covered
    if observed_events < 2:
        q_flags.append("insufficient_transits")
    if in_transit_points < 10:
        q_flags.append("insufficient_in_transit_points")
    if phase_cov < 0.75:
        q_flags.append("poor_phase_coverage")
        
    # Check parameters near boundaries
    if param_vals["b"] > 0.95 * (1.0 + param_vals["rp_rstar"]):
        q_flags.append("grazing_geometry")
    if param_vals["rp_rstar"] > 0.3:
        q_flags.append("likely_eclipsing_binary")
    if param_vals["rp_rstar"] < 0.005:
        q_flags.append("depth_implausible")
        
    # Convergence warnings
    if level in ("standard", "rigorous"):
        if not mcmc_success:
            q_flags.append("sampler_failed")
        elif not mcmc_passed:
            q_flags.append("sampler_not_converged")
            
    # Fit boundary limit warnings
    if param_vals["b"] <= 0.01 or param_vals["rp_rstar"] >= 0.79:
        q_flags.append("parameter_at_boundary")
        
    # Red noise warnings
    if red_noise["warning"]:
        q_flags.append("correlated_noise_detected")
    if contamination_err > 0.15:
        q_flags.append("dilution_uncertain")
    if not stellar_radius:
        q_flags.append("stellar_parameters_missing")
        
    # Set overall fit status
    # SUCCESS, SUCCESS_WITH_WARNINGS, APPROXIMATE, FAILED
    if not fit_opt.get("success", False):
        fit_status = "FAILED"
    elif len(q_flags) == 0:
        fit_status = "SUCCESS"
    elif any(f in ("sampler_failed", "insufficient_transits", "poor_phase_coverage") for f in q_flags):
        fit_status = "APPROXIMATE"
    else:
        fit_status = "SUCCESS_WITH_WARNINGS"
        
    # Assemble fit quality
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((flux - np.mean(flux)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0
    
    return {
        "fit_status": fit_status,
        "quality_flags": q_flags,
        "period_days": float(param_vals["period"]),
        "period_uncertainty_days": float(param_errs["period_err_upper"]),
        "epoch_btjd": float(param_vals["t0"]),
        "epoch_uncertainty_days": float(param_errs["t0_err_upper"]),
        "depth": observed_depth,
        "depth_uncertainty": observed_depth_err,
        "duration_days": duration_t14,
        "duration_uncertainty_days": duration_t14_err,
        "rp_rstar": float(param_vals["rp_rstar"]),
        "rp_rstar_err_lower": float(param_errs["rp_rstar_err_lower"]),
        "rp_rstar_err_upper": float(param_errs["rp_rstar_err_upper"]),
        "a_rstar": float(param_vals["a_rstar"]),
        "a_rstar_err_lower": float(param_errs["a_rstar_err_lower"]),
        "a_rstar_err_upper": float(param_errs["a_rstar_err_upper"]),
        "b": float(param_vals["b"]),
        "b_err_lower": float(param_errs["b_err_lower"]),
        "b_err_upper": float(param_errs["b_err_upper"]),
        "u1": float(param_vals["u1"]),
        "u2": float(param_vals["u2"]),
        "baseline": float(param_vals["baseline"]),
        "slope": float(param_vals["slope"]),
        "jitter": float(param_vals["jitter"]),
        
        # Diagnostics
        "fit_quality": max(0.0, r2),
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
        "bic": float(chi2 + 10 * np.log(n_points)),
        "aic": float(chi2 + 20),
        "residual_rms": residual_rms,
        "durbin_watson": red_noise["durbin_watson"],
        "beta_factor": red_noise["beta_factor"],
        "autocorr_lag1": red_noise["autocorr_lag1"],
        
        # MCMC specific
        "mcmc_passed": mcmc_passed,
        "mcmc_rhat": mcmc_rhat,
        "mcmc_ess": mcmc_ess,
        "mcmc_samples": mcmc_res.get("posteriors"),
        
        # Derived
        "observed_depth": observed_depth,
        "observed_depth_uncertainty": observed_depth_err,
        "corrected_depth": corrected_depth,
        "corrected_depth_uncertainty": corrected_depth_err,
        "planet_radius_earth": planet_radius_earth,
        "planet_radius_earth_err_lower": planet_radius_earth_err_lower,
        "planet_radius_earth_err_upper": planet_radius_earth_err_upper,
        "inferred_density": float(inferred_density),
        "inclination_deg": inclination_deg,
        "observed_transits": observed_events,
        "in_transit_cadences": in_transit_points,
        "phase_coverage_fraction": phase_cov,
        "alias_warning": alias_res["alias_warning"],
        "alias_type": alias_res["alias_type"],
        "alias_reason": alias_res["reason"],
        "odd_even_delta": alias_res["odd_even_delta"],
        "secondary_depth": alias_res["secondary_depth"],
        "residuals": residuals,
        "best_model": best_model,
        "uncertainty_method": uncertainty_method,
    }


def _failed_fit_result(p, t0, dur, dep, reason) -> dict:
    """Returns a dictionary structures with failures."""
    return {
        "fit_status": "FAILED",
        "quality_flags": ["optimizer_failed", reason],
        "period_days": p,
        "period_uncertainty_days": 0.05 * p,
        "epoch_btjd": t0,
        "epoch_uncertainty_days": 0.05 * dur,
        "depth": dep,
        "depth_uncertainty": 0.05 * dep,
        "duration_days": dur,
        "duration_uncertainty_days": 0.05 * dur,
        "rp_rstar": np.sqrt(dep),
        "rp_rstar_err_lower": 0.0,
        "rp_rstar_err_upper": 0.0,
        "a_rstar": 8.0,
        "a_rstar_err_lower": 0.0,
        "a_rstar_err_upper": 0.0,
        "b": 0.0,
        "b_err_lower": 0.0,
        "b_err_upper": 0.0,
        "u1": 0.4,
        "u2": 0.3,
        "baseline": 1.0,
        "slope": 0.0,
        "jitter": 0.0,
        "fit_quality": 0.0,
        "residual_rms": 0.001,
        "durbin_watson": 2.0,
        "beta_factor": 1.0,
        "autocorr_lag1": 0.0,
        "mcmc_passed": False,
        "mcmc_rhat": 1.0,
        "mcmc_ess": 0,
        "observed_depth": dep,
        "observed_depth_uncertainty": 0.0,
        "corrected_depth": dep,
        "corrected_depth_uncertainty": 0.0,
        "planet_radius_earth": None,
        "inferred_density": 0.0,
        "inclination_deg": 90.0,
        "observed_transits": 0,
        "in_transit_cadences": 0,
        "phase_coverage_fraction": 0.0,
        "alias_warning": False,
        "alias_type": "none",
        "alias_reason": reason,
        "odd_even_delta": 0.0,
        "secondary_depth": 0.0,
        "residuals": np.array([]),
        "best_model": np.array([]),
        "uncertainty_method": "failed",
    }
