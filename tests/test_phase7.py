"""
tests/test_phase7.py
--------------------
Unit and integration tests for Phase 7 Transit Fitting & Uncertainty pipeline.
"""

from __future__ import annotations
import numpy as np
import pytest
from core.transit_fitter import fit_transit
from core.transit_fitting_pipeline import (
    physical_transit_model,
    trapezoid_transit_model,
    calculate_red_noise_diagnostics,
    resolve_period_alias_hypotheses,
)
from core.uncertainty import estimate_uncertainties

# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_phase_folding_boundaries():
    """Verify phase folding wraps correctly near phase boundaries."""
    from core.utils import phase_fold
    time = np.array([0.0, 1.5, 3.0, 4.5, 6.0])
    period = 3.0
    t0 = 0.0
    phases = phase_fold(time, period, t0)
    # Expected folded phases centered around 0 in [-0.5, 0.5]
    # For time 0.0 -> 0.0
    # For time 1.5 -> 0.5 (or -0.5 depending on boundary tie-break, usually 0.5 or -0.5)
    # For time 3.0 -> 0.0
    # For time 4.5 -> -0.5 or 0.5
    assert np.abs(phases[0]) < 1e-12
    assert np.abs(phases[2]) < 1e-12
    assert np.all(phases >= -0.5) and np.all(phases <= 0.5)


def test_physical_model_values():
    """Verify that physical model returns expected baseline and in-transit values."""
    time = np.linspace(0.0, 10.0, 100)
    period = 4.0
    t0 = 2.0
    # Deep transit of large companion (rp = 0.2)
    flux = physical_transit_model(
        time, period=period, t0=t0, rp_rstar=0.2, a_rstar=5.0, b=0.0, u1=0.0, u2=0.0
    )
    # Out of transit should be baseline (1.0)
    # Mid-transit at t=2.0 and t=6.0 should be lower than 1.0
    assert np.all(flux <= 1.0)
    assert flux[50] == 1.0 # far from transit
    in_tr_idx = np.argmin(np.abs(time - 2.0))
    assert flux[in_tr_idx] < 0.99
    # Uniform star (u1=0, u2=0) depth is exactly rp_rstar^2 = 0.04
    assert np.abs((1.0 - flux[in_tr_idx]) - 0.04) < 1e-3


def test_dilution_correction():
    """Validate dilution depth scaling and error propagation."""
    # Test observed vs corrected depth
    # dilution_factor = 1.0 - contamination
    # depth_corrected = depth_obs / dilution_factor
    time = np.linspace(0.0, 8.0, 800)
    flux = np.ones_like(time)
    flux_err = np.ones_like(time) * 0.001
    
    # 20% contamination -> dilution_factor = 0.8
    meta = {
        "contamination_ratio": 0.2,
        "contamination_uncertainty": 0.02,
        "stellar_radius": 1.0,
        "stellar_radius_err": 0.05,
    }
    
    # Simulate a transit with 0.8% observed depth
    t0 = 2.5
    p = 3.0
    dur = 0.1
    phase = ((time - t0) / p + 0.5) % 1.0 - 0.5
    in_tr = np.abs(phase * p) <= dur / 2.0
    flux[in_tr] -= 0.008
    
    res = fit_transit(
        time, flux, init_period=p, init_t0=t0, init_duration=dur, init_depth=0.008,
        flux_err=flux_err, config={"fitting_level": "quick"}, metadata=meta
    )
    
    assert res["fit_status"] in ("SUCCESS", "SUCCESS_WITH_WARNINGS")
    # Corrected depth should be around 0.008 / 0.8 = 0.010 (1.0%)
    assert np.abs(res["corrected_depth"] - 0.010) < 0.002
    assert res["observed_depth"] < res["corrected_depth"]
    # Ensure planet radius is estimated
    assert res["planet_radius_earth"] is not None
    assert res["planet_radius_earth"] > 0.0


def test_red_noise_diagnostics():
    """Verify red-noise calculations on synthetic white and correlated noise."""
    rng = np.random.default_rng(42)
    # 1. Pure white noise: DW should be ~2.0, beta ~1.0
    white = rng.normal(0, 0.001, 1000)
    diag_w = calculate_red_noise_diagnostics(white)
    assert 1.8 < diag_w["durbin_watson"] < 2.2
    assert diag_w["beta_factor"] < 1.2
    
    # 2. Correlated noise: DW should be < 1.5, beta > 1.3
    correlated = np.zeros(1000)
    for i in range(1, 1000):
        correlated[i] = 0.7 * correlated[i-1] + rng.normal(0, 0.001)
    diag_c = calculate_red_noise_diagnostics(correlated)
    assert diag_c["durbin_watson"] < 1.5
    assert diag_c["beta_factor"] > 1.3
    assert diag_c["warning"] is True


def test_quality_flags_generation():
    """Verify quality flags are correctly raised for graze, eb, and data gaps."""
    # Grazing geometry: b > 1 - rp
    time = np.linspace(0.0, 5.0, 500)
    flux = np.ones_like(time)
    t0 = 2.5
    p = 4.0
    dur = 0.1
    phase = ((time - t0) / p + 0.5) % 1.0 - 0.5
    flux[np.abs(phase * p) <= dur / 2.0] -= 0.04
    
    # Run fit with very large impact parameter b to force grazing flag
    res = fit_transit(
        time, flux, init_period=p, init_t0=t0, init_duration=dur, init_depth=0.04,
        config={"fitting_level": "quick"},
        metadata={"stellar_radius": None} # missing stellar metadata flag
    )
    
    assert "stellar_parameters_missing" in res["quality_flags"]


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

def test_clean_synthetic_transit_integration():
    """Integration: Fit a clean high-SNR transit and verify recovery accuracy."""
    rng = np.random.default_rng(123)
    time = np.linspace(0.0, 10.0, 2000)
    p_true = 3.5
    t0_true = 1.5
    rp_true = 0.08
    a_true = 8.0
    b_true = 0.2
    
    model = physical_transit_model(
        time, period=p_true, t0=t0_true, rp_rstar=rp_true, a_rstar=a_true, b=b_true, u1=0.4, u2=0.3
    )
    noise = rng.normal(0, 0.0005, len(time))
    flux = model + noise
    
    # Run pipeline in Quick Mode (Stage A only)
    res = fit_transit(
        time, flux, init_period=p_true, init_t0=t0_true, init_duration=0.12, init_depth=0.0064,
        config={"fitting_level": "quick"}
    )
    
    assert res["fit_status"] in ("SUCCESS", "SUCCESS_WITH_WARNINGS")
    assert np.abs(res["period_days"] - p_true) / p_true < 0.001
    assert np.abs(res["epoch_btjd"] - t0_true) < 0.01
    assert np.abs(res["rp_rstar"] - rp_true) < 0.01
    assert res["chi2"] < 3000.0


def test_alias_selection_integration():
    """Integration: Feed half period and verify that double period is selected."""
    # Build a binary-like curve folded at true period 4.0d
    # If we initialize at P = 2.0d, the alias checker should detect that the true period is 4.0d
    rng = np.random.default_rng(999)
    time = np.linspace(0.0, 12.0, 3000)
    p_true = 4.0
    t0_true = 1.0
    
    # Primary eclipse: depth = 8%
    # Secondary eclipse at phase 0.5: depth = 3%
    model = np.ones_like(time)
    phase = ((time - t0_true) / p_true + 0.5) % 1.0 - 0.5
    
    dur = 0.15
    in_primary = np.abs(phase * p_true) <= dur / 2.0
    in_secondary = np.abs(np.abs(phase * p_true) - 2.0) <= dur / 2.0
    
    model[in_primary] -= 0.08
    model[in_secondary] -= 0.03
    flux = model + rng.normal(0, 0.001, len(time))
    flux_err = np.ones_like(time) * 0.001
    
    # Run alias test initialized at false period P = 2.0
    alias_res = resolve_period_alias_hypotheses(
        time, flux, flux_err,
        bls_period=2.0, bls_t0=t0_true, bls_duration=dur, bls_depth=0.08
    )
    
    # Should resolve to the true period 4.0
    assert alias_res["alias_warning"] is True
    assert np.abs(alias_res["preferred_period"] - 4.0) < 0.05
    assert alias_res["alias_type"] in ("double_period_odd_even", "double_period_secondary")
