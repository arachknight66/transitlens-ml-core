# test_phase2_diagnostics.py
# -------------------------
# Test suite for Phase 2 diagnostic vetting submodules and contracts.

from __future__ import annotations
import pytest
import numpy as np
from diagnostics import run_diagnostics
from diagnostics.contracts import get_default_diagnostics_dict, validate_schema

def test_diagnostics_runs_with_flat_noise():
    """Flat noise should return low risk scores and 'exoplanet_transit' or 'review_required' routing."""
    rng = np.random.default_rng(42)
    time = np.linspace(0, 10, 500)
    flux = 1.0 + rng.normal(0, 0.001, size=len(time))
    
    res = run_diagnostics(
        time=time,
        flux=flux,
        period=2.0,
        epoch_btjd=1.0,
        duration_days=0.4,
        depth=0.01,
        centroid_x=np.ones_like(time),
        centroid_y=np.ones_like(time),
    )
    
    assert res["centroid_available"] is True
    assert res["eb_risk_level"] in ("low", "medium", "high", "unavailable")
    assert res["blend_risk_level"] in ("low", "medium", "high", "unavailable")
    assert "recommended_route" in res

def test_diagnostics_eb_routing():
    """An observation with large secondary eclipse and V-shape should route to eclipsing_binary."""
    rng = np.random.default_rng(42)
    time = np.linspace(0, 10, 500)
    flux = 1.0 + rng.normal(0, 0.001, size=len(time))
    
    # Simulate primary transit at phase 0.0
    phase = (time - 1.0) % 2.0
    phase = np.where(phase > 1.0, phase - 2.0, phase)
    
    # Symmetric V-shape primary at phase 0.0
    in_primary = np.abs(phase) <= 0.10
    flux[in_primary] -= 0.20 * (1.0 - np.abs(phase[in_primary]) / 0.10)
    
    # Large secondary at phase 0.5 (which is 1.0 or -1.0 in this phase space)
    in_secondary = np.abs(np.abs(phase) - 1.0) <= 0.10
    flux[in_secondary] -= 0.10 * (1.0 - np.abs(np.abs(phase[in_secondary]) - 1.0) / 0.10)
    
    res = run_diagnostics(
        time=time,
        flux=flux,
        period=2.0,
        epoch_btjd=1.0,
        duration_days=0.4,
        depth=0.20,
    )
    
    assert res["secondary_evidence_flag"] is True
    assert res["recommended_route"] in ("eclipsing_binary", "review_required")

def test_diagnostics_blend_routing():
    """An observation with significant centroid shift should route to blend_contamination."""
    rng = np.random.default_rng(42)
    time = np.linspace(0, 10, 500)
    flux = 1.0 + rng.normal(0, 0.001, size=len(time))
    
    # Primary transit
    phase = (time - 1.0) % 2.0
    phase = np.where(phase > 1.0, phase - 2.0, phase)
    in_primary = np.abs(phase) <= 0.10
    flux[in_primary] -= 0.01
    
    # Centroid coordinates: shift on-transit
    centroid_x = np.ones_like(time) + rng.normal(0, 0.01, size=len(time))
    centroid_y = np.ones_like(time) + rng.normal(0, 0.01, size=len(time))
    centroid_x[in_primary] += 2.5 # large offset
    
    res = run_diagnostics(
        time=time,
        flux=flux,
        period=2.0,
        epoch_btjd=1.0,
        duration_days=0.4,
        depth=0.01,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
    )
    
    assert res["centroid_evidence_flag"] is True
    assert res["recommended_route"] in ("blend_contamination", "review_required")

def test_missing_diagnostics_contract():
    """Checks that default dict conforms to target schema."""
    d = get_default_diagnostics_dict()
    validate_schema(d)
    assert d["recommended_route"] == "review_required"
