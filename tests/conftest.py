"""
tests/conftest.py
-----------------
Shared pytest fixtures for transitlens-ml-core test suite.

Fixtures provided:
    tiny_lc_clean       — 1000-point clean light curve, no transit
    tiny_lc_transit     — 1000-point clean light curve with injected transit
    synthetic_cases     — dict with candidate_a/b/c raw time+flux arrays
    mock_bls_result     — pre-built BLSResult for feature extractor testing
    rule_config         — parsed rule_config.yaml as dict
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# Minimal clean light curves
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_lc_clean():
    """1000-point preprocessed light curve with no transit signal."""
    rng = np.random.default_rng(12345)
    n = 1000
    time = np.linspace(0.0, 20.0, n)
    flux = 1.0 + rng.normal(0, 0.001, n)
    return time, flux


@pytest.fixture
def tiny_lc_transit():
    """1000-point preprocessed light curve with a clear injected transit."""
    rng = np.random.default_rng(54321)
    n = 1000
    time = np.linspace(0.0, 20.0, n)
    flux = 1.0 + rng.normal(0, 0.001, n)

    # Inject a transit at period=4.0 days, depth=0.015, duration=0.1 days
    period, depth, duration, t0 = 4.0, 0.015, 0.1, 2.0
    phase = ((time - t0) / period) % 1.0
    in_transit = (phase < duration / period) | (phase > 1.0 - duration / period)
    flux[in_transit] -= depth

    return time, flux, {"period": period, "depth": depth, "duration": duration, "t0": t0}


# ---------------------------------------------------------------------------
# Full synthetic cases (18000 points each)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_cases():
    """
    Three synthetic light curves matching the hackathon spec.

    Returns a dict with keys 'a', 'b', 'c', each containing:
        time, flux, metadata (including true_label, true_period, etc.)
    """
    rng = np.random.default_rng(42)
    n = 18000
    time = np.linspace(0.0, 27.0, n)
    noise = 0.001

    # Candidate A — exoplanet
    flux_a = 1.0 + rng.normal(0, noise, n)
    period_a, depth_a, dur_a = 3.42, 0.013, 0.12
    phase_a = ((time - 1.5) / period_a) % 1.0
    mask_a = (phase_a < dur_a / period_a) | (phase_a > 1.0 - dur_a / period_a)
    flux_a[mask_a] -= depth_a

    # Candidate B — eclipsing binary
    flux_b = 1.0 + rng.normal(0, noise, n)
    period_b, depth_b, dur_b = 1.87, 0.18, 0.15
    hp_b = (dur_b / period_b) / 2.0
    phase_b = ((time - 0.8) / period_b) % 1.0
    for i, ph in enumerate(phase_b):
        if ph < hp_b:
            flux_b[i] -= depth_b * (1.0 - ph / hp_b)
        elif ph > 1.0 - hp_b:
            flux_b[i] -= depth_b * (1.0 - (1.0 - ph) / hp_b)
        elif abs(ph - 0.5) < hp_b:
            flux_b[i] -= 0.08 * (1.0 - abs(ph - 0.5) / hp_b)

    # Candidate C — noise
    flux_c = 1.0 + np.random.default_rng(3).normal(0, noise, n)

    return {
        "a": {
            "time": time.copy(), "flux": flux_a,
            "metadata": {
                "target_id": "candidate_a", "true_period": period_a,
                "true_depth": depth_a, "true_label": "exoplanet_transit",
            },
        },
        "b": {
            "time": time.copy(), "flux": flux_b,
            "metadata": {
                "target_id": "candidate_b", "true_period": period_b,
                "true_depth": depth_b, "true_label": "eclipsing_binary",
            },
        },
        "c": {
            "time": time.copy(), "flux": flux_c,
            "metadata": {
                "target_id": "candidate_c", "true_label": "stellar_variability_or_other",
            },
        },
    }


# ---------------------------------------------------------------------------
# Mock BLS result
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_bls_result():
    """Pre-built BLSResult for testing feature extractor in isolation."""
    from core.bls_detector import BLSResult

    return BLSResult(
        candidate_detected=True,
        best_period=3.42,
        best_t0=1.5,
        best_duration=0.12,
        best_depth=0.013,
        bls_power_peak=0.45,
        snr=15.0,
        periods=np.linspace(0.5, 13.0, 5000),
        power=np.random.default_rng(99).uniform(0.0, 0.1, 5000),
        alias_warning=False,
        backend="test",
        detection_reason="test fixture",
    )


# ---------------------------------------------------------------------------
# Rule config
# ---------------------------------------------------------------------------

@pytest.fixture
def rule_config():
    """Parsed models/rule_config.yaml as a Python dict."""
    path = Path(__file__).parent.parent / "models" / "rule_config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)
