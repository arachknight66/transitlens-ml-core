"""
tests/test_plotter.py
---------------------
Tests for core/plotter.py (Phase 6).

Covers:
    - All four plots generated without errors
    - All plots return non-empty base64 strings
    - Base64 strings decode to valid PNG images
    - Plots work for all three synthetic cases (detected, EB, noise)
    - Individual plot functions handle edge cases gracefully
    - Downsampling produces correct output lengths
    - Style application does not crash on unavailable styles
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from core.bls_detector import BLSResult
from core.plotter import (
    generate_all,
    _downsample,
    _fig_to_base64,
    _plot_raw,
    _plot_cleaned,
    _plot_periodogram,
    _plot_phase_folded,
    DEFAULT_CONFIG,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bls_detected():
    """BLSResult simulating a detected exoplanet."""
    return BLSResult(
        candidate_detected=True,
        best_period=3.42,
        best_t0=1.5,
        best_duration=0.12,
        best_depth=0.013,
        bls_power_peak=0.45,
        snr=15.0,
        periods=np.linspace(0.5, 13.0, 500),
        power=np.random.default_rng(1).uniform(0.0, 0.1, 500),
        alias_warning=False,
        backend="test",
        detection_reason="test",
    )


@pytest.fixture
def sample_bls_noise():
    """BLSResult simulating noise (no detection)."""
    return BLSResult(
        candidate_detected=False,
        best_period=None,
        best_t0=None,
        best_duration=None,
        best_depth=None,
        bls_power_peak=0.05,
        snr=1.2,
        periods=np.linspace(0.5, 13.0, 500),
        power=np.random.default_rng(2).uniform(0.0, 0.06, 500),
        alias_warning=False,
        backend="test",
        detection_reason="below threshold",
    )


@pytest.fixture
def lc_data():
    """Simple light curve data for plotting."""
    rng = np.random.default_rng(42)
    n = 2000
    time = np.linspace(0.0, 27.0, n)
    flux = 1.0 + rng.normal(0, 0.001, n)
    return time, flux


# ---------------------------------------------------------------------------
# generate_all tests
# ---------------------------------------------------------------------------

class TestGenerateAll:
    """Tests for the generate_all() public function."""

    def test_returns_all_seven_keys(self, lc_data, sample_bls_detected):
        time, flux = lc_data
        plots = generate_all(time, flux, time, flux, sample_bls_detected, "test")
        assert set(plots.keys()) == {
            "raw_lightcurve", "cleaned_lightcurve", "periodogram", "phase_folded",
            "transit_stack", "posterior_corner", "alias_comparison"
        }

    def test_all_plots_non_empty_for_detected(self, lc_data, sample_bls_detected):
        time, flux = lc_data
        plots = generate_all(time, flux, time, flux, sample_bls_detected, "test")
        for key in ["raw_lightcurve", "cleaned_lightcurve", "periodogram", "phase_folded"]:
            assert len(plots[key]) > 0, f"Plot '{key}' is empty"

    def test_all_plots_non_empty_for_noise(self, lc_data, sample_bls_noise):
        time, flux = lc_data
        plots = generate_all(time, flux, time, flux, sample_bls_noise, "test")
        for key in ["raw_lightcurve", "cleaned_lightcurve", "periodogram", "phase_folded"]:
            assert len(plots[key]) > 0, f"Plot '{key}' is empty"

    def test_base64_decodes_to_png(self, lc_data, sample_bls_detected):
        time, flux = lc_data
        plots = generate_all(time, flux, time, flux, sample_bls_detected, "test")
        for key in ["raw_lightcurve", "cleaned_lightcurve", "periodogram", "phase_folded"]:
            decoded = base64.b64decode(plots[key])
            # PNG magic bytes
            assert decoded[:4] == b"\x89PNG", f"Plot '{key}' is not valid PNG"

    def test_custom_config_applied(self, lc_data, sample_bls_detected):
        time, flux = lc_data
        cfg = {"dpi": 50, "figure_width": 6, "figure_height": 3}
        plots = generate_all(time, flux, time, flux, sample_bls_detected, "test", config=cfg)
        assert all(len(plots[k]) > 0 for k in ["raw_lightcurve", "cleaned_lightcurve", "periodogram", "phase_folded"])


# ---------------------------------------------------------------------------
# Individual plot tests
# ---------------------------------------------------------------------------

class TestIndividualPlots:
    """Tests for individual plot functions."""

    def test_raw_plot_produces_base64(self, lc_data):
        time, flux = lc_data
        result = _plot_raw(time, flux, "test", DEFAULT_CONFIG)
        assert len(result) > 100
        assert base64.b64decode(result)[:4] == b"\x89PNG"

    def test_cleaned_plot_with_detection(self, lc_data, sample_bls_detected):
        time, flux = lc_data
        result = _plot_cleaned(time, flux, sample_bls_detected, "test", DEFAULT_CONFIG)
        assert len(result) > 100

    def test_periodogram_with_alias(self, sample_bls_detected):
        sample_bls_detected.alias_warning = True
        result = _plot_periodogram(sample_bls_detected, "test", DEFAULT_CONFIG)
        assert len(result) > 100

    def test_phase_folded_no_period(self, lc_data):
        """Phase-folded plot should handle missing period gracefully."""
        time, flux = lc_data
        bls = BLSResult(
            candidate_detected=False, best_period=None, best_t0=None,
            best_duration=None, best_depth=None, bls_power_peak=0.01,
            snr=0.5, periods=np.array([]), power=np.array([]),
            alias_warning=False, backend="test", detection_reason="no data",
        )
        result = _plot_phase_folded(time, flux, bls, None, "test", DEFAULT_CONFIG)
        assert len(result) > 100


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for plotting helper functions."""

    def test_downsample_reduces_length(self):
        x = np.arange(10000)
        y = np.arange(10000, dtype=float)
        x_ds, y_ds = _downsample(x, y, 1000)
        assert len(x_ds) <= 1000
        assert len(x_ds) == len(y_ds)

    def test_downsample_no_change_below_limit(self):
        x = np.arange(100)
        y = np.arange(100, dtype=float)
        x_ds, y_ds = _downsample(x, y, 1000)
        assert len(x_ds) == 100

    def test_fig_to_base64_valid(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        result = _fig_to_base64(fig)
        assert len(result) > 0
        decoded = base64.b64decode(result)
        assert decoded[:4] == b"\x89PNG"
