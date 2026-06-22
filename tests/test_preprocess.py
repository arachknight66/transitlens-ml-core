"""
tests/test_preprocess.py
------------------------
Tests for core/preprocess.py and core/utils.py (Phase 1).

Covers:
    - NaN removal (synchronised on time and flux)
    - Sigma clipping
    - Detrending (both methods)
    - Re-normalisation
    - Gap detection
    - Minimum data quality gates
    - Input validation (InvalidInputError)
    - Full pipeline end-to-end
"""

from __future__ import annotations

import numpy as np
import pytest

from core.exceptions import (
    InsufficientDataError,
    InvalidInputError,
)
from core.preprocess import (
    DEFAULT_CONFIG,
    PreprocessResult,
    clean,
    _check_quality_gates,
    _remove_nans,
    _renormalise,
    _sigma_clip_flux,
)
from core.utils import (
    Gap,
    PhaseBins,
    bin_phase_folded,
    detect_gaps,
    estimate_cadence,
    phase_fold,
    running_median,
    sigma_clip,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_lc_500():
    """Minimal clean light curve with 500 points and no transit."""
    rng = np.random.default_rng(42)
    time = np.linspace(0, 27.0, 500)
    flux = 1.0 + rng.normal(0, 0.001, size=500)
    return time, flux


@pytest.fixture
def transit_lc_18000():
    """
    Synthetic 18 000-point TESS-like light curve with an injected transit.

    Transit: period=3.42 days, depth=0.013, duration=0.1 days.
    """
    rng = np.random.default_rng(0)
    n = 18_000
    time = np.linspace(0, 27.0, n)
    flux = 1.0 + rng.normal(0, 0.001, size=n)

    # Inject box transit
    period = 3.42
    duration = 0.1
    t0 = 1.5
    phase = ((time - t0) / period) % 1.0
    phase[phase >= 0.5] -= 1.0
    in_transit = np.abs(phase) < (duration / (2 * period))
    flux[in_transit] -= 0.013

    return time, flux


@pytest.fixture
def lc_with_nans(transit_lc_18000):
    """Light curve with NaNs injected at regular intervals."""
    time, flux = transit_lc_18000
    time_nan = time.copy().astype(float)
    flux_nan = flux.copy().astype(float)
    # Inject 50 NaNs
    nan_indices = np.arange(10, 500, 10)
    flux_nan[nan_indices] = np.nan
    return time_nan, flux_nan, nan_indices


@pytest.fixture
def lc_with_outliers(transit_lc_18000):
    """Light curve with extreme outliers injected."""
    time, flux = transit_lc_18000
    flux_out = flux.copy()
    # Inject upward spikes (flares)
    flux_out[100] = 1.5
    flux_out[500] = 1.8
    flux_out[1000] = 2.0
    # Inject downward spike (not a transit)
    flux_out[200] = 0.1
    return time, flux_out


@pytest.fixture
def lc_with_gap():
    """Light curve with a 2-day gap in the middle."""
    rng = np.random.default_rng(7)
    # First segment: 0–12 days
    t1 = np.linspace(0, 12.0, 6500)
    f1 = 1.0 + rng.normal(0, 0.001, size=6500)
    # Second segment: 14–27 days (2-day gap)
    t2 = np.linspace(14.0, 27.0, 7000)
    f2 = 1.0 + rng.normal(0, 0.001, size=7000)

    time = np.concatenate([t1, t2])
    flux = np.concatenate([f1, f2])
    return time, flux


# ---------------------------------------------------------------------------
# Tests: core/utils.py — phase_fold
# ---------------------------------------------------------------------------

class TestPhaseFold:
    def test_range_is_minus_half_to_half(self, clean_lc_500):
        time, _ = clean_lc_500
        phase = phase_fold(time, period=3.42, t0=1.5)
        assert np.all(phase >= -0.5)
        assert np.all(phase < 0.5)

    def test_transit_centre_at_zero(self):
        """t0 should map exactly to phase 0."""
        time = np.array([1.5, 1.5 + 3.42, 1.5 + 2 * 3.42])
        phase = phase_fold(time, period=3.42, t0=1.5)
        np.testing.assert_allclose(phase, [0.0, 0.0, 0.0], atol=1e-10)

    def test_half_period_maps_to_plus_minus_half(self):
        time = np.array([1.5 + 3.42 / 2])
        phase = phase_fold(time, period=3.42, t0=1.5)
        assert abs(abs(phase[0]) - 0.5) < 1e-10

    def test_raises_on_non_positive_period(self, clean_lc_500):
        time, _ = clean_lc_500
        with pytest.raises(ValueError, match="positive"):
            phase_fold(time, period=0.0, t0=0.0)

        with pytest.raises(ValueError, match="positive"):
            phase_fold(time, period=-1.0, t0=0.0)


# ---------------------------------------------------------------------------
# Tests: core/utils.py — sigma_clip
# ---------------------------------------------------------------------------

class TestSigmaClip:
    def test_removes_large_outlier(self):
        rng = np.random.default_rng(1)
        values = rng.normal(1.0, 0.001, size=1000)
        values[50] = 5.0   # 4000-sigma outlier
        mask = sigma_clip(values, sigma_upper=5.0, sigma_lower=5.0, max_iter=3)
        assert not mask[50], "Extreme outlier should be clipped"

    def test_keeps_inlier_fraction(self):
        rng = np.random.default_rng(2)
        values = rng.normal(1.0, 0.001, size=1000)
        values[0] = 100.0
        mask = sigma_clip(values, sigma_upper=5.0, sigma_lower=5.0, max_iter=3)
        assert mask.sum() >= 995, "Should keep ≥99.5% of inliers"

    def test_returns_bool_array(self):
        values = np.linspace(0, 1, 100)
        mask = sigma_clip(values, sigma_upper=3.0, sigma_lower=3.0, max_iter=2)
        assert mask.dtype == bool

    def test_convergence_on_uniform_data(self):
        values = np.ones(500)
        mask = sigma_clip(values, sigma_upper=5.0, sigma_lower=5.0, max_iter=5)
        assert mask.all(), "Uniform data — nothing should be clipped"

    def test_asymmetric_upper_clip(self):
        """A tight upper sigma should clip upward spikes but not downward."""
        rng = np.random.default_rng(3)
        values = rng.normal(1.0, 0.001, size=500)
        values[10] = 1.05   # large upward spike
        values[20] = 0.95   # deep downward spike (transit-like)
        mask = sigma_clip(values, sigma_upper=3.0, sigma_lower=30.0, max_iter=3)
        assert not mask[10], "Upward spike should be clipped"
        # Downward should survive with loose lower sigma
        assert mask[20], "Deep downward spike should survive with sigma_lower=30"


# ---------------------------------------------------------------------------
# Tests: core/utils.py — running_median
# ---------------------------------------------------------------------------

class TestRunningMedian:
    def test_length_preserved(self):
        values = np.random.default_rng(4).normal(1.0, 0.01, size=200)
        result = running_median(values, window_size=11)
        assert len(result) == len(values)

    def test_flat_signal_unchanged(self):
        values = np.ones(100)
        result = running_median(values, window_size=5)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_smooths_trend(self):
        """Running median of a linear ramp should be close to the ramp itself."""
        values = np.linspace(0, 1, 200)
        result = running_median(values, window_size=5)
        np.testing.assert_allclose(result[5:-5], values[5:-5], atol=0.01)

    def test_raises_on_zero_window(self):
        with pytest.raises(ValueError):
            running_median(np.ones(50), window_size=0)


# ---------------------------------------------------------------------------
# Tests: core/utils.py — bin_phase_folded
# ---------------------------------------------------------------------------

class TestBinPhaseFolded:
    def test_returns_named_tuple(self):
        phase = np.linspace(-0.5, 0.5, 500, endpoint=False)
        flux = np.ones(500)
        result = bin_phase_folded(phase, flux, n_bins=50)
        assert isinstance(result, PhaseBins)

    def test_bin_centres_span_minus_half_to_half(self):
        phase = np.linspace(-0.5, 0.5, 500, endpoint=False)
        flux = np.ones(500)
        result = bin_phase_folded(phase, flux, n_bins=100)
        assert result.bin_centres[0] >= -0.5
        assert result.bin_centres[-1] < 0.5

    def test_flat_flux_gives_uniform_means(self):
        rng = np.random.default_rng(5)
        phase = rng.uniform(-0.5, 0.5, size=5000)
        flux = np.ones(5000)
        result = bin_phase_folded(phase, flux, n_bins=100)
        valid = ~np.isnan(result.bin_means)
        np.testing.assert_allclose(result.bin_means[valid], 1.0, atol=1e-10)

    def test_transit_dip_visible_in_bins(self):
        """Injected dip should appear as lower mean in transit bins."""
        n = 10_000
        rng = np.random.default_rng(6)
        phase = rng.uniform(-0.5, 0.5, size=n)
        flux = np.ones(n)
        in_transit = np.abs(phase) < 0.02
        flux[in_transit] -= 0.02

        result = bin_phase_folded(phase, flux, n_bins=100)
        # Central bins (near phase 0) should have lower mean
        central = np.abs(result.bin_centres) < 0.02
        assert result.bin_means[central].mean() < 0.99

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError, match="equal length"):
            bin_phase_folded(np.ones(10), np.ones(11))


# ---------------------------------------------------------------------------
# Tests: core/utils.py — detect_gaps
# ---------------------------------------------------------------------------

class TestDetectGaps:
    def test_detects_two_day_gap(self, lc_with_gap):
        time, _ = lc_with_gap
        gaps = detect_gaps(time, threshold_factor=5.0)
        assert len(gaps) >= 1
        # The gap should be around 2 days
        assert any(g.gap_days > 1.5 for g in gaps)

    def test_no_gaps_in_uniform_series(self, clean_lc_500):
        time, _ = clean_lc_500
        gaps = detect_gaps(time, threshold_factor=5.0)
        assert len(gaps) == 0

    def test_gap_indices_are_adjacent(self, lc_with_gap):
        time, _ = lc_with_gap
        gaps = detect_gaps(time, threshold_factor=5.0)
        for gap in gaps:
            assert gap.end_idx == gap.start_idx + 1

    def test_returns_gap_named_tuples(self, lc_with_gap):
        time, _ = lc_with_gap
        gaps = detect_gaps(time, threshold_factor=5.0)
        for g in gaps:
            assert isinstance(g, Gap)
            assert isinstance(g.start_idx, int)
            assert isinstance(g.end_idx, int)
            assert isinstance(g.gap_days, float)

    def test_empty_array_returns_empty(self):
        gaps = detect_gaps(np.array([1.0]), threshold_factor=5.0)
        assert gaps == []


# ---------------------------------------------------------------------------
# Tests: core/preprocess.py — _remove_nans
# ---------------------------------------------------------------------------

class TestRemoveNans:
    def test_removes_nan_flux(self, lc_with_nans):
        time, flux, nan_idx = lc_with_nans
        t_clean, f_clean = _remove_nans(time, flux)
        assert not np.any(np.isnan(f_clean))

    def test_time_flux_remain_synchronised(self, lc_with_nans):
        time, flux, _ = lc_with_nans
        t_clean, f_clean = _remove_nans(time, flux)
        assert len(t_clean) == len(f_clean)

    def test_no_nans_input_unchanged(self, transit_lc_18000):
        time, flux = transit_lc_18000
        t_clean, f_clean = _remove_nans(time, flux)
        assert len(t_clean) == len(time)

    def test_removes_nan_time(self):
        time = np.array([1.0, np.nan, 3.0, 4.0])
        flux = np.array([1.0, 1.0, 1.0, 1.0])
        t_clean, f_clean = _remove_nans(time, flux)
        assert len(t_clean) == 3
        assert np.all(np.isfinite(t_clean))


# ---------------------------------------------------------------------------
# Tests: core/preprocess.py — _sigma_clip_flux
# ---------------------------------------------------------------------------

class TestSigmaClipFlux:
    def test_removes_extreme_outliers(self, lc_with_outliers):
        time, flux = lc_with_outliers
        t_clean, f_clean = _sigma_clip_flux(
            time, flux, sigma_upper=5.0, sigma_lower=5.0, max_iter=3
        )
        assert np.max(f_clean) < 1.1, "Extreme upward spike should be removed"

    def test_arrays_stay_same_length(self, transit_lc_18000):
        time, flux = transit_lc_18000
        t_clip, f_clip = _sigma_clip_flux(
            time, flux, sigma_upper=5.0, sigma_lower=5.0, max_iter=3
        )
        assert len(t_clip) == len(f_clip)

    def test_inliers_preserved(self, transit_lc_18000):
        time, flux = transit_lc_18000
        t_clip, f_clip = _sigma_clip_flux(
            time, flux, sigma_upper=5.0, sigma_lower=5.0, max_iter=3
        )
        # With σ=0.001 noise, 5σ = 0.005; transit depth 0.013 should survive
        assert len(f_clip) > 0.97 * len(flux)


# ---------------------------------------------------------------------------
# Tests: core/preprocess.py — _renormalise
# ---------------------------------------------------------------------------

class TestRenormalise:
    def test_output_median_is_one(self):
        flux = np.array([1.1, 1.2, 1.1, 1.0, 1.15, 1.05])
        result = _renormalise(flux)
        assert abs(np.median(result) - 1.0) < 1e-6

    def test_shape_preserved(self, transit_lc_18000):
        _, flux = transit_lc_18000
        result = _renormalise(flux)
        assert result.shape == flux.shape


# ---------------------------------------------------------------------------
# Tests: core/preprocess.py — _check_quality_gates
# ---------------------------------------------------------------------------

class TestQualityGates:
    def test_raises_on_too_few_points(self):
        with pytest.raises(InsufficientDataError, match="Insufficient data"):
            _check_quality_gates(
                n_points=100,
                time_span_days=20.0,
                fraction_retained=0.99,
                cfg=DEFAULT_CONFIG,
            )

    def test_raises_on_short_baseline(self):
        with pytest.raises(InsufficientDataError, match="time baseline"):
            _check_quality_gates(
                n_points=1000,
                time_span_days=2.0,
                fraction_retained=0.99,
                cfg=DEFAULT_CONFIG,
            )

    def test_raises_on_low_retention(self):
        with pytest.raises(InsufficientDataError, match="outliers"):
            _check_quality_gates(
                n_points=1000,
                time_span_days=20.0,
                fraction_retained=0.60,
                cfg=DEFAULT_CONFIG,
            )

    def test_passes_on_good_data(self):
        # Should not raise
        _check_quality_gates(
            n_points=500,
            time_span_days=10.0,
            fraction_retained=0.95,
            cfg=DEFAULT_CONFIG,
        )


# ---------------------------------------------------------------------------
# Tests: core/preprocess.py — clean() input validation
# ---------------------------------------------------------------------------

class TestCleanValidation:
    def test_raises_on_length_mismatch(self):
        with pytest.raises(InvalidInputError, match="equal length"):
            clean(np.ones(100), np.ones(200))

    def test_raises_on_non_monotonic_time(self):
        time = np.linspace(0, 27, 1000)
        time[500] = time[499]  # non-strictly increasing
        flux = np.ones(1000)
        with pytest.raises(InvalidInputError, match="monotonically"):
            clean(time, flux)

    def test_raises_on_infinite_flux(self):
        time = np.linspace(0, 27, 1000)
        flux = np.ones(1000)
        flux[50] = np.inf
        with pytest.raises(InvalidInputError, match="infinite"):
            clean(time, flux)

    def test_raises_on_empty_arrays(self):
        with pytest.raises(InvalidInputError, match="empty"):
            clean(np.array([]), np.array([]))

    def test_raises_on_too_few_points(self):
        """200 clean points is below the 500-point minimum."""
        time = np.linspace(0, 27, 200)
        flux = np.ones(200)
        with pytest.raises(InsufficientDataError):
            clean(time, flux)


# ---------------------------------------------------------------------------
# Tests: core/preprocess.py — clean() end-to-end
# ---------------------------------------------------------------------------

class TestCleanEndToEnd:
    def test_returns_preprocess_result(self, transit_lc_18000):
        time, flux = transit_lc_18000
        result = clean(time, flux)
        assert isinstance(result, PreprocessResult)

    def test_output_flux_has_median_one(self, transit_lc_18000):
        time, flux = transit_lc_18000
        result = clean(time, flux)
        assert abs(np.median(result.flux) - 1.0) < 1e-3

    def test_output_has_no_nans(self, lc_with_nans):
        time, flux, _ = lc_with_nans
        result = clean(time, flux)
        assert not np.any(np.isnan(result.flux))
        assert not np.any(np.isnan(result.time))

    def test_output_time_is_monotonic(self, transit_lc_18000):
        time, flux = transit_lc_18000
        result = clean(time, flux)
        diffs = np.diff(result.time)
        assert np.all(diffs > 0)

    def test_fraction_retained_stored(self, transit_lc_18000):
        time, flux = transit_lc_18000
        result = clean(time, flux)
        assert 0.0 < result.fraction_retained <= 1.0

    def test_gaps_detected_in_gapped_lc(self, lc_with_gap):
        time, flux = lc_with_gap
        result = clean(time, flux)
        assert len(result.gaps) >= 1

    def test_no_gaps_in_continuous_lc(self, transit_lc_18000):
        time, flux = transit_lc_18000
        result = clean(time, flux)
        assert len(result.gaps) == 0

    def test_config_override_respected(self, transit_lc_18000):
        """Passing config={"detrend_method": "polynomial"} should use that method."""
        time, flux = transit_lc_18000
        result = clean(time, flux, config={"detrend_method": "polynomial"})
        assert result.detrend_method == "polynomial"

    def test_outliers_removed(self, lc_with_outliers):
        time, flux = lc_with_outliers
        result = clean(time, flux)
        assert np.max(result.flux) < 1.1, "Extreme flux spike should be removed"

    def test_provenance_counts_are_consistent(self, lc_with_nans):
        time, flux, nan_idx = lc_with_nans
        result = clean(time, flux)
        assert result.n_original == len(time)
        assert result.n_after_nan < result.n_original
        assert result.n_after_clip <= result.n_after_nan
        assert result.n_points == len(result.flux)

    def test_transit_dip_survives_preprocessing(self, transit_lc_18000):
        """The 1.3% transit dip must not be removed by 5-sigma clipping."""
        time, flux = transit_lc_18000
        result = clean(time, flux)
        # Minimum cleaned flux should be below 0.99 (transit dip present)
        assert np.min(result.flux) < 0.99

    def test_polynomial_detrend_method(self, transit_lc_18000):
        time, flux = transit_lc_18000
        result = clean(time, flux, config={"detrend_method": "polynomial"})
        assert isinstance(result, PreprocessResult)
        assert abs(np.median(result.flux) - 1.0) < 1e-3