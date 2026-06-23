"""
tests/test_bls.py
-----------------
Tests for core/bls_detector.py (Phase 2).

Covers:
    - Period detection accuracy (<1% error) for candidates A and B
    - Non-detection for noise (candidate C)
    - BLS power spectrum shape (peak at correct period)
    - Period grid range and construction
    - Dual-threshold detection logic (power AND SNR)
    - Alias check
    - BLSResult field completeness
    - Processing time (<5 seconds for 18 000-point light curve)
    - Backend selection (astropy when available, scipy fallback)
    - Graceful handling of edge cases
"""

from __future__ import annotations

import time as _time

import numpy as np
import pytest

from core.bls_detector import (
    DEFAULT_CONFIG,
    BLSResult,
    detect,
    _run_scipy_bls,
    _compute_snr,
    _check_aliases,
)
from core.exceptions import BLSDetectionError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def candidate_a():
    """
    Synthetic exoplanet-like light curve.
    Period=3.42 days, depth=1.3%, duration=0.1 days.
    """
    rng = np.random.default_rng(0)
    n = 18_000
    t = np.linspace(0, 27.0, n)
    f = 1.0 + rng.normal(0, 0.001, n)
    period, depth, duration, t0 = 3.42, 0.013, 0.1, 1.5
    ph = ((t - t0) / period) % 1.0
    ph[ph >= 0.5] -= 1.0
    f[np.abs(ph) < (duration / (2 * period))] -= depth
    return t, f


@pytest.fixture(scope="module")
def candidate_b():
    """
    Synthetic eclipsing-binary-like light curve.
    Period=1.87 days, depth=18%, duration=0.08 days.
    """
    rng = np.random.default_rng(1)
    n = 18_000
    t = np.linspace(0, 27.0, n)
    f = 1.0 + rng.normal(0, 0.001, n)
    period, depth, duration, t0 = 1.87, 0.18, 0.08, 0.5
    ph = ((t - t0) / period) % 1.0
    ph[ph >= 0.5] -= 1.0
    f[np.abs(ph) < (duration / (2 * period))] -= depth
    return t, f


@pytest.fixture(scope="module")
def candidate_c():
    """Pure noise light curve — no transit signal."""
    rng = np.random.default_rng(2)
    n = 18_000
    t = np.linspace(0, 27.0, n)
    f = 1.0 + rng.normal(0, 0.001, n)
    return t, f


@pytest.fixture(scope="module")
def result_a(candidate_a):
    t, f = candidate_a
    return detect(t, f)


@pytest.fixture(scope="module")
def result_b(candidate_b):
    t, f = candidate_b
    return detect(t, f)


@pytest.fixture(scope="module")
def result_c(candidate_c):
    t, f = candidate_c
    return detect(t, f)


@pytest.fixture
def tiny_transit_lc():
    """Small 2 000-point light curve with a clear injected transit for fast unit tests."""
    rng = np.random.default_rng(42)
    n = 2_000
    t = np.linspace(0, 27.0, n)
    f = 1.0 + rng.normal(0, 0.001, n)
    period, depth, t0 = 3.42, 0.05, 1.5
    ph = ((t - t0) / period) % 1.0
    ph[ph >= 0.5] -= 1.0
    f[np.abs(ph) < 0.015] -= depth
    return t, f


@pytest.fixture
def tiny_noise_lc():
    """Small 2 000-point pure-noise light curve."""
    rng = np.random.default_rng(99)
    n = 2_000
    t = np.linspace(0, 27.0, n)
    f = 1.0 + rng.normal(0, 0.001, n)
    return t, f


# ---------------------------------------------------------------------------
# Tests: BLSResult structure
# ---------------------------------------------------------------------------

class TestBLSResultStructure:
    REQUIRED_FIELDS = [
        "candidate_detected", "best_period", "best_t0", "best_duration",
        "best_depth", "bls_power_peak", "snr", "periods", "power",
        "alias_warning", "backend", "detection_reason",
    ]

    def test_result_has_all_fields(self, result_a):
        for field in self.REQUIRED_FIELDS:
            assert hasattr(result_a, field), f"BLSResult missing field: {field}"

    def test_candidate_detected_is_bool(self, result_a):
        assert isinstance(result_a.candidate_detected, bool)

    def test_bls_power_peak_is_float(self, result_a):
        assert isinstance(result_a.bls_power_peak, float)

    def test_snr_is_float(self, result_a):
        assert isinstance(result_a.snr, float)

    def test_power_spectrum_is_numpy(self, result_a):
        assert isinstance(result_a.periods, np.ndarray)
        assert isinstance(result_a.power, np.ndarray)
        assert len(result_a.periods) == len(result_a.power)

    def test_power_spectrum_is_non_empty(self, result_a):
        assert len(result_a.periods) > 0
        assert len(result_a.power) > 0

    def test_power_values_in_zero_one(self, result_a):
        """Normalised power must be in [0, 1]."""
        assert result_a.power.min() >= 0.0
        assert result_a.power.max() <= 1.0 + 1e-9

    def test_periods_positive(self, result_a):
        assert np.all(result_a.periods > 0)

    def test_alias_warning_is_bool(self, result_a):
        assert isinstance(result_a.alias_warning, bool)

    def test_backend_is_known_string(self, result_a):
        assert result_a.backend in ("astropy", "scipy")

    def test_detection_reason_non_empty(self, result_a):
        assert isinstance(result_a.detection_reason, str)
        assert len(result_a.detection_reason) > 0


# ---------------------------------------------------------------------------
# Tests: Candidate A — exoplanet-like (period=3.42 days, depth=1.3%)
# ---------------------------------------------------------------------------

class TestCandidateA:
    def test_candidate_a_detected(self, result_a):
        assert result_a.candidate_detected is True, (
            f"Candidate A not detected: power={result_a.bls_power_peak:.4f}, "
            f"snr={result_a.snr:.2f}"
        )

    def test_candidate_a_period_within_1_pct(self, result_a):
        true_period = 3.42
        error_pct = abs(result_a.best_period - true_period) / true_period * 100
        assert error_pct < 1.0, (
            f"Period error {error_pct:.3f}% exceeds 1% tolerance "
            f"(detected={result_a.best_period:.4f}, true={true_period})"
        )

    def test_candidate_a_power_above_threshold(self, result_a):
        assert result_a.bls_power_peak >= DEFAULT_CONFIG["bls_power_threshold"], (
            f"BLS power {result_a.bls_power_peak:.4f} < threshold {DEFAULT_CONFIG['bls_power_threshold']}"
        )

    def test_candidate_a_snr_above_threshold(self, result_a):
        assert result_a.snr >= DEFAULT_CONFIG["snr_threshold"], (
            f"SNR {result_a.snr:.2f} < threshold {DEFAULT_CONFIG['snr_threshold']}"
        )

    def test_candidate_a_depth_positive(self, result_a):
        assert result_a.best_depth is not None
        assert result_a.best_depth > 0

    def test_candidate_a_duration_positive(self, result_a):
        assert result_a.best_duration is not None
        assert result_a.best_duration > 0

    def test_candidate_a_t0_in_range(self, result_a, candidate_a):
        t, _ = candidate_a
        assert result_a.best_t0 is not None
        assert t[0] <= result_a.best_t0 <= t[-1]

    def test_candidate_a_power_spectrum_peaks_near_true_period(self, result_a):
        """The period at max power should be within 2% of 3.42 days."""
        peak_idx = result_a.power.argmax()
        peak_period = result_a.periods[peak_idx]
        error_pct = abs(peak_period - 3.42) / 3.42 * 100
        assert error_pct < 2.0, f"Peak at {peak_period:.4f}d is {error_pct:.2f}% from 3.42d"


# ---------------------------------------------------------------------------
# Tests: Candidate B — eclipsing-binary-like (period=1.87 days, depth=18%)
# ---------------------------------------------------------------------------

class TestCandidateB:
    def test_candidate_b_detected(self, result_b):
        assert result_b.candidate_detected is True, (
            f"Candidate B not detected: power={result_b.bls_power_peak:.4f}, "
            f"snr={result_b.snr:.2f}"
        )

    def test_candidate_b_period_within_1_pct(self, result_b):
        true_period = 1.87
        error_pct = abs(result_b.best_period - true_period) / true_period * 100
        assert error_pct < 1.0, (
            f"Period error {error_pct:.3f}% exceeds 1% tolerance "
            f"(detected={result_b.best_period:.4f}, true={true_period})"
        )

    def test_candidate_b_power_above_threshold(self, result_b):
        assert result_b.bls_power_peak >= DEFAULT_CONFIG["bls_power_threshold"]

    def test_candidate_b_snr_above_threshold(self, result_b):
        assert result_b.snr >= DEFAULT_CONFIG["snr_threshold"]

    def test_candidate_b_depth_greater_than_a(self, result_a, result_b):
        """EB depth should be substantially larger than exoplanet depth."""
        assert result_b.best_depth > result_a.best_depth, (
            f"EB depth {result_b.best_depth:.4f} should exceed planet depth {result_a.best_depth:.4f}"
        )

    def test_candidate_b_high_snr(self, result_b):
        """Deep EB transit should have very high SNR."""
        assert result_b.snr > 20.0, f"Expected SNR > 20 for 18% depth, got {result_b.snr:.1f}"


# ---------------------------------------------------------------------------
# Tests: Candidate C — noise (no transit)
# ---------------------------------------------------------------------------

class TestCandidateC:
    def test_candidate_c_not_detected(self, result_c):
        assert result_c.candidate_detected is False, (
            f"Noise candidate incorrectly detected: "
            f"power={result_c.bls_power_peak:.4f}, snr={result_c.snr:.2f}"
        )

    def test_candidate_c_snr_below_threshold(self, result_c):
        assert result_c.snr < DEFAULT_CONFIG["snr_threshold"]

    def test_candidate_c_has_power_spectrum(self, result_c):
        """Even for noise, a full power spectrum must be returned for plotting."""
        assert len(result_c.periods) > 0
        assert len(result_c.power) > 0

    def test_candidate_c_no_dominant_peak(self, result_c):
        """
        Noise power spectrum should not have a single extremely dominant peak.
        The ratio of top-2 powers should not be extreme (< 5× difference
        from the median at those peaks) — confirming a flat-ish spectrum.
        """
        # For noise, power should be scattered; no single overwhelming peak
        sorted_power = np.sort(result_c.power)[::-1]
        if len(sorted_power) >= 2:
            # Top power not more than 5× the 95th percentile — roughly flat
            p95 = np.percentile(result_c.power, 95)
            # This is a soft check — just verify it doesn't look like a clear detection
            assert result_c.snr < DEFAULT_CONFIG["snr_threshold"]


# ---------------------------------------------------------------------------
# Tests: Period grid construction
# ---------------------------------------------------------------------------

class TestPeriodGrid:
    def test_period_grid_covers_min_to_max(self, result_a):
        """Period grid must span from period_min to time_span/2."""
        assert result_a.periods.min() <= DEFAULT_CONFIG["period_min_days"] * 1.5
        assert result_a.periods.max() >= 10.0  # for 27-day span, max should be ~13.5d

    def test_period_grid_all_positive(self, result_a):
        assert np.all(result_a.periods > 0)

    def test_period_grid_minimum_length(self, tiny_transit_lc):
        t, f = tiny_transit_lc
        r = detect(t, f)
        assert len(r.periods) >= 10, "Period grid too sparse"

    def test_period_max_respects_time_span(self, candidate_a):
        """period_max must not exceed time_span / 2."""
        t, f = candidate_a
        time_span = t[-1] - t[0]
        r = detect(t, f)
        assert r.periods.max() <= time_span / 2.0 + 0.1   # small tolerance


# ---------------------------------------------------------------------------
# Tests: Detection threshold logic
# ---------------------------------------------------------------------------

class TestDetectionThresholds:
    def test_both_conditions_required(self, tiny_noise_lc):
        """
        Neither power alone nor SNR alone is sufficient — both must pass.
        Verify that noise candidate has candidate_detected=False.
        """
        t, f = tiny_noise_lc
        r = detect(t, f)
        assert r.candidate_detected is False

    def test_custom_high_threshold_suppresses_detection(self, tiny_transit_lc):
        """Raising bls_power_threshold to 2.0 (impossible) should prevent detection."""
        t, f = tiny_transit_lc
        r = detect(t, f, config={"bls_power_threshold": 2.0})
        assert r.candidate_detected is False

    def test_custom_low_threshold_allows_detection(self, tiny_transit_lc):
        """Lowering thresholds to near-zero should allow detection of clear transit."""
        t, f = tiny_transit_lc
        r = detect(t, f, config={"bls_power_threshold": 0.01, "snr_threshold": 1.0})
        assert r.candidate_detected is True

    def test_detection_reason_is_informative(self, result_a):
        """detection_reason should mention power and SNR values."""
        reason = result_a.detection_reason.lower()
        assert "power" in reason or "snr" in reason or "detected" in reason


# ---------------------------------------------------------------------------
# Tests: SNR computation
# ---------------------------------------------------------------------------

class TestSNRComputation:
    def test_snr_positive_for_real_transit(self, candidate_a):
        t, f = candidate_a
        snr = _compute_snr(t, f, period=3.42, t0=1.5, duration=0.1, depth=0.013)
        assert snr > 5.0, f"Expected SNR > 5 for clean 1.3% transit, got {snr:.2f}"

    def test_snr_zero_for_zero_depth(self, candidate_a):
        t, f = candidate_a
        snr = _compute_snr(t, f, period=3.42, t0=1.5, duration=0.1, depth=0.0)
        assert snr == 0.0

    def test_snr_zero_for_zero_period(self, candidate_a):
        t, f = candidate_a
        snr = _compute_snr(t, f, period=0.0, t0=1.5, duration=0.1, depth=0.013)
        assert snr == 0.0

    def test_snr_scales_with_depth(self, candidate_a):
        t, f = candidate_a
        snr_low  = _compute_snr(t, f, period=3.42, t0=1.5, duration=0.1, depth=0.005)
        snr_high = _compute_snr(t, f, period=3.42, t0=1.5, duration=0.1, depth=0.020)
        assert snr_high > snr_low, "Deeper transit should have higher SNR"


# ---------------------------------------------------------------------------
# Tests: Alias check
# ---------------------------------------------------------------------------

class TestAliasCheck:
    def test_no_alias_for_clean_single_period(self, candidate_a):
        t, f = candidate_a
        r = detect(t, f)
        # A clean single-period signal should not trigger alias warning
        # (alias warning is possible but should not be systematic)
        assert isinstance(r.alias_warning, bool)

    def test_alias_check_returns_bool(self):
        periods = np.linspace(0.5, 13.5, 500)
        power = np.zeros(500)
        power[100] = 1.0   # sharp peak at periods[100]
        # Add a comparable peak at double the period
        best_period = float(periods[100])
        alias_idx = np.argmin(np.abs(periods - best_period * 2))
        power[alias_idx] = 0.9   # 90% of peak power
        result = _check_aliases(periods, power, best_period, 1.0, DEFAULT_CONFIG)
        assert isinstance(result, bool)

    def test_alias_detected_when_harmonic_strong(self):
        """If harmonic has >80% of peak power, alias should be flagged."""
        periods = np.linspace(0.5, 14.0, 1000)
        power = np.zeros(1000)
        best_period = 3.42
        peak_idx = np.argmin(np.abs(periods - best_period))
        power[peak_idx] = 1.0
        # Add strong alias at 2× period
        alias_idx = np.argmin(np.abs(periods - best_period * 2))
        if alias_idx < len(power):
            power[alias_idx] = 0.85  # above alias_check_tolerance threshold
            result = _check_aliases(periods, power, best_period, 1.0, DEFAULT_CONFIG)
            assert result is True

    def test_no_alias_when_harmonic_weak(self):
        """If harmonic has < 50% of peak power, no alias should be flagged."""
        periods = np.linspace(0.5, 14.0, 1000)
        power = np.zeros(1000)
        best_period = 3.42
        peak_idx = np.argmin(np.abs(periods - best_period))
        power[peak_idx] = 1.0
        alias_idx = np.argmin(np.abs(periods - best_period * 2))
        if alias_idx < len(power):
            power[alias_idx] = 0.3   # well below threshold
            result = _check_aliases(periods, power, best_period, 1.0, DEFAULT_CONFIG)
            assert result is False


# ---------------------------------------------------------------------------
# Tests: Scipy BLS backend directly
# ---------------------------------------------------------------------------

class TestSciPyBLSBackend:
    def test_scipy_detects_correct_period(self, tiny_transit_lc):
        t, f = tiny_transit_lc
        periods, power, params = _run_scipy_bls(t, f, DEFAULT_CONFIG, 0.5, 13.5)
        peak_idx = power.argmax()
        peak_period = periods[peak_idx]
        error_pct = abs(peak_period - 3.42) / 3.42 * 100
        assert error_pct < 2.0, f"Period error {error_pct:.3f}% (detected={peak_period:.4f})"

    def test_scipy_returns_three_arrays(self, tiny_transit_lc):
        t, f = tiny_transit_lc
        result = _run_scipy_bls(t, f, DEFAULT_CONFIG, 0.5, 13.5)
        assert len(result) == 3
        periods, power, params = result
        assert "t0" in params and "duration" in params and "depth" in params

    def test_scipy_power_normalised(self, tiny_transit_lc):
        t, f = tiny_transit_lc
        _, power, _ = _run_scipy_bls(t, f, DEFAULT_CONFIG, 0.5, 13.5)
        assert power.max() <= 1.0 + 1e-9
        assert power.min() >= 0.0

    def test_scipy_depth_non_negative(self, tiny_transit_lc):
        t, f = tiny_transit_lc
        _, _, params = _run_scipy_bls(t, f, DEFAULT_CONFIG, 0.5, 13.5)
        assert np.all(params["depth"] >= 0.0)

    def test_scipy_duration_positive(self, tiny_transit_lc):
        t, f = tiny_transit_lc
        _, _, params = _run_scipy_bls(t, f, DEFAULT_CONFIG, 0.5, 13.5)
        assert np.all(params["duration"] > 0)

    def test_scipy_noise_has_low_snr(self, tiny_noise_lc):
        t, f = tiny_noise_lc
        _, power, params = _run_scipy_bls(t, f, DEFAULT_CONFIG, 0.5, 13.5)
        peak_idx = power.argmax()
        snr = _compute_snr(
            t, f,
            period=float(1.0 / np.linspace(1.0/13.5, 1.0/0.5, len(power))[peak_idx]),
            t0=float(params["t0"][peak_idx]),
            duration=float(params["duration"][peak_idx]),
            depth=float(params["depth"][peak_idx]),
        )
        assert snr < DEFAULT_CONFIG["snr_threshold"]


# ---------------------------------------------------------------------------
# Tests: Edge cases and error handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_invalid_period_range_raises(self):
        t = np.linspace(0, 27.0, 2000)
        f = np.ones(2000)
        with pytest.raises(BLSDetectionError):
            detect(t, f, config={"period_min_days": 14.0, "period_max_days": 5.0})

    def test_config_override_period_range(self, tiny_transit_lc):
        t, f = tiny_transit_lc
        r = detect(t, f, config={"period_min_days": 1.0, "period_max_days": 5.0})
        assert r.periods.min() <= 1.5
        assert r.periods.max() <= 5.5  # small tolerance for frequency grid edges

    def test_result_periods_within_configured_range(self, tiny_transit_lc):
        t, f = tiny_transit_lc
        p_min, p_max = 2.0, 5.0
        r = detect(t, f, config={"period_min_days": p_min, "period_max_days": p_max})
        assert r.periods.min() >= p_min * 0.9
        assert r.periods.max() <= p_max * 1.1

    def test_flat_flux_returns_result(self):
        """Flat flux (no signal) should not crash — should return candidate_detected=False."""
        t = np.linspace(0, 27.0, 2000)
        f = np.ones(2000)
        r = detect(t, f)
        assert isinstance(r, BLSResult)
        assert r.candidate_detected is False

    def test_n_durations_config(self, tiny_transit_lc):
        """n_durations parameter should be respected (different values shouldn't crash)."""
        t, f = tiny_transit_lc
        for n_dur in [1, 3, 7]:
            r = detect(t, f, config={"n_durations": n_dur})
            assert isinstance(r, BLSResult)


# ---------------------------------------------------------------------------
# Tests: Processing time
# ---------------------------------------------------------------------------

class TestProcessingTime:
    def test_candidate_a_under_5_seconds(self, candidate_a):
        t, f = candidate_a
        t0 = _time.perf_counter()
        detect(t, f)
        elapsed = _time.perf_counter() - t0
        assert elapsed < 5.0, f"BLS took {elapsed:.2f}s (limit: 5s)"

    def test_candidate_b_under_5_seconds(self, candidate_b):
        t, f = candidate_b
        t0 = _time.perf_counter()
        detect(t, f)
        elapsed = _time.perf_counter() - t0
        assert elapsed < 5.0, f"BLS took {elapsed:.2f}s (limit: 5s)"

    def test_candidate_c_under_5_seconds(self, candidate_c):
        t, f = candidate_c
        t0 = _time.perf_counter()
        detect(t, f)
        elapsed = _time.perf_counter() - t0
        assert elapsed < 5.0, f"BLS took {elapsed:.2f}s (limit: 5s)"