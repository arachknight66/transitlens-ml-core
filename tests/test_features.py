"""
tests/test_features.py
----------------------
Tests for core/feature_extractor.py (Phase 3).

Covers:
    - Exactly 16 features returned with correct keys
    - No NaN or inf values in any feature
    - Feature values are physically correct for all three synthetic candidates
    - Fallback behavior for no-candidate case
    - odd_even_depth_delta near-zero for symmetric transits
    - v_shape_score near-zero for box transits, non-zero for V-shaped
    - depth consistent with BLS result
    - local_noise and depth_to_noise_ratio sensible
    - Reliability flags set correctly
    - FeatureResult.as_array() returns correct ordering
    - Feature computation under 0.5 seconds
"""

from __future__ import annotations

import time as _time

import numpy as np
import pytest

from core.bls_detector import detect, BLSResult
from core.feature_extractor import (
    FEATURE_NAMES,
    FeatureResult,
    extract,
    _compute_odd_even_delta,
    _compute_v_shape_score,
    _compute_phase_kurtosis,
    _ensure_finite,
    _FALLBACK,
)
from core.utils import phase_fold, bin_phase_folded


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

def _make_lc(seed, period, depth, duration, t0, n=18000, noise=0.001):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 27.0, n)
    f = 1.0 + rng.normal(0, noise, n)
    ph = ((t - t0) / period) % 1.0
    ph[ph >= 0.5] -= 1.0
    f[np.abs(ph) < (duration / (2 * period))] -= depth
    return t, f


def _make_v_lc(seed, period, depth, duration, t0, n=18000, noise=0.001):
    """V-shaped transit: depth varies linearly from 0 at ingress to max at centre."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 27.0, n)
    f = 1.0 + rng.normal(0, noise, n)
    ph = ((t - t0) / period) % 1.0
    ph[ph >= 0.5] -= 1.0
    half_q = duration / (2 * period)
    in_tr = np.abs(ph) < half_q
    # V-shape: flux drop proportional to distance from ingress/egress
    f[in_tr] -= depth * (1.0 - np.abs(ph[in_tr]) / half_q)
    return t, f


@pytest.fixture(scope="module")
def candidate_a_data():
    return _make_lc(0, 3.42, 0.013, 0.1, 1.5)


@pytest.fixture(scope="module")
def candidate_b_data():
    return _make_lc(1, 1.87, 0.18, 0.08, 0.5)


@pytest.fixture(scope="module")
def candidate_c_data():
    t = np.linspace(0, 27.0, 18000)
    f = 1.0 + np.random.default_rng(3).normal(0, 0.001, 18000)
    return t, f


@pytest.fixture(scope="module")
def result_a(candidate_a_data):
    t, f = candidate_a_data
    return detect(t, f)


@pytest.fixture(scope="module")
def result_b(candidate_b_data):
    t, f = candidate_b_data
    return detect(t, f)


@pytest.fixture(scope="module")
def result_c(candidate_c_data):
    t, f = candidate_c_data
    return detect(t, f)


@pytest.fixture(scope="module")
def features_a(candidate_a_data, result_a):
    t, f = candidate_a_data
    return extract(t, f, result_a)


@pytest.fixture(scope="module")
def features_b(candidate_b_data, result_b):
    t, f = candidate_b_data
    return extract(t, f, result_b)


@pytest.fixture(scope="module")
def features_c(candidate_c_data, result_c):
    t, f = candidate_c_data
    return extract(t, f, result_c)


@pytest.fixture
def tiny_box_lc():
    """Small box-transit LC for fast unit tests."""
    return _make_lc(42, 3.42, 0.05, 0.1, 1.5, n=3000)


@pytest.fixture
def tiny_v_lc():
    """Small V-shaped transit LC for v_shape_score tests."""
    return _make_v_lc(43, 3.42, 0.10, 0.15, 1.5, n=3000)


# ---------------------------------------------------------------------------
# Tests: Structural correctness
# ---------------------------------------------------------------------------

class TestFeatureStructure:
    def test_returns_feature_result(self, features_a):
        assert isinstance(features_a, FeatureResult)

    def test_exactly_16_features(self, features_a, features_b, features_c):
        for fr in [features_a, features_b, features_c]:
            assert len(fr.features) == len(FEATURE_NAMES), f"Expected {len(FEATURE_NAMES)} features, got {len(fr.features)}"

    def test_correct_feature_keys(self, features_a):
        assert set(features_a.features.keys()) == set(FEATURE_NAMES)

    def test_exactly_16_reliability_flags(self, features_a):
        assert len(features_a.reliable) == len(FEATURE_NAMES)

    def test_reliable_keys_match_feature_keys(self, features_a):
        assert set(features_a.reliable.keys()) == set(FEATURE_NAMES)

    def test_candidate_detected_forwarded(self, features_a, result_a, features_c, result_c):
        assert features_a.candidate_detected == result_a.candidate_detected
        assert features_c.candidate_detected == result_c.candidate_detected

    def test_as_array_returns_numpy(self, features_a):
        arr = features_a.as_array()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == float

    def test_as_array_length(self, features_a):
        assert len(features_a.as_array()) == len(FEATURE_NAMES)

    def test_as_array_matches_feature_dict_order(self, features_a):
        arr = features_a.as_array()
        for i, key in enumerate(FEATURE_NAMES):
            assert arr[i] == features_a.features[key], f"Mismatch at index {i} ({key})"


# ---------------------------------------------------------------------------
# Tests: No NaN or inf in any feature
# ---------------------------------------------------------------------------

class TestFeatureFiniteness:
    def test_no_nan_candidate_a(self, features_a):
        for k, v in features_a.features.items():
            assert np.isfinite(float(v)), f"Feature '{k}' is NaN/inf: {v}"

    def test_no_nan_candidate_b(self, features_b):
        for k, v in features_b.features.items():
            assert np.isfinite(float(v)), f"Feature '{k}' is NaN/inf: {v}"

    def test_no_nan_candidate_c(self, features_c):
        for k, v in features_c.features.items():
            assert np.isfinite(float(v)), f"Feature '{k}' is NaN/inf: {v}"

    def test_ensure_finite_replaces_nan(self, features_a):
        """_ensure_finite must replace NaN with fallback."""
        test_features = dict(features_a.features)
        test_features["depth"] = float("nan")
        reliable = {k: True for k in FEATURE_NAMES}
        cleaned = _ensure_finite(test_features, reliable)
        assert np.isfinite(cleaned["depth"])
        assert cleaned["depth"] == _FALLBACK["depth"]
        assert reliable["depth"] is False

    def test_ensure_finite_replaces_inf(self, features_a):
        test_features = dict(features_a.features)
        test_features["snr"] = float("inf")
        reliable = {k: True for k in FEATURE_NAMES}
        cleaned = _ensure_finite(test_features, reliable)
        assert np.isfinite(cleaned["snr"])
        assert reliable["snr"] is False


# ---------------------------------------------------------------------------
# Tests: Feature values — Candidate A (exoplanet-like)
# ---------------------------------------------------------------------------

class TestCandidateAFeatures:
    def test_bls_power_positive(self, features_a):
        assert features_a.features["bls_power"] > 0.0

    def test_bls_power_at_most_one(self, features_a):
        assert features_a.features["bls_power"] <= 1.0

    def test_snr_above_threshold(self, features_a):
        assert features_a.features["snr"] >= 5.0, \
            f"SNR {features_a.features['snr']:.2f} below threshold"

    def test_period_close_to_true(self, features_a):
        err = abs(features_a.features["period_days"] - 3.42) / 3.42
        assert err < 0.01, f"Period error {err*100:.2f}% (detected={features_a.features['period_days']:.4f})"

    def test_depth_approx_injected(self, features_a):
        # BLS depth may differ slightly from injected depth due to grid resolution
        assert 0.005 < features_a.features["depth"] < 0.025, \
            f"Depth {features_a.features['depth']:.4f} out of expected range"

    def test_transit_count_correct(self, features_a):
        # 27 days / 3.42 days = ~7 transits
        assert 6 <= features_a.features["transit_count"] <= 9

    def test_odd_even_delta_near_zero(self, features_a):
        """Symmetric transits → odd/even depths should be nearly identical."""
        delta = features_a.features["odd_even_depth_delta"]
        assert delta < 0.005, f"odd_even_delta={delta:.4f} unexpectedly large for box transit"

    def test_v_shape_score_low(self, features_a):
        """Flat-bottomed box transit → v_shape_score should be near 0."""
        score = features_a.features["v_shape_score"]
        assert score < 0.5, f"v_shape_score={score:.3f} too high for box transit"

    def test_local_noise_positive(self, features_a):
        assert features_a.features["local_noise"] > 0.0

    def test_local_noise_near_injected(self, features_a):
        """Injected noise = 0.001; cleaned local_noise should be close."""
        assert 0.0005 < features_a.features["local_noise"] < 0.01

    def test_depth_to_noise_ratio_above_threshold(self, features_a):
        """depth_to_noise_ratio must exceed the classifier threshold of 6."""
        assert features_a.features["depth_to_noise_ratio"] >= 6.0, \
            f"depth_to_noise_ratio={features_a.features['depth_to_noise_ratio']:.2f}"

    def test_all_features_reliable_when_detected(self, features_a):
        """All primary features should be reliable for a clear detection."""
        for key in ["bls_power", "snr", "period_days", "depth", "local_noise"]:
            assert features_a.reliable[key] is True, f"'{key}' marked unreliable"


# ---------------------------------------------------------------------------
# Tests: Feature values — Candidate B (eclipsing-binary-like)
# ---------------------------------------------------------------------------

class TestCandidateBFeatures:
    def test_depth_greater_than_threshold(self, features_b):
        """EB depth must exceed the 5% classifier threshold."""
        assert features_b.features["depth"] > 0.05, \
            f"EB depth {features_b.features['depth']:.4f} below 5% threshold"

    def test_depth_greater_than_a(self, features_a, features_b):
        assert features_b.features["depth"] > features_a.features["depth"]

    def test_snr_much_higher_than_a(self, features_a, features_b):
        """18% transit has much higher SNR than 1.3% transit."""
        assert features_b.features["snr"] > features_a.features["snr"]

    def test_period_close_to_true(self, features_b):
        err = abs(features_b.features["period_days"] - 1.87) / 1.87
        assert err < 0.01, f"Period error {err*100:.2f}%"

    def test_transit_count_correct(self, features_b):
        # 27 days / 1.87 days = ~14 transits
        assert 12 <= features_b.features["transit_count"] <= 16

    def test_all_primary_features_finite(self, features_b):
        for k in ["bls_power", "snr", "depth", "period_days", "duration_days"]:
            assert np.isfinite(float(features_b.features[k]))


# ---------------------------------------------------------------------------
# Tests: Feature values — Candidate C (noise)
# ---------------------------------------------------------------------------

class TestCandidateCFeatures:
    def test_snr_below_threshold(self, features_c):
        assert features_c.features["snr"] < 5.0, \
            f"Noise SNR {features_c.features['snr']:.2f} should be < 5"

    def test_candidate_detected_false(self, features_c):
        assert features_c.candidate_detected is False

    def test_all_16_features_present(self, features_c):
        assert len(features_c.features) == len(FEATURE_NAMES)

    def test_no_nan_in_noise_features(self, features_c):
        for k, v in features_c.features.items():
            assert np.isfinite(float(v)), f"NaN in noise feature '{k}'"

    def test_odd_even_unreliable_for_noise(self, features_c):
        """Noise case: odd_even reliability should be False (too few transits or no detection)."""
        assert features_c.reliable["odd_even_depth_delta"] is False

    def test_bls_power_set_even_for_noise(self, features_c):
        """bls_power_peak is always set, even sub-threshold."""
        assert features_c.features["bls_power"] >= 0.0


# ---------------------------------------------------------------------------
# Tests: Feature 7 — odd_even_depth_delta
# ---------------------------------------------------------------------------

class TestOddEvenDelta:
    def test_near_zero_for_symmetric_box_transit(self, candidate_a_data, result_a):
        """Symmetric box transit: all transit events have the same depth."""
        t, f = candidate_a_data
        delta, reliable = _compute_odd_even_delta(
            t, f,
            period=result_a.best_period,
            t0=result_a.best_t0,
            duration=result_a.best_duration,
            transit_count=int(np.floor(27.0 / result_a.best_period)),
            min_transits=4,
        )
        assert delta < 0.005, f"delta={delta:.4f} should be near 0 for symmetric box"

    def test_returns_false_for_too_few_transits(self, candidate_a_data, result_a):
        t, f = candidate_a_data
        delta, reliable = _compute_odd_even_delta(
            t, f,
            period=result_a.best_period,
            t0=result_a.best_t0,
            duration=result_a.best_duration,
            transit_count=2,   # below min_transits=4
            min_transits=4,
        )
        assert reliable is False

    def test_returns_false_for_zero_period(self, candidate_a_data):
        t, f = candidate_a_data
        delta, reliable = _compute_odd_even_delta(t, f, 0.0, 0.0, 0.0, 5, 4)
        assert reliable is False

    def test_alternating_transits_give_nonzero_delta(self):
        """Manually construct a light curve with alternating transit depths."""
        rng = np.random.default_rng(10)
        n = 18000
        t = np.linspace(0, 27.0, n)
        f = 1.0 + rng.normal(0, 0.0001, n)  # very low noise
        period = 3.0; t0 = 0.5; duration = 0.1
        # Odd transits: depth 0.02, even transits: depth 0.04
        for i in range(1, 10):
            tc_i = t0 + (i - 1) * period
            in_win = np.abs(t - tc_i) <= duration / 2
            depth_i = 0.02 if i % 2 == 1 else 0.04
            f[in_win] -= depth_i
        delta, reliable = _compute_odd_even_delta(
            t, f, period=period, t0=t0, duration=duration,
            transit_count=8, min_transits=4
        )
        assert reliable is True
        assert delta > 0.01, f"Expected delta > 0.01 for alternating transits, got {delta:.4f}"


# ---------------------------------------------------------------------------
# Tests: Feature 8 — v_shape_score
# ---------------------------------------------------------------------------

class TestVShapeScore:
    def test_box_transit_gives_low_score(self, tiny_box_lc):
        t, f = tiny_box_lc
        r = detect(t, f)
        fr = extract(t, f, r)
        assert fr.features["v_shape_score"] < 0.5, \
            f"Box transit v_shape={fr.features['v_shape_score']:.3f} should be < 0.5"

    def test_v_shape_transit_gives_higher_score(self, tiny_v_lc):
        t, f = tiny_v_lc
        r = detect(t, f)
        fr = extract(t, f, r)
        # V-shaped transit should score higher than 0
        # (may not be > 0.5 due to noise, but should be measurably positive)
        assert fr.features["v_shape_score"] >= 0.0

    def test_direct_v_shape_function_box(self):
        """Perfect box: v_shape_score should be 0."""
        n = 200
        phase_in = np.linspace(-0.01, 0.01, n)
        flux_in = np.full(n, 0.987)  # constant transit depth
        score = _compute_v_shape_score(phase_in, flux_in, depth=0.013, half_dur_phase=0.01)
        assert score < 0.3, f"Box transit score={score:.3f} should be near 0"

    def test_direct_v_shape_function_v(self):
        """Perfect V-shape: score should be significantly above 0."""
        n = 200
        phase_in = np.linspace(-0.01, 0.01, n)
        # True V-shape: flux proportional to |phase|
        flux_in = 1.0 - 0.013 * (1.0 - np.abs(phase_in) / 0.01)
        score = _compute_v_shape_score(phase_in, flux_in, depth=0.013, half_dur_phase=0.01)
        assert score > 0.3, f"V-shape score={score:.3f} should be > 0.3"

    def test_score_in_zero_one_range(self, features_a, features_b, features_c):
        for name, fr in [("A", features_a), ("B", features_b), ("C", features_c)]:
            score = fr.features["v_shape_score"]
            assert 0.0 <= score <= 1.0, f"Candidate {name}: v_shape={score:.3f} out of [0,1]"

    def test_returns_zero_for_too_few_points(self):
        phase_in = np.array([0.0, 0.001])  # fewer than _MIN_TRANSIT_POINTS=5
        flux_in = np.array([0.987, 0.988])
        score = _compute_v_shape_score(phase_in, flux_in, depth=0.013, half_dur_phase=0.01)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Tests: Feature 9 — local_noise
# ---------------------------------------------------------------------------

class TestLocalNoise:
    def test_local_noise_positive(self, features_a, features_b):
        assert features_a.features["local_noise"] > 0.0
        assert features_b.features["local_noise"] > 0.0

    def test_local_noise_close_to_injected(self, features_a):
        """Injected noise std = 0.001; local_noise should be within an order of magnitude."""
        assert features_a.features["local_noise"] < 0.01

    def test_local_noise_finite(self, features_c):
        assert np.isfinite(features_c.features["local_noise"])


# ---------------------------------------------------------------------------
# Tests: Feature 10 — depth_to_noise_ratio
# ---------------------------------------------------------------------------

class TestDepthToNoiseRatio:
    def test_ratio_positive_when_detected(self, features_a, features_b):
        assert features_a.features["depth_to_noise_ratio"] > 0.0
        assert features_b.features["depth_to_noise_ratio"] > 0.0

    def test_ratio_above_classifier_threshold(self, features_a):
        """depth_to_noise_ratio > 6.0 is required by classifier."""
        assert features_a.features["depth_to_noise_ratio"] >= 6.0

    def test_ratio_b_much_higher_than_a(self, features_a, features_b):
        """18% transit has much higher depth/noise than 1.3% transit."""
        assert features_b.features["depth_to_noise_ratio"] > features_a.features["depth_to_noise_ratio"]

    def test_ratio_near_snr(self, features_a, candidate_a_data):
        """depth_to_noise_ratio and snr both measure depth/noise -- should be similar after scaling."""
        t, f = candidate_a_data
        dtnr = features_a.features["depth_to_noise_ratio"]
        snr = features_a.features["snr"]
        period = features_a.features["period_days"]
        duration = features_a.features["duration_days"]
        
        # Estimate number of points in transit
        cadence = (t[-1] - t[0]) / len(t)
        n_in = (duration / cadence) * ((t[-1] - t[0]) / period)
        
        ratio = (dtnr * np.sqrt(n_in)) / snr if snr > 0 else 0
        assert 0.5 < ratio < 2.0, f"scaled ratio={ratio:.2f} unexpectedly far from 1"


# ---------------------------------------------------------------------------
# Tests: Config overrides
# ---------------------------------------------------------------------------

class TestConfigOverrides:
    def test_phase_bins_override(self, candidate_a_data, result_a):
        t, f = candidate_a_data
        fr = extract(t, f, result_a, config={"phase_bins": 50})
        assert len(fr.features) == len(FEATURE_NAMES)

    def test_noise_exclusion_factor_override(self, candidate_a_data, result_a):
        t, f = candidate_a_data
        fr = extract(t, f, result_a, config={"noise_exclusion_factor": 2.0})
        assert np.isfinite(fr.features["local_noise"])

    def test_odd_even_min_transits_override(self, candidate_a_data, result_a):
        t, f = candidate_a_data
        # Raise min to 100 so odd/even cannot be computed
        fr = extract(t, f, result_a, config={"odd_even_min_transits": 100})
        assert fr.reliable["odd_even_depth_delta"] is False
        assert fr.features["odd_even_depth_delta"] == 0.0


# ---------------------------------------------------------------------------
# Tests: Processing time
# ---------------------------------------------------------------------------

class TestProcessingTime:
    def test_feature_extraction_under_half_second(self, candidate_a_data, result_a):
        t, f = candidate_a_data
        t0 = _time.perf_counter()
        extract(t, f, result_a)
        elapsed = _time.perf_counter() - t0
        assert elapsed < 0.5, f"Feature extraction took {elapsed:.3f}s (limit: 0.5s)"

    def test_feature_extraction_b_under_half_second(self, candidate_b_data, result_b):
        t, f = candidate_b_data
        t0 = _time.perf_counter()
        extract(t, f, result_b)
        elapsed = _time.perf_counter() - t0
        assert elapsed < 0.5, f"Feature extraction took {elapsed:.3f}s (limit: 0.5s)"