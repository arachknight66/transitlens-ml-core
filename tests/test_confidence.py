"""
tests/test_confidence.py
------------------------
Tests for core/confidence.py.

Covers:
    - Confidence score range [0.0, 1.0] for all three classes
    - Component scoring (above/below direction)
    - Partial scoring (smooth transitions, not hard cutoffs)
    - score_with_breakdown returns detailed component info
    - Default confidence (0.5) when config is missing
    - Exoplanet case produces high confidence (>0.70)
    - EB case produces high confidence (>0.70)
    - Noise case produces high confidence (>0.70)
"""

from __future__ import annotations

import numpy as np
import pytest

from core.confidence import score, score_with_breakdown, _compute_component_score


# ---------------------------------------------------------------------------
# Component scoring unit tests
# ---------------------------------------------------------------------------

class TestComponentScore:
    """Tests for the _compute_component_score helper."""

    def test_above_at_threshold_gives_full_score(self):
        assert _compute_component_score(0.40, "above", 0.40) == 1.0

    def test_above_exceeding_threshold_gives_full_score(self):
        assert _compute_component_score(0.60, "above", 0.40) == 1.0

    def test_above_below_threshold_gives_partial_score(self):
        result = _compute_component_score(0.20, "above", 0.40)
        assert 0.0 < result < 1.0
        assert abs(result - 0.5) < 0.01  # 0.20/0.40 = 0.5

    def test_above_at_zero_gives_zero(self):
        assert _compute_component_score(0.0, "above", 0.40) == 0.0

    def test_below_at_threshold_gives_full_score(self):
        assert _compute_component_score(0.05, "below", 0.05) == 1.0

    def test_below_under_threshold_gives_full_score(self):
        assert _compute_component_score(0.01, "below", 0.05) == 1.0

    def test_below_above_threshold_gives_partial_score(self):
        result = _compute_component_score(0.08, "below", 0.05)
        assert 0.0 <= result < 1.0

    def test_non_finite_returns_zero(self):
        assert _compute_component_score(float("nan"), "above", 0.40) == 0.0
        assert _compute_component_score(float("inf"), "above", 0.40) == 0.0

    def test_unknown_direction_returns_half(self):
        assert _compute_component_score(0.5, "sideways", 0.40) == 0.5


# ---------------------------------------------------------------------------
# Score function tests
# ---------------------------------------------------------------------------

class TestScore:
    """Tests for the public score() function."""

    def test_exoplanet_features_give_high_confidence(self):
        features = {
            "bls_power": 0.50, "snr": 20.0, "period_days": 3.42,
            "duration_days": 0.12, "depth": 0.013, "transit_count": 7,
            "odd_even_depth_delta": 0.002, "v_shape_score": 0.05,
            "local_noise": 0.001, "depth_to_noise_ratio": 13.0,
            "phase_shape_kurtosis": -0.5,
        }
        conf = score(features, "exoplanet_like")
        assert 0.0 <= conf <= 1.0
        assert conf > 0.70

    def test_eb_features_give_high_confidence(self):
        features = {
            "bls_power": 0.60, "snr": 25.0, "period_days": 1.87,
            "duration_days": 0.15, "depth": 0.18, "transit_count": 14,
            "odd_even_depth_delta": 0.05, "v_shape_score": 0.65,
            "local_noise": 0.001, "depth_to_noise_ratio": 180.0,
            "phase_shape_kurtosis": 1.5,
        }
        conf = score(features, "eclipsing_binary_like")
        assert 0.0 <= conf <= 1.0
        assert conf > 0.70

    def test_noise_features_give_high_confidence(self):
        features = {
            "bls_power": 0.02, "snr": 1.5, "period_days": 5.0,
            "duration_days": 0.1, "depth": 0.001, "transit_count": 0,
            "odd_even_depth_delta": 0.0, "v_shape_score": 0.0,
            "local_noise": 0.001, "depth_to_noise_ratio": 1.0,
            "phase_shape_kurtosis": 0.0,
        }
        conf = score(features, "noise_or_other")
        assert 0.0 <= conf <= 1.0
        assert conf > 0.70

    def test_confidence_always_in_range(self):
        features = {k: 0.0 for k in [
            "bls_power", "snr", "period_days", "duration_days", "depth",
            "transit_count", "odd_even_depth_delta", "v_shape_score",
            "local_noise", "depth_to_noise_ratio", "phase_shape_kurtosis",
        ]}
        for cls in ["exoplanet_like", "eclipsing_binary_like", "noise_or_other"]:
            conf = score(features, cls)
            assert 0.0 <= conf <= 1.0

    def test_unknown_class_returns_default(self):
        features = {"bls_power": 0.5}
        conf = score(features, "alien_spacecraft")
        assert conf == 0.5


# ---------------------------------------------------------------------------
# Breakdown tests
# ---------------------------------------------------------------------------

class TestScoreWithBreakdown:
    """Tests for score_with_breakdown()."""

    def test_returns_tuple(self):
        features = {
            "bls_power": 0.50, "snr": 20.0, "period_days": 3.42,
            "duration_days": 0.12, "depth": 0.013, "transit_count": 7,
            "odd_even_depth_delta": 0.002, "v_shape_score": 0.05,
            "local_noise": 0.001, "depth_to_noise_ratio": 13.0,
            "phase_shape_kurtosis": -0.5,
        }
        result = score_with_breakdown(features, "exoplanet_like")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_breakdown_has_components(self):
        features = {
            "bls_power": 0.50, "snr": 20.0, "period_days": 3.42,
            "duration_days": 0.12, "depth": 0.013, "transit_count": 7,
            "odd_even_depth_delta": 0.002, "v_shape_score": 0.05,
            "local_noise": 0.001, "depth_to_noise_ratio": 13.0,
            "phase_shape_kurtosis": -0.5,
        }
        conf, breakdown = score_with_breakdown(features, "exoplanet_like")
        assert len(breakdown) > 0
        for comp in breakdown:
            assert "name" in comp
            assert "component_score" in comp
            assert "weight" in comp
            assert 0.0 <= comp["component_score"] <= 1.0

    def test_breakdown_confidence_matches_score(self):
        features = {
            "bls_power": 0.50, "snr": 20.0, "period_days": 3.42,
            "duration_days": 0.12, "depth": 0.013, "transit_count": 7,
            "odd_even_depth_delta": 0.002, "v_shape_score": 0.05,
            "local_noise": 0.001, "depth_to_noise_ratio": 13.0,
            "phase_shape_kurtosis": -0.5,
        }
        conf_breakdown, _ = score_with_breakdown(features, "exoplanet_like")
        conf_simple = score(features, "exoplanet_like")
        assert abs(conf_breakdown - conf_simple) < 0.001
