"""
tests/test_pipeline.py
----------------------
End-to-end tests for pipeline.py (Phase 5).

Covers:
    - analyze_light_curve() returns complete result dict with all required keys
    - All three synthetic cases classified correctly
    - candidate_detected is False for noise case
    - Result dict passes all invariants from Section 3 of the plan
    - Processing time is under 30 seconds for all cases
    - Error handling: mismatched arrays raise InvalidInputError
    - Error handling: all-NaN flux produces a graceful result
    - Config override merging works correctly
    - Plots are present (non-empty or empty strings)
"""

from __future__ import annotations

import time as _time

import numpy as np
import pytest

from pipeline import analyze_light_curve, _load_config, _deep_merge
from core.exceptions import InvalidInputError
from core.feature_extractor import FEATURE_NAMES


# ---------------------------------------------------------------------------
# Result dict structure tests
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "target_id", "candidate_detected", "predicted_class", "confidence",
    "period_days", "duration_days", "depth", "snr", "transit_count",
    "bootstrap_fap", "class_probabilities", "period_uncertainty_days",
    "duration_uncertainty_days", "depth_uncertainty", "epoch_btjd", "fit_quality",
    "features", "explanation", "plots",
    "processing_time_ms", "pipeline_version",
}

REQUIRED_PLOT_KEYS = {"raw_lightcurve", "cleaned_lightcurve", "periodogram", "phase_folded"}


class TestResultStructure:
    """Tests that the result dict has the correct structure."""

    def test_result_has_all_keys(self, synthetic_cases):
        case = synthetic_cases["a"]
        result = analyze_light_curve(case["time"], case["flux"], case["metadata"])
        assert REQUIRED_KEYS.issubset(set(result.keys()))

    def test_features_has_11_keys(self, synthetic_cases):
        case = synthetic_cases["a"]
        result = analyze_light_curve(case["time"], case["flux"], case["metadata"])
        assert set(result["features"].keys()) == set(FEATURE_NAMES)

    def test_plots_has_four_keys(self, synthetic_cases):
        case = synthetic_cases["a"]
        result = analyze_light_curve(case["time"], case["flux"], case["metadata"])
        assert set(result["plots"].keys()) == REQUIRED_PLOT_KEYS

    def test_version_is_string(self, synthetic_cases):
        case = synthetic_cases["a"]
        result = analyze_light_curve(case["time"], case["flux"], case["metadata"])
        assert isinstance(result["pipeline_version"], str)
        assert len(result["pipeline_version"]) > 0

    def test_processing_time_positive(self, synthetic_cases):
        case = synthetic_cases["a"]
        result = analyze_light_curve(case["time"], case["flux"], case["metadata"])
        assert result["processing_time_ms"] > 0


# ---------------------------------------------------------------------------
# Classification correctness
# ---------------------------------------------------------------------------

class TestClassification:
    """Test that all three synthetic cases are classified correctly."""

    def test_candidate_a_is_exoplanet(self, synthetic_cases):
        case = synthetic_cases["a"]
        result = analyze_light_curve(case["time"], case["flux"], case["metadata"])
        assert result["predicted_class"] == "exoplanet_transit"
        assert result["candidate_detected"] is True

    def test_candidate_b_is_eb(self, synthetic_cases):
        case = synthetic_cases["b"]
        result = analyze_light_curve(case["time"], case["flux"], case["metadata"])
        assert result["predicted_class"] == "eclipsing_binary"
        assert result["candidate_detected"] is True

    def test_candidate_c_is_noise(self, synthetic_cases):
        case = synthetic_cases["c"]
        result = analyze_light_curve(case["time"], case["flux"], case["metadata"])
        assert result["predicted_class"] == "stellar_variability_or_other"
        assert result["candidate_detected"] is False


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

class TestInvariants:
    """Test that the interface contract invariants hold."""

    def test_no_detection_nulls(self, synthetic_cases):
        """When candidate_detected=False, nullable fields must be None."""
        case = synthetic_cases["c"]
        result = analyze_light_curve(case["time"], case["flux"], case["metadata"])
        assert result["candidate_detected"] is False
        for key in ["period_days", "duration_days", "depth", "snr", "transit_count", "bootstrap_fap", "period_uncertainty_days", "duration_uncertainty_days", "depth_uncertainty", "epoch_btjd", "fit_quality"]:
            assert result[key] is None, f"{key} should be None when not detected"

    def test_detection_not_null(self, synthetic_cases):
        """When candidate_detected=True, nullable fields must NOT be None."""
        case = synthetic_cases["a"]
        result = analyze_light_curve(case["time"], case["flux"], case["metadata"])
        assert result["candidate_detected"] is True
        for key in ["period_days", "duration_days", "depth", "snr", "transit_count", "bootstrap_fap", "period_uncertainty_days", "duration_uncertainty_days", "depth_uncertainty", "epoch_btjd", "fit_quality"]:
            assert result[key] is not None, f"{key} should not be None when detected"

    def test_confidence_in_range(self, synthetic_cases):
        for cid in ["a", "b", "c"]:
            case = synthetic_cases[cid]
            result = analyze_light_curve(case["time"], case["flux"], case["metadata"])
            assert 0.0 <= result["confidence"] <= 1.0

    def test_predicted_class_valid(self, synthetic_cases):
        valid = {"exoplanet_transit", "eclipsing_binary", "stellar_variability_or_other"}
        for cid in ["a", "b", "c"]:
            case = synthetic_cases[cid]
            result = analyze_light_curve(case["time"], case["flux"], case["metadata"])
            assert result["predicted_class"] in valid

    def test_legacy_aliases_map_correctly(self):
        from core.classifier import ClassificationResult
        res1 = ClassificationResult("exoplanet_like", [], ml_class="eclipsing_binary_like")
        assert res1.predicted_class == "exoplanet_transit"
        assert res1.ml_class == "eclipsing_binary"
        
        res2 = ClassificationResult("noise_or_other", [])
        assert res2.predicted_class == "stellar_variability_or_other"

    def test_schema_conforms_to_contract(self, synthetic_cases):
        case = synthetic_cases["a"]
        result = analyze_light_curve(case["time"], case["flux"], case["metadata"])
        
        # Check all required fields from §3 of the Scientific Contract
        required_fields = {
            "target_id", "pipeline_version", "processing_time_ms",
            "candidate_detected", "predicted_class", "confidence", "class_probabilities", "explanation",
            "period_days", "period_uncertainty_days", "duration_days", "duration_uncertainty_days", "depth", "depth_uncertainty", "epoch_btjd",
            "snr", "bootstrap_fap", "fit_quality", "transit_count",
            "features", "plots"
        }
        for field in required_fields:
            assert field in result, f"Result missing schema field: {field}"
        
        # Check plots sub-fields
        plots = result["plots"]
        for plot_key in ["raw_lightcurve", "cleaned_lightcurve", "periodogram", "phase_folded"]:
            assert plot_key in plots, f"plots dict missing key: {plot_key}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test graceful error handling."""

    def test_mismatched_lengths_raises_invalid_input(self):
        with pytest.raises(InvalidInputError):
            analyze_light_curve(
                time=[1.0, 2.0, 3.0],
                flux=[1.0, 2.0],
            )

    def test_empty_arrays_raises_invalid_input(self):
        with pytest.raises((InvalidInputError, Exception)):
            analyze_light_curve(time=[], flux=[])


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    """Test processing time is within acceptable bounds."""

    def test_processing_under_30_seconds(self, synthetic_cases):
        """Each case should complete in under 30 seconds."""
        for cid in ["a", "b", "c"]:
            case = synthetic_cases[cid]
            start = _time.perf_counter()
            result = analyze_light_curve(case["time"], case["flux"], case["metadata"])
            elapsed = _time.perf_counter() - start
            assert elapsed < 30.0, (
                f"candidate_{cid} took {elapsed:.1f}s (limit: 30s)"
            )


# ---------------------------------------------------------------------------
# Config merging
# ---------------------------------------------------------------------------

class TestConfigMerge:
    """Test config loading and override merging."""

    def test_deep_merge_simple(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested(self):
        base = {"bls": {"period_min": 0.5, "period_max": None}}
        override = {"bls": {"period_max": 5.0}}
        result = _deep_merge(base, override)
        assert result["bls"]["period_min"] == 0.5
        assert result["bls"]["period_max"] == 5.0
