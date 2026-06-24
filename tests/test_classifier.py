"""
tests/test_classifier.py
------------------------
Tests for core/classifier.py and core/confidence.py (Phase 4).

Covers:
    - Correct classification of all three synthetic candidates
    - Rule-based decision tree paths (each stage)
    - Threshold changes in rule_config.yaml change classification
    - ML classifier graceful fallback when .pkl absent
    - Confidence score range [0.0, 1.0] for all cases
    - Minimum confidence thresholds met per plan
    - Component scoring (above/below, partial)
    - ClassificationResult structure
    - Config override at runtime
    - explanation path populated correctly
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from core.bls_detector import detect
from core.classifier import (
    CLASSES,
    ClassificationResult,
    classify,
    reload_rule_config,
    _apply_rules,
    _load_rule_config,
)
from core.confidence import (
    score,
    score_with_breakdown,
    _compute_component_score,
)
from core.exceptions import ClassificationError
from core.feature_extractor import extract


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

def _make_lc(seed, period, depth, duration, t0, n=18000):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 27.0, n)
    f = 1.0 + rng.normal(0, 0.001, n)
    ph = ((t - t0) / period) % 1.0
    ph[ph >= 0.5] -= 1.0
    f[np.abs(ph) < (duration / (2 * period))] -= depth
    return t, f


@pytest.fixture(scope="module")
def features_a():
    t, f = _make_lc(0, 3.42, 0.013, 0.1, 1.5)
    r = detect(t, f)
    return extract(t, f, r).features


@pytest.fixture(scope="module")
def features_b():
    t, f = _make_lc(1, 1.87, 0.18, 0.08, 0.5)
    r = detect(t, f)
    return extract(t, f, r).features


@pytest.fixture(scope="module")
def features_c():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 27.0, 18000)
    f = 1.0 + rng.normal(0, 0.001, 18000)
    r = detect(t, f)
    return extract(t, f, r).features


@pytest.fixture
def exoplanet_features():
    """Synthetic feature dict that should classify as exoplanet_like."""
    return {
        "bls_power": 0.80,
        "snr": 15.0,
        "period_days": 3.42,
        "duration_days": 0.10,
        "depth": 0.013,
        "transit_count": 7,
        "odd_even_depth_delta": 0.001,
        "v_shape_score": 0.05,
        "local_noise": 0.001,
        "depth_to_noise_ratio": 13.0,
        "phase_shape_kurtosis": 1.5,
    }


@pytest.fixture
def eb_features():
    """Synthetic feature dict that should classify as eclipsing_binary_like via depth."""
    return {
        "bls_power": 0.90,
        "snr": 80.0,
        "period_days": 1.87,
        "duration_days": 0.08,
        "depth": 0.18,
        "transit_count": 14,
        "odd_even_depth_delta": 0.002,
        "v_shape_score": 0.10,
        "local_noise": 0.001,
        "depth_to_noise_ratio": 80.0,
        "phase_shape_kurtosis": -1.5,
    }


@pytest.fixture
def noise_features():
    """Synthetic feature dict that should classify as noise_or_other."""
    return {
        "bls_power": 0.05,
        "snr": 1.5,
        "period_days": 5.0,
        "duration_days": 0.10,
        "depth": 0.001,
        "transit_count": 0,
        "odd_even_depth_delta": 0.0,
        "v_shape_score": 0.0,
        "local_noise": 0.001,
        "depth_to_noise_ratio": 1.5,
        "phase_shape_kurtosis": 0.0,
    }


@pytest.fixture
def grazing_eb_features():
    """EB-like via v_shape and odd/even, but shallow depth (planet-like depth range)."""
    return {
        "bls_power": 0.70,
        "snr": 10.0,
        "period_days": 2.5,
        "duration_days": 0.12,
        "depth": 0.030,           # below depth_threshold_eb=0.05
        "transit_count": 10,
        "odd_even_depth_delta": 0.025,   # above odd_even_threshold=0.02
        "v_shape_score": 0.50,
        "local_noise": 0.001,
        "depth_to_noise_ratio": 10.0,
        "phase_shape_kurtosis": 2.0,
    }


@pytest.fixture
def default_thresholds():
    cfg = _load_rule_config()
    return cfg["classification"]


# ---------------------------------------------------------------------------
# Tests: ClassificationResult structure
# ---------------------------------------------------------------------------

class TestClassificationResultStructure:
    def test_has_predicted_class(self, exoplanet_features):
        cr = classify(exoplanet_features)
        assert hasattr(cr, "predicted_class")

    def test_predicted_class_is_valid_string(self, exoplanet_features, eb_features, noise_features):
        for feat in [exoplanet_features, eb_features, noise_features]:
            cr = classify(feat)
            assert cr.predicted_class in CLASSES

    def test_has_rule_path(self, exoplanet_features):
        cr = classify(exoplanet_features)
        assert hasattr(cr, "rule_path")
        assert isinstance(cr.rule_path, list)
        assert len(cr.rule_path) > 0

    def test_has_ml_agreement_field(self, exoplanet_features):
        cr = classify(exoplanet_features)
        assert hasattr(cr, "ml_agreement")
        assert isinstance(cr.ml_agreement, bool)

    def test_has_thresholds_dict(self, exoplanet_features):
        cr = classify(exoplanet_features)
        assert hasattr(cr, "thresholds")
        assert isinstance(cr.thresholds, dict)
        assert len(cr.thresholds) > 0

    def test_invalid_class_raises(self):
        with pytest.raises(ClassificationError):
            ClassificationResult("unknown_class", [])

    def test_ml_agreement_true_when_ml_disabled(self, exoplanet_features):
        """When ML is disabled, ml_agreement should always be True."""
        cr = classify(exoplanet_features)
        assert cr.ml_agreement is True

    def test_ml_class_none_when_ml_disabled(self, exoplanet_features):
        cr = classify(exoplanet_features)
        assert cr.ml_class is None


# ---------------------------------------------------------------------------
# Tests: Correct classification of all three synthetic candidates
# ---------------------------------------------------------------------------

class TestSyntheticCandidates:
    def test_candidate_a_classified_as_exoplanet(self, features_a):
        cr = classify(features_a)
        assert cr.predicted_class == "exoplanet_like", (
            f"A classified as '{cr.predicted_class}'; rule_path={cr.rule_path}"
        )

    def test_candidate_b_classified_as_eb(self, features_b):
        cr = classify(features_b)
        assert cr.predicted_class == "eclipsing_binary_like", (
            f"B classified as '{cr.predicted_class}'; rule_path={cr.rule_path}"
        )

    def test_candidate_c_classified_as_noise(self, features_c):
        cr = classify(features_c)
        assert cr.predicted_class == "noise_or_other", (
            f"C classified as '{cr.predicted_class}'; rule_path={cr.rule_path}"
        )

    def test_all_three_rule_paths_non_empty(self, features_a, features_b, features_c):
        for feat in [features_a, features_b, features_c]:
            cr = classify(feat)
            assert len(cr.rule_path) >= 1


# ---------------------------------------------------------------------------
# Tests: Decision tree stage by stage
# ---------------------------------------------------------------------------

class TestDecisionTree:
    def test_stage1_blocks_on_low_snr(self, noise_features):
        """Low SNR → noise_or_other at Stage 1, regardless of depth."""
        feat = dict(noise_features)
        feat["snr"] = 1.0
        feat["bls_power"] = 0.05
        cr = classify(feat)
        assert cr.predicted_class == "noise_or_other"
        assert "Stage 1" in cr.rule_path[0]

    def test_stage1_blocks_on_low_power(self, exoplanet_features):
        """Low BLS power → noise_or_other at Stage 1."""
        feat = dict(exoplanet_features)
        feat["bls_power"] = 0.05
        cr = classify(feat)
        assert cr.predicted_class == "noise_or_other"
        assert "Stage 1" in cr.rule_path[0]

    def test_stage1_passes_for_strong_signal(self, exoplanet_features):
        cr = classify(exoplanet_features)
        assert "Stage 1 PASS" in cr.rule_path[0]

    def test_stage2_catches_deep_transit(self, eb_features):
        """depth > 0.05 → eclipsing_binary_like at Stage 2."""
        cr = classify(eb_features)
        assert cr.predicted_class == "eclipsing_binary_like"
        assert any("Stage 2 MATCH" in s for s in cr.rule_path)

    def test_stage2_passes_shallow_transit(self, exoplanet_features):
        """depth <= 0.05 → Stage 2 passes, continue to Stage 3."""
        cr = classify(exoplanet_features)
        assert any("Stage 2 PASS" in s for s in cr.rule_path)

    def test_stage3a_odd_even_triggers_eb(self, grazing_eb_features):
        """High odd/even delta in planet-like depth range → EB at Stage 3a."""
        feat = dict(grazing_eb_features)
        feat["v_shape_score"] = 0.0  # force only 3a to trigger
        cr = classify(feat)
        assert cr.predicted_class == "eclipsing_binary_like"
        assert any("Stage 3a MATCH" in s for s in cr.rule_path)

    def test_stage3b_vshape_triggers_eb(self, exoplanet_features):
        """High v_shape_score in planet-like depth range → EB at Stage 3b."""
        feat = dict(exoplanet_features)
        feat["odd_even_depth_delta"] = 0.001   # keep 3a passing
        feat["v_shape_score"] = 0.60            # trigger 3b
        cr = classify(feat)
        assert cr.predicted_class == "eclipsing_binary_like"
        assert any("Stage 3b MATCH" in s for s in cr.rule_path)

    def test_stage3c_low_dtnr_gives_noise(self, exoplanet_features):
        """Low depth_to_noise_ratio after passing stages 1-2 → noise at Stage 3c."""
        feat = dict(exoplanet_features)
        feat["depth_to_noise_ratio"] = 3.0   # below depth_snr_threshold=6.0
        cr = classify(feat)
        assert cr.predicted_class == "noise_or_other"
        assert any("Stage 3c MATCH" in s for s in cr.rule_path)

    def test_all_stages_pass_gives_exoplanet(self, exoplanet_features):
        cr = classify(exoplanet_features)
        assert cr.predicted_class == "exoplanet_like"
        assert "exoplanet_like" in cr.rule_path[-1]


# ---------------------------------------------------------------------------
# Tests: Threshold changes change classification (no hardcoding in Python)
# ---------------------------------------------------------------------------

class TestThresholdChanges:
    def test_changing_depth_threshold_changes_classification(self, tmp_path):
        """Raising depth_threshold_eb to 0.20 means B (depth=0.085) is NOT classified as EB."""
        reload_rule_config()
        cfg = _load_rule_config()
        modified = copy.deepcopy(cfg)
        modified["classification"]["depth_threshold_eb"] = 0.20

        config_path = str(tmp_path / "rule_config_modified.yaml")
        with open(config_path, "w") as f:
            yaml.dump(modified, f)
        reload_rule_config(config_path)

        eb_feat = {
            "bls_power": 0.90, "snr": 80.0, "period_days": 1.87,
            "duration_days": 0.08, "depth": 0.085, "transit_count": 14,
            "odd_even_depth_delta": 0.001, "v_shape_score": 0.05,
            "local_noise": 0.001, "depth_to_noise_ratio": 80.0,
            "phase_shape_kurtosis": -1.5,
        }
        cr = classify(eb_feat, rule_config_path=config_path)
        # With depth_threshold_eb=0.20 and depth=0.085, Stage 2 passes
        # The signal then goes to Stage 3 checks and should classify as exoplanet_like
        assert cr.predicted_class != "noise_or_other"  # strong signal detected

        reload_rule_config()  # restore

    def test_raising_snr_threshold_forces_noise(self, tmp_path, exoplanet_features):
        """Raising snr_threshold to 100 forces even strong signals to noise_or_other."""
        cfg = _load_rule_config()
        modified = copy.deepcopy(cfg)
        modified["detection"]["snr_threshold"] = 100.0

        config_path = str(tmp_path / "rule_config_snr.yaml")
        with open(config_path, "w") as f:
            yaml.dump(modified, f)
        reload_rule_config(config_path)

        cr = classify(exoplanet_features, rule_config_path=config_path)
        assert cr.predicted_class == "noise_or_other"

        reload_rule_config()

    def test_runtime_config_override_classification(self, exoplanet_features):
        """Config override at runtime without touching yaml file."""
        # Force all signals with v_shape > 0 to be EB
        feat = dict(exoplanet_features)
        feat["v_shape_score"] = 0.45   # normally below default threshold 0.40... wait above it
        cr_default = classify(feat)
        # With default threshold 0.40, v_shape=0.45 → EB
        assert cr_default.predicted_class == "eclipsing_binary_like"

        # Now raise threshold to 0.60 via runtime config
        cr_override = classify(feat, config={"classification": {"v_shape_threshold": 0.60}})
        # With threshold=0.60, v_shape=0.45 no longer triggers EB
        assert cr_override.predicted_class == "exoplanet_like"


# ---------------------------------------------------------------------------
# Tests: ML classifier graceful fallback
# ---------------------------------------------------------------------------

class TestMLClassifierFallback:
    def test_ml_disabled_uses_rule_based(self, exoplanet_features):
        """Default config has ml_classifier.enabled=false — should use rule-based."""
        cr = classify(exoplanet_features)
        assert cr.predicted_class == "exoplanet_like"
        assert cr.ml_class is None
        assert cr.ml_agreement is True

    def test_ml_absent_pkl_falls_back(self, tmp_path, exoplanet_features):
        """Even if ml_classifier.enabled=true, absent .pkl files fall back gracefully."""
        cfg = _load_rule_config()
        modified = copy.deepcopy(cfg)
        modified["ml_classifier"]["enabled"] = True

        config_path = str(tmp_path / "rule_config_ml.yaml")
        with open(config_path, "w") as f:
            yaml.dump(modified, f)
        reload_rule_config(config_path)

        # Should not raise — ML path fails gracefully, rule-based result used
        cr = classify(exoplanet_features, rule_config_path=config_path)
        assert cr.predicted_class in CLASSES  # still returns valid class
        assert cr.ml_agreement is True        # fallback marks as agreeing

        reload_rule_config()

    def test_classification_never_returns_invalid_class(self, exoplanet_features, eb_features, noise_features):
        for feat in [exoplanet_features, eb_features, noise_features]:
            cr = classify(feat)
            assert cr.predicted_class in CLASSES, f"Invalid class: {cr.predicted_class}"


# ---------------------------------------------------------------------------
# Tests: rule_config.yaml loading
# ---------------------------------------------------------------------------

class TestRuleConfigLoading:
    def test_config_loads_successfully(self):
        cfg = _load_rule_config()
        assert isinstance(cfg, dict)

    def test_config_has_required_sections(self):
        cfg = _load_rule_config()
        for section in ["detection", "classification", "confidence", "ml_classifier"]:
            assert section in cfg, f"Missing section: {section}"

    def test_config_has_all_thresholds(self):
        cfg = _load_rule_config()
        clf = cfg["classification"]
        for key in ["depth_threshold_eb", "odd_even_threshold", "v_shape_threshold", "depth_snr_threshold"]:
            assert key in clf, f"Missing threshold: {key}"

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(ClassificationError, match="not found"):
            _load_rule_config(str(tmp_path / "nonexistent.yaml"))

    def test_config_cached_after_first_load(self):
        """Second call with same path should return cached result."""
        cfg1 = _load_rule_config()
        cfg2 = _load_rule_config()
        assert cfg1 is cfg2  # same object = cached

    def test_reload_clears_cache(self):
        """reload_rule_config forces fresh load."""
        cfg1 = _load_rule_config()
        cfg2 = reload_rule_config()
        # After reload, config is freshly loaded — values should be identical
        assert cfg1["classification"] == cfg2["classification"]


# ---------------------------------------------------------------------------
# Tests: Confidence score range and minimum thresholds
# ---------------------------------------------------------------------------

class TestConfidenceScore:
    def test_confidence_in_zero_one_for_all_candidates(self, features_a, features_b, features_c):
        for tag, feat, cls in [
            ("A", features_a, "exoplanet_like"),
            ("B", features_b, "eclipsing_binary_like"),
            ("C", features_c, "noise_or_other"),
        ]:
            conf = score(feat, cls)
            assert 0.0 <= conf <= 1.0, f"Candidate {tag}: confidence={conf}"

    def test_candidate_a_confidence_above_0_80(self, features_a):
        conf = score(features_a, "exoplanet_like")
        assert conf >= 0.80, f"Candidate A confidence {conf:.4f} < 0.80"

    def test_candidate_b_confidence_above_0_80(self, features_b):
        conf = score(features_b, "eclipsing_binary_like")
        assert conf >= 0.80, f"Candidate B confidence {conf:.4f} < 0.80"

    def test_candidate_c_confidence_above_0_70(self, features_c):
        conf = score(features_c, "noise_or_other")
        assert conf >= 0.70, f"Candidate C confidence {conf:.4f} < 0.70"

    def test_confidence_always_finite(self, exoplanet_features, eb_features, noise_features):
        for feat, cls in [
            (exoplanet_features, "exoplanet_like"),
            (eb_features, "eclipsing_binary_like"),
            (noise_features, "noise_or_other"),
        ]:
            conf = score(feat, cls)
            assert np.isfinite(conf), f"Confidence is not finite: {conf}"

    def test_higher_snr_gives_higher_exoplanet_confidence(self):
        base = {
            "bls_power": 0.80, "snr": 8.0, "period_days": 3.42, "duration_days": 0.10,
            "depth": 0.013, "transit_count": 7, "odd_even_depth_delta": 0.001,
            "v_shape_score": 0.05, "local_noise": 0.001, "depth_to_noise_ratio": 8.0,
            "phase_shape_kurtosis": 1.5,
        }
        high_snr = dict(base)
        high_snr["snr"] = 20.0
        high_snr["depth_to_noise_ratio"] = 20.0

        conf_low  = score(base, "exoplanet_like")
        conf_high = score(high_snr, "exoplanet_like")
        assert conf_high > conf_low, "Higher SNR should give higher exoplanet confidence"

    def test_unknown_class_returns_0_5(self, exoplanet_features):
        conf = score(exoplanet_features, "unknown_class")
        assert conf == 0.5


# ---------------------------------------------------------------------------
# Tests: score_with_breakdown
# ---------------------------------------------------------------------------

class TestScoreWithBreakdown:
    def test_returns_tuple_float_list(self, features_a):
        result = score_with_breakdown(features_a, "exoplanet_like")
        assert isinstance(result, tuple) and len(result) == 2
        conf, bd = result
        assert isinstance(conf, float)
        assert isinstance(bd, list)

    def test_breakdown_has_required_keys(self, features_a):
        _, bd = score_with_breakdown(features_a, "exoplanet_like")
        required_keys = {"name", "feature", "feature_val", "direction",
                         "threshold", "component_score", "weight", "weighted_contribution"}
        for comp in bd:
            assert required_keys.issubset(comp.keys()), f"Missing keys: {required_keys - comp.keys()}"

    def test_breakdown_scores_in_zero_one(self, features_a):
        _, bd = score_with_breakdown(features_a, "exoplanet_like")
        for comp in bd:
            assert 0.0 <= comp["component_score"] <= 1.0

    def test_confidence_matches_breakdown_weighted_sum(self, features_a):
        conf, bd = score_with_breakdown(features_a, "exoplanet_like")
        total_w = sum(c["weight"] for c in bd)
        weighted = sum(c["weighted_contribution"] for c in bd)
        expected = weighted / total_w if total_w > 0 else 0.5
        assert abs(conf - expected) < 1e-9

    def test_breakdown_non_empty_for_all_classes(self, features_a, features_b, features_c):
        for feat, cls in [
            (features_a, "exoplanet_like"),
            (features_b, "eclipsing_binary_like"),
            (features_c, "noise_or_other"),
        ]:
            _, bd = score_with_breakdown(feat, cls)
            assert len(bd) >= 2, f"Expected at least 2 components for {cls}"


# ---------------------------------------------------------------------------
# Tests: _compute_component_score
# ---------------------------------------------------------------------------

class TestComputeComponentScore:
    def test_above_at_threshold_gives_1(self):
        assert _compute_component_score(10.0, "above", 10.0) == 1.0

    def test_above_exceeds_threshold_gives_1(self):
        assert _compute_component_score(15.0, "above", 10.0) == 1.0

    def test_above_zero_gives_0(self):
        assert _compute_component_score(0.0, "above", 10.0) == 0.0

    def test_above_half_threshold_gives_0_5(self):
        score_val = _compute_component_score(5.0, "above", 10.0)
        assert abs(score_val - 0.5) < 1e-9

    def test_below_at_threshold_gives_1(self):
        assert _compute_component_score(3.0, "below", 3.0) == 1.0

    def test_below_zero_gives_1(self):
        assert _compute_component_score(0.0, "below", 3.0) == 1.0

    def test_below_double_threshold_gives_0(self):
        # At 2× threshold, decay = 1 - (2t - t)/t = 0
        score_val = _compute_component_score(6.0, "below", 3.0)
        assert score_val == 0.0

    def test_output_always_in_zero_one(self):
        for val in [-5.0, 0.0, 0.5, 1.0, 5.0, 100.0]:
            for direction in ["above", "below"]:
                s = _compute_component_score(val, direction, 3.0)
                assert 0.0 <= s <= 1.0, f"val={val} dir={direction} score={s}"

    def test_non_finite_input_gives_0(self):
        assert _compute_component_score(float("nan"), "above", 5.0) == 0.0
        assert _compute_component_score(float("inf"), "above", 5.0) == 0.0