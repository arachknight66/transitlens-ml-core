"""
tests/test_mandatory_classifier.py
----------------------------------
Unit tests to verify mandatory trained ML classifier behavior and dev_fallback logic.
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from core.classifier import classify, reload_rule_config, _load_rule_config
from core.exceptions import ClassificationError
from core.feature_extractor import FEATURE_NAMES

@pytest.fixture
def sample_features():
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

def test_missing_model_raises_classification_error_when_fallback_disabled(tmp_path, sample_features):
    """When ML is enabled and dev_fallback is false, missing model files MUST raise ClassificationError."""
    cfg = _load_rule_config()
    modified = copy.deepcopy(cfg)
    modified["ml_classifier"]["enabled"] = True
    modified["ml_classifier"]["dev_fallback"] = False
    
    # Write to a temp directory so we look for model files there, which will be missing
    config_path = str(tmp_path / "rule_config_no_models.yaml")
    with open(config_path, "w") as f:
        yaml.dump(modified, f)
        
    reload_rule_config(config_path)
    
    with pytest.raises(ClassificationError) as exc_info:
        classify(sample_features, rule_config_path=config_path)
        
    assert "Please ensure the ML models are trained" in str(exc_info.value)
    
    reload_rule_config()  # Restore default cache

def test_missing_model_falls_back_when_fallback_enabled(tmp_path, sample_features):
    """When ML is enabled but dev_fallback is true, missing model files fall back to rule-based classification."""
    cfg = _load_rule_config()
    modified = copy.deepcopy(cfg)
    modified["ml_classifier"]["enabled"] = True
    modified["ml_classifier"]["dev_fallback"] = True
    
    config_path = str(tmp_path / "rule_config_fallback.yaml")
    with open(config_path, "w") as f:
        yaml.dump(modified, f)
        
    reload_rule_config(config_path)
    
    # Should not raise exception
    res = classify(sample_features, rule_config_path=config_path)
    assert res.predicted_class == "exoplanet_like"  # fallback rule-based classification
    assert res.ml_class is None
    assert res.ml_agreement is True
    
    reload_rule_config()

def test_schema_mismatch_raises_classification_error(tmp_path, sample_features):
    """If feature_order.json has mismatched columns, raise ClassificationError."""
    cfg = _load_rule_config()
    modified = copy.deepcopy(cfg)
    modified["ml_classifier"]["enabled"] = True
    modified["ml_classifier"]["dev_fallback"] = False
    
    config_path = str(tmp_path / "rule_config_mismatch.yaml")
    with open(config_path, "w") as f:
        yaml.dump(modified, f)
        
    # Write an invalid feature_order.json
    invalid_features = ["bls_power", "snr", "period_days", "invalid_feature"]
    with open(tmp_path / "feature_order.json", "w") as f:
        json.dump(invalid_features, f)
        
    reload_rule_config(config_path)
    
    with pytest.raises(ClassificationError) as exc_info:
        classify(sample_features, rule_config_path=config_path)
        
    assert "Feature schema mismatch" in str(exc_info.value)
    
    reload_rule_config()
