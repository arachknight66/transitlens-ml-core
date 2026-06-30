import sys
from pathlib import Path

import numpy as np
import pytest

DATA_PIPELINE = Path(__file__).resolve().parents[2] / "transitlens-data-pipeline"
if str(DATA_PIPELINE) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE))

from interface import load_light_curve
from pipeline import analyze_light_curve


@pytest.mark.parametrize(
    "candidate,expected_class,expected_period",
    [
        ("candidate_a", "exoplanet_transit", 3.4217),
        ("candidate_b", "eclipsing_binary", 1.87),
        ("candidate_c", "stellar_variability_or_other", None),
    ],
)
def test_demo_science_and_baseline(candidate, expected_class, expected_period):
    curve = load_light_curve("synthetic", candidate)
    assert curve["time"][-1] - curve["time"][0] >= 25.0
    metadata = dict(curve["metadata"])
    metadata["target_id"] = candidate
    result = analyze_light_curve(
        curve["time"], curve["flux"], metadata,
        {"significance": {"bootstrap_iterations": 3}, "plotting": {"downsample_points": 500}},
    )
    assert result["predicted_class"] == expected_class
    assert set(result["class_probabilities"]) == {
        "exoplanet_transit", "eclipsing_binary", "blend_contamination", "stellar_variability_or_other"
    }
    assert sum(result["class_probabilities"].values()) == pytest.approx(1.0, abs=2e-6)
    if expected_period is None:
        assert result["candidate_detected"] is False
        for field in ("period_days", "depth", "duration_days", "snr"):
            assert result[field] is None
    else:
        assert result["candidate_detected"] is True
        assert result["period_days"] == pytest.approx(expected_period, rel=0.01)
        assert result["period_uncertainty_days"] is not None
        assert result["depth_uncertainty"] is not None
        assert result["duration_uncertainty_days"] is not None


@pytest.mark.parametrize("candidate", ["candidate_a", "candidate_b", "candidate_c"])
def test_denoising_acceptance_gates(candidate):
    curve = load_light_curve("synthetic", candidate)
    metadata = dict(curve["metadata"])
    metadata["target_id"] = candidate
    result = analyze_light_curve(
        curve["time"], curve["flux"], metadata,
        {"significance": {"bootstrap_iterations": 2}, "plotting": {"downsample_points": 300}},
    )
    gate = result["denoising"]
    if result["candidate_detected"]:
        assert gate["accepted"] is True
        assert gate["noise_reduction_fraction"] >= 0.10
        assert gate["depth_attenuation_fraction"] <= 0.05
        assert gate["period_change_fraction"] < 0.01
        assert gate["duration_change_fraction"] <= 0.10
    else:
        assert result["predicted_class"] == "stellar_variability_or_other"
        assert gate["accepted"] is False
        assert any("new significant periodic detection" in reason for reason in gate["rejection_reasons"])


def test_denoising_is_deterministic():
    curve = load_light_curve("synthetic", "candidate_a")
    metadata = dict(curve["metadata"])
    metadata["target_id"] = "candidate_a"
    cfg = {"significance": {"bootstrap_iterations": 1}, "plotting": {"downsample_points": 200}}
    first = analyze_light_curve(curve["time"], curve["flux"], metadata, cfg)
    second = analyze_light_curve(curve["time"], curve["flux"], metadata, cfg)
    assert first["series"]["denoised"] == second["series"]["denoised"]
