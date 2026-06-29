import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml.calibration import expected_calibration_error, multiclass_brier
from ml.contracts import ContractError, PHYSICAL_CLASSES, validate_probability_vector
from ml.dataset import verify_inputs
from ml.preprocessing import FrozenPreprocessor
from ml.tabular import require_all_classes
from ml.timeseries import build_dual_views
from ml.uncertainty import ReviewPolicy, route_review

ROOT = Path(__file__).parents[1]
MANIFESTS = ROOT.parent / "data" / "manifests" / "phase1"

def test_fixed_physical_classes_excludes_review():
    assert len(PHYSICAL_CLASSES) == 4
    assert "review_required" not in PHYSICAL_CLASSES

def test_absent_class_hard_fails_without_padding():
    with pytest.raises(ContractError, match="absent"):
        require_all_classes([0, 0, 1, 2])

def test_probability_contract_rejects_fifth_class():
    values = {name: 0.25 for name in PHYSICAL_CLASSES}
    validate_probability_vector(values)
    values["review_required"] = 0.0
    with pytest.raises(ContractError): validate_probability_vector(values)

def test_preprocessor_is_train_fitted_and_schema_strict():
    train = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, 8.0]})
    processor = FrozenPreprocessor(["a", "b"]).fit(train)
    transformed = processor.transform(pd.DataFrame({"a": [np.nan], "b": [5.0]}))
    assert transformed.shape[0] == 1 and np.isfinite(transformed).all()
    with pytest.raises(ContractError): processor.transform(pd.DataFrame({"a": [1], "b": [2], "unknown": [3]}))

def test_review_routing_does_not_change_probability_classes():
    probs = np.array([[0.26, 0.25, 0.25, 0.24], [0.9, 0.04, 0.03, 0.03]])
    reasons = route_review(probs, ReviewPolicy(), ood=[False, True])
    assert reasons[0] and "out_of_distribution" in reasons[1]
    assert probs.shape[1] == 4

def test_calibration_metrics_known_perfect_case():
    y = np.array([0, 1, 2, 3])
    probs = np.eye(4)
    assert expected_calibration_error(y, probs) == pytest.approx(0)
    assert multiclass_brier(y, probs) == pytest.approx(0)

def test_dual_views_are_deterministic_and_masked():
    time = np.linspace(0, 10, 1000); flux = np.ones(1000)
    views = build_dual_views(time, flux, 2.0, 0.0, 0.1, global_length=101, local_length=31)
    assert views["global"].shape == (2, 101)
    assert views["local"].shape == (2, 31)
    assert set(np.unique(views["global"][1])) <= {0, 1}

def test_current_frozen_inputs_block_official_training():
    report = verify_inputs(MANIFESTS)
    assert report["status"] == "BLOCKED"
    assert report["phase1_status"] == "PASS"
    assert report["phase2_status"] == "PARTIAL"
    assert report["rows"] == {"train": 4827, "validation": 1037, "test": 1030}

def test_no_bypass_flags_in_entrypoints():
    combined = (ROOT / "train_model.py").read_text() + (ROOT / "promote_model.py").read_text()
    assert "bypass-eligibility" not in combined
    assert "bypass-sufficiency" not in combined
