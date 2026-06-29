from pathlib import Path
import json
import pandas as pd

from ml.prototype import PrototypeModel

ROOT=Path(__file__).parents[1]

def test_restricted_prototype_artifacts_are_hash_verified():
    model=PrototypeModel(ROOT/"phase3_prototype"/"three_class")
    assert model.record["production_eligible"] is False
    assert model.record["status"]=="DEVELOPMENT_RESTRICTED"

def test_prototype_outputs_three_normalized_probabilities():
    model=PrototypeModel(ROOT/"phase3_prototype"/"three_class")
    frame=pd.read_parquet(ROOT/"phase2_prototype"/"current"/"phase2_features_test.parquet").iloc[[0]]
    result=model.predict(frame)[0]
    assert set(result["probabilities"])=={"exoplanet_transit","eclipsing_binary","blend_contamination"}
    assert abs(sum(result["probabilities"].values())-1)<1e-9
    assert result["model_status"]=="development_restricted"

def test_active_pointer_cannot_claim_production():
    pointer=json.loads((ROOT/"phase3_prototype"/"active_model.json").read_text())
    assert pointer["production_eligible"] is False
    assert pointer["status"]=="development_restricted"
