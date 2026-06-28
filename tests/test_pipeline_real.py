import os
import sys
import json
import pickle
import pytest
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits

# Add project paths to system path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "transitlens-data-pipeline"))
sys.path.insert(0, str(REPO_ROOT / "transitlens-ml-core"))

from datasets.build_tess_training_manifest import normalize_tic_id
from real_tess.tesscut_downloader import validate_fits
from datasets.process_tesscut_lightcurves import perform_aperture_photometry
from core.feature_extractor import FEATURE_NAMES
from core.classifier import TransitLensClassifier, classify, ClassificationError

def test_comment_csv_parsing(tmp_path):
    """Test that CSV files with NASA comment headers are correctly parsed."""
    csv_content = """# Column 1: tid
# Column 2: tfopwg_disp
tid,tfopwg_disp,ra,dec,toi,pl_orbper,pl_trandurh,pl_trandep
50365310,CP,112.357708,-12.69596,1000.01,2.171348,2.017,656.8
88886371,PC,122.580465,-5.513852,1001.01,1.931646,3.166,1286.0
"""
    csv_file = tmp_path / "test_toi.csv"
    csv_file.write_text(csv_content)
    
    df = pd.read_csv(csv_file, comment="#")
    assert len(df) == 2
    assert list(df["tid"]) == [50365310, 88886371]
    assert list(df["tfopwg_disp"]) == ["CP", "PC"]

def test_tic_label_mapping():
    """Test normalization of TIC ID formats."""
    val, target_id = normalize_tic_id("TIC-50365310")
    assert val == 50365310
    assert target_id == "TIC-50365310"
    
    val, target_id = normalize_tic_id(" 50365310 ")
    assert val == 50365310
    assert target_id == "TIC-50365310"
    
    val, target_id = normalize_tic_id(np.nan)
    assert val is None

def test_no_split_overlap(tmp_path):
    """Verify that train, validation, and test splits have 0 overlapping TICs."""
    from datasets.build_tess_training_manifest import main as build_manifest
    
    toi_content = """# Comment
tid,tfopwg_disp,ra,dec,toi,pl_orbper,pl_trandurh,pl_trandep
50365310,CP,112.35,-12.69,1000.01,2.17,2.01,656.8
88886371,FA,122.58,-5.51,1001.01,1.93,3.16,1286.0
12470966,FP,104.72,-10.58,1002.01,1.86,1.40,1500.0
10699750,CP,110.55,-25.20,1003.01,2.74,3.16,383.4
"""
    tce_content = """tceid,ticid,tce_plnt_num,sectors,lastUpdate
0050365310-01,50365310,1,s0078,17-07-2024
"""
    toi_file = tmp_path / "toi.csv"
    toi_file.write_text(toi_content)
    
    tce_file = tmp_path / "tce.csv"
    tce_file.write_text(tce_content)
    
    policy_content = """
version: "1.0.0"
mappings:
  CP:
    label: "exoplanet_transit"
    strength: "strong"
    action: "include"
  FA:
    label: "stellar_variability_or_other"
    strength: "strong"
    action: "include"
  FP:
    label: "stellar_variability_or_other"
    strength: "medium"
    action: "include"
  PC:
    label: "exclude"
    action: "exclude"
"""
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(policy_content)
    
    manifest_parquet = tmp_path / "manifest.parquet"
    
    sys.argv = [
        "build_tess_training_manifest.py",
        "--archive", str(toi_file),
        "--tce", str(tce_file),
        "--output", str(manifest_parquet),
        "--label-policy", str(policy_file)
    ]
    
    build_manifest()
    
    df = pd.read_parquet(manifest_parquet)
    assert len(df) > 0
    
    train_tics = set(df[df["split"] == "train"]["tic_id"])
    val_tics = set(df[df["split"] == "val"]["tic_id"])
    test_tics = set(df[df["split"] == "test"]["tic_id"])
    
    assert train_tics.isdisjoint(val_tics)
    assert train_tics.isdisjoint(test_tics)
    assert val_tics.isdisjoint(test_tics)

def test_corrupt_fits_rejected(tmp_path):
    """Test that validate_fits correctly rejects bad FITS files."""
    bad_file = tmp_path / "corrupt.fits"
    bad_file.write_text("not a fits file")
    
    is_valid, msg = validate_fits(bad_file)
    assert not is_valid
    assert len(msg) > 0

def test_aperture_fallback(tmp_path):
    """Test that performing aperture photometry falls back to 3x3 window when threshold is empty."""
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["OBJECT"] = "TIC 50365310"
    primary_hdu.header["SECTOR"] = 78
    
    time_col = fits.Column(name="TIME", format="D", array=np.linspace(0.0, 10.0, 150))
    flux_array = np.ones((150, 15, 15)) * 1.0
    flux_array[:, 0, 0] = 1000.0  # Make standard deviation high
    flux_array[:, 6:9, 6:9] = 5.0
    
    flux_col = fits.Column(name="FLUX", format="225E", dim="(15,15)", array=flux_array)
    flux_err_col = fits.Column(name="FLUX_ERR", format="225E", dim="(15,15)", array=np.ones((150, 15, 15)) * 1.0)
    quality_col = fits.Column(name="QUALITY", format="J", array=np.zeros(150, dtype=np.int32))
    
    cols = fits.ColDefs([time_col, flux_col, flux_err_col, quality_col])
    table_hdu = fits.BinTableHDU.from_columns(cols)
    table_hdu.header["EXTNAME"] = "LIGHTCURVE"
    
    hdul = fits.HDUList([primary_hdu, table_hdu])
    fits_path = tmp_path / "mock_tpf.fits"
    hdul.writeto(fits_path, overwrite=True)
    
    res = perform_aperture_photometry(fits_path, cutout_size=15)
    assert res["metadata"]["is_fallback"] == True
    assert len(res["time"]) == 150
    assert len(res["flux"]) == 150
    assert np.all(np.isfinite(res["flux"]))

def test_feature_order_matches_inference():
    """Verify that FEATURE_NAMES has the correct dimensions and order matches classifier expectations."""
    assert len(FEATURE_NAMES) == 18
    assert FEATURE_NAMES[0] == "bls_power"
    assert FEATURE_NAMES[-1] == "secondary_eclipse_significance"

def test_inference_final_feature_order_validation(tmp_path):
    """Test that classifier raises ClassificationError when feature order mismatches or is missing."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    # Save a fake final_classifier.pkl
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(np.zeros((10, len(FEATURE_NAMES))))
    clf = RandomForestClassifier()
    clf.fit(np.zeros((10, len(FEATURE_NAMES))), ["exoplanet_transit"] * 5 + ["stellar_variability_or_other"] * 5)
    
    wrapper = TransitLensClassifier(model=clf, scaler=scaler, classes=["exoplanet_transit", "stellar_variability_or_other"])
    with open(models_dir / "final_classifier.pkl", "wb") as f:
        pickle.dump(wrapper, f)
        
    # Write a mismatched feature order file
    with open(models_dir / "final_feature_order.json", "w") as f:
        json.dump(["mismatched_feature_name"], f)
        
    # Test classifier mock config
    rule_config = {
        "detection": {"bls_power_threshold": 0.15, "snr_threshold": 5.0},
        "classification": {"depth_threshold_eb": 0.050},
        "confidence": {},
        "ml_classifier": {"enabled": True, "dev_fallback": False, "model_path": str(models_dir / "final_classifier.pkl")}
    }
    config_path = models_dir / "rule_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(rule_config, f)
        
    # Try classifying — should fail on feature schema validation
    mock_features = {k: 0.1 for k in FEATURE_NAMES}
    mock_features["snr"] = 10.0
    mock_features["bls_power"] = 0.8
    with pytest.raises(ClassificationError) as excinfo:
        classify(mock_features, rule_config_path=str(config_path))
    assert "Feature schema mismatch" in str(excinfo.value)
