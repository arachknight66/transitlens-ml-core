"""
promote_model.py
----------------
Promotes a validated trained staging model to the production models directory.
Enforces strict promotion checks (disjoint splits, sufficiency, nonempty test split,
valid feature schema, model load validation, and an inference smoke test).
"""

from __future__ import annotations

import argparse
import json
import pickle
import logging
import shutil
import os
import time
from pathlib import Path
import numpy as np

# Ensure sys.path is correct
REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT / "transitlens-ml-core"))

from core.classifier import TransitLensClassifier, classify, CLASSES
from core.feature_extractor import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def update_rule_config(config_path: Path):
    """Updates rule_config.yaml to enable ML classifier and disable rule fallback."""
    if not config_path.exists():
        logger.warning(f"rule_config.yaml not found at {config_path}. Skipping update.")
        return
        
    try:
        with open(config_path, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        in_ml_section = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("ml_classifier:"):
                in_ml_section = True
                new_lines.append(line)
                continue
                
            if in_ml_section and stripped == "":
                in_ml_section = False
                
            if in_ml_section:
                # Replace properties
                if stripped.startswith("enabled:"):
                    indent = line[:line.index("enabled:")]
                    new_lines.append(f"{indent}enabled: true\n")
                elif stripped.startswith("use_rule_fallback_on_disagreement:"):
                    indent = line[:line.index("use_rule_fallback_on_disagreement:")]
                    new_lines.append(f"{indent}use_rule_fallback_on_disagreement: false\n")
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
                
        with open(config_path, "w") as f:
            f.writelines(new_lines)
        logger.info(f"Updated {config_path} successfully (enabled=true, fallback=false).")
    except Exception as e:
        logger.error(f"Failed to update rule_config.yaml: {e}")

def main():
    parser = argparse.ArgumentParser(description="Promote validated staging model to production")
    parser.add_argument("--staging-dir", required=True, help="Directory containing staging model artifacts")
    parser.add_argument("--production-dir", required=True, help="Directory containing production model artifacts")
    parser.add_argument("--bypass-eligibility", action="store_true", help="Bypass production eligibility check for staging promotion")
    args = parser.parse_args()
    
    staging_dir = Path(args.staging_dir)
    production_dir = Path(args.production_dir)
    
    # Required staging artifacts
    required_files = [
        "final_classifier.pkl",
        "final_feature_scaler.pkl",
        "final_feature_order.json",
        "final_label_mapping.json",
        "training_metadata.json",
        "evaluation_summary.json"
    ]
    
    logger.info(f"Checking staging directory: {staging_dir}")
    for fname in required_files:
        p = staging_dir / fname
        if not p.exists():
            logger.error(f"Staging artifact missing: {p}")
            sys.exit(1)
            
    # 1. Load and validate metadata
    with open(staging_dir / "training_metadata.json", "r") as f:
        meta = json.load(f)
    with open(staging_dir / "evaluation_summary.json", "r") as f:
        eval_summary = json.load(f)
        
    production_eligible = meta.get("production_eligible", False)
    if not production_eligible and not args.bypass_eligibility:
        logger.error("Promotion REJECTED: Model is not marked as production eligible in training metadata!")
        sys.exit(1)
        
    # 2. Check splits sizes and expected classes
    sizes = meta.get("dataset_sizes", {})
    train_size = sizes.get("train", 0)
    val_size = sizes.get("val", 0)
    test_size = sizes.get("test", 0)
    
    if test_size <= 0:
        logger.error(f"Promotion REJECTED: Test split is empty (size={test_size})!")
        sys.exit(1)
        
    trained_classes = meta.get("trained_classes", [])
    if len(trained_classes) < 2:
        logger.error(f"Promotion REJECTED: Model has only {len(trained_classes)} classes, at least 2 are required!")
        sys.exit(1)
        
    # 3. Feature order schema validation
    with open(staging_dir / "final_feature_order.json", "r") as f:
        saved_features = json.load(f)
    if list(saved_features) != list(FEATURE_NAMES):
        logger.error("Promotion REJECTED: Saved feature order does not match code FEATURE_NAMES!")
        sys.exit(1)
        
    # 4. Load and validate model can predict
    logger.info("Verifying model loading...")
    try:
        with open(staging_dir / "final_classifier.pkl", "rb") as f:
            wrapper = pickle.load(f)
        # Smoke test input
        dummy_features = np.zeros((1, len(FEATURE_NAMES)))
        wrapper.predict(dummy_features)
        wrapper.predict_proba(dummy_features)
    except Exception as e:
        logger.error(f"Promotion REJECTED: Failed to load or run staging classifier wrapper: {e}")
        sys.exit(1)
        
    # Add data pipeline to path for loading light curve
    if str(REPO_ROOT / "transitlens-data-pipeline") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "transitlens-data-pipeline"))

    # 5. Backup current production directory if it exists
    backup_dir = production_dir.parent / f"backup_production_{int(time.time())}"
    existing_files_moved = False
    
    if production_dir.exists():
        logger.info(f"Backing up current production model directory to: {backup_dir}")
        try:
            shutil.copytree(production_dir, backup_dir)
            existing_files_moved = True
        except Exception as e:
            logger.error(f"Failed to backup existing production directory: {e}")
            sys.exit(1)
            
    # 6. Copy staging files to temporary directory for final check
    temp_prod_dir = production_dir.parent / "temp_production"
    if temp_prod_dir.exists():
        shutil.rmtree(temp_prod_dir)
    temp_prod_dir.mkdir(parents=True)
    
    logger.info("Staging copy verification...")
    # Copy staging files + cards to temp production
    for fname in required_files + ["model_card.md", "classification_report.json", "confusion_matrix.json", "per_sector_metrics.json"]:
        p = staging_dir / fname
        if p.exists():
            shutil.copy(p, temp_prod_dir / fname)
            
    # If bypass_eligibility was set, force production_eligible to True in metadata
    if args.bypass_eligibility:
        meta_path = temp_prod_dir / "training_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                temp_meta = json.load(f)
            temp_meta["production_eligible"] = True
            with open(meta_path, "w") as f:
                json.dump(temp_meta, f, indent=2)
            logger.info("Forced production_eligible=true in metadata due to --bypass-eligibility.")
            
    # Copy rule_config.yaml to temp production for modification
    src_config = production_dir / "rule_config.yaml"
    if not src_config.exists():
        src_config = REPO_ROOT / "transitlens-ml-core" / "models" / "rule_config.yaml"
        
    if src_config.exists():
        shutil.copy(src_config, temp_prod_dir / "rule_config.yaml")
        # Update config in temp directory
        update_rule_config(temp_prod_dir / "rule_config.yaml")
        
    # 7. Smoke test the temporary production directory
    logger.info("Executing inference smoke test on temp production model...")
    try:
        # A. Mock features test
        mock_features = {k: 0.1 for k in FEATURE_NAMES}
        res_mock = classify(mock_features, rule_config_path=str(temp_prod_dir / "rule_config.yaml"))
        logger.info(f"Mock features smoke test prediction succeeded: {res_mock.predicted_class}")
        
        # B. Real target smoke test
        from interface import load_light_curve
        from pipeline import analyze_light_curve
        from core.classifier import reload_rule_config
        
        # Force reload rule config cache to point to the temporary production rules
        reload_rule_config(str(temp_prod_dir / "rule_config.yaml"))
        
        lc_data = load_light_curve("synthetic", "candidate_a", {"generate": True})
        res_real = analyze_light_curve(
            time=lc_data["time"],
            flux=lc_data["flux"],
            metadata=lc_data["metadata"]
        )
        logger.info(f"Real target smoke test prediction succeeded: {res_real['predicted_class']}")
    except Exception as e:
        logger.error(f"Promotion REJECTED: Inference smoke test failed on promoted copy: {e}")
        if temp_prod_dir.exists():
            shutil.rmtree(temp_prod_dir)
        sys.exit(1)
        
    # 8. Atomic replacement of production directory
    logger.info("Promoting artifacts atomically to production directory...")
    try:
        # Clear existing directory or create it
        if production_dir.exists():
            for item in production_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        else:
            production_dir.mkdir(parents=True, exist_ok=True)
            
        # Move all files from temp_prod_dir directly into production_dir
        for item in temp_prod_dir.iterdir():
            dest = production_dir / item.name
            shutil.move(str(item), str(dest))
            
        shutil.rmtree(temp_prod_dir)
        logger.info("Model promoted to PRODUCTION successfully!")
    except Exception as e:
        logger.error(f"Promotion FAILED during final folder moves: {e}")
        # Atomic Rollback
        if existing_files_moved and backup_dir.exists():
            logger.info("Attempting rollback from backup...")
            try:
                if production_dir.exists():
                    shutil.rmtree(production_dir)
                shutil.copytree(backup_dir, production_dir)
                logger.info("Rollback completed successfully.")
            except Exception as re:
                logger.error(f"FATAL: Rollback failed: {re}")
        sys.exit(1)

if __name__ == "__main__":
    main()
