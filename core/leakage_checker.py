"""
core/leakage_checker.py
-----------------------
Utility to perform strict leakage checks across train, validation, and test splits.
Checks target overlaps, system/sector overlaps, and empty splits.
"""

import pandas as pd
import json
from pathlib import Path

def extract_system_id(target_id: str) -> str:
    """
    Extracts the base system ID (e.g., TIC number) to prevent sector/observation cross-leakage.
    Examples:
      - "TIC-261136679_sec98" -> "261136679"
      - "KIC-9221398"         -> "9221398"
      - "candidate_a"        -> "candidate_a"
    """
    # Strip sector suffixes if present
    base = target_id.split("_")[0].strip()
    digits = "".join(c for c in base if c.isdigit())
    if digits:
        return digits
    return base

def run_leakage_audit(splits_dir: Path, output_json_path: Path | None = None) -> dict:
    """
    Performs data split leakage check.
    Saves a JSON report to output_json_path if provided.
    """
    train_file = splits_dir / "train_targets.csv"
    val_file = splits_dir / "val_targets.csv"
    test_file = splits_dir / "test_targets.csv"
    
    audit_report = {
        "status": "PASSED",
        "timestamp": pd.Timestamp.now().isoformat(),
        "splits_found": {
            "train": train_file.exists(),
            "val": val_file.exists(),
            "test": test_file.exists(),
        },
        "target_counts": {"train": 0, "val": 0, "test": 0},
        "target_overlaps": [],
        "system_overlaps": [],
        "empty_splits": [],
        "errors": []
    }
    
    # Check if files exist
    for split_name, f in [("train", train_file), ("val", val_file), ("test", test_file)]:
        if not f.exists():
            audit_report["errors"].append(f"Missing split file: {f.name}")
            audit_report["status"] = "FAILED"
            
    if audit_report["status"] == "FAILED":
        if output_json_path:
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, "w") as out:
                json.dump(audit_report, out, indent=2)
        return audit_report
        
    # Read files
    try:
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
    except Exception as exc:
        audit_report["errors"].append(f"Failed to read split files: {exc}")
        audit_report["status"] = "FAILED"
        if output_json_path:
            with open(output_json_path, "w") as out:
                json.dump(audit_report, out, indent=2)
        return audit_report
        
    splits_data = {
        "train": set(train_df["target_id"].dropna().astype(str).tolist()),
        "val": set(val_df["target_id"].dropna().astype(str).tolist()),
        "test": set(test_df["target_id"].dropna().astype(str).tolist()),
    }
    
    for split_name, t_set in splits_data.items():
        audit_report["target_counts"][split_name] = len(t_set)
        if len(t_set) == 0:
            audit_report["empty_splits"].append(split_name)
            audit_report["status"] = "FAILED"
            
    # 1. Target ID Overlaps Check
    overlaps_pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for s1, s2 in overlaps_pairs:
        intersect = splits_data[s1].intersection(splits_data[s2])
        if intersect:
            audit_report["target_overlaps"].append({
                "split1": s1,
                "split2": s2,
                "overlap_count": len(intersect),
                "overlapping_targets": list(intersect)[:10]
            })
            audit_report["status"] = "FAILED"
            
    # 2. System ID Leakage Check (checking base astronomical objects)
    splits_systems = {
        name: {extract_system_id(tid) for tid in t_set}
        for name, t_set in splits_data.items()
    }
    
    for s1, s2 in overlaps_pairs:
        intersect_sys = splits_systems[s1].intersection(splits_systems[s2])
        if intersect_sys:
            audit_report["system_overlaps"].append({
                "split1": s1,
                "split2": s2,
                "overlap_count": len(intersect_sys),
                "overlapping_systems": list(intersect_sys)[:10]
            })
            audit_report["status"] = "FAILED"
            
    # Save the output json
    if output_json_path:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w") as out:
            json.dump(audit_report, out, indent=2)
            
    return audit_report
