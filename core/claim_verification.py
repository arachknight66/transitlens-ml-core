"""
core/claim_verification.py
--------------------------
Validates current run execution metrics against the reference results manifest.
"""

import json
from pathlib import Path

def get_nested_val(d: dict, dotted_key: str):
    """Retrieves a nested value from a dictionary using dot notation (e.g. 'val_metrics.accuracy')."""
    parts = dotted_key.split(".")
    current = d
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current

def verify_claims(
    metrics_json_path: Path,
    reference_json_path: Path
) -> dict:
    """
    Compares metrics from metrics_json_path against reference_json_path.
    Returns audit details with PASS/FAIL status.
    """
    audit = {
        "status": "PASSED",
        "timestamp": "",
        "results": []
    }
    
    if not metrics_json_path.exists():
        audit["status"] = "FAILED"
        audit["error"] = f"Metrics file not found: {metrics_json_path.name}"
        return audit
        
    if not reference_json_path.exists():
        audit["status"] = "FAILED"
        audit["error"] = f"Reference manifest not found: {reference_json_path.name}"
        return audit
        
    try:
        with open(metrics_json_path, "r") as f:
            metrics = json.load(f)
        with open(reference_json_path, "r") as f:
            reference = json.load(f)
    except Exception as exc:
        audit["status"] = "FAILED"
        audit["error"] = f"Failed to parse files: {exc}"
        return audit
        
    for claim_name, spec in reference.get("claims", {}).items():
        field = spec.get("field")
        expected_val = spec.get("value")
        tolerance = spec.get("tolerance", 0.0)
        
        actual_val = get_nested_val(metrics, field)
        if actual_val is None:
            audit["status"] = "FAILED"
            audit["results"].append({
                "claim": claim_name,
                "field": field,
                "status": "MISSING",
                "expected": expected_val,
                "actual": None
            })
            continue
            
        # Tolerant comparison
        diff = abs(actual_val - expected_val)
        passed = diff <= tolerance
        
        audit["results"].append({
            "claim": claim_name,
            "field": field,
            "status": "PASSED" if passed else "FAILED",
            "expected": expected_val,
            "actual": actual_val,
            "difference": diff,
            "tolerance": tolerance
        })
        
        if not passed:
            audit["status"] = "FAILED"
            
    return audit
