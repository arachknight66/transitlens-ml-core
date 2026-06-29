from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json

def write_blocked_release(output: Path, verification: dict) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "phase3_status": "PARTIAL",
        "production_eligible": False,
        "ml_enabled": False,
        "model_id": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_verification": verification,
        "blind_test_metrics": None,
        "external_sector_metrics": None,
        "reason": "Scientific training was not run because frozen prerequisites failed.",
    }
    gate = {
        "production_eligible": False,
        "checks": {"phase1_pass": verification["phase1_status"] == "PASS",
                   "phase2_features_nonempty": all(n > 0 for n in verification["rows"].values()),
                   "all_physical_classes_supported": all(all(n > 0 for n in split.values()) for split in verification["class_counts"].values()),
                   "zero_target_overlap": not any(verification["tic_overlap"].values()),
                   "blind_test_gates": False},
        "failures": verification["reasons"],
    }
    (output / "input_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    (output / "leakage_audit.json").write_text(json.dumps({"tic_overlap": verification["tic_overlap"], "pass": not any(verification["tic_overlap"].values())}, indent=2), encoding="utf-8")
    (output / "evaluation_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (output / "training_metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (output / "promotion_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    report = f"""# TransitLens Phase 3 scientific report

**Status: PARTIAL**  
**Production eligible: false**  
**ML enabled: false**

The frozen scientific prerequisites block an official training run. Phase 1 is `{verification['phase1_status']}`. Frozen Phase 2 train/validation/test feature rows are `{verification['rows']}`. No blind-test metrics were computed and no model was promoted.

## Release blockers

""" + "\n".join(f"- {reason}" for reason in verification["reasons"]) + "\n"
    (output / "phase3_scientific_report.md").write_text(report, encoding="utf-8")
    card = """# TransitLens Phase 3 model card

No model card is available because no scientifically eligible model was trained. This record is generated from the same immutable release record as training metadata.

- Model ID: none
- Physical classes: exoplanet_transit, eclipsing_binary, blend_contamination, stellar_variability_or_other
- `review_required`: abstention/routing outcome only
- Production eligible: false
- ML enabled: false
- Intended status: rule-only restricted; probabilities unavailable
- Limitation: frozen Phase 2 feature splits are empty and Phase 1 is not PASS
"""
    (output / "model_card.md").write_text(card, encoding="utf-8")
    return record
