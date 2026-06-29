"""Frozen-input verification and target-level tabular loading."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import pandas as pd

from .checksums import sha256_file, verify_registry
from .contracts import ContractError, PHYSICAL_CLASSES, PROHIBITED_FEATURES

SPLIT_FILES = {
    "train": "phase2_features_train.parquet",
    "validation": "phase2_features_validation.parquet",
    "test": "phase2_features_test.parquet",
}

def verify_inputs(manifest_dir: Path) -> dict:
    reasons: list[str] = []
    phase1 = json.loads((manifest_dir / "validation_report.json").read_text())
    phase2 = json.loads((manifest_dir / "phase2_validation_report.json").read_text())
    phase1_summary = json.loads((manifest_dir / "dataset_summary.json").read_text())
    split_integrity = json.loads((manifest_dir / "split_integrity_report.json").read_text())
    if phase1.get("status") != "PASS":
        reasons.append(f"Phase 1 release status is {phase1.get('status')}, not PASS")
    if phase2.get("status") not in {"PASS", "SUCCESS"}:
        reasons.append(f"Phase 2 release status is {phase2.get('status')}, not PASS")
    checksum_failures = verify_registry(manifest_dir / "checksums.sha256", manifest_dir)
    if checksum_failures:
        reasons.append(f"Phase 1 checksum mismatches: {checksum_failures}")
    registered = {line.split(maxsplit=1)[1].strip() for line in (manifest_dir / "checksums.sha256").read_text().splitlines() if line.strip()}
    phase2_required = set(SPLIT_FILES.values()) | {"phase2_feature_schema.json", "phase2_feature_order.json"}
    phase2_registry_path = manifest_dir / "phase2_artifact_checksums.json"
    phase2_registry = json.loads(phase2_registry_path.read_text()) if phase2_registry_path.exists() else {}
    phase2_unregistered = sorted(phase2_required - set(phase2_registry))
    phase2_checksum_failures = [name for name, expected in phase2_registry.items()
                                if (manifest_dir / name).exists() and sha256_file(manifest_dir / name) != expected]
    if phase2_unregistered: reasons.append(f"Phase 2 inputs are absent from its checksum registry: {phase2_unregistered}")
    if phase2_checksum_failures: reasons.append(f"Phase 2 checksum mismatches: {phase2_checksum_failures}")

    metadata = pd.read_parquet(manifest_dir / "phase2_feature_metadata.parquet")

    frames: dict[str, pd.DataFrame] = {}
    counts: dict[str, dict[str, int]] = {}
    tic_sets: dict[str, set] = {}
    for split, filename in SPLIT_FILES.items():
        frame = pd.read_parquet(manifest_dir / filename)
        frames[split] = frame
        if frame.empty:
            reasons.append(f"{filename} has zero rows")
        labelled = frame.merge(metadata[["observation_id", "canonical_label"]], on="observation_id", how="left", validate="one_to_one")
        label_col = "canonical_label"
        counts[split] = {c: int((labelled[label_col] == c).sum()) for c in PHYSICAL_CLASSES}
        missing_classes = [c for c, n in counts[split].items() if n == 0]
        if missing_classes:
            reasons.append(f"{split} has no support for: {', '.join(missing_classes)}")
        if labelled[label_col].isin(["review_required", "unlabeled"]).any():
            reasons.append(f"{split} contains a non-physical supervised label")
        tic_sets[split] = set(frame.get("tic_id", []))
        feature_columns = set(frame.columns) - {"tic_id", "observation_id", "sector", "split", "source_checksum",
                                                    "diagnostics_version", "feature_schema_version", "ephemeris_mode",
                                                    "candidate_detected", "diagnostic_status", "diagnostic_failure_reason"}
        leaked = feature_columns & PROHIBITED_FEATURES
        if leaked:
            reasons.append(f"{split} contains prohibited model features: {sorted(leaked)}")

    overlaps = {
        "train_validation": len(tic_sets["train"] & tic_sets["validation"]),
        "train_test": len(tic_sets["train"] & tic_sets["test"]),
        "validation_test": len(tic_sets["validation"] & tic_sets["test"]),
    }
    if any(overlaps.values()):
        reasons.append(f"TIC overlap detected: {overlaps}")

    feature_order_path = manifest_dir / "phase2_feature_order.json"
    schema_path = manifest_dir / "phase2_feature_schema.json"
    feature_order = json.loads(feature_order_path.read_text())
    schema = json.loads(schema_path.read_text())
    if list(schema) != feature_order:
        reasons.append("Phase 2 feature schema and order disagree")

    return {
        "status": "PASS" if not reasons else "BLOCKED",
        "phase1_status": phase1.get("status"),
        "phase2_status": phase2.get("status"),
        "phase1_dataset_version": phase1_summary.get("dataset_version"),
        "phase1_unique_tics": {
            "train": split_integrity.get("total_train_tics", 0),
            "validation": split_integrity.get("total_val_tics", 0),
            "test": split_integrity.get("total_test_tics", 0),
        },
        "phase1_class_counts": split_integrity.get("class_counts", {}),
        "phase1_checksum_registry_entries": len(registered),
        "phase1_checksum_failures": checksum_failures,
        "phase2_unregistered_inputs": phase2_unregistered,
        "phase2_checksum_failures": phase2_checksum_failures,
        "rows": {k: len(v) for k, v in frames.items()},
        "unique_tics": {k: len(v) for k, v in tic_sets.items()},
        "class_counts": counts,
        "tic_overlap": overlaps,
        "feature_schema_sha256": sha256_file(schema_path),
        "feature_order_sha256": sha256_file(feature_order_path),
        "reasons": reasons,
    }

def load_official_splits(manifest_dir: Path, feature_order: list[str]):
    report = verify_inputs(manifest_dir)
    if report["status"] != "PASS":
        raise ContractError("official training is blocked: " + "; ".join(report["reasons"]))
    result = {}
    for split, filename in SPLIT_FILES.items():
        df = pd.read_parquet(manifest_dir / filename)
        unknown = set(df.columns) - set(feature_order) - {"tic_id", "observation_id", "sector", "split", "source_checksum",
                                                              "diagnostics_version", "feature_schema_version", "ephemeris_mode",
                                                              "candidate_detected", "diagnostic_status", "diagnostic_failure_reason"}
        if unknown:
            raise ContractError(f"unknown columns in {split}: {sorted(unknown)}")
        metadata = pd.read_parquet(manifest_dir / "phase2_feature_metadata.parquet")
        result[split] = df.merge(metadata[["observation_id", "canonical_label"]], on="observation_id", how="left", validate="one_to_one")
    return result
