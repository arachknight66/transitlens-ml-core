"""Hash-verified, atomic model registry with strict promotion and rollback."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json, os, shutil, uuid

from .checksums import sha256_file

class RegistryError(RuntimeError): pass

def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)

def validate_candidate(candidate: Path) -> dict:
    record = json.loads((candidate / "evaluation_record.json").read_text())
    gate = json.loads((candidate / "promotion_gate.json").read_text())
    metadata = json.loads((candidate / "training_metadata.json").read_text())
    if not gate.get("production_eligible"):
        raise RegistryError("candidate failed the immutable promotion gate")
    if metadata.get("production_eligible") is not True or record.get("production_eligible") is not True:
        raise RegistryError("evaluation record and metadata do not agree")
    hashes = json.loads((candidate / "artifact_checksums.json").read_text())
    for relative, expected in hashes.items():
        path = candidate / relative
        if not path.exists() or sha256_file(path) != expected:
            raise RegistryError(f"artifact checksum mismatch: {relative}")
    required = ["model.joblib", "preprocessor.joblib", "calibration.joblib", "model_card.md"]
    if any(not (candidate / name).exists() for name in required):
        raise RegistryError("candidate is missing a required artifact")
    return record

def promote(candidate: Path, registry: Path) -> dict:
    record = validate_candidate(candidate)
    model_id = record["model_id"]
    destination = registry / "models" / model_id
    if destination.exists():
        raise RegistryError(f"immutable model already exists: {model_id}")
    stage = registry / ".staging" / uuid.uuid4().hex
    stage.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(candidate, stage)
    validate_candidate(stage)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, destination)
    active_path = registry / "active_model.json"
    previous = json.loads(active_path.read_text()).get("model_id") if active_path.exists() else None
    pointer = {"model_id": model_id, "previous_model_id": previous, "activated_at": datetime.now(timezone.utc).isoformat()}
    _atomic_json(active_path, pointer)
    with (registry / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"action": "promote", **pointer}) + "\n")
    return pointer

def rollback(registry: Path, reason: str) -> dict:
    active_path = registry / "active_model.json"
    pointer = json.loads(active_path.read_text())
    previous = pointer.get("previous_model_id")
    if not previous:
        raise RegistryError("no rollback target is recorded")
    validate_candidate(registry / "models" / previous)
    new_pointer = {"model_id": previous, "previous_model_id": pointer["model_id"],
                   "activated_at": datetime.now(timezone.utc).isoformat(), "rollback_reason": reason}
    _atomic_json(active_path, new_pointer)
    with (registry / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"action": "rollback", **new_pointer}) + "\n")
    return new_pointer
