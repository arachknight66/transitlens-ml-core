"""Immutable class, feature, and evaluation contracts for Phase 3."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

PHYSICAL_CLASSES = (
    "exoplanet_transit",
    "eclipsing_binary",
    "blend_contamination",
    "stellar_variability_or_other",
)
ROUTING_OUTCOME = "review_required"
METADATA_COLUMNS = {
    "target_id", "tic_id", "observation_id", "sector", "split",
    "canonical_label", "label_strength", "evidence_level", "source_checksum",
    "phase2_diagnostics_version", "feature_schema_version", "ephemeris_mode",
    "candidate_detected", "source_type",
}
PROHIBITED_FEATURES = {
    "true_period", "catalog_period", "true_depth", "true_duration", "disposition",
    "canonical_label", "label", "label_strength", "label_source", "source_catalog",
    "source_catalogs", "confirmed_planet_flag", "split", "target_id", "tic_id",
    "observation_id", "toi_id", "eb_id", "benchmark_status", "review_outcome",
    "test_membership",
}

class ContractError(ValueError):
    """Raised when frozen scientific input violates a Phase 3 contract."""

@dataclass(frozen=True)
class PromotionGates:
    macro_f1: float = 0.80
    planet_precision: float = 0.80
    planet_recall: float = 0.80
    eb_recall: float = 0.90
    blend_recall: float = 0.80
    blend_precision: float = 0.75
    ece: float = 0.05
    minimum_class_support: int = 1

def read_feature_contract(contract_dir: Path) -> tuple[list[str], set[str]]:
    allowed = json.loads((contract_dir / "allowed_features.json").read_text())
    prohibited = set(json.loads((contract_dir / "prohibited_features.json").read_text()))
    if len(allowed) != len(set(allowed)):
        raise ContractError("allowed feature order contains duplicates")
    overlap = set(allowed) & prohibited
    if overlap:
        raise ContractError(f"allowed/prohibited feature overlap: {sorted(overlap)}")
    return allowed, prohibited

def validate_probability_vector(probabilities: dict[str, float]) -> None:
    if tuple(probabilities) != PHYSICAL_CLASSES:
        raise ContractError("probability keys or class order violate the four-class contract")
    values = list(probabilities.values())
    if any((not isinstance(x, (int, float))) or x < 0 or x > 1 for x in values):
        raise ContractError("probabilities must be finite values in [0, 1]")
    if abs(sum(values) - 1.0) > 1e-6:
        raise ContractError("probabilities must sum to one")
