"""
tests/test_reproducibility.py
-----------------------------
Unit and integration tests for Phase 8 reproducibility modules:
  - config_schema (Pydantic validation)
  - seeds (deterministic derivation)
  - structured_logger (JSON formatter, telemetry)
  - leakage_checker (split disjointness)
  - run_manager (run directories, checksums, resumability)
  - claim_verification (metric tolerance checks)
"""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


# =========================================================================
# 1. Config Schema
# =========================================================================

class TestConfigSchema:
    """Pydantic schema validation for TransitLens configuration."""

    def test_valid_default_config_passes(self):
        """The shipped config.yaml must pass schema validation."""
        from core.config_schema import validate_config

        cfg_path = Path(__file__).parent.parent / "config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        result = validate_config(cfg)
        assert result.version == "0.1.0"

    def test_extra_key_rejected(self):
        """An unknown top-level key must raise ValidationError."""
        from core.config_schema import validate_config
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="extra_forbidden"):
            validate_config({"version": "0.1.0", "bogus_key": 42})

    def test_extra_nested_key_rejected(self):
        """An unknown key inside a nested section must raise ValidationError."""
        from core.config_schema import validate_config
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            validate_config({"preprocessing": {"sigma_upper": 5.0, "banana": 3}})

    def test_out_of_range_value_rejected(self):
        """A negative sigma_upper (ge=0) must be rejected."""
        from core.config_schema import validate_config
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            validate_config({"preprocessing": {"sigma_upper": -1.0}})

    def test_empty_dict_uses_defaults(self):
        """An empty dict should succeed using all defaults."""
        from core.config_schema import validate_config

        result = validate_config({})
        assert result.bls.period_min_days == 0.5
        assert result.fitting.random_seed == 42

    def test_partial_override_merges(self):
        """Providing only bls.period_min_days should leave other BLS defaults intact."""
        from core.config_schema import validate_config

        result = validate_config({"bls": {"period_min_days": 1.0}})
        assert result.bls.period_min_days == 1.0
        assert result.bls.n_oversample == 10  # default preserved

    def test_api_port_bounds(self):
        """Port outside 1-65535 must be rejected."""
        from core.config_schema import validate_config
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            validate_config({"api": {"port": 70000}})


# =========================================================================
# 2. Deterministic Seed Derivation
# =========================================================================

class TestSeeds:
    """Deterministic, platform-independent seed derivation."""

    def test_same_inputs_same_seed(self):
        from core.seeds import derive_seed
        s1 = derive_seed(42, "mcmc", "TIC-12345")
        s2 = derive_seed(42, "mcmc", "TIC-12345")
        assert s1 == s2

    def test_different_stage_different_seed(self):
        from core.seeds import derive_seed
        s1 = derive_seed(42, "mcmc", "TIC-12345")
        s2 = derive_seed(42, "bootstrap", "TIC-12345")
        assert s1 != s2

    def test_different_target_different_seed(self):
        from core.seeds import derive_seed
        s1 = derive_seed(42, "mcmc", "TIC-12345")
        s2 = derive_seed(42, "mcmc", "TIC-67890")
        assert s1 != s2

    def test_different_master_different_seed(self):
        from core.seeds import derive_seed
        s1 = derive_seed(42, "mcmc", "TIC-12345")
        s2 = derive_seed(99, "mcmc", "TIC-12345")
        assert s1 != s2

    def test_seed_fits_32bit(self):
        from core.seeds import derive_seed
        s = derive_seed(42, "mcmc", "TIC-12345")
        assert 0 <= s < 2**32

    def test_seed_usable_with_numpy_rng(self):
        """Derived seed must work with np.random.default_rng without error."""
        from core.seeds import derive_seed
        s = derive_seed(42, "injection", "candidate_a")
        rng = np.random.default_rng(s)
        val = rng.random()
        assert 0.0 <= val < 1.0


# =========================================================================
# 3. Structured Logger
# =========================================================================

class TestStructuredLogger:
    """JSON formatter and telemetry tracking."""

    def test_json_formatter_produces_valid_json(self):
        from core.structured_logger import JsonFormatter
        fmt = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None,
        )
        line = fmt.format(record)
        parsed = json.loads(line)
        assert parsed["message"] == "hello world"
        assert parsed["level"] == "INFO"

    def test_json_formatter_includes_extras(self):
        from core.structured_logger import JsonFormatter
        fmt = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="stage done", args=(), exc_info=None,
        )
        record.target_id = "TIC-12345"
        record.elapsed_ms = 500.0
        line = fmt.format(record)
        parsed = json.loads(line)
        assert parsed["target_id"] == "TIC-12345"
        assert parsed["elapsed_ms"] == 500.0

    def test_setup_and_finalize_telemetry(self):
        from core.structured_logger import setup_structured_logging, finalize_telemetry, run_telemetry
        setup_structured_logging(logging.WARNING, run_id="test-run-001")
        assert run_telemetry["run_id"] == "test-run-001"
        assert run_telemetry["status"] == "RUNNING"

        result = finalize_telemetry("COMPLETED")
        assert result["status"] == "COMPLETED"
        assert result["end_time"] != ""


# =========================================================================
# 4. Leakage Checker
# =========================================================================

class TestLeakageChecker:
    """Data-split leakage auditing."""

    def _make_splits(self, tmp: Path, train_ids, val_ids, test_ids):
        """Helper to write split CSVs."""
        for name, ids in [("train_targets", train_ids), ("val_targets", val_ids), ("test_targets", test_ids)]:
            df = pd.DataFrame({"target_id": ids, "label": ["x"] * len(ids)})
            df.to_csv(tmp / f"{name}.csv", index=False)

    def test_disjoint_splits_pass(self, tmp_path):
        from core.leakage_checker import run_leakage_audit
        self._make_splits(tmp_path, ["A", "B"], ["C", "D"], ["E", "F"])
        audit = run_leakage_audit(tmp_path)
        assert audit["status"] == "PASSED"
        assert len(audit["target_overlaps"]) == 0

    def test_overlapping_targets_detected(self, tmp_path):
        from core.leakage_checker import run_leakage_audit
        self._make_splits(tmp_path, ["A", "B"], ["B", "C"], ["D"])
        audit = run_leakage_audit(tmp_path)
        assert audit["status"] == "FAILED"
        assert any(o["split1"] == "train" and o["split2"] == "val" for o in audit["target_overlaps"])

    def test_system_id_leakage_detected(self, tmp_path):
        from core.leakage_checker import run_leakage_audit
        # Same TIC number appears in train and test under different sector suffixes
        self._make_splits(tmp_path, ["TIC-12345_sec1"], ["C"], ["TIC-12345_sec2"])
        audit = run_leakage_audit(tmp_path)
        assert audit["status"] == "FAILED"
        assert len(audit["system_overlaps"]) > 0

    def test_empty_split_flagged(self, tmp_path):
        from core.leakage_checker import run_leakage_audit
        self._make_splits(tmp_path, ["A"], [], ["B"])
        audit = run_leakage_audit(tmp_path)
        assert audit["status"] == "FAILED"
        assert "val" in audit["empty_splits"]

    def test_missing_file_flagged(self, tmp_path):
        from core.leakage_checker import run_leakage_audit
        # Only create train, not val or test
        pd.DataFrame({"target_id": ["A"]}).to_csv(tmp_path / "train_targets.csv", index=False)
        audit = run_leakage_audit(tmp_path)
        assert audit["status"] == "FAILED"
        assert len(audit["errors"]) > 0

    def test_output_json_written(self, tmp_path):
        from core.leakage_checker import run_leakage_audit
        self._make_splits(tmp_path, ["A"], ["B"], ["C"])
        out_file = tmp_path / "audit.json"
        run_leakage_audit(tmp_path, out_file)
        assert out_file.exists()
        with open(out_file) as f:
            data = json.load(f)
        assert data["status"] == "PASSED"


# =========================================================================
# 5. Run Manager
# =========================================================================

class TestRunManager:
    """Run directory creation, checksums, and resumability."""

    def test_setup_creates_standard_dirs(self, tmp_path):
        from core.run_manager import RunManager
        mgr = RunManager(tmp_path, "run-001")
        run_dir = mgr.setup_directories({"version": "0.1.0"})
        assert (run_dir / "logs").is_dir()
        assert (run_dir / "predictions").is_dir()
        assert (run_dir / "plots").is_dir()
        assert (run_dir / "resolved_config.yaml").exists()
        assert (run_dir / "environment.json").exists()

    def test_overwrite_protection(self, tmp_path):
        from core.run_manager import RunManager
        mgr1 = RunManager(tmp_path, "run-001")
        mgr1.setup_directories({"version": "0.1.0"})
        # Second manager without resume should fail
        mgr2 = RunManager(tmp_path, "run-001", resume=False)
        with pytest.raises(FileExistsError):
            mgr2.setup_directories({"version": "0.1.0"})

    def test_resume_loads_existing_manifest(self, tmp_path):
        from core.run_manager import RunManager
        mgr = RunManager(tmp_path, "run-001")
        mgr.setup_directories({"version": "0.1.0"})
        mgr.finalize_run("COMPLETED")

        mgr2 = RunManager(tmp_path, "run-001", resume=True)
        mgr2.setup_directories({"version": "0.1.0"})
        assert mgr2.manifest["status"] == "RESUMED"

    def test_record_artifact_computes_hash(self, tmp_path):
        from core.run_manager import RunManager
        mgr = RunManager(tmp_path, "run-001")
        run_dir = mgr.setup_directories({"version": "0.1.0"})
        # Create a dummy artifact file
        art_file = run_dir / "predictions" / "test.json"
        art_file.write_text('{"class": "exoplanet"}')
        entry = mgr.record_artifact("target_x", "classify", "predictions/test.json", "ClassResult", "0.1")
        assert entry["hash"]
        assert entry["size_bytes"] > 0

    def test_finalize_creates_checksums_file(self, tmp_path):
        from core.run_manager import RunManager
        mgr = RunManager(tmp_path, "run-001")
        run_dir = mgr.setup_directories({"version": "0.1.0"})
        mgr.finalize_run("COMPLETED")
        checksums = run_dir / "checksums.sha256"
        assert checksums.exists()
        content = checksums.read_text()
        # Should contain entries for resolved_config.yaml, environment.json, manifest.json
        assert "resolved_config.yaml" in content
        assert "environment.json" in content

    def test_is_stage_completed_false_without_resume(self, tmp_path):
        from core.run_manager import RunManager
        mgr = RunManager(tmp_path, "run-001")
        mgr.setup_directories({"version": "0.1.0"})
        assert mgr.is_stage_completed("t1", "classify", "predictions/t1.json") is False


# =========================================================================
# 6. Claim Verification
# =========================================================================

class TestClaimVerification:
    """Metric tolerance comparison against reference manifest."""

    def _write_json(self, path: Path, data: dict):
        path.write_text(json.dumps(data))

    def test_matching_claims_pass(self, tmp_path):
        from core.claim_verification import verify_claims
        metrics = {"accuracy": 1.0, "nested": {"f1": 0.95}}
        ref = {"claims": {
            "acc": {"field": "accuracy", "value": 1.0, "tolerance": 0.01},
            "f1":  {"field": "nested.f1", "value": 0.95, "tolerance": 0.01},
        }}
        self._write_json(tmp_path / "m.json", metrics)
        self._write_json(tmp_path / "r.json", ref)
        audit = verify_claims(tmp_path / "m.json", tmp_path / "r.json")
        assert audit["status"] == "PASSED"
        assert all(r["status"] == "PASSED" for r in audit["results"])

    def test_out_of_tolerance_fails(self, tmp_path):
        from core.claim_verification import verify_claims
        metrics = {"accuracy": 0.80}
        ref = {"claims": {"acc": {"field": "accuracy", "value": 1.0, "tolerance": 0.01}}}
        self._write_json(tmp_path / "m.json", metrics)
        self._write_json(tmp_path / "r.json", ref)
        audit = verify_claims(tmp_path / "m.json", tmp_path / "r.json")
        assert audit["status"] == "FAILED"

    def test_missing_field_fails(self, tmp_path):
        from core.claim_verification import verify_claims
        metrics = {}
        ref = {"claims": {"acc": {"field": "accuracy", "value": 1.0, "tolerance": 0.01}}}
        self._write_json(tmp_path / "m.json", metrics)
        self._write_json(tmp_path / "r.json", ref)
        audit = verify_claims(tmp_path / "m.json", tmp_path / "r.json")
        assert audit["status"] == "FAILED"
        assert audit["results"][0]["status"] == "MISSING"

    def test_missing_metrics_file(self, tmp_path):
        from core.claim_verification import verify_claims
        ref = {"claims": {}}
        self._write_json(tmp_path / "r.json", ref)
        audit = verify_claims(tmp_path / "nonexistent.json", tmp_path / "r.json")
        assert audit["status"] == "FAILED"

    def test_nested_dot_key_access(self, tmp_path):
        from core.claim_verification import verify_claims
        metrics = {"a": {"b": {"c": 42.0}}}
        ref = {"claims": {"deep": {"field": "a.b.c", "value": 42.0, "tolerance": 0.5}}}
        self._write_json(tmp_path / "m.json", metrics)
        self._write_json(tmp_path / "r.json", ref)
        audit = verify_claims(tmp_path / "m.json", tmp_path / "r.json")
        assert audit["status"] == "PASSED"


# =========================================================================
# 7. extract_system_id helper
# =========================================================================

class TestExtractSystemId:
    """Unit tests for the system-ID extraction helper in leakage_checker."""

    def test_tic_with_sector(self):
        from core.leakage_checker import extract_system_id
        assert extract_system_id("TIC-261136679_sec98") == "261136679"

    def test_kic_no_sector(self):
        from core.leakage_checker import extract_system_id
        assert extract_system_id("KIC-9221398") == "9221398"

    def test_plain_name(self):
        from core.leakage_checker import extract_system_id
        assert extract_system_id("candidate_a") == "candidate"
