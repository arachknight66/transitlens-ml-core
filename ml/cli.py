from __future__ import annotations
import argparse, json
from pathlib import Path

from .config import load_config
from .contracts import ContractError, read_feature_contract
from .dataset import verify_inputs
from .registry import promote, rollback
from .reporting import write_blocked_release

COMMANDS = ["verify-inputs", "audit-features", "build-views", "train-tabular", "train-timeseries",
            "train-fusion", "calibrate", "tune-abstention", "evaluate-validation", "freeze-candidate",
            "evaluate-test", "evaluate-external-sector", "robustness", "explain", "validate-artifacts",
            "promote", "registry-status", "rollback", "run-all"]

def _paths(config_path: Path, config: dict):
    core = config_path.parent.parent
    project = core.parent
    manifest = (core / config["inputs"]["manifest_dir"]).resolve()
    output = (core / config["outputs"]["run_dir"] / "latest").resolve()
    registry = (core / config["outputs"]["registry_dir"]).resolve()
    return manifest, output, registry

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TransitLens scientifically gated Phase 3")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "config" / "phase3_training.yaml"))
    parser.add_argument("--model-id"); parser.add_argument("--model-type"); parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--output-dir"); parser.add_argument("--run-id")
    parser.add_argument("--reason", default="operator requested rollback")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve(); config = load_config(config_path)
    manifest, output, registry = _paths(config_path, config)
    if args.output_dir: output = Path(args.output_dir).resolve()
    if args.limit and args.command in {"freeze-candidate", "evaluate-test", "promote", "run-all"}:
        raise ContractError("--limit is development-only and cannot create promotion evidence")

    if args.command == "audit-features":
        allowed, prohibited = read_feature_contract(config_path.parent / "contracts")
        print(json.dumps({"allowed_feature_count": len(allowed), "prohibited_feature_count": len(prohibited), "status": "PASS"}, indent=2)); return 0
    if args.command in {"verify-inputs", "run-all"}:
        report = verify_inputs(manifest)
        print(json.dumps(report, indent=2))
        if args.command == "run-all" and report["status"] != "PASS": write_blocked_release(output, report)
        return 0 if report["status"] == "PASS" else 2
    if args.command == "registry-status":
        pointer = registry / "active_model.json"
        print(pointer.read_text() if pointer.exists() else json.dumps({"status":"restricted", "active_model":None})); return 0
    if args.command == "promote":
        if not args.model_id: raise ContractError("--model-id is required")
        print(json.dumps(promote(output / args.model_id, registry), indent=2)); return 0
    if args.command == "rollback":
        print(json.dumps(rollback(registry, args.reason), indent=2)); return 0

    report = verify_inputs(manifest)
    if report["status"] != "PASS":
        write_blocked_release(output, report)
        raise ContractError(f"{args.command} blocked by frozen input verification")
    raise ContractError(f"{args.command} requires a completed prior artifact stage")

if __name__ == "__main__":
    raise SystemExit(main())
