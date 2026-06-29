from __future__ import annotations
from pathlib import Path
import yaml
from .contracts import ContractError, PHYSICAL_CLASSES

TOP_LEVEL = {"experiment", "inputs", "outputs", "tabular", "timeseries", "fusion", "calibration", "uncertainty", "promotion", "inference"}

def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    unknown = set(config) - TOP_LEVEL
    missing = TOP_LEVEL - set(config)
    if unknown or missing:
        raise ContractError(f"invalid configuration sections; unknown={sorted(unknown)}, missing={sorted(missing)}")
    if tuple(config["experiment"]["allowed_classes"]) != PHYSICAL_CLASSES:
        raise ContractError("configuration violates fixed physical class order")
    if config["experiment"]["grouping_key"] != "tic_id" or config["experiment"]["evaluation_unit"] != "target":
        raise ContractError("TIC grouping and target-level evaluation are mandatory")
    if config["calibration"]["split_policy"] != "validation_only":
        raise ContractError("calibration must use validation only")
    if config["timeseries"]["ephemeris_mode"] != "detected":
        raise ContractError("official views must use detected ephemerides")
    return config
