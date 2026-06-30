"""Tests for validated application configuration."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from transitlens_ml_core.config import AppConfig, load_config


@pytest.fixture
def valid_config() -> dict[str, object]:
    """Return a minimal valid configuration mapping."""
    return {
        "project": {"name": "test", "version": "1.0.0", "seed": 42},
        "data": {
            "input_directory": "data/processed",
            "time_field": "time",
            "normalized_flux_field": "normalized_flux",
            "wavelet_flux_field": "wavelet_flux",
            "metadata_field": "metadata",
            "model_input_field": "wavelet_flux",
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
        },
        "model": {
            "input_channels": 1,
            "convolution_channels": [16, 32],
            "kernel_size": 5,
            "pool_size": 2,
        },
        "training": {
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer_betas": [0.9, 0.999],
            "optimizer_epsilon": 1e-8,
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0.0001,
            "scheduler_patience": 2,
            "scheduler_factor": 0.5,
            "scheduler_min_lr": 1e-6,
            "checkpoint_directory": "weights",
            "best_checkpoint_filename": "best.pt",
            "latest_checkpoint_filename": "latest.pt",
            "experiment_directory": "experiments",
            "history_filename": "history.json",
        },
        "evaluation": {
            "classification_threshold": 0.5,
            "report_directory": "experiments",
            "report_filename": "evaluation.json",
        },
        "export": {
            "onnx_opset_version": 17,
            "sample_length": 128,
            "output_directory": "weights",
            "pytorch_filename": "model.pt",
            "onnx_filename": "model.onnx",
        },
    }


def write_config(path: Path, config: object) -> None:
    """Write test configuration as YAML."""
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_load_config_returns_validated_immutable_model(
    tmp_path: Path, valid_config: dict[str, object]
) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path, valid_config)

    config = load_config(config_path)

    assert isinstance(config, AppConfig)
    assert config.project.seed == 42
    assert config.data.model_input_field == "wavelet_flux"
    assert config.model.convolution_channels == (16, 32)
    with pytest.raises(ValidationError):
        config.project.seed = 7  # type: ignore[misc]


def test_load_config_rejects_invalid_split(
    tmp_path: Path, valid_config: dict[str, object]
) -> None:
    data_config = valid_config["data"]
    assert isinstance(data_config, dict)
    data_config["test_fraction"] = 0.2
    config_path = tmp_path / "config.yaml"
    write_config(config_path, valid_config)

    with pytest.raises(ValidationError, match="must sum to 1.0"):
        load_config(config_path)


def test_load_config_rejects_unknown_keys(
    tmp_path: Path, valid_config: dict[str, object]
) -> None:
    valid_config["unexpected"] = True
    config_path = tmp_path / "config.yaml"
    write_config(config_path, valid_config)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config(config_path)


@pytest.mark.parametrize("document", [None, [], "configuration"])
def test_load_config_rejects_non_mapping_root(tmp_path: Path, document: object) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path, document)

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_config(config_path)


def test_load_config_propagates_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does-not-exist.yaml")


@pytest.mark.parametrize(
    ("field", "filename", "suffix"),
    [
        ("pytorch_filename", "nested/model.pt", ".pt"),
        ("pytorch_filename", "model.bin", ".pt"),
        ("onnx_filename", "nested/model.onnx", ".onnx"),
        ("onnx_filename", "model.bin", ".onnx"),
    ],
)
def test_export_config_rejects_unsafe_filenames(
    valid_config: dict[str, object], field: str, filename: str, suffix: str
) -> None:
    export_config = valid_config["export"]
    assert isinstance(export_config, dict)
    export_config[field] = filename

    with pytest.raises(ValidationError, match=rf"{suffix}"):
        AppConfig.model_validate(valid_config)
