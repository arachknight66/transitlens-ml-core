"""Tests for the checked-in prototype configuration."""

from pathlib import Path

from transitlens_ml_core.config import load_config


def test_prototype_configuration_is_valid() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    config = load_config(repository_root / "configs" / "prototype.yaml")

    assert config.project.name == "transitlens-baseline"
    assert config.data.model_input_field == "wavelet_flux"
