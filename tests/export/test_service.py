"""Tests for composed platform model export."""

from pathlib import Path

from transitlens_ml_core.export import ExportArtifacts, export_models

from .conftest import configured_model, export_config


def test_export_models_produces_both_platform_artifacts(tmp_path: Path) -> None:
    config = export_config(tmp_path)

    artifacts = export_models(configured_model(config), config)

    assert isinstance(artifacts, ExportArtifacts)
    assert artifacts.pytorch_checkpoint.is_file()
    assert artifacts.onnx_model.is_file()
    assert artifacts.pytorch_checkpoint.suffix == ".pt"
    assert artifacts.onnx_model.suffix == ".onnx"
