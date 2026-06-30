"""Tests for self-describing PyTorch model export."""

from pathlib import Path

import pytest
import torch

from transitlens_ml_core.export import (
    export_pytorch_checkpoint,
    load_pytorch_export,
)
from transitlens_ml_core.inference import Predictor

from .conftest import configured_model, export_config, sample_curve


def test_pytorch_export_is_self_describing_and_loadable(tmp_path: Path) -> None:
    config = export_config(tmp_path / "nested")
    model = configured_model(config)

    path = export_pytorch_checkpoint(model, config)
    payload = torch.load(path, weights_only=True)
    loaded = load_pytorch_export(path, torch.device("cpu"))

    assert path == config.export.output_directory / config.export.pytorch_filename
    assert payload["schema_version"] == 1
    assert payload["format"] == "transitlens-pytorch"
    assert payload["model_version"] == config.project.version
    assert payload["model_config"] == config.model.model_dump(mode="json")
    assert payload["input_field"] == config.data.model_input_field
    assert payload["classification_threshold"] == 0.5
    assert payload["input_name"] == "light_curve"
    assert payload["output_name"] == "transit_probability"
    assert not loaded.model.training
    assert loaded.model_version == config.project.version


def test_exported_predictor_matches_original_model(tmp_path: Path) -> None:
    config = export_config(tmp_path)
    model = configured_model(config)
    path = export_pytorch_checkpoint(model, config)
    curve = sample_curve()
    original = Predictor(
        model,
        config.data.model_input_field,
        config.evaluation.classification_threshold,
        config.project.version,
        torch.device("cpu"),
    )
    exported = Predictor.from_exported_checkpoint(path, torch.device("cpu"))

    expected = original.predict(curve)
    actual = exported.predict(curve)

    assert actual.probability == expected.probability
    assert actual.confidence == expected.confidence
    assert actual.predicted_class == expected.predicted_class
    assert actual.model_version == expected.model_version


def test_pytorch_export_rejects_incompatible_model(tmp_path: Path) -> None:
    config = export_config(tmp_path)
    incompatible = configured_model(config)
    incompatible.classifier.output = torch.nn.Linear(16, 1)

    with pytest.raises(ValueError, match="incompatible"):
        export_pytorch_checkpoint(incompatible, config)


def test_load_pytorch_export_rejects_missing_and_corrupt_artifacts(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_pytorch_export(tmp_path / "missing.pt", torch.device("cpu"))

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="unable to read"):
        load_pytorch_export(corrupt, torch.device("cpu"))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "root must be a mapping"),
        ({"schema_version": 1}, "missing fields"),
        (
            {
                "schema_version": 99,
                "format": "transitlens-pytorch",
                "model_version": "1.0",
                "model_config": {},
                "input_field": "wavelet_flux",
                "classification_threshold": 0.5,
                "input_name": "light_curve",
                "output_name": "transit_probability",
                "model_state_dict": {},
            },
            "schema version",
        ),
        (
            {
                "schema_version": 1,
                "format": "other",
                "model_version": "1.0",
                "model_config": {},
                "input_field": "wavelet_flux",
                "classification_threshold": 0.5,
                "input_name": "light_curve",
                "output_name": "transit_probability",
                "model_state_dict": {},
            },
            "format",
        ),
    ],
)
def test_load_pytorch_export_rejects_invalid_envelope(
    tmp_path: Path, payload: object, message: str
) -> None:
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match=message):
        load_pytorch_export(path, torch.device("cpu"))


def test_load_pytorch_export_rejects_invalid_embedded_state(tmp_path: Path) -> None:
    config = export_config(tmp_path)
    path = export_pytorch_checkpoint(configured_model(config), config)
    payload = torch.load(path, weights_only=True)
    payload["input_field"] = "flux"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="invalid model state"):
        load_pytorch_export(path, torch.device("cpu"))
