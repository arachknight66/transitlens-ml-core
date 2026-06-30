"""Tests for checkpoint loading and single-record inference."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from transitlens_ml_core.config import ModelConfig, load_config
from transitlens_ml_core.datasets import ProcessedLightCurve
from transitlens_ml_core.inference import (
    PredictionResult,
    Predictor,
    load_baseline_checkpoint,
)
from transitlens_ml_core.models import BaselineCNN


class FluxMeanModel(nn.Module):
    """Return mean input flux as one probability."""

    input_channels = 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Average channel and cadence dimensions."""
        return inputs.mean(dim=(1, 2), keepdim=True)


class FixedOutputModel(nn.Module):
    """Return a configured output tensor for validation tests."""

    input_channels = 1

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_output", output)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the fixed output regardless of input."""
        del inputs
        return self.fixed_output


class MultiChannelModel(FluxMeanModel):
    """Advertise an incompatible processed-input channel count."""

    input_channels = 2


def model_config() -> ModelConfig:
    """Create a compact baseline architecture configuration."""
    return ModelConfig(
        input_channels=1,
        convolution_channels=(2, 4),
        kernel_size=3,
        pool_size=2,
    )


def light_curve() -> ProcessedLightCurve:
    """Create a valid record whose flux representations differ."""
    return ProcessedLightCurve(
        time=np.arange(64, dtype=np.float64),
        normalized_flux=np.full(64, 0.8, dtype=np.float64),
        wavelet_flux=np.full(64, 0.6, dtype=np.float64),
        metadata={"schema_version": "1.0"},
    )


def write_checkpoint(path: Path, config: ModelConfig | None = None) -> Path:
    """Write a valid baseline model-state checkpoint."""
    model = BaselineCNN.from_config(config or model_config())
    torch.save({"model_state_dict": model.state_dict()}, path)
    return path


def test_predict_returns_stable_platform_contract_for_selected_flux() -> None:
    predictor = Predictor(
        FluxMeanModel(),  # type: ignore[arg-type]
        input_field="wavelet_flux",
        classification_threshold=0.5,
        model_version="1.2.3",
        device=torch.device("cpu"),
    )

    result = predictor.predict(light_curve())

    assert isinstance(result, PredictionResult)
    assert result.probability == pytest.approx(0.6)
    assert result.confidence == pytest.approx(0.2)
    assert result.predicted_class == 1
    assert result.model_version == "1.2.3"
    assert result.inference_time >= 0.0
    assert list(result.to_dict()) == [
        "probability",
        "confidence",
        "predicted_class",
        "model_version",
        "inference_time",
    ]


def test_predict_can_select_normalized_flux_and_threshold_is_inclusive() -> None:
    predictor = Predictor(
        FluxMeanModel(),  # type: ignore[arg-type]
        input_field="normalized_flux",
        classification_threshold=0.8,
        model_version="1.0.0",
        device=torch.device("cpu"),
    )

    result = predictor.predict(light_curve())

    assert result.probability == pytest.approx(0.8)
    assert result.confidence == pytest.approx(0.0, abs=1e-6)
    assert result.predicted_class == 1


def test_loaded_baseline_prediction_is_deterministic_and_fast(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    predictor = Predictor.from_checkpoint(
        checkpoint,
        model_config(),
        input_field="wavelet_flux",
        classification_threshold=0.5,
        model_version="1.0.0",
        device=torch.device("cpu"),
    )

    predictor.predict(light_curve())
    first = predictor.predict(light_curve())
    second = predictor.predict(light_curve())

    assert first.probability == second.probability
    assert first.confidence == second.confidence
    assert first.predicted_class == second.predicted_class
    assert first.inference_time < 100.0
    assert second.inference_time < 100.0
    assert not predictor.model.training


def test_predictor_from_application_config(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config = load_config(repository_root / "configs" / "prototype.yaml")
    checkpoint = write_checkpoint(tmp_path / "model.pt", config.model)

    predictor = Predictor.from_config(checkpoint, config, torch.device("cpu"))
    result = predictor.predict(light_curve())

    assert predictor.input_field == config.data.model_input_field
    assert result.model_version == config.project.version


@pytest.mark.parametrize(
    ("input_field", "threshold", "version", "model", "message"),
    [
        ("flux", 0.5, "1.0", FluxMeanModel(), "input_field"),
        ("wavelet_flux", 0.0, "1.0", FluxMeanModel(), "threshold"),
        ("wavelet_flux", float("nan"), "1.0", FluxMeanModel(), "threshold"),
        ("wavelet_flux", 0.5, "", FluxMeanModel(), "model_version"),
        ("wavelet_flux", 0.5, "1.0", MultiChannelModel(), "one input channel"),
    ],
)
def test_predictor_rejects_invalid_configuration(
    input_field: str,
    threshold: float,
    version: str,
    model: nn.Module,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Predictor(
            model,  # type: ignore[arg-type]
            input_field,  # type: ignore[arg-type]
            threshold,
            version,
            torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (torch.tensor([[0.2, 0.8]]), "exactly one"),
        (torch.tensor([[float("nan")]]), "finite"),
        (torch.tensor([[1.1]]), r"range \[0, 1\]"),
    ],
)
def test_predict_rejects_invalid_model_output(
    output: torch.Tensor, message: str
) -> None:
    predictor = Predictor(
        FixedOutputModel(output),  # type: ignore[arg-type]
        "wavelet_flux",
        0.5,
        "1.0",
        torch.device("cpu"),
    )

    with pytest.raises(ValueError, match=message):
        predictor.predict(light_curve())


def test_load_checkpoint_rejects_missing_and_unreadable_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_baseline_checkpoint(
            tmp_path / "missing.pt", model_config(), torch.device("cpu")
        )

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="unable to read"):
        load_baseline_checkpoint(corrupt, model_config(), torch.device("cpu"))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "root must be a mapping"),
        ({"schema_version": 1}, "missing model_state_dict"),
        ({"model_state_dict": {}}, "incompatible"),
    ],
)
def test_load_checkpoint_rejects_invalid_state(
    tmp_path: Path, payload: object, message: str
) -> None:
    checkpoint = tmp_path / "invalid.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=message):
        load_baseline_checkpoint(checkpoint, model_config(), torch.device("cpu"))


def test_prediction_result_values_are_json_compatible() -> None:
    result = PredictionResult(0.5, 0.0, 1, "1.0", 0.25)

    assert all(
        isinstance(value, (float, int, str)) and not isinstance(value, complex)
        for value in result.to_dict().values()
    )
    assert math.isfinite(result.inference_time)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((float("nan"), 0.0, 0, "1.0", 1.0), "probability"),
        ((0.5, 1.1, 0, "1.0", 1.0), "confidence"),
        ((0.5, 0.0, 2, "1.0", 1.0), "predicted_class"),
        ((0.5, 0.0, True, "1.0", 1.0), "predicted_class"),
        ((0.5, 0.0, 0, " ", 1.0), "model_version"),
        ((0.5, 0.0, 0, "1.0", -1.0), "inference_time"),
    ],
)
def test_prediction_result_rejects_invalid_contract_values(
    values: tuple[float, float, int, str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PredictionResult(*values)
