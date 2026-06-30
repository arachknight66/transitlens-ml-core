"""Tests for the composed baseline CNN."""

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from transitlens_ml_core.config import ModelConfig, load_config
from transitlens_ml_core.models import BaselineCNN, CNNFeatureExtractor


def model_config() -> ModelConfig:
    """Create a compact valid model configuration."""
    return ModelConfig(
        input_channels=1,
        convolution_channels=(4, 8),
        kernel_size=3,
        pool_size=2,
    )


def test_feature_extractor_exposes_modular_sequence_features() -> None:
    extractor = CNNFeatureExtractor(1, (4, 8), kernel_size=3, pool_size=2)

    features = extractor(torch.randn(2, 1, 32))

    assert features.shape == (2, 8, 16)


@pytest.mark.parametrize("sample_count", [16, 31, 128])
def test_baseline_forward_returns_one_probability_per_sample(
    sample_count: int,
) -> None:
    model = BaselineCNN.from_config(model_config())

    probabilities = model(torch.randn(4, 1, sample_count))

    assert probabilities.shape == (4, 1)
    assert torch.all(torch.isfinite(probabilities))
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_baseline_forward_supports_gradient_propagation() -> None:
    model = BaselineCNN.from_config(model_config())
    inputs = torch.randn(2, 1, 24, requires_grad=True)

    model(inputs).sum().backward()

    assert inputs.grad is not None
    assert torch.all(torch.isfinite(inputs.grad))
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_baseline_is_deterministic_in_evaluation_mode() -> None:
    model = BaselineCNN.from_config(model_config()).eval()
    inputs = torch.randn(2, 1, 24)

    with torch.inference_mode():
        first = model(inputs)
        second = model(inputs)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_checked_in_configuration_constructs_model() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config = load_config(repository_root / "configs" / "prototype.yaml")

    model = BaselineCNN.from_config(config.model)
    output = model(torch.ones(1, 1, 64))

    assert output.shape == (1, 1)


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (torch.ones(1, 16), "shape"),
        (torch.ones(1, 2, 16), "expected 1 input channels"),
        (torch.ones(1, 1, 1), "at least 2 samples"),
    ],
)
def test_baseline_rejects_invalid_inputs(inputs: torch.Tensor, message: str) -> None:
    model = BaselineCNN.from_config(model_config())

    with pytest.raises(ValueError, match=message):
        model(inputs)


def test_baseline_rejects_invalid_pool_size() -> None:
    with pytest.raises(ValueError, match="pool_size"):
        BaselineCNN(1, (4, 8), kernel_size=3, pool_size=0)


def test_model_configuration_rejects_even_kernel() -> None:
    with pytest.raises(ValidationError, match="must be odd"):
        ModelConfig(
            input_channels=1,
            convolution_channels=(4, 8),
            kernel_size=4,
            pool_size=2,
        )
