"""Tests for model evaluation and report generation."""

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from transitlens_ml_core.config import EvaluationConfig
from transitlens_ml_core.evaluation import evaluate_model


class IdentityProbabilityModel(nn.Module):
    """Return the single input feature as a probability."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Flatten one probability per sample."""
        return inputs.flatten(start_dim=1)


class MisalignedModel(nn.Module):
    """Deliberately return one fewer prediction than targets."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Drop the final batch prediction."""
        return inputs[:-1].flatten(start_dim=1)


def evaluation_config(root: Path) -> EvaluationConfig:
    """Create an isolated evaluation configuration."""
    return EvaluationConfig(
        classification_threshold=0.5,
        report_directory=root,
        report_filename="evaluation.json",
    )


def test_evaluate_model_computes_metrics_and_writes_report(tmp_path: Path) -> None:
    features = torch.tensor([0.1, 0.6, 0.4, 0.9]).reshape(-1, 1, 1)
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
    loader = DataLoader(TensorDataset(features, targets), batch_size=3)
    model = IdentityProbabilityModel()
    model.train()

    result = evaluate_model(
        model,
        loader,
        torch.device("cpu"),
        evaluation_config(tmp_path / "nested"),
    )

    assert model.training
    assert result.sample_count == 4
    assert result.metrics.accuracy == 0.5
    assert result.metrics.roc_auc == 0.75
    assert result.report_path.is_file()
    document = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["sample_count"] == 4
    assert document["classification_threshold"] == 0.5
    assert document["metrics"]["confusion_matrix"] == [[1, 1], [1, 1]]


def test_evaluate_model_preserves_evaluation_mode(tmp_path: Path) -> None:
    model = IdentityProbabilityModel().eval()
    loader = DataLoader(
        TensorDataset(torch.tensor([0.2, 0.8]).reshape(-1, 1, 1), torch.tensor([0, 1])),
        batch_size=2,
    )

    evaluate_model(model, loader, torch.device("cpu"), evaluation_config(tmp_path))

    assert not model.training


def test_evaluate_model_restores_mode_after_model_error(tmp_path: Path) -> None:
    model = MisalignedModel().train()
    loader = DataLoader(
        TensorDataset(torch.tensor([0.2, 0.8]).reshape(-1, 1, 1), torch.tensor([0, 1])),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="one probability per target"):
        evaluate_model(model, loader, torch.device("cpu"), evaluation_config(tmp_path))

    assert model.training


def test_evaluate_model_rejects_empty_loader(tmp_path: Path) -> None:
    loader = DataLoader(
        TensorDataset(torch.empty(0, 1, 1), torch.empty(0)), batch_size=2
    )

    with pytest.raises(ValueError, match="no samples"):
        evaluate_model(
            IdentityProbabilityModel(),
            loader,
            torch.device("cpu"),
            evaluation_config(tmp_path),
        )


@pytest.mark.parametrize("filename", ["report.txt", "nested/report.json"])
def test_evaluation_config_rejects_unsafe_report_filename(
    tmp_path: Path, filename: str
) -> None:
    with pytest.raises(ValidationError, match="JSON report|json basename"):
        EvaluationConfig(
            classification_threshold=0.5,
            report_directory=tmp_path,
            report_filename=filename,
        )
