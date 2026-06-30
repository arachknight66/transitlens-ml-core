"""Shared model export test helpers."""

from pathlib import Path

import numpy as np

from transitlens_ml_core.config import AppConfig, load_config
from transitlens_ml_core.datasets import ProcessedLightCurve
from transitlens_ml_core.models import BaselineCNN


def export_config(root: Path) -> AppConfig:
    """Load prototype configuration with isolated export paths and small shapes."""
    repository_root = Path(__file__).resolve().parents[2]
    config = load_config(repository_root / "configs" / "prototype.yaml")
    export = config.export.model_copy(
        update={
            "output_directory": root,
            "pytorch_filename": "model.pt",
            "onnx_filename": "model.onnx",
            "sample_length": 32,
        }
    )
    return config.model_copy(update={"export": export})


def configured_model(config: AppConfig) -> BaselineCNN:
    """Create a deterministic configured baseline in evaluation mode."""
    model = BaselineCNN.from_config(config.model).eval()
    return model


def sample_curve(length: int = 32) -> ProcessedLightCurve:
    """Create one valid predictor input record."""
    return ProcessedLightCurve(
        time=np.arange(length, dtype=np.float64),
        normalized_flux=np.ones(length, dtype=np.float64),
        wavelet_flux=np.linspace(0.95, 1.05, length, dtype=np.float64),
        metadata={},
    )
