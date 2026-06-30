"""Self-describing PyTorch inference checkpoint export."""

import math
import os
import pickle
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from transitlens_ml_core.config import AppConfig, ModelConfig
from transitlens_ml_core.datasets.loader import ModelInputField
from transitlens_ml_core.models import BaselineCNN

PYTORCH_EXPORT_SCHEMA_VERSION = 1
PYTORCH_EXPORT_FORMAT = "transitlens-pytorch"
MODEL_INPUT_NAME = "light_curve"
MODEL_OUTPUT_NAME = "transit_probability"


@dataclass(frozen=True, slots=True)
class ExportedModel:
    """Loaded self-describing PyTorch inference artifact."""

    model: BaselineCNN
    input_field: ModelInputField
    classification_threshold: float
    model_version: str


def export_pytorch_checkpoint(model: BaselineCNN, config: AppConfig) -> Path:
    """Atomically export a self-describing PyTorch inference artifact.

    Args:
        model: Trained baseline CNN whose weights will be exported.
        config: Complete validated application and artifact configuration.

    Returns:
        Completed PyTorch artifact path.

    Raises:
        ValueError: If model state is incompatible with configured architecture.

    """
    _validate_model_compatibility(model, config.model)
    destination = config.export.output_directory / config.export.pytorch_filename
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    payload = {
        "schema_version": PYTORCH_EXPORT_SCHEMA_VERSION,
        "format": PYTORCH_EXPORT_FORMAT,
        "model_version": config.project.version,
        "model_config": config.model.model_dump(mode="json"),
        "input_field": config.data.model_input_field,
        "classification_threshold": config.evaluation.classification_threshold,
        "input_name": MODEL_INPUT_NAME,
        "output_name": MODEL_OUTPUT_NAME,
        "model_state_dict": state,
    }
    _atomic_torch_save(payload, destination)
    return destination


def load_pytorch_export(
    artifact_path: str | Path, device: torch.device
) -> ExportedModel:
    """Load and validate a self-describing PyTorch inference artifact.

    Args:
        artifact_path: Exported TransitLens ``.pt`` artifact.
        device: Device receiving model weights and inference execution.

    Returns:
        Loaded model and embedded inference metadata.

    Raises:
        FileNotFoundError: If the artifact does not exist.
        ValueError: If the artifact is unreadable, incomplete, or incompatible.

    """
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError) as error:
        raise ValueError("unable to read PyTorch export") from error
    if not isinstance(payload, Mapping):
        raise ValueError("PyTorch export root must be a mapping")
    required = {
        "schema_version",
        "format",
        "model_version",
        "model_config",
        "input_field",
        "classification_threshold",
        "input_name",
        "output_name",
        "model_state_dict",
    }
    if missing := sorted(required.difference(payload)):
        raise ValueError(f"PyTorch export is missing fields: {missing}")
    if payload["schema_version"] != PYTORCH_EXPORT_SCHEMA_VERSION:
        raise ValueError("unsupported PyTorch export schema version")
    if payload["format"] != PYTORCH_EXPORT_FORMAT:
        raise ValueError("unsupported PyTorch export format")
    if (
        payload["input_name"] != MODEL_INPUT_NAME
        or payload["output_name"] != MODEL_OUTPUT_NAME
    ):
        raise ValueError("PyTorch export tensor names are incompatible")

    try:
        model_config = ModelConfig.model_validate(payload["model_config"])
        input_field = _validate_input_field(payload["input_field"])
        threshold = float(payload["classification_threshold"])
        version = str(payload["model_version"])
        if not math.isfinite(threshold) or threshold <= 0.0 or threshold >= 1.0:
            raise ValueError("classification threshold is invalid")
        if not version.strip():
            raise ValueError("model version is empty")
        model = BaselineCNN.from_config(model_config)
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("PyTorch export contains invalid model state") from error
    return ExportedModel(
        model=model.to(device).eval(),
        input_field=input_field,
        classification_threshold=threshold,
        model_version=version,
    )


def _validate_input_field(value: object) -> ModelInputField:
    """Validate an embedded processed-input field name."""
    if value not in ("normalized_flux", "wavelet_flux"):
        raise ValueError("export input field is invalid")
    return cast("ModelInputField", value)


def _validate_model_compatibility(
    model: BaselineCNN, model_config: ModelConfig
) -> None:
    """Ensure model weights match the configured baseline architecture."""
    configured_model = BaselineCNN.from_config(model_config)
    try:
        configured_model.load_state_dict(model.state_dict(), strict=True)
    except RuntimeError as error:
        raise ValueError(
            "model state is incompatible with export configuration"
        ) from error


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    """Atomically serialize a restricted-values PyTorch artifact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}-",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
