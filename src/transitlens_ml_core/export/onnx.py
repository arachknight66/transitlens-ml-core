"""Dynamic-shape ONNX export for the baseline CNN."""

import copy
import json
import os
import tempfile
import warnings
from pathlib import Path

import onnx
import torch

from transitlens_ml_core.config import AppConfig
from transitlens_ml_core.export.checkpoint import (
    MODEL_INPUT_NAME,
    MODEL_OUTPUT_NAME,
    _validate_model_compatibility,
)
from transitlens_ml_core.models import BaselineCNN

ONNX_EXPORT_SCHEMA_VERSION = 1


def export_onnx(model: BaselineCNN, config: AppConfig) -> Path:
    """Atomically export and validate a dynamic-length ONNX model.

    Args:
        model: Trained baseline CNN whose weights will be exported.
        config: Complete validated application and artifact configuration.

    Returns:
        Completed, checker-validated ONNX artifact path.

    Raises:
        ValueError: If model state is incompatible with configured architecture.
        RuntimeError: If ONNX conversion or validation fails.

    """
    _validate_model_compatibility(model, config.model)
    destination = config.export.output_directory / config.export.onnx_filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_model = copy.deepcopy(model).cpu().eval()
    example = torch.zeros(
        1,
        config.model.input_channels,
        config.export.sample_length,
        dtype=torch.float32,
    )
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}-",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using the legacy TorchScript-based ONNX export",
            )
            warnings.filterwarnings("ignore", message="The feature will be removed.*")
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            torch.onnx.export(
                export_model,
                example,
                temporary_path,
                export_params=True,
                opset_version=config.export.onnx_opset_version,
                do_constant_folding=True,
                input_names=[MODEL_INPUT_NAME],
                output_names=[MODEL_OUTPUT_NAME],
                dynamic_axes={
                    MODEL_INPUT_NAME: {0: "batch", 2: "samples"},
                    MODEL_OUTPUT_NAME: {0: "batch"},
                },
                dynamo=False,
            )
        onnx_model = onnx.load(temporary_path)
        onnx.checker.check_model(onnx_model)
        onnx.helper.set_model_props(onnx_model, _metadata(config))
        onnx.save_model(onnx_model, temporary_path)
        onnx.checker.check_model(onnx.load(temporary_path))
        os.replace(temporary_path, destination)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("failed to export validated ONNX model") from error
    return destination


def _metadata(config: AppConfig) -> dict[str, str]:
    """Build platform-consumable ONNX model metadata."""
    return {
        "transitlens.schema_version": str(ONNX_EXPORT_SCHEMA_VERSION),
        "transitlens.model_version": config.project.version,
        "transitlens.model_config": json.dumps(
            config.model.model_dump(mode="json"), sort_keys=True
        ),
        "transitlens.input_field": config.data.model_input_field,
        "transitlens.classification_threshold": str(
            config.evaluation.classification_threshold
        ),
        "transitlens.input_name": MODEL_INPUT_NAME,
        "transitlens.output_name": MODEL_OUTPUT_NAME,
    }
