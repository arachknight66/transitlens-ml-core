"""Composed export of all platform model formats."""

from dataclasses import dataclass
from pathlib import Path

from transitlens_ml_core.config import AppConfig
from transitlens_ml_core.export.checkpoint import export_pytorch_checkpoint
from transitlens_ml_core.export.onnx import export_onnx
from transitlens_ml_core.models import BaselineCNN


@dataclass(frozen=True, slots=True)
class ExportArtifacts:
    """Paths produced by a complete model export."""

    pytorch_checkpoint: Path
    onnx_model: Path


def export_models(model: BaselineCNN, config: AppConfig) -> ExportArtifacts:
    """Export PyTorch and ONNX platform artifacts.

    Args:
        model: Trained baseline CNN to export.
        config: Complete validated application and export configuration.

    Returns:
        Paths to both completed export artifacts.

    """
    return ExportArtifacts(
        pytorch_checkpoint=export_pytorch_checkpoint(model, config),
        onnx_model=export_onnx(model, config),
    )
