"""Inference-only model evaluation and report generation."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from transitlens_ml_core.config import EvaluationConfig
from transitlens_ml_core.evaluation.metrics import (
    EvaluationMetrics,
    calculate_binary_metrics,
)

_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Metrics and artifact information from one model evaluation."""

    metrics: EvaluationMetrics
    sample_count: int
    classification_threshold: float
    report_path: Path


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader[Any],
    device: torch.device,
    config: EvaluationConfig,
) -> EvaluationResult:
    """Evaluate a probability model and atomically generate its report.

    Args:
        model: Binary model returning one probability per sample.
        data_loader: Labeled batches used for evaluation.
        device: Device on which model evaluation runs.
        config: Validated threshold and report configuration.

    Returns:
        Computed metrics and the generated report path.

    Raises:
        ValueError: If no samples are produced or model outputs are misaligned.

    """
    was_training = model.training
    model.to(device)
    model.eval()
    targets: list[float] = []
    probabilities: list[float] = []
    try:
        with torch.inference_mode():
            for features, batch_targets in data_loader:
                batch_probabilities = model(features.to(device)).reshape(-1)
                flattened_targets = batch_targets.to(device).reshape(-1)
                if batch_probabilities.numel() != flattened_targets.numel():
                    raise ValueError(
                        "model must return exactly one probability per target"
                    )
                probabilities.extend(batch_probabilities.cpu().tolist())
                targets.extend(flattened_targets.cpu().tolist())
    finally:
        model.train(was_training)

    if not targets:
        raise ValueError("evaluation loader produced no samples")
    metrics = calculate_binary_metrics(
        targets,
        probabilities,
        config.classification_threshold,
    )
    report_path = config.report_directory / config.report_filename
    result = EvaluationResult(
        metrics=metrics,
        sample_count=len(targets),
        classification_threshold=config.classification_threshold,
        report_path=report_path,
    )
    _write_report(result)
    return result


def _write_report(result: EvaluationResult) -> None:
    """Atomically write a versioned JSON evaluation report."""
    result.report_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "sample_count": result.sample_count,
        "classification_threshold": result.classification_threshold,
        "metrics": asdict(result.metrics),
    }
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=result.report_path.parent,
        prefix=f".{result.report_path.name}-",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write(content)
    try:
        os.replace(temporary_path, result.report_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
