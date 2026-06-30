"""Checkpoint loading and deterministic single-light-curve inference."""

import math
import pickle
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from transitlens_ml_core.config import AppConfig, ModelConfig
from transitlens_ml_core.datasets.loader import ModelInputField, ProcessedLightCurve
from transitlens_ml_core.inference.confidence import estimate_confidence
from transitlens_ml_core.models import BaselineCNN


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Stable platform-facing prediction output.

    Attributes:
        probability: Predicted exoplanet-transit probability.
        confidence: Normalized distance from the classification threshold.
        predicted_class: Binary class selected using the configured threshold.
        model_version: Version identifier supplied by project configuration.
        inference_time: Model execution time in milliseconds, excluding loading.

    """

    probability: float
    confidence: float
    predicted_class: int
    model_version: str
    inference_time: float

    def __post_init__(self) -> None:
        """Validate every field at the public output boundary."""
        if (
            not math.isfinite(self.probability)
            or self.probability < 0.0
            or self.probability > 1.0
        ):
            raise ValueError("prediction probability must be finite and in [0, 1]")
        if (
            not math.isfinite(self.confidence)
            or self.confidence < 0.0
            or self.confidence > 1.0
        ):
            raise ValueError("prediction confidence must be finite and in [0, 1]")
        if isinstance(self.predicted_class, bool) or self.predicted_class not in (0, 1):
            raise ValueError("predicted_class must be binary")
        if not self.model_version.strip():
            raise ValueError("model_version must not be empty")
        if not math.isfinite(self.inference_time) or self.inference_time < 0.0:
            raise ValueError("inference_time must be finite and non-negative")

    def to_dict(self) -> dict[str, float | int | str]:
        """Serialize the stable output contract.

        Returns:
            Dictionary containing exactly the five public output fields.

        """
        return asdict(self)


def load_baseline_checkpoint(
    checkpoint_path: str | Path,
    model_config: ModelConfig,
    device: torch.device,
) -> BaselineCNN:
    """Load a baseline model from a trusted local PyTorch checkpoint.

    Args:
        checkpoint_path: Phase 4 training checkpoint containing model state.
        model_config: Architecture configuration used to create the checkpoint.
        device: Device receiving the model and checkpoint tensors.

    Returns:
        Loaded baseline CNN in evaluation mode.

    Raises:
        FileNotFoundError: If the checkpoint path does not exist.
        ValueError: If the checkpoint cannot be read or has incompatible state.

    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError) as error:
        raise ValueError("unable to read model checkpoint") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("model checkpoint root must be a mapping")
    if "model_state_dict" not in checkpoint:
        raise ValueError("model checkpoint is missing model_state_dict")

    model = BaselineCNN.from_config(model_config)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (TypeError, RuntimeError) as error:
        raise ValueError("model checkpoint state is incompatible") from error
    return model.to(device).eval()


class Predictor:
    """Deterministic single-record predictor for processed light curves."""

    def __init__(
        self,
        model: BaselineCNN,
        input_field: ModelInputField,
        classification_threshold: float,
        model_version: str,
        device: torch.device,
    ) -> None:
        """Initialize a ready-to-run predictor.

        Args:
            model: Loaded baseline CNN.
            input_field: Processed flux representation consumed by the model.
            classification_threshold: Inclusive positive-class threshold.
            model_version: Non-empty model version identifier.
            device: Device used for model execution.

        Raises:
            ValueError: If configuration is invalid or model expects multiple channels.

        """
        if input_field not in ("normalized_flux", "wavelet_flux"):
            raise ValueError("input_field must be normalized_flux or wavelet_flux")
        if (
            not math.isfinite(classification_threshold)
            or classification_threshold <= 0.0
            or classification_threshold >= 1.0
        ):
            raise ValueError("classification threshold must be between zero and one")
        if not model_version.strip():
            raise ValueError("model_version must not be empty")
        if model.input_channels != 1:
            raise ValueError(
                "processed light-curve inference requires one input channel"
            )

        self.model = model.to(device).eval()
        self.input_field = input_field
        self.classification_threshold = classification_threshold
        self.model_version = model_version
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_config: ModelConfig,
        input_field: ModelInputField,
        classification_threshold: float,
        model_version: str,
        device: torch.device,
    ) -> "Predictor":
        """Construct a predictor by loading a PyTorch checkpoint.

        Args:
            checkpoint_path: Phase 4 training checkpoint.
            model_config: Architecture configuration for the saved model.
            input_field: Processed flux representation consumed by the model.
            classification_threshold: Inclusive positive-class threshold.
            model_version: Non-empty model version identifier.
            device: Device used for loading and execution.

        Returns:
            Loaded deterministic predictor.

        """
        model = load_baseline_checkpoint(checkpoint_path, model_config, device)
        return cls(
            model,
            input_field,
            classification_threshold,
            model_version,
            device,
        )

    @classmethod
    def from_config(
        cls,
        checkpoint_path: str | Path,
        config: AppConfig,
        device: torch.device,
    ) -> "Predictor":
        """Construct a predictor from the complete application configuration.

        Args:
            checkpoint_path: Phase 4 training checkpoint.
            config: Validated application configuration.
            device: Device used for loading and execution.

        Returns:
            Loaded deterministic predictor.

        """
        return cls.from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_config=config.model,
            input_field=config.data.model_input_field,
            classification_threshold=config.evaluation.classification_threshold,
            model_version=config.project.version,
            device=device,
        )

    @classmethod
    def from_exported_checkpoint(
        cls,
        artifact_path: str | Path,
        device: torch.device,
    ) -> "Predictor":
        """Construct a predictor from a self-describing Phase 7 artifact.

        Args:
            artifact_path: Exported TransitLens PyTorch inference artifact.
            device: Device used for loading and execution.

        Returns:
            Predictor configured entirely from embedded artifact metadata.

        """
        from transitlens_ml_core.export.checkpoint import load_pytorch_export

        exported = load_pytorch_export(artifact_path, device)
        return cls(
            exported.model,
            exported.input_field,
            exported.classification_threshold,
            exported.model_version,
            device,
        )

    def predict(self, light_curve: ProcessedLightCurve) -> PredictionResult:
        """Predict one processed light curve and measure model execution time.

        Args:
            light_curve: Validated processed input record.

        Returns:
            Stable probability, confidence, class, version, and timing output.

        Raises:
            ValueError: If the model does not return one valid probability.

        """
        flux = getattr(light_curve, self.input_field)
        inputs = (
            torch.from_numpy(flux.copy())
            .to(device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self._synchronize_device()
        start = time.perf_counter_ns()
        with torch.inference_mode():
            output = self.model(inputs)
        self._synchronize_device()
        elapsed_milliseconds = (time.perf_counter_ns() - start) / 1_000_000.0

        flattened = output.detach().cpu().reshape(-1)
        if flattened.numel() != 1:
            raise ValueError("model must return exactly one probability")
        probability = float(flattened.item())
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError("model probability must be finite and in the range [0, 1]")
        return PredictionResult(
            probability=probability,
            confidence=estimate_confidence(probability, self.classification_threshold),
            predicted_class=int(probability >= self.classification_threshold),
            model_version=self.model_version,
            inference_time=elapsed_milliseconds,
        )

    def _synchronize_device(self) -> None:
        """Synchronize accelerator work when timing CUDA inference."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
