"""Validated public schemas for the inference service."""

from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from transitlens_ml_core.datasets import ProcessedLightCurve


class ServiceSchema(BaseModel):
    """Strict immutable base for service request and response contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ProcessedLightCurveRequest(ServiceSchema):
    """Public processed-light-curve request contract."""

    time: list[float] = Field(min_length=1)
    normalized_flux: list[float] = Field(min_length=1)
    wavelet_flux: list[float] = Field(min_length=1)
    metadata: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_cadences(self) -> Self:
        """Validate aligned arrays and strictly increasing observation time.

        Returns:
            The validated request.

        Raises:
            ValueError: If cadence arrays are misaligned or time is not increasing.

        """
        lengths = {len(self.time), len(self.normalized_flux), len(self.wavelet_flux)}
        if len(lengths) != 1:
            raise ValueError("processed light-curve arrays must have equal lengths")
        if any(
            current <= previous
            for previous, current in zip(self.time, self.time[1:], strict=False)
        ):
            raise ValueError("time values must be strictly increasing")
        return self

    def to_domain(self) -> ProcessedLightCurve:
        """Convert the API request into the existing inference-domain record.

        Returns:
            Validated immutable processed light curve.

        """
        return ProcessedLightCurve(
            time=np.asarray(self.time, dtype=np.float64),
            normalized_flux=np.asarray(self.normalized_flux, dtype=np.float64),
            wavelet_flux=np.asarray(self.wavelet_flux, dtype=np.float64),
            metadata=self.metadata,
        )


class HealthResponse(ServiceSchema):
    """Service and model readiness response."""

    status: str
    model_loaded: bool
    model_version: str | None


class ModelResponse(ServiceSchema):
    """Loaded-model identity and supported request contract."""

    model_version: str
    architecture: str
    training_timestamp: str | None
    supported_input_schema: dict[str, str]


class PredictionResponse(ServiceSchema):
    """Public scientific prediction response."""

    prediction: int = Field(ge=0, le=1)
    probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    transit_depth: float | None
    transit_duration: float | None
    estimated_period: float | None
    signal_to_noise_ratio: float | None
    model_version: str = Field(min_length=1)
    inference_time: float = Field(ge=0.0)


def metadata_number(metadata: dict[str, JsonValue], *names: str) -> float | None:
    """Extract the first finite numeric scientific descriptor from metadata.

    Args:
        metadata: Processed-light-curve metadata mapping.
        *names: Accepted descriptor names in priority order.

    Returns:
        First finite non-boolean number found, otherwise ``None``.

    """
    containers: list[dict[str, Any]] = [metadata]
    for container_name in ("statistics", "features", "metadata"):
        value = metadata.get(container_name)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        for name in names:
            value = container.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                number = float(value)
                if np.isfinite(number):
                    return number
    return None
