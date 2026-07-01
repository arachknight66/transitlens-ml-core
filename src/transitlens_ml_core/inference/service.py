"""FastAPI adapter for exported TransitLens model inference."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request, status

from transitlens_ml_core.inference.predictor import Predictor
from transitlens_ml_core.inference.schemas import (
    HealthResponse,
    ModelResponse,
    PredictionResponse,
    ProcessedLightCurveRequest,
    metadata_number,
)

_MODEL_PATH_ENVIRONMENT_VARIABLE = "TRANSITLENS_MODEL_PATH"
_SUPPORTED_INPUT_SCHEMA = {
    "time": "array[finite float], strictly increasing",
    "normalized_flux": "array[finite float], cadence-aligned",
    "wavelet_flux": "array[finite float], cadence-aligned",
    "metadata": "JSON object",
}


def create_app(
    model_path: str | Path | None = None,
    *,
    device: torch.device | None = None,
    predictor: Predictor | None = None,
    training_timestamp: str | None = None,
) -> FastAPI:
    """Create the public inference application.

    Args:
        model_path: Optional self-describing exported PyTorch checkpoint path.
        device: Optional inference device; defaults to CPU.
        predictor: Optional injected predictor, primarily for composed deployment.
        training_timestamp: Optional ISO-8601 training provenance from deployment.

    Returns:
        Configured FastAPI application with OpenAPI documentation.

    """
    inference_device = device or torch.device("cpu")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Load the exported model while preserving degraded health reporting."""
        application.state.predictor = predictor
        application.state.model_error = None
        application.state.training_timestamp = training_timestamp
        configured_path = model_path or os.getenv(_MODEL_PATH_ENVIRONMENT_VARIABLE)
        if application.state.predictor is None:
            if configured_path is None:
                application.state.model_error = (
                    "model path is not configured; set "
                    f"{_MODEL_PATH_ENVIRONMENT_VARIABLE}"
                )
            else:
                try:
                    application.state.predictor = Predictor.from_exported_checkpoint(
                        configured_path, inference_device
                    )
                except (FileNotFoundError, ValueError) as error:
                    application.state.model_error = str(error)
        yield

    application = FastAPI(
        title="TransitLens ML Inference Service",
        description="Deterministic inference over processed exoplanet light curves.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Return service and model readiness."""
        loaded_predictor = _optional_predictor(request)
        return HealthResponse(
            status="healthy" if loaded_predictor is not None else "degraded",
            model_loaded=loaded_predictor is not None,
            model_version=(
                loaded_predictor.model_version if loaded_predictor is not None else None
            ),
        )

    @application.get("/model", response_model=ModelResponse)
    def model_information(request: Request) -> ModelResponse:
        """Return loaded-model identity and supported input schema."""
        loaded_predictor = _required_predictor(request)
        return ModelResponse(
            model_version=loaded_predictor.model_version,
            architecture=loaded_predictor.model.__class__.__name__,
            training_timestamp=request.app.state.training_timestamp,
            supported_input_schema=dict(_SUPPORTED_INPUT_SCHEMA),
        )

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: ProcessedLightCurveRequest, request: Request
    ) -> PredictionResponse:
        """Validate one processed light curve and return its transit prediction."""
        loaded_predictor = _required_predictor(request)
        try:
            result = loaded_predictor.predict(payload.to_domain())
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"invalid inference input: {error}",
            ) from error
        except RuntimeError as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="model inference failed",
            ) from error
        return PredictionResponse(
            prediction=result.predicted_class,
            probability=result.probability,
            confidence=result.confidence,
            transit_depth=metadata_number(payload.metadata, "transit_depth", "depth"),
            transit_duration=metadata_number(
                payload.metadata, "transit_duration", "duration"
            ),
            estimated_period=metadata_number(
                payload.metadata, "estimated_period", "period"
            ),
            signal_to_noise_ratio=metadata_number(
                payload.metadata, "signal_to_noise_ratio", "snr"
            ),
            model_version=result.model_version,
            inference_time=result.inference_time,
        )

    return application


def _optional_predictor(request: Request) -> Predictor | None:
    """Return the loaded predictor when available."""
    return request.app.state.predictor


def _required_predictor(request: Request) -> Predictor:
    """Return the predictor or raise a descriptive readiness error."""
    predictor = _optional_predictor(request)
    if predictor is None:
        detail = request.app.state.model_error or "model is not loaded"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"model unavailable: {detail}",
        )
    return predictor


app = create_app()
