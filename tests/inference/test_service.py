"""Integration tests for the public FastAPI inference service."""

from pathlib import Path
from types import SimpleNamespace

import torch
from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from torch import nn

from transitlens_ml_core.config import load_config
from transitlens_ml_core.export import export_pytorch_checkpoint
from transitlens_ml_core.inference.predictor import Predictor
from transitlens_ml_core.inference.service import create_app
from transitlens_ml_core.models import BaselineCNN


class FluxMeanModel(nn.Module):
    """Return mean input flux as one probability."""

    input_channels = 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Average channel and cadence dimensions."""
        return inputs.mean(dim=(1, 2), keepdim=True)


class FailingModel(FluxMeanModel):
    """Raise a runtime failure during prediction."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Raise a deterministic runtime error."""
        del inputs
        raise RuntimeError("device failure")


class InvalidResponsePredictor:
    """Return values that violate the public response schema."""

    model_version = "1.0"
    model = FluxMeanModel()

    def predict(self, payload: object) -> SimpleNamespace:
        """Return an invalid probability for response validation."""
        del payload
        return SimpleNamespace(
            predicted_class=1,
            probability=2.0,
            confidence=0.5,
            model_version="1.0",
            inference_time=1.0,
        )


def predictor(model: nn.Module | None = None) -> Predictor:
    """Create a deterministic injected predictor."""
    return Predictor(
        model or FluxMeanModel(),  # type: ignore[arg-type]
        input_field="wavelet_flux",
        classification_threshold=0.5,
        model_version="1.2.3",
        device=torch.device("cpu"),
    )


def payload() -> dict[str, object]:
    """Return a valid request with scientific metadata."""
    return {
        "time": [1.0, 2.0, 3.0, 4.0],
        "normalized_flux": [0.9, 0.9, 0.9, 0.9],
        "wavelet_flux": [0.75, 0.75, 0.75, 0.75],
        "metadata": {
            "transit_depth": 0.02,
            "transit_duration": 3.5,
            "estimated_period": 12.25,
            "statistics": {"signal_to_noise_ratio": 9.0},
        },
    }


def test_health_model_and_prediction_contracts() -> None:
    application = create_app(
        predictor=predictor(), training_timestamp="2026-06-30T10:00:00Z"
    )

    with TestClient(application) as client:
        health = client.get("/health")
        model = client.get("/model")
        prediction = client.post("/predict", json=payload())

    assert health.status_code == 200
    assert health.json() == {
        "status": "healthy",
        "model_loaded": True,
        "model_version": "1.2.3",
    }
    assert model.status_code == 200
    assert model.json()["model_version"] == "1.2.3"
    assert model.json()["architecture"] == "FluxMeanModel"
    assert model.json()["training_timestamp"] == "2026-06-30T10:00:00Z"
    assert set(model.json()["supported_input_schema"]) == {
        "time",
        "normalized_flux",
        "wavelet_flux",
        "metadata",
    }
    assert prediction.status_code == 200
    assert prediction.json()["prediction"] == 1
    assert prediction.json()["probability"] == 0.75
    assert prediction.json()["confidence"] == 0.5
    assert prediction.json()["transit_depth"] == 0.02
    assert prediction.json()["transit_duration"] == 3.5
    assert prediction.json()["estimated_period"] == 12.25
    assert prediction.json()["signal_to_noise_ratio"] == 9.0
    assert prediction.json()["model_version"] == "1.2.3"
    assert prediction.json()["inference_time"] >= 0.0


def test_prediction_is_deterministic_and_optional_descriptors_are_null() -> None:
    request = payload()
    request["metadata"] = {}

    with TestClient(create_app(predictor=predictor())) as client:
        first = client.post("/predict", json=request).json()
        second = client.post("/predict", json=request).json()

    assert first["prediction"] == second["prediction"]
    assert first["probability"] == second["probability"]
    assert first["confidence"] == second["confidence"]
    assert first["transit_depth"] is None
    assert first["transit_duration"] is None
    assert first["estimated_period"] is None
    assert first["signal_to_noise_ratio"] is None


def test_service_loads_exported_checkpoint(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config = load_config(repository_root / "configs" / "prototype.yaml")
    export_config = config.export.model_copy(
        update={"output_directory": tmp_path, "pytorch_filename": "service.pt"}
    )
    config = config.model_copy(update={"export": export_config})
    checkpoint = export_pytorch_checkpoint(
        BaselineCNN.from_config(config.model).eval(), config
    )

    with TestClient(create_app(model_path=checkpoint)) as client:
        health = client.get("/health")
        response = client.post("/predict", json=payload())

    assert health.json()["model_loaded"] is True
    assert health.json()["model_version"] == config.project.version
    assert response.status_code == 200
    assert response.json()["model_version"] == config.project.version


def test_degraded_service_remains_health_checkable(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("TRANSITLENS_MODEL_PATH", raising=False)

    with TestClient(create_app()) as client:
        health = client.get("/health")
        model = client.get("/model")
        prediction = client.post("/predict", json=payload())

    assert health.json() == {
        "status": "degraded",
        "model_loaded": False,
        "model_version": None,
    }
    assert model.status_code == 503
    assert "model path is not configured" in model.json()["detail"]
    assert prediction.status_code == 503


def test_invalid_model_path_reports_descriptive_degraded_error(tmp_path: Path) -> None:
    with TestClient(create_app(model_path=tmp_path / "missing.pt")) as client:
        response = client.get("/model")

    assert response.status_code == 503
    assert "missing.pt" in response.json()["detail"]


def test_request_validation_and_inference_errors_are_descriptive() -> None:
    invalid = payload()
    invalid["wavelet_flux"] = [0.5]

    with TestClient(create_app(predictor=predictor())) as client:
        validation_response = client.post("/predict", json=invalid)
    with TestClient(create_app(predictor=predictor(FailingModel()))) as client:
        inference_response = client.post("/predict", json=payload())

    assert validation_response.status_code == 422
    assert "equal lengths" in str(validation_response.json())
    assert inference_response.status_code == 500
    assert inference_response.json()["detail"] == "model inference failed"


def test_response_validation_rejects_invalid_predictor_output() -> None:
    application = create_app(predictor=InvalidResponsePredictor())  # type: ignore[arg-type]

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/predict", json=payload())

    assert response.status_code == 500


def test_openapi_documents_all_public_routes() -> None:
    with TestClient(create_app(predictor=predictor())) as client:
        document = client.get("/openapi.json").json()
        docs = client.get("/docs")

    assert {"/health", "/model", "/predict"}.issubset(document["paths"])
    assert document["paths"]["/predict"]["post"]["requestBody"]["required"]
    assert docs.status_code == 200
