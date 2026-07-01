"""Tests for public inference service schemas."""

import pytest
from pydantic import ValidationError

from transitlens_ml_core.inference.schemas import (
    PredictionResponse,
    ProcessedLightCurveRequest,
    metadata_number,
)


def valid_request() -> dict[str, object]:
    """Return a valid service request mapping."""
    return {
        "time": [1.0, 2.0, 3.0],
        "normalized_flux": [1.0, 0.99, 1.0],
        "wavelet_flux": [1.0, 0.995, 1.0],
        "metadata": {"statistics": {"signal_to_noise_ratio": 8.5}},
    }


def test_request_converts_to_validated_domain_record() -> None:
    request = ProcessedLightCurveRequest.model_validate(valid_request())

    curve = request.to_domain()

    assert curve.time.tolist() == [1.0, 2.0, 3.0]
    assert curve.wavelet_flux.tolist() == [1.0, 0.995, 1.0]
    assert curve.metadata["statistics"] == {"signal_to_noise_ratio": 8.5}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("wavelet_flux", [1.0, 1.0], "equal lengths"),
        ("time", [1.0, 3.0, 2.0], "strictly increasing"),
        ("normalized_flux", [1.0, float("nan"), 1.0], "finite number"),
    ],
)
def test_request_rejects_invalid_cadences(
    field: str, value: object, message: str
) -> None:
    payload = valid_request()
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        ProcessedLightCurveRequest.model_validate(payload)


def test_request_rejects_unknown_fields() -> None:
    payload = valid_request()
    payload["flux"] = [1.0, 1.0, 1.0]

    with pytest.raises(ValidationError, match="Extra inputs"):
        ProcessedLightCurveRequest.model_validate(payload)


def test_metadata_number_searches_supported_containers_and_ignores_invalid() -> None:
    metadata = {
        "depth": True,
        "statistics": {"snr": 7, "duration": float("inf")},
        "features": {"transit_duration": 2.5},
    }

    assert metadata_number(metadata, "depth") is None
    assert metadata_number(metadata, "signal_to_noise_ratio", "snr") == 7.0
    assert metadata_number(metadata, "transit_duration", "duration") == 2.5
    assert metadata_number(metadata, "period") is None


def test_prediction_response_validates_public_ranges() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        PredictionResponse(
            prediction=2,
            probability=0.5,
            confidence=0.5,
            transit_depth=None,
            transit_duration=None,
            estimated_period=None,
            signal_to_noise_ratio=None,
            model_version="1.0",
            inference_time=1.0,
        )
