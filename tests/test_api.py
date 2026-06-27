"""
tests/test_api.py
-----------------
Tests for the FastAPI API (Phase 7).

Uses FastAPI's TestClient — no running server required.

Covers:
    - GET /health returns 200 with {status: ok}
    - POST /analyze with valid data returns 200 with complete result dict
    - POST /analyze with mismatched lengths returns 422
    - POST /analyze with too few points returns 422
    - GET /demo/a returns 200 with complete result
    - GET /demo/invalid returns 404
    - CORS headers are present
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
import numpy as np

from fastapi.testclient import TestClient
from api.app import app


client = TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealth:
    """Tests for GET /health."""

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_ok(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_has_version(self):
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert len(data["version"]) > 0

    def test_health_has_timestamp(self):
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    """Tests for POST /analyze."""

    @pytest.fixture
    def valid_payload(self):
        """Generate a minimal valid light curve payload."""
        rng = np.random.default_rng(42)
        n = 1000
        time = np.linspace(0.0, 20.0, n)
        flux = 1.0 + rng.normal(0, 0.001, n)
        return {
            "time": time.tolist(),
            "flux": flux.tolist(),
            "target_id": "test_target",
        }

    def test_analyze_returns_200(self, valid_payload):
        response = client.post("/analyze", json=valid_payload)
        assert response.status_code == 200

    def test_analyze_has_required_keys(self, valid_payload):
        response = client.post("/analyze", json=valid_payload)
        data = response.json()
        required = {
            "target_id", "candidate_detected", "predicted_class",
            "confidence", "features", "explanation", "plots",
            "processing_time_ms", "pipeline_version",
        }
        assert required.issubset(set(data.keys()))

    def test_analyze_mismatched_lengths_returns_422(self):
        payload = {
            "time": [1.0, 2.0, 3.0] * 50,  # 150 elements
            "flux": [1.0, 2.0] * 50,        # 100 elements
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422

    def test_analyze_too_few_points_returns_422(self):
        payload = {
            "time": [1.0, 2.0, 3.0],
            "flux": [1.0, 1.0, 1.0],
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /demo
# ---------------------------------------------------------------------------

class TestDemo:
    """Tests for GET /demo/{candidate_id}."""

    def test_demo_a_returns_200(self):
        response = client.get("/demo/a")
        assert response.status_code == 200

    def test_demo_a_has_complete_result(self):
        response = client.get("/demo/a")
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "features" in data

    def test_demo_invalid_returns_404(self):
        response = client.get("/demo/invalid")
        assert response.status_code == 404

    def test_demo_b_returns_200(self):
        response = client.get("/demo/b")
        assert response.status_code == 200

    def test_demo_c_returns_200(self):
        response = client.get("/demo/c")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

class TestCORS:
    """Tests that CORS headers are present."""

    def test_cors_header_on_health(self):
        response = client.get("/health", headers={"Origin": "http://localhost:8501"})
        # CORS middleware should add the header
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Timing header
# ---------------------------------------------------------------------------

class TestTimingHeader:
    """Test that X-Processing-Time-Ms header is present."""

    def test_timing_header_present(self):
        response = client.get("/health")
        assert "x-processing-time-ms" in response.headers


# ---------------------------------------------------------------------------
# POST /analyze/file and POST /analyze/tess
# ---------------------------------------------------------------------------

class TestAnalyzeFileAndTess:
    """Tests for POST /analyze/file and POST /analyze/tess."""

    def test_analyze_file_returns_200(self):
        fits_path = (
            Path(__file__).resolve().parents[2]
            / "transitlens-data-pipeline"
            / "real_tess"
            / "cache"
            / "TIC261136679_sector095.fits"
        )
        if not fits_path.exists():
            pytest.skip("cached TESS integration fixture is not available")
        with open(fits_path, "rb") as f:
            response = client.post(
                "/analyze/file",
                files={"file": ("TIC261136679_sector095.fits", f, "application/octet-stream")},
                data={"target_id": "TIC 261136679", "metadata": "{}"}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["target_id"] == "TIC 261136679"
        assert "predicted_class" in data

    def test_analyze_tess_returns_200(self, monkeypatch):
        mock_result = {
            "target_id": "TIC 261136679",
            "candidate_detected": True,
            "predicted_class": "exoplanet_transit",
            "confidence": 0.95,
            "period_days": 13.2316,
            "duration_days": 0.1120,
            "depth": 0.0002,
            "snr": 24.75,
            "transit_count": 2,
            "features": {
                "depth": 0.0002,
                "snr": 24.75,
                "bls_power": 1.0,
                "period": 13.2316,
                "duration": 0.1120,
                "odd_even_mismatch": 0.0,
                "v_shape_parameter": 0.0,
                "sde": 5.75,
                "secondary_depth": 0.0,
                "crowding_metric": 1.0,
                "centroid_offset_sigma": 0.0,
                "background_contamination_ratio": 0.0,
                "cdpp_noise": 0.0001,
                "rms_scatter": 0.0001,
                "depth_to_noise_ratio": 10.0,
                "phase_shape_kurtosis": 2.0
            },
            "explanation": "Mocked successful prediction.",
            "plots": {
                "raw_lightcurve": "",
                "cleaned_lightcurve": "",
                "periodogram": "",
                "phase_folded": ""
            },
            "processing_time_ms": 150.0,
            "pipeline_version": "0.1.0"
        }
        import pipeline
        monkeypatch.setattr(pipeline, "analyze_tess_multi_sector", lambda *args, **kwargs: mock_result)

        response = client.post(
            "/analyze/tess",
            data={"tic_id": "261136679"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["target_id"] == "TIC 261136679"
        assert "predicted_class" in data

    def test_analyze_tess_rejects_non_numeric_tic(self):
        response = client.post(
            "/analyze/tess",
            data={"tic_id": "TIC-not-a-number"}
        )
        assert response.status_code == 422

    def test_analyze_tess_rejects_invalid_config_json(self):
        response = client.post(
            "/analyze/tess",
            data={"tic_id": "261136679", "config": "not-json"}
        )
        assert response.status_code == 422
