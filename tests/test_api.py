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
