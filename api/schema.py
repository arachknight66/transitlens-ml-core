"""
api/schema.py
-------------
Pydantic v2 request and response models for the TransitLens ML Core API.

Models:
    AnalyzeRequest  — validates incoming light-curve data
    FeaturesSchema  — the 11-feature sub-dict
    PlotsSchema     — the 4-plot sub-dict
    AnalyzeResponse — mirrors the full result dict from analyze_light_curve()
    HealthResponse  — GET /health response

Used by: api/routes.py
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """
    Input payload for POST /analyze.

    Validates that time and flux are non-empty lists of equal length with
    at least 100 data points.
    """
    time: list[float] = Field(
        ..., min_length=500, description="BTJD timestamps, monotonically increasing"
    )
    flux: list[float] = Field(
        ..., min_length=500, description="Normalised flux values (median ≈ 1.0)"
    )
    target_id: str = Field(
        default="unknown", description="Identifier for this target"
    )
    metadata: Optional[dict] = Field(
        default=None, description="Optional metadata from data-pipeline"
    )
    config: Optional[dict] = Field(
        default=None, description="Optional pipeline config overrides"
    )

    @model_validator(mode="after")
    def check_equal_length(self) -> "AnalyzeRequest":
        if len(self.time) != len(self.flux):
            raise ValueError(
                f"time and flux must have equal length, got "
                f"len(time)={len(self.time)}, len(flux)={len(self.flux)}"
            )
        return self


# ---------------------------------------------------------------------------
# Response sub-models
# ---------------------------------------------------------------------------

class FeaturesSchema(BaseModel):
    """The 11 physically-interpretable features."""
    bls_power: float = 0.0
    snr: float = 0.0
    period_days: float = 0.0
    duration_days: float = 0.0
    depth: float = 0.0
    transit_count: float = 0.0
    odd_even_depth_delta: float = 0.0
    v_shape_score: float = 0.0
    local_noise: float = 0.0
    depth_to_noise_ratio: float = 0.0
    phase_shape_kurtosis: float = 0.0


class PlotsSchema(BaseModel):
    """The 4 diagnostic plots as base64-encoded PNG strings."""
    raw_lightcurve: str = ""
    cleaned_lightcurve: str = ""
    periodogram: str = ""
    phase_folded: str = ""


# ---------------------------------------------------------------------------
# Full response model
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    """
    Complete analysis result from analyze_light_curve().

    Mirrors the interface contract in the build plan §3.
    """
    target_id: str = "unknown"
    candidate_detected: bool = False
    predicted_class: str = "noise_or_other"
    confidence: float = 0.0

    period_days: Optional[float] = None
    duration_days: Optional[float] = None
    depth: Optional[float] = None
    snr: Optional[float] = None
    transit_count: Optional[int] = None

    features: FeaturesSchema = Field(default_factory=FeaturesSchema)
    explanation: str = ""
    plots: PlotsSchema = Field(default_factory=PlotsSchema)

    processing_time_ms: float = 0.0
    pipeline_version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """GET /health response."""
    status: str = "ok"
    version: str = "0.1.0"
    timestamp: str = ""
