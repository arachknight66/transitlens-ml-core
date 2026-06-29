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
    """The 16 physically-interpretable features."""
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
    bls_sde: float = 0.0
    secondary_eclipse_depth: float = 0.0
    centroid_shift: float = 0.0
    crowding_metric: float = 1.0
    gaia_neighbor_count: float = 0.0



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
    predicted_class: str = "stellar_variability_or_other"
    confidence: float = 0.0

    period_days: Optional[float] = None
    duration_days: Optional[float] = None
    depth: Optional[float] = None
    snr: Optional[float] = None
    transit_count: Optional[int] = None

    # Scientific uncertainties and significance
    bootstrap_fap: Optional[float] = None
    class_probabilities: Optional[dict[str, float]] = None
    class_probability_status: Optional[str] = None
    ml_inference_status: Optional[str] = None
    ml_predicted_class: Optional[str] = None
    ml_review_required: Optional[bool] = None
    ml_review_reasons: list[str] = Field(default_factory=list)
    ml_model_id: Optional[str] = None
    period_uncertainty_days: Optional[float] = None
    duration_uncertainty_days: Optional[float] = None
    depth_uncertainty: Optional[float] = None
    epoch_btjd: Optional[float] = None
    fit_quality: Optional[float] = None

    # New Phase 7 scientific parameters
    fit_status: Optional[str] = None
    quality_flags: Optional[list[str]] = None
    rp_rstar: Optional[float] = None
    rp_rstar_err_lower: Optional[float] = None
    rp_rstar_err_upper: Optional[float] = None
    a_rstar: Optional[float] = None
    a_rstar_err_lower: Optional[float] = None
    a_rstar_err_upper: Optional[float] = None
    b: Optional[float] = None
    b_err_lower: Optional[float] = None
    b_err_upper: Optional[float] = None
    u1: Optional[float] = None
    u2: Optional[float] = None
    baseline_offset: Optional[float] = None
    baseline_slope: Optional[float] = None
    jitter: Optional[float] = None
    chi2: Optional[float] = None
    reduced_chi2: Optional[float] = None
    bic: Optional[float] = None
    aic: Optional[float] = None
    residual_rms: Optional[float] = None
    durbin_watson: Optional[float] = None
    beta_factor: Optional[float] = None
    autocorr_lag1: Optional[float] = None
    mcmc_passed: Optional[bool] = None
    mcmc_rhat: Optional[float] = None
    mcmc_ess: Optional[int] = None
    observed_depth: Optional[float] = None
    observed_depth_uncertainty: Optional[float] = None
    corrected_depth: Optional[float] = None
    corrected_depth_uncertainty: Optional[float] = None
    planet_radius_earth: Optional[float] = None
    planet_radius_earth_err_lower: Optional[float] = None
    planet_radius_earth_err_upper: Optional[float] = None
    inferred_density: Optional[float] = None
    inclination_deg: Optional[float] = None
    observed_transits: Optional[int] = None
    in_transit_cadences: Optional[int] = None
    phase_coverage_fraction: Optional[float] = None
    alias_warning_fitter: Optional[bool] = None
    alias_type_fitter: Optional[str] = None
    alias_reason_fitter: Optional[str] = None
    odd_even_delta_fitter: Optional[float] = None
    secondary_depth_fitter: Optional[float] = None
    uncertainty_method: Optional[str] = None

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
