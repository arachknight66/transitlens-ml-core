"""
core/config_schema.py
---------------------
Pydantic v2 validation schema for TransitLens configuration.
Rejects extra fields and validates data types and ranges.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError

class PreprocessingSchema(BaseModel):
    model_config = {"extra": "forbid"}
    
    sigma_upper: float = Field(default=5.0, ge=0.0)
    sigma_lower: float = Field(default=50.0, ge=0.0)
    max_sigma_iter: int = Field(default=3, ge=1)
    detrend_method: str = Field(default="running_median")
    detrend_window_days: float = Field(default=1.5, gt=0.0)
    detrend_poly_degree: int = Field(default=2, ge=1)
    gap_threshold_factor: float = Field(default=5.0, gt=0.0)
    min_points: int = Field(default=500, ge=10)
    min_time_span_days: float = Field(default=5.0, gt=0.0)
    min_fraction_retained: float = Field(default=0.80, ge=0.0, le=1.0)

class BLSSchema(BaseModel):
    model_config = {"extra": "forbid"}
    
    period_min_days: float = Field(default=0.5, gt=0.0)
    period_max_days: Optional[float] = Field(default=None)
    n_oversample: int = Field(default=10, ge=1)
    n_durations: int = Field(default=5, ge=1)
    duration_min_days: float = Field(default=0.01, gt=0.0)
    duration_max_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    bls_power_threshold: float = Field(default=0.15, ge=0.0)
    snr_threshold: float = Field(default=5.0, ge=0.0)
    alias_check_tolerance: float = Field(default=0.20, ge=0.0)

class FeaturesSchema(BaseModel):
    model_config = {"extra": "forbid"}
    
    phase_bins: int = Field(default=100, ge=10)
    odd_even_min_transits: int = Field(default=4, ge=1)
    noise_exclusion_factor: float = Field(default=1.5, gt=0.0)

class ClassificationSchema(BaseModel):
    model_config = {"extra": "forbid"}
    
    depth_threshold_eb: float = Field(default=0.050, ge=0.0)
    odd_even_threshold: float = Field(default=0.020, ge=0.0)
    v_shape_threshold: float = Field(default=0.40, ge=0.0)
    depth_snr_threshold: float = Field(default=6.0, ge=0.0)

class MLClassifierSchema(BaseModel):
    model_config = {"extra": "forbid"}
    
    enabled: bool = Field(default=True)
    model_type: str = Field(default="rf")
    dev_fallback: bool = Field(default=False)
    blend_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    use_rule_fallback_on_disagreement: bool = Field(default=True)

class PlottingSchema(BaseModel):
    model_config = {"extra": "forbid"}
    
    dpi: int = Field(default=100, ge=10)
    figure_width: int = Field(default=10, ge=1)
    figure_height: int = Field(default=4, ge=1)
    downsample_points: int = Field(default=2000, ge=100)
    phase_bins: int = Field(default=100, ge=10)
    transit_shade_alpha: float = Field(default=0.15, ge=0.0, le=1.0)
    style: str = Field(default="seaborn-v0_8-whitegrid")

class FittingSchema(BaseModel):
    model_config = {"extra": "forbid"}
    
    fitting_level: str = Field(default="quick")
    random_seed: int = Field(default=42)

class APISchema(BaseModel):
    model_config = {"extra": "forbid"}
    
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: List[str] = Field(default=["*"])
    log_level: str = Field(default="info")

class GlobalConfigSchema(BaseModel):
    model_config = {"extra": "forbid"}
    
    version: str = Field(default="0.1.0")
    preprocessing: PreprocessingSchema = Field(default_factory=PreprocessingSchema)
    bls: BLSSchema = Field(default_factory=BLSSchema)
    features: FeaturesSchema = Field(default_factory=FeaturesSchema)
    classification: ClassificationSchema = Field(default_factory=ClassificationSchema)
    ml_classifier: MLClassifierSchema = Field(default_factory=MLClassifierSchema)
    plotting: PlottingSchema = Field(default_factory=PlottingSchema)
    fitting: FittingSchema = Field(default_factory=FittingSchema)
    api: APISchema = Field(default_factory=APISchema)

def validate_config(config_dict: dict) -> GlobalConfigSchema:
    """
    Validates the configuration dictionary against GlobalConfigSchema.
    Raises ValidationError if invalid.
    """
    return GlobalConfigSchema.model_validate(config_dict)
