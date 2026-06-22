"""
core/exceptions.py
------------------
Custom exception hierarchy for transitlens-ml-core.

All exceptions inherit from MLCoreError, allowing callers to catch all
ml-core errors with a single except clause.
"""


class MLCoreError(Exception):
    """Base exception for all transitlens-ml-core errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{detail_str}]"
        return self.message


class InvalidInputError(MLCoreError):
    """
    Raised when input arrays are fundamentally broken.

    Examples:
        - time and flux have different lengths
        - time is not monotonically increasing
        - flux contains infinite values
        - arrays are empty

    This is the only exception that propagates to the caller from pipeline.py.
    """
    pass


class InsufficientDataError(MLCoreError):
    """
    Raised when there is not enough data to run a valid analysis.

    Examples:
        - fewer than 500 data points after NaN removal
        - time span less than 5 days
        - more than 20% of points removed by sigma clipping
    """
    pass


class NoCandidateFoundError(MLCoreError):
    """
    Raised when BLS finds no significant periodic signal.

    This is a valid, expected outcome for noise_or_other light curves.
    The pipeline catches this internally and sets candidate_detected=False
    rather than propagating it to the caller.
    """
    pass


class PreprocessingError(MLCoreError):
    """
    Raised when a preprocessing step fails unexpectedly.

    This covers unexpected failures in normalisation, detrending, or
    gap detection that are not caused by bad input data.
    """
    pass


class BLSDetectionError(MLCoreError):
    """
    Raised when the BLS algorithm fails due to a numerical issue.

    Examples:
        - astropy BLS raises an internal exception
        - period grid construction fails
        - power spectrum is all NaN
    """
    pass


class FeatureExtractionError(MLCoreError):
    """
    Raised when feature computation fails.

    Examples:
        - phase folding produces an empty in-transit array
        - kurtosis computation fails due to degenerate distribution
    """
    pass


class ClassificationError(MLCoreError):
    """
    Raised when the classifier returns invalid output.

    Examples:
        - predicted class is not one of the three allowed strings
        - confidence is outside [0.0, 1.0]
        - rule_config.yaml is missing required keys
    """
    pass


class PlottingError(MLCoreError):
    """
    Raised when matplotlib rendering fails.

    Examples:
        - figure cannot be saved to BytesIO buffer
        - base64 encoding fails
        - display backend unavailable
    """
    pass