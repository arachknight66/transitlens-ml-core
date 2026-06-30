"""Classification confidence estimation."""

import math


def estimate_confidence(probability: float, threshold: float) -> float:
    """Estimate confidence as normalized distance from the class boundary.

    Confidence is zero at the configured classification threshold and one at
    either probability extreme. This is decision confidence, not statistical
    probability calibration.

    Args:
        probability: Predicted positive-class probability in ``[0, 1]``.
        threshold: Positive-class decision threshold strictly inside ``(0, 1)``.

    Returns:
        Normalized decision confidence in ``[0, 1]``.

    Raises:
        ValueError: If probability or threshold is non-finite or out of range.

    """
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError("probability must be finite and in the range [0, 1]")
    if not math.isfinite(threshold) or threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("threshold must be finite and between zero and one")
    if probability >= threshold:
        return (probability - threshold) / (1.0 - threshold)
    return (threshold - probability) / threshold
