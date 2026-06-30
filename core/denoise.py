"""Conservative transit-preserving denoising.

The cleaned/detrended series remains the detection authority.  This module
creates an optional second representation, masks the initially detected transit
windows, and accepts it only after explicit signal-preservation gates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.signal import savgol_filter


@dataclass
class DenoiseResult:
    flux: np.ndarray
    applied: bool
    accepted: bool
    method: str
    rejection_reasons: list[str]
    noise_before: float
    noise_after: float
    noise_reduction_fraction: float
    depth_attenuation_fraction: float | None
    period_change_fraction: float | None
    duration_change_fraction: float | None
    transit_masked_points: int
    detection_series: str = "cleaned_detrended"

    def metrics(self) -> dict:
        value = asdict(self)
        value.pop("flux")
        return value


def robust_noise(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


def transit_mask(time: np.ndarray, period: float | None, t0: float | None, duration: float | None) -> np.ndarray:
    if not period or not t0 or not duration or period <= 0 or duration <= 0:
        return np.zeros(len(time), dtype=bool)
    phase_time = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    return np.abs(phase_time) <= 0.75 * duration


def denoise(
    time: np.ndarray,
    flux: np.ndarray,
    initial_detection,
    *,
    config: dict | None = None,
    detector=None,
) -> DenoiseResult:
    cfg = {
        "enabled": True,
        "window_points": 7,
        "polyorder": 2,
        "minimum_noise_reduction": 0.10,
        "maximum_depth_attenuation": 0.05,
        "maximum_period_change": 0.01,
        "maximum_duration_change": 0.10,
        **(config or {}),
    }
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if not cfg["enabled"]:
        noise = robust_noise(flux - np.median(flux))
        return DenoiseResult(flux.copy(), False, False, "disabled", ["Denoising disabled by configuration."], noise, noise, 0.0, None, None, None, 0)

    mask = transit_mask(
        time,
        getattr(initial_detection, "best_period", None),
        getattr(initial_detection, "best_t0", None),
        getattr(initial_detection, "best_duration", None),
    )
    oot = ~mask
    if oot.sum() < 50:
        noise = robust_noise(flux)
        return DenoiseResult(flux.copy(), False, False, "savgol_segmented_transit_masked", ["Too few out-of-transit samples."], noise, noise, 0.0, None, None, None, int(mask.sum()))

    candidate = flux.copy()
    cadence = float(np.median(np.diff(time)))
    breaks = np.flatnonzero(np.diff(time) > 5.0 * cadence) + 1
    for indices in np.split(np.arange(len(time)), breaks):
        if len(indices) < 5:
            continue
        window = min(int(cfg["window_points"]), len(indices) if len(indices) % 2 else len(indices) - 1)
        window = max(5, window if window % 2 else window - 1)
        if window > len(indices):
            continue
        smoothed = savgol_filter(flux[indices], window, min(int(cfg["polyorder"]), window - 2), mode="interp")
        replace = ~mask[indices]
        candidate[indices[replace]] = smoothed[replace]

    before = robust_noise(flux[oot] - np.median(flux[oot]))
    after = robust_noise(candidate[oot] - np.median(candidate[oot]))
    reduction = 0.0 if not np.isfinite(before) or before <= 0 else 1.0 - after / before
    depth_change = period_change = duration_change = None
    reasons: list[str] = []

    comparison = None
    if detector is not None:
        try:
            comparison = detector(time, candidate)
        except Exception as exc:
            reasons.append(f"Second-pass detection failed: {exc}")
    initial_detected = bool(getattr(initial_detection, "candidate_detected", False))
    if initial_detected and comparison is not None:
        if not bool(getattr(comparison, "candidate_detected", False)):
            reasons.append("Denoised series lost the initial significant detection.")
        initial_depth = float(getattr(initial_detection, "best_depth", 0.0) or 0.0)
        initial_period = float(getattr(initial_detection, "best_period", 0.0) or 0.0)
        initial_duration = float(getattr(initial_detection, "best_duration", 0.0) or 0.0)
        if initial_depth > 0:
            depth_change = abs(float(getattr(comparison, "best_depth", 0.0)) - initial_depth) / initial_depth
        if initial_period > 0:
            period_change = abs(float(getattr(comparison, "best_period", 0.0)) - initial_period) / initial_period
        if initial_duration > 0:
            duration_change = abs(float(getattr(comparison, "best_duration", 0.0)) - initial_duration) / initial_duration
        if depth_change is not None and depth_change > float(cfg["maximum_depth_attenuation"]):
            reasons.append(f"Transit depth changed by {depth_change:.1%} (>5%).")
        if period_change is not None and period_change >= float(cfg["maximum_period_change"]):
            reasons.append(f"Period changed by {period_change:.1%} (>=1%).")
        if duration_change is not None and duration_change > float(cfg["maximum_duration_change"]):
            reasons.append(f"Duration changed by {duration_change:.1%} (>10%).")
    elif not initial_detected and comparison is not None and bool(getattr(comparison, "candidate_detected", False)):
        reasons.append("Denoising introduced a new significant periodic detection in a control curve.")

    if reduction < float(cfg["minimum_noise_reduction"]):
        reasons.append(f"Out-of-transit robust noise reduction was {reduction:.1%} (<10%).")
    accepted = not reasons
    return DenoiseResult(
        flux=candidate if accepted else flux.copy(),
        applied=accepted,
        accepted=accepted,
        method="savgol_segmented_transit_masked",
        rejection_reasons=reasons,
        noise_before=before,
        noise_after=after if accepted else before,
        noise_reduction_fraction=reduction if accepted else 0.0,
        depth_attenuation_fraction=depth_change,
        period_change_fraction=period_change,
        duration_change_fraction=duration_change,
        transit_masked_points=int(mask.sum()),
    )
