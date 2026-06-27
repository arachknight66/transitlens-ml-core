"""
core/blend_features.py
----------------------
Blend and contamination diagnostic computations for TransitLens.

Implements four independent diagnostic modules:
    1. Centroid shift analysis  — in-transit vs out-of-transit centroid motion
    2. Crowding/dilution        — CROWDSAP-based aperture contamination
    3. Neighbor diagnostics     — nearby-source risk from catalog data
    4. Blend risk scoring       — weighted aggregation of all evidence

Design principle: missing data is ALWAYS represented as unavailable, never
silently as safe defaults. A centroid_shift of 0.0 when no centroid data
exists would be dishonest; instead, centroid_available=False and
centroid_shift=None.

Used by: core/feature_extractor.py, pipeline.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum in-transit centroid points needed for a reliable shift measurement
_MIN_CENTROID_POINTS_IN_TRANSIT = 5
# Minimum out-of-transit points for scatter estimation
_MIN_CENTROID_POINTS_OUT_TRANSIT = 20
# Buffer factor around transit for out-of-transit centroid selection
_CENTROID_OOT_BUFFER = 1.5
# Significance threshold for flagging centroid shift
_CENTROID_SIGNIFICANCE_THRESHOLD = 3.0

# Crowding thresholds
_CROWDING_MODERATE_THRESHOLD = 0.80
_CROWDING_SEVERE_THRESHOLD = 0.50

# Neighbor risk thresholds
_NEIGHBOR_CLOSE_ARCSEC = 21.0   # ~1 TESS pixel
_NEIGHBOR_BRIGHT_DELTA_MAG = 3.0  # within 3 magnitudes

# Blend risk level boundaries
_RISK_LOW_THRESHOLD = 0.30
_RISK_HIGH_THRESHOLD = 0.70

# Allowed risk levels
BLEND_RISK_LEVELS = ("unavailable", "low", "medium", "high")


# ---------------------------------------------------------------------------
# 1. Centroid Shift Analysis
# ---------------------------------------------------------------------------

def compute_centroid_shift(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    centroid_x: Optional[np.ndarray] = None,
    centroid_y: Optional[np.ndarray] = None,
    quality: Optional[np.ndarray] = None,
    min_points_in: int = _MIN_CENTROID_POINTS_IN_TRANSIT,
    min_points_out: int = _MIN_CENTROID_POINTS_OUT_TRANSIT,
) -> dict:
    """
    Compute centroid shift between in-transit and out-of-transit positions.

    Uses phase-folding to stack all transit events and compares the robust
    median centroid position during transit vs. out-of-transit baseline.

    Parameters
    ----------
    time : np.ndarray
        Timestamps (BTJD), same length as centroid arrays.
    flux : np.ndarray
        Normalised flux, used for quality sanity checks.
    period : float
        Detected orbital period in days.
    t0 : float
        Reference transit epoch (BTJD).
    duration : float
        Transit duration in days.
    centroid_x, centroid_y : np.ndarray or None
        MOM_CENTR1 / MOM_CENTR2 pixel coordinates from TESS.
    quality : np.ndarray or None
        TESS quality flags (0 = good).
    min_points_in : int
        Minimum in-transit centroid measurements required.
    min_points_out : int
        Minimum out-of-transit centroid measurements required.

    Returns
    -------
    dict with keys:
        centroid_available : bool
        centroid_shift : float or None
        centroid_shift_significance : float or None
        centroid_in_transit_x : float or None
        centroid_in_transit_y : float or None
        centroid_out_transit_x : float or None
        centroid_out_transit_y : float or None
        centroid_shift_points_used : int or None
    """
    unavailable = {
        "centroid_available": False,
        "centroid_shift": None,
        "centroid_shift_significance": None,
        "centroid_in_transit_x": None,
        "centroid_in_transit_y": None,
        "centroid_out_transit_x": None,
        "centroid_out_transit_y": None,
        "centroid_shift_points_used": None,
    }

    # Gate: centroid data must exist
    if centroid_x is None or centroid_y is None:
        logger.debug("centroid_shift: no centroid data provided")
        return unavailable

    centroid_x = np.asarray(centroid_x, dtype=float)
    centroid_y = np.asarray(centroid_y, dtype=float)

    if len(centroid_x) != len(time) or len(centroid_y) != len(time):
        logger.warning("centroid_shift: centroid array length mismatch")
        return unavailable

    # Gate: need valid period/duration
    if period <= 0 or duration <= 0:
        logger.debug("centroid_shift: invalid period=%.4f or duration=%.4f", period, duration)
        return unavailable

    # Build validity mask: finite centroids, finite flux, good quality
    valid = np.isfinite(centroid_x) & np.isfinite(centroid_y) & np.isfinite(flux)
    if quality is not None:
        quality = np.asarray(quality)
        valid = valid & (quality == 0)

    if valid.sum() < (min_points_in + min_points_out):
        logger.debug("centroid_shift: too few valid points (%d)", valid.sum())
        return unavailable

    # Phase-fold
    phase = ((time - t0) / period) % 1.0
    # Centre phase at 0 (transit at phase=0)
    phase[phase > 0.5] -= 1.0

    half_dur_phase = (duration / period) / 2.0
    buffer_phase = _CENTROID_OOT_BUFFER * half_dur_phase

    # In-transit: |phase| <= half_dur_phase
    in_transit = valid & (np.abs(phase) <= half_dur_phase)
    # Out-of-transit: |phase| > buffer_phase (with buffer around transit)
    out_transit = valid & (np.abs(phase) > buffer_phase)

    n_in = int(in_transit.sum())
    n_out = int(out_transit.sum())

    if n_in < min_points_in:
        logger.debug("centroid_shift: too few in-transit points (%d < %d)", n_in, min_points_in)
        return unavailable
    if n_out < min_points_out:
        logger.debug("centroid_shift: too few out-of-transit points (%d < %d)", n_out, min_points_out)
        return unavailable

    # Robust median centroids
    cx_in = float(np.median(centroid_x[in_transit]))
    cy_in = float(np.median(centroid_y[in_transit]))
    cx_out = float(np.median(centroid_x[out_transit]))
    cy_out = float(np.median(centroid_y[out_transit]))

    # Shift distance (pixels)
    dx = cx_in - cx_out
    dy = cy_in - cy_out
    shift = float(np.sqrt(dx**2 + dy**2))

    # Uncertainty estimation
    # Standard error of median ≈ 1.253 * σ / sqrt(N) for Gaussian
    # Use out-of-transit scatter as σ estimate
    sigma_x_oot = float(np.std(centroid_x[out_transit], ddof=1))
    sigma_y_oot = float(np.std(centroid_y[out_transit], ddof=1))

    # Combined positional uncertainty for in-transit median
    sigma_combined = float(np.sqrt(
        (1.253 * sigma_x_oot / np.sqrt(n_in))**2 +
        (1.253 * sigma_y_oot / np.sqrt(n_in))**2
    ))

    # Significance
    if sigma_combined > 0:
        significance = shift / sigma_combined
    else:
        significance = 0.0

    logger.info(
        "centroid_shift: shift=%.6f pixels, significance=%.2f sigma "
        "(n_in=%d, n_out=%d, σx=%.6f, σy=%.6f)",
        shift, significance, n_in, n_out, sigma_x_oot, sigma_y_oot,
    )

    return {
        "centroid_available": True,
        "centroid_shift": round(shift, 8),
        "centroid_shift_significance": round(significance, 4),
        "centroid_in_transit_x": round(cx_in, 6),
        "centroid_in_transit_y": round(cy_in, 6),
        "centroid_out_transit_x": round(cx_out, 6),
        "centroid_out_transit_y": round(cy_out, 6),
        "centroid_shift_points_used": n_in,
    }


# ---------------------------------------------------------------------------
# 2. Crowding / Dilution Diagnostics
# ---------------------------------------------------------------------------

def compute_crowding_diagnostics(
    observed_depth: float,
    crowding_metric: Optional[float] = None,
    flux_fraction: Optional[float] = None,
) -> dict:
    """
    Compute dilution/contamination diagnostics from TESS CROWDSAP metadata.

    CROWDSAP is the ratio of target flux to total flux in the photometric
    aperture. A value of 1.0 means no contamination; lower values indicate
    more third-light dilution.

    Parameters
    ----------
    observed_depth : float
        Observed fractional transit depth from BLS.
    crowding_metric : float or None
        CROWDSAP value from FITS header (0.0 to 1.0).
    flux_fraction : float or None
        FLFRCSAP value from FITS header (optional).

    Returns
    -------
    dict with keys:
        crowding_available : bool
        crowding_metric : float or None
        flux_fraction : float or None
        dilution_factor : float or None
        dilution_corrected_depth : float or None
        contamination_ratio : float or None
    """
    unavailable = {
        "crowding_available": False,
        "crowding_metric": None,
        "flux_fraction": None,
        "dilution_factor": None,
        "dilution_corrected_depth": None,
        "contamination_ratio": None,
    }

    if crowding_metric is None:
        return unavailable

    # Validate range
    if not (0.0 < crowding_metric <= 1.0):
        logger.warning(
            "crowding_diagnostics: invalid crowding_metric=%.4f (must be in (0, 1])",
            crowding_metric,
        )
        return unavailable

    # Dilution factor: how much the depth is diluted
    # True depth ≈ observed_depth / crowding_metric
    dilution_factor = 1.0 / crowding_metric
    dilution_corrected_depth = observed_depth * dilution_factor if observed_depth > 0 else 0.0
    contamination_ratio = (1.0 - crowding_metric) / crowding_metric

    logger.info(
        "crowding_diagnostics: CROWDSAP=%.4f, dilution_factor=%.4f, "
        "corrected_depth=%.6f, contamination_ratio=%.4f",
        crowding_metric, dilution_factor, dilution_corrected_depth, contamination_ratio,
    )

    return {
        "crowding_available": True,
        "crowding_metric": round(float(crowding_metric), 6),
        "flux_fraction": round(float(flux_fraction), 6) if flux_fraction is not None else None,
        "dilution_factor": round(dilution_factor, 6),
        "dilution_corrected_depth": round(dilution_corrected_depth, 8),
        "contamination_ratio": round(contamination_ratio, 6),
    }


# ---------------------------------------------------------------------------
# 3. Neighbor Diagnostics
# ---------------------------------------------------------------------------

def load_neighbor_catalog(path: str | Path) -> dict:
    """
    Load a local neighbor catalog CSV into a dict keyed by target_id.

    Expected CSV columns:
        target_id, neighbor_source_id, neighbor_ra, neighbor_dec,
        separation_arcsec, delta_mag, flux_ratio

    Returns
    -------
    dict[str, list[dict]]
        Mapping from target_id to list of neighbor records.
    """
    import csv

    path = Path(path)
    if not path.exists():
        logger.debug("neighbor_catalog: file not found at %s", path)
        return {}

    catalog: dict[str, list[dict]] = {}
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = str(row.get("target_id", "")).strip()
                if not tid:
                    continue
                entry = {
                    "neighbor_source_id": row.get("neighbor_source_id", ""),
                    "separation_arcsec": float(row["separation_arcsec"]) if row.get("separation_arcsec") else None,
                    "delta_mag": float(row["delta_mag"]) if row.get("delta_mag") else None,
                    "flux_ratio": float(row["flux_ratio"]) if row.get("flux_ratio") else None,
                }
                catalog.setdefault(tid, []).append(entry)
    except Exception as exc:
        logger.warning("neighbor_catalog: failed to load %s: %s", path, exc)
        return {}

    logger.info("neighbor_catalog: loaded %d targets from %s", len(catalog), path)
    return catalog


def compute_neighbor_diagnostics(
    target_id: str,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    neighbor_catalog: Optional[dict] = None,
    aperture_radius_arcsec: float = 21.0,
) -> dict:
    """
    Compute neighbor contamination risk from a pre-cached catalog.

    Parameters
    ----------
    target_id : str
        Target identifier for catalog lookup.
    ra, dec : float or None
        Target coordinates (for future online query support).
    neighbor_catalog : dict or None
        Pre-loaded catalog from load_neighbor_catalog().
    aperture_radius_arcsec : float
        TESS pixel aperture radius for risk computation.

    Returns
    -------
    dict with keys:
        neighbor_available : bool
        gaia_neighbor_count : int or None
        nearest_neighbor_sep_arcsec : float or None
        nearest_neighbor_delta_mag : float or None
        neighbor_flux_ratio_sum : float or None
        aperture_neighbor_risk : float or None
    """
    unavailable = {
        "neighbor_available": False,
        "gaia_neighbor_count": None,
        "nearest_neighbor_sep_arcsec": None,
        "nearest_neighbor_delta_mag": None,
        "neighbor_flux_ratio_sum": None,
        "aperture_neighbor_risk": None,
    }

    if neighbor_catalog is None or not neighbor_catalog:
        return unavailable

    # Normalise target_id for lookup
    neighbors = neighbor_catalog.get(target_id)
    if neighbors is None:
        # Try without prefix
        for prefix in ("TIC-", "TIC", "KIC-", "KIC"):
            if target_id.startswith(prefix):
                alt_id = target_id[len(prefix):]
                neighbors = neighbor_catalog.get(alt_id)
                if neighbors:
                    break
    if neighbors is None:
        return unavailable

    # Filter to those within aperture
    in_aperture = [
        n for n in neighbors
        if n.get("separation_arcsec") is not None and n["separation_arcsec"] <= aperture_radius_arcsec
    ]

    if not in_aperture:
        return {
            "neighbor_available": True,
            "gaia_neighbor_count": 0,
            "nearest_neighbor_sep_arcsec": None,
            "nearest_neighbor_delta_mag": None,
            "neighbor_flux_ratio_sum": 0.0,
            "aperture_neighbor_risk": 0.0,
        }

    count = len(in_aperture)

    # Nearest neighbor
    nearest = min(in_aperture, key=lambda n: n["separation_arcsec"])
    nearest_sep = nearest["separation_arcsec"]
    nearest_dmag = nearest.get("delta_mag")

    # Sum of flux ratios
    flux_ratio_sum = sum(
        n["flux_ratio"] for n in in_aperture
        if n.get("flux_ratio") is not None
    )

    # Aperture neighbor risk: fraction of total flux from neighbors
    # risk = sum(flux_ratios) / (1 + sum(flux_ratios))
    aperture_risk = flux_ratio_sum / (1.0 + flux_ratio_sum) if flux_ratio_sum > 0 else 0.0

    logger.info(
        "neighbor_diagnostics: %d neighbors in aperture, nearest=%.1f arcsec, "
        "flux_ratio_sum=%.4f, risk=%.4f",
        count, nearest_sep, flux_ratio_sum, aperture_risk,
    )

    return {
        "neighbor_available": True,
        "gaia_neighbor_count": count,
        "nearest_neighbor_sep_arcsec": round(nearest_sep, 4) if nearest_sep is not None else None,
        "nearest_neighbor_delta_mag": round(nearest_dmag, 4) if nearest_dmag is not None else None,
        "neighbor_flux_ratio_sum": round(flux_ratio_sum, 6),
        "aperture_neighbor_risk": round(aperture_risk, 6),
    }


# ---------------------------------------------------------------------------
# 4. Blend Risk Score
# ---------------------------------------------------------------------------

def compute_blend_risk_score(
    centroid_diag: dict,
    crowding_diag: dict,
    neighbor_diag: dict,
    transit_features: Optional[dict] = None,
) -> dict:
    """
    Compute an aggregate blend risk score from individual diagnostics.

    The score is a weighted combination of available evidence components.
    If no diagnostics are available, the risk level is 'unavailable'.

    Parameters
    ----------
    centroid_diag : dict
        Output of compute_centroid_shift().
    crowding_diag : dict
        Output of compute_crowding_diagnostics().
    neighbor_diag : dict
        Output of compute_neighbor_diagnostics().
    transit_features : dict or None
        Feature dict with secondary_eclipse_depth, depth, etc.

    Returns
    -------
    dict with keys:
        blend_risk_score : float or None
        blend_risk_level : str  ("unavailable", "low", "medium", "high")
        blend_evidence_flags : list[str]
    """
    evidence_flags: list[str] = []
    weighted_scores: list[tuple[float, float]] = []  # (score, weight)

    transit_features = transit_features or {}

    # --- Centroid evidence ---
    if centroid_diag.get("centroid_available"):
        sig = centroid_diag.get("centroid_shift_significance") or 0.0
        # Map significance to [0, 1]: 0 at sig=0, 1 at sig >= 5
        centroid_score = float(np.clip(sig / 5.0, 0.0, 1.0))
        weighted_scores.append((centroid_score, 3.0))  # high weight

        if sig >= _CENTROID_SIGNIFICANCE_THRESHOLD:
            evidence_flags.append(f"centroid_shift_{sig:.1f}sigma")
        if sig >= 5.0:
            evidence_flags.append("strong_centroid_displacement")

    # --- Crowding evidence ---
    if crowding_diag.get("crowding_available"):
        crowd = crowding_diag.get("crowding_metric") or 1.0
        # Map crowding to risk: 1.0 → 0 risk, 0.5 → 1.0 risk
        crowding_score = float(np.clip((1.0 - crowd) / 0.5, 0.0, 1.0))
        weighted_scores.append((crowding_score, 2.0))  # medium weight

        if crowd < _CROWDING_SEVERE_THRESHOLD:
            evidence_flags.append(f"severe_crowding_{crowd:.2f}")
        elif crowd < _CROWDING_MODERATE_THRESHOLD:
            evidence_flags.append(f"moderate_crowding_{crowd:.2f}")

        # Check if dilution-corrected depth pushes signal into EB range
        corr_depth = crowding_diag.get("dilution_corrected_depth")
        obs_depth = transit_features.get("depth", 0.0)
        if corr_depth is not None and obs_depth > 0 and corr_depth > 0.05 and obs_depth < 0.05:
            evidence_flags.append("dilution_corrected_depth_exceeds_eb_threshold")

    # --- Neighbor evidence ---
    if neighbor_diag.get("neighbor_available"):
        risk = neighbor_diag.get("aperture_neighbor_risk") or 0.0
        weighted_scores.append((float(np.clip(risk * 2, 0.0, 1.0)), 1.5))

        count = neighbor_diag.get("gaia_neighbor_count") or 0
        nearest_sep = neighbor_diag.get("nearest_neighbor_sep_arcsec")
        nearest_dmag = neighbor_diag.get("nearest_neighbor_delta_mag")

        if count > 0:
            evidence_flags.append(f"neighbors_in_aperture_{count}")
        if nearest_sep is not None and nearest_sep < _NEIGHBOR_CLOSE_ARCSEC:
            if nearest_dmag is not None and nearest_dmag < _NEIGHBOR_BRIGHT_DELTA_MAG:
                evidence_flags.append(
                    f"close_bright_neighbor_{nearest_sep:.1f}arcsec_dmag{nearest_dmag:.1f}"
                )

    # --- Secondary eclipse + centroid/crowding compound ---
    sec_depth = transit_features.get("secondary_eclipse_depth", 0.0)
    if sec_depth > 0.001:
        has_spatial = centroid_diag.get("centroid_available") or crowding_diag.get("crowding_available")
        has_spatial_risk = any(
            "centroid_shift" in f or "crowding" in f for f in evidence_flags
        )
        if has_spatial and has_spatial_risk:
            evidence_flags.append("secondary_eclipse_with_spatial_anomaly")

    # --- Aggregate ---
    if not weighted_scores:
        return {
            "blend_risk_score": None,
            "blend_risk_level": "unavailable",
            "blend_evidence_flags": [],
        }

    total_weight = sum(w for _, w in weighted_scores)
    risk_score = sum(s * w for s, w in weighted_scores) / total_weight

    if risk_score < _RISK_LOW_THRESHOLD:
        level = "low"
    elif risk_score < _RISK_HIGH_THRESHOLD:
        level = "medium"
    else:
        level = "high"

    logger.info(
        "blend_risk_score: score=%.4f, level=%s, flags=%s",
        risk_score, level, evidence_flags,
    )

    return {
        "blend_risk_score": round(float(risk_score), 6),
        "blend_risk_level": level,
        "blend_evidence_flags": evidence_flags,
    }


# ---------------------------------------------------------------------------
# 5. Combined blend diagnostics (convenience wrapper)
# ---------------------------------------------------------------------------

def extract_blend_diagnostics(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    observed_depth: float,
    metadata: Optional[dict] = None,
    centroid_x: Optional[np.ndarray] = None,
    centroid_y: Optional[np.ndarray] = None,
    quality: Optional[np.ndarray] = None,
    neighbor_catalog: Optional[dict] = None,
    transit_features: Optional[dict] = None,
) -> dict:
    """
    Run all blend diagnostics and return a unified result dict.

    This is the main entry point called by feature_extractor.py.

    Returns
    -------
    dict with all diagnostic fields plus classifier-safe numeric features:
        centroid_shift, crowding_metric, gaia_neighbor_count
    """
    metadata = metadata or {}

    # 1. Centroid
    centroid_diag = compute_centroid_shift(
        time=time,
        flux=flux,
        period=period,
        t0=t0,
        duration=duration,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        quality=quality,
    )

    # 2. Crowding
    crowd_val = metadata.get("crowding_metric") or metadata.get("CROWDSAP")
    flfr_val = metadata.get("flux_fraction") or metadata.get("FLFRCSAP")
    crowding_diag = compute_crowding_diagnostics(
        observed_depth=observed_depth,
        crowding_metric=float(crowd_val) if crowd_val is not None else None,
        flux_fraction=float(flfr_val) if flfr_val is not None else None,
    )

    # 3. Neighbors
    target_id = metadata.get("target_id", "unknown")
    ra = metadata.get("ra")
    dec = metadata.get("dec")
    neighbor_diag = compute_neighbor_diagnostics(
        target_id=str(target_id),
        ra=float(ra) if ra is not None else None,
        dec=float(dec) if dec is not None else None,
        neighbor_catalog=neighbor_catalog,
    )

    # 4. Risk score
    risk_diag = compute_blend_risk_score(
        centroid_diag=centroid_diag,
        crowding_diag=crowding_diag,
        neighbor_diag=neighbor_diag,
        transit_features=transit_features,
    )

    # Assemble full diagnostics
    diagnostics = {
        **centroid_diag,
        **{k: v for k, v in crowding_diag.items() if k != "crowding_available"},
        "crowding_available": crowding_diag["crowding_available"],
        **{k: v for k, v in neighbor_diag.items() if k != "neighbor_available"},
        "neighbor_available": neighbor_diag["neighbor_available"],
        **risk_diag,
    }

    # Classifier-safe numeric features (for backward compatibility with
    # the 16-feature schema). These use 0.0/1.0/0 defaults when unavailable
    # because the ML model requires finite floats, but the diagnostics dict
    # above preserves the honest None/unavailable representation.
    classifier_features = {
        "centroid_shift": centroid_diag["centroid_shift"] if centroid_diag["centroid_available"] else 0.0,
        "crowding_metric": crowding_diag["crowding_metric"] if crowding_diag["crowding_available"] else 1.0,
        "gaia_neighbor_count": neighbor_diag["gaia_neighbor_count"] if neighbor_diag["neighbor_available"] else 0,
    }

    return {
        "diagnostics": diagnostics,
        "classifier_features": classifier_features,
    }


# ---------------------------------------------------------------------------
# 6. Blend explanation (availability-aware)
# ---------------------------------------------------------------------------

def get_blend_explanation(diagnostics: dict, predicted_class: str) -> str:
    """
    Generate a human-readable explanation of blend diagnostics.

    Clearly distinguishes between:
    - "no contamination detected" (diagnostics ran, nothing found)
    - "contamination diagnostics unavailable" (no data to run diagnostics)
    - "contamination likely" (diagnostics found evidence)
    """
    parts: list[str] = []

    centroid_avail = diagnostics.get("centroid_available", False)
    crowding_avail = diagnostics.get("crowding_available", False)
    neighbor_avail = diagnostics.get("neighbor_available", False)
    risk_level = diagnostics.get("blend_risk_level", "unavailable")

    # Availability summary
    avail_sources = []
    unavail_sources = []
    for name, avail in [("centroid", centroid_avail), ("crowding/CROWDSAP", crowding_avail), ("neighbor/Gaia", neighbor_avail)]:
        if avail:
            avail_sources.append(name)
        else:
            unavail_sources.append(name)

    if risk_level == "unavailable":
        parts.append(
            "Blend diagnostics unavailable: no centroid, crowding, or neighbor metadata was provided."
        )
        if unavail_sources:
            parts.append(f"Missing sources: {', '.join(unavail_sources)}.")
        return " ".join(parts)

    # We have at least some diagnostics
    if risk_level == "high":
        parts.append("High blend risk detected.")
    elif risk_level == "medium":
        parts.append("Moderate blend risk detected.")
    else:
        parts.append("Low blend risk: no significant contamination evidence found.")

    # Centroid details
    if centroid_avail:
        sig = diagnostics.get("centroid_shift_significance", 0.0)
        shift = diagnostics.get("centroid_shift", 0.0)
        if sig is not None and sig >= _CENTROID_SIGNIFICANCE_THRESHOLD:
            parts.append(
                f"In-transit centroid shift detected at {sig:.1f}σ "
                f"(displacement = {shift:.4f} pixels)."
            )
        else:
            parts.append(
                f"No significant centroid shift (significance = {sig:.1f}σ)."
            )

    # Crowding details
    if crowding_avail:
        crowd = diagnostics.get("crowding_metric", 1.0)
        parts.append(f"CROWDSAP = {crowd:.3f}.")
        if crowd is not None and crowd < _CROWDING_MODERATE_THRESHOLD:
            corr_depth = diagnostics.get("dilution_corrected_depth")
            if corr_depth is not None:
                parts.append(
                    f"Significant aperture contamination: dilution-corrected depth = {corr_depth:.4f}."
                )

    # Neighbor details
    if neighbor_avail:
        count = diagnostics.get("gaia_neighbor_count", 0)
        if count and count > 0:
            nearest = diagnostics.get("nearest_neighbor_sep_arcsec")
            dmag = diagnostics.get("nearest_neighbor_delta_mag")
            parts.append(
                f"{count} neighbor(s) in aperture"
                + (f", nearest at {nearest:.1f} arcsec" if nearest else "")
                + (f" (Δmag = {dmag:.1f})" if dmag is not None else "")
                + "."
            )
        else:
            parts.append("No neighbors detected in aperture.")

    # Unavailable sources
    if unavail_sources:
        parts.append(f"Unavailable diagnostics: {', '.join(unavail_sources)}.")

    # Evidence flags
    flags = diagnostics.get("blend_evidence_flags", [])
    if flags:
        parts.append(f"Evidence flags: {', '.join(flags)}.")

    return " ".join(parts)
