"""
api/routes.py
-------------
FastAPI route handlers for the TransitLens ML Core API.

Endpoints:
    GET  /health           — service health check
    POST /analyze          — run full pipeline on a light curve
    GET  /demo/{candidate} — run pipeline on a synthetic demo case

Used by: api/app.py
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from api.schema import AnalyzeRequest, AnalyzeResponse, HealthResponse
from pipeline import analyze_light_curve, _load_config
from core.exceptions import InvalidInputError

logger = logging.getLogger("api")

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns status, version, and timestamp. Used by platform to verify ml-core is running.",
)
async def health() -> HealthResponse:
    cfg = _load_config()
    return HealthResponse(
        status="ok",
        version=str(cfg.get("version", "0.1.0")),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyse a light curve",
    description="Run the full TransitLens pipeline on a raw light curve and return classification, features, confidence, and diagnostic plots.",
)
async def analyze(request: AnalyzeRequest) -> dict:
    """
    Accept a light curve and return the full analysis result.

    Pydantic validates array lengths and types automatically (422 on failure).
    InvalidInputError from the pipeline is converted to a 422 response.
    All other errors are caught by middleware (500 JSON response).
    """
    metadata = request.metadata or {}
    metadata["target_id"] = request.target_id

    try:
        result = analyze_light_curve(
            time=request.time,
            flux=request.flux,
            metadata=metadata,
            config=request.config,
        )
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return result


# ---------------------------------------------------------------------------
# GET /demo/{candidate_id}
# ---------------------------------------------------------------------------

# Synthetic demo data — maps candidate ID to generation parameters
_DEMO_CANDIDATES = {"a", "b", "c"}


@router.get(
    "/demo/{candidate_id}",
    response_model=AnalyzeResponse,
    summary="Run a synthetic demo case",
    description=(
        "Convenience endpoint for the hackathon demo. "
        "Accepts 'a' (exoplanet), 'b' (eclipsing binary), or 'c' (noise). "
        "Generates a synthetic light curve internally and runs the full pipeline."
    ),
)
async def demo(candidate_id: str) -> dict:
    """
    Run the pipeline on a synthetic light curve.

    Tries to load from data-pipeline first; falls back to built-in generator.
    """
    cid = candidate_id.lower().strip()
    if cid not in _DEMO_CANDIDATES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown candidate_id '{candidate_id}'. Must be 'a', 'b', or 'c'.",
        )

    time, flux, metadata = _load_demo_data(cid)

    try:
        result = analyze_light_curve(
            time=time,
            flux=flux,
            metadata=metadata,
        )
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return result


def _load_demo_data(
    candidate_id: str,
) -> tuple[list[float], list[float], dict]:
    """
    Load or generate synthetic demo data for a given candidate.

    Attempts to import from data-pipeline first. Falls back to a built-in
    minimal synthetic generator for standalone operation.
    """
    # Try data-pipeline first
    try:
        import sys
        from pathlib import Path

        # Try to find transitlens-data-pipeline as a sibling repo
        repo_root = Path(__file__).resolve().parent.parent
        dp_path = repo_root.parent / "transitlens-data-pipeline"
        if dp_path.exists() and str(dp_path) not in sys.path:
            sys.path.insert(0, str(dp_path))

        from interface import load_light_curve  # type: ignore
        result = load_light_curve(f"candidate_{candidate_id}")
        return (
            result["time"].tolist(),
            result["flux"].tolist(),
            result.get("metadata", {"target_id": f"candidate_{candidate_id}"}),
        )
    except Exception:
        logger.debug("demo: data-pipeline not available, using built-in generator")

    # Built-in minimal synthetic generator
    return _generate_synthetic(candidate_id)


def _generate_synthetic(
    candidate_id: str,
) -> tuple[list[float], list[float], dict]:
    """
    Generate a minimal synthetic light curve for demo purposes.

    candidate_a: exoplanet-like (P=3.42d, depth=1.3%)
    candidate_b: eclipsing-binary-like (P=1.87d, depth=18%)
    candidate_c: noise only
    """
    import numpy as np

    rng = np.random.default_rng(42)
    n_points = 18000
    t_start = 0.0
    t_end = 27.0
    time = np.linspace(t_start, t_end, n_points)
    cadence = (t_end - t_start) / n_points

    noise_level = 0.001
    flux = 1.0 + rng.normal(0, noise_level, n_points)

    if candidate_id == "a":
        # Exoplanet: flat-bottomed transit
        period, depth, duration = 3.42, 0.013, 0.12
        t0 = 1.5
        phase = ((time - t0) / period) % 1.0
        in_transit = (phase < duration / period) | (phase > 1.0 - duration / period)
        flux[in_transit] -= depth
        metadata = {"target_id": "candidate_a", "true_period": period, "true_depth": depth}

    elif candidate_id == "b":
        # Eclipsing binary: deep V-shaped eclipses with secondary
        period, depth_primary, depth_secondary, duration = 1.87, 0.18, 0.08, 0.15
        t0 = 0.8
        phase = ((time - t0) / period) % 1.0
        # Primary eclipse: V-shape
        half_phase = duration / period / 2.0
        for i, ph in enumerate(phase):
            if ph < half_phase:
                flux[i] -= depth_primary * (1.0 - ph / half_phase)
            elif ph > 1.0 - half_phase:
                flux[i] -= depth_primary * (1.0 - (1.0 - ph) / half_phase)
            # Secondary eclipse at phase 0.5
            elif abs(ph - 0.5) < half_phase:
                flux[i] -= depth_secondary * (1.0 - abs(ph - 0.5) / half_phase)
        metadata = {"target_id": "candidate_b", "true_period": period, "true_depth": depth_primary}

    else:  # candidate_c
        # Pure noise — no signal
        metadata = {"target_id": "candidate_c"}

    return time.tolist(), flux.tolist(), metadata
