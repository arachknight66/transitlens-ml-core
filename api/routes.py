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

from fastapi import APIRouter, HTTPException, File, UploadFile, Form
import tempfile
import os
import json
from pathlib import Path

from api.schema import AnalyzeRequest, AnalyzeResponse, HealthResponse, TesscutRequest
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
# POST /analyze/file
# ---------------------------------------------------------------------------

@router.post(
    "/analyze/file",
    response_model=AnalyzeResponse,
    summary="Analyse a light curve from an uploaded file (e.g. FITS)",
    description="Accepts a raw FITS file, parses it using the data pipeline, and runs the full analysis.",
)
async def analyze_file(
    file: UploadFile = File(...),
    target_id: str = Form("unknown"),
    metadata: str = Form("{}"),
    pipeline_config: str = Form("{}", alias="config"),
) -> dict:
    try:
        parsed_metadata = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="metadata must be a valid JSON string")
    try:
        parsed_config = json.loads(pipeline_config)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="config must be valid JSON")
    if not isinstance(parsed_config, dict):
        raise HTTPException(status_code=422, detail="config must be a JSON object")

    original_name = Path(file.filename or "upload").name
    lower_name = original_name.lower()
    allowed = (".csv", ".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz")
    if not lower_name.endswith(allowed):
        raise HTTPException(status_code=415, detail="Supported uploads: CSV, FITS/FIT/FTS, and compressed FITS variants")
    source = "csv" if lower_name.endswith(".csv") else "fits"
    suffix = ".csv" if source == "csv" else (".fits.gz" if lower_name.endswith(".gz") else ".fits")
    fd, tmp_path = tempfile.mkstemp(prefix="transitlens_upload_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            total = 0
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > 256 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Upload exceeds the 256 MiB limit")
                f.write(chunk)

        _add_data_pipeline_path()
        from interface import load_light_curve  # type: ignore
        try:
            lc_data = load_light_curve(
                source=source,
                target_id=target_id,
                config={"path": tmp_path, "original_filename": original_name, "source_type": f"{source}_upload"},
            )
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Authoritative file parsing failed: {e}") from e

        extracted_metadata = lc_data.get("metadata", {})
        parsed_metadata.update(extracted_metadata)
        parsed_metadata["target_id"] = lc_data.get("target_id", target_id)
        for key in ("flux_err", "quality", "centroid_x", "centroid_y"):
            if lc_data.get(key) is not None:
                parsed_metadata[key] = lc_data[key]
        try:
            result = analyze_light_curve(
                time=lc_data["time"],
                flux=lc_data["flux"],
                metadata=parsed_metadata,
                config=parsed_config,
            )
        except InvalidInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        result["source_status"] = "live"
        result["source_provenance"] = extracted_metadata
        result["parser_warnings"] = lc_data.get("warnings", [])
        result["processing_stages"] = _stage_records(extracted_metadata.get("source_type", source))
        return result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# TESScut acquisition
# ---------------------------------------------------------------------------

@router.get("/tesscut/sectors/{tic_id}", summary="Resolve a TIC and list TESScut sectors")
async def tesscut_sectors(tic_id: str) -> dict:
    _add_data_pipeline_path()
    try:
        from real_tess.tesscut_service import list_sectors
        return list_sectors(tic_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/analyze/tesscut", response_model=AnalyzeResponse, summary="Acquire and analyze a TESScut cube")
async def analyze_tesscut(request: TesscutRequest) -> dict:
    return _run_tesscut(request.tic_id, request.sector, request.cutout_size, request.config or {})

@router.post(
    "/analyze/tess",
    response_model=AnalyzeResponse,
    summary="Analyse a TESS light curve by TIC ID",
    description="Accepts a TIC ID, downloads/fetches the light curve from MAST or loads it from cache, and runs the analysis.",
)
async def analyze_tess(
    tic_id: str = Form(...),
    sector: int | None = Form(None),
    cutout_size: int = Form(15),
    metadata: str = Form("{}"),
    pipeline_config: str = Form("{}", alias="config"),
) -> dict:
    """
    Accepts a TIC ID, retrieves the time/flux series via the data-pipeline,
    and runs the full analysis pipeline.
    """
    try:
        parsed_metadata = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="metadata must be a valid JSON string")

    try:
        parsed_config = json.loads(pipeline_config)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="config must be a valid JSON string")
    if not isinstance(parsed_config, dict):
        raise HTTPException(status_code=422, detail="config must be a JSON object")

    clean_id = tic_id.upper().replace("TIC", "").replace("-", "")
    clean_id = "".join(clean_id.split())
    if not clean_id.isdigit():
        raise HTTPException(status_code=422, detail="tic_id must contain a numeric TIC identifier")
    if sector is not None and sector <= 0:
        raise HTTPException(status_code=422, detail="sector must be a positive integer")

    if not 5 <= cutout_size <= 31:
        raise HTTPException(status_code=422, detail="cutout_size must be between 5 and 31")
    return _run_tesscut(clean_id, sector, cutout_size, parsed_config, parsed_metadata)


def _add_data_pipeline_path() -> None:
    import sys
    dp_path = Path(__file__).resolve().parents[2] / "transitlens-data-pipeline"
    if dp_path.exists() and str(dp_path) not in sys.path:
        sys.path.insert(0, str(dp_path))


def _run_tesscut(tic_id: str, sector: int | None, cutout_size: int, config: dict, metadata: dict | None = None) -> dict:
    _add_data_pipeline_path()
    try:
        from real_tess.tesscut_service import acquire_tesscut
        record = acquire_tesscut(
            tic_id,
            sector=sector,
            cutout_size=cutout_size,
            cache_dir=Path(__file__).resolve().parents[2] / "transitlens-data-pipeline" / "real_tess" / "tesscut_cache",
        )
        analysis_metadata = dict(metadata or {})
        analysis_metadata.update(record["metadata"])
        analysis_metadata["target_id"] = record["target_id"]
        for key in ("flux_err", "quality", "centroid_x", "centroid_y"):
            if record.get(key) is not None:
                analysis_metadata[key] = record[key]
        result = analyze_light_curve(record["time"], record["flux"], metadata=analysis_metadata, config=config)
        result["source_status"] = record["metadata"].get("cache_status", "downloaded")
        result["source_provenance"] = record["metadata"]
        result["parser_warnings"] = record.get("warnings", [])
        result["processing_stages"] = _stage_records("tesscut")
        return result
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TESScut acquisition or analysis failed: {exc}") from exc


def _stage_records(source: str) -> list[dict]:
    names = ["authoritative_parse"]
    if source == "tesscut":
        names = ["tic_catalog_lookup", "sector_query", "download_or_cache", "aperture_photometry"]
    names += ["preprocessing", "bls_search", "feature_extraction", "classification", "transit_fitting", "plot_generation"]
    return [{"stage": name, "status": "complete"} for name in names]


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
    except ImportError as e:
        logger.warning(f"demo: data-pipeline not available ({e}), using built-in generator")
        return _generate_synthetic(candidate_id)

    try:
        result = load_light_curve(
            source="synthetic",
            target_id=f"candidate_{candidate_id}",
            config={"generate": True}
        )
        metadata = result.get("metadata", {})
        metadata["target_id"] = f"candidate_{candidate_id}"
        return (
            result["time"],
            result["flux"],
            metadata,
        )
    except Exception as e:
        logger.error(f"demo: load_light_curve failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Integration error: load_light_curve failed: {str(e)}"
        ) from e

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
