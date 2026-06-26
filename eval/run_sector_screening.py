"""
eval/run_sector_screening.py
----------------------------
Automated entry point for TESS Sector Integration.
Orchestrates manifest building, downloads, processing, and candidate screening.

Usage:
    python -m eval.run_sector_screening --sector 98 --limit 10 --download
"""

from __future__ import annotations

import argparse
import sys
import os
import logging
from pathlib import Path

# Ensure ml-core root is on the path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Sibling path to data-pipeline
_DP_PATH = _REPO_ROOT.parent / "transitlens-data-pipeline"
if str(_DP_PATH) not in sys.path:
    sys.path.insert(0, str(_DP_PATH))

# Add real_tess folder to path
sys.path.append(str(_DP_PATH / "real_tess"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="TransitLens Sector Ingestion and Screening workflow.")
    parser.add_argument("--sector", type=int, default=98, help="TESS Sector number (default: 98)")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of targets to build/download/screen")
    parser.add_argument("--cadence", type=str, default="short", help="Target cadence (default: 'short' for <=2min)")
    parser.add_argument("--use-cache", action="store_true", help="Utilize locally cached FITS files (default: False)")
    parser.add_argument("--download", action="store_true", help="Query MAST and download pending FITS files (default: False)")
    parser.add_argument("--process-only", action="store_true", help="Build manifest and process arrays without running ML core classifier")
    parser.add_argument("--screen-only", action="store_true", help="Execute ML classification on already processed NPZ files only")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    
    args = parser.parse_args()
    
    # Setup directories
    tess_sector_dir = _DP_PATH / "datasets" / "processed" / "tess_sector"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = tess_sector_dir
        
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sector_manifest.csv"
    cache_dir = _DP_PATH / "real_tess" / "cache"
    
    logger.info("=" * 70)
    logger.info(f"  TransitLens — Sector Ingestion and Screening (Sector {args.sector})")
    logger.info("=" * 70)
    
    # Import pipeline scripts
    from sector_manifest import build_sector_manifest
    from sector_downloader import download_manifest_targets
    from sector_processor import process_sector_manifest
    from sector_screening import screen_sector_targets
    
    if not args.screen_only:
        # Step 1: Build Sector Manifest
        logger.info("[Step 1] Building/loading target manifest...")
        build_sector_manifest(
            sector=args.sector,
            output_path=str(manifest_path),
            limit=args.limit,
            cadence=args.cadence,
            cache_dir=str(cache_dir)
        )
        
        # Step 2: Download files if requested
        if args.download:
            logger.info("[Step 2] Ingesting/downloading TESS FITS products from MAST...")
            download_manifest_targets(
                manifest_path=str(manifest_path),
                limit=args.limit,
                cache_dir=str(cache_dir)
            )
        else:
            logger.info("[Step 2] Skipping download (run with --download to query and download from MAST).")
            
        # Step 3: Process FITS files to normalized NPZ arrays
        logger.info("[Step 3] Processing FITS files to normalized NPZ arrays...")
        process_sector_manifest(
            manifest_path=str(manifest_path),
            output_dir=str(output_dir)
        )
    else:
        logger.info("Skipping manifest build, download, and parse stages (--screen-only specified).")
        
    if not args.process_only:
        # Step 4: Screen processed targets
        logger.info("[Step 4] Screening processed targets using TransitLens ML Core...")
        screen_sector_targets(
            manifest_path=str(manifest_path),
            output_dir=str(output_dir)
        )
    else:
        logger.info("Skipping ML core classification stage (--process-only specified).")
        
    logger.info("=" * 70)
    logger.info("  Ingestion and Screening workflow finished!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
