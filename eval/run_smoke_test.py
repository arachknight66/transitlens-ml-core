"""
eval/run_smoke_test.py
----------------------
End-to-end smoke test to verify pipeline analysis on a real cached TESS target.
Designed to run in clean environments or CI.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

# Setup paths
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DP_PATH = _REPO_ROOT.parent / "transitlens-data-pipeline"
if str(_DP_PATH) not in sys.path:
    sys.path.insert(0, str(_DP_PATH))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    try:
        from pipeline import analyze_light_curve
        from real_tess.fits_parser import load_fits_and_normalize
    except ImportError as e:
        logger.error(f"Failed to import core modules: {e}")
        sys.exit(1)

    # Locate cached FITS target
    cache_dir = _DP_PATH / "real_tess" / "cache"
    fits_file = cache_dir / "TIC261136679_sector095.fits"
    if not fits_file.exists():
        # Fall back to any available fits in cache
        fits_files = list(cache_dir.glob("*.fits"))
        if len(fits_files) > 0:
            fits_file = fits_files[0]
        else:
            logger.error(f"No cached FITS files found in {cache_dir}. Cannot run smoke test.")
            sys.exit(1)

    logger.info(f"Running end-to-end smoke test on target FITS: {fits_file}")
    
    try:
        # Load and normalize
        parsed = load_fits_and_normalize(str(fits_file))
        time = parsed["time"]
        flux = parsed["flux"]
        metadata = parsed["metadata"]
        target_id = parsed["target_id"]

        logger.info(f"Loaded target {target_id} with {len(time)} points. Executing analyze_light_curve...")

        # Run pipeline
        res = analyze_light_curve(
            time=time,
            flux=flux,
            metadata=metadata
        )

        # Invariant validations
        required_keys = {
            "target_id", "candidate_detected", "predicted_class",
            "confidence", "features", "explanation", "processing_time_ms"
        }
        missing_keys = required_keys - set(res.keys())
        if missing_keys:
            logger.error(f"Smoke test failed: missing keys in result dict: {missing_keys}")
            sys.exit(1)

        logger.info(f"Target ID: {res['target_id']}")
        logger.info(f"Candidate Detected: {res['candidate_detected']}")
        logger.info(f"Predicted Class: {res['predicted_class']}")
        logger.info(f"Confidence: {res['confidence']:.4f}")
        logger.info(f"Processing Time: {res['processing_time_ms']:.1f} ms")

        # Success
        logger.info("SMOKE TEST SUCCESSFUL!")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Smoke test failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
