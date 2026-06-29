# ephemeris_matching.py
# ------------------
# Ephemeris matching against catalogs of known EBs, TCEs, and variable stars.

from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def run_ephemeris_matching(
    target_id: str,
    period: float,
    epoch_btjd: float,
    ra: float | None,
    dec: float | None,
    config: dict,
) -> dict:
    """
    Cross-checks target orbital period and epoch against lists of known binaries.
    """
    unavailable = {
        "matched_source": "none",
        "angular_separation_arcsec": None,
        "period_ratio": None,
        "period_agreement_pct": None,
        "epoch_agreement_days": None,
        "match_score": None,
        "source_catalogue": "none",
        "evidence_strength": "none",
        "ephemeris_match_evidence_flag": False,
    }
    
    if period <= 0 or ra is None or dec is None:
        return unavailable
        
    # Attempt to load a catalog of known EBs if present
    # Look under data/catalogs/known_variables.csv or similar
    catalog_path = Path("data/catalogs/known_variables.csv")
    if not catalog_path.exists():
        # Search parent levels
        catalog_path = Path(__file__).parent.parent.parent / "data" / "catalogs" / "known_variables.csv"
        
    if not catalog_path.exists():
        return unavailable
        
    try:
        df = pd.read_csv(catalog_path)
        if df.empty or "period" not in df.columns or "ra" not in df.columns or "dec" not in df.columns:
            return unavailable
            
        # Select sources within 105 arcsec (5 TESS pixels)
        dra = (df["ra"] - ra) * np.cos(np.radians(dec)) * 3600.0
        ddec = (df["dec"] - dec) * 3600.0
        df["sep"] = np.sqrt(dra**2 + ddec**2)
        
        matches = df[df["sep"] <= 105.0].copy()
        if matches.empty:
            return unavailable
            
        best_match = None
        best_score = -1.0
        
        # Test hypotheses: P_target ≈ P_cat, P_target ≈ 2 * P_cat, 2 * P_target ≈ P_cat, etc.
        for idx, row in matches.iterrows():
            p_cat = float(row["period"])
            t0_cat = float(row.get("epoch", epoch_btjd))
            
            p_ratio = period / p_cat
            
            # Look for close integer ratios: 0.5, 1.0, 2.0
            ratios_to_test = [0.5, 1.0, 2.0]
            best_ratio = 1.0
            min_diff = 999.0
            
            for r in ratios_to_test:
                diff = abs(p_ratio - r) / r
                if diff < min_diff:
                    min_diff = diff
                    best_ratio = r
                    
            # Check period agreement (must be within 1%)
            if min_diff <= 0.01:
                # Re-calculate epoch difference phase offset
                # dt = (epoch_btjd - t0_cat) / p_cat
                # phase_offset = abs(dt - round(dt))
                dt = abs(epoch_btjd - t0_cat) % p_cat
                if dt > p_cat / 2.0:
                    dt = p_cat - dt
                    
                # Match score: 1.0 is perfect period/epoch/spatial match
                score = (1.0 - min_diff) * (1.0 - (row["sep"] / 105.0))
                
                if score > best_score:
                    best_score = score
                    best_match = {
                        "matched_source": str(row.get("source_id", row.get("name", "unknown"))),
                        "angular_separation_arcsec": float(row["sep"]),
                        "period_ratio": float(best_ratio),
                        "period_agreement_pct": float(min_diff * 100.0),
                        "epoch_agreement_days": float(dt),
                        "match_score": float(score),
                        "source_catalogue": str(row.get("catalog", "known_variable_catalog")),
                        "evidence_strength": "strong" if min_diff < 0.001 else "moderate",
                        "ephemeris_match_evidence_flag": True,
                    }
                    
        if best_match is not None:
            return best_match
            
        return unavailable
    except Exception as e:
        logger.warning(f"Ephemeris matching failed: {e}")
        return unavailable
