# gaia_neighbors.py
# -----------------
# Gaia neighbor diagnostics: cone queries, proper-motion propagation, and flux contamination ratios.

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
import time
import numpy as np

logger = logging.getLogger(__name__)

# TESS to Gaia flux relation helper (Stassun et al. 2019 relation)
# Tmag = G - 0.430 for typical stars, or using BP/RP colours if available:
# Tmag = G - 0.0053*(BP-RP)^2 - 0.36*(BP-RP) - 0.03
def estimate_tess_magnitude(g_mag: float, bp_mag: float | None = None, rp_mag: float | None = None) -> float:
    """Calculates approximate TESS magnitude from Gaia G, BP, and RP magnitudes."""
    if bp_mag is not None and rp_mag is not None:
        color = bp_mag - rp_mag
        # Stassun 2019 conversion formula
        tmag = g_mag - 0.0053 * (color ** 2) - 0.360 * color - 0.030
        if np.isfinite(tmag):
            return float(tmag)
    # Fallback to simple offset
    return float(g_mag - 0.430)

def run_gaia_neighbor_query(
    target_id: str,
    ra: float | None,
    dec: float | None,
    config: dict,
) -> dict:
    """
    Queries Gaia for nearby sources around the target coordinates and calculates contamination statistics.
    Uses local cache directory to avoid repeat network requests.
    """
    g_config = config.get("Gaia", {})
    cache_dir = Path(g_config.get("cache_directory", "data/cache/gaia"))
    radius_arcsec = g_config.get("cone_search_radius_arcsec", 105.0)
    mag_limit_diff = g_config.get("maximum_neighbor_magnitude_difference", 10.0)
    pm_prop = g_config.get("proper_motion_propagation", True)
    
    unavailable = {
        "gaia_available": False,
        "gaia_release": "DR3",
        "gaia_target_source_id": None,
        "gaia_target_match_sep_arcsec": None,
        "gaia_neighbor_count": None,
        "nearest_neighbor_source_id": None,
        "nearest_neighbor_sep_arcsec": None,
        "nearest_neighbor_delta_gmag": None,
        "nearest_neighbor_delta_tmag": None,
        "summed_neighbor_flux_ratio": None,
        "aperture_weighted_neighbor_flux_ratio": None,
        "brightest_neighbor_position_angle": None,
        "gaia_query_timestamp": "",
        "gaia_cache_key": "",
        "gaia_evidence_flag": False,
        "gaia_quality": "unavailable",
    }
    
    if ra is None or dec is None:
        return unavailable
        
    # Generate cache key based on coordinates
    cache_key = f"gaia_ra{ra:.5f}_dec{dec:.5f}_r{radius_arcsec:.1f}"
    cache_file = cache_dir / f"{cache_key}.json"
    
    rows = []
    loaded_from_cache = False
    
    # ── Check offline cache ──
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                rows = json.load(f)
            loaded_from_cache = True
            logger.debug(f"Gaia neighbor data loaded from cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to read Gaia cache file: {e}")
            
    # ── Online Gaia query (Astroquery) ──
    if not loaded_from_cache and g_config.get("offline_only", False):
        unavailable["gaia_cache_key"] = cache_key
        unavailable["gaia_quality"] = "unavailable_cache_miss"
        return unavailable

    if not loaded_from_cache:
        # Check if we should query online
        try:
            from astroquery.gaia import Gaia
            Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
            
            # Execute cone search query
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
            width = u.Quantity(radius_arcsec, u.arcsec)
            
            logger.info(f"Querying Gaia DR3 for target {target_id} around ra={ra:.5f}, dec={dec:.5f}")
            job = Gaia.cone_search_async(coordinate=coord, radius=width)
            results = job.get_results()
            
            # Convert astropy table to JSON-serializable list of dicts
            rows = []
            if len(results) > 0:
                for r in results:
                    rows.append({
                        "source_id": int(r["source_id"]),
                        "ra": float(r["ra"]),
                        "dec": float(r["dec"]),
                        "pmra": float(r["pmra"]) if not np.isnan(r["pmra"]) else 0.0,
                        "pmdec": float(r["pmdec"]) if not np.isnan(r["pmdec"]) else 0.0,
                        "phot_g_mean_mag": float(r["phot_g_mean_mag"]) if not np.isnan(r["phot_g_mean_mag"]) else None,
                        "phot_bp_mean_mag": float(r["phot_bp_mean_mag"]) if not np.isnan(r["phot_bp_mean_mag"]) else None,
                        "phot_rp_mean_mag": float(r["phot_rp_mean_mag"]) if not np.isnan(r["phot_rp_mean_mag"]) else None,
                    })
                    
            # Write to cache
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
            logger.info(f"Cached Gaia query to {cache_file}")
            
        except Exception as exc:
            logger.debug(f"Gaia online query failed (or astroquery not installed): {exc}")
            # If not in cache and online fails, return unavailable
            return unavailable
            
    if not rows:
        return {
            **unavailable,
            "gaia_available": True,
            "gaia_neighbor_count": 0,
            "summed_neighbor_flux_ratio": 0.0,
            "aperture_weighted_neighbor_flux_ratio": 0.0,
            "gaia_quality": "clean_no_neighbors",
        }
        
    # Process rows
    # 1. Match target star: the closest source to search RA/Dec
    target_idx = -1
    min_dist = 999.0
    
    # Calculate separations
    seps = []
    for i, r in enumerate(rows):
        dra = (r["ra"] - ra) * np.cos(np.radians(dec)) * 3600.0
        ddec = (r["dec"] - dec) * 3600.0
        dist = np.sqrt(dra**2 + ddec**2)
        seps.append(dist)
        if dist < min_dist:
            min_dist = dist
            target_idx = i
            
    if target_idx == -1:
        return unavailable
        
    target_row = rows[target_idx]
    target_source_id = target_row["source_id"]
    target_g_mag = target_row["phot_g_mean_mag"] or 15.0
    target_tmag = estimate_tess_magnitude(target_g_mag, target_row.get("phot_bp_mean_mag"), target_row.get("phot_rp_mean_mag"))
    
    # 2. Extract neighbors (all sources except target)
    neighbors = []
    for i, r in enumerate(rows):
        if i == target_idx:
            continue
        g_mag = r["phot_g_mean_mag"]
        if g_mag is None:
            continue
            
        # Proper motion propagation to TESS epoch (~2024.0, Gaia baseline is 2016.0)
        # dt = 8.0 years
        # ra_corr = ra + pmra * dt / 1000.0 / 3600.0 / cos(dec)
        n_ra = r["ra"]
        n_dec = r["dec"]
        if pm_prop and r.get("pmra") and r.get("pmdec"):
            dt = 8.0 # years
            n_ra += (r["pmra"] * dt / 1000.0) / 3600.0 / np.cos(np.radians(dec))
            n_dec += (r["pmdec"] * dt / 1000.0) / 3600.0
            
        # Recompute distance
        dra = (n_ra - ra) * np.cos(np.radians(dec)) * 3600.0
        ddec = (n_dec - dec) * 3600.0
        dist = float(np.sqrt(dra**2 + ddec**2))
        
        tmag = estimate_tess_magnitude(g_mag, r.get("phot_bp_mean_mag"), r.get("phot_rp_mean_mag"))
        
        # Position angle (PA)
        pa = float(np.degrees(np.arctan2(dra, ddec))) % 360.0
        
        neighbors.append({
            "source_id": r["source_id"],
            "sep_arcsec": dist,
            "g_mag": g_mag,
            "t_mag": tmag,
            "pa": pa,
        })
        
    # Sort neighbors by separation
    neighbors = sorted(neighbors, key=lambda n: n["sep_arcsec"])
    
    n_count = len(neighbors)
    if n_count == 0:
        return {
            "gaia_available": True,
            "gaia_release": "DR3",
            "gaia_target_source_id": int(target_source_id),
            "gaia_target_match_sep_arcsec": round(min_dist, 4),
            "gaia_neighbor_count": 0,
            "nearest_neighbor_source_id": None,
            "nearest_neighbor_sep_arcsec": None,
            "nearest_neighbor_delta_gmag": None,
            "nearest_neighbor_delta_tmag": None,
            "summed_neighbor_flux_ratio": 0.0,
            "aperture_weighted_neighbor_flux_ratio": 0.0,
            "brightest_neighbor_position_angle": None,
            "gaia_query_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gaia_cache_key": cache_key,
            "gaia_evidence_flag": False,
            "gaia_quality": "clean",
        }
        
    nearest = neighbors[0]
    nearest_sep = nearest["sep_arcsec"]
    nearest_dg = nearest["g_mag"] - target_g_mag
    nearest_dt = nearest["t_mag"] - target_tmag
    
    # Summed neighbor flux ratio within standard aperture (e.g. 42 arcsec, ~2 pixels)
    sum_flux_ratio = 0.0
    weighted_flux_ratio = 0.0
    
    for n in neighbors:
        # relative flux = 10 ** (-0.4 * delta_mag)
        f_ratio = 10.0 ** (-0.4 * (n["t_mag"] - target_tmag))
        sum_flux_ratio += f_ratio
        
        # Aperture weighting: Gaussian profile or simple linear falloff (closer = more weight)
        # Let's use simple exponential falloff: weight = exp(-d / aperture_radius)
        ap_radius = g_config.get("contamination_aperture_radius_arcsec", 42.0)
        weight = float(np.exp(-n["sep_arcsec"] / ap_radius))
        weighted_flux_ratio += f_ratio * weight
        
    # Brightest neighbor position angle
    brightest = min(neighbors, key=lambda n: n["t_mag"])
    brightest_pa = brightest["pa"]
    
    # Evidence flag: neighbor within 21 arcsec (1 TESS pixel) and delta_mag < mag_limit_diff (e.g. 10)
    # or cumulative contamination > 0.10
    evidence_flag = (nearest_sep <= 21.0 and nearest_dt < 5.0) or (sum_flux_ratio >= 0.10)
    
    quality = "crowded" if evidence_flag else "isolated"
    
    return {
        "gaia_available": True,
        "gaia_release": "DR3",
        "gaia_target_source_id": int(target_source_id),
        "gaia_target_match_sep_arcsec": round(min_dist, 4),
        "gaia_neighbor_count": int(n_count),
        "nearest_neighbor_source_id": int(nearest["source_id"]),
        "nearest_neighbor_sep_arcsec": round(nearest_sep, 4),
        "nearest_neighbor_delta_gmag": round(nearest_dg, 4),
        "nearest_neighbor_delta_tmag": round(nearest_dt, 4),
        "summed_neighbor_flux_ratio": round(sum_flux_ratio, 6),
        "aperture_weighted_neighbor_flux_ratio": round(weighted_flux_ratio, 6),
        "brightest_neighbor_position_angle": round(brightest_pa, 4),
        "gaia_query_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gaia_cache_key": cache_key,
        "gaia_evidence_flag": bool(evidence_flag),
        "gaia_quality": quality,
    }
