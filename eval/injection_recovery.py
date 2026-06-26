import os
import logging
import numpy as np
import pandas as pd
from core.preprocess import clean
from core.bls_detector import detect

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def inject_transit(time, flux, period, t0, duration, depth):
    """Injects a periodic box-shaped transit signal into a flux array."""
    injected_flux = flux.copy()
    phase = ((time - t0) / period) % 1.0
    in_transit = (phase < (duration / period)) | (phase > 1.0 - (duration / period))
    injected_flux[in_transit] -= depth
    return injected_flux

def run_suite(n_trials=30):
    """Runs a series of injection-recovery trials and writes a summary CSV."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(repo_root, "eval", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate a quiet base light curve
    t = np.linspace(0, 27.0, 10000)
    noise_level = 0.001
    
    rng = np.random.default_rng(42)
    
    records = []
    
    # Define grids for injection parameters
    periods = [1.5, 3.5, 7.0]
    depths = [0.001, 0.005, 0.015] # 0.1%, 0.5%, 1.5%
    durations = [0.1, 0.15]
    
    logger.info("Starting injection-recovery simulation...")
    
    trial_num = 0
    for p in periods:
        for d in depths:
            for dur in durations:
                if trial_num >= n_trials:
                    break
                trial_num += 1
                
                # Create clean light curve + noise
                flux_base = 1.0 + rng.normal(0, noise_level, len(t))
                t0 = rng.uniform(0.1, p / 2.0)
                
                flux_injected = inject_transit(t, flux_base, p, t0, dur, d)
                
                try:
                    # Run clean -> BLS detect
                    clean_res = clean(t, flux_injected)
                    bls_res = detect(clean_res.time, clean_res.flux)
                    
                    recovered = False
                    recovered_period = 0.0
                    if bls_res.candidate_detected and bls_res.best_period is not None:
                        # Check if recovered period is within 1% of injected period
                        period_err = abs(bls_res.best_period - p) / p
                        if period_err < 0.01:
                            recovered = True
                        recovered_period = bls_res.best_period
                        
                    records.append({
                        "trial": trial_num,
                        "injected_period": p,
                        "injected_depth": d,
                        "injected_duration": dur,
                        "noise_rms": noise_level,
                        "recovered": recovered,
                        "recovered_period": recovered_period,
                        "recovered_depth": bls_res.best_depth if bls_res.best_depth else 0.0,
                        "snr": bls_res.snr
                    })
                except Exception as e:
                    logger.warning("Trial %d failed: %s", trial_num, e)
                    
    df_summary = pd.DataFrame(records)
    summary_path = os.path.join(results_dir, "injection_recovery_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    logger.info("Saved injection-recovery summary to %s", summary_path)
    
    # Calculate recovery fraction per depth group
    if len(df_summary) > 0:
        summary_by_depth = df_summary.groupby("injected_depth")["recovered"].mean().reset_index()
        logger.info("Recovery rate by injected depth:\n%s", summary_by_depth.to_string(index=False))

if __name__ == "__main__":
    run_suite()
