# phase_windows.py
# ----------------
# Shared implementation for phase folding, event masking, and sector windowing.

from __future__ import annotations
import numpy as np

def fold_phase(
    time: np.ndarray,
    period: float,
    epoch_btjd: float,
) -> np.ndarray:
    """
    Folds time series timestamps onto [-0.5, 0.5) centered on the epoch.
    Phase 0 corresponds to the transit center.
    """
    if period <= 0:
        raise ValueError(f"Period must be > 0, got {period}")
    phase = ((time - epoch_btjd) / period) % 1.0
    phase[phase >= 0.5] -= 1.0
    return phase

def get_event_mask(
    time: np.ndarray,
    period: float,
    epoch_btjd: float,
    duration_days: float,
    window_multiplier: float = 1.0,
) -> np.ndarray:
    """
    Returns a boolean mask of in-transit points.
    True = point is inside (transit_center - half_duration, transit_center + half_duration).
    """
    if period <= 0 or duration_days <= 0:
        return np.zeros(len(time), dtype=bool)
    phase = fold_phase(time, period, epoch_btjd)
    half_dur_phase = (duration_days / period) / 2.0
    return np.abs(phase) <= (half_dur_phase * window_multiplier)

def assign_cycle_numbers(
    time: np.ndarray,
    period: float,
    epoch_btjd: float,
) -> np.ndarray:
    """
    Assigns an integer cycle number to each timestamp.
    Cycle 0 is the transit event closest to epoch_btjd.
    """
    if period <= 0:
        return np.zeros(len(time), dtype=int)
    return np.round((time - epoch_btjd) / period).astype(int)

def analyze_sector_coverage(
    time: np.ndarray,
    period: float,
    epoch_btjd: float,
    duration_days: float,
    gap_threshold_days: float = 5.0,
) -> dict:
    """
    Analyzes observation gaps, sector coverage, and counts unique observed transits.
    """
    if len(time) == 0 or period <= 0 or duration_days <= 0:
        return {
            "n_cadences": 0,
            "n_unique_observed_events": 0,
            "sectors": [],
            "cycle_coverage": {},
        }
        
    cycles = assign_cycle_numbers(time, period, epoch_btjd)
    unique_cycles = np.unique(cycles)
    
    # Check coverage per cycle: require at least 3 points within the transit duration
    observed_cycles = []
    half_dur = duration_days / 2.0
    for cycle in unique_cycles:
        tc = epoch_btjd + cycle * period
        in_transit = np.abs(time - tc) <= half_dur
        if in_transit.sum() >= 3:
            observed_cycles.append(int(cycle))
            
    # Detect sectors by grouping gaps > gap_threshold_days
    diffs = np.diff(time)
    gap_indices = np.where(diffs > gap_threshold_days)[0]
    
    sector_boundaries = [0]
    for idx in gap_indices:
        sector_boundaries.append(idx + 1)
    sector_boundaries.append(len(time))
    
    sectors_list = []
    for i in range(len(sector_boundaries) - 1):
        s_start = sector_boundaries[i]
        s_end = sector_boundaries[i+1]
        s_time = time[s_start:s_end]
        sectors_list.append({
            "start": float(s_time[0]),
            "end": float(s_time[-1]),
            "points": int(len(s_time)),
        })
        
    return {
        "n_cadences": len(time),
        "n_unique_observed_events": len(observed_cycles),
        "observed_cycle_list": observed_cycles,
        "sectors": sectors_list,
        "unique_cycles_present": [int(c) for c in unique_cycles],
    }
