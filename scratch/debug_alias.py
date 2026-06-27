import numpy as np
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.evaluate_phase7 import generate_synthetic_curve
from core.transit_fitting_pipeline import resolve_period_alias_hypotheses
from core.utils import phase_fold

rng = np.random.default_rng(42)
t, f, fe = generate_synthetic_curve(rng, 1.58040, 1.05, 0.0135, 0.036, 0.0005)

# If BLS found the half period alias (0.7902)
res = resolve_period_alias_hypotheses(
    t, f, fe,
    bls_period=0.79020, bls_t0=1.05, bls_duration=0.036, bls_depth=0.0135
)

print("Preferred Period:", res["preferred_period"])
print("Alias Warning:", res["alias_warning"])
print("Alias Type:", res["alias_type"])
print("Reason:", res["reason"])
print("Odd-Even Delta:", res["odd_even_delta"])
print("Secondary Depth:", res["secondary_depth"])
print("Local Noise:", np.std(np.diff(f)) / np.sqrt(2.0))

# Print counts
p_cand = res["hypotheses"]["P"]["period"]
t0_cand = res["hypotheses"]["P"]["t0"]
dur_cand = res["hypotheses"]["P"]["duration"]
phase_c = phase_fold(t, p_cand, t0_cand)
cycles_c = np.round((t - t0_cand) / p_cand)
half_dur = (dur_cand / p_cand) / 2.0
in_transit_c = np.abs(phase_c) < half_dur
odd_mask = (cycles_c % 2 == 1) & in_transit_c
even_mask = (cycles_c % 2 == 0) & in_transit_c

print("Odd Mask Count:", np.sum(odd_mask))
print("Even Mask Count:", np.sum(even_mask))
if np.sum(odd_mask) > 0:
    print("Odd median flux:", np.median(f[odd_mask]))
if np.sum(even_mask) > 0:
    print("Even median flux:", np.median(f[even_mask]))
