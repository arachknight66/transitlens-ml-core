"""
core/seeds.py
-------------
Deterministic seed derivation for reproducible multi-stage exoplanet candidate pipeline runs.
"""

import hashlib

def derive_seed(master_seed: int, stage: str, target_id: str) -> int:
    """
    Derive a deterministic 32-bit unsigned integer seed for a specific
    target_id and pipeline stage from a global master_seed.

    Parameters
    ----------
    master_seed : int
        Master random seed for the execution run.
    stage : str
        Name of the pipeline stage (e.g. 'injection', 'bootstrap_fap', 'mcmc').
    target_id : str
        The identifier of the light curve target.

    Returns
    -------
    int
        A derived 32-bit integer seed (0 to 2^32-1) suitable for np.random.default_rng().
    """
    # Combine ingredients into a unique string key
    key = f"{master_seed}:{stage}:{target_id}".encode("utf-8")
    
    # Hash using SHA-256
    hash_hex = hashlib.sha256(key).hexdigest()
    
    # Convert the first 8 hex characters (32-bits) to unsigned integer
    seed_32 = int(hash_hex[:8], 16)
    
    return seed_32
