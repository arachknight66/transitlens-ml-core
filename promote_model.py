"""Compatibility entry point for strict, evidence-only Phase 3 promotion.

No eligibility bypass exists. Promotion validates the immutable gate, metadata,
required artifacts, and every registered checksum before atomically activating.
"""
from ml.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["promote", *__import__("sys").argv[1:]]))
