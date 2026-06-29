"""Compatibility entry point for the safe Phase 3 tabular training stage.

There is intentionally no sufficiency bypass. Missing real class support is a
hard scientific error enforced by :mod:`ml.dataset` and :mod:`ml.tabular`.
"""
from ml.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train-tabular", *__import__("sys").argv[1:]]))
