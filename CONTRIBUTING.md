# Contributing to transitlens-ml-core

This repository is the central brain of the TransitLens system. Its only job is to receive raw, normalized light curves and perform scientific analysis, signal detection, feature extraction, and classification.

---

## Coding Conventions

- **Functions and variables:** Use `snake_case` (e.g. `analyze_light_curve`, `snr_threshold`).
- **Classes:** Use `PascalCase` (e.g. `InvalidInputError`, `BLSResult`).
- **Module-private helpers:** Prefix with a single underscore (e.g. `_load_config`, `_deep_merge`) so internal routines are clearly distinguished from the public interface.
- **Docstrings:** Every public function and API endpoint gets a NumPy-style docstring (`Parameters` / `Returns` / `Raises` sections) — match the style already used throughout `pipeline.py` and `core/`.
- **Thresholds and Config:** Do not hardcode parameters. All detection and classification thresholds belong in `config.yaml` or `models/rule_config.yaml`.
- **Performance:** Ensure array operations are vectorised using NumPy where possible to maintain low processing times (under 500ms on TESS-cadence datasets).

---

## Repository Structure and Architecture

The code is organized logically by analysis stage:

1. `pipeline.py` — The single public entry point `analyze_light_curve()`.
2. `core/preprocess.py` — Outlier clipping, detrending, and validation.
3. `core/bls_detector.py` — Period searching using Box Least Squares.
4. `core/feature_extractor.py` — Extraction of physical parameters.
5. `core/classifier.py` — Rule-based routing and machine learning classification.
6. `core/confidence.py` — Confidence score calibration.
7. `core/plotter.py` — base64-encoded PNG chart generator.

If you are modifying the output contract of `analyze_light_curve()`, make sure to update `api/schema.py` in lockstep.

---

## Testing Requirements

- **Every change must be tested.** Unit tests are located in the `tests/` directory.
- Run the full suite before submitting your code:
  ```bash
  python -m pytest tests/ -v
  ```
- Ensure all 290+ tests pass with zero failures and that performance invariants are not violated.
- Test new endpoints or API contract additions in `tests/test_api.py` using FastAPI's `TestClient`.

---

## Tri-Repository Boundary Rules

- `transitlens-ml-core` imports from `transitlens-data-pipeline` only when generating demo targets dynamically (in `api/routes.py`). It never imports from `transitlens-platform`.
- Business logic for astronomical analysis belongs in this repository, not in the visualization dashboard.
