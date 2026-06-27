# Phase 8 Repository Audit Note

This document summarizes the comprehensive repository audit performed for TransitLens in Phase 8.

---

## 1. Project Layout & Components

TransitLens consists of three primary subprojects:

1. **`transitlens-data-pipeline/`**
   - **Purpose**: Handles real TESS data downloads, synthetic light curve generation, and split manifestations.
   - **Dependencies**: numpy, pandas, scipy, pyyaml, lightkurve, astroquery, pytest.
   - **Key Scripts**: `datasets/build_dataset.py`, `datasets/build_real_evaluation_dataset.py`, `datasets/validate_dataset.py`.
   - **Datasets**: Curated gold set (`datasets/gold_set.csv`) and NPZ splits (`datasets/processed/lightcurves/splits/`).

2. **`transitlens-ml-core/`**
   - **Purpose**: Feature extraction, classification rules + ML, physical transit fitting, parameter estimation, uncertainty quantification, and FastAPI server.
   - **Dependencies**: numpy, scipy, astropy, scikit-learn, xgboost, matplotlib, fastapi, uvicorn, pydantic, pyyaml, pytest, httpx, emcee, corner.
   - **Key Entry Points**: `pipeline.py` (main entry), `train_model.py` (classifier training), `prepare_ml.py` (feature extraction).
   - **Evaluation**: Scripts in `eval/` (`run_full_evaluation.py`, `evaluate_phase7.py`, `run_injection_recovery.py`, `run_sector_screening.py`).
   - **Models**: Pre-trained Random Forest and XGBoost classifiers in `models/`.

3. **`transitlens-platform/`**
   - **Purpose**: Dashboard user interface.
   - **Dependencies**: streamlit, plotly, pandas, numpy, requests, jinja2, pyyaml, pillow.
   - **Key Script**: `main.py` (starts Streamlit).

---

## 2. Current Setup & Testing Status

- **Setup Process**: Manual execution of `pip install -r requirements.txt` across all three subdirectories. No dependency lock files exist.
- **Testing Process**: Executed via `python -m pytest` inside `transitlens-ml-core`.
- **Audit Findings**:
  - Matplotlib tests in `tests/test_plotter.py` failed due to key layout modifications in Phase 7 (returned 7 plots instead of 4, with empty strings for non-fitted plots). Fixed in this phase.
  - Overall 375 tests collected and passing (100% green).

---

## 3. Identified Reproducibility & QA Gaps

- **Lack of Lockfiles**: Dependencies are defined in loose `requirements.txt` with minimum versions (`numpy>=1.24`, etc.). A clean install is vulnerable to upstream version changes.
- **Randomness Gaps**:
  - Random Forest and XGBoost models are trained with `--seed` command options, but seed propagation across target stages is missing.
  - MCMC posterior sampling lacks derived child seeds for parallelization/ordering robustness.
- **Missing Global CLI**: The pipeline consists of independent scripts requiring complex positional execution. There is no unified entry point.
- **Leakage Prevention Check**: There is a disjoint target validation check in `datasets/validate_dataset.py`, but no check verifying that multiple sectors of the same star do not cross splits, or that synthetic injections derived from base curves remain in the same split.
- **Schema Versions**: Serialization outputs (detections, features, fits, predictions) lack explicit schema identifiers or schema validation at boundaries.
- **Run Directory & Manifests**: Runs overwrite each other or place results in static folders (`eval/results/`) without a timestamped run ID or configuration hash.

---

## 4. Scientific Claims Audit

The official claimed metrics registry in `docs/evaluation_claims.md` contains:
- **Curated Gold Set Accuracy**: 100% (on $N=12$ targets).
- **Validation/Test Split Accuracy**: 100% (evaluated on $N=3$ toy targets).
- **Standalone Classifier Validation**: 64% RF, 62% XGBoost (on $N=200$ targets).
- **Parameter Estimation**: Period error $0.0167\%$, depth error $2.77\%$, duration error $30\%$ (on $N=1$ test target).

All claims are mathematically trace-calibrated to generated evidence files. We must ensure that the unified CLI can programmatically verify these claims.
