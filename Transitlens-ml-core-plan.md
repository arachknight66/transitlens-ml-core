# transitlens-ml-core — Phased Build Plan

> **Project:** TransitLens — Bharatiya Antariksh Hackathon 2026
> **Problem Statement:** PS7 — AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves
> **Repo Role:** All signal processing, BLS detection, feature extraction, classification, confidence scoring, plotting, and HTTP API
> **Document Type:** Engineering build plan — no code, phases only
> **Last Updated:** 2026

---

## Table of Contents

1. [Repo Purpose and Boundaries](#1-repo-purpose-and-boundaries)
2. [Folder Structure Reference](#2-folder-structure-reference)
3. [The Central Interface Contract](#3-the-central-interface-contract)
4. [Phase Overview](#4-phase-overview)
5. [Phase 1 — Preprocessing and Signal Cleaning](#5-phase-1--preprocessing-and-signal-cleaning)
6. [Phase 2 — BLS Transit Detection](#6-phase-2--bls-transit-detection)
7. [Phase 3 — Feature Extraction](#7-phase-3--feature-extraction)
8. [Phase 4 — Classification and Confidence Scoring](#8-phase-4--classification-and-confidence-scoring)
9. [Phase 5 — Pipeline Orchestration](#9-phase-5--pipeline-orchestration)
10. [Phase 6 — Plotting and Visualisation](#10-phase-6--plotting-and-visualisation)
11. [Phase 7 — FastAPI Wrapper](#11-phase-7--fastapi-wrapper)
12. [Phase 8 — Evaluation and Benchmarking](#12-phase-8--evaluation-and-benchmarking)
13. [Phase 9 — Tests and Polish](#13-phase-9--tests-and-polish)
14. [Phase 10 — Stretch Goals](#14-phase-10--stretch-goals)
15. [File-by-File Responsibility Matrix](#15-file-by-file-responsibility-matrix)
16. [Algorithm Reference](#16-algorithm-reference)
17. [Feature Engineering Reference](#17-feature-engineering-reference)
18. [Classification Logic Reference](#18-classification-logic-reference)
19. [Dependencies and Install Plan](#19-dependencies-and-install-plan)
20. [Configuration Reference](#20-configuration-reference)
21. [API Endpoint Specification](#21-api-endpoint-specification)
22. [Risk Register](#22-risk-register)
23. [Hackathon Priority Tiers](#23-hackathon-priority-tiers)
24. [Definition of Done](#24-definition-of-done)

---

## 1. Repo Purpose and Boundaries

### What this repo does

`transitlens-ml-core` is the **brain** of the TransitLens system. It receives a raw light curve from `data-pipeline` and returns a complete analysis result. Every number a judge sees — the period, the depth, the confidence score, the class label — is computed here.

It handles:

- Preprocessing and normalising raw light curves
- Running Box Least Squares (BLS) period search to detect transit signals
- Extracting interpretable features from detected signals
- Classifying candidates using rule-based logic (primary) and optionally RF/XGBoost (secondary)
- Computing a calibrated confidence score
- Generating all four diagnostic plots as base64-encoded PNG strings
- Exposing the full pipeline as a FastAPI HTTP endpoint for `transitlens-platform`

### What this repo does NOT do

This repo must never:

- Generate or modify light curve data (that belongs in `data-pipeline`)
- Render a user interface or produce HTML reports (that belongs in `platform`)
- Store results in a database or file system (it is stateless — in, process, out)
- Import anything from `transitlens-platform`

### The one function that matters most

Everything in this repo serves one public interface:

```
analyze_light_curve(time, flux, metadata=None, config=None) → result dict
```

All module boundaries, all phase sequencing, and all testing priorities are organised around making this function correct, fast, and explainable.

### Position in the tri-repo system

```
transitlens-data-pipeline  →  transitlens-ml-core  →  transitlens-platform
       load_light_curve()       analyze_light_curve()     POST /analyze
          (feeds)                    (analyses)               (displays)
```

---

## 2. Folder Structure Reference

```
transitlens-ml-core/
│
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── .gitignore
├── setup.py
├── config.yaml                           ← global pipeline config
├── pipeline.py                           ← analyze_light_curve() — single public entry point
│
├── core/
│   ├── __init__.py
│   ├── preprocess.py                     ← normalise, sigma-clip, gap fill, outlier removal
│   ├── bls_detector.py                   ← BLS period search, peak extraction, grid scan
│   ├── feature_extractor.py              ← all 11 features
│   ├── classifier.py                     ← rule-based logic + optional RF/XGBoost wrapper
│   ├── confidence.py                     ← confidence score calculation
│   ├── plotter.py                        ← raw, cleaned, periodogram, phase-folded plots
│   ├── exceptions.py                     ← custom exception hierarchy
│   └── utils.py                          ← phase folding, sigma clipping, interpolation
│
├── models/
│   ├── rule_config.yaml                  ← all rule-based classifier thresholds
│   ├── rf_model.pkl                      ← trained Random Forest (placeholder)
│   ├── xgb_model.pkl                     ← trained XGBoost (placeholder)
│   ├── feature_scaler.pkl                ← StandardScaler fitted on training data
│   ├── model_card.md                     ← training data, accuracy, known limitations
│   └── .gitkeep
│
├── api/
│   ├── __init__.py
│   ├── app.py                            ← FastAPI app, CORS, startup config
│   ├── routes.py                         ← POST /analyze, GET /health, GET /demo
│   ├── schema.py                         ← Pydantic request and response models
│   └── middleware.py                     ← request logging, error handling, timing
│
├── eval/
│   ├── __init__.py
│   ├── evaluate.py                       ← runs pipeline on labeled dataset
│   ├── metrics.py                        ← precision, recall, F1, period recovery rate
│   ├── benchmark.py                      ← speed benchmarks per light curve
│   └── results/
│       ├── .gitkeep
│       ├── confusion_matrix.png
│       ├── classification_report.txt
│       └── benchmark_summary.csv
│
├── notebooks/
│   ├── bls_exploration.ipynb
│   ├── feature_analysis.ipynb
│   ├── model_training.ipynb
│   └── confidence_tuning.ipynb
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_preprocess.py
    ├── test_bls.py
    ├── test_features.py
    ├── test_classifier.py
    ├── test_confidence.py
    ├── test_plotter.py
    ├── test_pipeline.py
    └── test_api.py
```

---

## 3. The Central Interface Contract

### Input specification

```
analyze_light_curve(
    time:     float[]          # BTJD timestamps, monotonically increasing
    flux:     float[]          # normalised flux, median ≈ 1.0
    metadata: dict | None      # optional — from data-pipeline load_light_curve()
    config:   dict | None      # optional — override any pipeline parameter
) → dict
```

`time` and `flux` must have equal length. All other validation is performed internally.

`metadata` is the metadata sub-dict from `data-pipeline`'s output. It provides optional ground truth (`true_period`, `true_depth`, `true_duration`) for evaluation purposes, but the pipeline must work correctly even when `metadata` is `None`.

### Output specification — the full result dict

```
{
    "target_id":           str              # from metadata or "unknown"
    "candidate_detected":  bool             # true if BLS found a significant peak
    "predicted_class":     str              # "exoplanet_like" | "eclipsing_binary_like"
                                            # | "noise_or_other"
    "confidence":          float            # 0.0 to 1.0
    "period_days":         float | null     # detected period (null if no candidate)
    "duration_days":       float | null     # detected transit duration
    "depth":               float | null     # detected fractional flux drop
    "snr":                 float | null     # signal-to-noise ratio of detection
    "transit_count":       int | null       # number of transits in the time series

    "features": {
        "bls_power":              float     # normalised BLS peak power
        "snr":                    float     # signal-to-noise of best peak
        "period_days":            float     # period at BLS peak
        "duration_days":          float     # duration at BLS peak
        "depth":                  float     # depth at BLS peak
        "transit_count":          int       # floor(time_span / period)
        "odd_even_depth_delta":   float     # |depth_odd - depth_even|
        "v_shape_score":          float     # 0 = flat-bottomed, 1 = fully V-shaped
        "local_noise":            float     # RMS of out-of-transit flux
        "depth_to_noise_ratio":   float     # depth / local_noise
        "phase_shape_kurtosis":   float     # kurtosis of phase-folded in-transit profile
    }

    "explanation":  str     # human-readable classification justification

    "plots": {
        "raw_lightcurve":    str    # base64-encoded PNG
        "cleaned_lightcurve": str   # base64-encoded PNG
        "periodogram":       str    # base64-encoded PNG
        "phase_folded":      str    # base64-encoded PNG
    }

    "processing_time_ms":   float   # total wall-clock time for the analysis
    "pipeline_version":     str     # semver string from config.yaml
}
```

### Invariants that must always hold

- If `candidate_detected` is `false`, all of `period_days`, `duration_days`, `depth`, `snr`, `transit_count` must be `null`
- If `candidate_detected` is `true`, none of those fields may be `null`
- `confidence` is always a float between 0.0 and 1.0 regardless of detection outcome
- `predicted_class` is always one of the three allowed strings — never null, even for noise
- All four `plots` keys are always present — even for noise cases, the raw and periodogram plots are generated
- `explanation` is always a non-empty string

---

## 4. Phase Overview

| Phase | Name | Priority | Estimated Effort | Hackathon Tier |
|-------|------|----------|-----------------|----------------|
| 1 | Preprocessing and Signal Cleaning | Critical | 2–3 hours | Must-have |
| 2 | BLS Transit Detection | Critical | 3–4 hours | Must-have |
| 3 | Feature Extraction | Critical | 2–3 hours | Must-have |
| 4 | Classification and Confidence Scoring | Critical | 2–3 hours | Must-have |
| 5 | Pipeline Orchestration | Critical | 1–2 hours | Must-have |
| 6 | Plotting and Visualisation | High | 2–3 hours | Should-have |
| 7 | FastAPI Wrapper | High | 1–2 hours | Should-have |
| 8 | Evaluation and Benchmarking | Medium | 2–3 hours | Should-have |
| 9 | Tests and Polish | High | 2–3 hours | Should-have |
| 10 | Stretch Goals | Low | Open-ended | Future |

**Build order:** Phase 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10

Phases 1 through 5 must be complete before `transitlens-platform` can show any real results. Phase 6 (plotting) is technically optional for the pipeline but is the single biggest visual impact for judges. Phase 7 (API) is required for the platform to call ml-core over HTTP.

**Critical dependency:** `data-pipeline` Tier 1 must be complete before this repo can be built. ml-core has nothing to analyse until the three synthetic CSVs exist and `load_light_curve()` works.

---

## 5. Phase 1 — Preprocessing and Signal Cleaning

### Goal

Transform a raw normalised light curve into a clean, analysis-ready flux array. The BLS detector is sensitive to outliers and trends — any uncleaned artefacts will produce spurious period detections. This phase must produce a flux array where the only significant deviations from 1.0 are genuine astrophysical signals.

### Deliverables

- `core/preprocess.py` — complete preprocessing module
- `core/utils.py` — shared utility functions used by preprocess and other modules

### Step 1.1 — Define preprocessing responsibilities

`preprocess.py` must perform these operations in this exact order. The order matters because each step depends on the output of the previous one.

**Operation 1: Input validation**

Before doing anything else, validate that `time` and `flux` have equal length, that `time` is monotonically increasing, and that `flux` contains no infinite values. If these checks fail, raise `InvalidInputError` with a message that tells the caller exactly what is wrong. Do not attempt to repair invalid inputs — fail fast and loudly.

**Operation 2: NaN removal**

Remove any NaN values from both `time` and `flux` simultaneously. NaNs appear in real TESS data at cadences flagged by the instrument pipeline (cosmic rays, scattered light, momentum dumps). For synthetic data, NaNs should not exist but the check must be there for robustness. Use mask-based removal to keep `time` and `flux` synchronised.

**Operation 3: Outlier removal (sigma clipping)**

Remove flux values that are more than `sigma_upper` standard deviations above the median or more than `sigma_lower` standard deviations below the median. Default values from `config.yaml`: `sigma_upper = 5.0`, `sigma_lower = 5.0`. Run for `max_iterations = 3` iterations, recomputing the median and standard deviation after each pass. This is the standard astronomical sigma-clipping procedure.

**Why asymmetric sigma bounds matter:** Stellar flares produce extreme upward spikes (flux >> 1.0) that are not transit signals. Transits are downward dips. Setting a tighter upper bound (e.g., `sigma_upper = 3.0`) is sometimes appropriate for flare stars but would remove legitimate signals in other contexts. For the hackathon, symmetric 5-sigma clipping is safe for all three synthetic cases.

**Operation 4: Trend removal (detrending)**

Real TESS light curves contain long-term instrumental systematics — gradual brightness trends over hours to days caused by spacecraft pointing drift, thermal effects, and detector saturation recovery. These trends appear as low-frequency variations in the flux that have nothing to do with transits.

For the hackathon, use a simple polynomial detrending approach: fit a low-degree polynomial (degree 2 or 3) to the out-of-transit flux and divide it out. The polynomial baseline represents the slow trend, and dividing by it removes the trend while preserving the transit signal.

An alternative is running median detrending: compute a running median with a window of `detrend_window_days` (default: 1.5 days) and divide the flux by it. Running median is more robust than polynomial fitting for light curves with gaps or complex systematics.

For synthetic data, detrending should have minimal effect because synthetic light curves have no instrumental trends. The detrending step is included for robustness when real TESS data is added in Phase 3 of `data-pipeline`.

**Operation 5: Re-normalisation**

After outlier removal and detrending, divide the flux by its median to ensure the baseline is exactly 1.0. This step compensates for any small baseline drift introduced by the detrending operation.

**Operation 6: Gap detection**

Identify gaps in the time array where the spacing between consecutive timestamps exceeds `gap_threshold_factor × cadence` (default factor: 5). Record the gap locations and lengths. This information is passed to the BLS detector so it can handle gapped time arrays correctly.

For real TESS data, the standard mid-sector download gap of ~1 day must not be treated as a data quality issue — it is expected and handled by ignoring it in the phase-folding step.

### Step 1.2 — Define `core/utils.py` responsibilities

`utils.py` is a collection of pure mathematical functions used by multiple modules. It has no state and no side effects.

**`phase_fold(time, period, t0)`** — the most-used utility in the entire repo. Given a time array, a period, and a reference epoch `t0`, returns a phase array where values range from -0.5 to 0.5. Phase 0 corresponds to the transit centre. This function is used by the BLS detector, the feature extractor, and the plotter.

**`sigma_clip(values, sigma, max_iter)`** — the core sigma-clipping logic. Returns a boolean mask where `True` means the value is within the sigma bounds. Separated from `preprocess.py` because it is also used during feature extraction to compute local noise statistics.

**`running_median(values, window_size)`** — computes a centred running median. Used in detrending and in the local noise estimator.

**`bin_phase_folded(phase, flux, n_bins)`** — bins a phase-folded light curve into `n_bins` evenly spaced phase bins, returning `(bin_centres, bin_means, bin_stds)`. Used in `plotter.py` for the phase-folded plot and in `feature_extractor.py` for shape statistics.

**`detect_gaps(time, cadence_min, threshold_factor)`** — returns a list of `(start_idx, end_idx, gap_days)` tuples for all detected gaps. Used in preprocessing and to annotate the raw light curve plot.

### Step 1.3 — Define `core/exceptions.py`

All custom exceptions must inherit from `MLCoreError` as the base class. This allows callers to catch all ml-core errors with a single except clause.

The exception hierarchy:

```
MLCoreError
├── InvalidInputError          # bad input arrays (wrong length, NaN, non-monotonic)
├── InsufficientDataError      # not enough data points to run BLS
├── NoCandidateFoundError      # BLS found no significant peak (not an error — normal for noise)
├── PreprocessingError         # preprocessing step failed unexpectedly
├── BLSDetectionError          # BLS algorithm failed (numerical issue)
├── FeatureExtractionError     # feature computation failed
├── ClassificationError        # classifier returned invalid output
└── PlottingError              # matplotlib/plotly rendering failed
```

`NoCandidateFoundError` is different from the others — it is a valid, expected outcome for `noise_or_other` light curves. The pipeline catches this internally and sets `candidate_detected = false` rather than propagating it to the caller.

### Step 1.4 — Minimum data requirements

The preprocessing module must check that the input passes minimum quality gates before proceeding:

- Minimum 500 data points (otherwise BLS cannot search a meaningful period range)
- Time span of at least 5 days (need at least 2 transit events to measure a period)
- At least 80% of points retained after sigma clipping (if more are clipped, the data is too noisy to analyse)

If any gate fails, raise `InsufficientDataError` with a clear message. Do not attempt to run BLS on data that cannot support a valid period search.

### Phase 1 Completion Checklist

- [ ] `preprocess.py` performs all six operations in the correct order
- [ ] NaN removal keeps `time` and `flux` synchronised
- [ ] Sigma clipping uses iterative recomputation with configurable sigma and max_iter
- [ ] Detrending removes low-frequency trends without affecting transit signals
- [ ] Re-normalisation ensures median flux = 1.0 ± 0.0001 after cleaning
- [ ] Gap detection returns a list of gaps with start index, end index, and duration
- [ ] All three minimum data quality gates are checked before returning
- [ ] `utils.py` defines `phase_fold`, `sigma_clip`, `running_median`, `bin_phase_folded`, `detect_gaps`
- [ ] All exceptions inherit from `MLCoreError`
- [ ] Preprocessing the three synthetic cases produces clean flux arrays with visible transit dips

---

## 6. Phase 2 — BLS Transit Detection

### Goal

Find the most significant periodic box-shaped signal in the cleaned light curve. The BLS (Box Least Squares) algorithm is the gold standard for transit detection in radial velocity and photometric data. This is the most scientifically critical module in the entire repo.

### Deliverables

- `core/bls_detector.py` — complete BLS detection module

### Step 2.1 — Understand what BLS does

BLS searches for periodic signals shaped like a box — a sudden drop in brightness, a flat bottom, and a sudden return to baseline. For each candidate period in a grid, it asks: "if I fold the light curve at this period, does a box dip consistently appear at the same phase?"

The BLS power spectrum is a function of period (and optionally phase and duration) that peaks at the period where a box dip is most consistently detected. The period at the peak of the power spectrum is the best estimate of the true transit period.

The key parameters BLS searches over:

- **Period:** the candidate orbital period. Searched over a grid from `period_min` to `period_max`
- **Duration:** the fraction of the period spent in transit. BLS tries multiple duration values at each period
- **Phase:** which fraction of the period corresponds to the transit centre

The output of BLS is:
- The period at the power peak (`best_period`)
- The phase at the power peak (`best_t0`)
- The duration at the power peak (`best_duration`)
- The depth at the power peak (`best_depth`)
- The BLS power spectrum array
- The power value at the peak (`bls_power_peak`)

### Step 2.2 — Implementation choice

For the hackathon, use **Astropy's `BoxLeastSquares`** implementation when available. This is a production-quality, well-tested BLS implementation that handles the period grid, duration grid, and power computation internally. It also computes a false-alarm probability for the peak, which is useful for the SNR calculation.

If astropy is not available (e.g., offline environment without it), implement a simplified BLS using scipy. The simplified version:

1. Creates a period grid using `numpy.linspace` or `numpy.geomspace`
2. For each period, folds the time array to get phases
3. For each phase window (transit duration), computes the average in-transit flux and out-of-transit flux
4. Computes the BLS statistic as `(depth^2 × n_in) / (variance × n_out)` where `n_in` is the number of in-transit points and `n_out` is the number of out-of-transit points
5. Records the power and parameters at the maximum

The astropy version is strongly preferred because it is vectorised, much faster, and handles edge cases correctly.

### Step 2.3 — Period grid design

The period grid must be chosen carefully to balance sensitivity and compute time.

**Period range:**

- `period_min`: 0.5 days (the shortest physically plausible close-in planet period)
- `period_max`: `time_span / 2.0` days (need at least 2 transits to confirm a period — this is the fundamental constraint on transit detection)

For a 27-day TESS sector, this gives a range of 0.5 to 13.5 days.

**Period grid spacing:**

Use a frequency grid with uniform spacing in frequency (`1/period`) rather than uniform spacing in period. This ensures equal sensitivity across all periods — a uniform period grid over-samples long periods and under-samples short periods.

The frequency step size: `df = 1 / (n_oversample × time_span)` where `n_oversample` is an oversampling factor (default: 10). A higher oversampling factor gives finer period resolution at the cost of compute time.

**Duration grid:**

Try multiple transit durations at each period. For the hackathon, use 5 duration values logarithmically spaced between `duration_min = 0.01 days` (about 15 minutes) and `duration_max = 0.5 × period` (half the period — no transit can last longer than half an orbit). Astropy's BLS handles this automatically.

**Compute time estimate:**

For a 27-day light curve with 18000 points and a period grid of ~5000 periods with 5 durations each, astropy BLS takes approximately 0.5–2 seconds on a modern laptop. This is acceptable for the hackathon demo. Document this timing in `README.md` so judges know what to expect during the live demo.

### Step 2.4 — Peak extraction and significance testing

After computing the BLS power spectrum, extract the peak:

**Finding the peak:** Use `numpy.argmax` on the power spectrum to find the period index with the highest power. Record the period, duration, depth, phase, and power at this index.

**Secondary peak check:** Verify that the peak is not an alias. The most common aliasing patterns are:
- Half the true period (2:1 alias) — if the best period is very close to half of some longer period with similar power, the longer period may be the true one
- Double the true period — can occur when alternate transit events are deeper (eclipsing binary signal)

For the hackathon, a simple check suffices: compute the BLS power at `best_period × 2` and `best_period / 2`. If either is within 20% of the peak power, flag the result as potentially aliased and note this in the explanation string.

**Significance threshold:**

A detection is considered significant if the BLS power peak exceeds `bls_power_threshold` (default from config: 0.15 normalised power). Below this threshold, set `candidate_detected = false`.

The SNR calculation: `snr = best_depth / local_noise` where `local_noise` is the RMS of the out-of-transit flux after folding at the best period. SNR above `snr_threshold` (default: 5.0) is required for a detection.

Both conditions — BLS power AND SNR — must be satisfied for `candidate_detected = true`. Either alone is insufficient.

### Step 2.5 — BLS detector output structure

`bls_detector.py` must return an intermediate result dict that is consumed by `feature_extractor.py`:

```
{
    "candidate_detected":  bool
    "best_period":         float | null
    "best_t0":             float | null
    "best_duration":       float | null
    "best_depth":          float | null
    "bls_power_peak":      float
    "snr":                 float
    "power_spectrum": {
        "periods":  float[]     # period grid
        "power":    float[]     # BLS power at each period
    }
    "alias_warning":       bool    # true if possible aliasing detected
}
```

### Step 2.6 — Handling the noise case

For `candidate_c` (noise_or_other), the BLS power spectrum should show no significant peak. The detector should return `candidate_detected = false` with `bls_power_peak` below threshold. This is the expected and correct outcome. The pipeline must handle this cleanly without raising any exception.

The key diagnostic that distinguishes a genuine noise case from a missed detection: in a noise case, the power spectrum is roughly flat with no dominant peak. In a missed detection (signal too weak), the power spectrum has a peak but it is below the threshold. Both result in `candidate_detected = false`, but the explanation string should describe the difference when possible.

### Phase 2 Completion Checklist

- [ ] `bls_detector.py` imports and uses `astropy.timeseries.BoxLeastSquares` when available
- [ ] Fallback to scipy-based BLS when astropy is not available
- [ ] Period grid covers 0.5 to `time_span/2` days with frequency-uniform spacing
- [ ] Duration grid uses 5 logarithmically-spaced values per period
- [ ] BLS correctly detects candidate_a period within 1% of 3.42 days
- [ ] BLS correctly detects candidate_b period within 1% of 1.87 days
- [ ] BLS returns `candidate_detected = false` for candidate_c
- [ ] Significance requires both BLS power > threshold AND SNR > threshold
- [ ] Alias check is performed and `alias_warning` is set correctly
- [ ] Complete BLS power spectrum (periods and powers arrays) is returned for plotting
- [ ] Processing completes in under 5 seconds for an 18000-point light curve

---

## 7. Phase 3 — Feature Extraction

### Goal

Convert the BLS detection result and the phase-folded light curve into a vector of 11 interpretable numerical features. These features are the inputs to the classifier. They must be physically meaningful — each one should correspond to something a human astronomer would look at when deciding whether a signal is a planet, a binary, or noise.

### Deliverables

- `core/feature_extractor.py` — all 11 features computed

### Step 3.1 — The eleven features

Every feature must be computable from the combination of `(time, flux_cleaned, bls_result)`. The feature extractor takes these three inputs and returns a flat dict of 11 floats.

---

**Feature 1: `bls_power` (float, 0 to 1)**

The normalised BLS peak power from `bls_result["bls_power_peak"]`. This is the single most important feature — it measures how consistently the box dip appears across all folded transits. High BLS power means many transit events all have similar depth and phase. Low BLS power means the signal is inconsistent or absent.

Range interpretation:
- 0.0 to 0.05: no significant detection
- 0.05 to 0.15: marginal, likely noise
- 0.15 to 0.40: moderate, warrants investigation
- Above 0.40: strong detection, likely astrophysical

---

**Feature 2: `snr` (float, ≥ 0)**

Signal-to-noise ratio of the transit detection. Computed as `best_depth / local_noise` where `local_noise` is the RMS of the out-of-transit flux in the phase-folded light curve.

SNR interpretation:
- Below 5: sub-threshold, unreliable detection
- 5 to 10: marginal, worth investigating
- Above 10: confident detection
- Above 20: high-quality detection (Candidate A has SNR ≈ 21.4)

---

**Feature 3: `period_days` (float)**

The best-fit period from BLS. While this is a measurement rather than a classification feature, it is included in the feature vector because very short periods (<0.5 days) or very long periods (>10 days) carry information about the likely signal type. Extremely short periods are physically implausible for most exoplanets but common for eclipsing binaries.

---

**Feature 4: `duration_days` (float)**

The best-fit transit duration from BLS. A transit duration that is an implausibly large fraction of the period (e.g., duration > 0.15 × period) is a red flag that the signal is not a grazing transit or a very elongated orbit — it may be an eclipsing binary secondary eclipse or a blended signal.

---

**Feature 5: `depth` (float)**

The fractional flux drop at the transit centre. This is the single most powerful feature for separating exoplanets from eclipsing binaries:
- Exoplanet transits: typically 0.001 to 0.030 (0.1% to 3%)
- Eclipsing binaries: typically 0.050 to 0.600 (5% to 60%)
- Grey zone 0.030 to 0.050: ambiguous, use other features

The physical reason: no planet can be large enough to block more than about 3% of a Sun-like star's light. A depth greater than 3% almost certainly means a stellar-sized companion.

---

**Feature 6: `transit_count` (int)**

The number of individual transit events expected in the time series: `floor(time_span_days / period_days)`. More transits mean the BLS period estimate is more reliable. Fewer than 2 transits means the period cannot be confirmed from this light curve alone. For the hackathon demo:
- Candidate A (period 3.42 days, 27-day span): ~7 transits
- Candidate B (period 1.87 days): ~14 transits

---

**Feature 7: `odd_even_depth_delta` (float)**

The absolute difference between the average depth of odd-numbered transits and even-numbered transits: `|depth_odd - depth_even|`.

This is the most powerful single discriminator between exoplanets and eclipsing binaries:

- **Exoplanet:** all transits are caused by the same planet passing in front of the same star. Odd and even transits should have identical depth. `odd_even_depth_delta ≈ 0`
- **Eclipsing binary:** odd transits are the primary eclipse (one star in front of the other) and even transits are the secondary eclipse (reversed geometry). These often have different depths because the two stars have different temperatures and sizes. `odd_even_depth_delta > 0.02` is a strong eclipsing binary indicator

To compute this: phase-fold the light curve at the best period, separate transit events by index (1st, 2nd, 3rd... = odd/even), compute the in-transit median flux for each, then take the absolute difference of the odd mean and even mean.

If `transit_count < 4`, this feature cannot be computed reliably — set it to 0 and do not use it as a classifier input.

---

**Feature 8: `v_shape_score` (float, 0 to 1)**

A measure of how V-shaped the transit profile is. 0 means a perfectly flat-bottomed box (consistent with a planet with a small radius ratio). 1 means a perfectly V-shaped profile (consistent with a grazing eclipse or an eclipsing binary).

How to compute it: Phase-fold the light curve at the best period. Extract the in-transit points. Fit two models to the in-transit profile:
- Model A: flat box (the expected depth, constant across the transit window)
- Model B: linear V-shape (deepest at phase 0, linearly increasing toward ingress/egress)

Compute the residuals of each model. The `v_shape_score` is the fraction of the total variance explained by Model B relative to Model A. A value near 0 means the flat box fits better (exoplanet). A value near 1 means the V-shape fits better (eclipsing binary).

For the hackathon, a simpler approximation: compute the ratio `depth_at_phase_centre / depth_at_quarter_transit`. For a flat-bottomed transit this ratio is near 1.0. For a V-shaped transit this ratio is near 2.0 (deeper at centre than at quarter-transit). Normalise to 0–1 range.

---

**Feature 9: `local_noise` (float)**

The RMS (root mean square) scatter of the out-of-transit flux in the phase-folded light curve. This quantifies the intrinsic noise level of the light curve after all preprocessing. A lower local noise means the transit signal stands out more clearly from the background.

Computation: Phase-fold at best period. Identify out-of-transit points (phase outside the transit window by at least `1.5 × duration`). Compute RMS of these points minus 1.0.

---

**Feature 10: `depth_to_noise_ratio` (float)**

Closely related to SNR: `best_depth / local_noise`. This is the transit detection significance expressed in units of the local noise. While similar to Feature 2 (SNR), this version uses the local phase-folded noise rather than the global noise estimate, making it more sensitive to local systematics around the transit.

In the rule-based classifier, `depth_to_noise_ratio > 6.0` is required for a confident detection.

---

**Feature 11: `phase_shape_kurtosis` (float)**

The excess kurtosis of the in-transit flux distribution in the phase-folded light curve. Kurtosis measures how "peaky" or "flat" a distribution is relative to a Gaussian.

- A flat-bottomed transit has a roughly uniform in-transit flux distribution → low kurtosis
- A V-shaped or spiky transit has most in-transit points near the ingress/egress with a sharp minimum at phase 0 → high kurtosis
- Noise has a Gaussian-like distribution → kurtosis near 0

Use `scipy.stats.kurtosis` on the in-transit portion of the binned phase-folded light curve.

### Step 3.2 — Handling the no-candidate case

If `bls_result["candidate_detected"] == false`, the feature extractor still runs but uses the BLS spectrum's best (sub-threshold) peak to compute what features it can. Features that require phase-folded data (`odd_even_depth_delta`, `v_shape_score`, `phase_shape_kurtosis`) are set to 0 and flagged as unreliable. `bls_power` and `snr` are computed from the sub-threshold peak. `transit_count` is set to 0.

This allows the classifier to receive a complete feature vector even for noise cases, which is important for the optional RF/XGBoost path.

### Step 3.3 — Feature validation

Before returning, the extractor must check that all 11 features are finite (not NaN or inf). If any feature fails this check, replace it with a sensible default (0 for scores, -1 for unreliable indicators) and log a warning. Never return NaN in the feature dict — the classifier must always receive a complete vector.

### Phase 3 Completion Checklist

- [ ] All 11 features are computed for candidate_a (exoplanet case)
- [ ] All 11 features are computed for candidate_b (eclipsing binary case)
- [ ] All 11 features are computed for candidate_c (noise case) with appropriate fallback values
- [ ] `odd_even_depth_delta` correctly shows near-zero for candidate_a and non-zero for candidate_b
- [ ] `v_shape_score` correctly shows near-zero for candidate_a and non-zero for candidate_b
- [ ] `depth` correctly shows ~0.013 for candidate_a and ~0.18 for candidate_b
- [ ] No NaN or inf values in any returned feature dict
- [ ] Feature computation adds less than 0.5 seconds to total processing time
- [ ] Feature dict has exactly the 11 keys specified — no more, no fewer

---

## 8. Phase 4 — Classification and Confidence Scoring

### Goal

Take the 11-feature vector and return a predicted class label and a calibrated confidence score. The classification system has two layers: a rule-based primary classifier (always active) and an optional ML classifier (RF or XGBoost, active only if trained models are available). The rule-based classifier is the foundation — it must produce correct results on the three synthetic cases without any training data.

### Deliverables

- `core/classifier.py` — rule-based logic and optional ML wrapper
- `core/confidence.py` — confidence score calculation
- `models/rule_config.yaml` — all decision thresholds

### Step 4.1 — Rule-based classifier design

The rule-based classifier implements a decision tree of logical conditions. The thresholds in these conditions are stored in `models/rule_config.yaml` — not hardcoded. This is important because judges may ask "how did you tune these?" and the answer is "they are configurable, here is the file."

**Stage 1: Detection gate**

If `candidate_detected == false` (BLS power below threshold and/or SNR below threshold):
→ Classify as `noise_or_other` immediately. Skip Stages 2 and 3.

This is the correct handling for Candidate C.

**Stage 2: Depth threshold (primary discriminator)**

If `depth > depth_threshold_eb` (default: 0.050, i.e., 5%):
→ Candidate is `eclipsing_binary_like`. The physical reasoning is unambiguous: no transiting planet produces a depth greater than about 3% around a Sun-like star.

This catches the majority of eclipsing binaries (Candidate B has depth 18%).

**Stage 3: Secondary discriminators (within the planet-like depth range)**

If `depth ≤ 0.050`, apply additional checks to distinguish true exoplanet-like signals from grazing eclipsing binaries and other contaminants:

Check 3a — Odd/even depth delta:
- If `odd_even_depth_delta > odd_even_threshold` (default: 0.02): classify as `eclipsing_binary_like`
- Reason: an odd/even depth difference is the clearest possible sign of a binary system with two different eclipse depths

Check 3b — V-shape score:
- If `v_shape_score > v_shape_threshold` (default: 0.40): classify as `eclipsing_binary_like`
- Reason: exoplanet transits are flat-bottomed; V-shaped transits indicate a stellar companion

Check 3c — Depth-to-noise ratio:
- If `depth_to_noise_ratio < depth_snr_threshold` (default: 6.0): classify as `noise_or_other`
- Reason: even if BLS detects a signal, if it's too close to the noise floor it's unreliable

If all checks 3a, 3b, 3c pass:
→ Classify as `exoplanet_like`

**Decision tree summary:**

```
candidate_detected == false       →  noise_or_other
depth > 0.050                     →  eclipsing_binary_like
odd_even_depth_delta > 0.020      →  eclipsing_binary_like
v_shape_score > 0.40              →  eclipsing_binary_like
depth_to_noise_ratio < 6.0        →  noise_or_other
all checks pass                   →  exoplanet_like
```

### Step 4.2 — Optional ML classifier (RF/XGBoost)

When `models/rf_model.pkl` or `models/xgb_model.pkl` exists (trained models from the `notebooks/model_training.ipynb` process), the classifier can optionally use them as a second opinion.

The ML path:

1. Scale the 11-feature vector using `models/feature_scaler.pkl` (the StandardScaler fitted on the training data)
2. Call `model.predict_proba(scaled_features)` to get class probabilities
3. The predicted class is `numpy.argmax(probabilities)`
4. The confidence is `max(probabilities)`

**How to combine rule-based and ML predictions:**

For the hackathon, use the following priority:
- If both rule-based and ML agree: use that class with the ML confidence
- If they disagree: use the rule-based prediction (it is more interpretable and explainable to judges), but note the disagreement in the `explanation` string
- If ML model not available: use rule-based only (this is the default hackathon state)

This approach is conservative and honest — we prioritise explainability over black-box accuracy.

### Step 4.3 — `models/rule_config.yaml` specification

All decision thresholds must be stored here:

```yaml
detection:
  bls_power_threshold: 0.15
  snr_threshold: 5.0

classification:
  depth_threshold_eb: 0.050
  odd_even_threshold: 0.020
  v_shape_threshold: 0.40
  depth_snr_threshold: 6.0

ml_classifier:
  enabled: false          # set to true when trained models exist
  model_type: "rf"        # "rf" or "xgb"
  blend_weight: 0.0       # 0.0 = rule-based only, 1.0 = ML only, 0.5 = equal blend
```

The `ml_classifier.enabled: false` default ensures the hackathon demo always uses the explainable rule-based path unless a trained model is explicitly available.

### Step 4.4 — Confidence score design

The confidence score is not simply the ML probability output. It is a calibrated composite score that reflects how convincingly the features support the predicted class. It must be interpretable — if a judge asks "why is the confidence 88%?", there must be a clear answer.

**Confidence components:**

Each component contributes a partial score from 0 to 1. The final confidence is a weighted average of all applicable components.

For `exoplanet_like` predictions:

| Component | What it measures | Weight | Full score condition |
|-----------|-----------------|--------|---------------------|
| BLS power | Strength of periodic signal | 0.30 | bls_power > 0.40 |
| Depth SNR | Signal clearly above noise | 0.25 | depth_to_noise_ratio > 10.0 |
| Odd/even consistency | Transits are identical | 0.20 | odd_even_depth_delta < 0.010 |
| V-shape absence | Flat-bottomed transit | 0.15 | v_shape_score < 0.15 |
| Transit count | Multiple events confirmed | 0.10 | transit_count ≥ 5 |

For `eclipsing_binary_like` predictions:

| Component | What it measures | Weight | Full score condition |
|-----------|-----------------|--------|---------------------|
| Depth magnitude | Clearly stellar depth | 0.35 | depth > 0.10 |
| BLS power | Strong periodic signal | 0.25 | bls_power > 0.40 |
| V-shape presence | V-shaped profile | 0.20 | v_shape_score > 0.50 |
| Odd/even asymmetry | Depth alternation | 0.20 | odd_even_depth_delta > 0.040 |

For `noise_or_other` predictions:

| Component | What it measures | Weight | Full score condition |
|-----------|-----------------|--------|---------------------|
| Low BLS power | No significant peak | 0.50 | bls_power < 0.05 |
| Low SNR | Below detection threshold | 0.30 | snr < 3.0 |
| Low depth SNR | Consistent with noise | 0.20 | depth_to_noise_ratio < 3.0 |

**Partial scoring:** Components that do not fully satisfy their condition receive a fractional score proportional to how close they are to the threshold. This produces smooth confidence values rather than abrupt jumps between e.g. 90% and 30%.

### Step 4.5 — Explanation string generation

The `explanation` field in the result dict is a human-readable, one-paragraph summary of why the classifier made its decision. It must:

1. State the predicted class and confidence
2. Name the top 2-3 features that drove the decision
3. Give the feature values with plain-language interpretation
4. Note any caveats (alias warning, low transit count, disagreement with ML classifier)

The explanation is generated programmatically by filling a template string with the actual feature values. It must never be a generic placeholder like "analysis complete" — judges will read it and it needs to be specific.

**Example explanation for Candidate A:**

"Classified as exoplanet_like with 88% confidence. A periodic transit signal was detected at 3.42 days with a depth of 1.3% (21.4σ above noise). The depth is consistent with a sub-Jupiter-sized planet. Odd/even transit depths are nearly identical (delta = 0.04), ruling out an eclipsing binary. The transit profile is flat-bottomed (V-shape score = 0.12), consistent with a planetary disc crossing a stellar surface. 7 transit events were detected across the 27-day observation window."

### Phase 4 Completion Checklist

- [ ] Rule-based classifier correctly classifies all three synthetic cases
- [ ] Decision thresholds are all stored in `rule_config.yaml` — none hardcoded in Python
- [ ] `classifier.py` reads `rule_config.yaml` at startup (not at every call)
- [ ] ML classifier path exists and gracefully skips when `.pkl` files are absent
- [ ] Confidence score is a float between 0.0 and 1.0 for all three cases
- [ ] Candidate A confidence > 0.80 (strong exoplanet signal)
- [ ] Candidate B confidence > 0.80 (strong EB signal)
- [ ] Candidate C confidence > 0.70 (clear noise case)
- [ ] `explanation` string is specific, names the driving features, gives actual values
- [ ] Changing a threshold in `rule_config.yaml` changes the classification without touching Python code

---

## 9. Phase 5 — Pipeline Orchestration

### Goal

Wire Phases 1 through 4 into a single callable function: `analyze_light_curve()`. This function is the only thing ml-core exports to the outside world. Everything else is internal.

### Deliverables

- `pipeline.py` — the public entry point

### Step 5.1 — `pipeline.py` structure

`pipeline.py` lives at the repo root, not inside `core/`. This placement is deliberate — it is the face of the repo. When someone opens the repo and wants to understand what it does, this is the first file they read.

**What `pipeline.py` does:**

1. Load config from `config.yaml` (with optional override from the `config` parameter)
2. Record the start time for `processing_time_ms`
3. Call `preprocess.clean(time, flux)` → `(time_clean, flux_clean)`
4. Call `bls_detector.detect(time_clean, flux_clean, config)` → `bls_result`
5. Call `feature_extractor.extract(time_clean, flux_clean, bls_result)` → `features`
6. Call `classifier.classify(features, config)` → `(predicted_class, rule_path)`
7. Call `confidence.score(features, predicted_class, config)` → `confidence_float`
8. Call `plotter.generate_all(time, flux, time_clean, flux_clean, bls_result)` → `plots`
9. Build the `explanation` string
10. Assemble and return the complete result dict
11. Record and store `processing_time_ms`

### Step 5.2 — Error handling strategy

The pipeline must never raise an unhandled exception to the caller. All internal exceptions are caught, logged, and converted to a graceful result dict with `candidate_detected = false` and an `explanation` describing what went wrong.

This means: if BLS crashes on a particularly pathological light curve, the caller gets a clean JSON response explaining the failure, not a Python stack trace.

**The only exception to this rule:** `InvalidInputError` — if the input `time` and `flux` arrays are fundamentally broken (different lengths, all NaN, etc.), raise this to the caller because there is nothing the pipeline can do.

### Step 5.3 — Config loading and override

The config is loaded once from `config.yaml` at module level (not at every call). The optional `config` parameter allows overriding any top-level key. The merge strategy: start with the file config, then update with the override dict. Nested dicts are merged recursively.

This allows platform to pass `config={"bls": {"period_max": 5.0}}` to restrict the period search range for a specific use case, without needing to modify `config.yaml`.

### Step 5.4 — Timing and versioning

Record the wall-clock start time before calling `preprocess` and the end time after assembling the result dict. Store the difference in milliseconds as `processing_time_ms`.

Read the `pipeline_version` from `config.yaml`. This should be a semver string (e.g., `"0.1.0"` for the hackathon submission). The version is included in every result dict so platform can log which version of ml-core produced a given result.

### Phase 5 Completion Checklist

- [ ] `analyze_light_curve("candidate_a_time", "candidate_a_flux")` returns a complete result dict
- [ ] All 5 pipeline stages execute in order without errors
- [ ] Total processing time is under 10 seconds for an 18000-point light curve (target: under 5 seconds)
- [ ] Error handling converts internal exceptions to graceful result dicts
- [ ] `InvalidInputError` is propagated to the caller for fundamentally broken inputs
- [ ] Config override merges correctly at the top level and nested levels
- [ ] `processing_time_ms` is recorded accurately
- [ ] `pipeline_version` is read from `config.yaml`
- [ ] The result dict passes all invariant checks from Section 3

---

## 10. Phase 6 — Plotting and Visualisation

### Goal

Generate all four diagnostic plots as base64-encoded PNG strings. These plots are what judges will remember. The phase-folded plot in particular is the single most visually impactful output of the entire project.

### Deliverables

- `core/plotter.py` — all four plot generators

### Step 6.1 — Plot 1: Raw light curve

**What it shows:** The raw, unprocessed flux versus time.

**Visual design requirements:**

- X-axis: time in BTJD, labelled "Time (BTJD)"
- Y-axis: normalised flux, labelled "Normalised Flux"
- Data: blue dots or thin blue line (whichever is readable at 18000 points — dots at full resolution become solid blobs; use a thin line or binned representation)
- Background: dark grey or white depending on the Matplotlib style chosen
- If gaps were detected, mark them with vertical dashed lines in a lighter colour
- Title: "Raw Light Curve — {target_id}"
- Font size: large enough to read as a thumbnail

**Downsampling for speed:** 18000 points plotted individually is slow in Matplotlib. Downsample to 2000 points for plotting (keeping every 9th point) for the raw curve. This is invisible to the eye at normal figure sizes.

### Step 6.2 — Plot 2: Cleaned light curve

**What it shows:** The preprocessed flux after sigma clipping and detrending.

**Visual design:**

- Same axes as Plot 1
- Data: same blue line or dots
- Overplot: mark the in-transit windows at the detected period (if `candidate_detected == true`) as shaded green/teal boxes. This makes the transit timing visually clear — a judge can count the dips and see that they are evenly spaced.
- Title: "Cleaned Light Curve — {target_id}"

The difference between Plot 1 and Plot 2 should be visible for real TESS data (trend removal, outlier removal). For synthetic data, the difference will be subtle because the synthetic data has no instrumental trends. Note this in the `explanation`.

### Step 6.3 — Plot 3: BLS Periodogram

**What it shows:** The BLS power spectrum as a function of period.

**Why this is important to judges:** The periodogram is the direct evidence that a signal was detected. A sharp, isolated peak at a specific period is compelling. A flat or noisy periodogram confirms there is no periodic signal.

**Visual design:**

- X-axis: period in days on a logarithmic scale, labelled "Period (days)"
- Y-axis: BLS power, labelled "BLS Power"
- Data: solid grey line showing the full power spectrum
- Highlight: a vertical red dashed line at the best period with a label showing the period value
- If `alias_warning` is true: mark the harmonic periods (×2 and ÷2) with orange dashed lines
- Title: "BLS Periodogram — Best Period: {period:.4f} days"

The logarithmic period axis is standard in astronomy because transit detection sensitivity falls off at longer periods. A logarithmic axis shows the full search range without compressing the interesting short-period region.

### Step 6.4 — Plot 4: Phase-folded light curve

**This is the most important plot in the entire project.** When a judge sees a clean, symmetric dip perfectly centred at phase zero with a flat baseline on either side — that is the moment of visual proof that the AI found a transit.

**What it shows:** The flux folded at the best period so all transit events align at phase 0.

**Visual design requirements:**

- X-axis: orbital phase from -0.5 to 0.5, labelled "Orbital Phase"
- Y-axis: normalised flux, labelled "Normalised Flux"
- Raw data: light grey dots showing all individual phase-folded points
- Binned model: a thick dark line showing the average flux in each phase bin (use `utils.bin_phase_folded()` with 100 bins)
- Transit window: a shaded purple rectangle behind the dip, spanning the detected duration. This makes the transit region visually obvious even for shallow transits.
- The dip must be centred at phase 0 by construction (this is controlled by the `t0` parameter in the BLS result)
- Title: "Phase-folded at P = {period:.4f} days, Depth = {depth:.4f}"

**For the noise case:** Still generate this plot using the sub-threshold BLS peak. The result should be a flat, featureless phase-folded curve — which is itself evidence of no transit signal.

### Step 6.5 — Output format

All four plots are returned as base64-encoded PNG strings. The encoding process:

1. Render the Matplotlib figure to a BytesIO buffer using `savefig(buffer, format='png', dpi=100, bbox_inches='tight')`
2. Encode the buffer contents as base64: `base64.b64encode(buffer.getvalue()).decode('utf-8')`
3. Return the encoded string

The platform receives these strings and can either embed them directly in HTML (`<img src="data:image/png;base64,{string}">`) or decode them for display in Streamlit (`PIL.Image.open(io.BytesIO(base64.b64decode(string)))`).

**DPI and size:** Use `dpi=100` and a figure size of `(10, 4)` inches (1000×400 pixels). This is large enough for details to be visible but small enough to transmit quickly over the API.

**Speed:** Matplotlib figure generation should take under 1 second per plot. Total plotting time for all four plots should be under 4 seconds. If performance is an issue, use `Agg` backend (non-interactive) which is faster than GUI backends.

### Phase 6 Completion Checklist

- [ ] `plotter.py` generates all four plots without errors for all three synthetic cases
- [ ] All plots are returned as non-empty base64-encoded PNG strings
- [ ] Raw light curve plot shows the time series correctly
- [ ] Cleaned light curve plot shows in-transit windows as shaded regions for detected candidates
- [ ] Periodogram shows a clear peak for candidate_a and candidate_b
- [ ] Periodogram shows a flat/noisy spectrum for candidate_c
- [ ] Phase-folded plot shows a clear symmetric dip for candidate_a centred at phase 0
- [ ] Phase-folded plot shows a clear V-shaped dip for candidate_b
- [ ] Phase-folded plot shows a flat curve for candidate_c
- [ ] Total plotting time for all four plots is under 4 seconds
- [ ] Decoded PNG images open correctly in a web browser

---

## 11. Phase 7 — FastAPI Wrapper

### Goal

Expose `analyze_light_curve()` as an HTTP API so `transitlens-platform` can call it from a different process (or machine). This decouples the science from the display.

### Deliverables

- `api/app.py` — FastAPI application
- `api/routes.py` — endpoint definitions
- `api/schema.py` — Pydantic input/output models
- `api/middleware.py` — logging and error handling

### Step 7.1 — Endpoint specification

**`GET /health`**

Returns a simple status check. Used by `api_client.py` in the platform to verify that ml-core is running before attempting an analysis.

Response:
```json
{
    "status": "ok",
    "version": "0.1.0",
    "timestamp": "2026-01-01T00:00:00Z"
}
```

**`POST /analyze`**

The main endpoint. Accepts a light curve and returns the full analysis result.

Request body:
```json
{
    "time":       [float, ...],
    "flux":       [float, ...],
    "target_id":  "string",
    "metadata":   { ... } | null,
    "config":     { ... } | null
}
```

Response body: the full result dict from `analyze_light_curve()`.

Validation: The Pydantic model must enforce that `time` and `flux` are non-empty lists of the same length. Pydantic raises a 422 Unprocessable Entity response automatically if this validation fails — no manual checking needed in the route handler.

**`GET /demo/{candidate_id}`**

Convenience endpoint for the hackathon demo. Accepts `candidate_id` as `"a"`, `"b"`, or `"c"`, loads the corresponding synthetic case using `data-pipeline`'s `interface.load_light_curve()`, runs the analysis, and returns the result. This lets the platform demo the system without uploading any data.

**`GET /docs`**

FastAPI generates Swagger/OpenAPI documentation automatically. This should be enabled in production. Judges can open `localhost:8000/docs` and see the full API schema without reading any code.

### Step 7.2 — `api/schema.py` design

Use Pydantic v2 for schema definitions. Key models:

**`AnalyzeRequest`** — validates the input. Must enforce:
- `time` and `flux` are lists of floats
- `len(time) == len(flux)` — use a Pydantic validator
- `len(time) >= 100` — minimum viable light curve
- `target_id` defaults to `"unknown"` if not provided

**`AnalyzeResponse`** — mirrors the result dict structure from Section 3. All fields must be typed. Optional fields use `Optional[float]` with default `None`.

**`FeaturesSchema`** — the nested features sub-dict with all 11 feature fields typed as `float`.

**`PlotsSchema`** — the nested plots sub-dict with all four plot fields typed as `str` (base64 PNG).

### Step 7.3 — CORS configuration

The platform runs on a different port (default: 8501 for Streamlit). FastAPI's CORS middleware must allow cross-origin requests from the platform's origin. For the hackathon, allow all origins (`"*"`) to avoid configuration complexity.

### Step 7.4 — `api/middleware.py` design

**Request timing middleware:** Record the start time at the beginning of each request and add a `X-Processing-Time-Ms` header to the response. This is useful for debugging and shows up in browser developer tools.

**Error handling middleware:** Catch any unhandled exception and return a 500 response with a JSON body explaining the error. This prevents the API from returning raw Python stack traces to the platform.

**Request logging:** Log each request's method, path, status code, and processing time to stdout. Use Python's standard `logging` module. The format: `2026-01-01 00:00:00 POST /analyze 200 1234ms`

### Step 7.5 — Running the API

The API is started with:

```
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Document this command prominently in `README.md`. The `--reload` flag enables hot reload during development (restart on file change). Remove it for production.

For the hackathon demo, run both the API and the Streamlit platform simultaneously in two terminal windows.

### Phase 7 Completion Checklist

- [ ] `GET /health` returns `{"status": "ok"}` within 100ms
- [ ] `POST /analyze` with candidate_a data returns the full result dict
- [ ] Pydantic validation rejects requests where `len(time) != len(flux)` with a 422 response
- [ ] `GET /demo/a`, `/demo/b`, `/demo/c` all return complete result dicts
- [ ] `GET /docs` opens Swagger UI with the full API schema
- [ ] CORS is configured to allow requests from any origin
- [ ] All requests are logged with method, path, status code, and timing
- [ ] Unhandled exceptions return a 500 JSON response, not a Python stack trace
- [ ] The API starts successfully with `uvicorn api.app:app --port 8000`

---

## 12. Phase 8 — Evaluation and Benchmarking

### Goal

Quantify how well the pipeline performs on the labeled dataset from `data-pipeline`. This section produces the numbers that appear in the project presentation and that judges will ask about.

### Deliverables

- `eval/evaluate.py` — main evaluation runner
- `eval/metrics.py` — metric computation functions
- `eval/benchmark.py` — speed benchmarks
- `eval/results/` — output files

### Step 8.1 — `eval/evaluate.py`

Loads the `labeled_dataset.csv` from `data-pipeline`, groups by `target_id`, runs `analyze_light_curve()` on each target, and compares the predicted class and detected period against the ground truth.

**Outputs:**

1. A classification report (printed to stdout and saved to `eval/results/classification_report.txt`) showing precision, recall, and F1 for each class
2. A confusion matrix saved to `eval/results/confusion_matrix.png`
3. A per-target result CSV saved to `eval/results/per_target_results.csv` showing for each target: target_id, true_label, predicted_label, true_period, detected_period, period_recovery_error_pct, confidence

### Step 8.2 — `eval/metrics.py`

**Period recovery rate:** The fraction of targets with a known `true_period` where the detected period is within `period_tolerance_pct` (default: 1%) of the true period. Report this separately from classification accuracy because a correct classification can still have a wrong period estimate.

**Classification accuracy per class:** Standard precision (what fraction of predicted X are truly X) and recall (what fraction of true X were predicted as X) for each of the three classes.

**Mean confidence per correct/incorrect prediction:** This quantifies whether the confidence score is calibrated — correct predictions should have higher average confidence than incorrect predictions.

### Step 8.3 — `eval/benchmark.py`

Runs the full `analyze_light_curve()` pipeline on each synthetic case 10 times and reports:

- Mean processing time
- Standard deviation of processing time
- 95th percentile processing time (the worst-case scenario)
- Time breakdown by stage (preprocessing, BLS, features, classification, plotting)

The time breakdown requires temporarily instrumenting each stage call with `time.perf_counter()`. This is for the benchmark only — the production pipeline should not have this overhead.

**Target benchmarks for the hackathon:**

| Stage | Target time |
|-------|------------|
| Preprocessing | < 0.1 s |
| BLS detection | < 2.0 s |
| Feature extraction | < 0.5 s |
| Classification + confidence | < 0.1 s |
| Plotting (all 4) | < 2.0 s |
| **Total** | **< 5.0 s** |

### Phase 8 Completion Checklist

- [ ] `evaluate.py` runs on all three synthetic cases and prints a classification report
- [ ] All three cases are classified correctly (accuracy = 100% on synthetic data)
- [ ] Period recovery rate is 100% for candidate_a and candidate_b
- [ ] `confusion_matrix.png` is generated and saved
- [ ] `per_target_results.csv` is written with correct fields
- [ ] `benchmark.py` reports mean processing time under 5 seconds
- [ ] The classification report is honest — it reports on synthetic data only and says so

---

## 13. Phase 9 — Tests and Polish

### Goal

Write a complete test suite, harden the code, and complete all documentation. This is what distinguishes a professional-quality submission from a prototype.

### Step 9.1 — `tests/conftest.py`

Shared pytest fixtures:

- `tiny_lc_clean`: a minimal preprocessed light curve with 500 points and no transit — for fast unit tests
- `tiny_lc_transit`: a minimal preprocessed light curve with 500 points and a clearly injected transit — for testing BLS and feature extractor
- `candidate_a_result`: the full output of `analyze_light_curve()` on candidate_a — for testing the result dict structure
- `candidate_b_result`: same for candidate_b
- `candidate_c_result`: same for candidate_c
- `mock_bls_result`: a pre-constructed BLS result dict for testing the feature extractor in isolation
- `rule_config`: the parsed `rule_config.yaml` as a Python dict

### Step 9.2 — `tests/test_preprocess.py`

- Output has no NaN values regardless of NaN presence in input
- Output flux has median 1.0 ± 0.001
- Sigma clipping removes points beyond 5σ
- Time array is monotonically increasing in output
- `InsufficientDataError` is raised for inputs with fewer than 500 points
- Gap detection correctly identifies a 2-day gap in a simulated light curve

### Step 9.3 — `tests/test_bls.py`

- Detects the correct period (within 1%) for a synthetic transit with known period
- Returns `candidate_detected = false` for a pure noise light curve
- Power spectrum has a peak at the correct period
- Period grid covers the correct range
- BLS result dict has all required keys
- Processing completes in under 5 seconds for an 18000-point array

### Step 9.4 — `tests/test_features.py`

- Returns exactly 11 features in the feature dict
- No NaN or inf values in any feature
- `depth` feature matches the BLS result depth within 1%
- `odd_even_depth_delta` is near zero for a symmetric transit
- `v_shape_score` is near zero for a box-shaped transit and near one for a V-shaped transit
- All features are floats (not ints, not strings)

### Step 9.5 — `tests/test_classifier.py`

- Classifies a feature vector with depth 0.013, low odd/even delta, low v-shape as `exoplanet_like`
- Classifies a feature vector with depth 0.18 as `eclipsing_binary_like`
- Classifies a feature vector with low BLS power as `noise_or_other`
- Changing `depth_threshold_eb` in the rule config changes the classification boundary
- ML classifier path returns `exoplanet_like` when model files are absent (falls back to rule-based)

### Step 9.6 — `tests/test_pipeline.py`

- `analyze_light_curve()` returns a dict with all required keys for all three synthetic cases
- All three cases are classified correctly
- `candidate_detected` is `false` for candidate_c
- Result dict passes all invariants from Section 3
- Processing time is under 10 seconds for all three cases
- Error handling: passing arrays of different lengths raises `InvalidInputError`
- Error handling: passing all-NaN flux raises `InsufficientDataError` (after NaN removal, no data remains)

### Step 9.7 — `tests/test_api.py`

Use FastAPI's `TestClient` for these tests (no running server required).

- `GET /health` returns 200 with `{"status": "ok"}`
- `POST /analyze` with valid candidate_a data returns 200 with complete result dict
- `POST /analyze` with mismatched array lengths returns 422
- `GET /demo/a` returns 200 with complete result dict
- `GET /demo/invalid` returns 404

### Step 9.8 — Complete `models/model_card.md`

The model card must document:

1. **Model type:** Rule-based classifier with optional RF/XGBoost
2. **Training data:** Synthetic TESS-like light curves (n=3 for hackathon, expandable)
3. **Features used:** List all 11 features with descriptions
4. **Rule thresholds:** All thresholds from `rule_config.yaml` with justification
5. **Performance:** Classification accuracy on the labeled dataset (100% on synthetic cases)
6. **Known limitations:**
   - Trained/tuned on synthetic data only — real TESS performance may vary
   - Rule thresholds are conservative — may miss shallow or short-duration transits
   - Only three target classes — blend and stellar variability are classified as noise_or_other
7. **How to improve:** Add more labeled real TESS cases, tune thresholds on real data, train RF/XGBoost on expanded dataset

### Phase 9 Completion Checklist

- [ ] `pytest tests/` runs with zero failures and zero warnings
- [ ] At least 6 tests per test file
- [ ] All three pipeline end-to-end tests pass
- [ ] API tests pass without a running server
- [ ] `model_card.md` is complete with all 7 sections
- [ ] `README.md` has a working quick-start in under 5 commands
- [ ] `CONTRIBUTING.md` explains the module boundaries and how to add a new feature
- [ ] `config.yaml` is fully documented with comments explaining each parameter

---

## 14. Phase 10 — Stretch Goals

These are post-hackathon enhancements.

### 10.1 — CNN-based transit classifier

Train a 1D Convolutional Neural Network on the phase-folded light curve directly, bypassing the feature extraction step. Input: 100-point binned phase curve. Output: class probabilities for 3 classes. Architecture suggestion: two 1D convolution layers with ReLU activation, global average pooling, two fully-connected layers, softmax output.

### 10.2 — LSTM sequence model

Process the raw cleaned light curve as a time sequence using an LSTM. No phase folding required — the LSTM learns to detect periodic patterns in the raw sequence. More general than the BLS approach but requires significantly more training data.

### 10.3 — Multi-sector analysis

If multiple TESS sectors are available for a target, combine them before running BLS. Increases the time baseline dramatically — a 270-day light curve can detect planets with periods up to 135 days. Requires sector stitching (normalising each sector separately before concatenating).

### 10.4 — Secondary eclipse search

After detecting the primary transit, search for a secondary eclipse at phase 0.5. A detected secondary at half the primary depth is a strong eclipsing binary indicator. New feature added to `feature_extractor.py` and a new rule in the classifier.

### 10.5 — Centroid analysis

For real TESS data, a flux centroid that shifts during the transit indicates the signal may originate from a background star. Add a `centroid_shift_score` feature using centroid pixel data — a powerful false-positive discriminator.

### 10.6 — Confidence calibration

Use Platt scaling or isotonic regression to calibrate confidence scores against actual accuracy on a held-out test set. Calibrated probabilities are more trustworthy — a 90% confidence score should correspond to 90% accuracy.

---

## 15. File-by-File Responsibility Matrix

| File | Owned by | Input | Output | Used by |
|------|----------|-------|--------|---------|
| `pipeline.py` | ml-core public | time[], flux[], metadata, config | full result dict | api/routes, platform, tests |
| `core/preprocess.py` | ml-core internal | time[], flux[] | time_clean[], flux_clean[] | pipeline |
| `core/bls_detector.py` | ml-core internal | time_clean[], flux_clean[], config | bls_result dict | pipeline, feature_extractor |
| `core/feature_extractor.py` | ml-core internal | time_clean[], flux_clean[], bls_result | features dict (11 keys) | pipeline, classifier |
| `core/classifier.py` | ml-core internal | features dict, config | (predicted_class, rule_path) | pipeline |
| `core/confidence.py` | ml-core internal | features dict, predicted_class, config | float 0-1 | pipeline |
| `core/plotter.py` | ml-core internal | time[], flux[], time_clean[], flux_clean[], bls_result | plots dict (4 base64 strings) | pipeline |
| `core/utils.py` | ml-core internal | arrays, scalars | arrays, scalars | preprocess, bls, features, plotter |
| `core/exceptions.py` | ml-core internal | — | — | all core modules |
| `models/rule_config.yaml` | ml-core config | — | thresholds | classifier, confidence |
| `models/*.pkl` | ml-core trained | — | — | classifier (optional) |
| `models/model_card.md` | ml-core docs | — | — | judges, README |
| `api/app.py` | ml-core API | — | FastAPI app | uvicorn |
| `api/routes.py` | ml-core API | HTTP request | HTTP response | app |
| `api/schema.py` | ml-core API | — | Pydantic models | routes |
| `api/middleware.py` | ml-core API | — | — | app |
| `eval/evaluate.py` | ml-core eval | labeled_dataset.csv | metrics, CSV, PNG | standalone script |
| `eval/metrics.py` | ml-core eval | predictions, labels | precision/recall/F1 | evaluate |
| `eval/benchmark.py` | ml-core eval | — | timing CSV | standalone script |
| `tests/` | ml-core tests | fixtures | pass/fail | pytest |

---

## 16. Algorithm Reference

### Box Least Squares (BLS)

BLS was introduced by Kovacs, Zucker & Mazeh (2002) specifically for detecting planetary transit signals in photometric time series. It is the standard algorithm used by NASA's Kepler and TESS mission pipelines.

The core idea: for a grid of candidate periods P, fold the light curve to get a phase array. Slide a box (defined by phase centre and width q = duration/period) across the phase axis. For each position, compute the weighted average in-transit flux and out-of-transit flux. The BLS statistic is maximised when the box exactly captures the transit signal.

Why BLS and not Lomb-Scargle: Lomb-Scargle is designed for sinusoidal signals. Transit signals are box-shaped. BLS has much higher sensitivity for transit detection.

Astropy implementation: `astropy.timeseries.BoxLeastSquares` is a fully vectorised implementation using NumPy broadcasting. It also provides a `false_alarm_probability` method.

### Sigma Clipping

Iterative outlier rejection: compute median and standard deviation, remove points beyond k-sigma, repeat. Using the median makes the procedure robust to the very outliers being removed.

### Phase Folding

Given time array t, period P, reference epoch t0:

```
phase = ((t - t0) / P) mod 1.0
```

Subtracting 0.5 re-centres to [-0.5, 0.5) where phase 0 is the transit centre. All transits collapse to the same phase, allowing the average transit signal to be measured even when individual transits are below the noise threshold.

---

## 17. Feature Engineering Reference

### Why these 11 features

Selected to satisfy three criteria: physical interpretability (each feature corresponds to something a human astronomer examines), discriminative power (each separates at least two classes), and computability from BLS output (all derivable without additional period search algorithms).

### Feature correlations

`snr` and `depth_to_noise_ratio` are highly correlated but use different noise estimates (global vs local). Both are included because their difference can indicate non-uniform noise clustering around the transit phase.

`depth` and `transit_count` are anti-correlated in practice — deeper transits tend to have shorter periods and thus more transits per sector. This is physically correct.

---

## 18. Classification Logic Reference

### Why rule-based first, ML second

Rule-based classifiers are explainable (every decision has a clear reason), data-efficient (no training data required), robust (not overfitted to a small dataset), and debuggable (the specific failing rule is immediately identifiable).

ML classifiers are more accurate with sufficient labeled data and better at non-linear decision boundaries, but harder to explain to judges.

The hybrid approach is professionally correct: start with physics-based rules, upgrade with ML when data supports it. For the hackathon, rule-based is the right default.

### Why three classes

The three-class vocabulary covers the three most common transit-like signal types in TESS data. They correspond to natural clusters in feature space with clear boundaries. "noise_or_other" is scientifically honest — not every light curve contains a detectable signal.

---

## 19. Dependencies and Install Plan

### Production dependencies

| Package | Version | Why needed | Phase introduced |
|---------|---------|-----------|-----------------|
| `numpy` | >= 1.24 | Array operations throughout | Phase 1 |
| `scipy` | >= 1.11 | Statistics, sigma clipping, BLS fallback | Phase 1 |
| `astropy` | >= 5.3 | BLS implementation (primary) | Phase 2 |
| `scikit-learn` | >= 1.3 | RF classifier, StandardScaler, metrics | Phase 4 |
| `xgboost` | >= 2.0 | XGBoost classifier (optional) | Phase 4 |
| `matplotlib` | >= 3.7 | All four diagnostic plots | Phase 6 |
| `fastapi` | >= 0.110 | HTTP API wrapper | Phase 7 |
| `uvicorn` | >= 0.27 | ASGI server for FastAPI | Phase 7 |
| `pydantic` | >= 2.5 | Request/response validation | Phase 7 |
| `pyyaml` | >= 6.0 | Reading config files | Phase 1 |

### Install strategy for hackathon

Minimal install (Phases 1-5, no plotting, no API):

```
pip install numpy scipy astropy scikit-learn pyyaml
```

Standard install (all phases):

```
pip install numpy scipy astropy scikit-learn xgboost matplotlib fastapi uvicorn pydantic pyyaml
```

---

## 20. Configuration Reference

### `config.yaml` — complete specification

```yaml
version: "0.1.0"

preprocessing:
  sigma_upper: 5.0
  sigma_lower: 5.0
  max_sigma_iter: 3
  detrend_method: "running_median"
  detrend_window_days: 1.5
  detrend_poly_degree: 2
  gap_threshold_factor: 5.0
  min_points: 500
  min_time_span_days: 5.0
  min_fraction_retained: 0.80

bls:
  period_min_days: 0.5
  period_max_days: null
  n_oversample: 10
  n_durations: 5
  duration_min_days: 0.01
  duration_max_fraction: 0.5
  bls_power_threshold: 0.15
  snr_threshold: 5.0
  alias_check_tolerance: 0.20

features:
  phase_bins: 100
  odd_even_min_transits: 4
  noise_exclusion_factor: 1.5

classification:
  depth_threshold_eb: 0.050
  odd_even_threshold: 0.020
  v_shape_threshold: 0.40
  depth_snr_threshold: 6.0

ml_classifier:
  enabled: false
  model_type: "rf"
  blend_weight: 0.0

plotting:
  dpi: 100
  figure_width: 10
  figure_height: 4
  downsample_points: 2000
  phase_bins: 100
  transit_shade_alpha: 0.15
  style: "seaborn-v0_8-whitegrid"

api:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
  log_level: "info"
```

---

## 21. API Endpoint Specification

### `GET /health`

Response 200:
```json
{"status": "ok", "version": "0.1.0", "timestamp": "2026-01-01T00:00:00Z"}
```

### `POST /analyze`

Request body: `{"time": [...], "flux": [...], "target_id": "...", "metadata": {...}, "config": null}`

Response 200: Full result dict as specified in Section 3

Response 422: Pydantic validation failure (array length mismatch, wrong types)

Response 500: Internal pipeline error with JSON body

### `GET /demo/{candidate_id}`

Path parameter: `candidate_id` = `"a"` | `"b"` | `"c"`

Response 200: Full result dict for the selected synthetic case

Response 404: Unknown candidate_id

### `GET /docs`

Swagger UI auto-generated by FastAPI. Judges can open `localhost:8000/docs` and see the full API schema.

---

## 22. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| BLS fails to recover period within 1% tolerance | Low | High | Verify period recovery on all three cases before demo; adjust n_oversample if needed |
| Plotting too slow for live demo | Medium | Medium | Pre-generate plots for demo cases and cache them |
| FastAPI port conflict at hackathon venue | Low | Medium | Document alternative port config; allow platform to specify ml-core URL |
| Confidence scores seem arbitrary to judges | Medium | Medium | Document the weighting scheme; show calculation in explanation string |
| Rule thresholds wrong for real TESS data | High | Medium | Flag in model_card.md; real TESS tuning is stretch goal |
| Classifier wrong for one of the three demo cases | Low | High | Run full pipeline on all three before submitting and verify classes |
| astropy BLS raises exception on gapped time array | Low | Medium | Test with simulated gapped light curve in test_bls.py; implement scipy fallback |
| matplotlib rendering fails in headless server | Medium | Medium | Set `matplotlib.use('Agg')` at top of plotter.py; test in headless environment |
| Memory error for very long light curves | Low | Low | Implement downsampling in preprocessing for inputs > 50000 points |

---

## 23. Hackathon Priority Tiers

### Tier 1 — Must-have (complete before platform can show results)

- `core/preprocess.py` — clean the three synthetic cases without errors
- `core/bls_detector.py` — detect correct periods for candidate_a and candidate_b
- `core/feature_extractor.py` — return 11 valid features for all three cases
- `core/classifier.py` — correctly classify all three cases
- `core/confidence.py` — return a sensible confidence float for all three cases
- `pipeline.py` — `analyze_light_curve()` returns a valid result dict
- `core/exceptions.py` — exception hierarchy in place
- `core/utils.py` — phase_fold, sigma_clip, bin_phase_folded implemented
- `models/rule_config.yaml` — all thresholds defined

**Estimated effort: 10-15 hours. Everything else builds on this.**

### Tier 2 — Should-have (adds major visual and demo value)

- `core/plotter.py` — all four plots as base64 PNG
- `api/app.py`, `api/routes.py`, `api/schema.py` — FastAPI wrapper working
- `eval/evaluate.py` — classification report generated
- `tests/test_pipeline.py` — end-to-end tests passing
- `models/model_card.md` — complete for judge questions

**Estimated effort: 6-8 hours.**

### Tier 3 — Nice-to-have (professional quality signal)

- `eval/benchmark.py` — timing numbers documented
- `api/middleware.py` — request logging and error handling
- Full test suite with all 10 test files
- `notebooks/bls_exploration.ipynb` — shows BLS parameter tuning
- `config.yaml` — fully documented with comments

**Estimated effort: 3-4 hours.**

### Tier 4 — Stretch (post-hackathon)

Everything in Phase 10 — CNN classifier, LSTM, multi-sector, centroid analysis, confidence calibration.

---

## 24. Definition of Done

The `transitlens-ml-core` repo is considered complete for hackathon submission when:

1. `analyze_light_curve(time, flux)` called with candidate_a data returns `predicted_class == "exoplanet_like"`, `candidate_detected == true`, and detected period within 1% of 3.42 days
2. Same function with candidate_b returns `predicted_class == "eclipsing_binary_like"` with `candidate_detected == true`
3. Same function with candidate_c returns `candidate_detected == false` and `predicted_class == "noise_or_other"`
4. All four plot keys in `result["plots"]` are non-empty base64 strings that decode to valid PNG images
5. `confidence` is between 0.0 and 1.0 for all three cases
6. `explanation` is a specific, non-empty string that mentions the key features driving the decision
7. `pytest tests/` runs with zero failures
8. `uvicorn api.app:app --port 8000` starts without errors and `GET /health` returns 200
9. `POST /analyze` with candidate_a data returns 200 with a complete result dict in under 10 seconds
10. `processing_time_ms` is under 10000 for all three cases (target: under 5000)
11. `models/rule_config.yaml` contains all thresholds — none hardcoded in Python files
12. No import from `transitlens-platform` exists anywhere in the repo
13. `README.md` contains a working quick-start in under 5 commands
14. `models/model_card.md` documents training data, features, performance, and limitations

---

*This document covers the complete engineering plan for `transitlens-ml-core`. No code is included. All algorithm choices, feature definitions, classification rules, API contracts, and evaluation metrics are documented here for use during the hackathon build.*

*Previous document: `transitlens-data-pipeline-PLAN.md`*
*Next document: `transitlens-platform-PLAN.md`*