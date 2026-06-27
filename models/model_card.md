# Model Card: TransitLens ML Classifier

## Model Overview
- **Model Type**: Sigmoid-Calibrated Random Forest Classifier (via `CalibratedClassifierCV`)
- **Wrapped In**: `TransitLensClassifier` (see `core/classifier.py`)
- **Training Date**: 2026-06-26
- **Evidence Level**: `sufficient` — all four classes meet minimum sample thresholds (≥20 train, ≥10 test per class)

## Intended Use
Classification of light curve transit-like signals detected by the BLS periodogram into one of four astrophysical categories:

| Class Label | Physical Meaning |
|:---|:---|
| `exoplanet_transit` | Signal consistent with a transiting exoplanet |
| `eclipsing_binary` | Signal consistent with an eclipsing binary star |
| `blend_contamination` | Signal from a nearby contaminating source (diluted/blended) |
| `stellar_variability_or_other` | No credible transit signal; noise, systematics, or intrinsic variability |

## Feature Schema
The model uses 16 canonical physically-motivated features defined in [`feature_schema.md`](feature_schema.md) and ordered in [`feature_order.json`](feature_order.json). Feature extraction is performed by `core/feature_extractor.py`.

**No label-leaked features are used.** Target identifiers, catalog dispositions, and ground-truth parameters are excluded from the feature vector.

## Training Data
- **Train Samples**: 604 (target-disjoint from validation and test)
- **Validation Samples**: 202
- **Test Samples**: 201
- **Split Strategy**: Target-disjoint; no target appears in multiple splits
- **Source**: Synthetic light curves generated from catalog parameters (periods, depths, durations) with TESS-realistic noise models and cadence

### Per-Class Distribution (Test Split)

| Class | Support |
|:---|:---|
| `exoplanet_transit` | 51 |
| `eclipsing_binary` | 50 |
| `blend_contamination` | 50 |
| `stellar_variability_or_other` | 50 |

## Performance Metrics (Test Split)

| Metric | Value |
|:---|:---|
| **Accuracy** | 100.00% |
| **Macro F1 Score** | 1.0000 |
| **Expected Calibration Error (ECE)** | 0.0195 |
| **Brier Score** | 0.0005 |

### Per-Class Performance

| Class Label | Precision | Recall | F1-Score | Support |
|:---|:---|:---|:---|:---|
| `exoplanet_transit` | 1.0000 | 1.0000 | 1.0000 | 51 |
| `eclipsing_binary` | 1.0000 | 1.0000 | 1.0000 | 50 |
| `blend_contamination` | 1.0000 | 1.0000 | 1.0000 | 50 |
| `stellar_variability_or_other` | 1.0000 | 1.0000 | 1.0000 | 50 |

## Probability Calibration
- **Method**: Sigmoid (Platt scaling) via `sklearn.calibration.CalibratedClassifierCV` on 3-fold CV
- **ECE**: 0.0195 — well-calibrated for the synthetic evaluation distribution
- **Brier Score**: 0.0005 — excellent sharpness

The `TransitLensClassifier` wrapper supports `calibrated=True` (default, uses calibrated model) and `calibrated=False` (unwraps to base estimator) for diagnostic purposes.

## Pipeline Integration
The classifier is invoked by `core/classifier.py` via the `classify()` function:
1. **Rule-based decision tree** always runs first (provides interpretability and a fallback)
2. **ML classifier** runs when `ml_classifier.enabled = true` in `rule_config.yaml`
3. **Disagreement handling**: Configurable via `use_rule_fallback_on_disagreement`
4. **Fallback**: If ML model files are missing and `dev_fallback = true`, the rule-based result is used

## Warnings & Limitations

### Synthetic-Only Training
> **CRITICAL**: This model is trained and evaluated entirely on **synthetic light curves** with stochastically simulated noise and blend features. Performance on real TESS/Kepler photometry has NOT been validated. The 100% test accuracy reflects the separability of the synthetic distribution, NOT guaranteed performance on real astronomical data.

### Small Dataset
The training set contains 604 samples total (~151 per class). This is statistically adequate for the feature dimensionality (16) but far below the thousands-to-millions scale typical of production astronomical classifiers (e.g., Robovetter, Astronet).

### Known Failure Modes
- **Subtle blends with high crowding**: If `crowding_metric` is near 1.0 but the signal is still diluted, the blend class may be missed
- **Period aliases**: The classifier operates on features from the best BLS period; if the period is aliased (e.g., half-period for EBs), classification features may be distorted. The alias resolver (`core/alias_resolver.py`) mitigates this but does not guarantee alias-free inputs.
- **Out-of-distribution signals**: Signals not well-represented in the training set (e.g., heartbeat stars, disintegrating planets, circumbinary transits) will be classified into the nearest known class

### Calibration Caveat
ECE and Brier scores are computed on the same synthetic distribution. Calibration may degrade on real photometry with different noise characteristics, systematics, or class imbalance.

## Reproducibility
```bash
# Generate feature matrices (requires transitlens-data-pipeline datasets)
python prepare_ml.py

# Train and calibrate
python train_model.py

# Evaluate
python -m eval.run_full_evaluation
```

## Files
| File | Description |
|:---|:---|
| `final_classifier.pkl` | Serialized `TransitLensClassifier` wrapper with calibrated RF model |
| `final_feature_scaler.pkl` | `StandardScaler` fitted on training features |
| `final_feature_order.json` | Canonical feature order used during training |
| `final_label_mapping.json` | Integer-to-class-label mapping |
| `training_metadata.json` | Training run metadata, dataset sizes, and metrics |
| `feature_schema.md` | Feature definitions and physical interpretations |
| `rule_config.yaml` | All rule-based thresholds and ML classifier configuration |
