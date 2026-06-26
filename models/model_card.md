# Model Card: TransitLens ML Classifier

## Model Details
- **Model Type**: Sigmoid-Calibrated RandomForest Classifier
- **Training Date**: 2026-06-26 23:55:37
- **Evidence Level**: `sufficient`
- **Sufficiency Notes**: All classes satisfy target sizes.

## Intended Use
The model is designed to classify light curve transiting events into one of four categories: `exoplanet_transit`, `eclipsing_binary`, `blend_contamination`, and `stellar_variability_or_other`.

## Performance Metrics (Test Split)
- **Accuracy**: 100.0000%
- **Macro F1 Score**: 1.0000
- **Expected Calibration Error (ECE)**: 0.0195
- **Brier Score**: 0.0005

### Per-Class Performance Summary

| Class Label | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| exoplanet_transit | 1.0000 | 1.0000 | 1.0000 | 51 |
| eclipsing_binary | 1.0000 | 1.0000 | 1.0000 | 50 |
| blend_contamination | 1.0000 | 1.0000 | 1.0000 | 50 |
| stellar_variability_or_other | 1.0000 | 1.0000 | 1.0000 | 50 |

## Training & Split Distribution
- **Train Samples**: 604
- **Validation Samples**: 202
- **Test Samples**: 201

## Warnings & Limitations
