# Model Card: Retrained TransitLens Classifier

## Model Details
- **Model Type**: Calibrated RandomForest Classifier (Sigmoid Platt Scaling)
- **Training Date**: 2026-06-27 18:19:22
- **Evidence Level**: `restricted`
- **Sufficiency Notes**: exoplanet_transit train count 3 < 20; exoplanet_transit test count 0 < 10

## Parameters & Training Metadata
- **Random Seed**: 42
- **Label Policy Version**: `1.0.0`
- **Aperture Photometry Version**: `connected_threshold_v1.0`
- **Cutout Size**: 15x15 pixels
- **Archive Hashes (TOI)**: `7302f42b6af38f7aa1266f0cb314e11f54b0cb08b8468e09160e3794ada69f5f`
- **Archive Hashes (TCE)**: `d432b93dccd9759965a998df1eb225e3fb71de66510617ac85e461fa931c4a17`

## Performance Metrics (Test Split)
- **Accuracy**: 0.0000%
- **Macro F1 Score**: 0.0000
- **Expected Calibration Error (ECE)**: 0.0000
- **Brier Score**: 0.0000

### Per-Class Performance Summary

| Class Label | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| exoplanet_transit | 0.0000 | 0.0000 | 0.0000 | 0 |
| eclipsing_binary | 0.0000 | 0.0000 | 0.0000 | 0 |
| blend_contamination | 0.0000 | 0.0000 | 0.0000 | 0 |
| stellar_variability_or_other | 0.0000 | 0.0000 | 0.0000 | 0 |

## Dataset Partition Sizes
- **Train Samples**: 3
- **Validation Samples**: 1
- **Test Samples**: 0

## Scientific Warnings & Limitations
- **Offline Vetted Retraining**: This model has been explicitly retrained offline with vetted archives.
- **No Continual Self-Training**: Model predictions are never automatically injected back into the training catalog.
- **Class Exclusions**: Eclipsing Binary and Blend Contamination classes are intentionally empty for this TESS run due to a lack of vetted TESS EB/blend catalogs. They will consistently output 0.0% probability during inference.