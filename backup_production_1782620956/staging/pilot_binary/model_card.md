# Model Card: Retrained TransitLens Classifier

## Model Details
- **Model Type**: CalibratedRandomForest
- **Training Date**: 2026-06-27 19:24:52
- **Evidence Level**: `restricted`
- **Sufficiency Notes**: Class 'exoplanet_transit' train count 4 < 20; Class 'exoplanet_transit' val count 4 < 5; Class 'exoplanet_transit' test count 0 < 10; Class 'stellar_variability_or_other' train count 3 < 20; Class 'stellar_variability_or_other' val count 2 < 5; Class 'stellar_variability_or_other' test count 2 < 10
- **Production Eligible**: `False`

## Parameters & Training Metadata
- **Label Mode**: `binary`
- **Trained Classes**: ['exoplanet_transit', 'stellar_variability_or_other']
- **Random Seed**: 42
- **Label Policy Version**: `1.0.0`
- **Aperture Photometry Version**: `connected_threshold_v1.0`
- **Cutout Size**: 15x15 pixels
- **Archive Hashes (TOI)**: `7302f42b6af38f7aa1266f0cb314e11f54b0cb08b8468e09160e3794ada69f5f`
- **Archive Hashes (TCE)**: `d432b93dccd9759965a998df1eb225e3fb71de66510617ac85e461fa931c4a17`

## Performance Metrics (Test Split)
- **Accuracy**: 0.0000%
- **Macro F1 Score**: 0.0000
- **Expected Calibration Error (ECE)**: 0.7216
- **Brier Score**: 1.0465

### Per-Class Performance Summary

| Class Label | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| exoplanet_transit | 0.0000 | 0.0000 | 0.0000 | 0 |
| stellar_variability_or_other | 0.0000 | 0.0000 | 0.0000 | 2 |

## Dataset Partition Sizes
- **Train Samples**: 7
- **Validation Samples**: 6
- **Test Samples**: 2

## Scientific Warnings & Limitations
- **Offline Vetted Retraining**: This model has been explicitly retrained offline with vetted archives.
- **No Continual Self-Training**: Model predictions are never automatically injected back into the training catalog.
- **Trained Sectors**: [2, 4, 6, 7, 14, 16, 19, 44]