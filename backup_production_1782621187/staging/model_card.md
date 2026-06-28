# Model Card: Retrained TransitLens Classifier

## Model Details
- **Model Type**: CalibratedXgboost
- **Training Date**: 2026-06-28 10:02:59
- **Evidence Level**: `restricted`
- **Sufficiency Notes**: Class 'exoplanet_transit' train count 3 < 20; Class 'exoplanet_transit' val count 1 < 5; Class 'exoplanet_transit' test count 1 < 10; Class 'eclipsing_binary' train count 0 < 20; Class 'eclipsing_binary' val count 1 < 5; Class 'eclipsing_binary' test count 0 < 10; Class 'blend_contamination' train count 0 < 20; Class 'blend_contamination' val count 0 < 5; Class 'blend_contamination' test count 0 < 10; Class 'stellar_variability_or_other' train count 19 < 20; Class 'stellar_variability_or_other' val count 2 < 5; Class 'stellar_variability_or_other' test count 1 < 10; Class 'review_required' train count 0 < 20; Class 'review_required' val count 0 < 5; Class 'review_required' test count 0 < 10
- **Production Eligible**: `False`

## Parameters & Training Metadata
- **Label Mode**: `four_class`
- **Trained Classes**: ('exoplanet_transit', 'eclipsing_binary', 'blend_contamination', 'stellar_variability_or_other', 'review_required')
- **Random Seed**: 42
- **Label Policy Version**: `1.0.0`
- **Aperture Photometry Version**: `connected_threshold_v1.0`
- **Cutout Size**: 15x15 pixels
- **Archive Hashes (TOI)**: `7302f42b6af38f7aa1266f0cb314e11f54b0cb08b8468e09160e3794ada69f5f`
- **Archive Hashes (TCE)**: `d432b93dccd9759965a998df1eb225e3fb71de66510617ac85e461fa931c4a17`

## Performance Metrics (Test Split)
- **Accuracy**: 50.0000%
- **Macro F1 Score**: 0.1333
- **Expected Calibration Error (ECE)**: 0.1890
- **Brier Score**: 0.6929

### Per-Class Performance Summary

| Class Label | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| exoplanet_transit | 0.0000 | 0.0000 | 0.0000 | 1 |
| eclipsing_binary | 0.0000 | 0.0000 | 0.0000 | 0 |
| blend_contamination | 0.0000 | 0.0000 | 0.0000 | 0 |
| stellar_variability_or_other | 0.5000 | 1.0000 | 0.6667 | 1 |
| review_required | 0.0000 | 0.0000 | 0.0000 | 0 |

## Dataset Partition Sizes
- **Train Samples**: 22
- **Validation Samples**: 4
- **Test Samples**: 2

## Scientific Warnings & Limitations
- **Offline Vetted Retraining**: This model has been explicitly retrained offline with vetted archives.
- **No Continual Self-Training**: Model predictions are never automatically injected back into the training catalog.
- **Trained Sectors**: [1, 29, 47, 65, 78, 87, 98]