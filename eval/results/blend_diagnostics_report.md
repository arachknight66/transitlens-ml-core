# TransitLens Blend & Contamination Diagnostics Performance Report

## 1. Executive Summary
- **Diagnostic Availability Rate**: 33.3%
- **Centroid Availability Rate**: 33.3%
- **Crowding Availability Rate**: 0.0%
- **Neighbor Availability Rate**: 0.0%
- **Blend Classification Precision**: 0.0%
- **Blend Classification Recall**: 0.0%
- **Blend Classification F1-Score**: 0.0000
- **False Blend Flag Rate on Clean Planets**: 50.0%

## 2. Confusion Matrix (Blend Contamination Slice)
| | Predicted Blend | Predicted Non-Blend |
|---|---|---|
| **True Blend** | 0 (TP) | 0 (FN) |
| **True Non-Blend** | 0 (FP) | 3 (TN) |

## 3. False Blend Flags
Listed below are the clean exoplanet transit targets that were flagged with high blend risk or classified as blend:

| target_id | true_label | predicted_class | centroid_shift | centroid_shift_significance | crowding_metric | blend_risk_level | blend_evidence_flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TIC-261136679 | exoplanet_transit | stellar_variability_or_other | 0.04205875 | 16.3212 | None | high | centroid_shift_16.3sigma,strong_centroid_displacement |
