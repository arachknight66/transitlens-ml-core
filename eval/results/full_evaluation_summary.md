# TransitLens Scientific Performance Evaluation Summary

## 1. Executive Summary
- **Overall Period Recovery Rate**: 50.00% (tolerance < 1.0%)
- **Validation Split Classification Accuracy**: 100.00%
- **Blind Test Split Classification Accuracy**: 100.00%
- **Gold Target Set Accuracy**: 100.00%
- **Average Pipeline Execution Latency**: 4372.2 ms per target

## 2. Classification Performance (Test Split)
| Class Label | Precision | Recall | F1-Score |
|---|---|---|---|
| exoplanet_transit | 100.0% | 100.0% | 100.0% |
| eclipsing_binary | 0.0% | 0.0% | 0.0% |
| blend_contamination | 0.0% | 0.0% | 0.0% |
| stellar_variability_or_other | 0.0% | 0.0% | 0.0% |

## 3. Parameter Estimation Accuracy
- **Mean Period Error**: 0.0167%
- **Mean Transit Depth Error**: 2.77%
- **Mean Transit Duration Error**: 30.00%

*Parameter errors are computed relative to synthetic/archive catalogue ground truth.*
