# TransitLens Scientific Performance Evaluation Summary

## 1. Executive Summary
- **Overall Period Recovery Rate**: 33.33% (tolerance < 1.0%)
- **Validation Split Classification Accuracy**: 50.00%
- **Blind Test Split Classification Accuracy**: 50.00%
- **Average Pipeline Execution Latency**: 7534.1 ms per target


## 2. Classification Performance (Test Split)
| Class Label | Precision | Recall | F1-Score |
|---|---|---|---|
| exoplanet_transit | 100.0% | 100.0% | 100.0% |
| eclipsing_binary | 0.0% | 0.0% | 0.0% |
| blend_contamination | 0.0% | 0.0% | 0.0% |
| stellar_variability_or_other | 0.0% | 0.0% | 0.0% |

## 3. Parameter Estimation Accuracy
- **Mean Period Error**: 0.0035%
- **Mean Transit Depth Error**: 2.78%
- **Mean Transit Duration Error**: 30.00%

*Parameter errors are computed relative to synthetic/archive catalogue ground truth.*

## 4. Phase 4 Injection-Recovery Summary (Synthetic Evidence Only)

> Evidence type: Synthetic injection-recovery benchmark. NOT real-TESS evidence.
> Run `python -m eval.run_injection_recovery --mode standard` for a full benchmark.

- **Detection Recall (all SNR)**: (not run - use --injection to enable)
- **Detection Recall (SNR >= 7)**: (not run - use --injection to enable)
- **Period Recovery +/- 1% (all)**: (not run - use --injection to enable)
- **Period Recovery +/- 1% (SNR >= 7)**: (not run - use --injection to enable)
- **False-Positive Rate (controls)**: (not run - use --injection to enable)

See `eval/results/phase4_injection_recovery_report.md` for the full Phase 4 report.
