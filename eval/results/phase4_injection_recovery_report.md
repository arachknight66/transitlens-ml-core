# TransitLens Phase 4: Injection-Recovery Report

> **EVIDENCE LEVEL**: Phase 4 (Injection-Recovery Benchmark)
> **Evidence Type**: Synthetic injection on simulated light curves.
> **NOT real-TESS-sector evidence.** Do not conflate with Level 4 sector screening.
> All metrics are on synthetic data with simulated noise models.

---

## 1. Run Configuration

| Parameter | Value |
|-----------|-------|
| Mode | **quick** |
| Random Seed | 42 |
| Total Trials | 100 |
| Injection Trials | 75 |
| Control Trials | 25 |
| Injection Failures | 0 |
| Control Failures | 0 |
| Period Grid | [0.75, 1.5, 3.5, 7.0, 12.0] |
| Depth Grid | [0.0003, 0.001, 0.003, 0.005, 0.015] |
| Duration Grid | [0.06, 0.1, 0.15, 0.25] |
| Noise Grid | [0.0005, 0.001, 0.002] |
| Variability Modes | ['none', 'sinusoidal', 'quasi_periodic'] |
| Gap Modes | ['none', 'random_gaps', 'tess_downlink_gap'] |
| Dilution Factors | [1.0, 0.75, 0.5] |
| Cadence | 2.0 min |
| Time Span | 27.0 days |
| Mean Runtime/Trial | 785.0 ms |

---

## 2. Overall Results

| Metric | All Injections | High-SNR (≥7) |
|--------|---------------|---------------|
| N Trials | 75 | 65 |
| Detection Recall | 57.3% | 63.1% |
| Period Recovery ±1% | 53.3% | 58.5% |
| Period Recovery ±5% | 57.3% | 63.1% |
| Median Period Error | 0.171 % | — |
| Median Depth Error | 20.313 % | — |
| Median Duration Error | 16.667 % | — |
| FP Rate (Controls) | 64.0% | — |

**Strict targets for SNR ≥ 7 (required for 95+ score):**
- Detection Recall ≥ 90%: ❌ FAIL (63.1%)
- Period Recovery ±1% ≥ 90%: ❌ FAIL (58.5%)
- Period Recovery ±5% ≥ 95%: ❌ FAIL (63.1%)
- FP Rate Controls < 15%: ❌ FAIL (64.0%)

---

## 3. Results by SNR Bin

| SNR Bin | N | Detection Recall | Period Recovery 1% | Period Recovery 5% | Median Period Err% | Median Depth Err% | Median Dur Err% |
|---------|---|-----------------|-------------------|-------------------|-------------------|------------------|----------------|
| [0, 3) | 3 | 0.0% | 0.0% | 0.0% | N/A | N/A | N/A |
| [3, 5) | 5 | 20.0% | 20.0% | 20.0% | 0.042 | 7.120 | 16.667 |
| [5, 7) | 2 | 50.0% | 50.0% | 50.0% | 0.042 | 11.998 | 12.000 |
| [7, 10) | 2 | 50.0% | 50.0% | 50.0% | 25.048 | 33.350 | 6.000 |
| [10, 15) | 5 | 80.0% | 80.0% | 80.0% | 0.042 | 5.984 | 25.333 |
| [15, 20) | 6 | 83.3% | 66.7% | 83.3% | 0.096 | 14.673 | 14.333 |
| [20, 999) | 51 | 58.8% | 54.9% | 58.8% | 0.171 | 22.129 | 16.667 |


---

## 4. Results by Depth and Period

### Detection Recall by Injected Depth

| Depth (ppm) | N | Detection Recall | Period Recovery 1% | Median Period Err% |
|-------------|---|-----------------|-------------------|-------------------|
| 300 | 10 | 40.0% | 40.0% | 0.042 |
| 1000 | 16 | 75.0% | 68.8% | 0.096 |
| 3000 | 15 | 53.3% | 46.7% | 1.194 |
| 5000 | 16 | 62.5% | 56.2% | 0.274 |
| 15000 | 18 | 50.0% | 50.0% | 25.075 |


### Detection Recall by Injected Period

| Period (days) | N | Detection Recall | Period Recovery 1% | Median Period Err% |
|--------------|---|-----------------|-------------------|-------------------|
| 0.75 | 16 | 100.0% | 100.0% | 0.096 |
| 1.50 | 8 | 100.0% | 100.0% | 0.171 |
| 3.50 | 18 | 88.9% | 88.9% | 0.042 |
| 7.00 | 20 | 15.0% | 0.0% | 49.979 |
| 12.00 | 13 | 0.0% | 0.0% | 50.055 |


---

## 5. False-Positive Controls

| Control Type | N Controls | N FP | FP Rate | Mean Conf (FP) | Median SNR (FP) |
|---|---|---|---|---|---|
| white_noise | 5 | 0 | 0.0% | N/A | N/A |
| red_noise | 5 | 5 | 100.0% | 0.000 | 9.019 |
| sinusoidal | 5 | 1 | 20.0% | 0.000 | 8.309 |
| quasi_periodic | 5 | 5 | 100.0% | 0.000 | 9.173 |
| systematics_gap | 5 | 5 | 100.0% | 0.000 | 9.033 |


**Interpretation**: A false positive is defined as `candidate_detected=True` with
`recovered_snr >= 5.0` on a light curve with NO injected transit.
Values above 10-15% indicate the BLS detector is too sensitive to noise patterns.

---

## 6. Alias Behavior

| Metric | Value |
|--------|-------|
| Half-period alias rate (P/2 recovered instead of P) | 30.7% |
| Double-period alias rate (2P recovered instead of P) | 0.0% |
| Any harmonic match rate (within 5%) | 88.0% |

**Note**: Alias rates are computed only over detected injection trials.
High half-period alias rates indicate the BLS is finding the dominant harmonic
of the true period (common for short-duration, long-period signals).

---

## 7. Confidence Score Behavior

| Group | Mean Confidence |
|-------|----------------|
| Correctly detected injections | 0.000 |
| Missed injections | 0.000 |
| False positives (controls) | 0.000 |

A well-calibrated system should have: detected_correct > missed > false_positive.

---

## 8. Weak Regimes

The following conditions showed detection recall < 50% or FP rate > 15%:

- **Low SNR [0-3)**: detection recall = 0.0% (N=3)
- **Low SNR [3-5)**: detection recall = 20.0% (N=5)
- **Depth 300 ppm**: detection recall = 40.0% (N=10)
- **Period 7.00 d**: detection recall = 15.0% (N=20)
- **Period 12.00 d**: detection recall = 0.0% (N=13)
- **FP elevated for red_noise**: FP rate = 100.0% (N=5)
- **FP elevated for sinusoidal**: FP rate = 20.0% (N=5)
- **FP elevated for quasi_periodic**: FP rate = 100.0% (N=5)
- **FP elevated for systematics_gap**: FP rate = 100.0% (N=5)

These regimes should be the focus of Phase 5 classifier strengthening.

---

## 9. Strict Conclusion

**Phase 4 useful but not strong enough for 95+ evidence** — see Weak Regimes section

---

## 10. Caveats and Limitations

1. **Synthetic noise only**: These results use white noise, AR(1) red noise, and
   sinusoidal stellar variability. Real TESS systematics (momentum dumps, scattered
   light, systematics correlations) are not modelled.
2. **No real TESS data**: All light curves are fully synthetic. Performance on real
   TESS light curves may differ substantially, especially for low-SNR signals.
3. **SNR is estimated analytically**: The `injected_snr_estimate` field is an
   analytical upper bound. Actual SNR after preprocessing/detrending will be lower.
4. **Preprocessing removes signal**: The running-median detrending in `preprocess.clean()`
   can partially remove transit signals with duration > detrend window. This effect
   is not corrected for in SNR estimates.
5. **Classification confidence is 0.0 in BLS-only mode**: This suite runs only
   `preprocess.clean()` + `bls_detector.detect()`, not the full `analyze_light_curve()`
   pipeline. Confidence scores reflect BLS properties, not the ML classifier.
6. **Mode = quick**: 75 injection trials. For statistical confidence in each
   grid cell, run `standard` (500+) or `full` (2000+) mode.
7. **Do not cite these as Level 4 evidence**: Real TESS sector screening with ≥100
   targets must also be performed for a Level 4 claim.

---

*Generated by TransitLens Phase 4 Injection-Recovery Suite. Seed=42. Mode=quick.*
*This report is auto-generated and reflects observed pipeline performance, not targets.*
