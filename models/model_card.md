# Model Card: TransitLens Stochastic Classifier
- Trained at: 2026-06-26T06:23:07.632677+00:00
- Samples: 150 total targets (Kepler KOIs & TESS TOIs combined)
- RF Validation Accuracy: 86.8421%
- XGBoost Validation Accuracy: 81.5789%

## Label Mapping
{
  "0": "eclipsing_binary_like",
  "1": "exoplanet_like",
  "2": "noise_or_other"
}

## Features (in canonical order)
[
  "bls_power",
  "snr",
  "period_days",
  "duration_days",
  "depth",
  "transit_count",
  "odd_even_depth_delta",
  "v_shape_score",
  "local_noise",
  "depth_to_noise_ratio",
  "phase_shape_kurtosis"
]
