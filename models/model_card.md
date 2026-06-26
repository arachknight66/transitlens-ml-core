# Model Card: TransitLens Stochastic Classifier
- Trained at: 2026-06-26T07:13:19.830092+00:00
- Samples: 200 total targets (Kepler KOIs & TESS TOIs combined)
- RF Validation Accuracy: 64.0000%
- XGBoost Validation Accuracy: 62.0000%

## Label Mapping
{
  "0": "blend_contamination",
  "1": "eclipsing_binary",
  "2": "exoplanet_transit",
  "3": "stellar_variability_or_other"
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
  "phase_shape_kurtosis",
  "bls_sde",
  "secondary_eclipse_depth",
  "centroid_shift",
  "crowding_metric",
  "gaia_neighbor_count"
]
