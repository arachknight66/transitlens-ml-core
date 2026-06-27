# Feature Generation Report
Generated on: 2026-06-27 18:19:13

## Summary

| Split | Total Targets | Success | Failed | Exoplanet Transit | Eclipsing Binary | Blend Contam | Stellar Var/Other |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| train | 4 | 3 | 1 | 3 | 0 | 0 | 0 |
| val | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Configuration
- Real-only mode: True
- Resume: False
- Feature count: 16 features

## Excluded / Included Features
The feature matrix retains metadata columns for downstream evaluation, but ML models MUST only use the following feature columns for training and evaluation:

- `bls_power`
- `snr`
- `period_days`
- `duration_days`
- `depth`
- `transit_count`
- `odd_even_depth_delta`
- `v_shape_score`
- `local_noise`
- `depth_to_noise_ratio`
- `phase_shape_kurtosis`
- `bls_sde`
- `secondary_eclipse_depth`
- `centroid_shift`
- `crowding_metric`
- `gaia_neighbor_count`