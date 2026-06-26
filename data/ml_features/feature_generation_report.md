# Feature Generation Report
Generated on: 2026-06-26 23:54:51

## Summary

| Split | Total Targets | Success | Failed | Exoplanet Transit | Eclipsing Binary | Blend Contam | Stellar Var/Other |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| train | 604 | 604 | 0 | 153 | 150 | 150 | 151 |
| val | 202 | 202 | 0 | 51 | 51 | 50 | 50 |
| test | 201 | 201 | 0 | 51 | 50 | 50 | 50 |

## Configuration
- Target count per class: Train=150, Val=50, Test=50
- Resume: False
- Feature count: 16 features

## Excluded / Included Features
The feature matrix retains `target_id`, `class_label`, and `split` columns for downstream splitting and evaluation, but ML models MUST only use the following feature columns for training and evaluation:

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