# Feature Generation Report
Generated on: 2026-06-28 09:55:57

## Summary

| Split | Total Targets | Success Features | Failed | Suspicious | Exoplanet Transit | Stellar Var/Other |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| train | 22 | 22 | 0 | 0 | 3 | 19 |
| val | 4 | 4 | 0 | 0 | 1 | 2 |
| test | 2 | 2 | 0 | 0 | 1 | 1 |

## Configuration
- Real-only mode: True
- Resume: False
- Include Suspicious: False
- Feature count: 18 features

## Checked Features List

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
- `odd_even_significance`
- `secondary_eclipse_significance`