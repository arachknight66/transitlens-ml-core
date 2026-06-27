# Feature Generation Report
Generated on: 2026-06-27 19:23:08

## Summary

| Split | Total Targets | Success Features | Failed | Suspicious | Exoplanet Transit | Stellar Var/Other |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| train | 41 | 7 | 1 | 33 | 4 | 3 |
| val | 14 | 6 | 0 | 8 | 4 | 2 |
| test | 10 | 2 | 0 | 8 | 0 | 2 |

## Configuration
- Real-only mode: True
- Resume: False
- Include Suspicious: False
- Feature count: 16 features

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