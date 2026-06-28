# TransitLens Feature Schema

This document defines the 16 canonical features used by the TransitLens classification models. Every feature represents a physically meaningful property that distinguishes exoplanet transits from other astronomical events.

## Feature Schema Table

| Index | Feature Name | DataType | Unit | Physical Interpretation / Usage |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `bls_power` | float | None | Normalized Box Least Squares peak power. Measures periodogram signal strength. |
| 2 | `snr` | float | None | Signal-to-Noise Ratio of the detection. |
| 3 | `period_days` | float | Days | Detected orbital period of the candidate transit. |
| 4 | `duration_days` | float | Days | Best-fit transit duration. |
| 5 | `depth` | float | None | Fractional flux drop at transit center. |
| 6 | `transit_count` | int | Count | Number of transit events in the time series. |
| 7 | `odd_even_depth_delta` | float | None | Difference between odd-numbered and even-numbered transit depths. Eclipsing binary discriminator. |
| 8 | `v_shape_score` | float | None | Fit ratio between a flat box (0.0) and a linear V-shape (1.0). Indicates grazing/stellar eclipse. |
| 9 | `local_noise` | float | None | Out-of-transit flux scatter (RMS). |
| 10 | `depth_to_noise_ratio` | float | None | Local transit depth divided by local noise. |
| 11 | `phase_shape_kurtosis` | float | None | Excess kurtosis of the in-transit binned profile. High kurtosis implies spiky profile (EB-like). |
| 12 | `bls_sde` | float | None | Signal Detection Efficiency. Standard measurement of power spectrum peak significance. |
| 13 | `secondary_eclipse_depth` | float | None | Fractional depth of secondary eclipse at half phase, resolving binary aliases. |
| 14 | `centroid_shift` | float | Pixels | Mean pixel displacement of the photocenter during transit compared to out-of-transit. |
| 15 | `crowding_metric` | float | None | CROWDSAP ratio of target star flux to total aperture flux. Low (<0.8) implies dilution/blend. |
| 16 | `gaia_neighbor_count` | int | Count | Count of neighboring sources in Gaia DR2 inside aperture. |

## Excluded Metadata / Leaky Parameters
The ML models MUST NOT train or predict using target identifiers (`target_id`, `tic_id`), raw dispositions, catalog names, or ground-truth physical parameters (such as `true_period` or `true_depth` unless they are pipeline estimates).
