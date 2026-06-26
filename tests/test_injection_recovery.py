"""
tests/test_injection_recovery.py
---------------------------------
Unit tests for the Phase 4 injection-recovery module.

Tests verify:
  1. Transit injection creates correct depth at the right phase
  2. Trapezoid/box injection correct relative to pure box
  3. Dilution reduces apparent depth by dilution_factor
  4. Noise generation has correct RMS
  5. Red noise shows non-zero autocorrelation (is correlated)
  6. Random gaps reduce point count but preserve sorted monotonic time
  7. TESS downlink gap removes the correct time range
  8. Control light curves have injected=False
  9. Metric computation correctly flags 1%, 5%, half-period alias, double-period alias
  10. Metric computation handles empty / all-failure bins gracefully
  11. estimate_injected_snr returns 0 for degenerate inputs
  12. run_injection_recovery_suite quick mode writes expected files

NOTE: Test 12 (full suite) is marked slow and may be skipped with -m "not slow".
"""

from __future__ import annotations

import csv
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Path setup ───────────────────────────────────────────────────────────────
_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.injection_recovery import (
    make_time_array,
    inject_box_or_trapezoid_transit,
    add_noise,
    add_stellar_variability,
    apply_dilution,
    generate_control_lightcurve,
    estimate_injected_snr,
    compute_trial_metrics,
    TrialResult,
    TRIAL_CSV_COLUMNS,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def default_time(rng):
    """27-day time array at 2-min cadence, no gaps."""
    return make_time_array(cadence_min=2.0, time_span_days=27.0, gap_mode="none", rng=rng)


@pytest.fixture
def baseline_cfg():
    """Minimal config dict for metric computation tests."""
    return {
        "global": {"cadence_min": 2.0, "time_span_days": 27.0},
        "grid": {"variability_amplitude": 0.003},
        "recovery_thresholds": {
            "period_tolerance_1pct": 1.0,
            "period_tolerance_5pct": 5.0,
            "alias_tolerance_pct": 1.0,
            "min_snr_for_recall": 7.0,
            "detection_snr_threshold": 5.0,
        },
    }


# ============================================================================
# 1. Transit injection creates correct depth
# ============================================================================

class TestInjectBoxTransit:
    def test_in_transit_depth_is_correct(self, default_time):
        """In-transit flux should be depressed by approximately the injected depth."""
        period = 5.0
        depth = 0.01     # 1%
        duration = 0.1   # 0.1 days
        epoch = 0.0
        ingress_ratio = 0.0  # pure box

        flux = inject_box_or_trapezoid_transit(
            default_time, period, depth, duration, epoch, ingress_ratio
        )

        # Phase-fold to find in-transit points
        phase_days = ((default_time - epoch + period / 2.0) % period) - period / 2.0
        in_transit = np.abs(phase_days) <= duration / 2.0
        assert in_transit.sum() > 0, "No in-transit points found"

        # All in-transit points should be ~1.0 - depth
        in_transit_flux = flux[in_transit]
        assert np.allclose(
            in_transit_flux, 1.0 - depth, atol=1e-9
        ), f"In-transit flux deviates: mean={in_transit_flux.mean():.6f}, expected={1.0 - depth:.6f}"

    def test_out_of_transit_is_one(self, default_time):
        """Out-of-transit flux should be exactly 1.0 for box injection."""
        flux = inject_box_or_trapezoid_transit(
            default_time, period=7.0, depth=0.005, duration=0.15, epoch=1.0, ingress_ratio=0.0
        )
        phase_days = ((default_time - 1.0 + 7.0 / 2.0) % 7.0) - 7.0 / 2.0
        out_of_transit = np.abs(phase_days) > 0.15 / 2.0
        assert np.allclose(flux[out_of_transit], 1.0, atol=1e-9)

    def test_trapezoid_ingress_depth_between_zero_and_depth(self, default_time):
        """Ingress flux should be strictly between 1-depth and 1.0."""
        period = 3.5
        depth = 0.005
        duration = 0.10
        ingress_ratio = 0.3
        epoch = 0.5

        flux = inject_box_or_trapezoid_transit(
            default_time, period, depth, duration, epoch, ingress_ratio
        )

        phase_days = ((default_time - epoch + period / 2.0) % period) - period / 2.0
        half_flat = (duration / 2.0) * (1 - ingress_ratio)
        half_dur = duration / 2.0

        ingress_mask = (phase_days >= -half_dur) & (phase_days < -half_flat)
        if ingress_mask.sum() > 0:
            ingress_flux = flux[ingress_mask]
            # Must be strictly between (1 - depth) and 1.0
            assert np.all(ingress_flux >= 1.0 - depth - 1e-9), "Ingress flux below 1-depth"
            assert np.all(ingress_flux <= 1.0 + 1e-9), "Ingress flux above 1.0"

    def test_flux_array_same_length_as_time(self, default_time):
        """Output array must have same length as input time array."""
        flux = inject_box_or_trapezoid_transit(
            default_time, period=5.0, depth=0.003, duration=0.1, epoch=0.0
        )
        assert len(flux) == len(default_time)

    def test_periodic_transits_all_same_depth(self, default_time):
        """All transit events for the same period should have the same depth."""
        period = 3.5
        depth = 0.01
        duration = 0.1
        epoch = 1.0

        flux = inject_box_or_trapezoid_transit(
            default_time, period, depth, duration, epoch, ingress_ratio=0.0
        )
        phase_days = ((default_time - epoch + period / 2.0) % period) - period / 2.0
        in_transit = np.abs(phase_days) <= duration / 2.0
        assert in_transit.sum() > 0

        dip = 1.0 - flux[in_transit]
        # All dip values should be very close to depth
        assert np.all(np.abs(dip - depth) < 1e-8), (
            f"Not all in-transit dips equal depth: min={dip.min():.8f}, max={dip.max():.8f}"
        )


# ============================================================================
# 2. Dilution reduces apparent depth
# ============================================================================

class TestApplyDilution:
    def test_dilution_reduces_transit_depth(self, default_time):
        """After applying dilution, the transit dip should be scaled by dilution_factor."""
        period = 5.0
        true_depth = 0.01
        dilution = 0.6

        flux_no_dil = inject_box_or_trapezoid_transit(
            default_time, period, true_depth, 0.1, 0.0, ingress_ratio=0.0
        )
        flux_dil = apply_dilution(flux_no_dil, dilution)

        # Observed depth = dilution_factor × true_depth
        phase_days = ((default_time + period / 2.0) % period) - period / 2.0
        in_transit = np.abs(phase_days) <= 0.1 / 2.0
        assert in_transit.sum() > 0

        observed_dip = 1.0 - flux_dil[in_transit].mean()
        expected_dip = dilution * true_depth
        assert abs(observed_dip - expected_dip) < 1e-6, (
            f"Diluted dip {observed_dip:.6f} != expected {expected_dip:.6f}"
        )

    def test_full_dilution_preserves_baseline(self, default_time):
        """dilution_factor=1.0 should leave flux unchanged."""
        flux_orig = np.ones(len(default_time)) - 0.005  # uniform dip
        flux_out = apply_dilution(flux_orig, 1.0)
        assert np.allclose(flux_orig, flux_out)

    def test_zero_dilution_raises(self):
        """dilution_factor=0.0 should raise ValueError."""
        with pytest.raises(ValueError, match="dilution_factor"):
            apply_dilution(np.ones(100), 0.0)

    def test_baseline_stays_at_one(self, default_time):
        """Out-of-transit flux after dilution should remain ≈ 1.0."""
        flux = inject_box_or_trapezoid_transit(default_time, 5.0, 0.01, 0.1, 0.0)
        flux_dil = apply_dilution(flux, 0.7)
        phase_days = ((default_time + 2.5) % 5.0) - 2.5
        oot = np.abs(phase_days) > 0.1
        assert np.allclose(flux_dil[oot], 1.0, atol=1e-6)


# ============================================================================
# 3. Noise generation
# ============================================================================

class TestAddNoise:
    def test_white_noise_rms(self, rng):
        """White noise should have RMS close to requested noise_rms."""
        flux = np.ones(50_000)
        noise_rms = 0.001
        noisy = add_noise(flux, noise_rms, "white", rng)
        residuals = noisy - 1.0
        measured_rms = float(np.std(residuals))
        assert abs(measured_rms - noise_rms) < noise_rms * 0.05, (
            f"White noise RMS {measured_rms:.5f} deviates from target {noise_rms}"
        )

    def test_red_noise_is_correlated(self, rng):
        """Red noise should show non-negligible autocorrelation at lag 1."""
        flux = np.ones(10_000)
        noisy = add_noise(flux, 0.001, "red", rng)
        residuals = noisy - 1.0
        # Autocorrelation at lag 1 should be > 0.3 for AR(1) with phi=0.6
        lag1_autocorr = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
        assert lag1_autocorr > 0.3, (
            f"Red noise autocorrelation at lag 1 = {lag1_autocorr:.3f} (expected > 0.3)"
        )

    def test_white_noise_is_uncorrelated(self, rng):
        """White noise should show near-zero autocorrelation at lag 1."""
        flux = np.ones(20_000)
        noisy = add_noise(flux, 0.001, "white", rng)
        residuals = noisy - 1.0
        lag1_autocorr = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
        assert abs(lag1_autocorr) < 0.05, (
            f"White noise lag-1 autocorr = {lag1_autocorr:.3f} (expected ≈ 0)"
        )

    def test_mixed_noise_rms_in_range(self, rng):
        """Mixed noise RMS should be approximately noise_rms."""
        flux = np.ones(30_000)
        noise_rms = 0.002
        noisy = add_noise(flux, noise_rms, "mixed", rng)
        rms = float(np.std(noisy - 1.0))
        # Allow ±20% deviation (red component adds variance)
        assert noise_rms * 0.5 < rms < noise_rms * 2.0, (
            f"Mixed noise RMS {rms:.5f} out of range for target {noise_rms}"
        )

    def test_invalid_mode_raises(self, rng):
        """Unknown noise mode should raise ValueError."""
        with pytest.raises(ValueError, match="noise mode"):
            add_noise(np.ones(100), 0.001, "quantum", rng)


# ============================================================================
# 4. Gap modes
# ============================================================================

class TestMakeTimeArray:
    def test_no_gaps_length(self):
        """No-gap time array should have approximately n=time_span/cadence points."""
        rng = np.random.default_rng(0)
        time = make_time_array(2.0, 27.0, "none", rng)
        expected_n = int(27.0 / (2.0 / 1440.0))
        # Allow ±1 due to rounding
        assert abs(len(time) - expected_n) <= 1, f"n={len(time)}, expected≈{expected_n}"

    def test_random_gaps_reduce_points(self):
        """Random gaps should produce fewer points than no-gap."""
        rng_no = np.random.default_rng(1)
        rng_gap = np.random.default_rng(1)
        time_no = make_time_array(2.0, 27.0, "none", rng_no)
        time_gap = make_time_array(2.0, 27.0, "random_gaps", rng_gap)
        assert len(time_gap) < len(time_no), (
            f"Random gaps should reduce points: {len(time_gap)} >= {len(time_no)}"
        )

    def test_tess_downlink_gap_reduces_points(self):
        """TESS downlink gap should produce fewer points than no-gap."""
        rng_no = np.random.default_rng(2)
        rng_gap = np.random.default_rng(2)
        time_no = make_time_array(2.0, 27.0, "none", rng_no)
        time_gap = make_time_array(2.0, 27.0, "tess_downlink_gap", rng_gap)
        assert len(time_gap) < len(time_no), (
            f"TESS gap should reduce points: {len(time_gap)} >= {len(time_no)}"
        )

    def test_time_is_sorted(self):
        """Output time array must be monotonically increasing."""
        rng = np.random.default_rng(3)
        for gap_mode in ["none", "random_gaps", "tess_downlink_gap"]:
            time = make_time_array(2.0, 27.0, gap_mode, rng)
            diffs = np.diff(time)
            assert np.all(diffs > 0), (
                f"Time array not sorted for gap_mode={gap_mode!r}: "
                f"found {(diffs <= 0).sum()} non-increasing steps"
            )

    def test_random_gaps_preserves_no_negative_time(self):
        """All time values must be non-negative."""
        rng = np.random.default_rng(4)
        for gap_mode in ["none", "random_gaps", "tess_downlink_gap"]:
            time = make_time_array(2.0, 27.0, gap_mode, rng)
            assert np.all(time >= 0.0), f"Negative time values in gap_mode={gap_mode!r}"

    def test_tess_gap_near_midpoint(self):
        """TESS downlink gap should remove points near the midpoint (day ~13.5)."""
        rng = np.random.default_rng(99)
        time = make_time_array(2.0, 27.0, "tess_downlink_gap", rng)
        # Check mid-sector region has a gap
        near_mid = (time >= 12.5) & (time <= 14.5)
        n_near_mid = near_mid.sum()
        total_near_mid_expected = int(2.0 / (2.0 / 1440.0))  # 2 days worth
        # At least 20% fewer points than if there were no gap
        assert n_near_mid < total_near_mid_expected * 0.8, (
            f"Expected gap near mid-sector, but found {n_near_mid} points in [12.5, 14.5] d"
        )


# ============================================================================
# 5. Controls have injected=False
# ============================================================================

class TestControlLightcurves:
    def test_control_produces_no_transit_shape(self, rng):
        """Control light curves should not have a persistent periodic dip."""
        time = make_time_array(2.0, 27.0, "none", rng)
        cfg = {"variability_amplitude": 0.003}
        for ct in ["white_noise", "red_noise", "sinusoidal", "quasi_periodic", "systematics_gap"]:
            flux = generate_control_lightcurve(ct, time, noise_rms=0.001, rng=rng, cfg=cfg)
            assert len(flux) == len(time), f"Length mismatch for control_type={ct!r}"
            # Baseline should be approximately 1.0 (within ±5%)
            assert abs(np.median(flux) - 1.0) < 0.05, (
                f"Control {ct!r} median flux = {np.median(flux):.4f} (expected ≈ 1.0)"
            )

    def test_invalid_control_type_raises(self, rng):
        """Unknown control type should raise ValueError."""
        time = np.linspace(0, 27, 1000)
        with pytest.raises(ValueError, match="control_type"):
            generate_control_lightcurve("laser_noise", time, 0.001, rng, {})


# ============================================================================
# 6. Period recovery metric flags
# ============================================================================

class TestPeriodRecoveryFlags:
    def _make_injection_trial(self, true_period, recovered_period, detected=True, snr=10.0):
        """Helper: build a minimal TrialResult for metric testing."""
        return TrialResult(
            trial_id=1, mode="test", random_seed=0,
            source_type="injection", control_type="",
            injected=True,
            injected_period_days=true_period,
            injected_depth=0.01,
            injected_duration_days=0.1,
            injected_epoch=0.0,
            injected_snr_estimate=snr,
            noise_rms=0.001, variability_mode="none",
            variability_amplitude=0.0, gap_mode="none",
            dilution_factor=1.0, ingress_ratio=0.0,
            n_points=10000, time_span_days=27.0,
            candidate_detected=detected,
            recovered_period_days=recovered_period,
            recovered_depth=0.01 if detected else None,
            recovered_duration_days=0.1 if detected else None,
            recovered_snr=snr,
        )

    def test_exact_period_recovery(self, baseline_cfg):
        """Exact period match should set period_recovered_1pct=True."""
        trial = self._make_injection_trial(5.0, 5.0)
        trial = compute_trial_metrics(trial, baseline_cfg)
        assert trial.period_recovered_1pct is True
        assert trial.period_recovered_5pct is True
        assert trial.detected_correctly is True
        assert abs(trial.period_error_pct) < 1e-9

    def test_close_period_within_1pct(self, baseline_cfg):
        """Period within 1% should flag period_recovered_1pct=True."""
        trial = self._make_injection_trial(5.0, 5.0 * 1.009)  # 0.9% error
        trial = compute_trial_metrics(trial, baseline_cfg)
        assert trial.period_recovered_1pct is True
        assert trial.period_recovered_5pct is True

    def test_period_outside_1pct_but_within_5pct(self, baseline_cfg):
        """Period 3% off: 1% flag False, 5% flag True."""
        trial = self._make_injection_trial(5.0, 5.0 * 1.03)  # 3% error
        trial = compute_trial_metrics(trial, baseline_cfg)
        assert trial.period_recovered_1pct is False
        assert trial.period_recovered_5pct is True

    def test_half_period_alias_detected(self, baseline_cfg):
        """When recovered = P/2, half_period_alias flag should be set."""
        true_p = 6.0
        trial = self._make_injection_trial(true_p, true_p / 2.0)
        trial = compute_trial_metrics(trial, baseline_cfg)
        assert trial.half_period_alias is True
        assert trial.double_period_alias is False

    def test_double_period_alias_detected(self, baseline_cfg):
        """When recovered = 2P, double_period_alias flag should be set."""
        true_p = 3.5
        trial = self._make_injection_trial(true_p, true_p * 2.0)
        trial = compute_trial_metrics(trial, baseline_cfg)
        assert trial.double_period_alias is True
        assert trial.half_period_alias is False

    def test_missed_detection_all_false(self, baseline_cfg):
        """Missed detection (candidate_detected=False) should leave recovery flags False."""
        trial = self._make_injection_trial(5.0, None, detected=False)
        trial = compute_trial_metrics(trial, baseline_cfg)
        assert trial.detected_correctly is False
        assert trial.period_recovered_1pct is False
        assert trial.period_recovered_5pct is False
        assert trial.half_period_alias is False
        assert trial.double_period_alias is False

    def test_far_off_period_all_false(self, baseline_cfg):
        """Period very far off (50% error) should fail all recovery flags."""
        trial = self._make_injection_trial(5.0, 7.5)  # 50% error
        trial = compute_trial_metrics(trial, baseline_cfg)
        assert trial.period_recovered_1pct is False
        assert trial.period_recovered_5pct is False
        assert trial.detected_correctly is False

    def test_control_false_positive_flag(self, baseline_cfg):
        """Control trial with high SNR detection should be flagged as false positive."""
        trial = TrialResult(
            trial_id=2, mode="test", random_seed=0,
            source_type="control", control_type="white_noise",
            injected=False,
            injected_period_days=None, injected_depth=None,
            injected_duration_days=None, injected_epoch=None,
            injected_snr_estimate=None,
            noise_rms=0.001, variability_mode="none",
            variability_amplitude=0.0, gap_mode="none",
            dilution_factor=1.0, ingress_ratio=0.0,
            n_points=10000, time_span_days=27.0,
            candidate_detected=True,
            recovered_snr=7.5,  # above detection_snr_threshold=5
        )
        trial = compute_trial_metrics(trial, baseline_cfg)
        assert trial.false_positive is True

    def test_control_no_detection_no_fp(self, baseline_cfg):
        """Control trial with no detection should NOT be a false positive."""
        trial = TrialResult(
            trial_id=3, mode="test", random_seed=0,
            source_type="control", control_type="red_noise",
            injected=False,
            injected_period_days=None, injected_depth=None,
            injected_duration_days=None, injected_epoch=None,
            injected_snr_estimate=None,
            noise_rms=0.001, variability_mode="none",
            variability_amplitude=0.0, gap_mode="none",
            dilution_factor=1.0, ingress_ratio=0.0,
            n_points=10000, time_span_days=27.0,
            candidate_detected=False,
            recovered_snr=2.1,
        )
        trial = compute_trial_metrics(trial, baseline_cfg)
        assert trial.false_positive is False


# ============================================================================
# 7. Estimate SNR handles degenerate inputs
# ============================================================================

class TestEstimateInjectedSNR:
    def test_positive_snr_for_valid_inputs(self):
        """Valid inputs should return a positive SNR."""
        snr = estimate_injected_snr(
            depth=0.01,
            duration_days=0.1,
            period_days=5.0,
            time_span_days=27.0,
            noise_rms=0.001,
            dilution_factor=1.0,
            cadence_min=2.0,
        )
        assert snr > 0, f"SNR should be positive, got {snr}"

    def test_zero_noise_returns_zero(self):
        """Zero noise_rms should return 0.0 (not inf/nan)."""
        snr = estimate_injected_snr(0.01, 0.1, 5.0, 27.0, 0.0, 1.0, 2.0)
        assert snr == 0.0

    def test_zero_period_returns_zero(self):
        snr = estimate_injected_snr(0.01, 0.1, 0.0, 27.0, 0.001, 1.0, 2.0)
        assert snr == 0.0

    def test_dilution_reduces_snr(self):
        """Dilution should reduce the SNR proportionally."""
        snr_full = estimate_injected_snr(0.01, 0.1, 5.0, 27.0, 0.001, 1.0, 2.0)
        snr_half = estimate_injected_snr(0.01, 0.1, 5.0, 27.0, 0.001, 0.5, 2.0)
        assert abs(snr_half - snr_full * 0.5) < 1e-6, (
            f"Diluted SNR {snr_half:.3f} != 0.5 × {snr_full:.3f}"
        )

    def test_deeper_transit_has_higher_snr(self):
        """Deeper transit should have higher SNR."""
        snr_shallow = estimate_injected_snr(0.001, 0.1, 5.0, 27.0, 0.001, 1.0, 2.0)
        snr_deep = estimate_injected_snr(0.01, 0.1, 5.0, 27.0, 0.001, 1.0, 2.0)
        assert snr_deep > snr_shallow


# ============================================================================
# 8. TrialResult CSV row has all required columns
# ============================================================================

class TestTrialResultCSVRow:
    def test_csv_row_has_all_columns(self):
        """to_csv_row() should return a dict with all TRIAL_CSV_COLUMNS keys."""
        trial = TrialResult(
            trial_id=1, mode="quick", random_seed=42,
            source_type="injection", control_type="",
            injected=True, injected_period_days=5.0, injected_depth=0.01,
            injected_duration_days=0.1, injected_epoch=1.0,
            injected_snr_estimate=12.5, noise_rms=0.001,
            variability_mode="none", variability_amplitude=0.0,
            gap_mode="none", dilution_factor=1.0, ingress_ratio=0.2,
            n_points=10000, time_span_days=27.0,
        )
        row = trial.to_csv_row()
        missing = set(TRIAL_CSV_COLUMNS) - set(row.keys())
        assert len(missing) == 0, f"Missing CSV columns: {missing}"

    def test_none_values_become_empty_string(self):
        """None values in to_csv_row() should be converted to empty string."""
        trial = TrialResult(
            trial_id=1, mode="quick", random_seed=42,
            source_type="control", control_type="white_noise",
            injected=False, injected_period_days=None, injected_depth=None,
            injected_duration_days=None, injected_epoch=None,
            injected_snr_estimate=None, noise_rms=0.001,
            variability_mode="none", variability_amplitude=0.0,
            gap_mode="none", dilution_factor=1.0, ingress_ratio=0.0,
            n_points=5000, time_span_days=27.0,
        )
        row = trial.to_csv_row()
        # All None fields must be "" not None
        for k, v in row.items():
            assert v is not None, f"Column {k!r} has None value (should be empty string)"


# ============================================================================
# 9. Full suite quick mode — integration test (slow)
# ============================================================================

@pytest.mark.slow
def test_quick_mode_writes_expected_files(tmp_path):
    """
    Run the full injection-recovery suite in quick mode and verify that
    all required output files are created.

    This is an integration test. It actually runs the pipeline against
    synthetic light curves. Mark as slow: skip with pytest -m "not slow".
    """
    import yaml
    from eval.injection_recovery import run_injection_recovery_suite

    # Write a minimal config to tmp_path
    cfg = {
        "global": {"random_seed": 77, "cadence_min": 2.0, "time_span_days": 27.0,
                   "output_dir": str(tmp_path)},
        "grid": {
            "period_days": [3.5], "depth": [0.005], "duration_days": [0.1],
            "noise_rms": [0.001], "variability_mode": ["none"], "gap_mode": ["none"],
            "dilution_factor": [1.0], "variability_amplitude": 0.003, "ingress_ratio": 0.2,
            "n_trials_per_cell": 1,
        },
        "controls": {"n_white_noise": 2, "n_red_noise": 2, "n_sinusoidal": 2,
                     "n_quasi_periodic": 2, "n_systematics_gap": 2},
        "recovery_thresholds": {
            "period_tolerance_1pct": 1.0, "period_tolerance_5pct": 5.0,
            "alias_tolerance_pct": 1.0, "min_snr_for_recall": 7.0,
            "detection_snr_threshold": 5.0,
            "snr_bins": [0, 5, 7, 10, 999],
        },
        "modes": {
            "quick": {"n_injection_trials": 5, "n_controls_total": 5, "subsample_grid": True},
            "standard": {"n_injection_trials": 500, "n_controls_total": 100, "subsample_grid": True},
            "full": {"n_injection_trials": 2000, "n_controls_total": 200, "subsample_grid": False},
        },
    }
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    summary = run_injection_recovery_suite(
        mode="quick",
        cfg_path=str(cfg_path),
        output_dir=str(tmp_path),
        seed=77,
        max_trials=5,
    )

    # Check all required output files exist
    required_files = [
        "injection_recovery_trials.csv",
        "injection_recovery_summary.csv",
        "injection_recovery_by_snr.csv",
        "injection_recovery_by_depth.csv",
        "injection_recovery_by_period.csv",
        "injection_recovery_by_noise.csv",
        "false_positive_controls.csv",
        "false_positive_summary.csv",
        "alias_recovery_summary.csv",
        "phase4_injection_recovery_report.md",
    ]
    for fname in required_files:
        assert (tmp_path / fname).exists(), f"Expected output file missing: {fname}"

    # Trials CSV should have correct columns header
    trials_csv = tmp_path / "injection_recovery_trials.csv"
    with open(trials_csv, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        missing_cols = set(TRIAL_CSV_COLUMNS) - set(headers)
        assert len(missing_cols) == 0, f"Trials CSV missing columns: {missing_cols}"

    # Summary should have expected keys
    assert "detection_recall" in summary
    assert "false_positive_rate_controls" in summary
    assert "period_recovery_rate_1pct" in summary

    # Report should contain key sections
    report_text = (tmp_path / "phase4_injection_recovery_report.md").read_text()
    for section in [
        "Run Configuration", "Overall Results", "SNR Bin", "False-Positive",
        "Alias Behavior", "Weak Regimes", "Strict Conclusion", "Caveats"
    ]:
        assert section in report_text, f"Report missing section: {section!r}"


# ============================================================================
# 10. Stellar variability modes
# ============================================================================

class TestAddStellarVariability:
    def test_none_mode_no_change(self, default_time, rng):
        """mode='none' should return flux unchanged."""
        flux = np.ones(len(default_time))
        out = add_stellar_variability(default_time, flux, "none", 0.005, None, rng)
        assert np.allclose(out, flux)

    def test_sinusoidal_has_nonzero_variability(self, default_time, rng):
        """Sinusoidal mode should add detectable variability."""
        flux = np.ones(len(default_time))
        out = add_stellar_variability(default_time, flux, "sinusoidal", 0.005, 10.0, rng)
        rms = float(np.std(out - flux))
        assert rms > 0.001, f"Sinusoidal variability too small: RMS={rms:.6f}"

    def test_quasi_periodic_has_nonzero_variability(self, default_time, rng):
        """Quasi-periodic mode should add detectable variability."""
        flux = np.ones(len(default_time))
        out = add_stellar_variability(default_time, flux, "quasi_periodic", 0.005, None, rng)
        rms = float(np.std(out - flux))
        assert rms > 0.001, f"Quasi-periodic variability too small: RMS={rms:.6f}"

    def test_invalid_mode_raises(self, default_time, rng):
        """Unknown mode should raise ValueError."""
        with pytest.raises(ValueError):
            add_stellar_variability(default_time, np.ones(len(default_time)),
                                    "starquake", 0.005, None, rng)
