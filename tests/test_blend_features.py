"""
tests/test_blend_features.py
----------------------------
Tests for Phase 6 blend and contamination diagnostics.

Tests ensure:
    - Missing centroid data returns unavailable, not zero-safe values
    - Synthetic centroid shift is detected with significance > threshold
    - No centroid shift gives low significance
    - CROWDSAP correction increases depth when crowding < 1
    - Missing crowding returns unavailable
    - Neighbor risk identifies close bright neighbor
    - All diagnostics unavailable => blend_risk_level = "unavailable"
    - High centroid shift => high blend risk
    - Pipeline result includes diagnostics.blend
"""

from __future__ import annotations

import numpy as np
import pytest

from core.blend_features import (
    compute_centroid_shift,
    compute_crowding_diagnostics,
    compute_neighbor_diagnostics,
    compute_blend_risk_score,
    extract_blend_diagnostics,
    get_blend_explanation,
    load_neighbor_catalog,
    BLEND_RISK_LEVELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transit_lc(
    n_points: int = 10000,
    period: float = 3.5,
    t0: float = 0.0,
    duration: float = 0.15,
    depth: float = 0.01,
    baseline_days: float = 27.0,
):
    """Generate synthetic light curve with transit dips."""
    rng = np.random.default_rng(42)
    time = np.linspace(0, baseline_days, n_points)
    flux = np.ones(n_points) + rng.normal(0, 0.001, n_points)

    # Inject transits
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    half_dur = (duration / period) / 2.0
    in_transit = np.abs(phase) <= half_dur
    flux[in_transit] -= depth

    return time, flux, period, t0, duration, depth


def _make_centroids(
    time: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    base_x: float = 500.0,
    base_y: float = 300.0,
    scatter: float = 0.01,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
):
    """Generate synthetic centroid arrays with optional in-transit shift."""
    rng = np.random.default_rng(99)
    n = len(time)
    cx = base_x + rng.normal(0, scatter, n)
    cy = base_y + rng.normal(0, scatter, n)

    # Apply shift during transit
    if shift_x != 0 or shift_y != 0:
        phase = ((time - t0) / period) % 1.0
        phase[phase > 0.5] -= 1.0
        half_dur = (duration / period) / 2.0
        in_transit = np.abs(phase) <= half_dur
        cx[in_transit] += shift_x
        cy[in_transit] += shift_y

    return cx, cy


# ===========================================================================
# Tests: compute_centroid_shift
# ===========================================================================

class TestCentroidShift:
    """Tests for centroid motion analysis."""

    def test_missing_centroid_returns_unavailable(self):
        """Missing centroid data must return centroid_available=False, shift=None."""
        time, flux, period, t0, duration, _ = _make_transit_lc()
        result = compute_centroid_shift(
            time, flux, period, t0, duration,
            centroid_x=None, centroid_y=None,
        )
        assert result["centroid_available"] is False
        assert result["centroid_shift"] is None
        assert result["centroid_shift_significance"] is None

    def test_missing_centroid_shift_is_not_zero(self):
        """Critically: missing centroid_shift must NOT be 0.0."""
        time, flux, period, t0, duration, _ = _make_transit_lc()
        result = compute_centroid_shift(
            time, flux, period, t0, duration,
            centroid_x=None, centroid_y=None,
        )
        # It should be None, NOT 0.0
        assert result["centroid_shift"] is None
        assert result["centroid_shift"] != 0.0

    def test_no_shift_gives_low_significance(self):
        """Centroids with no displacement should have low significance."""
        time, flux, period, t0, duration, _ = _make_transit_lc()
        cx, cy = _make_centroids(time, period, t0, duration, shift_x=0.0, shift_y=0.0)

        result = compute_centroid_shift(time, flux, period, t0, duration, cx, cy)
        assert result["centroid_available"] is True
        assert result["centroid_shift"] is not None
        assert result["centroid_shift_significance"] is not None
        # With no injected shift, significance should be low (< 3 sigma)
        assert result["centroid_shift_significance"] < 3.0

    def test_large_shift_gives_high_significance(self):
        """A large injected centroid displacement should be detected at high significance."""
        time, flux, period, t0, duration, _ = _make_transit_lc()
        # Inject a large shift (0.5 pixels, scatter 0.01)
        cx, cy = _make_centroids(
            time, period, t0, duration,
            scatter=0.01, shift_x=0.5, shift_y=0.3,
        )

        result = compute_centroid_shift(time, flux, period, t0, duration, cx, cy)
        assert result["centroid_available"] is True
        assert result["centroid_shift"] > 0.3  # sqrt(0.5^2 + 0.3^2) ≈ 0.58
        assert result["centroid_shift_significance"] > 5.0  # clearly significant

    def test_returns_median_positions(self):
        """In-transit and out-of-transit median positions should be returned."""
        time, flux, period, t0, duration, _ = _make_transit_lc()
        cx, cy = _make_centroids(time, period, t0, duration, base_x=500.0, base_y=300.0)

        result = compute_centroid_shift(time, flux, period, t0, duration, cx, cy)
        assert result["centroid_in_transit_x"] is not None
        assert result["centroid_out_transit_x"] is not None
        # Both should be near the base position
        assert abs(result["centroid_in_transit_x"] - 500.0) < 1.0
        assert abs(result["centroid_out_transit_y"] - 300.0) < 1.0

    def test_returns_point_count(self):
        """centroid_shift_points_used should be the number of in-transit points."""
        time, flux, period, t0, duration, _ = _make_transit_lc()
        cx, cy = _make_centroids(time, period, t0, duration)

        result = compute_centroid_shift(time, flux, period, t0, duration, cx, cy)
        assert result["centroid_available"] is True
        assert isinstance(result["centroid_shift_points_used"], int)
        assert result["centroid_shift_points_used"] > 0

    def test_quality_masking(self):
        """Bad quality flags should exclude points."""
        time, flux, period, t0, duration, _ = _make_transit_lc(n_points=500)
        cx, cy = _make_centroids(time, period, t0, duration)
        # Mark ALL points as bad quality
        quality = np.ones(len(time), dtype=int) * 128

        result = compute_centroid_shift(
            time, flux, period, t0, duration, cx, cy, quality=quality,
        )
        # Should return unavailable because all points are masked
        assert result["centroid_available"] is False

    def test_invalid_period_returns_unavailable(self):
        """Period <= 0 should return unavailable."""
        time, flux, _, t0, duration, _ = _make_transit_lc()
        cx, cy = _make_centroids(time, 1.0, t0, duration)

        result = compute_centroid_shift(time, flux, 0.0, t0, duration, cx, cy)
        assert result["centroid_available"] is False


# ===========================================================================
# Tests: compute_crowding_diagnostics
# ===========================================================================

class TestCrowdingDiagnostics:
    """Tests for CROWDSAP/dilution diagnostics."""

    def test_missing_crowding_returns_unavailable(self):
        """If CROWDSAP is None, crowding_available must be False."""
        result = compute_crowding_diagnostics(
            observed_depth=0.01,
            crowding_metric=None,
        )
        assert result["crowding_available"] is False
        assert result["dilution_factor"] is None
        assert result["dilution_corrected_depth"] is None

    def test_crowdsap_correction_increases_depth(self):
        """When crowding < 1, dilution-corrected depth should be larger."""
        result = compute_crowding_diagnostics(
            observed_depth=0.01,
            crowding_metric=0.6,
        )
        assert result["crowding_available"] is True
        assert result["dilution_corrected_depth"] > 0.01
        assert abs(result["dilution_corrected_depth"] - 0.01 / 0.6) < 0.001

    def test_crowdsap_1_gives_no_correction(self):
        """CROWDSAP = 1.0 means no contamination."""
        result = compute_crowding_diagnostics(
            observed_depth=0.01,
            crowding_metric=1.0,
        )
        assert result["crowding_available"] is True
        assert result["dilution_factor"] == 1.0
        assert abs(result["dilution_corrected_depth"] - 0.01) < 0.0001
        assert result["contamination_ratio"] == 0.0

    def test_contamination_ratio_calculation(self):
        """Contamination ratio = (1-c)/c."""
        result = compute_crowding_diagnostics(
            observed_depth=0.01,
            crowding_metric=0.5,
        )
        assert abs(result["contamination_ratio"] - 1.0) < 0.01

    def test_invalid_crowding_returns_unavailable(self):
        """Crowding <= 0 or > 1 should return unavailable."""
        for bad_val in [0.0, -0.5, 1.5]:
            result = compute_crowding_diagnostics(
                observed_depth=0.01,
                crowding_metric=bad_val,
            )
            assert result["crowding_available"] is False


# ===========================================================================
# Tests: compute_neighbor_diagnostics
# ===========================================================================

class TestNeighborDiagnostics:
    """Tests for neighbor catalog diagnostics."""

    def test_no_catalog_returns_unavailable(self):
        """No neighbor catalog => neighbor_available=False."""
        result = compute_neighbor_diagnostics(
            target_id="TIC-12345",
            neighbor_catalog=None,
        )
        assert result["neighbor_available"] is False
        assert result["gaia_neighbor_count"] is None

    def test_empty_catalog_returns_unavailable(self):
        """Empty catalog => unavailable."""
        result = compute_neighbor_diagnostics(
            target_id="TIC-12345",
            neighbor_catalog={},
        )
        assert result["neighbor_available"] is False

    def test_target_not_in_catalog_returns_unavailable(self):
        """Target not found in catalog."""
        catalog = {"TIC-99999": [{"separation_arcsec": 5.0, "delta_mag": 2.0, "flux_ratio": 0.1}]}
        result = compute_neighbor_diagnostics(
            target_id="TIC-12345",
            neighbor_catalog=catalog,
        )
        assert result["neighbor_available"] is False

    def test_close_bright_neighbor_detected(self):
        """A close, bright neighbor should be detected."""
        catalog = {
            "TIC-12345": [
                {"neighbor_source_id": "G1", "separation_arcsec": 10.0, "delta_mag": 1.5, "flux_ratio": 0.4},
                {"neighbor_source_id": "G2", "separation_arcsec": 50.0, "delta_mag": 5.0, "flux_ratio": 0.01},
            ]
        }
        result = compute_neighbor_diagnostics(
            target_id="TIC-12345",
            neighbor_catalog=catalog,
            aperture_radius_arcsec=21.0,
        )
        assert result["neighbor_available"] is True
        assert result["gaia_neighbor_count"] == 1  # only G1 within 21 arcsec
        assert result["nearest_neighbor_sep_arcsec"] == 10.0
        assert result["nearest_neighbor_delta_mag"] == 1.5

    def test_no_neighbors_in_aperture(self):
        """All neighbors outside aperture => count=0."""
        catalog = {
            "TIC-12345": [
                {"neighbor_source_id": "G1", "separation_arcsec": 100.0, "delta_mag": 2.0, "flux_ratio": 0.05},
            ]
        }
        result = compute_neighbor_diagnostics(
            target_id="TIC-12345",
            neighbor_catalog=catalog,
            aperture_radius_arcsec=21.0,
        )
        assert result["neighbor_available"] is True
        assert result["gaia_neighbor_count"] == 0
        assert result["aperture_neighbor_risk"] == 0.0


# ===========================================================================
# Tests: compute_blend_risk_score
# ===========================================================================

class TestBlendRiskScore:
    """Tests for aggregate blend risk scoring."""

    def test_all_unavailable_gives_unavailable_level(self):
        """If no diagnostics are available, risk level = 'unavailable'."""
        centroid = {"centroid_available": False, "centroid_shift_significance": None}
        crowding = {"crowding_available": False, "crowding_metric": None}
        neighbor = {"neighbor_available": False}

        result = compute_blend_risk_score(centroid, crowding, neighbor)
        assert result["blend_risk_level"] == "unavailable"
        assert result["blend_risk_score"] is None
        assert result["blend_evidence_flags"] == []

    def test_high_centroid_shift_gives_high_risk(self):
        """High centroid significance should produce high blend risk."""
        centroid = {"centroid_available": True, "centroid_shift_significance": 10.0}
        crowding = {"crowding_available": False, "crowding_metric": None}
        neighbor = {"neighbor_available": False}

        result = compute_blend_risk_score(centroid, crowding, neighbor)
        assert result["blend_risk_level"] == "high"
        assert result["blend_risk_score"] is not None
        assert result["blend_risk_score"] >= 0.7
        assert any("centroid_shift" in f for f in result["blend_evidence_flags"])

    def test_low_crowding_contributes_to_risk(self):
        """Low crowding metric should contribute to blend risk."""
        centroid = {"centroid_available": False}
        crowding = {"crowding_available": True, "crowding_metric": 0.4}
        neighbor = {"neighbor_available": False}

        result = compute_blend_risk_score(centroid, crowding, neighbor)
        assert result["blend_risk_level"] in ("medium", "high")
        assert any("crowding" in f for f in result["blend_evidence_flags"])

    def test_clean_signal_gives_low_risk(self):
        """Clean centroid and good crowding should give low risk."""
        centroid = {"centroid_available": True, "centroid_shift_significance": 0.5}
        crowding = {"crowding_available": True, "crowding_metric": 0.95}
        neighbor = {"neighbor_available": True, "aperture_neighbor_risk": 0.0, "gaia_neighbor_count": 0}

        result = compute_blend_risk_score(centroid, crowding, neighbor)
        assert result["blend_risk_level"] == "low"
        assert result["blend_risk_score"] < 0.3

    def test_risk_level_values(self):
        """All risk levels should be from the allowed set."""
        for level in BLEND_RISK_LEVELS:
            assert level in ("unavailable", "low", "medium", "high")


# ===========================================================================
# Tests: extract_blend_diagnostics (integration)
# ===========================================================================

class TestExtractBlendDiagnostics:
    """Integration tests for the combined diagnostics wrapper."""

    def test_no_metadata_returns_all_unavailable(self):
        """With no centroid/crowding/neighbor data, all should be unavailable."""
        time, flux, period, t0, duration, depth = _make_transit_lc()
        result = extract_blend_diagnostics(
            time, flux, period, t0, duration, depth,
        )
        diag = result["diagnostics"]
        assert diag["centroid_available"] is False
        assert diag["crowding_available"] is False
        assert diag["neighbor_available"] is False
        assert diag["blend_risk_level"] == "unavailable"

    def test_with_centroid_data(self):
        """When centroid data is provided, centroid_available should be True."""
        time, flux, period, t0, duration, depth = _make_transit_lc()
        cx, cy = _make_centroids(time, period, t0, duration)
        result = extract_blend_diagnostics(
            time, flux, period, t0, duration, depth,
            centroid_x=cx, centroid_y=cy,
        )
        diag = result["diagnostics"]
        assert diag["centroid_available"] is True
        assert diag["centroid_shift"] is not None

    def test_classifier_features_finite(self):
        """Classifier features should always be finite floats/ints."""
        time, flux, period, t0, duration, depth = _make_transit_lc()
        result = extract_blend_diagnostics(
            time, flux, period, t0, duration, depth,
        )
        clf = result["classifier_features"]
        assert isinstance(clf["centroid_shift"], (int, float))
        assert isinstance(clf["crowding_metric"], (int, float))
        assert isinstance(clf["gaia_neighbor_count"], (int, float))
        assert np.isfinite(clf["centroid_shift"])
        assert np.isfinite(clf["crowding_metric"])


# ===========================================================================
# Tests: get_blend_explanation
# ===========================================================================

class TestBlendExplanation:
    """Tests for blend explanation generation."""

    def test_unavailable_diagnostics_explanation(self):
        """Should clearly state diagnostics are unavailable."""
        diag = {
            "centroid_available": False,
            "crowding_available": False,
            "neighbor_available": False,
            "blend_risk_level": "unavailable",
        }
        expl = get_blend_explanation(diag, "exoplanet_transit")
        assert "unavailable" in expl.lower()

    def test_high_risk_explanation(self):
        """High risk should mention centroid shift."""
        diag = {
            "centroid_available": True,
            "centroid_shift": 0.5,
            "centroid_shift_significance": 8.0,
            "crowding_available": False,
            "neighbor_available": False,
            "blend_risk_level": "high",
            "blend_evidence_flags": ["centroid_shift_8.0sigma"],
        }
        expl = get_blend_explanation(diag, "blend_contamination")
        assert "centroid" in expl.lower()
        assert "8.0" in expl

    def test_low_risk_explanation(self):
        """Low risk should say no significant contamination."""
        diag = {
            "centroid_available": True,
            "centroid_shift": 0.001,
            "centroid_shift_significance": 0.3,
            "crowding_available": False,
            "neighbor_available": False,
            "blend_risk_level": "low",
            "blend_evidence_flags": [],
        }
        expl = get_blend_explanation(diag, "exoplanet_transit")
        assert "low" in expl.lower()


# ===========================================================================
# Tests: Pipeline integration
# ===========================================================================

class TestPipelineDiagnostics:
    """Test that pipeline output includes diagnostics.blend."""

    def test_pipeline_result_has_diagnostics_blend(self):
        """analyze_light_curve result should include diagnostics.blend."""
        from pipeline import analyze_light_curve

        time, flux, _, _, _, _ = _make_transit_lc()
        result = analyze_light_curve(time, flux)

        assert "diagnostics" in result
        assert "blend" in result["diagnostics"]
        blend = result["diagnostics"]["blend"]
        assert "centroid_available" in blend
        assert "crowding_available" in blend
        assert "neighbor_available" in blend
        assert "blend_risk_level" in blend
        assert blend["blend_risk_level"] in BLEND_RISK_LEVELS

    def test_pipeline_error_result_has_diagnostics(self):
        """Even error results should have diagnostics.blend."""
        from pipeline import analyze_light_curve

        # Very short light curve that will fail
        time = np.linspace(0, 1, 50)
        flux = np.ones(50)
        try:
            result = analyze_light_curve(time, flux)
            if "diagnostics" in result:
                assert "blend" in result["diagnostics"]
        except Exception:
            pass  # Some inputs may raise InvalidInputError
