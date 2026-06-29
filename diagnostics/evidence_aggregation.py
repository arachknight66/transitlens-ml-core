# evidence_aggregation.py
# ----------------------
# Evidence aggregation: transparent risk scoring, levels, and candidate routing.

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

def run_evidence_aggregation(
    oddeven_res: dict,
    secondary_res: dict,
    morphology_res: dict,
    harmonics_res: dict,
    centroid_res: dict,
    diff_imaging_res: dict,
    source_loc_res: dict,
    gaia_res: dict,
    crowding_res: dict,
    multi_aperture_res: dict,
    ephemeris_res: dict,
    config: dict,
) -> dict:
    """
    Aggregates all morphological and spatial evidence into transparent risk levels and routing recommendations.
    """
    agg_config = config.get("EvidenceAggregation", {})
    eb_weights = agg_config.get("eb_evidence_weights", {})
    blend_weights = agg_config.get("blend_evidence_weights", {})
    
    eb_flags = []
    blend_flags = []
    contradictory_flags = []
    missing_diagnostics = []
    
    # ── Gather EB Evidence ──
    eb_score_num = 0.0
    eb_weight_sum = 0.0
    
    # 1. Odd/Even asymmetry
    if oddeven_res.get("odd_even_available"):
        sig = oddeven_res.get("odd_even_significance", 0.0)
        weight = eb_weights.get("odd_even_asymmetry", 3.0)
        # Map significance to [0, 1] range: 0 at sig=0, 1 at sig >= 5
        score = min(1.0, sig / 5.0)
        eb_score_num += score * weight
        eb_weight_sum += weight
        
        if oddeven_res.get("odd_even_evidence_flag"):
            eb_flags.append(f"odd_even_depth_difference_{sig:.1f}sigma")
    else:
        missing_diagnostics.append("odd_even")
        
    # 2. Secondary eclipse
    if secondary_res.get("secondary_available"):
        sig = secondary_res.get("secondary_significance", 0.0)
        weight = eb_weights.get("secondary_eclipse", 3.0)
        # Map significance to [0, 1] range
        score = min(1.0, sig / 5.0)
        eb_score_num += score * weight
        eb_weight_sum += weight
        
        if secondary_res.get("secondary_evidence_flag"):
            eb_flags.append(f"secondary_eclipse_detected_{sig:.1f}sigma")
            if secondary_res.get("secondary_quality") == "eb_like":
                eb_flags.append("large_secondary_primary_ratio")
    else:
        missing_diagnostics.append("secondary_eclipse")
        
    # 3. V-shape profile
    if morphology_res.get("morphology_available"):
        v_score = morphology_res.get("v_shape_score", 0.0)
        weight = eb_weights.get("v_shape", 2.0)
        eb_score_num += v_score * weight
        eb_weight_sum += weight
        
        if morphology_res.get("morphology_evidence_flag"):
            eb_flags.append(f"v_shape_score_{v_score:.2f}")
    else:
        missing_diagnostics.append("morphology")
        
    # 4. Harmonics (ellipsoidal / beaming / reflection)
    if harmonics_res.get("harmonic_available"):
        sig_ell = harmonics_res.get("ellipsoidal_significance", 0.0)
        weight = eb_weights.get("harmonic_ellipsoidal", 2.0)
        score = min(1.0, sig_ell / 5.0)
        eb_score_num += score * weight
        eb_weight_sum += weight
        
        if harmonics_res.get("harmonic_evidence_flag"):
            eb_flags.append(f"ellipsoidal_variability_{sig_ell:.1f}sigma")
    else:
        missing_diagnostics.append("harmonic_variability")
        
    # ── Gather Blend/Contamination Evidence ──
    blend_score_num = 0.0
    blend_weight_sum = 0.0
    
    # 1. Centroid Shift
    if centroid_res.get("centroid_available"):
        sig = centroid_res.get("centroid_shift_significance", 0.0)
        weight = blend_weights.get("centroid_shift", 3.0)
        score = min(1.0, sig / 5.0)
        blend_score_num += score * weight
        blend_weight_sum += weight
        
        if centroid_res.get("centroid_evidence_flag"):
            blend_flags.append(f"centroid_shift_{sig:.1f}sigma")
    else:
        missing_diagnostics.append("centroid_shift")
        
    # 2. Difference image source offset
    if source_loc_res.get("source_target_offset_pixels") is not None:
        offset = source_loc_res.get("source_target_offset_pixels", 0.0)
        weight = blend_weights.get("difference_offset", 3.0)
        # Map offset in pixels to [0, 1] range: 0 at 0 pixels, 1 at >= 2 pixels
        score = min(1.0, offset / 2.0)
        blend_score_num += score * weight
        blend_weight_sum += weight
        
        if source_loc_res.get("difference_image_evidence_flag"):
            blend_flags.append(f"difference_source_offset_{offset:.2f}pixels")
    else:
        missing_diagnostics.append("difference_imaging")
        
    # 3. Gaia neighbor contamination
    if gaia_res.get("gaia_available"):
        cont_ratio = gaia_res.get("summed_neighbor_flux_ratio", 0.0)
        weight = blend_weights.get("gaia_neighbor", 2.0)
        # Map ratio to [0, 1]: 0 at 0.0, 1 at >= 0.50
        score = min(1.0, cont_ratio / 0.50)
        blend_score_num += score * weight
        blend_weight_sum += weight
        
        if gaia_res.get("gaia_evidence_flag"):
            blend_flags.append(f"gaia_flux_contamination_{cont_ratio:.2f}")
    else:
        missing_diagnostics.append("gaia_neighbors")
        
    # 4. Crowding (CROWDSAP)
    if crowding_res.get("crowding_available"):
        crowdsap = crowding_res.get("crowdsap", 1.0)
        weight = blend_weights.get("crowding_dilution", 1.5)
        # Map (1 - crowdsap) to [0, 1]: 0 at crowdsap=1, 1 at crowdsap <= 0.50
        score = min(1.0, (1.0 - crowdsap) / 0.50)
        blend_score_num += score * weight
        blend_weight_sum += weight
        
        if crowding_res.get("crowding_evidence_flag"):
            blend_flags.append(f"aperture_crowding_crowdsap_{crowdsap:.2f}")
    else:
        missing_diagnostics.append("crowdsap")
        
    # 5. Multi-aperture chi2
    if multi_aperture_res.get("multi_aperture_available"):
        p_val = multi_aperture_res.get("aperture_depth_consistency_p_value", 1.0)
        weight = blend_weights.get("aperture_trend", 2.0)
        # Map p_value to score: p=1 -> 0, p=0 -> 1
        score = float(1.0 - p_val)
        blend_score_num += score * weight
        blend_weight_sum += weight
        
        if multi_aperture_res.get("multi_aperture_evidence_flag"):
            blend_flags.append("aperture_depth_inconsistency")
    else:
        missing_diagnostics.append("multi_aperture")
        
    # 6. Ephemeris match
    if ephemeris_res.get("ephemeris_match_evidence_flag"):
        weight = blend_weights.get("ephemeris_match", 3.0)
        blend_score_num += 1.0 * weight
        blend_weight_sum += weight
        
        blend_flags.append(f"ephemeris_matched_variable_{ephemeris_res['matched_source']}")
    else:
        if ephemeris_res.get("source_catalogue") == "none":
            # catalog was missing
            pass
            
    # Calculate final scores
    eb_score = float(eb_score_num / eb_weight_sum) if eb_weight_sum > 0 else None
    blend_score = float(blend_score_num / blend_weight_sum) if blend_weight_sum > 0 else None
    
    # Map scores to levels
    def get_level(score):
        if score is None:
            return "unavailable"
        if score < 0.30:
            return "low"
        if score < 0.70:
            return "medium"
        return "high"
        
    eb_level = get_level(eb_score)
    blend_level = get_level(blend_score)
    
    # ── Contradictions Check ──
    # If we have clean morphology but significant centroid shift
    if (morphology_res.get("morphology_available") and not morphology_res.get("morphology_evidence_flag")) and centroid_res.get("centroid_evidence_flag"):
        contradictory_flags.append("clean_morphology_with_centroid_shift")
        
    # If we have V-shape profile but zero centroid/spatial shift
    if morphology_res.get("morphology_evidence_flag") and (centroid_res.get("centroid_available") and not centroid_res.get("centroid_evidence_flag")):
        contradictory_flags.append("v_shape_with_zero_centroid_shift")
        
    # ── Vetting Routing Recommendation ──
    route = "review_required"
    reason = "insufficient diagnostics"
    review_required = True
    
    # Count how many independent families are high quality/available
    indep_eb_families = int(sum([
        oddeven_res.get("odd_even_available", False),
        secondary_res.get("secondary_available", False),
        morphology_res.get("morphology_available", False),
        harmonics_res.get("harmonic_available", False)
    ]))
    
    indep_blend_families = int(sum([
        centroid_res.get("centroid_available", False),
        diff_imaging_res.get("difference_image_available", False),
        gaia_res.get("gaia_available", False),
        crowding_res.get("crowding_available", False),
        multi_aperture_res.get("multi_aperture_available", False)
    ]))
    
    if eb_level == "high" and blend_level == "low":
        route = "eclipsing_binary"
        reason = "high eclipsing-binary morphology evidence, low spatial contamination"
        review_required = False
    elif blend_level == "high" and eb_level == "low":
        # Check if localized offset is off-target
        if source_loc_res.get("difference_image_evidence_flag") or centroid_res.get("centroid_evidence_flag"):
            route = "blend_contamination"
            reason = "strong spatial offset localization (off-target contaminant)"
            review_required = False
        else:
            route = "review_required"
            reason = "high blend risk but unresolved source localization offset"
            review_required = True
    elif eb_level == "high" and blend_level == "high":
        route = "eclipsing_binary" # Likely background/nearby EB
        reason = "high eclipsing binary and high spatial blend flags (nearby/background eclipsing binary)"
        review_required = False
    elif eb_level == "low" and blend_level == "low":
        if indep_eb_families >= 2 and indep_blend_families >= 2:
            route = "exoplanet_transit"
            reason = "clean morphology profiles and clean spatial diagnostics"
            review_required = False
        else:
            route = "review_required"
            reason = "clean diagnostics but insufficient independent evidence coverage"
            review_required = True
    else:
        route = "review_required"
        reason = "moderate risk scores or contradictory evidence"
        review_required = True
        
    # Check if contradictory flags force review
    if contradictory_flags:
        route = "review_required"
        reason = f"contradictory evidence: {', '.join(contradictory_flags)}"
        review_required = True
        
    return {
        "eb_risk_score": round(eb_score, 6) if eb_score is not None else None,
        "blend_risk_score": round(blend_score, 6) if blend_score is not None else None,
        "eb_risk_level": eb_level,
        "blend_risk_level": blend_level,
        "eb_evidence_flags": eb_flags,
        "blend_evidence_flags": blend_flags,
        "independent_eb_evidence_count": indep_eb_families,
        "independent_blend_evidence_count": indep_blend_families,
        "contradictory_evidence_flags": contradictory_flags,
        "recommended_route": route,
        "recommendation_reason": reason,
        "review_required": review_required,
        "missing_diagnostics": missing_diagnostics,
        "quality_flags": contradictory_flags,
        "threshold_policy_version": "2.0.0",
    }
