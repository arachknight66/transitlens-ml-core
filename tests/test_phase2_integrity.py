import numpy as np
from diagnostics.contracts import get_default_diagnostics_dict, validate_schema
from diagnostics.gaia_neighbors import run_gaia_neighbor_query
from diagnostics.phase_windows import fold_phase, assign_cycle_numbers

def test_default_contract_uses_null_for_missing_measurements():
    result=get_default_diagnostics_dict()
    assert result["centroid_available"] is False and result["centroid_shift_pixels"] is None
    assert result["gaia_available"] is False and result["gaia_neighbor_count"] is None
    assert validate_schema(result)

def test_gaia_offline_cache_miss_is_unavailable_not_clean(tmp_path):
    cfg={"Gaia":{"cache_directory":str(tmp_path),"offline_only":True}}
    result=run_gaia_neighbor_query("TIC-1",10.,20.,cfg)
    assert result["gaia_available"] is False
    assert result["gaia_neighbor_count"] is None
    assert result["gaia_quality"]=="unavailable_cache_miss"

def test_phase_and_cycle_assignment_are_stable_across_gaps():
    time=np.array([0.,1.,2.,20.,21.]); phase=fold_phase(time,2.,0.)
    cycles=assign_cycle_numbers(time,2.,0.)
    assert np.all((phase>=-.5)&(phase<.5))
    assert cycles.tolist()==[0,0,1,10,10]
