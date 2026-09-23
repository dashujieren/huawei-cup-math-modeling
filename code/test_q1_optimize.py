"""第一问精确组批：候选覆盖、目标与独立回代测试。"""

from pathlib import Path

import pytest

from q1_baseline import _read_inputs, find_latest_validated_run, run_model
from q1_optimize import enumerate_candidates, replay_selection, solve_partition, zone_lower_bounds


def small_instance():
    model = {
        "type_id": "T", "empty_mass_kg": 5.0, "max_payload_kg": 10.0,
        "capacity_m3": 1.0, "cruise_speed_mps": 10.0,
        "range_empty_m": 10000.0, "range_full_m": 5000.0,
        "energy_use_kwh": 10.0, "return_soc_min": 0.2,
        "prep_s": 30.0, "load_per_box_s": 2.0,
        "handoff_base_s": 5.0, "handoff_per_box_s": 1.0,
        "climb_speed_mps": 2.0, "descent_speed_mps": 2.0,
        "climb_efficiency": 0.72,
    }
    arc = {"distance_m": 100.0, "climb_m": 0.0, "descent_m": 0.0}
    boxes = [
        {"box_id": "B01", "zone_id": "S001", "mass_kg": 6.0, "volume_m3": 0.2},
        {"box_id": "B02", "zone_id": "S001", "mass_kg": 4.0, "volume_m3": 0.2},
        {"box_id": "B03", "zone_id": "S001", "mass_kg": 4.0, "volume_m3": 0.2},
    ]
    arcs = {("O01", "S001"): arc, ("S001", "O01"): arc}
    return boxes, {"T": model}, arcs


def test_exact_cover_and_replay_on_small_instance():
    boxes, models, arcs = small_instance()
    candidates = enumerate_candidates(boxes, models, arcs)
    assert not any(len(item.box_ids) == 3 for item in candidates)
    solved = solve_partition(candidates, [box["box_id"] for box in boxes],
                             objective="sorties", time_limit_s=20)
    assert solved["status"] == "optimal"
    assert len(solved["selected"]) == 2
    plan = replay_selection(solved["selected"], boxes, models, arcs)
    assert sorted(box_id for batch in plan for box_id in batch["box_ids"]) == ["B01", "B02", "B03"]
    assert all(batch["feasible"] for batch in plan)


def test_fixed_sorties_and_time_cap():
    boxes, models, arcs = small_instance()
    candidates = enumerate_candidates(boxes, models, arcs)
    energy = solve_partition(candidates, [box["box_id"] for box in boxes],
                             objective="energy", exact_sorties=2, time_limit_s=20)
    assert energy["status"] == "optimal"
    total_time = sum(item.operation_s for item in energy["selected"])
    constrained = solve_partition(candidates, [box["box_id"] for box in boxes],
                                  objective="energy", exact_sorties=2,
                                  time_cap_s=total_time + 1e-6, time_limit_s=20)
    assert constrained["status"] == "optimal"
    assert sum(item.operation_s for item in constrained["selected"]) <= total_time + 1e-5


def test_official_candidate_replay_matches_baseline_physics():
    try:
        data_run = find_latest_validated_run()
    except FileNotFoundError:
        pytest.skip("没有已验收的官方数据")
    boxes, models, arcs = _read_inputs(data_run)
    baseline = run_model(data_run)
    ids = {box["box_id"] for box in boxes}
    assert len(ids) == 80
    assert len(baseline["ffd"]) == 18
    candidates = enumerate_candidates(boxes, models, arcs)
    assert any(set(item.box_ids) == set(baseline["ffd"][0]["box_ids"])
               for item in candidates)
    lower = zone_lower_bounds(boxes, models, baseline["max_payloads"], baseline["ffd"])
    assert sum(row["sortie_lower_bound"] for row in lower) == 18
    assert all(row["matched_lower_bound"] for row in lower)
