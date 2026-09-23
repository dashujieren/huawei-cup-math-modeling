"""第一问基线的物理边界、FFD 与正式附件集成测试。"""

from pathlib import Path

import pandas as pd
import pytest
import q1_baseline

from q1_baseline import (
    best_type_for_batch,
    equivalent_range_m,
    max_safe_payload_kg,
    run_model,
    save_results,
    sortie_metrics,
)


CODE_DIR = Path(__file__).resolve().parent


def toy_inputs():
    model = {
        "type_id": "T", "empty_mass_kg": 5.0, "max_payload_kg": 10.0,
        "capacity_m3": 1.0, "cruise_speed_mps": 10.0,
        "range_empty_m": 1000.0, "range_full_m": 500.0,
        "energy_use_kwh": 2.0, "return_soc_min": 0.2,
        "prep_s": 30.0, "load_per_box_s": 2.0,
        "handoff_base_s": 5.0, "handoff_per_box_s": 1.0,
        "climb_speed_mps": 2.0, "descent_speed_mps": 2.0,
        "climb_efficiency": 0.72,
    }
    outward = {"distance_m": 300.0, "climb_m": 0.0, "descent_m": 0.0}
    homeward = dict(outward)
    box = {"box_id": "B01", "zone_id": "S001", "mass_kg": 3.0, "volume_m3": 0.1}
    return model, outward, homeward, box


def test_range_boundary_and_energy_limited_payload():
    model, outward, homeward, _ = toy_inputs()
    assert equivalent_range_m(model, 0.0) == 1000.0
    assert equivalent_range_m(model, 10.0) == 500.0
    result = max_safe_payload_kg(model, outward, homeward)
    assert result["status"] == "energy_limited"
    assert 0 < result["q_max_kg"] < 10
    assert result["energy_at_qmax_kwh"] <= 1.6 + 1e-8
    unreachable = dict(outward, distance_m=500.0)
    assert max_safe_payload_kg(model, unreachable, unreachable)["status"] == "unreachable"


def test_sortie_time_and_return_soc():
    model, outward, homeward, box = toy_inputs()
    result = sortie_metrics(model, outward, homeward, [box])
    assert result["feasible"]
    assert result["flight_s"] == pytest.approx(60.0)
    assert result["operation_s"] == pytest.approx(98.0)
    assert result["return_soc"] == pytest.approx(1 - result["energy_kwh"] / 2.0)
    assert result["energy_kwh"] <= 1.6
    assert result["energy_back_kwh"] != result["energy_out_kwh"]


def test_two_box_handoff_base_is_charged_once():
    model, outward, homeward, box = toy_inputs()
    second = dict(box, box_id="B02", mass_kg=2.0)
    result = sortie_metrics(model, outward, homeward, [box, second])
    assert result["handoff_s"] == pytest.approx(7.0)
    assert result["loading_s"] == pytest.approx(4.0)
    assert result["operation_s"] == pytest.approx(101.0)


def test_volume_is_independent_of_mass():
    model, outward, homeward, box = toy_inputs()
    oversized = dict(box, volume_m3=1.01)
    assert best_type_for_batch([oversized], {"T": model}, outward, homeward) is None


def test_choose_lower_energy_type_then_type_id():
    model, outward, homeward, box = toy_inputs()
    alternative = dict(model, type_id="U", energy_use_kwh=2.5)
    chosen = best_type_for_batch([box], {"T": model, "U": alternative}, outward, homeward)
    assert chosen is not None
    assert chosen["type_id"] == "T"
    equal = dict(model, type_id="A")
    chosen_equal = best_type_for_batch([box], {"T": model, "A": equal}, outward, homeward)
    assert chosen_equal is not None
    assert chosen_equal["type_id"] == "A"


def test_one_type_unreachable_does_not_invalidate_feasible_problem(monkeypatch):
    model, outward, homeward, _ = toy_inputs()
    boxes = [{"box_id": f"B{i:03d}", "zone_id": f"S{min(i // 5 + 1, 15):03d}",
              "mass_kg": 9.475, "volume_m3": 2.011 / 80}
             for i in range(80)]
    a = dict(model, type_id="A", return_soc_min=0.99)
    b = dict(model, type_id="B", max_payload_kg=100.0, energy_use_kwh=10.0,
             range_empty_m=10000.0, range_full_m=5000.0)
    c = dict(b, type_id="C")
    arcs = {}
    for zone in {box["zone_id"] for box in boxes}:
        arcs[("O01", zone)] = dict(outward, valid_flag=True)
        arcs[(zone, "O01")] = dict(homeward, valid_flag=True)
    monkeypatch.setattr(q1_baseline, "_read_inputs", lambda _: (boxes, {"A": a, "B": b, "C": c}, arcs))
    result = run_model(Path("synthetic"))
    assert any(row["status"] == "unreachable" for row in result["max_payloads"])
    assert all(check["status"] == "PASS" for check in result["checks"])
    assert len(result["single_box"]) == 80


def test_official_data_baseline_replays_all_boxes():
    runs = sorted((CODE_DIR / "outputs").glob("run_*"), reverse=True)
    if not runs:
        pytest.skip("尚无已验收的数据运行目录")
    result = run_model(runs[0])
    assert len(result["max_payloads"]) == 45
    assert len(result["single_box"]) == 80
    assert sum(len(batch["box_ids"]) for batch in result["ffd"]) == 80
    assert len(result["ffd"]) <= len(result["single_box"])
    assert all(check["status"] == "PASS" for check in result["checks"])


def test_official_hand_calculation_anchors():
    runs = sorted((CODE_DIR / "outputs").glob("run_*"), reverse=True)
    if not runs:
        pytest.skip("尚无已验收的数据运行目录")
    types = pd.read_csv(runs[0] / "clean" / "transport_types.csv").set_index("type_id")
    arcs = pd.read_csv(runs[0] / "derived" / "arc_geometry.csv").set_index(["from_id", "to_id"])
    a = types.loc["A"].to_dict()
    c = types.loc["C"].to_dict()
    a_out = arcs.loc[("O01", "S001")].to_dict()
    a_back = arcs.loc[("S001", "O01")].to_dict()
    a_result = max_safe_payload_kg(a, a_out, a_back)
    assert a_result["energy_empty_kwh"] == pytest.approx(1.161539158, abs=1e-7)
    assert a_result["q_max_kg"] == 25.0
    assert a_result["energy_at_qmax_kwh"] == pytest.approx(1.312722009, abs=1e-7)
    c_result = max_safe_payload_kg(c, arcs.loc[("O01", "S008")].to_dict(),
                                   arcs.loc[("S008", "O01")].to_dict())
    assert c_result["q_max_kg"] == pytest.approx(58.903108, abs=1e-6)
    assert c_result["energy_at_qmax_kwh"] <= c_result["energy_limit_kwh"]


def test_export_reopens_csv_before_ready(tmp_path):
    runs = sorted((CODE_DIR / "outputs").glob("run_*"), reverse=True)
    if not runs:
        pytest.skip("尚无已验收的数据运行目录")
    result = run_model(runs[0])
    destination = tmp_path / "q1_baseline"
    save_results(result, destination)
    assert (destination / "Q1_READY.txt").exists()
    assert list(pd.read_csv(destination / "Q1_单点组批.csv").columns) == q1_baseline.TEMPLATE_COLUMNS
