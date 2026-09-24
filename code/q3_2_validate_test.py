"""问题三连续通信证明器的无需 DEM 文件的单元测试。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from q3_2_validate import certify_link_interval, verify_continuous_coverage


@dataclass(frozen=True)
class Segment:
    stage: str = "巡航"
    start_s: float = 0.0
    end_s: float = 10.0
    x0: float = 1000.0
    x1: float = 2000.0
    z: float = 100.0

    def position(self, instant: float) -> tuple[float, float, float]:
        ratio = (instant - self.start_s) / (self.end_s - self.start_s)
        return self.x0 + ratio * (self.x1 - self.x0), 0.0, self.z


@dataclass(frozen=True)
class Radio:
    tx_power_dbm: float = 30.0
    gain_dbi: float = 0.0
    sensitivity_dbm: float = -100.0
    fade_margin_db: float = 0.0


class FakeLinks:
    def __init__(self, obstructed_function=None, radio=None):
        self.gateway = (0.0, 0.0, 0.0)
        self.radios = {name: radio or Radio()
                       for name in ("transport", "gateway", "relay_access",
                                    "relay_backhaul")}
        self.system_loss_db = 0.0
        self.frequency_mhz = 2400.0
        self.obstruction_loss_db = 10.0
        self.dem = SimpleNamespace(
            to_xy=SimpleNamespace(transform=lambda lon, lat: (lon, lat)),
            obstructed=obstructed_function or (lambda _a, _b: False),
        )

    def link(self, moving, moving_radio, fixed, fixed_radio):
        raise AssertionError("验收器不得复用求解器的链路结果")


def comm(start=0.0, end=10.0, mode="直连", relay_id="", stage="巡航"):
    return {"运输架次编号": "T01", "通信阶段": stage,
            "开始时刻（s）": start, "结束时刻（s）": end,
            "保障方式": mode, "中继架次编号": relay_id}


def relay(ready=5.0, service_end=10.0):
    return {"中继架次编号": "Q01", "开始时刻（s）": 0.0,
            "悬停经度（°）": 1000.0, "悬停纬度（°）": 0.0,
            "悬停海拔（m）": 100.0, "建链完成时刻（s）": ready,
            "服务结束时刻（s）": service_end, "返回O01时刻（s）": 12.0}


def verify(rows, relay_rows=(), links=None, segment=None):
    return verify_continuous_coverage(
        {"T01": [segment or Segment()]}, links or FakeLinks(), rows, relay_rows)


def test_worst_obstruction_and_max_endpoint_distance_prove_full_interval():
    result = certify_link_interval(Segment(), 0.0, 10.0, (0.0, 0.0, 0.0),
                                   "transport", "gateway", FakeLinks())
    assert result["status"] == "PASS"
    assert result["method"] == "worst_obstruction_endpoint_distance"
    assert result["pessimistic_obstructed"] is True
    assert result["worst_distance_m"] > 2000.0


def test_exact_midpoint_counterexample_is_fail():
    weak = Radio(tx_power_dbm=28.0, sensitivity_dbm=-80.0)
    links = FakeLinks(lambda a, _b: a[0] == 1500.0, weak)
    result = certify_link_interval(Segment(), 0.0, 10.0, links.gateway,
                                   "transport", "gateway", links)
    assert result["status"] == "FAIL"
    assert result["witness_s"] == 5.0
    assert result["method"] == "exact_point_counterexample"


def test_all_sampled_points_pass_is_still_unverified():
    weak = Radio(tx_power_dbm=28.0, sensitivity_dbm=-80.0)
    links = FakeLinks(radio=weak)
    result = certify_link_interval(Segment(), 0.0, 10.0, links.gateway,
                                   "transport", "gateway", links)
    assert result["status"] == "UNVERIFIED"
    assert result["checked_min_margin_db"] > 0.0
    assert result["margin_db"] < 0.0


def test_short_outage_between_sample_times_can_never_be_marked_pass():
    weak = Radio(tx_power_dbm=28.0, sensitivity_dbm=-80.0)
    links = FakeLinks(lambda a, _b: 1299.0 < a[0] < 1301.0, weak)
    # 遮挡仅出现在 t≈3 s，端点与中点均通；采样式算法会漏报。
    assert all(not links.dem.obstructed(Segment().position(t), links.gateway)
               for t in (0.0, 5.0, 10.0))
    result = certify_link_interval(Segment(), 0.0, 10.0, links.gateway,
                                   "transport", "gateway", links)
    assert result["status"] == "UNVERIFIED"
    assert result["method"] == "insufficient_interval_bound"


def test_stationary_handoff_uses_exact_dem_link_not_pessimistic_bound():
    stationary = Segment(stage="投送", x0=2000.0, x1=2000.0)
    weak = Radio(tx_power_dbm=28.0, sensitivity_dbm=-80.0)
    links = FakeLinks(radio=weak)
    result = certify_link_interval(stationary, 0.0, 10.0, links.gateway,
                                   "transport", "gateway", links)
    assert result["status"] == "PASS"
    assert result["method"] == "exact_point"


def test_zero_distance_outside_radio_model_is_not_certified():
    stationary = Segment(x0=0.0, x1=0.0)
    links = FakeLinks()
    links.gateway = (0.0, 0.0, 100.0)
    result = certify_link_interval(stationary, 0.0, 10.0, links.gateway,
                                   "transport", "gateway", links)
    assert result["status"] == "UNVERIFIED"


def test_exact_point_check_does_not_call_solver_link_method():
    stationary = Segment(stage="投送", x0=1000.0, x1=1000.0)
    links = FakeLinks()
    links.link = lambda *_args: (_ for _ in ()).throw(AssertionError("solver link called"))
    result = certify_link_interval(stationary, 0.0, 10.0, links.gateway,
                                   "transport", "gateway", links)
    assert result["status"] == "PASS"


def test_entire_sortie_direct_pass_and_json_serializable():
    result = verify([comm()])
    assert result["status"] == "PASS"
    assert result["certified_interval_count"] == 1
    json.dumps(result, allow_nan=False)


def test_exact_direct_to_relay_handoff_and_simultaneous_backhaul_pass():
    result = verify([comm(0.0, 5.0), comm(5.0, 10.0, "中继", "Q01")], [relay()])
    assert result["status"] == "PASS"
    assert {item["mode"] for item in result["checks"]} == {"直连", "中继"}


def test_positive_length_gap_fails_even_when_both_links_pass():
    result = verify([comm(0.0, 4.0), comm(5.0, 10.0)])
    assert result["status"] == "FAIL"
    assert any(issue["kind"] == "coverage_partition" for issue in result["issues"])


def test_roundoff_gap_at_handoff_is_normalized_and_certified():
    result = verify([comm(0.0, 5.0), comm(5.0 + 1e-12, 10.0)])
    assert result["status"] == "PASS"
    assert result["checks"][1]["start_s"] == 5.0


def test_roundoff_at_sortie_endpoints_is_normalized_and_certified():
    result = verify([comm(1e-12, 10.0 - 1e-12)])
    assert result["status"] == "PASS"
    assert result["checks"][0]["start_s"] == 0.0
    assert result["checks"][0]["end_s"] == 10.0


def test_gap_larger_than_time_tolerance_still_fails():
    result = verify([comm(0.0, 5.0), comm(5.0 + 1e-5, 10.0)])
    assert result["status"] == "FAIL"


def test_positive_length_overlap_fails_even_when_both_links_pass():
    result = verify([comm(0.0, 6.0), comm(5.0, 10.0)])
    assert result["status"] == "FAIL"
    assert any(issue["kind"] == "coverage_partition" for issue in result["issues"])


def test_relay_not_ready_before_service_interval_fails():
    result = verify([comm(0.0, 4.0), comm(4.0, 10.0, "中继", "Q01")], [relay()])
    assert result["status"] == "FAIL"
    assert any(issue["kind"] == "relay_window" for issue in result["issues"])


def test_relay_backhaul_failure_is_not_hidden_by_good_access():
    weak = Radio(tx_power_dbm=28.0, sensitivity_dbm=-80.0)
    links = FakeLinks(lambda a, _b: a[0] == 1000.0, weak)
    result = verify([comm(0.0, 5.0), comm(5.0, 10.0, "中继", "Q01")],
                    [relay()], links)
    assert result["status"] == "FAIL"
    assert any(issue["kind"] == "relay_backhaul" for issue in result["issues"])


def test_stage_mismatch_is_fail():
    result = verify([comm(stage="下降")])
    assert result["status"] == "FAIL"
    assert any(issue["kind"] == "stage_mismatch" for issue in result["issues"])


def test_unknown_relay_reference_is_fail():
    result = verify([comm(mode="中继", relay_id="MISSING")])
    assert result["status"] == "FAIL"
    assert any(issue["kind"] == "relay_reference" for issue in result["issues"])


def test_sample_counterexample_propagates_to_report():
    weak = Radio(tx_power_dbm=28.0, sensitivity_dbm=-80.0)
    links = FakeLinks(lambda a, _b: a[0] == 1500.0, weak)
    result = verify([comm()], links=links)
    assert result["status"] == "FAIL"
    assert result["checks"][0]["witness_s"] == 5.0


@pytest.mark.parametrize("bad", ["nan", "inf", "not a number"])
def test_invalid_numeric_coverage_never_passes(bad):
    result = verify([comm(end=bad)])
    assert result["status"] == "FAIL"
    assert any(issue["kind"] == "comm_row" for issue in result["issues"])


def test_incomplete_final_coverage_fails():
    result = verify([comm(0.0, 9.0)])
    assert result["status"] == "FAIL"
    assert any(issue["kind"] == "coverage_partition" for issue in result["issues"])


def test_unknown_sortie_cannot_satisfy_known_sortie():
    row = comm()
    row["运输架次编号"] = "OTHER"
    result = verify([row])
    assert result["status"] == "FAIL"
    assert any(issue["kind"] == "unknown_sortie" for issue in result["issues"])


def test_direct_certificate_clips_only_roundoff_sized_boundary_error():
    segment = Segment()
    result = certify_link_interval(segment, -1e-12, 10.0 + 1e-12,
                                   (0.0, 0.0, 0.0), "transport", "gateway",
                                   FakeLinks())
    assert result["status"] == "PASS"
    with pytest.raises(ValueError):
        certify_link_interval(segment, -1e-5, 10.0, (0.0, 0.0, 0.0),
                              "transport", "gateway", FakeLinks())
