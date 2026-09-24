"""Focused tests for the independent Q3 relay-resource audit."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from q3_3_resources import validate_relay_rows


class IdentityProjection:
    def transform(self, lon: float, lat: float) -> tuple[float, float]:
        return lon, lat


class FlatDem:
    to_xy = IdentityProjection()

    def terrain_at(self, x: float, y: float) -> float:
        return 100.0

    def arc_geometry(self, a: tuple[float, float, float],
                     b: tuple[float, float, float]) -> dict[str, float]:
        return {
            "distance_m": math.hypot(b[0] - a[0], b[1] - a[1]),
            "cruise_alt_m": 200.0,
            "climb_m": max(0.0, 200.0 - a[2]),
            "descent_m": max(0.0, 200.0 - b[2]),
        }


@dataclass
class Link:
    available: bool
    margin_db: float


class Links:
    gateway = (0.0, 0.0, 120.0)

    def backhaul(self, hover: tuple[float, float, float]) -> Link:
        return Link(True, 12.0)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def data_run(tmp_path: Path) -> Path:
    clean = tmp_path / "clean"
    _write_csv(clean / "relay_type.csv", [{
        "takeoff_mass_kg": "23.5", "cruise_speed_mps": "15",
        "cruise_power_kw": "1.15", "energy_use_kwh": "3.2",
        "return_soc_min": "0.2", "prep_s": "180", "link_setup_s": "30",
        "turnaround_s": "300", "climb_speed_mps": "4",
        "descent_speed_mps": "3", "climb_efficiency": "0.72",
        "hover_power_kw": "1.05", "comm_extra_power_kw": "0.05",
        "max_hover_agl_m": "300",
    }])
    _write_csv(clean / "relay_uavs.csv", [{"uav_id": "R01"},
                                           {"uav_id": "R02"}])
    _write_csv(clean / "relay_energy_units.csv", [
        {"unit_id": f"R-E{i:02d}"} for i in range(1, 7)
    ])
    _write_csv(clean / "relay_energy_pool.csv", [{"full_charge_s": "1800"}])
    _write_csv(clean / "nodes.csv", [{
        "node_id": "O01", "x_m": "0", "y_m": "0",
        "operation_alt_m": "100",
    }])
    return tmp_path


def _row(mission_id: str = "R001", uav: str = "R01",
         unit: str = "R-E01", start: float = 0.0,
         service_s: float = 100.0) -> dict[str, str]:
    outward_s = 100 / 4 + 150 / 15 + 50 / 3
    inward_s = 50 / 4 + 150 / 15 + 100 / 3
    ready = start + 180 + outward_s + 30
    end = ready + service_s
    returned = end + inward_s
    travel_energy = (1.15 * (2 * 150 / 15) / 3600 +
                     23.5 * 9.80665 * (100 + 50) / (3_600_000 * 0.72))
    energy = travel_energy + 1.1 * (30 + service_s) / 3600
    return {
        "中继架次编号": mission_id, "中继无人机编号": uav,
        "能源组件编号": unit, "开始时刻（s）": repr(start),
        "悬停经度（°）": "150.000000000", "悬停纬度（°）": "0.000000000",
        "悬停海拔（m）": "150.000000000", "建链完成时刻（s）": repr(ready),
        "服务结束时刻（s）": repr(end), "返回O01时刻（s）": repr(returned),
        "架次能耗（kWh）": f"{energy:.9f}",
    }


def _failed(report: dict, check: str) -> bool:
    return any(item["check"] == check for item in report["issues"])


def test_valid_two_sortie_reuse(data_run: Path) -> None:
    report = validate_relay_rows([_row(), _row("R002", start=800)],
                                 data_run, Links(), FlatDem())
    assert report["status"] == "PASS"
    assert report["first_failure"] is None
    assert report["summary"]["audited_sorties"] == 2


def test_energy_tamper_is_detected(data_run: Path) -> None:
    row = _row()
    row["架次能耗（kWh）"] = "0.010000000"
    report = validate_relay_rows([row], data_run, Links(), FlatDem())
    assert report["status"] == "FAIL"
    assert _failed(report, "energy_replay")


def test_return_soc_floor_is_detected(data_run: Path) -> None:
    report = validate_relay_rows([_row(service_s=9000)],
                                 data_run, Links(), FlatDem())
    assert _failed(report, "return_soc")


def test_turnaround_is_detected(data_run: Path) -> None:
    first = _row()
    second = _row("R002", unit="R-E02",
                  start=float(first["返回O01时刻（s）"]) + 20)
    report = validate_relay_rows([first, second], data_run, Links(), FlatDem())
    assert _failed(report, "uav_turnaround")
    assert not _failed(report, "component_recharge")


def test_component_recharge_is_detected(data_run: Path) -> None:
    first = _row()
    second = _row("R002", uav="R02",
                  start=float(first["返回O01时刻（s）"]) + 5)
    report = validate_relay_rows([first, second], data_run, Links(), FlatDem())
    assert _failed(report, "component_recharge")
    assert not _failed(report, "uav_turnaround")


def test_hover_height_and_replay_are_detected(data_run: Path) -> None:
    row = _row()
    row["悬停海拔（m）"] = "450.000000000"
    report = validate_relay_rows([row], data_run, Links(), FlatDem())
    assert _failed(report, "hover_agl")
    assert _failed(report, "relay_arc_altitude")


def test_unknown_ids_and_duplicate_are_detected(data_run: Path) -> None:
    rows = [_row(), _row(uav="R99", unit="R-X01", start=800)]
    report = validate_relay_rows(rows, data_run, Links(), FlatDem())
    assert _failed(report, "mission_id_unique")
    assert _failed(report, "known_uav")
    assert _failed(report, "known_component")


def test_malformed_numeric_is_fail_not_exception(data_run: Path) -> None:
    row = _row()
    row["建链完成时刻（s）"] = "nan"
    report = validate_relay_rows([row], data_run, Links(), FlatDem())
    assert report["status"] == "FAIL"
    assert _failed(report, "finite_numeric_fields")


def test_wrong_table_type_is_serializable_failure(data_run: Path) -> None:
    report = validate_relay_rows(None, data_run, Links(), FlatDem())  # type: ignore[arg-type]
    assert report["status"] == "FAIL"
    assert _failed(report, "table_type")
    json.dumps(report, allow_nan=False)


def test_link_ready_tamper_is_detected(data_run: Path) -> None:
    row = _row()
    row["建链完成时刻（s）"] = repr(float(row["建链完成时刻（s）"]) - 5)
    report = validate_relay_rows([row], data_run, Links(), FlatDem())
    assert _failed(report, "link_ready_replay")


def test_microsecond_turnaround_overlap_is_not_accepted(data_run: Path) -> None:
    first = _row()
    earliest = float(first["返回O01时刻（s）"]) + 300
    second = _row("R002", unit="R-E02", start=earliest - 1e-6)
    report = validate_relay_rows([first, second], data_run, Links(), FlatDem())
    assert _failed(report, "uav_turnaround")


def test_soc_floor_does_not_inherit_physics_replay_tolerance(data_run: Path) -> None:
    row = _row()
    row["架次能耗（kWh）"] = "2.560000005"
    report = validate_relay_rows([row], data_run, Links(), FlatDem())
    assert _failed(report, "return_soc")
