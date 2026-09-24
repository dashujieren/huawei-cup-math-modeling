"""Independent resource and flight-physics audit for Q3 relay sorties.

This module consumes only the official relay table and the cleaned input data.
It deliberately does not read the searcher's candidate or resource calendars.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from q3_0_baseline import DemGrid, LinkEvaluator


RELAY_COLUMNS = (
    "中继架次编号", "中继无人机编号", "能源组件编号", "开始时刻（s）",
    "悬停经度（°）", "悬停纬度（°）", "悬停海拔（m）", "建链完成时刻（s）",
    "服务结束时刻（s）", "返回O01时刻（s）", "架次能耗（kWh）",
)
TIME_TOL_S = 2e-5
ENERGY_TOL_KWH = 2e-6
HEIGHT_TOL_M = 2e-3
RESOURCE_TOL_S = 1e-8
HARD_ENERGY_TOL_KWH = 1e-9


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _number(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 不是数值") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} 不是有限数值")
    return value


def _recharge_duration_s(return_soc: float, full_charge_s: float) -> float:
    """Official 0–90% / 90–100% split, recomputed without scheduler helpers."""
    if not (0.0 <= return_soc <= 1.0 and full_charge_s > 0):
        raise ValueError("SOC 或充满时长无效")
    if return_soc < 0.9:
        return full_charge_s * (0.65 * (0.9 - return_soc) / 0.9 + 0.35)
    return full_charge_s * 0.35 * (1.0 - return_soc) / 0.1


def _flight(arc: dict[str, float], params: dict[str, float]) -> tuple[float, float]:
    """Reconstruct one leg from length and ascent, not the scheduler's cache."""
    climb = float(arc["climb_m"])
    descent = float(arc["descent_m"])
    distance = float(arc["distance_m"])
    cruise_s = distance / params["cruise_speed_mps"]
    flight_s = (climb / params["climb_speed_mps"] + cruise_s +
                descent / params["descent_speed_mps"])
    energy_kwh = (params["cruise_power_kw"] * cruise_s / 3600.0 +
                  params["takeoff_mass_kg"] * 9.80665 * climb /
                  (3_600_000.0 * params["climb_efficiency"]))
    return flight_s, energy_kwh


def validate_relay_rows(relay_rows: list[dict[str, str]], data_run: Path,
                        links: LinkEvaluator, dem: DemGrid) -> dict[str, Any]:
    """Audit official Q3 relay rows; return a JSON-serializable PASS/FAIL report.

    A missing/invalid source table or a malformed sortie is a FAIL, not a silent
    skip. The resource clocks are rebuilt chronologically from each audited row.
    """
    checks: list[dict[str, str]] = []
    issues: list[dict[str, str]] = []

    def add(name: str, passed: bool, detail: str = "",
            mission_id: str = "") -> None:
        record = {"check": name, "status": "PASS" if passed else "FAIL",
                  "mission_id": mission_id, "detail": detail}
        checks.append(record)
        if not passed:
            issues.append(record)

    def result(total_energy: float = 0.0, audited_count: int = 0,
               uav_count: int = 0, component_count: int = 0) -> dict[str, Any]:
        return {
            "status": "FAIL" if issues else "PASS",
            "checks": checks,
            "issues": issues,
            "first_failure": issues[0] if issues else None,
            "summary": {
                "relay_sorties": len(relay_rows) if isinstance(relay_rows, list) else 0,
                "audited_sorties": audited_count,
                "uav_count": uav_count,
                "component_count": component_count,
                "total_reported_energy_kwh": round(total_energy, 9),
            },
        }

    if not isinstance(relay_rows, list) or any(not isinstance(row, dict)
                                                for row in relay_rows):
        add("table_type", False, "中继架次必须是字典行列表")
        return result()
    for index, row in enumerate(relay_rows, 1):
        missing = [field for field in RELAY_COLUMNS if field not in row]
        if missing:
            add("official_columns", False,
                f"第 {index} 行缺少字段：{', '.join(missing)}")
    if issues:
        return result()

    try:
        clean = data_run / "clean"
        type_rows = _read_csv(clean / "relay_type.csv")
        uav_rows = _read_csv(clean / "relay_uavs.csv")
        unit_rows = _read_csv(clean / "relay_energy_units.csv")
        pool_rows = _read_csv(clean / "relay_energy_pool.csv")
        center_rows = [row for row in _read_csv(clean / "nodes.csv")
                       if row.get("node_id") == "O01"]
        if not (len(type_rows) == 1 and len(pool_rows) == 1 and
                len(center_rows) == 1):
            raise ValueError("R 型号、能源池或 O01 节点缺失/重复")
        if len(uav_rows) != 2 or len(unit_rows) != 6:
            raise ValueError("中继资源不是 2 架无人机和 6 个能源组件")
        required = (
            "takeoff_mass_kg", "cruise_speed_mps", "cruise_power_kw",
            "energy_use_kwh", "return_soc_min", "prep_s", "link_setup_s",
            "turnaround_s", "climb_speed_mps", "descent_speed_mps",
            "climb_efficiency", "hover_power_kw", "comm_extra_power_kw",
            "max_hover_agl_m",
        )
        params = {key: _number(type_rows[0][key], key) for key in required}
        params["full_charge_s"] = _number(pool_rows[0]["full_charge_s"],
                                            "full_charge_s")
        if any(params[key] <= 0 for key in (
                "takeoff_mass_kg", "cruise_speed_mps", "cruise_power_kw",
                "energy_use_kwh", "climb_speed_mps", "descent_speed_mps",
                "climb_efficiency", "full_charge_s", "max_hover_agl_m")):
            raise ValueError("飞行/充电参数必须为正")
        if not (0 < params["return_soc_min"] < 1):
            raise ValueError("返航 SOC 下限无效")
        if any(params[key] < 0 for key in ("prep_s", "link_setup_s",
                                            "turnaround_s", "hover_power_kw",
                                            "comm_extra_power_kw")):
            raise ValueError("时间或悬停功率参数为负")
        uav_ids = [row["uav_id"] for row in uav_rows]
        unit_ids = [row["unit_id"] for row in unit_rows]
        if len(set(uav_ids)) != 2 or len(set(unit_ids)) != 6:
            raise ValueError("中继无人机或能源组件编号重复")
        center = center_rows[0]
        home = (_number(center["x_m"], "O01 x_m"),
                _number(center["y_m"], "O01 y_m"),
                _number(center["operation_alt_m"], "O01 operation_alt_m"))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        add("official_resources", False, str(exc))
        return result()
    add("official_resources", True, "2 架中继机、6 个组件和 O01 坐标已加载")
    add("gateway_location", math.hypot(home[0] - links.gateway[0],
                                        home[1] - links.gateway[1]) <= 2.0,
        "G01 的平面坐标应与 O01 相同")

    parsed: list[dict[str, Any]] = []
    seen_missions: set[str] = set()
    total_energy = 0.0
    numeric_fields = {
        "start": "开始时刻（s）", "lon": "悬停经度（°）",
        "lat": "悬停纬度（°）", "alt": "悬停海拔（m）",
        "ready": "建链完成时刻（s）", "end": "服务结束时刻（s）",
        "return": "返回O01时刻（s）", "energy": "架次能耗（kWh）",
    }
    for index, row in enumerate(relay_rows, 1):
        mission_id = row["中继架次编号"].strip()
        if not mission_id or mission_id in seen_missions:
            add("mission_id_unique", False,
                f"第 {index} 行中继架次编号为空或重复", mission_id)
        else:
            add("mission_id_unique", True, mission_id=mission_id)
            seen_missions.add(mission_id)
        uav_id, unit_id = row["中继无人机编号"].strip(), row["能源组件编号"].strip()
        add("known_uav", uav_id in uav_ids, uav_id, mission_id)
        add("known_component", unit_id in unit_ids, unit_id, mission_id)
        try:
            values = {name: _number(row[column], column)
                      for name, column in numeric_fields.items()}
        except ValueError as exc:
            add("finite_numeric_fields", False, str(exc), mission_id)
            continue
        add("finite_numeric_fields", True, mission_id=mission_id)
        add("geographic_coordinates",
            -180.0 <= values["lon"] <= 180.0 and
            -90.0 <= values["lat"] <= 90.0,
            "悬停点经纬度应落在合法范围", mission_id)
        if not (-180.0 <= values["lon"] <= 180.0 and
                -90.0 <= values["lat"] <= 90.0):
            continue
        parsed.append({"mission_id": mission_id, "uav_id": uav_id,
                       "unit_id": unit_id, **values})
        total_energy += values["energy"]

    uav_ready = {uav_id: 0.0 for uav_id in uav_ids}
    unit_ready = {unit_id: 0.0 for unit_id in unit_ids}
    audited = 0
    for item in sorted(parsed, key=lambda entry: (entry["start"],
                                                   entry["mission_id"])):
        mission_id = item["mission_id"]
        start, ready = item["start"], item["ready"]
        end, returned = item["end"], item["return"]
        energy = item["energy"]
        add("time_order", start >= -RESOURCE_TOL_S and
            ready >= start - RESOURCE_TOL_S and
            end >= ready - RESOURCE_TOL_S and
            returned >= end - RESOURCE_TOL_S,
            f"start={start}, ready={ready}, end={end}, return={returned}",
            mission_id)
        add("nonnegative_energy", energy >= -HARD_ENERGY_TOL_KWH,
            f"reported={energy:.9f} kWh", mission_id)
        try:
            x, y = dem.to_xy.transform(item["lon"], item["lat"])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("经纬度投影得到非有限坐标")
            hover = (x, y, item["alt"])
            ground = dem.terrain_at(x, y)
            agl = hover[2] - ground
            add("hover_agl", agl > -HEIGHT_TOL_M and
                agl <= params["max_hover_agl_m"] + HEIGHT_TOL_M,
                f"AGL={agl:.6f} m, limit={params['max_hover_agl_m']:.6f} m",
                mission_id)
            backhaul = links.backhaul(hover)
            add("bidirectional_backhaul", backhaul.available,
                f"margin={backhaul.margin_db:.6f} dB", mission_id)
            outward = dem.arc_geometry(home, hover)
            inward = dem.arc_geometry(hover, home)
            add("relay_arc_altitude",
                hover[2] <= min(outward["cruise_alt_m"],
                                inward["cruise_alt_m"]) + HEIGHT_TOL_M,
                "PDF 基线航段巡航高度不能低于悬停点", mission_id)
            outward_s, outward_energy = _flight(outward, params)
            inward_s, inward_energy = _flight(inward, params)
            setup = params["link_setup_s"]
            expected_ready = start + params["prep_s"] + outward_s + setup
            expected_return = end + inward_s
            expected_energy = (outward_energy + inward_energy +
                               (params["hover_power_kw"] +
                                params["comm_extra_power_kw"]) *
                               (setup + max(0.0, end - ready)) / 3600.0)
            add("link_ready_replay",
                abs(ready - expected_ready) <= TIME_TOL_S,
                f"reported={ready:.9f}, replay={expected_ready:.9f}",
                mission_id)
            add("return_time_replay",
                abs(returned - expected_return) <= TIME_TOL_S,
                f"reported={returned:.9f}, replay={expected_return:.9f}",
                mission_id)
            add("energy_replay",
                abs(energy - expected_energy) <= ENERGY_TOL_KWH,
                f"reported={energy:.9f}, replay={expected_energy:.9f} kWh",
                mission_id)
        except (ValueError, KeyError, OverflowError, ZeroDivisionError) as exc:
            add("physical_replay", False, str(exc), mission_id)

        soc = 1.0 - energy / params["energy_use_kwh"]
        add("return_soc",
            soc >= params["return_soc_min"] -
            HARD_ENERGY_TOL_KWH / params["energy_use_kwh"] and
            soc <= 1.0 + HARD_ENERGY_TOL_KWH / params["energy_use_kwh"],
            f"SOC={soc:.9f}, minimum={params['return_soc_min']:.9f}",
            mission_id)
        uav_id, unit_id = item["uav_id"], item["unit_id"]
        if uav_id in uav_ready:
            add("uav_turnaround", start >= uav_ready[uav_id] - RESOURCE_TOL_S,
                f"start={start:.9f}, earliest={uav_ready[uav_id]:.9f}",
                mission_id)
            uav_ready[uav_id] = max(uav_ready[uav_id],
                                    returned + params["turnaround_s"])
        if unit_id in unit_ready:
            add("component_recharge", start >= unit_ready[unit_id] - RESOURCE_TOL_S,
                f"start={start:.9f}, earliest={unit_ready[unit_id]:.9f}",
                mission_id)
            if 0.0 <= soc <= 1.0:
                charging = _recharge_duration_s(soc, params["full_charge_s"])
                unit_ready[unit_id] = max(unit_ready[unit_id],
                                          returned + charging)
            else:
                # Invalid SOC must not leave the component apparently reusable.
                unit_ready[unit_id] = math.inf
        audited += 1
    return result(total_energy, audited, len(uav_ids), len(unit_ids))
