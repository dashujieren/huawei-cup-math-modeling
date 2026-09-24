"""问题三通信连续性独立验收。

本模块不导入求解器，接收求解器重放的三维分段轨迹和官方 Q3 表行。
对运动链路，使用区间端点的最大距离及最不利遮挡损耗证明整段可用；
证明不足时可以找到确定失联反例，但绝不会把有限采样点通过当作连续通过。
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping


DECISION_GUARD_DB = 1e-6
TIME_TOL_S = 1e-7


@dataclass(frozen=True)
class _RelayWindow:
    relay_id: str
    hover: tuple[float, float, float]
    ready_s: float
    service_end_s: float
    backhaul_status: str
    backhaul_margin_db: float | None


@dataclass(frozen=True)
class _Allocation:
    row_index: int
    sortie_id: str
    stage: str
    start_s: float
    end_s: float
    mode: str
    relay_id: str


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} 不是数字：{value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} 不是有限数字：{value!r}")
    return number


def _bidirectional_limit(a: Any, b: Any, system_loss_db: float) -> float:
    """独立重算上下行预算，取更严格的一向。"""
    forward = (a.tx_power_dbm + a.gain_dbi + b.gain_dbi - system_loss_db -
               b.sensitivity_dbm - b.fade_margin_db)
    backward = (b.tx_power_dbm + b.gain_dbi + a.gain_dbi - system_loss_db -
                a.sensitivity_dbm - a.fade_margin_db)
    return min(forward, backward)


def _fspl_db(distance_m: float, frequency_mhz: float) -> float:
    if distance_m <= 0 or frequency_mhz <= 0:
        raise ValueError("传播距离或频率不合法")
    return 32.45 + 20.0 * math.log10(frequency_mhz) + 20.0 * math.log10(
        distance_m / 1000.0)


def _point_certificate(links: Any, moving: tuple[float, float, float],
                       moving_radio: str, fixed: tuple[float, float, float],
                       fixed_radio: str) -> dict[str, Any]:
    try:
        distance = math.dist(moving, fixed)
        obstructed = bool(links.dem.obstructed(moving, fixed))
        budget = _bidirectional_limit(links.radios[moving_radio],
                                      links.radios[fixed_radio],
                                      _finite_number(links.system_loss_db, "系统损耗"))
        loss = (_fspl_db(distance, _finite_number(links.frequency_mhz, "频率")) +
                (_finite_number(links.obstruction_loss_db, "遮挡损耗")
                 if obstructed else 0.0))
        margin = _finite_number(budget - loss, "点链路余量")
    except (ValueError, KeyError, ArithmeticError, TypeError, AttributeError) as exc:
        return {"status": "UNVERIFIED", "method": "exact_point",
                "margin_db": None, "reason": f"无法计算精确点链路：{exc}"}
    if margin > DECISION_GUARD_DB:
        status = "PASS"
    elif margin < -DECISION_GUARD_DB:
        status = "FAIL"
    else:
        status = "UNVERIFIED"
    return {"status": status, "method": "exact_point", "margin_db": margin,
            "obstructed": obstructed,
            "reason": "独立计算 DEM 遮挡、自由空间损耗与双向预算"}


def certify_link_interval(segment: Any, start_s: float, end_s: float,
                          fixed: tuple[float, float, float],
                          moving_radio: str, fixed_radio: str,
                          links: Any) -> dict[str, Any]:
    """证明一段线性轨迹到固定端点的连续链路。

    返回 JSON 可序列化字典。PASS 是全区间证明，FAIL 含确切失联时刻；
    UNVERIFIED 表示保守最坏界不通过、已检查点又未提供反例。
    """
    start = _finite_number(start_s, "区间起点")
    end = _finite_number(end_s, "区间终点")
    if (start < segment.start_s - TIME_TOL_S or
            end > segment.end_s + TIME_TOL_S):
        raise ValueError("待验收区间不在轨迹分段内")
    start = max(start, segment.start_s)
    end = min(end, segment.end_s)
    if not start < end:
        raise ValueError("裁剪后待验收区间非正长")
    if not all(math.isfinite(float(v)) for v in fixed):
        raise ValueError("固定通信端点坐标不合法")
    p0, p1 = segment.position(start), segment.position(end)
    stationary = p0 == p1
    if stationary:
        point = _point_certificate(links, p0, moving_radio, fixed, fixed_radio)
        point.update({"witness_s": start if point["status"] == "FAIL" else None,
                      "worst_distance_m": math.dist(p0, fixed),
                      "pessimistic_obstructed": None})
        return point

    # 对线性轨迹到固定点的距离平方是凸二次函数，闭区间最大值在端点。
    distance = max(math.dist(p0, fixed), math.dist(p1, fixed))
    try:
        budget = _bidirectional_limit(links.radios[moving_radio],
                                      links.radios[fixed_radio],
                                      _finite_number(links.system_loss_db, "系统损耗"))
        frequency = _finite_number(links.frequency_mhz, "频率")
        obstruction = _finite_number(links.obstruction_loss_db, "遮挡损耗")
        if not math.isfinite(budget):
            raise ValueError("双向预算不是有限值")
        # 遮挡状态可以任意变化，但每时刻附加损耗不超过以下上界。
        worst_loss = _fspl_db(distance, frequency) + max(0.0, obstruction)
        worst_margin = budget - worst_loss
    except (ValueError, KeyError, ArithmeticError) as exc:
        return {"status": "UNVERIFIED", "method": "interval_bound",
                "margin_db": None, "worst_distance_m": distance,
                "pessimistic_obstructed": True, "witness_s": None,
                "reason": f"无法计算保守区间界：{exc}"}
    if worst_margin > DECISION_GUARD_DB:
        return {"status": "PASS", "method": "worst_obstruction_endpoint_distance",
                "margin_db": worst_margin, "worst_distance_m": distance,
                "pessimistic_obstructed": True, "witness_s": None,
                "reason": "端点最大距离且全时段按遮挡计损仍满足双向预算"}

    # 点验收只用于查找 FAIL 反例，绝不用来宣称整个连续区间 PASS。
    checks = []
    for instant in (start, (start + end) / 2.0, end):
        point = _point_certificate(links, segment.position(instant), moving_radio,
                                   fixed, fixed_radio)
        checks.append((instant, point))
        if point["status"] == "FAIL":
            return {"status": "FAIL", "method": "exact_point_counterexample",
                    "margin_db": point["margin_db"],
                    "worst_distance_m": distance,
                    "pessimistic_obstructed": True, "witness_s": instant,
                    "reason": "区间内存在精确计算出的失联时刻"}
    checked_margins = [point["margin_db"] for _, point in checks
                       if point["margin_db"] is not None]
    return {"status": "UNVERIFIED", "method": "insufficient_interval_bound",
            "margin_db": worst_margin, "worst_distance_m": distance,
            "pessimistic_obstructed": True, "witness_s": None,
            "checked_min_margin_db": min(checked_margins) if checked_margins else None,
            "reason": "最坏遮挡界不能证明整段；有限点通过不等于连续通过"}


def _issue(issues: list[dict[str, str]], kind: str, detail: str,
           status: str = "FAIL") -> None:
    issues.append({"status": status, "kind": kind, "detail": detail})


def _relay_windows(relay_rows: Iterable[Mapping[str, Any]], links: Any,
                   issues: list[dict[str, str]]) -> dict[str, _RelayWindow]:
    relays: dict[str, _RelayWindow] = {}
    for row_index, row in enumerate(relay_rows, 1):
        relay_id = str(row.get("中继架次编号", "")).strip()
        if not relay_id or relay_id in relays:
            _issue(issues, "relay_id", f"中继表第 {row_index} 行编号为空或重复")
            continue
        try:
            start = _finite_number(row["开始时刻（s）"], "中继开始")
            ready = _finite_number(row["建链完成时刻（s）"], "建链完成")
            service_end = _finite_number(row["服务结束时刻（s）"], "服务结束")
            returned = _finite_number(row["返回O01时刻（s）"], "中继返回")
            lon = _finite_number(row["悬停经度（°）"], "中继悬停经度")
            lat = _finite_number(row["悬停纬度（°）"], "中继悬停纬度")
            altitude = _finite_number(row["悬停海拔（m）"], "中继悬停海拔")
            if not 0 <= start < ready <= service_end <= returned:
                raise ValueError("中继准备/建链/服务/返航时序无效")
            x, y = links.dem.to_xy.transform(lon, lat)
            hover = (_finite_number(x, "悬停 x"), _finite_number(y, "悬停 y"), altitude)
            backhaul = _point_certificate(links, hover, "relay_backhaul",
                                           links.gateway, "gateway")
        except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
            _issue(issues, "relay_row", f"中继表第 {row_index} 行无效：{exc}")
            continue
        relays[relay_id] = _RelayWindow(relay_id, hover, ready, service_end,
                                        backhaul["status"], backhaul["margin_db"])
        if backhaul["status"] != "PASS":
            _issue(issues, "relay_backhaul",
                   f"{relay_id} 固定回传链路 {backhaul['status']}：{backhaul['reason']}",
                   backhaul["status"])
    return relays


def _allocations(comm_rows: Iterable[Mapping[str, Any]],
                 issues: list[dict[str, str]]) -> list[_Allocation]:
    allocations = []
    for row_index, row in enumerate(comm_rows, 1):
        try:
            sortie_id = str(row["运输架次编号"]).strip()
            stage = str(row["通信阶段"]).strip()
            mode = str(row["保障方式"]).strip()
            relay_id = str(row.get("中继架次编号", "")).strip()
            start = _finite_number(row["开始时刻（s）"], "通信开始")
            end = _finite_number(row["结束时刻（s）"], "通信结束")
            if not sortie_id or not stage or not start < end:
                raise ValueError("架次/阶段为空或通信区间非正长")
            if mode == "直连" and relay_id:
                raise ValueError("直连行不得填写中继架次")
            if mode == "中继" and not relay_id:
                raise ValueError("中继行缺少中继架次编号")
            if mode not in {"直连", "中继"}:
                raise ValueError(f"未知保障方式 {mode!r}")
        except (KeyError, ValueError, TypeError) as exc:
            _issue(issues, "comm_row", f"通信表第 {row_index} 行无效：{exc}")
            continue
        allocations.append(_Allocation(row_index, sortie_id, stage,
                                       start, end, mode, relay_id))
    return allocations


def verify_continuous_coverage(
    trajectories: Mapping[str, list[Any]], links: Any,
    comm_rows: Iterable[Mapping[str, Any]],
    relay_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """核实 Q3 官方通信表声明是否覆盖 Q2 每架次的完整飞行与投送轨迹。

    数据行使用官方中文表头；时间建议以可往返的浮点文本（如
    ``repr(float)``）输出。小于等于 ``TIME_TOL_S`` 的时序舍入差会被
    吸附到上一行/轨迹端点，并对吸附后的完整区间重新证明链路；超过
    容差的空档或重叠一律 FAIL。返回值可用
    ``json.dump(..., allow_nan=False)`` 保存。
    """
    issues: list[dict[str, str]] = []
    if not trajectories:
        _issue(issues, "empty_trajectories", "未提供任何运输轨迹")
    relays = _relay_windows(relay_rows, links, issues)
    allocations = _allocations(comm_rows, issues)
    by_sortie: dict[str, list[_Allocation]] = defaultdict(list)
    used_relays: set[str] = set()
    for row in allocations:
        by_sortie[row.sortie_id].append(row)
        if row.mode == "中继":
            used_relays.add(row.relay_id)
    for relay_id in sorted(set(relays) - used_relays):
        _issue(issues, "unused_relay", f"中继架次 {relay_id} 未保障任何运输区间")
    for sortie_id in sorted(set(by_sortie) - set(trajectories)):
        _issue(issues, "unknown_sortie", f"通信表引用未知运输架次 {sortie_id}")

    checks: list[dict[str, Any]] = []
    for sortie_id, segments in sorted(trajectories.items()):
        ordered_segments = sorted(segments, key=lambda seg: seg.start_s)
        if not ordered_segments:
            _issue(issues, "empty_trajectory", f"{sortie_id} 没有飞行轨迹")
            continue
        for before, after in zip(ordered_segments, ordered_segments[1:]):
            if abs(before.end_s - after.start_s) > TIME_TOL_S:
                _issue(issues, "trajectory_discontinuity",
                       f"{sortie_id} 轨迹在 {before.end_s} 与 {after.start_s} 之间不连续")
        rows = sorted(by_sortie[sortie_id], key=lambda row: (row.start_s, row.end_s))
        cursor = ordered_segments[0].start_s
        if not rows:
            _issue(issues, "missing_coverage", f"{sortie_id} 没有通信保障行")
            continue
        for row_index, raw_row in enumerate(rows):
            row = raw_row
            if abs(row.start_s - cursor) <= TIME_TOL_S:
                row = replace(row, start_s=cursor)
            else:
                relation = "空档" if row.start_s > cursor else "重叠"
                _issue(issues, "coverage_partition",
                       f"{sortie_id} 通信保障在 {cursor} 与 {row.start_s} 之间{relation}")
            if (row_index == len(rows) - 1 and
                    abs(row.end_s - ordered_segments[-1].end_s) <= TIME_TOL_S):
                row = replace(row, end_s=ordered_segments[-1].end_s)
            if row.start_s >= row.end_s:
                _issue(issues, "coverage_partition",
                       f"{sortie_id} 第 {row.row_index} 行吸附后区间非正长")
                continue
            cursor = row.end_s
            if (row.start_s < ordered_segments[0].start_s - TIME_TOL_S or
                    row.end_s > ordered_segments[-1].end_s + TIME_TOL_S):
                _issue(issues, "coverage_scope", f"{sortie_id} 第 {row.row_index} 行超出飞行轨迹")
            relay = relays.get(row.relay_id) if row.mode == "中继" else None
            if row.mode == "中继":
                if relay is None:
                    _issue(issues, "relay_reference",
                           f"{sortie_id} 第 {row.row_index} 行引用未知中继 {row.relay_id}")
                    continue
                if row.start_s < relay.ready_s or row.end_s > relay.service_end_s:
                    _issue(issues, "relay_window",
                           f"{sortie_id} 第 {row.row_index} 行不在 {row.relay_id} 的已建链服务时段")
            intersections = 0
            for segment in ordered_segments:
                start = max(row.start_s, segment.start_s)
                end = min(row.end_s, segment.end_s)
                if abs(start - segment.start_s) <= TIME_TOL_S:
                    start = segment.start_s
                if abs(end - segment.end_s) <= TIME_TOL_S:
                    end = segment.end_s
                if not start < end:
                    continue
                intersections += 1
                if row.stage != segment.stage:
                    _issue(issues, "stage_mismatch",
                           f"{sortie_id} 第 {row.row_index} 行写 {row.stage}，轨迹为 {segment.stage}")
                fixed = links.gateway if relay is None else relay.hover
                target_radio = "gateway" if relay is None else "relay_access"
                try:
                    certificate = certify_link_interval(
                        segment, start, end, fixed, "transport", target_radio, links)
                except (ValueError, KeyError, ArithmeticError) as exc:
                    certificate = {"status": "UNVERIFIED", "method": "invalid_interval",
                                   "margin_db": None, "worst_distance_m": None,
                                   "pessimistic_obstructed": None, "witness_s": None,
                                   "reason": f"无法复核该通信分段：{exc}"}
                checks.append({"sortie_id": sortie_id, "stage": segment.stage,
                               "start_s": start, "end_s": end,
                               "mode": row.mode, "relay_id": row.relay_id,
                               "comm_row_index": row.row_index, **certificate})
            if not intersections:
                _issue(issues, "coverage_scope",
                       f"{sortie_id} 第 {row.row_index} 行未覆盖任何飞行轨迹")
        if abs(cursor - ordered_segments[-1].end_s) > TIME_TOL_S:
            _issue(issues, "coverage_partition",
                   f"{sortie_id} 通信保障终点 {cursor} 不等于返航 {ordered_segments[-1].end_s}")

    statuses = [item["status"] for item in issues] + [item["status"] for item in checks]
    status = ("FAIL" if "FAIL" in statuses else
              "UNVERIFIED" if "UNVERIFIED" in statuses else "PASS")
    certified_margins = [item["margin_db"] for item in checks
                         if item["status"] == "PASS" and item["margin_db"] is not None]
    return {
        "status": status,
        "time_tolerance_s": TIME_TOL_S,
        "certified_interval_count": sum(x["status"] == "PASS" for x in checks),
        "failed_interval_count": sum(x["status"] == "FAIL" for x in checks),
        "unverified_interval_count": sum(x["status"] == "UNVERIFIED" for x in checks),
        "worst_certified_margin_db": min(certified_margins) if certified_margins else None,
        "issues": issues,
        "checks": checks,
    }
