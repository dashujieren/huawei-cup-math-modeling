"""问题三方案 3：候选列整数规划 + 软任务修复的运输/中继联合排程。

本程序不读取问题二的排程结果。有限空间候选、时间网格及求解预算不能
证明无解或全局最优；只有原有独立验收全部通过才生成 Q3_READY.txt。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

import q3_1_optimize as q3
from q2_0_baseline import charge_to_full_s
from q2_1_optimize import Context, check_plan


CODE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = CODE_DIR / "3_outputs" / "3_global_joint"
EPS = 1e-6


@dataclass(frozen=True)
class BlindBlock:
    """运输候选的绝对时间盲段；任一 sites 中的点可覆盖完整区间。"""

    start_s: float
    end_s: float
    sites: frozenset[int]


@dataclass(frozen=True)
class TransportColumn:
    column_id: str
    task: q3.RouteTask
    kind: str
    option: dict[str, Any]
    profile: q3.RouteProfile
    start_s: float
    return_s: float
    charge_end_s: float
    blocks: tuple[BlindBlock, ...]

    @property
    def box_ids(self) -> tuple[str, ...]:
        return tuple(self.task.route.box_ids)


@dataclass(frozen=True)
class RelayColumn:
    column_id: str
    site_index: int
    candidate: q3.RelayCandidate
    launch_s: float
    ready_s: float
    service_end_s: float
    return_s: float
    uav_end_s: float
    charge_end_s: float
    energy_kwh: float


@dataclass
class MasterResult:
    status: str
    transport_indices: list[int]
    relay_indices: list[int]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ProfileChoice:
    task: q3.RouteTask
    kind: str
    option: dict[str, Any]
    profile: q3.RouteProfile


class Rows:
    """稀疏 MILP 行构造器；各行均是 lb <= Ax <= ub。"""

    def __init__(self, variable_count: int):
        self.variable_count = variable_count
        self.ri: list[int] = []
        self.ci: list[int] = []
        self.values: list[float] = []
        self.lb: list[float] = []
        self.ub: list[float] = []

    def add(self, terms: Iterable[tuple[int, float]], lb: float, ub: float) -> None:
        index = len(self.lb)
        for column, value in terms:
            if value:
                self.ri.append(index)
                self.ci.append(column)
                self.values.append(value)
        self.lb.append(lb)
        self.ub.append(ub)

    def constraint(self) -> LinearConstraint:
        matrix = coo_matrix(
            (self.values, (self.ri, self.ci)),
            shape=(len(self.lb), self.variable_count), dtype=float).tocsr()
        return LinearConstraint(matrix, np.asarray(self.lb), np.asarray(self.ub))


def add_capacity_rows(rows: Rows, columns: Iterable[tuple[int, float, float]],
                      capacity: int) -> int:
    """同质资源的半开占用区间；任一事件时刻的重叠数不超库存。"""
    if capacity < 0:
        raise ValueError("资源容量不能为负")
    events: dict[float, list[tuple[int, int]]] = defaultdict(list)
    for index, begin, end in columns:
        if not 0 <= begin < end:
            raise ValueError("资源占用区间必须有正长度")
        events[begin].append((1, index))
        events[end].append((-1, index))
    active: set[int] = set()
    count = 0
    previous: tuple[int, ...] | None = None
    for moment in sorted(events):
        at_time = events[moment]
        for delta, index in at_time:
            if delta == -1:
                active.remove(index)
        starts = [index for delta, index in at_time if delta == 1]
        active.update(starts)
        if starts and len(active) > capacity:
            signature = tuple(sorted(active))
            if signature != previous:
                rows.add(((index, 1.0) for index in signature),
                         -math.inf, float(capacity))
                count += 1
                previous = signature
    return count


def covers(block: BlindBlock, mission: RelayColumn) -> bool:
    return (mission.site_index in block.sites and
            mission.ready_s <= block.start_s + EPS and
            block.end_s <= mission.service_end_s + EPS)


def hard_route_pool(ctx: Context,
                    route_sets: list[tuple[str, list[q3.RouteTask]]]
                    ) -> list[q3.RouteTask]:
    """从原始组批构造可替选的硬箱路线，不依赖 Q2 排程。"""
    routes: dict[tuple[Any, ...], Any] = {}
    for _, tasks in route_sets:
        for task in tasks:
            if all(not pd.isna(ctx.boxes[bid]["hard_deadline_s"])
                   for bid in task.route.box_ids):
                routes.setdefault(tuple(task.route.visits), task.route)
    required = {bid for bid, box in ctx.boxes.items()
                if not pd.isna(box["hard_deadline_s"])}
    covered = {bid for route in routes.values() for bid in route.box_ids}
    if covered != required:
        raise ValueError(f"硬箱候选路线遗漏：{sorted(required - covered)}")
    return [q3.RouteTask(f"H{i:03d}", route)
            for i, route in enumerate(routes.values(), 1)]


def build_profile_choices(
    ctx: Context, tasks: list[q3.RouteTask],
    candidates: list[q3.RelayCandidate], links: q3.LinkEvaluator,
    positions: dict[str, tuple[float, float, float]], sample_step_s: float,
    site_tree: cKDTree | None,
    access_cache: dict[tuple[Any, ...], bool],
    show_progress: bool = False,
) -> tuple[list[ProfileChoice], dict[str, Any]]:
    choices: list[ProfileChoice] = []
    missing: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, 1):
        for kind, option in sorted(ctx.options(task.route).items()):
            profile = q3._route_profile(
                ctx, task, kind, option, positions, links, sample_step_s,
                candidates, site_tree, access_cache)
            if profile.uncovered_slices:
                missing.append({"batch_id": task.batch_id, "type_id": kind,
                                "uncovered_slices": len(profile.uncovered_slices)})
            else:
                choices.append(ProfileChoice(task, kind, option, profile))
        if show_progress and (index == 1 or index % 5 == 0 or index == len(tasks)):
            print(f"Q3 硬任务通信证书 {index}/{len(tasks)} 条路线；"
                  f"可用路线×机型 {len(choices)}", flush=True)
    return choices, {"profile_choices": len(choices),
                     "spatially_uncovered_profiles": missing}


def choose_sites(profiles: list[ProfileChoice],
                 candidates: list[q3.RelayCandidate], site_budget: int
                 ) -> tuple[set[int], dict[str, Any]]:
    """按可覆盖盲段数量做空间集合覆盖，预算耗尽需显式报告。"""
    if site_budget < 1:
        raise ValueError("中继位置预算必须为正")
    incidence: dict[int, set[tuple[int, int]]] = defaultdict(set)
    universe: set[tuple[int, int]] = set()
    for pidx, choice in enumerate(profiles):
        for slice_id, sites in choice.profile.support.items():
            token = (pidx, slice_id)
            universe.add(token)
            for site in sites:
                incidence[site].add(token)
    remaining = set(universe)
    selected: set[int] = set()
    while remaining and len(selected) < site_budget:
        site = max(incidence, key=lambda i: (
            len(incidence[i] & remaining),
            -candidates[i].travel_energy_kwh,
            candidates[i].max_service_s,
            candidates[i].candidate_id))
        benefit = incidence[site] & remaining
        if not benefit:
            break
        selected.add(site)
        remaining.difference_update(benefit)
        incidence.pop(site)
    # 集合覆盖的最小位置集只保证空间可达，并不能保证两架中继机的
    # 时间日历可排。继续保留不同网格点的替代位置供主问题交换。
    xy_uses = Counter((round(candidates[i].x_m), round(candidates[i].y_m))
                      for i in selected)
    while incidence and len(selected) < site_budget:
        site = max(incidence, key=lambda i: (
            xy_uses[(round(candidates[i].x_m),
                     round(candidates[i].y_m))] == 0,
            len(incidence[i]), -candidates[i].travel_energy_kwh,
            candidates[i].max_service_s, candidates[i].candidate_id))
        if not incidence[site]:
            break
        selected.add(site)
        xy_uses[(round(candidates[site].x_m),
                 round(candidates[site].y_m))] += 1
        incidence.pop(site)
    return selected, {
        "selected_sites": len(selected), "site_budget": site_budget,
        "profile_blind_slices": len(universe),
        "uncovered_profile_slices_after_pruning": len(remaining),
    }


def preselect_sites(
    source_slices: list[q3.TimeSlice],
    candidates: list[q3.RelayCandidate], links: q3.LinkEvaluator,
    access_cache: dict[tuple[Any, ...], bool],
    budget: int, nearest_count: int,
) -> tuple[list[q3.RelayCandidate], dict[str, Any]]:
    """先用原始飞行走廊筛位，减少每条组批路线的精确地形判定量。"""
    if budget < 1 or nearest_count < 1:
        raise ValueError("空间候选预算和近邻数必须为正")
    blind = [item for item in source_slices if not item.direct_available]
    if not blind or not candidates:
        return [], {"source_blind_slices": len(blind),
                    "preselected_sites": 0,
                    "source_slices_without_nearby_access": len(blind)}
    tree = cKDTree(np.asarray([site.hover for site in candidates], dtype=float))
    incidence: dict[int, set[int]] = defaultdict(set)
    uncertain = 0
    for slice_number, item in enumerate(blind):
        found: set[int] = set()
        for count in (nearest_count, min(len(candidates), nearest_count * 4)):
            _, indices = tree.query(item.midpoint, k=min(count, len(candidates)))
            for raw in np.atleast_1d(indices):
                site = int(raw)
                if site in found:
                    continue
                if q3._access_certified(candidates[site], item,
                                        links, access_cache):
                    found.add(site)
            if len(found) >= 2:
                break
        if not found:
            uncertain += 1
        for site in found:
            incidence[site].add(slice_number)
    remaining = set(range(len(blind)))
    selected: set[int] = set()
    while remaining and incidence and len(selected) < budget:
        site = max(incidence, key=lambda i: (
            len(incidence[i] & remaining),
            -candidates[i].travel_energy_kwh,
            candidates[i].max_service_s, candidates[i].candidate_id))
        gained = incidence.pop(site) & remaining
        if not gained:
            break
        selected.add(site)
        remaining.difference_update(gained)
    # 保留少量额外覆盖点，避免一个早期中继站把后续资源日历锁死。
    while incidence and len(selected) < budget:
        site = max(incidence, key=lambda i: (
            len(incidence[i]), -candidates[i].travel_energy_kwh,
            candidates[i].max_service_s, candidates[i].candidate_id))
        if not incidence[site]:
            break
        selected.add(site)
        incidence.pop(site)
    keep = sorted(selected)
    return [candidates[index] for index in keep], {
        "source_blind_slices": len(blind),
        "original_relay_candidates": len(candidates),
        "preselected_sites": len(keep),
        "preselection_budget": budget,
        "source_slices_without_nearby_access": uncertain,
        "source_slices_uncovered_after_budget": len(remaining),
        "nearest_count": nearest_count,
    }


def blind_blocks(profile: q3.RouteProfile,
                 selected_sites: set[int], max_block_s: float = 300.0
                 ) -> tuple[BlindBlock, ...] | None:
    """相邻盲段仅在存在共同可用位置时合并；否则分块。"""
    if max_block_s <= 0:
        raise ValueError("最长通信块必须为正")
    result: list[BlindBlock] = []
    current: BlindBlock | None = None
    for item in profile.slices:
        if item.direct_available:
            if current is not None:
                result.append(current)
                current = None
            continue
        available = profile.support[item.slice_id] & selected_sites
        if not available:
            return None
        if current is not None:
            common = current.sites & available
            if (abs(current.end_s - item.start_s) < EPS and common and
                    item.end_s - current.start_s <= max_block_s + EPS):
                current = BlindBlock(current.start_s, item.end_s,
                                     frozenset(common))
                continue
            result.append(current)
        current = BlindBlock(item.start_s, item.end_s, frozenset(available))
    if current is not None:
        result.append(current)
    return tuple(result)


def build_transport_columns(
    ctx: Context, profiles: list[ProfileChoice], selected_sites: set[int],
    start_step_s: int,
) -> tuple[list[TransportColumn], dict[str, Any]]:
    if start_step_s < 1:
        raise ValueError("运输起飞网格必须为正")
    by_type: dict[str, set[float]] = defaultdict(set)
    for battery in ctx.batteries:
        by_type[battery.type_id].add(float(battery.full_charge_s))
    if any(len(values) != 1 for values in by_type.values()):
        raise ValueError("同机型共享电池充电时间不一致，不能按同质容量聚合")
    result: list[TransportColumn] = []
    discarded = 0
    for choice in profiles:
        blocks = blind_blocks(choice.profile, selected_sites)
        if blocks is None:
            discarded += 1
            continue
        latest = q3._latest_hard_start(ctx, choice.task.route, choice.option)
        if not math.isfinite(latest) or latest < -EPS:
            discarded += 1
            continue
        starts = list(range(0, max(0, math.floor(latest / start_step_s)) *
                            start_step_s + 1, start_step_s))
        starts.append(max(0.0, latest))
        duration = float(choice.option["duration_s"])
        full = next(iter(by_type[choice.kind]))
        charge = charge_to_full_s(float(choice.option["return_soc"]), full)
        for start in sorted(set(float(value) for value in starts)):
            if start > latest + EPS:
                continue
            result.append(TransportColumn(
                f"T{len(result):06d}", choice.task, choice.kind, choice.option,
                choice.profile, start, start + duration,
                start + duration + charge,
                tuple(BlindBlock(start + block.start_s, start + block.end_s,
                                 block.sites) for block in blocks)))
    return result, {"transport_columns": len(result),
                    "discarded_profiles_after_site_pruning": discarded,
                    "start_step_s": start_step_s}


def build_relay_columns(
    transports: list[TransportColumn],
    candidates: list[q3.RelayCandidate], relay: dict[str, Any],
    ready_step_s: int,
) -> tuple[list[RelayColumn], dict[str, Any]]:
    """按实际盲段时间生成可共享的中继服务窗口。"""
    if ready_step_s < 1:
        raise ValueError("中继就绪网格必须为正")
    blocks_by_site: dict[int, list[BlindBlock]] = defaultdict(list)
    for column in transports:
        for block in column.blocks:
            for site in block.sites:
                blocks_by_site[site].append(block)
    params = relay["params"]
    power = params["hover_power_kw"] + params["comm_extra_power_kw"]
    max_energy = (1.0 - params["return_soc_min"]) * params["energy_use_kwh"]
    result: list[RelayColumn] = []
    durations = (300, 600, 900, 1200, 1800, 2400, 3600, 4800, 7200)
    for site, blocks in sorted(blocks_by_site.items()):
        candidate = candidates[site]
        lead = (params["prep_s"] + candidate.outward_s +
                params["link_setup_s"])
        ready_times = {max(lead, math.floor(block.start_s / ready_step_s) *
                           ready_step_s) for block in blocks
                       if block.start_s + EPS >= lead}
        for ready in sorted(ready_times):
            useful = [block for block in blocks
                      if ready <= block.start_s + EPS and
                      block.end_s <= ready + candidate.max_service_s + EPS]
            if not useful:
                continue
            needed = {math.ceil((block.end_s - ready) / ready_step_s) *
                      ready_step_s for block in useful}
            allowed = set(durations) | needed
            # 每个就绪时刻最多保留若干长短不同的有效服务终点。
            allowed = sorted(value for value in allowed
                             if 0 < value <= candidate.max_service_s + EPS)
            if len(allowed) > 14:
                indexes = np.linspace(0, len(allowed) - 1, 14).round().astype(int)
                allowed = [allowed[i] for i in sorted(set(indexes))]
            for duration in allowed:
                end = ready + duration
                if not any(block.end_s <= end + EPS for block in useful):
                    continue
                energy = (candidate.travel_energy_kwh + power *
                          (params["link_setup_s"] + duration) / 3600.0)
                if energy > max_energy + EPS:
                    continue
                launch = ready - lead
                returned = end + candidate.homeward_s
                soc = 1.0 - energy / params["energy_use_kwh"]
                result.append(RelayColumn(
                    f"M{len(result):06d}", site, candidate,
                    launch, ready, end, returned,
                    returned + params["turnaround_s"],
                    returned + charge_to_full_s(soc, params["full_charge_s"]),
                    energy))
    return result, {"relay_columns": len(result),
                    "sites_with_demand": len(blocks_by_site),
                    "ready_step_s": ready_step_s}


def solve_master(
    required_boxes: set[str], transports: list[TransportColumn],
    relays: list[RelayColumn], fleet_by_type: dict[str, int],
    batteries_by_type: dict[str, int], relay_fleet: int,
    relay_energy_units: int, time_limit_s: float,
) -> MasterResult:
    """在有限候选列内同时选择运输、通信和四类资源日历。"""
    started = perf_counter()
    nt, nr = len(transports), len(relays)
    if not required_boxes or not transports:
        return MasterResult("CANDIDATE_INCOMPLETE", [], [], {
            "reason": "缺少必需货箱或运输列"})
    box_to_columns: dict[str, list[int]] = defaultdict(list)
    for index, column in enumerate(transports):
        for box_id in column.box_ids:
            if box_id not in required_boxes:
                raise ValueError(f"非本轮硬任务进入主问题：{box_id}")
            box_to_columns[box_id].append(index)
    missing = sorted(required_boxes - box_to_columns.keys())
    if missing:
        return MasterResult("CANDIDATE_INCOMPLETE", [], [], {
            "boxes_without_transport_columns": missing})

    rows = Rows(nt + nr)
    for box_id in sorted(required_boxes):
        rows.add(((i, 1.0) for i in box_to_columns[box_id]), 1.0, 1.0)
    capacities: dict[str, int] = {}
    for kind in sorted({col.kind for col in transports}):
        kind_cols = [(i, col) for i, col in enumerate(transports)
                     if col.kind == kind]
        capacities[f"uav_{kind}"] = add_capacity_rows(
            rows, ((i, col.start_s, col.return_s) for i, col in kind_cols),
            fleet_by_type.get(kind, 0))
        capacities[f"battery_{kind}"] = add_capacity_rows(
            rows, ((i, col.start_s, col.charge_end_s)
                   for i, col in kind_cols), batteries_by_type.get(kind, 0))
    capacities["relay_uav"] = add_capacity_rows(
        rows, ((nt + j, col.launch_s, col.uav_end_s)
               for j, col in enumerate(relays)), relay_fleet)
    # 中继机周转时间和组件充电分别写进 uav_end_s / charge_end_s。
    capacities["relay_energy"] = add_capacity_rows(
        rows, ((nt + j, col.launch_s, col.charge_end_s)
               for j, col in enumerate(relays)), relay_energy_units)

    by_site: dict[int, list[tuple[int, RelayColumn]]] = defaultdict(list)
    for j, mission in enumerate(relays):
        by_site[mission.site_index].append((j, mission))
    uncovered_columns: set[int] = set()
    coverage_rows = 0
    for i, transport in enumerate(transports):
        for block in transport.blocks:
            eligible = {j for site in block.sites
                        for j, mission in by_site.get(site, ())
                        if covers(block, mission)}
            if not eligible:
                uncovered_columns.add(i)
                break
            rows.add([(i, -1.0),
                      *((nt + j, 1.0) for j in sorted(eligible))],
                     0.0, math.inf)
            coverage_rows += 1
    if uncovered_columns:
        for i in sorted(uncovered_columns):
            rows.add([(i, 1.0)], 0.0, 0.0)
    if any(all(i in uncovered_columns for i in box_to_columns[bid])
           for bid in required_boxes):
        return MasterResult("CANDIDATE_INCOMPLETE", [], [], {
            "reason": "至少一箱的全部运输候选缺少中继服务窗口",
            "boxes_without_covered_columns": sorted(
                bid for bid in required_boxes if all(
                    i in uncovered_columns for i in box_to_columns[bid])),
            "transport_columns": nt, "relay_columns": nr})
    matrix = rows.constraint()
    costs = np.asarray([
        1.0 + 0.02 * float(col.option.get("energy_kwh", 0.0)) +
        0.00005 * col.start_s for col in transports
    ] + [
        1.2 + 0.02 * col.energy_kwh for col in relays
    ], dtype=float)
    solution = milp(
        costs, integrality=np.ones(nt + nr, dtype=np.int8),
        bounds=Bounds(np.zeros(nt + nr), np.ones(nt + nr)),
        constraints=matrix,
        options={"time_limit": time_limit_s, "mip_rel_gap": 0.02,
                 "presolve": True})
    diagnostics = {
        "solver_status": int(solution.status), "solver_message": str(solution.message),
        "elapsed_s": round(perf_counter() - started, 2),
        "transport_columns": nt, "relay_columns": nr,
        "coverage_rows": coverage_rows,
        "resource_rows": capacities,
        "disabled_uncovered_transport_columns": len(uncovered_columns),
        "constraint_rows": len(rows.lb),
    }
    if solution.x is None:
        return MasterResult("SEARCH_INCOMPLETE", [], [], diagnostics)
    chosen = np.rint(solution.x)
    lhs = matrix.A @ chosen
    valid = (np.all(np.abs(solution.x - chosen) < 1e-5) and
             np.all(lhs >= matrix.lb - 1e-5) and
             np.all(lhs <= matrix.ub + 1e-5))
    if not valid:
        return MasterResult("FAILED_MILP_RECHECK", [], [], diagnostics)
    ti = [i for i in range(nt) if chosen[i] > 0.5]
    ri = [j for j in range(nr) if chosen[nt + j] > 0.5]
    diagnostics["selected_transport_columns"] = len(ti)
    diagnostics["selected_relay_columns"] = len(ri)
    diagnostics["objective"] = float(costs @ chosen)
    return MasterResult("CANDIDATE_FEASIBLE", ti, ri, diagnostics)


def assign_entities(
    columns: list[tuple[int, str, float, float]],
    entities_by_type: dict[str, list[str]],
) -> dict[int, str]:
    """区间图按开始时刻贪心着色；容量约束通过时应能分配实体。"""
    free_at = {entity: 0.0 for names in entities_by_type.values()
               for entity in names}
    assigned: dict[int, str] = {}
    for index, kind, begin, end in sorted(columns,
                                          key=lambda row: (row[2], row[3], row[0])):
        available = [entity for entity in entities_by_type.get(kind, ())
                     if free_at[entity] <= begin + EPS]
        if not available:
            raise ValueError(f"主问题容量约束后仍无法分配实体：{kind}, t={begin:.3f}")
        entity = min(available, key=lambda name: (free_at[name], name))
        assigned[index] = entity
        free_at[entity] = end
    return assigned


def state_from_master(
    ctx: Context, transport_columns: list[TransportColumn],
    relay_columns: list[RelayColumn], selected: MasterResult,
) -> q3.JointState:
    """整数列还原成题面实体编号、逐箱时刻和连续通信分配。"""
    chosen_t = [(i, transport_columns[i]) for i in selected.transport_indices]
    chosen_r = [(j, relay_columns[j]) for j in selected.relay_indices]
    uav_by_type: dict[str, list[str]] = defaultdict(list)
    battery_by_type: dict[str, list[str]] = defaultdict(list)
    for uav in ctx.uavs:
        uav_by_type[uav.type_id].append(uav.uav_id)
    for battery in ctx.batteries:
        battery_by_type[battery.type_id].append(battery.battery_id)
    uav_assign = assign_entities(
        [(i, col.kind, col.start_s, col.return_s) for i, col in chosen_t],
        uav_by_type)
    battery_assign = assign_entities(
        [(i, col.kind, col.start_s, col.charge_end_s)
         for i, col in chosen_t], battery_by_type)
    relay_data = q3.load_relay_resources(ctx.data_run)
    relay_uavs = assign_entities(
        [(j, "R", col.launch_s, col.uav_end_s) for j, col in chosen_r],
        {"R": relay_data["uav_ids"]})
    relay_units = assign_entities(
        [(j, "R", col.launch_s, col.charge_end_s) for j, col in chosen_r],
        {"R": relay_data["unit_ids"]})
    missions = [q3.RelayMission(
        f"R{order:03d}", relay_uavs[j], relay_units[j], col.candidate,
        col.launch_s, col.ready_s, col.service_end_s, col.return_s,
        col.energy_kwh)
        for order, (j, col) in enumerate(
            sorted(chosen_r, key=lambda pair: (pair[1].launch_s, pair[0])), 1)]
    mission_by_column = {j: mission for (j, _), mission in zip(
        sorted(chosen_r, key=lambda pair: (pair[1].launch_s, pair[0])),
        missions)}
    batteries = {item.battery_id: item for item in ctx.batteries}
    state = q3._empty_joint_state()
    state.missions = missions
    for i, column in sorted(chosen_t,
                            key=lambda pair: (pair[1].start_s, pair[0])):
        battery_id = battery_assign[i]
        sortie, deliveries = q3._make_sortie(
            ctx, column.task, column.kind, column.option,
            uav_assign[i], batteries[battery_id], column.start_s)
        if column.task.batch_id in state.trajectories:
            raise AssertionError("同一批次被主问题重复选择")
        shifted, slices = q3._shift_template(
            column.profile.segments, column.profile.slices,
            column.start_s, len(state.slices))
        for local_id, item in enumerate(slices):
            if item.direct_available:
                continue
            possible = [mission_by_column[j] for j in selected.relay_indices
                        if relay_columns[j].site_index in
                        column.profile.support[local_id] and
                        relay_columns[j].ready_s <= item.start_s + EPS and
                        item.end_s <= relay_columns[j].service_end_s + EPS]
            if not possible:
                raise ValueError(f"主问题选择的中继未覆盖 {column.task.batch_id}"
                                 f" 第 {local_id} 个盲段")
            mission = min(possible, key=lambda m: (m.energy_kwh, m.mission_id))
            state.assigned[item.slice_id] = mission.mission_id
        state.sorties.append(sortie)
        state.deliveries.extend(deliveries)
        state.trajectories[column.task.batch_id] = shifted
        state.slices.extend(slices)
    return state


def check_partial_state(ctx: Context, state: q3.JointState,
                        links: q3.LinkEvaluator, dem: q3.DemGrid
                        ) -> dict[str, Any]:
    """硬任务阶段允许尚未交付软箱，其余硬约束必须已成立。"""
    transport_bad = [row for row in check_plan(
        ctx, {"sorties": state.sorties, "deliveries": state.deliveries})
        if row["status"] != "PASS" and row["check_id"] != "all_80_boxes_once"]
    bounds = {row["batch_id"]: (row["start_s"], row["return_s"])
              for row in state.sorties}
    relay_rows = q3.relay_rows_from_missions(state.missions)
    comm = q3.verify_continuous_coverage(
        state.trajectories, links,
        q3.communication_rows(state.slices, state.assigned, bounds),
        relay_rows)
    resources = q3.validate_relay_rows(
        relay_rows, ctx.data_run, links, dem)
    return {
        "status": ("PASS" if not transport_bad and
                   comm["status"] == "PASS" and
                   resources["status"] == "PASS" else "FAILED_VALIDATION"),
        "transport_failure": transport_bad[0] if transport_bad else None,
        "communication_status": comm["status"],
        "communication_failure": comm["issues"][0] if comm["issues"] else None,
        "relay_resource_status": resources["status"],
        "relay_resource_failure": resources.get("first_failure"),
    }


def soft_task_sets(ctx: Context,
                   route_sets: list[tuple[str, list[q3.RouteTask]]]
                   ) -> list[tuple[str, list[q3.RouteTask]]]:
    soft = {bid for bid, box in ctx.boxes.items()
            if pd.isna(box["hard_deadline_s"])}
    result = []
    for name, tasks in route_sets:
        chosen = [q3.RouteTask(f"S{index:03d}", task.route)
                  for index, task in enumerate(tasks, 1)
                  if all(bid in soft for bid in task.route.box_ids)]
        listed = [bid for task in chosen for bid in task.route.box_ids]
        if len(listed) != len(soft) or set(listed) != soft:
            raise AssertionError(f"软箱组批 {name} 未恰好覆盖全部软箱")
        result.append((name, chosen))
    return result


def serial_ready_time(state: q3.JointState,
                      relay: dict[str, Any]) -> float:
    """软任务故障时可退回所有既有资源已周转、充满的保守时间。"""
    params = relay["params"]
    return max(
        [row["charge_end_s"] for row in state.sorties] +
        [mission.return_s + max(
            params["turnaround_s"],
            charge_to_full_s(1.0 - mission.energy_kwh /
                             params["energy_use_kwh"],
                             params["full_charge_s"]))
         for mission in state.missions], default=0.0)


def schedule_soft(
    ctx: Context, initial: q3.JointState,
    tasks: list[q3.RouteTask], candidates: list[q3.RelayCandidate],
    relay: dict[str, Any], links: q3.LinkEvaluator,
    positions: dict[str, tuple[float, float, float]],
    site_tree: cKDTree | None,
    access_cache: dict[tuple[Any, ...], bool],
    sample_step_s: float, beam_width: int, candidate_limit: int,
    start_probes: int, max_repairs: int, max_rewinds: int,
    soft_horizon_s: float, max_seconds: float,
) -> tuple[q3.JointState, dict[str, Any]]:
    """在已验证的硬任务后安排软箱；失败时拆批、换序和串行兜底。"""
    pending = sorted(tasks, key=lambda t: (
        min(float(ctx.boxes[bid]["expected_s"])
            for bid in t.route.box_ids), -len(t.route.box_ids)))
    beam = [initial]
    snapshots: list[list[q3.JointState]] = [beam]
    best = initial
    profiles: dict[tuple[Any, ...], q3.RouteProfile] = {}
    repairs = 0
    rewinds = 0
    index = 0
    failures: list[dict[str, Any]] = []
    started = perf_counter()
    while index < len(pending):
        if max_seconds and perf_counter() - started >= max_seconds:
            return best, {"status": "SEARCH_TIMEOUT", "failures": failures,
                          "adaptive_splits": repairs, "rewinds": rewinds,
                          "elapsed_s": round(perf_counter() - started, 2),
                          "scheduled_boxes": len(best.deliveries)}
        task = pending[index]
        successors: list[q3.JointState] = []
        reasons: Counter[str] = Counter()
        for site_limit in dict.fromkeys((
            min(candidate_limit, len(candidates)),
            min(max(4 * candidate_limit, 64), len(candidates)),
            len(candidates))):
            for state in beam:
                options, failed = q3._insert_task(
                    ctx, state, task, candidates, links, relay, positions,
                    sample_step_s, site_tree, profiles, access_cache,
                    max(3, beam_width // 2), site_limit, start_probes,
                    soft_horizon_s)
                successors.extend(options)
                reasons.update(failed)
            if successors:
                break
        if not successors:
            # 原有窗口太拥挤时，试一次所有资源已周转后的保守串行位置。
            for state in beam:
                options, failed = q3._insert_task(
                    ctx, state, task, candidates, links, relay, positions,
                    sample_step_s, site_tree, profiles, access_cache,
                    max(3, beam_width // 2), len(candidates),
                    start_probes, soft_horizon_s,
                    earliest_start_s=serial_ready_time(state, relay))
                successors.extend(options)
                reasons.update(failed)
        if successors:
            ordered = sorted(successors, key=q3._state_score)
            distinct: list[q3.JointState] = []
            seen: set[tuple[Any, ...]] = set()
            for state in ordered:
                latest = state.sorties[-1]
                signature = (latest["type_id"], latest["uav_id"],
                             round(latest["start_s"] / 60),
                             tuple(sorted(m.candidate.candidate_id
                                          for m in state.missions)))
                if signature in seen:
                    continue
                seen.add(signature)
                distinct.append(state)
                if len(distinct) >= beam_width:
                    break
            beam = distinct
            snapshots.append(beam)
            index += 1
            if len(beam[0].deliveries) > len(best.deliveries):
                best = beam[0]
            print(f"Q3 软箱 {index}/{len(pending)} 批；"
                  f"累计 {len(beam[0].deliveries)}/80 箱", flush=True)
            continue
        failures.append({"batch_id": task.batch_id,
                         "box_ids": list(task.route.box_ids),
                         "reasons": dict(reasons),
                         "scheduled_boxes": len(beam[0].deliveries)})
        pieces = (q3._split_current_task(ctx, task, pending)
                  if repairs < max_repairs else None)
        if pieces is not None:
            pending[index:index + 1] = pieces
            repairs += 1
            continue
        if index > 0 and rewinds < max_rewinds:
            target = max(0, index - (2 if rewinds < 2 else 5))
            failed_task = pending.pop(index)
            pending.insert(target, failed_task)
            beam = snapshots[target]
            snapshots = snapshots[:target + 1]
            index = target
            rewinds += 1
            continue
        return best, {"status": "SEARCH_INCOMPLETE", "failures": failures,
                      "adaptive_splits": repairs,
                      "rewinds": rewinds,
                      "elapsed_s": round(perf_counter() - started, 2),
                      "scheduled_boxes": len(best.deliveries)}
    return min(beam, key=q3._state_score), {
        "status": "CANDIDATE_FEASIBLE", "failures": failures,
        "adaptive_splits": repairs, "rewinds": rewinds,
        "elapsed_s": round(perf_counter() - started, 2),
        "scheduled_boxes": len(beam[0].deliveries)}


def save_partial_debug(path: Path, state: q3.JointState,
                       reason: dict[str, Any]) -> None:
    """仅作故障分析；文件名明确标识不属于官方可提交方案。"""
    q3.json_dump(path, {
        "status": "NOT_VALIDATED_PARTIAL",
        "warning": "不得用于论文结果或问题四输入",
        "delivered_box_count": len(state.deliveries),
        "transport_sorties": [{
            "batch_id": row["batch_id"], "box_ids": row["box_ids"],
            "uav_id": row["uav_id"], "battery_id": row["battery_id"],
            "start_s": row["start_s"], "return_s": row["return_s"],
            "charge_end_s": row["charge_end_s"]} for row in state.sorties],
        "relay_missions": [{
            "mission_id": m.mission_id, "site": m.candidate.candidate_id,
            "uav_id": m.uav_id, "energy_unit_id": m.unit_id,
            "launch_s": m.start_s, "ready_s": m.ready_s,
            "service_end_s": m.service_end_s, "return_s": m.return_s,
            "energy_kwh": m.energy_kwh} for m in state.missions],
        "reason": reason,
    })


def validate_complete_state(ctx: Context, state: q3.JointState,
                            links: q3.LinkEvaluator, dem: q3.DemGrid
                            ) -> dict[str, Any]:
    transport_bad = [row for row in check_plan(
        ctx, {"sorties": state.sorties, "deliveries": state.deliveries})
        if row["status"] != "PASS"]
    partial = check_partial_state(ctx, state, links, dem)
    return {**partial,
            "status": ("PASS" if not transport_bad and
                       partial["status"] == "PASS" else "FAILED_VALIDATION"),
            "transport_failure": transport_bad[0] if transport_bad else None}


def run_global(args: argparse.Namespace) -> dict[str, Any]:
    data_run = (args.data_run.resolve() if args.data_run else
                q3.find_data_run(CODE_DIR, q3.EXPECTED_MANIFEST))
    relay, dem, links, source = q3._scene(data_run)
    ctx = Context(data_run)
    positions = q3._node_positions(data_run)
    route_sets = q3._raw_route_sets(ctx)
    source_slices = q3._source_slices_from_raw(
        ctx, positions, links, args.sample_step_s, route_sets)
    print("Q3 全局排程：生成 DEM 内通信可达的悬停候选", flush=True)
    candidates, candidate_stats = q3.generate_candidates(
        dem, links, source_slices, relay, positions["O01"],
        spacing_m=args.candidate_spacing_m, max_points=12000)
    access_cache: dict[tuple[Any, ...], bool] = {}
    candidates, preselect_stats = preselect_sites(
        source_slices, candidates, links, access_cache,
        args.preselect_budget, args.preselect_neighbors)
    print(f"Q3 空间预筛：{preselect_stats.get('original_relay_candidates', 0)}"
          f" → {len(candidates)} 个悬停点", flush=True)
    tree = (cKDTree(np.asarray([site.hover for site in candidates], dtype=float))
            if candidates else None)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    table_dir = args.output_root.resolve() / f"q3_global_{stamp}_table"
    figure_dir = args.output_root.resolve() / f"q3_global_{stamp}_figure"
    table_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "status": "RUNNING", "method": "candidate-column MILP and soft-task repair",
        "source": source, "parameters": {
            "site_budget": args.site_budget,
            "preselect_budget": args.preselect_budget,
            "preselect_neighbors": args.preselect_neighbors,
            "transport_start_step_s": args.transport_start_step_s,
            "relay_ready_step_s": args.relay_ready_step_s,
            "sample_step_s": args.sample_step_s,
            "candidate_spacing_m": args.candidate_spacing_m,
            "master_time_limit_s": args.master_time_limit_s,
            "soft_horizon_s": args.soft_horizon_s,
            "soft_seconds_per_grouping": args.soft_seconds_per_grouping,
        }, "candidate_stats": candidate_stats,
        "spatial_preselection": preselect_stats,
        "global_optimality_proven": False,
        "table_dir": str(table_dir), "figure_dir": None,
    }

    def record(status: str, **details: Any) -> dict[str, Any]:
        summary.update(details)
        summary["status"] = status
        q3.json_dump(table_dir / "Q3_运行摘要.json", summary)
        return summary

    hard_tasks = hard_route_pool(ctx, route_sets)
    required_hard = {bid for bid, box in ctx.boxes.items()
                     if not pd.isna(box["hard_deadline_s"])}
    print(f"Q3 全局排程：{len(required_hard)} 箱硬时限、"
          f"{len(hard_tasks)} 种组批路线；计算通信证书", flush=True)
    profiles, profile_stats = build_profile_choices(
        ctx, hard_tasks, candidates, links, positions,
        args.sample_step_s, tree, access_cache, show_progress=True)
    selected_sites, site_stats = choose_sites(
        profiles, candidates, args.site_budget)
    print(f"Q3 中继位置：保留 {len(selected_sites)}/{len(candidates)} 个；"
          f"裁剪后仍未覆盖 {site_stats['uncovered_profile_slices_after_pruning']}"
          " 个路线盲片（相关路线会禁用）", flush=True)
    transports, transport_stats = build_transport_columns(
        ctx, profiles, selected_sites, args.transport_start_step_s)
    relays, relay_stats = build_relay_columns(
        transports, candidates, relay, args.relay_ready_step_s)
    summary["candidate_preparation"] = {
        **profile_stats, **site_stats, **transport_stats, **relay_stats}
    q3.json_dump(table_dir / "Q3_候选准备统计.json",
                 summary["candidate_preparation"])
    if (len(transports) > args.max_transport_columns or
            len(relays) > args.max_relay_columns):
        return record("CANDIDATE_BUDGET_EXCEEDED", interpretation=(
            "候选列超过显式内存预算；可增大时间网格或预算，不能据此判定无解"))
    fleet = Counter(uav.type_id for uav in ctx.uavs)
    batteries = Counter(battery.type_id for battery in ctx.batteries)
    print(f"Q3 全局排程：MILP 同时选 {len(transports)} 个运输列与"
          f" {len(relays)} 个中继服务列", flush=True)
    answer = solve_master(
        required_hard, transports, relays, dict(fleet), dict(batteries),
        len(relay["uav_ids"]), len(relay["unit_ids"]),
        args.master_time_limit_s)
    summary["hard_master"] = answer.diagnostics
    if answer.status != "CANDIDATE_FEASIBLE":
        return record(answer.status, interpretation=(
            "有限候选或求解预算内没有形成硬任务方案；未证明赛题无解"))
    hard_state = state_from_master(ctx, transports, relays, answer)
    hard_check = check_partial_state(ctx, hard_state, links, dem)
    summary["hard_validation"] = hard_check
    if hard_check["status"] != "PASS":
        save_partial_debug(table_dir / "Q3_未验收部分排程.json",
                           hard_state, hard_check)
        return record("FAILED_VALIDATION", interpretation=(
            "候选列解未通过秒级独立验收；不得作为正式结果"))
    print(f"Q3 全局排程：硬时限 {len(hard_state.deliveries)}/"
          f"{len(required_hard)} 箱通过独立验收，继续安排软箱", flush=True)
    soft_sets = soft_task_sets(ctx, route_sets)
    preferred = ("neighbor_pair", "balanced", "small_batches",
                 "hard_split", "single_box_fallback")
    soft_sets.sort(key=lambda row: preferred.index(row[0])
                   if row[0] in preferred else len(preferred))
    attempts = []
    best_partial = hard_state
    selected: q3.JointState | None = None
    for grouping, soft_tasks in soft_sets[:args.max_soft_groupings]:
        print(f"Q3 全局排程：软箱组批 {grouping}，{len(soft_tasks)} 批",
              flush=True)
        state, outcome = schedule_soft(
            ctx, hard_state, soft_tasks, candidates, relay, links,
            positions, tree, access_cache, args.sample_step_s,
            args.soft_beam_width, args.soft_candidate_limit,
            args.soft_start_probes, args.soft_max_splits,
            args.soft_max_rewinds, args.soft_horizon_s,
            args.soft_seconds_per_grouping)
        attempts.append({"grouping": grouping, **outcome})
        q3.json_dump(table_dir / "Q3_搜索进展.json", attempts)
        if len(state.deliveries) > len(best_partial.deliveries):
            best_partial = state
        if outcome["status"] != "CANDIDATE_FEASIBLE":
            continue
        final_check = validate_complete_state(ctx, state, links, dem)
        attempts[-1]["final_check"] = final_check
        if final_check["status"] == "PASS":
            selected = state
            break
    summary["soft_attempts"] = attempts
    if selected is None:
        save_partial_debug(table_dir / "Q3_未验收部分排程.json",
                           best_partial, {"soft_attempts": attempts})
        return record("SEARCH_INCOMPLETE", interpretation=(
            "未得到80箱完整可行解；部分排程只用于诊断，不可用于论文或问题四"),
            best_partial_delivered_boxes=len(best_partial.deliveries))

    q3._save_joint_tables(table_dir, selected, relay)
    validation = q3.validate_official_tables(table_dir, data_run, links, dem)
    q3._save_verification(table_dir, validation)
    energy_transport = sum(row["energy_kwh"] for row in selected.sorties)
    energy_relay = sum(row.energy_kwh for row in selected.missions)
    result = record(
        validation["status"],
        delivered_boxes=len(selected.deliveries),
        transport_sorties=len(selected.sorties),
        relay_sorties=len(selected.missions),
        joint_makespan_s=max(
            [row["return_s"] for row in selected.sorties] +
            [row.return_s for row in selected.missions]),
        transport_energy_kwh=energy_transport,
        relay_energy_kwh=energy_relay,
        total_energy_kwh=energy_transport + energy_relay,
        weighted_soft_lateness_s=sum(
            row["priority"] * row["soft_lateness_s"]
            for row in selected.deliveries
            if row["hard_deadline_s"] is None),
        transport_status=validation["transport_status"],
        communication_status=validation["communication"]["status"],
        relay_resource_status=validation["relay_resources"]["status"],
    )
    if validation["status"] == "PASS":
        q3._save_figures(figure_dir, selected.slices, selected.assigned,
                         selected.missions)
        (table_dir / "Q3_READY.txt").write_text(
            "Q3 global candidate-column search; independent validation PASS\n"
            f"source_manifest_sha256={source['source_manifest_sha256']}\n",
            encoding="utf-8")
        result["figure_dir"] = str(figure_dir)
        q3.json_dump(table_dir / "Q3_运行摘要.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-run", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--candidate-spacing-m", type=int, default=500)
    parser.add_argument("--sample-step-s", type=float, default=20.0)
    parser.add_argument("--site-budget", type=int, default=20)
    parser.add_argument("--preselect-budget", type=int, default=128)
    parser.add_argument("--preselect-neighbors", type=int, default=32)
    parser.add_argument("--transport-start-step-s", type=int, default=300)
    parser.add_argument("--relay-ready-step-s", type=int, default=300)
    parser.add_argument("--max-transport-columns", type=int, default=10000)
    parser.add_argument("--max-relay-columns", type=int, default=16000)
    parser.add_argument("--master-time-limit-s", type=float, default=900.0)
    parser.add_argument("--max-soft-groupings", type=int, default=5)
    parser.add_argument("--soft-beam-width", type=int, default=8)
    parser.add_argument("--soft-candidate-limit", type=int, default=16)
    parser.add_argument("--soft-start-probes", type=int, default=30)
    parser.add_argument("--soft-max-splits", type=int, default=20)
    parser.add_argument("--soft-max-rewinds", type=int, default=12)
    parser.add_argument("--soft-horizon-s", type=float, default=86400.0)
    parser.add_argument("--soft-seconds-per-grouping", type=float,
                        default=600.0,
                        help="每种软箱组批搜索预算；0 表示不设上限")
    parser.add_argument("--validate-only", type=Path, default=None)
    args = parser.parse_args(argv)
    numeric_positive = (
        args.candidate_spacing_m, args.sample_step_s, args.site_budget,
        args.preselect_budget, args.preselect_neighbors,
        args.transport_start_step_s, args.relay_ready_step_s,
        args.max_transport_columns, args.max_relay_columns,
        args.master_time_limit_s, args.max_soft_groupings,
        args.soft_beam_width, args.soft_candidate_limit,
        args.soft_start_probes, args.soft_horizon_s)
    if any(not math.isfinite(float(value)) or value <= 0
           for value in numeric_positive):
        parser.error("所有网格、预算、时间与束宽参数必须为正且有限")
    if min(args.soft_max_splits, args.soft_max_rewinds) < 0:
        parser.error("拆批与回退次数不能为负")
    if (not math.isfinite(args.soft_seconds_per_grouping) or
            args.soft_seconds_per_grouping < 0):
        parser.error("软箱组批时间预算必须非负且有限")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_only is not None:
        data_run = (args.data_run.resolve() if args.data_run else
                    q3.find_data_run(CODE_DIR, q3.EXPECTED_MANIFEST))
        _, dem, links, _ = q3._scene(data_run)
        report = q3.validate_official_tables(
            args.validate_only.resolve(), data_run, links, dem)
        print(json.dumps({"status": report["status"],
                          "transport_status": report["transport_status"],
                          "communication_status": report["communication"]["status"],
                          "relay_resource_status": report["relay_resources"]["status"]},
                         ensure_ascii=False, indent=2))
        return 0 if report["status"] == "PASS" else 2
    summary = run_global(args)
    print(json.dumps({key: summary.get(key) for key in (
        "status", "delivered_boxes", "transport_sorties", "relay_sorties",
        "joint_makespan_s", "total_energy_kwh", "table_dir")},
        ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
