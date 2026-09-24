"""问题二优化方案 2：紧急箱 A 型直送、普通箱多点合并、机电池排程。

先从原始箱单按硬时限分层；紧急箱同区直送，普通箱按节约里程合并
为多站路线，再联合安排 8 架实体机与 14 组共享电池。若不可行则明确
输出 FAIL 和冲突诊断，不回退为其他优化方案；任何结果都不声称全局最优。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from _0_pipeline import verify_ready
from q1_0_baseline import CODE_DIR, find_latest_validated_run
from q2_0_baseline import charge_to_full_s, optional_seconds, required_seconds, write_csv
from q2_1_optimize import Context, Route, TOL, check_plan, save_figures, save_tables, simulate


class Matrix:
    """稀疏 MILP 行构造器。"""

    def __init__(self) -> None:
        self.cost: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.integer: list[int] = []
        self.rows: list[dict[int, float]] = []
        self.row_lower: list[float] = []
        self.row_upper: list[float] = []

    def var(self, cost: float = 0.0, lo: float = 0.0,
            hi: float = 1.0, integer: bool = True) -> int:
        index = len(self.cost)
        self.cost.append(cost)
        self.lower.append(lo)
        self.upper.append(hi)
        self.integer.append(int(integer))
        return index

    def row(self, terms: dict[int, float], lo: float = -math.inf,
            hi: float = math.inf) -> None:
        self.rows.append({i: v for i, v in terms.items() if abs(v) > 1e-12})
        self.row_lower.append(lo)
        self.row_upper.append(hi)

    def solve(self, seconds: float):
        rr, cc, vv = [], [], []
        for i, row in enumerate(self.rows):
            for j, value in row.items():
                rr.append(i)
                cc.append(j)
                vv.append(value)
        matrix = coo_array((np.asarray(vv, dtype=float),
                            (np.asarray(rr, dtype=np.int32),
                             np.asarray(cc, dtype=np.int32))),
                           shape=(len(self.rows), len(self.cost))).tocsc()
        return milp(np.asarray(self.cost),
                    integrality=np.asarray(self.integer, dtype=np.int32),
                    bounds=Bounds(self.lower, self.upper),
                    constraints=LinearConstraint(matrix, self.row_lower, self.row_upper),
                    options={"time_limit": max(0.1, seconds), "mip_rel_gap": 0.0})


def is_urgent(ctx: Context, route: Route) -> bool:
    return any(optional_seconds(ctx.boxes[bid]["hard_deadline_s"]) is not None
               for bid in route.box_ids)


def eligible_options(ctx: Context, route: Route) -> dict[str, dict[str, Any]]:
    """紧急层固定 A 型，普通层留给 B/C 型，防止两层抢占 A 型资源。"""
    options = ctx.options(route)
    if is_urgent(ctx, route):
        if not all(optional_seconds(ctx.boxes[bid]["hard_deadline_s"]) is not None
                   for bid in route.box_ids):
            raise ValueError("紧急路线混入普通箱")
        return {"A": options["A"]} if "A" in options else {}
    return {kind: options[kind] for kind in ("B", "C") if kind in options}


def urgency_key(ctx: Context, route: Route) -> tuple[float, float, str]:
    return (min((optional_seconds(ctx.boxes[bid]["hard_deadline_s"])
                 for bid in route.box_ids
                 if optional_seconds(ctx.boxes[bid]["hard_deadline_s"]) is not None),
                default=math.inf),
            min(required_seconds(ctx.boxes[bid]["expected_s"], bid)
                for bid in route.box_ids), route.box_ids[0])


def routine_cost(ctx: Context, route: Route) -> tuple[float, float] | None:
    options = eligible_options(ctx, route)
    candidates = []
    fallback = []
    for option in options.values():
        score = (float(option["energy_kwh"]), float(option["duration_s"]))
        fallback.append(score)
        if all(float(option["completion_offsets"][bid]) <=
               required_seconds(ctx.boxes[bid]["expected_s"], bid) + TOL
               for bid in route.box_ids):
            candidates.append(score)
    return min(candidates or fallback) if fallback else None


def best_merge(ctx: Context, left: Route, right: Route,
               max_stops: int) -> tuple[Route, tuple[float, float]] | None:
    groups: dict[str, list[str]] = {}
    for route in (left, right):
        for zone, ids in route.visits:
            groups.setdefault(zone, []).extend(ids)
    if len(groups) > max_stops:
        return None
    winner = None
    for order in itertools.permutations(sorted(groups)):
        route = ctx.route([(zone, groups[zone]) for zone in order])
        cost = routine_cost(ctx, route)
        if cost is not None and (winner is None or cost < winner[1]):
            winner = route, cost
    return winner


def two_tier_routes(ctx: Context, deadline: float, max_stops: int,
                    energy_slack: float) -> tuple[list[Route], list[list[Route]], dict[str, Any]]:
    """紧急箱按站 A 型直送；普通箱从单箱起逐次做物理可行的邻近合并。"""
    zones: dict[str, list[str]] = {}
    hard_boxes = []
    routine_boxes = []
    for bid, box in ctx.boxes.items():
        if optional_seconds(box["hard_deadline_s"]) is not None:
            zones.setdefault(str(box["zone_id"]), []).append(bid)
            hard_boxes.append(bid)
        else:
            routine_boxes.append(bid)
    urgent = [ctx.route([(zone, ids)]) for zone, ids in sorted(zones.items())]
    if any("A" not in ctx.options(route) for route in urgent):
        raise ValueError("至少一个紧急同区直送批次不满足 A 型机物理约束")
    urgent.sort(key=lambda route: urgency_key(ctx, route))
    routine = [ctx.route([(str(ctx.boxes[bid]["zone_id"]), [bid])])
               for bid in sorted(routine_boxes)]
    if any(routine_cost(ctx, route) is None for route in routine):
        raise ValueError("至少一个普通箱不能由 B/C 型机在期望时间前单独交付")
    snapshots: list[list[Route]] = []
    targets = {20, 17, 15, 13, 11}
    while len(routine) > min(targets) and time.perf_counter() < deadline:
        candidates = []
        costs = [routine_cost(ctx, route) for route in routine]
        for i, j in itertools.combinations(range(len(routine)), 2):
            if time.perf_counter() >= deadline:
                break
            merged = best_merge(ctx, routine[i], routine[j], max_stops)
            if merged is None:
                continue
            route, cost = merged
            saving = costs[i][0] + costs[j][0] - cost[0]
            if saving >= -energy_slack:
                candidates.append((-saving, cost[1], len(route.zones), i, j, route))
        if not candidates:
            break
        _, _, _, i, j, merged_route = min(candidates)
        routine = [route for k, route in enumerate(routine) if k not in (i, j)] + [merged_route]
        if len(routine) in targets:
            snapshots.append(list(routine))
    if not snapshots or snapshots[-1] != routine:
        snapshots.append(list(routine))
    coverage = Counter(bid for route in [*urgent, *routine] for bid in route.box_ids)
    if coverage != Counter(ctx.boxes.keys()):
        raise AssertionError("两级路线未将所有货箱恰好覆盖一次")
    return urgent, snapshots, {"urgent_boxes": len(hard_boxes),
                               "routine_boxes": len(routine_boxes),
                               "urgent_a_routes": len(urgent),
                               "routine_routes_final": len(routine),
                               "route_sets": len(snapshots)}


class RestrictedContext:
    """仅把候选机型限制给已有的逐段仿真器，验收仍用原始 Context。"""

    def __init__(self, source: Context):
        self.source = source
        self.boxes = source.boxes
        self.uavs = source.uavs
        self.batteries = source.batteries

    def options(self, route: Route) -> dict[str, dict[str, Any]]:
        return eligible_options(self.source, route)


def scheduled_plan(ctx: Context, routes: list[Route], seconds: float,
                   enforce_soft: bool) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """固定组批后，把机型、实体机、电池和连续开始时刻联合做 MILP。"""
    n = len(routes)
    model_options = [eligible_options(ctx, route) for route in routes]
    if any(not options for options in model_options):
        return None, {"status": "no_eligible_type", "enforce_zero_soft_lateness": enforce_soft}
    m = Matrix()
    horizon = 20000.0
    big_m = 30000.0
    start = [m.var(cost=0.0001, hi=horizon, integer=False) for _ in routes]
    xu = {(i, u.uav_id): m.var(cost=float(model_options[i][u.type_id]["energy_kwh"]))
          for i in range(n) for u in ctx.uavs if u.type_id in model_options[i]}
    zb = {(i, b.battery_id): m.var()
          for i in range(n) for b in ctx.batteries if b.type_id in model_options[i]}
    for i in range(n):
        m.row({xu[i, u.uav_id]: 1 for u in ctx.uavs if (i, u.uav_id) in xu}, 1, 1)
        m.row({zb[i, b.battery_id]: 1 for b in ctx.batteries if (i, b.battery_id) in zb}, 1, 1)
        for kind in model_options[i]:
            row = {xu[i, u.uav_id]: 1.0 for u in ctx.uavs if u.type_id == kind}
            row.update({zb[i, b.battery_id]: -1.0 for b in ctx.batteries
                        if b.type_id == kind})
            m.row(row, 0, 0)
        for bid in routes[i].box_ids:
            box = ctx.boxes[bid]
            due = optional_seconds(box["hard_deadline_s"])
            if due is None and enforce_soft:
                due = required_seconds(box["expected_s"], bid)
            if due is not None:
                row = {start[i]: 1.0}
                for u in ctx.uavs:
                    key = (i, u.uav_id)
                    if key in xu:
                        row[xu[key]] = float(model_options[i][u.type_id]["completion_offsets"][bid])
                m.row(row, hi=due + TOL)

    # 对同一架实体机/同一组电池上的两架次，择一决定先后。
    for i, j in itertools.combinations(range(n), 2):
        yu = m.var()
        yb = m.var()
        for u in ctx.uavs:
            ai, aj = (i, u.uav_id), (j, u.uav_id)
            if ai not in xu or aj not in xu:
                continue
            dur_i = {xu[i, v.uav_id]: float(model_options[i][v.type_id]["duration_s"])
                     for v in ctx.uavs if (i, v.uav_id) in xu}
            dur_j = {xu[j, v.uav_id]: float(model_options[j][v.type_id]["duration_s"])
                     for v in ctx.uavs if (j, v.uav_id) in xu}
            before = {start[i]: 1.0, start[j]: -1.0,
                      xu[ai]: big_m, xu[aj]: big_m, yu: big_m}
            for key, value in dur_i.items():
                before[key] = before.get(key, 0.0) + value
            m.row(before, hi=3 * big_m)
            after = {start[j]: 1.0, start[i]: -1.0,
                     xu[ai]: big_m, xu[aj]: big_m, yu: -big_m}
            for key, value in dur_j.items():
                after[key] = after.get(key, 0.0) + value
            m.row(after, hi=2 * big_m)
        for battery in ctx.batteries:
            ai, aj = (i, battery.battery_id), (j, battery.battery_id)
            if ai not in zb or aj not in zb:
                continue
            # 电池复用间隔 = 返航时间 + 按返航 SOC 的两阶段补能时间。
            def occupied(k: int) -> dict[int, float]:
                return {
                    xu[k, u.uav_id]: float(model_options[k][u.type_id]["duration_s"])
                    + charge_to_full_s(float(model_options[k][u.type_id]["return_soc"]),
                                       battery.full_charge_s)
                    for u in ctx.uavs if (k, u.uav_id) in xu
                }
            before = {start[i]: 1.0, start[j]: -1.0,
                      zb[ai]: big_m, zb[aj]: big_m, yb: big_m}
            for key, value in occupied(i).items():
                before[key] = before.get(key, 0.0) + value
            m.row(before, hi=3 * big_m)
            after = {start[j]: 1.0, start[i]: -1.0,
                     zb[ai]: big_m, zb[aj]: big_m, yb: -big_m}
            for key, value in occupied(j).items():
                after[key] = after.get(key, 0.0) + value
            m.row(after, hi=2 * big_m)
    result = m.solve(seconds)
    info = {"status": int(result.status), "message": str(result.message),
            "variables": len(m.cost), "constraints": len(m.rows),
            "enforce_zero_soft_lateness": enforce_soft}
    if result.x is None:
        return None, info
    sorties, deliveries = [], []
    for i, route in enumerate(routes):
        selected_u = [u for u in ctx.uavs if (i, u.uav_id) in xu
                      and result.x[xu[i, u.uav_id]] > 0.5]
        selected_b = [b for b in ctx.batteries if (i, b.battery_id) in zb
                      and result.x[zb[i, b.battery_id]] > 0.5]
        if len(selected_u) != 1 or len(selected_b) != 1:
            return None, info
        u, battery = selected_u[0], selected_b[0]
        if u.type_id != battery.type_id:
            return None, info
        option = model_options[i][u.type_id]
        t0 = max(0.0, float(result.x[start[i]]))
        end = t0 + float(option["duration_s"])
        charge = charge_to_full_s(float(option["return_soc"]), battery.full_charge_s)
        batch = f"Q{i + 1:03d}"
        sorties.append({
            "batch_id": batch, "type_id": u.type_id, "uav_id": u.uav_id,
            "battery_id": battery.battery_id, "route": route,
            "visit_order": ">".join(route.zones), "box_ids": list(route.box_ids),
            "start_s": t0, "launch_s": t0 + float(option["launch_offset_s"]),
            "return_s": end, "charge_start_s": end, "charge_end_s": end + charge,
            "charge_s": charge, "mass_kg": option["mass_kg"],
            "volume_m3": option["volume_m3"], "energy_kwh": option["energy_kwh"],
            "return_soc": option["return_soc"], "legs": option["legs"],
            "visits": option["visits"],
        })
        for bid in route.box_ids:
            box = ctx.boxes[bid]
            complete = t0 + float(option["completion_offsets"][bid])
            hard = optional_seconds(box["hard_deadline_s"])
            expected = required_seconds(box["expected_s"], bid)
            deliveries.append({
                "box_id": bid, "batch_id": batch, "zone_id": str(box["zone_id"]),
                "sequence": option["sequence"][bid], "complete_s": complete,
                "hard_deadline_s": hard, "expected_s": expected,
                "priority": float(box["priority"]),
                "hard_slack_s": None if hard is None else hard - complete,
                "soft_lateness_s": max(0.0, complete - expected),
            })
    sorties.sort(key=lambda s: (s["start_s"], s["batch_id"]))
    deliveries.sort(key=lambda d: d["box_id"])
    hard_fail = [d for d in deliveries if d["hard_slack_s"] is not None
                 and d["hard_slack_s"] < -TOL]
    score = (len(hard_fail), sum(-d["hard_slack_s"] for d in hard_fail),
             sum(d["priority"] * d["soft_lateness_s"] for d in deliveries
                 if d["hard_deadline_s"] is None),
             max(s["return_s"] for s in sorties),
             sum(s["energy_kwh"] for s in sorties), n)
    plan = {"routes": routes, "sorties": sorties, "deliveries": deliveries, "score": score}
    if any(c["status"] != "PASS" for c in check_plan(ctx, plan)):
        return None, info
    return plan, info


def better(plan: dict[str, Any], best: dict[str, Any]) -> bool:
    """先比硬违约、软迟到，再比较架次、能耗和最晚返航。"""
    a, b = plan["score"], best["score"]
    return (a[0], a[1], a[2], a[5], a[4], a[3]) < (b[0], b[1], b[2], b[5], b[4], b[3])


def ordered_candidate(ctx: Context, urgent: list[Route], routine: list[Route],
                      rng: random.Random, randomize: bool) -> list[Route]:
    urgent_groups: dict[float, list[Route]] = {}
    for route in urgent:
        urgent_groups.setdefault(urgency_key(ctx, route)[0], []).append(route)
    early: list[Route] = []
    for due in sorted(urgent_groups):
        group = urgent_groups[due]
        if randomize:
            rng.shuffle(group)
        else:
            group.sort(key=lambda route: urgency_key(ctx, route))
        early.extend(group)
    later = list(routine)
    if randomize:
        rng.shuffle(later)
    else:
        later.sort(key=lambda route: urgency_key(ctx, route))
    return early + later


def search_two_tier(ctx: Context, urgent: list[Route],
                    route_sets: list[list[Route]], deadline: float,
                    trials: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)
    restricted = RestrictedContext(ctx)
    best = None
    tries = 0
    for routine in route_sets:
        for attempt in range(trials):
            if time.perf_counter() >= deadline:
                break
            routes = ordered_candidate(ctx, urgent, routine, rng, attempt > 0)
            plan = simulate(restricted, routes)
            tries += 1
            if plan is not None and (best is None or better(plan, best)):
                best = plan
    if best is None:
        raise ValueError("两级路线没有形成可排程方案；请检查机型或输入数据")
    return best, {"greedy_schedules": tries, "best_greedy_score": best["score"]}


COMPARISON_COLUMNS = [
    "方案", "状态", "最晚返航（s）", "运输总能耗（kWh）", "架次数",
    "硬时限违约箱数", "软箱加权迟到（s）", "同口径可比",
    "本表返航最早", "本表能耗最低", "本表架次数最低", "本表非劣解",
    "说明", "结果目录",
]


def compare_all_methods(output_root: Path, current: dict[str, Any],
                        current_table_dir: Path, soft_budget: float) -> list[dict[str, Any]]:
    """只取同一数据哈希下每种方法的最新一次结果，标出三个指标的权衡。"""
    rows: list[dict[str, Any]] = []
    source_hash = current["source_manifest_sha256"]
    methods = (
        ("0_baseline", "q2_base_*_table", "三机型同区基线", "q2_baseline_abc_single_zone_v2"),
        ("1_optimize", "q2_opt_*_table", "优化方案1", None),
        ("3_optimize", "q2_opt3_*_table", "优化方案3（均衡推荐）", None),
        ("4_optimize", "q2_opt4_*_table", "优化方案4", None),
    )

    def make_row(label: str, summary: dict[str, Any] | None,
                 directory: Path | None) -> dict[str, Any]:
        if summary is None:
            return {"方案": label, "状态": "未找到同源结果", "说明": "请先运行该方案",
                    "结果目录": ""}
        return {
            "方案": label, "状态": summary["status"],
            "最晚返航（s）": summary["makespan_s"],
            "运输总能耗（kWh）": summary["total_energy_kwh"],
            "架次数": summary["sorties"],
            "硬时限违约箱数": summary["hard_deadline_violations"],
            "软箱加权迟到（s）": summary["soft_weighted_lateness_s"],
            "说明": "硬时限违约，仅作诊断" if summary["status"] != "PASS" else "",
            "结果目录": str(directory) if directory is not None else "",
        }

    for folder, pattern, label, method_id in methods:
        candidates = [*(output_root / folder).glob(pattern),
                      *output_root.glob(pattern)]  # 兼容整理前的平铺结果
        found: tuple[dict[str, Any], Path] | None = None
        for directory in sorted(candidates, key=lambda path: path.stat().st_mtime,
                                reverse=True):
            path = directory / "Q2_运行摘要.json"
            if not path.is_file():
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            if (summary.get("source_manifest_sha256") == source_hash and
                    (method_id is None or summary.get("baseline_method_id") == method_id)):
                found = summary, directory
                break
        rows.append(make_row(label, found[0] if found else None,
                             found[1] if found else None))
    rows.insert(2, make_row("优化方案2", current, current_table_dir))
    rows[2]["说明"] = "独立构造的两级配送方案；请按验收状态解读"

    feasible = []
    for row in rows:
        same_scope = (
            row["状态"] == "PASS"
            and int(row["硬时限违约箱数"]) == 0
            and float(row["软箱加权迟到（s）"]) <= soft_budget + TOL
        ) if row["状态"] != "未找到同源结果" else False
        row["同口径可比"] = "是" if same_scope else "否"
        if same_scope:
            feasible.append(row)
    metrics = ("最晚返航（s）", "运输总能耗（kWh）", "架次数")
    minima = {metric: min(float(row[metric]) for row in feasible)
              for metric in metrics} if feasible else {}
    flags = {"最晚返航（s）": "本表返航最早",
             "运输总能耗（kWh）": "本表能耗最低",
             "架次数": "本表架次数最低"}
    for row in rows:
        comparable = row["同口径可比"] == "是"
        for metric, column in flags.items():
            row[column] = ("是" if comparable and
                           math.isclose(float(row[metric]), minima[metric],
                                        rel_tol=0, abs_tol=TOL) else "")
        dominated = comparable and any(
            other is not row
            and all(float(other[metric]) <= float(row[metric]) + TOL
                    for metric in metrics)
            and any(float(other[metric]) < float(row[metric]) - TOL
                    for metric in metrics)
            for other in feasible
        )
        row["本表非劣解"] = "是" if comparable and not dominated else ""
        if dominated:
            row["说明"] = (row["说明"] + "；" if row["说明"] else "") + "本表存在三指标均不劣的方案"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2 方案2：紧急 A 型直送 + 普通多点合并 + 资源排程")
    parser.add_argument("--data-run", type=Path, help="已验收数据目录，默认最新")
    parser.add_argument("--output-root", type=Path, help="默认 code/2_outputs")
    parser.add_argument("--time-limit-s", type=float, default=240.0,
                        help="路线构造与排程的总时间预算，默认 240 秒")
    parser.add_argument("--greedy-trials", type=int, default=30,
                        help="每组路线尝试的列表排程次序，默认 30")
    parser.add_argument("--max-stops", type=int, default=3,
                        help="普通路线最多停靠服务区数，默认 3")
    parser.add_argument("--merge-energy-slack-kwh", type=float, default=1.0,
                        help="每次合并允许的额外能耗上限，默认 1 kWh")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--soft-budget-s", type=float, default=0.0,
                        help="方案对照中允许的软箱加权迟到秒数，默认 0")
    args = parser.parse_args()
    if (args.time_limit_s <= 0 or args.greedy_trials < 1 or
            not 1 <= args.max_stops <= 4 or args.merge_energy_slack_kwh < 0 or
            args.soft_budget_s < 0):
        parser.error("时间和次数必须为正；停靠数为 1–4；能耗余量与软迟到预算不能为负")
    data_run = args.data_run.resolve() if args.data_run else find_latest_validated_run()
    if not verify_ready(data_run):
        raise ValueError(f"输入数据未通过哈希验收：{data_run}")
    ctx = Context(data_run)
    output_root = args.output_root.resolve() if args.output_root else CODE_DIR / "2_outputs"
    clock = time.perf_counter()
    deadline = clock + args.time_limit_s
    urgent, route_sets, construction = two_tier_routes(
        ctx, clock + 0.45 * args.time_limit_s, args.max_stops,
        args.merge_energy_slack_kwh)
    print(f"two-tier routes: {construction['urgent_a_routes']} A urgent, "
          f"{construction['routine_routes_final']} routine; "
          f"{construction['urgent_boxes']} hard boxes", flush=True)
    seed_plan, search = search_two_tier(
        ctx, urgent, route_sets, clock + 0.75 * args.time_limit_s,
        args.greedy_trials, args.seed)
    best = seed_plan
    milp_info = []
    remaining = deadline - time.perf_counter()
    if remaining > 5 and len(best["routes"]) <= 35:
        # 路线已固定；仅优化实体机、电池和开始时刻。紧急层始终固定为 A 型。
        plan, info = scheduled_plan(ctx, best["routes"], max(1.0, remaining * 0.65), True)
        milp_info.append({"strict_soft": info})
        if plan is not None and better(plan, best):
            best = plan
        remaining = deadline - time.perf_counter()
        if remaining > 5 and best["score"][0] > 0:
            plan, info = scheduled_plan(ctx, seed_plan["routes"], remaining - 1, False)
            milp_info.append({"hard_only": info})
            if plan is not None and better(plan, best):
                best = plan
    checks = check_plan(ctx, best)
    checks.append({
        "check_id": "urgent_direct_by_A",
        "status": "PASS" if all(
            s["type_id"] == "A" and len(s["route"].visits) == 1
            for s in best["sorties"] if is_urgent(ctx, s["route"])) else "FAIL",
        "detail": "每个硬时限架次必须由 A 型机同区直送",
    })
    checks.append({
        "check_id": "routine_by_B_or_C",
        "status": "PASS" if all(
            s["type_id"] in {"B", "C"} for s in best["sorties"]
            if not is_urgent(ctx, s["route"])) else "FAIL",
        "detail": "普通箱由 B/C 型机执行",
    })
    if not verify_ready(data_run):
        raise ValueError("搜索期间输入数据发生改变")
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    method_dir = output_root / "2_optimize"
    table_dir = method_dir / f"q2_opt2_{stamp}_table"
    figure_dir = method_dir / f"q2_opt2_{stamp}_figure"
    if table_dir.exists() or figure_dir.exists():
        raise FileExistsError("本次结果目录已存在；请稍后重试")
    stats = {"algorithm": "A-urgent direct + routine route-first merge + resource scheduling",
             "construction": construction, "greedy": search, "milp": milp_info,
             "seconds": time.perf_counter() - clock, "seed": args.seed,
             "max_stops": args.max_stops,
             "merge_energy_slack_kwh": args.merge_energy_slack_kwh,
             "global_optimality_proven": False,
             "note": "固定分层和有限次合并/排程搜索，不保证全局最优"}
    summary = save_tables(table_dir, ctx, best, seed_plan, stats, checks, output_root)
    summary.update({"method": stats["algorithm"], "construction": construction,
                     "independent_of_other_optimizers": True,
                    "global_optimality_proven": False})
    route_rows = [{
        "架次编号": s["batch_id"],
        "配送层": "紧急A型直送" if is_urgent(ctx, s["route"]) else "普通多点配送",
        "机型": s["type_id"], "服务区顺序": s["visit_order"],
        "货箱数": len(s["box_ids"]), "货箱编号": ";".join(s["box_ids"]),
        "载重（kg）": s["mass_kg"], "能耗（kWh）": s["energy_kwh"],
    } for s in best["sorties"]]
    write_csv(table_dir / "Q2_分层路线.csv", route_rows, list(route_rows[0]))
    comparison = compare_all_methods(output_root, summary, table_dir,
                                     args.soft_budget_s)
    write_csv(table_dir / "Q2_方案对照.csv", comparison, COMPARISON_COLUMNS)
    (table_dir / "Q2_运行摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (table_dir / "Q2_运行说明.txt").write_text(
        "方案2从原始箱单重新分层：全部硬时限箱按服务区组成 A 型直送架次；\n"
        "普通箱从单箱路线起，按节约能耗逐步合并，并检查载重、体积、返航 SOC 和交付期望。\n"
        "对已构造路线多次列表排程，并可用 MILP 调整实体机、电池和起飞时刻；\n"
        "14 组电池按两阶段模型充满才可复用。全程不复用其他优化方案的结果。\n"
        "Q2_方案对照.csv 同时列出基线及方案1–4的最晚返航、能耗和架次。\n"
        "如本次为 FAIL，表和图只用于定位硬时限冲突，不应作为可行解。\n"
        "本方法是有限路线和排程搜索，PASS 不等于全局最优。\n",
        encoding="utf-8")
    print(f"tables: {table_dir}", flush=True)
    save_figures(figure_dir, ctx, best, summary)
    print(f"figures: {figure_dir}", flush=True)
    print(f"Q2 method 2 {summary['status']}: {summary['sorties']} sorties; "
          f"hard={summary['hard_deadline_violations']}; "
          f"energy={summary['total_energy_kwh']:.6f} kWh; "
          "global optimum not proven", flush=True)


if __name__ == "__main__":
    main()
