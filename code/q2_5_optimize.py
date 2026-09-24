"""问题二方案 5：路线池整数组批 + 实体机/共享电池可行排程。

从同一已验收箱单重新生成候选组批与跨区路线。整数规划在已生成的
路线池中精确覆盖全部货箱；随后用原物理模型排程并独立验收。
分别搜索最晚返航、能耗与架次三个目标。路线池未穷尽，不宣称全局最优。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from _0_pipeline import sha256, verify_ready
from q1_0_baseline import CODE_DIR, find_latest_validated_run
from q2_0_baseline import DELIVERY_COLUMNS, SORTIE_COLUMNS, optional_seconds, write_csv
from q2_1_optimize import Context, Route, check_plan, initial_routes, save_figures, save_tables, simulate


TOL = 1e-6
METHODS = (
    ("1_optimize", "q2_opt_*_table", "优化1"),
    ("2_optimize", "q2_opt2_*_table", "优化2"),
    ("3_optimize", "q2_opt3_*_table", "优化3"),
    ("4_optimize", "q2_opt4_*_table", "优化4"),
)
PROFILES = ("time", "energy", "sorties")
NAMES = {"time": "最快返航", "energy": "最低能耗", "sorties": "最少架次"}


@dataclass(frozen=True)
class Column:
    route: Route
    boxes: frozenset[str]
    energy_lb: float
    duration_lb: float


def admissible(ctx: Context, plan: dict[str, Any] | None) -> bool:
    return (plan is not None and plan["score"][0] == 0 and
            plan["score"][2] <= TOL and
            Counter(bid for route in plan["routes"] for bid in route.box_ids) ==
            Counter(ctx.boxes.keys()))


def metric(plan: dict[str, Any], profile: str) -> tuple[float, ...]:
    score = plan["score"]
    if profile == "time":
        return score[3], score[4], score[5]
    if profile == "energy":
        return score[4], score[3], score[5]
    return score[5], score[4], score[3]


def load_existing(ctx: Context, output_root: Path) -> list[tuple[str, dict[str, Any]]]:
    """只读取同一哈希、PASS 且重新仿真/验收仍通过的历史方案。"""
    source_hash = sha256(ctx.data_run / "meta" / "validated_artifacts.json")
    valid: list[tuple[str, dict[str, Any]]] = []
    for folder, pattern, label in METHODS:
        directories = [*(output_root / folder).glob(pattern), *output_root.glob(pattern)]
        for directory in sorted(directories, key=lambda p: p.stat().st_mtime, reverse=True):
            summary_file = directory / "Q2_运行摘要.json"
            detail_file = directory / "Q2_架次与电池周转.csv"
            if not summary_file.is_file() or not detail_file.is_file():
                continue
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            if (summary.get("status") != "PASS" or
                    summary.get("source_manifest_sha256") != source_hash):
                continue
            rows = pd.read_csv(detail_file, dtype=str).to_dict("records")
            routes = []
            for row in sorted(rows, key=lambda r: r["batch_id"]):
                zones = str(row["visit_order"]).split(">")
                ids = str(row["box_ids"]).split(";")
                routes.append(ctx.route([
                    (zone, [bid for bid in ids if ctx.boxes[bid]["zone_id"] == zone])
                    for zone in zones
                ]))
            plan = simulate(ctx, routes)
            if admissible(ctx, plan) and all(c["status"] == "PASS" for c in check_plan(ctx, plan)):
                valid.append((label, plan))
                break
    return valid


class RoutePool:
    def __init__(self, ctx: Context, limit: int):
        self.ctx = ctx
        self.limit = limit
        self.columns: dict[Route, Column] = {}

    def add(self, route: Route) -> None:
        if route in self.columns or len(self.columns) >= self.limit:
            return
        if len(route.visits) > 4:
            return
        mass = sum(float(self.ctx.boxes[bid]["mass_kg"]) for bid in route.box_ids)
        volume = sum(float(self.ctx.boxes[bid]["volume_m3"]) for bid in route.box_ids)
        if mass > 80 + TOL or volume > 0.25 + TOL:
            return
        options = []
        for option in self.ctx.options(route).values():
            if all(
                option["completion_offsets"][bid] <=
                (float(deadline) if (deadline := optional_seconds(
                    self.ctx.boxes[bid]["hard_deadline_s"])) is not None
                 else float(self.ctx.boxes[bid]["expected_s"])) + TOL
                for bid in route.box_ids
            ):
                options.append(option)
        if options:
            self.columns[route] = Column(
                route, frozenset(route.box_ids),
                min(float(o["energy_kwh"]) for o in options),
                min(float(o["duration_s"]) for o in options),
            )


def route_without(ctx: Context, route: Route, box_id: str) -> Route | None:
    visits = [(z, [bid for bid in ids if bid != box_id]) for z, ids in route.visits]
    visits = [(z, ids) for z, ids in visits if ids]
    return ctx.route(visits) if visits else None


def route_with(ctx: Context, route: Route, box_id: str) -> list[Route]:
    if box_id in route.box_ids:
        return []
    zone = str(ctx.boxes[box_id]["zone_id"])
    visits = [(z, list(ids)) for z, ids in route.visits]
    if zone in route.zones:
        return [ctx.route([(z, [*ids, box_id] if z == zone else ids) for z, ids in visits])]
    if len(visits) >= 4:
        return []
    return [ctx.route([*visits[:at], (zone, [box_id]), *visits[at:]])
            for at in range(len(visits) + 1)]


def merged_routes(ctx: Context, first: Route, second: Route) -> list[Route]:
    groups: dict[str, list[str]] = defaultdict(list)
    for route in (first, second):
        for zone, ids in route.visits:
            groups[zone].extend(ids)
    if len(groups) > 4 or len(set(itertools.chain.from_iterable(groups.values()))) != sum(
            len(ids) for ids in groups.values()):
        return []
    return [ctx.route([(zone, groups[zone]) for zone in order])
            for order in itertools.permutations(groups)]


def build_pool(ctx: Context, plans: list[dict[str, Any]], limit: int,
               seed: int) -> RoutePool:
    rng = random.Random(seed)
    pool = RoutePool(ctx, limit)
    base = [route for plan in plans for route in plan["routes"]]
    for route in base:
        pool.add(route)
    for route in initial_routes(ctx):
        pool.add(route)
    by_zone: dict[str, list[str]] = defaultdict(list)
    for bid, box in ctx.boxes.items():
        by_zone[str(box["zone_id"])].append(bid)
        pool.add(ctx.route([(str(box["zone_id"]), [bid])]))
    for zone, ids in by_zone.items():
        hard = [bid for bid in ids if optional_seconds(ctx.boxes[bid]["hard_deadline_s"]) is not None]
        if hard:
            pool.add(ctx.route([(zone, hard)]))
        pool.add(ctx.route([(zone, ids)]))
        for a, b in itertools.combinations(ids, 2):
            pool.add(ctx.route([(zone, [a, b])]))
    for route in base:
        for zone, ids in route.visits:
            pool.add(ctx.route([(zone, list(ids))]))
        for bid in route.box_ids:
            reduced = route_without(ctx, route, bid)
            if reduced:
                pool.add(reduced)
        if len(route.visits) > 1:
            for order in itertools.permutations(route.visits):
                pool.add(Route(order))
    for plan in plans:
        for first, second in itertools.combinations(plan["routes"], 2):
            if len(pool.columns) >= limit:
                break
            for route in merged_routes(ctx, first, second):
                pool.add(route)
    for _ in range(min(8000, len(base) * 70)):
        if len(pool.columns) >= limit:
            break
        route = rng.choice(base)
        bid = rng.choice(tuple(ctx.boxes))
        if bid not in route.box_ids:
            for candidate in route_with(ctx, route, bid):
                pool.add(candidate)
    return pool


def solve_cover(columns: list[Column], required: frozenset[str], profile: str,
                seconds: float, rng: random.Random, max_routes: int | None = None
                ) -> tuple[list[Route] | None, dict[str, Any]]:
    available = [column for column in columns if column.boxes <= required]
    if not available or any(not any(bid in col.boxes for col in available)
                            for bid in required):
        return None, {"status": "missing_columns", "columns": len(available)}
    ids = sorted(required)
    row_of = {bid: i for i, bid in enumerate(ids)}
    rr, cc, vv = [], [], []
    for j, col in enumerate(available):
        for bid in col.boxes:
            rr.append(row_of[bid]); cc.append(j); vv.append(1.0)
        if max_routes is not None:
            rr.append(len(ids)); cc.append(j); vv.append(1.0)
    rows = len(ids) + int(max_routes is not None)
    matrix = coo_array((np.asarray(vv),
                        (np.asarray(rr, dtype=np.int32), np.asarray(cc, dtype=np.int32))),
                       shape=(rows, len(available))).tocsc()
    lower = np.ones(rows)
    upper = np.ones(rows)
    if max_routes is not None:
        lower[-1] = 0.0
        upper[-1] = float(max_routes)
    if profile == "energy":
        cost = [col.energy_lb + 0.001 for col in available]
    elif profile == "sorties":
        cost = [1.0 + 0.00001 * col.energy_lb for col in available]
    else:
        cost = [col.duration_lb + 0.01 * col.energy_lb for col in available]
    cost = np.asarray([value * (1.0 + rng.uniform(-0.0002, 0.0002))
                       for value in cost])
    result = milp(cost, integrality=np.ones(len(available), dtype=np.int32),
                  bounds=Bounds(np.zeros(len(available)), np.ones(len(available))),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={"time_limit": max(0.1, seconds), "mip_rel_gap": 0.005})
    info = {"status": int(result.status), "message": str(result.message),
            "columns": len(available),
            "objective": None if result.fun is None else float(result.fun),
            "pool_dual_bound": None if getattr(result, "mip_dual_bound", None) is None
            else float(result.mip_dual_bound),
            "pool_gap": None if getattr(result, "mip_gap", None) is None
            else float(result.mip_gap)}
    if result.x is None:
        return None, info
    chosen = [col.route for col, value in zip(available, result.x) if value > 0.5]
    if Counter(bid for route in chosen for bid in route.box_ids) != Counter(required):
        raise AssertionError("整数覆盖结果没有恰好覆盖待修复货箱")
    return chosen, info


def route_priority(ctx: Context, route: Route) -> tuple[float, float, str]:
    hard = min((d for bid in route.box_ids
                if (d := optional_seconds(ctx.boxes[bid]["hard_deadline_s"])) is not None),
               default=math.inf)
    expected = min(float(ctx.boxes[bid]["expected_s"]) for bid in route.box_ids)
    return hard, expected, route.box_ids[0]


def schedule_orders(ctx: Context, base: list[Route], selected: list[Route],
                    removed: set[int], rng: random.Random, attempts: int
                    ) -> list[list[Route]]:
    anchor = min(removed) if removed else 0
    kept = [route for i, route in enumerate(base) if i not in removed]
    variants = [sorted(selected, key=lambda r: route_priority(ctx, r)),
                sorted(selected, key=lambda r: min(
                    float(o["duration_s"]) for o in ctx.options(r).values())),
                sorted(selected, key=lambda r: min(
                    float(o["energy_kwh"]) for o in ctx.options(r).values()))]
    variants.append(selected)
    for _ in range(max(0, attempts - len(variants))):
        shuffled = list(selected)
        rng.shuffle(shuffled)
        variants.append(shuffled)
    orders = []
    for variant in variants:
        # 原块位置及紧急任务靠前两种拼接方式。
        orders.append([*kept[:anchor], *variant, *kept[anchor:]])
        orders.append(sorted([*kept, *variant], key=lambda r: route_priority(ctx, r)))
    return orders


def evaluate_repair(ctx: Context, incumbent: dict[str, Any],
                    selected: list[Route], removed: set[int], profile: str,
                    rng: random.Random) -> dict[str, Any] | None:
    best = None
    for routes in schedule_orders(ctx, incumbent["routes"], selected, removed,
                                 rng, attempts=10):
        if routes == incumbent["routes"]:
            continue
        trial = simulate(ctx, routes)
        if admissible(ctx, trial) and (best is None or metric(trial, profile) < metric(best, profile)):
            best = trial
    return best


def search_profile(ctx: Context, pool: RoutePool, seed_plan: dict[str, Any],
                   profile: str, seconds: float, rng: random.Random
                   ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = time.perf_counter() + seconds
    best = seed_plan
    logs = []
    columns = list(pool.columns.values())
    all_boxes = frozenset(ctx.boxes)
    round_no = 0
    while time.perf_counter() < deadline:
        round_no += 1
        remaining = deadline - time.perf_counter()
        if round_no == 1:
            removed = set(range(len(best["routes"])))
            budget = min(max(1.0, seconds * 0.18), remaining)
        else:
            count = rng.randint(4, min(12, len(best["routes"])))
            removed = set(rng.sample(range(len(best["routes"])), count))
            budget = min(3.0, remaining)
        required = (all_boxes if len(removed) == len(best["routes"])
                    else frozenset(bid for i in removed for bid in best["routes"][i].box_ids))
        max_routes = (len(removed) if profile == "sorties" else len(removed) + 2)
        selected, info = solve_cover(columns, required, profile, budget, rng, max_routes)
        improved = False
        if selected is not None:
            trial = evaluate_repair(ctx, best, selected, removed, profile, rng)
            if trial is not None and metric(trial, profile) < metric(best, profile):
                if all(c["status"] == "PASS" for c in check_plan(ctx, trial)):
                    best = trial
                    improved = True
        logs.append({"profile": profile, "round": round_no,
                     "boxes_repaired": len(required), "columns": info["columns"],
                     "master_status": info["status"], "master_gap": info.get("pool_gap"),
                     "master_objective": info.get("objective"),
                     "scheduled_improvement": improved,
                     "best_makespan_s": best["score"][3],
                     "best_energy_kwh": best["score"][4],
                     "best_sorties": best["score"][5]})
        if improved:
            print(f"{NAMES[profile]} improved: return={best['score'][3]:.1f}s, "
                  f"energy={best['score'][4]:.3f}kWh, sorties={best['score'][5]}", flush=True)
    return best, logs


def profile_rows(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sorties = [{"架次编号": s["batch_id"], "无人机编号": s["uav_id"],
                "机型编号": s["type_id"], "电池编号": s["battery_id"],
                "开始时刻（s）": s["start_s"], "访问服务区顺序": s["visit_order"],
                "返回O01时刻（s）": s["return_s"], "架次能耗（kWh）": s["energy_kwh"]}
               for s in plan["sorties"]]
    deliveries = [{"货箱编号": d["box_id"], "架次编号": d["batch_id"],
                   "服务区编号": d["zone_id"], "交付完成时刻（s）": d["complete_s"]}
                  for d in plan["deliveries"]]
    return sorties, deliveries


def comparison_rows(output_root: Path, source_hash: str,
                    plans: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for folder, pattern, label in (("0_baseline", "q2_base_*_table", "基线"), *METHODS):
        paths = [*(output_root / folder).glob(pattern), *output_root.glob(pattern)]
        for directory in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True):
            summary_file = directory / "Q2_运行摘要.json"
            if not summary_file.is_file():
                continue
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            if summary.get("source_manifest_sha256") != source_hash:
                continue
            rows.append({"方案": label, "状态": summary["status"],
                         "硬时限违约箱数": summary["hard_deadline_violations"],
                         "软箱加权迟到（s）": summary["soft_weighted_lateness_s"],
                         "最晚返航（s）": summary["makespan_s"],
                         "运输总能耗（kWh）": summary["total_energy_kwh"],
                         "架次数": summary["sorties"]})
            break
    for profile, plan in plans.items():
        rows.append({"方案": f"方案5·{NAMES[profile]}", "状态": "PASS",
                     "硬时限违约箱数": plan["score"][0],
                     "软箱加权迟到（s）": plan["score"][2],
                     "最晚返航（s）": plan["score"][3],
                     "运输总能耗（kWh）": plan["score"][4],
                     "架次数": plan["score"][5]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2 方案5：路线池整数组批及三目标资源排程")
    parser.add_argument("--data-run", type=Path, help="已验收数据目录，默认最新")
    parser.add_argument("--output-root", type=Path, help="默认 code/2_outputs")
    parser.add_argument("--time-limit-s", type=float, default=360.0,
                        help="三项搜索总秒数，默认360")
    parser.add_argument("--pool-limit", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--dry-run", action="store_true", help="只搜索和验收，不写输出")
    args = parser.parse_args()
    if args.time_limit_s <= 0 or args.pool_limit < 200:
        parser.error("时间必须为正，路线池上限至少200")
    data_run = args.data_run.resolve() if args.data_run else find_latest_validated_run()
    if not verify_ready(data_run):
        raise ValueError(f"输入数据未通过哈希验收：{data_run}")
    ctx = Context(data_run)
    output_root = args.output_root.resolve() if args.output_root else CODE_DIR / "2_outputs"
    existing = load_existing(ctx, output_root)
    if not existing:
        raise RuntimeError("没有同一输入数据下的已验收零迟到方案")
    print("warm starts:", [(name, p["score"][3:]) for name, p in existing], flush=True)
    pool = build_pool(ctx, [p for _, p in existing], args.pool_limit, args.seed)
    print(f"physical route columns: {len(pool.columns)}", flush=True)
    rng = random.Random(args.seed)
    solutions = {}
    logs = []
    for profile in PROFILES:
        seed_plan = min((p for _, p in existing), key=lambda p: metric(p, profile))
        best, profile_logs = search_profile(ctx, pool, seed_plan, profile,
                                            args.time_limit_s / 3, rng)
        solutions[profile] = best
        logs.extend(profile_logs)
        print(f"{NAMES[profile]}: {best['score']}", flush=True)
    for profile, plan in solutions.items():
        checks = check_plan(ctx, plan)
        if not admissible(ctx, plan) or any(c["status"] != "PASS" for c in checks):
            raise AssertionError(f"{profile} 最终方案未通过独立验收")
    if not verify_ready(data_run):
        raise ValueError("搜索期间输入数据发生改变")
    if args.dry_run:
        return
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    method_dir = output_root / "5_optimize"
    table_dir = method_dir / f"q2_opt5_{stamp}_table"
    figure_dir = method_dir / f"q2_opt5_{stamp}_figure"
    if table_dir.exists() or figure_dir.exists():
        raise FileExistsError("本次结果目录已存在")
    main_plan = solutions["energy"]
    search_info = {"seconds": args.time_limit_s,
                   "route_pool_size": len(pool.columns),
                   "master_runs": len(logs),
                   "route_pool_complete": False,
                   "global_optimality_proven": False}
    summary = save_tables(table_dir, ctx, main_plan,
                          min((p for _, p in existing), key=lambda p: metric(p, "energy")),
                          search_info, check_plan(ctx, main_plan), output_root)
    summary.update({"method": "route-pool set partitioning and feasible scheduling",
                    "global_optimality_proven": False,
                    "route_pool_complete": False,
                    "profiles": {profile: {"makespan_s": plan["score"][3],
                                           "energy_kwh": plan["score"][4],
                                           "sorties": plan["score"][5]}
                                 for profile, plan in solutions.items()},
                    "note": "MILP bounds apply only to the generated route pool and omit resource calendars; full Q2 global optimality is not certified."})
    for profile, plan in solutions.items():
        sorties, deliveries = profile_rows(plan)
        write_csv(table_dir / f"Q2_{profile}_运输架次.csv", sorties, SORTIE_COLUMNS)
        write_csv(table_dir / f"Q2_{profile}_逐箱交付.csv", deliveries, DELIVERY_COLUMNS)
    write_csv(table_dir / "Q2_路线池求解记录.csv", logs, list(logs[0]))
    comparison = comparison_rows(output_root, sha256(data_run / "meta" / "validated_artifacts.json"),
                                 solutions)
    write_csv(table_dir / "Q2_方案对照.csv", comparison, list(comparison[0]))
    (table_dir / "Q2_运行摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (table_dir / "Q2_运行说明.txt").write_text(
        "方案5重新生成跨区路线池，整数规划精确覆盖80箱，再做实体机/共享电池排程。\n"
        "time、energy、sorties 三套官方格式表对应三个单目标结果；无前缀表为energy。\n"
        "所有输出均经原物理模型与独立验收；路线池未穷尽，MILP的池内界不等于全局界。\n",
        encoding="utf-8")
    save_figures(figure_dir, ctx, main_plan, summary)
    print(f"tables: {table_dir}\nfigures: {figure_dir}", flush=True)


if __name__ == "__main__":
    main()
