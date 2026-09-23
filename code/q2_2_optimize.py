"""D 题问题二优化方案 2：可行性约束下的大邻域搜索。

复用 q2_1 的同一物理/事件模拟器，以已验证方案热启动。反复合并路线、
拆除低载重架次并逐箱重插，优先减少能耗。输出三种软时效预算的权衡方案。
算法为启发式；另报告一个严格但较宽松的最少架次物理下界。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from _0_pipeline import sha256, verify_ready
from q1_0_baseline import CODE_DIR, find_latest_validated_run
from q2_0_baseline import (
    DELIVERY_COLUMNS, SORTIE_COLUMNS, validate_template, write_csv,
)
from q2_1_optimize import (
    MAX_SEARCH_STOPS, TOL, Context, Route, check_plan, initial_routes,
    save_figures, simulate,
)


PHASES = (
    ("按时优先", 0.0, 0.0020, 0.0),
    ("均衡推荐", 15000.0, 0.0010, 0.00020),
    ("节能优先", 60000.0, 0.0005, 0.00005),
)


def is_admissible(plan: dict[str, Any] | None, soft_cap: float) -> bool:
    return (plan is not None and plan["score"][0] == 0 and
            plan["score"][2] <= soft_cap + TOL)


def objective(plan: dict[str, Any], time_weight: float,
              late_weight: float) -> float:
    score = plan["score"]
    return score[4] + time_weight * score[3] + late_weight * score[2]


def saved_solution(ctx: Context, output_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """仅使用同一份已验收输入且通过全部硬约束的本地历史方案。"""
    source_hash = sha256(ctx.data_run / "meta" / "validated_artifacts.json")
    directories = [
        *output_root.glob("q2_opt_*_table"),
        *output_root.glob("q2_opt2_*_table"),
    ]
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
            zone_order = str(row["visit_order"]).split(">")
            ids = str(row["box_ids"]).split(";")
            routes.append(ctx.route([
                (zone, [bid for bid in ids if ctx.boxes[bid]["zone_id"] == zone])
                for zone in zone_order
            ]))
        plan = simulate(ctx, routes)
        if (is_admissible(plan, 0.0) and
                Counter(bid for route in routes for bid in route.box_ids) ==
                Counter(ctx.boxes.keys()) and
                all(row["status"] == "PASS" for row in check_plan(ctx, plan))):
            return plan, str(directory)
    return None, None


def initial_solution(ctx: Context, output_root: Path) -> tuple[dict[str, Any], str]:
    warm, origin = saved_solution(ctx, output_root)
    seed = simulate(ctx, initial_routes(ctx))
    if seed is None:
        raise RuntimeError("构造初始排程失败")
    candidates = [(plan, name) for plan, name in ((warm, origin), (seed, "重新组批初始方案"))
                  if is_admissible(plan, 0.0)]
    if not candidates:
        raise RuntimeError("没有零硬时限违约、零软迟到的热启动方案")
    return min(candidates, key=lambda x: (x[0]["score"][4], x[0]["score"][3]))  # type: ignore[index]


def join_route_options(ctx: Context, first: Route, second: Route) -> list[Route]:
    grouped: dict[str, list[str]] = {}
    for route in (first, second):
        for zone, ids in route.visits:
            grouped.setdefault(zone, []).extend(ids)
    if len(grouped) > MAX_SEARCH_STOPS:
        return []
    results = []
    for order in itertools.permutations(grouped):
        route = ctx.route([(zone, grouped[zone]) for zone in order])
        if ctx.options(route):
            results.append(route)
    return results


def merge_step(ctx: Context, incumbent: dict[str, Any], soft_cap: float,
               time_weight: float, late_weight: float,
               deadline: float) -> dict[str, Any]:
    """穷查一次两条路线合并；只接受逐箱硬截止与软迟到预算均满足的方案。"""
    routes = incumbent["routes"]
    best = incumbent
    best_value = objective(best, time_weight, late_weight)
    # 小架次先尝试合并，优先消除固定开销。
    pairs = sorted(itertools.combinations(range(len(routes)), 2),
                   key=lambda ij: (len(routes[ij[0]].box_ids) +
                                   len(routes[ij[1]].box_ids), ij))
    for i, j in pairs:
        if time.perf_counter() >= deadline:
            break
        for joined in join_route_options(ctx, routes[i], routes[j]):
            remainder = [route for k, route in enumerate(routes) if k not in (i, j)]
            for place in {min(i, len(remainder)), min(j - 1, len(remainder)), 0}:
                candidate = list(remainder)
                candidate.insert(place, joined)
                trial = simulate(ctx, candidate)
                if is_admissible(trial, soft_cap):
                    value = objective(trial, time_weight, late_weight)  # type: ignore[arg-type]
                    if value + 1e-8 < best_value:
                        best, best_value = trial, value
    return best


def zone_distance(ctx: Context, zone: str, route: Route) -> float:
    if zone in route.zones:
        return 0.0
    return min(float(ctx.arcs[(zone, other)]["distance_m"])
               for other in route.zones)


def insertion_options(ctx: Context, route: Route, box_id: str) -> list[Route]:
    zone = str(ctx.boxes[box_id]["zone_id"])
    visits = [(z, list(ids)) for z, ids in route.visits]
    if zone in route.zones:
        for z, ids in visits:
            if z == zone:
                ids.append(box_id)
                break
        candidate = ctx.route(visits)
        return [candidate] if ctx.options(candidate) else []
    if len(visits) >= MAX_SEARCH_STOPS:
        return []
    options = []
    for place in range(len(visits) + 1):
        changed = list(visits)
        changed.insert(place, (zone, [box_id]))
        candidate = ctx.route(changed)
        if ctx.options(candidate):
            options.append(candidate)
    return options


def repair(ctx: Context, routes: list[Route], removed: list[str],
           soft_cap: float, time_weight: float, late_weight: float,
           deadline: float, rng: random.Random,
           max_targets: int) -> dict[str, Any] | None:
    """移除整架次后逐箱重插；每次候选都重新排具体机和共享电池。"""
    removed.sort(key=lambda bid: (
        ctx.boxes[bid]["hard_deadline_s"]
        if pd.notna(ctx.boxes[bid]["hard_deadline_s"]) else math.inf,
        float(ctx.boxes[bid]["expected_s"]),
        -float(ctx.boxes[bid]["mass_kg"]), bid,
    ))
    for box_id in removed:
        if time.perf_counter() >= deadline:
            return None
        zone = str(ctx.boxes[box_id]["zone_id"])
        ranked = sorted(range(len(routes)),
                        key=lambda i: (zone_distance(ctx, zone, routes[i]), i))
        targets = ranked[:max_targets]
        if len(ranked) > max_targets:
            targets.extend(rng.sample(ranked[max_targets:],
                                      min(2, len(ranked) - max_targets)))
        best_routes = None
        best_value = math.inf
        for i in targets:
            for inserted in insertion_options(ctx, routes[i], box_id):
                candidate = list(routes)
                candidate[i] = inserted
                trial = simulate(ctx, candidate)
                if is_admissible(trial, soft_cap):
                    value = objective(trial, time_weight, late_weight)  # type: ignore[arg-type]
                    if value < best_value:
                        best_value, best_routes = value, candidate
        singleton = ctx.route([(zone, [box_id])])
        if ctx.options(singleton):
            n = len(routes)
            places = {0, n, n // 4, n // 2, (3 * n) // 4}
            if ctx.boxes[box_id]["hard_deadline_s"] is not None and pd.notna(
                    ctx.boxes[box_id]["hard_deadline_s"]):
                places.update(range(min(9, n + 1)))
            for place in places:
                candidate = list(routes)
                candidate.insert(place, singleton)
                trial = simulate(ctx, candidate)
                if is_admissible(trial, soft_cap):
                    value = objective(trial, time_weight, late_weight)  # type: ignore[arg-type]
                    if value < best_value:
                        best_value, best_routes = value, candidate
        if best_routes is None:
            return None
        routes = best_routes
    if Counter(bid for route in routes for bid in route.box_ids) != Counter(ctx.boxes.keys()):
        raise AssertionError("拆除重插后未覆盖全部 80 箱")
    return simulate(ctx, routes)


def destroy_repair(ctx: Context, current: dict[str, Any], soft_cap: float,
                   time_weight: float, late_weight: float, deadline: float,
                   rng: random.Random, max_targets: int) -> dict[str, Any] | None:
    routes = current["routes"]
    if len(routes) < 2:
        return None
    # 更常选择单位货箱能耗较高的架次，仍保留少量随机探索。
    energy_by_id = {s["batch_id"]: s["energy_kwh"] for s in current["sorties"]}
    weighted = sorted(range(len(routes)), key=lambda i: (
        energy_by_id[f"Q{i + 1:03d}"] / len(routes[i].box_ids),
        rng.random(),
    ), reverse=True)
    remove_n = 1 if rng.random() < 0.7 else 2
    chosen = set(rng.sample(weighted[:max(3, len(weighted) // 2)],
                            min(remove_n, len(weighted))))
    removed = [bid for i in chosen for bid in routes[i].box_ids]
    remaining = [route for i, route in enumerate(routes) if i not in chosen]
    return repair(ctx, remaining, removed, soft_cap, time_weight, late_weight,
                  deadline, rng, max_targets)


def search_phase(ctx: Context, start_plan: dict[str, Any], soft_cap: float,
                 time_weight: float, late_weight: float, deadline: float,
                 rng: random.Random, max_targets: int) -> tuple[dict[str, Any], dict[str, Any]]:
    best = current = start_plan
    attempts = improvements = 0
    # 确定性合并先消掉明显多余的小架次。
    while time.perf_counter() < deadline:
        merged = merge_step(ctx, best, soft_cap, time_weight, late_weight, deadline)
        if objective(merged, time_weight, late_weight) >= objective(best, time_weight, late_weight) - 1e-8:
            break
        best = current = merged
        improvements += 1
    while time.perf_counter() < deadline:
        attempts += 1
        trial = destroy_repair(ctx, current, soft_cap, time_weight, late_weight,
                               deadline, rng, max_targets)
        if not is_admissible(trial, soft_cap):
            continue
        trial_value = objective(trial, time_weight, late_weight)  # type: ignore[arg-type]
        best_value = objective(best, time_weight, late_weight)
        if trial_value < best_value - 1e-8:
            best = trial
            improvements += 1
            print(f"  improved: energy={best['score'][4]:.3f} kWh, "
                  f"sorties={best['score'][5]}, "
                  f"soft late={best['score'][2]:.1f}", flush=True)
        current_value = objective(current, time_weight, late_weight)
        temperature = max(0.3, 3.0 * (1.0 - attempts / (attempts + 250)))
        if (trial_value <= current_value or
                rng.random() < math.exp(-min((trial_value - current_value) /
                                            temperature, 700))):
            current = trial
        if attempts % 20 == 0:
            current = best
            merged = merge_step(ctx, best, soft_cap, time_weight, late_weight, deadline)
            if objective(merged, time_weight, late_weight) < objective(best, time_weight, late_weight) - 1e-8:
                best = current = merged
                improvements += 1
    return best, {"attempts": attempts, "improvements": improvements,
                  "soft_lateness_cap_s": soft_cap, "time_weight": time_weight,
                  "late_weight": late_weight}


def physical_sortie_lower_bound(ctx: Context) -> dict[str, Any]:
    total_mass = sum(float(box["mass_kg"]) for box in ctx.boxes.values())
    total_volume = sum(float(box["volume_m3"]) for box in ctx.boxes.values())
    maximum_mass = max(float(model["max_payload_kg"]) for model in ctx.models.values())
    maximum_volume = max(float(model["capacity_m3"]) for model in ctx.models.values())
    by_mass = math.ceil((total_mass - TOL) / maximum_mass)
    by_volume = math.ceil((total_volume - TOL) / maximum_volume)
    return {
        "total_mass_kg": total_mass,
        "total_volume_m3": total_volume,
        "max_payload_kg": maximum_mass,
        "max_volume_m3": maximum_volume,
        "sortie_lower_bound_from_mass": by_mass,
        "sortie_lower_bound_from_volume": by_volume,
        "global_sortie_lower_bound": max(by_mass, by_volume),
        "energy_lower_bound_kwh": None,
        "interpretation": (
            "全球最少架次的必要物理下界；忽略路线、时限和电池，"
            "因此不能据此证明当前方案接近最优，也不构成能耗下界。"
        ),
    }


def table_rows(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sortie_rows = [{
        "架次编号": s["batch_id"], "无人机编号": s["uav_id"],
        "机型编号": s["type_id"], "电池编号": s["battery_id"],
        "开始时刻（s）": s["start_s"], "访问服务区顺序": s["visit_order"],
        "返回O01时刻（s）": s["return_s"], "架次能耗（kWh）": s["energy_kwh"],
    } for s in plan["sorties"]]
    delivery_rows = [{
        "货箱编号": d["box_id"], "架次编号": d["batch_id"],
        "服务区编号": d["zone_id"], "交付完成时刻（s）": d["complete_s"],
    } for d in plan["deliveries"]]
    return sortie_rows, delivery_rows


def previous_comparisons(output_root: Path, source_hash: str) -> list[dict[str, Any]]:
    records = []
    for pattern, label in (
        ("q2_base_*_table", "固定FFD基线"),
        ("q2_opt_*_table", "优化方案1"),
    ):
        for directory in sorted(output_root.glob(pattern), reverse=True):
            path = directory / "Q2_运行摘要.json"
            if not path.is_file():
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("source_manifest_sha256") != source_hash:
                continue
            records.append({
                "方案": label, "状态": summary["status"],
                "硬时限违约箱数": summary["hard_deadline_violations"],
                "软箱加权迟到": summary["soft_weighted_lateness_s"],
                "最晚返航（s）": summary["makespan_s"],
                "运输总能耗（kWh）": summary["total_energy_kwh"],
                "架次数": summary["sorties"],
            })
            break
    return records


def save_tables(table_dir: Path, ctx: Context,
                solutions: dict[str, dict[str, Any]],
                stats: dict[str, dict[str, Any]],
                origin: str, output_root: Path) -> dict[str, Any]:
    if table_dir.exists():
        raise FileExistsError(f"不覆盖已有结果：{table_dir}")
    validate_template()
    table_dir.mkdir(parents=True)
    main = solutions["均衡推荐"]
    validation = {}
    for name, plan in solutions.items():
        checks = check_plan(ctx, plan)
        validation[name] = checks
        rows, deliveries = table_rows(plan)
        prefix = "" if name == "均衡推荐" else f"备选_{name}_"
        write_csv(table_dir / f"{prefix}Q2_运输架次.csv",
                  rows, SORTIE_COLUMNS)
        write_csv(table_dir / f"{prefix}Q2_逐箱交付.csv",
                  deliveries, DELIVERY_COLUMNS)
        if name != "均衡推荐":
            write_csv(table_dir / f"备选_{name}_验收检查.csv",
                      checks, ["check_id", "status", "detail"])
    write_csv(table_dir / "Q2_验收检查.csv",
              validation["均衡推荐"], ["check_id", "status", "detail"])
    detail = [{key: value for key, value in s.items()
               if key not in {"route", "legs", "visits"}}
              for s in main["sorties"]]
    for row in detail:
        row["box_ids"] = ";".join(row["box_ids"])
    write_csv(table_dir / "Q2_架次与电池周转.csv", detail, list(detail[0]))
    write_csv(table_dir / "Q2_逐箱时限检查.csv",
              main["deliveries"], list(main["deliveries"][0]))
    leg_rows = []
    for sortie in main["sorties"]:
        for index, leg in enumerate(sortie["legs"], 1):
            leg_rows.append({
                "架次编号": sortie["batch_id"], "航段序号": index,
                "起点": leg["from_id"], "终点": leg["to_id"],
                "离开时载货质量（kg）": leg["load_kg"],
                "飞行时间（s）": leg["flight_s"],
                "航段能耗（kWh）": leg["energy_kwh"],
                "抵达时刻（s）": sortie["start_s"] + leg["arrival_offset_s"],
            })
    write_csv(table_dir / "Q2_路线航段.csv", leg_rows, list(leg_rows[0]))
    source_hash = sha256(ctx.data_run / "meta" / "validated_artifacts.json")
    comparison = previous_comparisons(output_root, source_hash)
    phase_summary = {}
    for name, plan in solutions.items():
        fails = [c for c in validation[name] if c["status"] == "FAIL"]
        state = "PASS" if not fails else "FAIL"
        score = plan["score"]
        phase_summary[name] = {
            "status": state, "hard_deadline_violations": score[0],
            "soft_weighted_lateness_s": score[2],
            "makespan_s": score[3], "total_energy_kwh": score[4],
            "sorties": score[5], "search": stats[name],
        }
        comparison.append({
            "方案": f"优化方案2·{name}", "状态": state,
            "硬时限违约箱数": score[0], "软箱加权迟到": score[2],
            "最晚返航（s）": score[3], "运输总能耗（kWh）": score[4],
            "架次数": score[5],
        })
    write_csv(table_dir / "Q2_方案对照.csv",
              comparison, list(comparison[0]))
    lower_bound = physical_sortie_lower_bound(ctx)
    (table_dir / "Q2_最少架次下界.json").write_text(
        json.dumps(lower_bound, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    selected = phase_summary["均衡推荐"]
    fails = [check for check in validation["均衡推荐"] if check["status"] == "FAIL"]
    summary = {
        "status": selected["status"],
        "method": "energy-oriented ruin-and-recreate large neighborhood search",
        "global_optimality_proven": False,
        "source_data_run": str(ctx.data_run),
        "source_manifest_sha256": source_hash,
        "warm_start": origin,
        "recommended_profile": "均衡推荐",
        "sorties": selected["sorties"], "delivered_boxes": len(main["deliveries"]),
        "multi_zone_sorties": sum(len(s["route"].visits) > 1 for s in main["sorties"]),
        "hard_deadline_violations": selected["hard_deadline_violations"],
        "soft_weighted_lateness_s": selected["soft_weighted_lateness_s"],
        "makespan_s": selected["makespan_s"],
        "total_energy_kwh": selected["total_energy_kwh"],
        "first_failure": fails[0] if fails else None,
        "profiles": phase_summary,
        "sortie_lower_bound": lower_bound["global_sortie_lower_bound"],
        "note": "Only the sortie count has a rigorous, loose lower bound. "
                "No global energy or makespan optimum is claimed.",
    }
    (table_dir / "Q2_运行摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    (table_dir / "Q2_优化状态.txt").write_text(
        f"{summary['status']}\n"
        f"first_failure={summary['first_failure']}\n"
        "global_optimality_proven=False\n", encoding="utf-8")
    (table_dir / "Q2_运行说明.txt").write_text(
        "主提交表为 Q2_运输架次.csv 与 Q2_逐箱交付.csv；采用“均衡推荐”。\n"
        "另有按时优先、节能优先两套备选表；各自均有独立验收检查。\n"
        "所有方案仍沿用同一逐段能耗、逐箱交付、实体机及共享电池事件模拟器。\n"
        "Q2_最少架次下界只给出质量/体积放松后的必要下界，不是能耗最优证明。\n"
        "若旧基线硬截止失败，对照表仅展示原始数值，不计算改善百分比。\n",
        encoding="utf-8")
    if summary["status"] == "PASS":
        (table_dir / "Q2_READY.txt").write_text(
            "Q2 plan passed independent checks; heuristic optimality is not proven.\n",
            encoding="utf-8")
    for filename, columns, count in (
        ("Q2_运输架次.csv", SORTIE_COLUMNS, len(main["sorties"])),
        ("Q2_逐箱交付.csv", DELIVERY_COLUMNS, len(main["deliveries"])),
    ):
        saved = pd.read_csv(table_dir / filename, encoding="utf-8-sig")
        if list(saved.columns) != columns or len(saved) != count:
            raise ValueError(f"官方格式表导出复核失败：{filename}")
    return summary


def save_tradeoff_figure(figure_dir: Path, solutions: dict[str, dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIR / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"按时优先": "#3A80C1", "均衡推荐": "#27A275", "节能优先": "#DB873F"}
    labels = {"按时优先": "On-time", "均衡推荐": "Balanced", "节能优先": "Energy"}
    for name, plan in solutions.items():
        score = plan["score"]
        ax.scatter(score[3] / 3600, score[4], s=80, color=colors[name], label=labels[name])
        ax.annotate(f"{labels[name]}: {score[5]} sorties, late={score[2]:.0f}",
                    (score[3] / 3600, score[4]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("Latest return (h)")
    ax.set_ylabel("Transport energy (kWh)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q2_权衡比较.{extension}", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="问题二优化方案2：大邻域搜索与三种权衡")
    parser.add_argument("--data-run", type=Path, help="已验收数据目录；默认最新")
    parser.add_argument("--output-root", type=Path, help="默认 code/2_outputs")
    parser.add_argument("--time-limit-s", type=float, default=180.0,
                        help="三种方案的搜索总时间，默认180秒")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--insertion-targets", type=int, default=8)
    args = parser.parse_args()
    if args.time_limit_s <= 0 or args.insertion_targets < 1:
        parser.error("time-limit-s 和 insertion-targets 必须为正")
    data_run = args.data_run.resolve() if args.data_run else find_latest_validated_run()
    if not verify_ready(data_run):
        raise ValueError(f"输入数据未通过哈希验收：{data_run}")
    ctx = Context(data_run)
    output_root = args.output_root.resolve() if args.output_root else CODE_DIR / "2_outputs"
    initial, origin = initial_solution(ctx, output_root)
    print(f"warm start: {origin}; energy={initial['score'][4]:.3f} kWh, "
          f"sorties={initial['score'][5]}", flush=True)
    rng = random.Random(args.seed)
    solutions: dict[str, dict[str, Any]] = {}
    stats: dict[str, dict[str, Any]] = {}
    per_phase = args.time_limit_s / len(PHASES)
    starting = initial
    for name, soft_cap, time_weight, late_weight in PHASES:
        print(f"search {name}: weighted soft-lateness cap={soft_cap}", flush=True)
        deadline = time.perf_counter() + per_phase
        best, phase_stats = search_phase(
            ctx, starting, soft_cap, time_weight, late_weight, deadline,
            rng, args.insertion_targets)
        if best["score"][4] > starting["score"][4] + TOL:
            best = starting
            phase_stats["energy_guard_reverted"] = True
        solutions[name] = best
        stats[name] = phase_stats
        starting = best
    if not verify_ready(data_run):
        raise ValueError("搜索期间输入数据发生改变")
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    table_dir = output_root / f"q2_opt2_{stamp}_table"
    figure_dir = output_root / f"q2_opt2_{stamp}_figure"
    if table_dir.exists() or figure_dir.exists():
        raise FileExistsError("输出目录已存在，请稍后重试或指定新的 --output-root")
    summary = save_tables(table_dir, ctx, solutions, stats, origin, output_root)
    print(f"tables: {table_dir}", flush=True)
    save_figures(figure_dir, ctx, solutions["均衡推荐"], summary)
    save_tradeoff_figure(figure_dir, solutions)
    print(f"figures: {figure_dir}", flush=True)
    for name, plan in solutions.items():
        score = plan["score"]
        print(f"{name}: hard={score[0]}, soft={score[2]:.1f}, "
              f"return={score[3]:.1f}s, energy={score[4]:.3f}kWh, "
              f"sorties={score[5]}")
    print("Heuristic solution; global optimum not proven.")


if __name__ == "__main__":
    main()
