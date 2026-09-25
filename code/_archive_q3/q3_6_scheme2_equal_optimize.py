"""问题三：以师兄已验收的方案二为起点，开放运输架次的四目标等权搜索。

保留原方案二表格作为回退；只有独立验收 PASS 且四项归一化等权分数
严格下降时才接纳新解。搜索可拆分原运输批次，绝不设置 31 架次上限。
有限时间搜索不构成全局最优证明。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import q3_1_optimize as base
import q3_2_optimize as senior
from q2_1_optimize import Context


CODE_DIR = Path(__file__).resolve().parent


def _latest_scheme2(root: Path) -> Path:
    for path in sorted(root.glob("q3_opt2_*_table"), reverse=True):
        if not (path / "Q3_READY.txt").is_file():
            continue
        summary = json.loads((path / "Q3_运行摘要.json").read_text(encoding="utf-8"))
        if summary.get("status") == "PASS":
            return path
    raise FileNotFoundError("找不到已验收的方案二；请指定 --scheme2-table")


def _default_data_run() -> Path:
    """优先使用当前 code；Git 检出的换行损坏验收哈希时用同步工作目录。"""
    roots = [CODE_DIR,
             Path("D:/Documents/ChatGPT/研究生数学建模比赛/D题/code")]
    for root in roots:
        try:
            return base.find_data_run(root, base.EXPECTED_MANIFEST).resolve()
        except FileNotFoundError:
            continue
    raise FileNotFoundError("没有完整通过哈希验收的 0_outputs；请用 --data-run 指定")


def _reference(metrics: dict[str, float | int]) -> dict[str, float]:
    return {
        "soft_weighted_lateness_s": float(metrics["weighted_lateness"]),
        "joint_makespan_s": float(metrics["makespan_s"]),
        "total_energy_kwh": float(metrics["energy_kwh"]),
        "total_sorties": float(metrics["transport_sorties"] + metrics["relay_sorties"]),
    }


def _score(metrics: dict[str, float | int], reference: dict[str, float]) -> float:
    return base.equal_weight_score(_reference(metrics), reference)


def _is_improvement(metrics: dict[str, float | int], best_score: float,
                    reference: dict[str, float]) -> bool:
    # 架次不是独立硬约束；它仅作为总架次进入等权目标。
    return _score(metrics, reference) < best_score - 1e-7


def _partition_routes(ctx: Context, choice: base.JointChoice) -> list[tuple[Any, Any]]:
    """每条原路线允许拆成两条非空路线，保留服务区访问顺序。"""
    route = choice.task.route
    ids = list(route.box_ids)
    if len(ids) < 2:
        return []
    partitions: set[frozenset[str]] = set()
    for bid in ids:
        partitions.add(frozenset((bid,)))
    if len(route.zones) == 2:
        partitions.add(frozenset(bid for bid in ids
                                 if str(ctx.boxes[bid]["zone_id"]) == route.zones[0]))
    routes: list[tuple[Any, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for first_ids in partitions:
        if not first_ids or len(first_ids) == len(ids):
            continue
        second_ids = set(ids) - first_ids
        halves = []
        for half_ids in (first_ids, second_ids):
            visits = [(zone, [bid for bid in ids
                              if bid in half_ids and
                              str(ctx.boxes[bid]["zone_id"]) == zone])
                      for zone in route.zones]
            halves.append(ctx.route(visits))
        key = tuple(sorted((halves[0].visits, halves[1].visits), key=str))
        if key not in seen:
            seen.add(key)
            routes.append((halves[0], halves[1]))
    return routes


def _estimated_lateness(ctx: Context, route: Any, option: dict[str, Any],
                        start: int) -> float:
    total = 0.0
    for bid in route.box_ids:
        box = ctx.boxes[bid]
        if not math.isnan(float(box["hard_deadline_s"])):
            continue
        total += float(box["priority"]) * max(
            0.0, start + float(option["completion_offsets"][bid]) -
            float(box["expected_s"]))
    return total


def _split_options(ctx: Context, incumbent: list[base.JointChoice],
                   starts: dict[tuple[Any, str], int],
                   reference: dict[str, float], maximum: int,
                   deadline: float) -> list[tuple[tuple[int, int], Any, str,
                                                  dict[str, Any], float]]:
    """先用物理可飞与粗分数筛选，再由逐段通信和 CP-SAT 严格认证。"""
    groups = []
    for i, choice in enumerate(incumbent):
        if time.monotonic() >= deadline:
            break
        start = starts[(choice.task.route.visits, choice.kind)]
        old_late = _estimated_lateness(ctx, choice.task.route, choice.option, start)
        old_energy = float(choice.option["energy_kwh"])
        best_group = None
        for first, second in _partition_routes(ctx, choice):
            children = []
            for route in (first, second):
                valid = [(kind, option) for kind, option in ctx.options(route).items()
                         if base._latest_hard_start(ctx, route, option) >= -base.EPS]
                if not valid:
                    break
                kind, option = min(valid, key=lambda item: (
                    _estimated_lateness(ctx, route, item[1], start) /
                    max(reference["soft_weighted_lateness_s"], 1.0) +
                    float(item[1]["energy_kwh"]) /
                    max(reference["total_energy_kwh"], 1.0), item[0]))
                children.append((route, kind, option))
            if len(children) != 2:
                continue
            late = sum(_estimated_lateness(ctx, route, option, start)
                       for route, _, option in children)
            energy = sum(float(option["energy_kwh"])
                         for _, _, option in children)
            gain = ((old_late - late) /
                    max(reference["soft_weighted_lateness_s"], 1.0) +
                    (old_energy - energy) /
                    max(reference["total_energy_kwh"], 1.0) -
                    1.0 / reference["total_sorties"]) / 4.0
            if best_group is None or gain > best_group[0]:
                best_group = gain, children
        if best_group is not None:
            groups.append((best_group[0], i, best_group[1]))
    groups.sort(key=lambda item: (-item[0], item[1]))
    result = []
    for gain, i, children in groups[:max(1, maximum // 2)]:
        result.extend(((i, i), route, kind, option, gain)
                      for route, kind, option in children)
    return result[:maximum]


def _attempt(label: str, ctx: Context, relay: dict[str, Any],
             sites: list[base.RelayCandidate], pool: list[base.JointChoice],
             hints: dict[str, Any], fixed: dict[tuple[Any, str],
             tuple[int, str, str]] | None, args: argparse.Namespace,
             data_run: Path, links: base.LinkEvaluator, dem: base.DemGrid,
             reference: dict[str, float], seconds: float,
             feasibility_only: bool = False,
             fixed_relays: dict[int, tuple[int, int, int]] | None = None
             ) -> tuple[dict[str, Any] | None, base.JointState | None, dict[str, Any]]:
    print(f"Q3 等权 {label}：{len(pool)} 条候选，预算 {seconds:.0f}s", flush=True)
    solution, report = base._solve_joint(
        ctx, relay, sites, pool, args.horizon_s, args.relay_slots, seconds,
        args.workers, False, hints=hints, fixed_partial_transport=fixed,
        fixed_relay_schedule=fixed_relays, feasibility_only=feasibility_only,
        objective_mode="equal_normalized", objective_reference=reference,
        explicit_relay_uavs=True)
    if solution is None:
        return None, None, {**report, "stage": label, "validation": "NO_SOLUTION"}
    state, validation = senior._inspect_solution(solution, ctx, relay, sites,
                                                  data_run, links, dem)
    metrics = senior._metrics(state)
    return solution, state, {**report, "stage": label,
                             "validation": validation["status"],
                             "metrics": metrics,
                             "equal_weight_score": _score(metrics, reference)}


def _write_result(args: argparse.Namespace, origin: Path,
                  original_summary: dict[str, Any], data_run: Path,
                  best_state: base.JointState | None,
                  best_metrics: dict[str, float | int],
                  reference: dict[str, float], attempts: list[dict[str, Any]],
                  relay: dict[str, Any], links: base.LinkEvaluator,
                  dem: base.DemGrid) -> dict[str, Any]:
    stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    table = args.output_root / f"q3_scheme2_equal_{stamp}_table"
    figure = args.output_root / f"q3_scheme2_equal_{stamp}_figure"
    table.mkdir(parents=True, exist_ok=False)
    if best_state is None:
        skip = {"Q3_运行摘要.json", "Q3_方案比较.csv", "Q3_已验收候选比较.csv",
                "Q3_方案二搜索轨迹.json", "Q3_验收检查.csv", "Q3_通信问题.json",
                "Q3_逐区间验收.csv"}
        for item in origin.iterdir():
            if item.is_file() and item.suffix in {".csv", ".json"} and item.name not in skip:
                shutil.copy2(item, table / item.name)
    else:
        base._save_joint_tables(table, best_state, relay)
    validation = base.validate_official_tables(table, data_run, links, dem)
    base._save_verification(table, validation, scheme_check_id="SCHEME2_EQUAL_STATUS")
    score = _score(best_metrics, reference)
    summary = {
        "status": validation["status"],
        "method": "scheme 2 incumbent + split/rebatch + equal-weight joint CP-SAT",
        "source_scheme2_table": str(origin),
        "source": {**original_summary["source"], "data_run": str(data_run)},
        "baseline_metrics": original_summary["final_metrics"],
        "final_metrics": best_metrics,
        "weights": {key: 0.25 for key in base.OBJECTIVE_KEYS},
        "baseline_score": 1.0,
        "final_score": score,
        "improved_from_scheme2": best_state is not None,
        "global_optimality_proven": False,
        "attempts": attempts,
        "table_dir": str(table),
        "figure_dir": None,
        "transport_status": validation["transport_status"],
        "communication_status": validation["communication"]["status"],
        "relay_resource_status": validation["relay_resources"]["status"],
    }
    rows = [{"方案": "师兄方案二已验收起点", **original_summary["final_metrics"],
             "F": 1.0},
            {"方案": "等权优化输出", **best_metrics, "F": score}]
    base.write_rows(table / "Q3_等权方案比较.csv", rows,
                    ["方案", "weighted_lateness", "makespan_s", "energy_kwh",
                     "transport_sorties", "relay_sorties", "F"])
    base.json_dump(table / "Q3_等权搜索轨迹.json", attempts)
    if validation["status"] == "PASS":
        if best_state is not None:
            base._save_figures(figure, best_state.slices, best_state.assigned,
                               best_state.missions, scheme_label="Q3 scheme 2 equal")
        else:
            old_figure = origin.with_name(origin.name.replace("_table", "_figure"))
            if old_figure.is_dir():
                shutil.copytree(old_figure, figure)
            else:
                figure.mkdir()
        (table / "Q3_READY.txt").write_text(
            "Q3 scheme 2 equal independently verified PASS\n"
            f"source_manifest_sha256={summary['source']['source_manifest_sha256']}\n",
            encoding="utf-8")
        summary["figure_dir"] = str(figure)
    base.json_dump(table / "Q3_运行摘要.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    origin = (args.scheme2_table or _latest_scheme2(
        CODE_DIR / "3_outputs" / "2_optimize")).resolve()
    if not (origin / "Q3_READY.txt").is_file():
        raise ValueError("方案二起点没有 Q3_READY.txt")
    original_summary = json.loads((origin / "Q3_运行摘要.json").read_text(encoding="utf-8"))
    if original_summary.get("status") != "PASS":
        raise ValueError("方案二起点不是 PASS")
    data_run = (args.data_run or _default_data_run()).resolve()
    relay, dem, links, meta = base._scene(data_run)
    if meta["source_manifest_sha256"] != original_summary["source"]["source_manifest_sha256"]:
        raise ValueError("方案二和当前数据清单哈希不一致")
    initial_check = base.validate_official_tables(origin, data_run, links, dem)
    if initial_check["status"] != "PASS":
        raise ValueError("方案二起点本机独立复验未通过")
    ctx = Context(data_run)
    positions = base._node_positions(data_run)
    relay_rows = base.read_rows(origin / "Q3_中继架次.csv")
    sites = senior._load_sites(origin, relay_rows, data_run, relay, dem, links,
                               ctx, positions, args.sample_step_s,
                               args.candidate_spacing_m, args.relay_sites)
    access_cache: dict[tuple[Any, ...], bool] = {}
    incumbent, initial_hints = senior._incumbent_choices(
        ctx, origin, positions, links, args.sample_step_s, sites, access_cache)
    reference = _reference(original_summary["final_metrics"])
    best_metrics = dict(original_summary["final_metrics"])
    best_score = 1.0
    best_state = None
    best_solution = None
    search_solution = None
    attempts: list[dict[str, Any]] = []
    started = time.monotonic()

    # 原解先映射到可继续优化的整数模型。映射失败时仍保留原始 PASS 表。
    solution, state, record = _attempt(
        "原解映射", ctx, relay, sites, incumbent, initial_hints,
        initial_hints["fixed_transport"], args, data_run, links, dem,
        reference, min(args.mapping_seconds, args.time_budget_s),
        feasibility_only=True,
        fixed_relays={slot: (site, ready, end)
                      for slot, site, _, _, ready, end in initial_hints["relays"]})
    attempts.append(record)
    if solution is not None and record["validation"] == "PASS":
        search_solution = solution
        measured = senior._metrics(state)
        if _is_improvement(measured, best_score, reference):
            best_solution, best_state, best_metrics = solution, state, measured
            best_score = _score(measured, reference)

    remaining = args.time_budget_s - (time.monotonic() - started)
    if search_solution is not None and remaining >= 5:
        solution, state, record = _attempt(
            "固定批次整体重排", ctx, relay, sites, incumbent,
            senior._solution_hints(search_solution), None,
            args, data_run, links, dem, reference,
            min(args.global_seconds, remaining))
        attempts.append(record)
        if state is not None and record["validation"] == "PASS":
            search_solution = solution
            measured = senior._metrics(state)
            if _is_improvement(measured, best_score, reference):
                best_solution, best_state, best_metrics = solution, state, measured
                best_score = _score(measured, reference)
                print(f"Q3 等权：整体重排接受 F={best_score:.6f}", flush=True)

    deadline = started + args.time_budget_s
    split = _split_options(ctx, incumbent, initial_hints["routes"], reference,
                           args.max_split_routes, deadline)
    other = senior._local_route_options(ctx, incumbent, args.max_new_routes,
                                        deadline) if time.monotonic() < deadline else []
    proposed = [*split, *other]
    print(f"Q3 等权：{len(split)} 条拆分子路线、{len(other)} 条其他重组路线", flush=True)
    certified = senior._certify_rebatch(ctx, proposed, positions, links,
                                        args.sample_step_s, sites, access_cache,
                                        deadline)
    by_pair: dict[tuple[int, int], list[base.JointChoice]] = {}
    rank: dict[tuple[int, int], float] = {}
    for pair, choice, gain in certified:
        by_pair.setdefault(pair, []).append(choice)
        rank[pair] = max(rank.get(pair, -math.inf), gain)
    # 拆分必须有两条子路线共同覆盖原批次；单条通过通信认证不能构成新方案。
    for pair in list(by_pair):
        if pair[0] != pair[1]:
            continue
        wanted = set(incumbent[pair[0]].task.route.box_ids)
        alternatives = by_pair[pair]
        complete = any(
            set(left.task.route.box_ids).isdisjoint(right.task.route.box_ids)
            and set(left.task.route.box_ids) | set(right.task.route.box_ids) == wanted
            for i, left in enumerate(alternatives)
            for right in alternatives[i + 1:])
        if not complete:
            del by_pair[pair]
    pair_order = sorted(by_pair, key=lambda pair: (-rank[pair], pair))
    for number, pair in enumerate(pair_order[:args.max_neighborhoods], 1):
        remaining = deadline - time.monotonic()
        if remaining < 5:
            break
        current = best_solution or search_solution
        if current is None:
            break
        chosen = {(current["pool"][i].task.route.visits,
                   current["pool"][i].kind) for i, *_ in current["routes"]}
        released = set(pair)
        originals = {(incumbent[i].task.route.visits, incumbent[i].kind)
                     for i in released}
        if not originals <= chosen:
            continue
        fixed = {(current["pool"][i].task.route.visits,
                  current["pool"][i].kind): (start, uav, battery)
                 for i, start, uav, battery in current["routes"]
                 if (current["pool"][i].task.route.visits,
                     current["pool"][i].kind) not in originals}
        pool = [current["pool"][i] for i, *_ in current["routes"]]
        known = {(item.task.route.visits, item.kind) for item in pool}
        for choice in by_pair[pair]:
            key = (choice.task.route.visits, choice.kind)
            if key not in known:
                pool.append(choice)
                known.add(key)
        solution, state, record = _attempt(
            f"邻域{number}/{pair}", ctx, relay, sites, pool,
            senior._solution_hints(current), fixed, args, data_run, links,
            dem, reference, min(args.local_seconds, remaining))
        attempts.append(record)
        if state is not None and record["validation"] == "PASS":
            measured = senior._metrics(state)
            if _is_improvement(measured, best_score, reference):
                best_solution, best_state, best_metrics = solution, state, measured
                best_score = _score(measured, reference)
                print(f"Q3 等权：邻域{number} 接受 F={best_score:.6f}，"
                      f"运输架次={measured['transport_sorties']}", flush=True)
    return _write_result(args, origin, original_summary, data_run, best_state,
                         best_metrics, reference, attempts, relay, links, dem)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme2-table", type=Path, default=None)
    parser.add_argument("--data-run", type=Path, default=None)
    parser.add_argument("--output-root", type=Path,
                        default=CODE_DIR / "3_outputs" / "3_scheme2_equal")
    parser.add_argument("--time-budget-s", type=float, default=600)
    parser.add_argument("--mapping-seconds", type=float, default=30)
    parser.add_argument("--global-seconds", type=float, default=80)
    parser.add_argument("--local-seconds", type=float, default=18)
    parser.add_argument("--max-neighborhoods", type=int, default=20)
    parser.add_argument("--max-split-routes", type=int, default=40)
    parser.add_argument("--max-new-routes", type=int, default=30)
    parser.add_argument("--relay-slots", type=int, default=18)
    parser.add_argument("--relay-sites", type=int, default=72)
    parser.add_argument("--horizon-s", type=int, default=21600)
    parser.add_argument("--sample-step-s", type=float, default=20)
    parser.add_argument("--candidate-spacing-m", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if any(getattr(args, name) <= 0 for name in (
            "time_budget_s", "mapping_seconds", "global_seconds", "local_seconds",
            "max_neighborhoods", "max_split_routes", "max_new_routes",
            "relay_slots", "relay_sites", "horizon_s", "sample_step_s",
            "candidate_spacing_m", "workers")):
        parser.error("预算、候选数和时域必须为正")
    summary = run(args)
    print(json.dumps({key: summary[key] for key in
                      ("status", "improved_from_scheme2", "baseline_metrics",
                       "final_metrics", "final_score", "table_dir")},
                     ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
