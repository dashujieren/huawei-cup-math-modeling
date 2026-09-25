"""问题三方案 2：从已验收方案出发的运输—中继联合大邻域改进。

运行后先写 3_outputs/2_optimize/*_table，再生成同批次 *_figure。
每个候选均须经过独立的运输、资源和连续通信验收；未改进时保留方案一。
有限候选、时间预算和 CP-SAT 状态均不构成全局最优证明。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

import q3_1_optimize as base
from q2_1_optimize import Context


CODE_DIR = Path(__file__).resolve().parent


def _latest_passed_run(root: Path) -> Path:
    runs = sorted(root.glob("q3_opt1_*_table"), reverse=True)
    for path in runs:
        if (path / "Q3_READY.txt").is_file():
            summary = json.loads((path / "Q3_运行摘要.json").read_text(encoding="utf-8"))
            if summary.get("status") == "PASS":
                return path
    raise FileNotFoundError("没有找到已验收的方案一输出；请用 --scheme1-table 指定目录")


def _same_site(site: base.RelayCandidate, row: dict[str, str]) -> bool:
    return (abs(site.lon_deg - float(row["悬停经度（°）"])) < 2e-7
            and abs(site.lat_deg - float(row["悬停纬度（°）"])) < 2e-7
            and abs(site.alt_m - float(row["悬停海拔（m）"])) < 2e-4)


def _load_sites(table_dir: Path, relay_rows: list[dict[str, str]], data_run: Path,
                relay: dict[str, Any], dem: base.DemGrid,
                links: base.LinkEvaluator, ctx: Context,
                positions: dict[str, tuple[float, float, float]],
                step_s: float, spacing_m: int, site_limit: int) -> list[base.RelayCandidate]:
    """复用完整匹配旧输出的站点几何；缓存仅作候选，结果仍重新验收。"""
    cache_root = Path(tempfile.gettempdir()) / "huawei_q3_geometry"
    for path in sorted(cache_root.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            sites = [base.RelayCandidate(**row) for row in record["sites"]]
            if (len(sites) <= site_limit and all(
                    any(_same_site(site, row) for site in sites)
                    for row in relay_rows)):
                print(f"Q3 方案二：复用 {len(sites)} 个已知站点；逐段通信仍重新认证", flush=True)
                return sites
        except (OSError, ValueError, TypeError, KeyError):
            continue
    print("Q3 方案二：未找到覆盖原方案的几何缓存，重新生成候选站点……", flush=True)
    route_sets = base._raw_route_sets(ctx)
    source_slices = base._source_slices_from_raw(ctx, positions, links, step_s, route_sets)
    all_sites, _ = base.generate_candidates(
        dem, links, source_slices, relay, positions["O01"],
        spacing_m=spacing_m, max_points=12000)
    if not all_sites:
        raise ValueError("没有物理可用的中继站点")
    selected = base._shortlist_sites(source_slices, all_sites, links,
                                    site_limit, {})
    for row in relay_rows:
        match = next((site for site in all_sites if _same_site(site, row)), None)
        if match is None:
            raise ValueError("新生成的站点不能重建方案一中继点，请恢复同一数据版本的缓存")
        if not any(site.candidate_id == match.candidate_id for site in selected):
            selected.append(match)
    return selected


def _incumbent_choices(ctx: Context, table_dir: Path,
                       positions: dict[str, tuple[float, float, float]],
                       links: base.LinkEvaluator, step_s: float,
                       sites: list[base.RelayCandidate],
                       access_cache: dict[tuple[Any, ...], bool]) -> tuple[
                           list[base.JointChoice], dict[str, Any]]:
    tree = cKDTree(np.asarray([site.hover for site in sites], dtype=float))
    rows = base.read_rows(table_dir / "Q3_运输明细.csv")
    relay_rows = base.read_rows(table_dir / "Q3_中继架次.csv")
    choices: list[base.JointChoice] = []
    route_hints: dict[tuple[Any, str], int] = {}
    transport_fixed: dict[tuple[Any, str], tuple[int, str, str]] = {}
    for row in rows:
        box_ids = row["box_ids"].split(";")
        zones = row["visit_order"].split(">")
        route = ctx.route([(zone, [bid for bid in box_ids
                                   if str(ctx.boxes[bid]["zone_id"]) == zone])
                           for zone in zones])
        kind = row["type_id"]
        option = ctx.evaluate(route, kind)
        if option is None:
            raise ValueError(f"方案一的 {row['batch_id']} 在现有数据下不可飞")
        task = base.RouteTask(row["batch_id"], route)
        profile = base._route_profile(ctx, task, kind, option, positions,
                                      links, step_s, sites, tree, access_cache)
        blocks = base._blind_blocks(profile, sites, split_on_site_change=True)
        if profile.uncovered_slices or blocks is None:
            raise ValueError(f"方案一的 {row['batch_id']} 无法用当前站点重建通信")
        choices.append(base.JointChoice(task, kind, option, profile, blocks))
        route_hints[(route.visits, kind)] = round(float(row["start_s"]))
        transport_fixed[(route.visits, kind)] = (
            round(float(row["start_s"])), row["uav_id"], row["battery_id"])
    covered = [bid for c in choices for bid in c.task.route.box_ids]
    if len(covered) != len(ctx.boxes) or set(covered) != set(ctx.boxes):
        raise ValueError("方案一运输表没有完整且唯一覆盖原始货箱")
    relays = []
    for slot, row in enumerate(relay_rows):
        site_index = next((j for j, site in enumerate(sites)
                           if _same_site(site, row)), None)
        if site_index is None:
            raise ValueError(f"方案一中继任务 {row['中继架次编号']} 的站点缺失")
        relays.append((slot, site_index, row["中继无人机编号"],
                       row["能源组件编号"], round(float(row["建链完成时刻（s）"])),
                       round(float(row["服务结束时刻（s）"]))))
    return choices, {"routes": route_hints, "relays": relays,
                     "fixed_transport": transport_fixed}


def _metrics(state: base.JointState) -> dict[str, float | int]:
    return {
        "weighted_lateness": sum(float(d["priority"]) * float(d["soft_lateness_s"])
                                 for d in state.deliveries if d["hard_deadline_s"] is None),
        "makespan_s": max([float(s["return_s"]) for s in state.sorties] +
                          [float(m.return_s) for m in state.missions]),
        "energy_kwh": sum(float(s["energy_kwh"]) for s in state.sorties) +
                      sum(float(m.energy_kwh) for m in state.missions),
        "transport_sorties": len(state.sorties),
        "relay_sorties": len(state.missions),
    }


def _base_metrics(summary: dict[str, Any]) -> dict[str, float | int]:
    return {
        "weighted_lateness": float(summary["soft_weighted_lateness_s"]),
        "makespan_s": float(summary["joint_makespan_s"]),
        "energy_kwh": float(summary["total_energy_kwh"]),
        "transport_sorties": int(summary["transport_sorties"]),
        "relay_sorties": int(summary["relay_sorties"]),
    }


def _dominates(new: dict[str, float | int], old: dict[str, float | int]) -> bool:
    tolerances = {"weighted_lateness": 1e-3, "makespan_s": 1e-4,
                  "energy_kwh": 1e-6, "transport_sorties": 0,
                  "relay_sorties": 0}
    return (all(new[key] <= old[key] + tolerances[key] for key in tolerances)
            and any(new[key] < old[key] - tolerances[key] for key in tolerances))


def _limits(best: dict[str, float | int], target: str) -> dict[str, int]:
    """CP 的保守秒级取整给小幅容差，最终仍用连续值严格接受。"""
    limits = {
        "weighted_lateness": math.ceil(float(best["weighted_lateness"]) + 5000),
        "makespan_s": math.ceil(float(best["makespan_s"]) + 2),
        "energy_j": math.ceil(float(best["energy_kwh"]) * 3_600_000 + 5000),
        "transport_sorties": int(best["transport_sorties"]),
        "relay_sorties": int(best["relay_sorties"]),
    }
    limits.pop(target, None)
    return limits


def _inspect_solution(solution: dict[str, Any], ctx: Context,
                      relay: dict[str, Any], sites: list[base.RelayCandidate],
                      data_run: Path, links: base.LinkEvaluator,
                      dem: base.DemGrid) -> tuple[base.JointState, dict[str, Any]]:
    state = base._materialize_solution(ctx, relay, sites, solution)
    with tempfile.TemporaryDirectory(prefix="q3_scheme2_verify_") as tmp:
        table = Path(tmp)
        base._save_joint_tables(table, state, relay)
        report = base.validate_official_tables(table, data_run, links, dem)
    return state, report


def _merged_routes(ctx: Context, first: base.JointChoice,
                   second: base.JointChoice) -> list[Any]:
    ids = list(dict.fromkeys([*first.task.route.box_ids, *second.task.route.box_ids]))
    by_zone: dict[str, list[str]] = {}
    for bid in ids:
        by_zone.setdefault(str(ctx.boxes[bid]["zone_id"]), []).append(bid)
    if len(by_zone) > 2:
        return []
    zones = list(by_zone)
    return [ctx.route([(zone, by_zone[zone]) for zone in ordering])
            for ordering in itertools.permutations(zones)]


def _local_route_options(ctx: Context, incumbent: list[base.JointChoice],
                         maximum: int, deadline: float) -> list[tuple[tuple[int, int], Any, str,
                                                      dict[str, Any], float]]:
    """按真实运输能耗和当前迟到瓶颈筛选待做完整通信认证的两架次重组。"""
    scored = []
    moved = []
    for i, j in itertools.combinations(range(len(incumbent)), 2):
        if time.monotonic() >= deadline:
            break
        a, b = incumbent[i], incumbent[j]
        old_energy = float(a.option["energy_kwh"]) + float(b.option["energy_kwh"])
        for route in _merged_routes(ctx, a, b):
            for kind, option in ctx.options(route).items():
                if base._latest_hard_start(ctx, route, option) < -base.EPS:
                    continue
                saving = old_energy - float(option["energy_kwh"])
                if saving > 1e-5:
                    scored.append(((i, j), route, kind, option, saving))
        # 同区不同批次之间交换 1 箱，特别允许硬箱与软箱混装。
        if len(a.task.route.zones) == len(b.task.route.zones) == 1 and (
                a.task.route.zones[0] == b.task.route.zones[0]):
            zone = a.task.route.zones[0]
            for donor, receiver in ((a, b), (b, a)):
                donor_ids = list(donor.task.route.box_ids)
                receiver_ids = list(receiver.task.route.box_ids)
                if len(donor_ids) < 2:
                    continue
                for bid in donor_ids:
                    first = ctx.route([(zone, [x for x in donor_ids if x != bid])])
                    second = ctx.route([(zone, [*receiver_ids, bid])])
                    if not ctx.options(first) or not ctx.options(second):
                        continue
                    best_pair_energy = (min(float(x["energy_kwh"])
                                            for x in ctx.options(first).values()) +
                                        min(float(x["energy_kwh"])
                                            for x in ctx.options(second).values()))
                    saving = old_energy - best_pair_energy
                    for route in (first, second):
                        for kind, option in ctx.options(route).items():
                            if base._latest_hard_start(ctx, route, option) >= -base.EPS:
                                moved.append(((i, j), route, kind, option, saving))
    scored.sort(key=lambda item: (-item[4], item[0], item[2]))
    moved.sort(key=lambda item: (-item[4], item[0], item[2]))
    selected = []
    pair_count: dict[tuple[int, int], int] = {}
    seen = set()
    for item in [*scored[:max(1, maximum * 2 // 3)],
                 *moved[:max(1, maximum // 3)]]:
        pair, route, kind, _, _ = item
        key = (route.visits, kind)
        if key in seen or pair_count.get(pair, 0) >= 6:
            continue
        seen.add(key)
        pair_count[pair] = pair_count.get(pair, 0) + 1
        selected.append(item)
    return selected[:maximum]


def _certify_rebatch(ctx: Context, options: list[tuple[Any, ...]],
                     positions: dict[str, tuple[float, float, float]],
                     links: base.LinkEvaluator, step_s: float,
                     sites: list[base.RelayCandidate],
                     access_cache: dict[tuple[Any, ...], bool],
                     deadline: float) -> list[tuple[tuple[int, int], base.JointChoice, float]]:
    tree = cKDTree(np.asarray([site.hover for site in sites], dtype=float))
    result = []
    progress = base.ProgressBar("方案二新路线通信认证", len(options))
    for n, (pair, route, kind, option, saving) in enumerate(options):
        if time.monotonic() >= deadline:
            break
        progress.update(n)
        task = base.RouteTask(f"N{n+1:03d}", route)
        profile = base._route_profile(ctx, task, kind, option, positions,
                                      links, step_s, sites, tree, access_cache)
        blocks = base._blind_blocks(profile, sites, split_on_site_change=True)
        if not profile.uncovered_slices and blocks is not None:
            result.append((pair, base.JointChoice(task, kind, option, profile, blocks),
                           saving))
    progress.close(completed=False)
    return result


def _solution_hints(solution: dict[str, Any]) -> dict[str, Any]:
    return {"routes": solution["route_hints"], "relays": solution["relays"]}


def _attempt(label: str, ctx: Context, relay: dict[str, Any],
             sites: list[base.RelayCandidate], pool: list[base.JointChoice],
             hints: dict[str, Any], fixed: dict[tuple[Any, str], tuple[int, str, str]] | None,
             objective: str, limits: dict[str, int], args: argparse.Namespace,
             data_run: Path, links: base.LinkEvaluator, dem: base.DemGrid,
             seconds: float, feasibility_only: bool = False,
             fixed_relays: dict[int, tuple[int, int, int]] | None = None) -> tuple[
                                      dict[str, Any] | None, base.JointState | None,
                                      dict[str, Any]]:
    print(f"Q3 方案二 {label}：{len(pool)} 个路线/机型候选，"
          f"目标={objective}，预算={seconds:.0f}s", flush=True)
    solution, report = base._solve_joint(
        ctx, relay, sites, pool, args.horizon_s, args.relay_slots, seconds,
        args.workers, False, hints=hints, fixed_partial_transport=fixed,
        fixed_relay_schedule=fixed_relays,
        feasibility_only=feasibility_only,
        objective_mode=("balanced" if feasibility_only else f"scheme2_{objective}"),
        metric_limits=limits,
        explicit_relay_uavs=True)
    if solution is None:
        return None, None, {"stage": label, **report, "validation": "NO_SOLUTION"}
    state, validation = _inspect_solution(solution, ctx, relay, sites,
                                           data_run, links, dem)
    return solution, state, {"stage": label, **report,
                             "validation": validation["status"],
                             "metrics": _metrics(state)}


def _write_result(args: argparse.Namespace, origin: Path,
                  origin_summary: dict[str, Any], best_state: base.JointState | None,
                  relay: dict[str, Any], data_run: Path, links: base.LinkEvaluator,
                  dem: base.DemGrid, best_metrics: dict[str, float | int],
                  attempts: list[dict[str, Any]]) -> dict[str, Any]:
    stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    table_dir = args.output_root / f"q3_opt2_{stamp}_table"
    figure_dir = args.output_root / f"q3_opt2_{stamp}_figure"
    table_dir.mkdir(parents=True, exist_ok=False)
    if best_state is None:
        for item in origin.iterdir():
            if item.is_file() and item.suffix in {".csv", ".json"}:
                shutil.copy2(item, table_dir / item.name)
    else:
        base._save_joint_tables(table_dir, best_state, relay)
    validation = base.validate_official_tables(table_dir, data_run, links, dem)
    base._save_verification(table_dir, validation,
                            scheme_check_id="SCHEME2_STATUS")
    summary = {
        "status": validation["status"],
        "method": "Q3 scheme 2: validated-incumbent joint large-neighborhood improvement",
        "interpretation": ("已发现并独立验收不劣于方案一的改进" if best_state is not None
                           else "在当前预算内未取得严格改进；保留原方案一可行解"),
        "source_scheme1_table": str(origin.resolve()),
        "source": origin_summary["source"],
        "global_optimality_proven": False,
        "improved_from_scheme1": best_state is not None,
        "baseline_metrics": _base_metrics(origin_summary),
        "final_metrics": best_metrics,
        "attempts": attempts,
        "table_dir": str(table_dir.resolve()),
        "figure_dir": None,
        "transport_status": validation["transport_status"],
        "communication_status": validation["communication"]["status"],
        "relay_resource_status": validation["relay_resources"]["status"],
    }
    comparison = [{"方案": "方案一已验收起点", **_base_metrics(origin_summary)}]
    global_candidates = [item for item in attempts
                         if item.get("stage", "").startswith("整体重排/")
                         and item.get("validation") == "PASS"]
    if global_candidates:
        chosen = min(global_candidates,
                     key=lambda item: (item["metrics"]["weighted_lateness"],
                                       item["metrics"]["makespan_s"],
                                       item["metrics"]["energy_kwh"]))
        comparison.append({"方案": "固定批次整体重排", **chosen["metrics"]})
    comparison.append({"方案": "方案二最终可行输出", **best_metrics})
    base.write_rows(table_dir / "Q3_方案比较.csv", comparison,
                    ["方案", "weighted_lateness", "makespan_s", "energy_kwh",
                     "transport_sorties", "relay_sorties"])
    candidate_rows = []
    reference = _base_metrics(origin_summary)
    for item in attempts:
        if item.get("validation") != "PASS":
            continue
        metrics = item["metrics"]
        candidate_rows.append({
            "阶段": item["stage"], "求解状态": item["solver_status"],
            "严格支配方案一": _dominates(metrics, reference),
            **metrics,
            "迟到变化": float(metrics["weighted_lateness"]) - float(reference["weighted_lateness"]),
            "最晚返航变化_s": float(metrics["makespan_s"]) - float(reference["makespan_s"]),
            "能耗变化_kwh": float(metrics["energy_kwh"]) - float(reference["energy_kwh"]),
        })
    base.write_rows(table_dir / "Q3_已验收候选比较.csv", candidate_rows,
                    ["阶段", "求解状态", "严格支配方案一", "weighted_lateness",
                     "makespan_s", "energy_kwh", "transport_sorties",
                     "relay_sorties", "迟到变化", "最晚返航变化_s", "能耗变化_kwh"])
    base.json_dump(table_dir / "Q3_方案二搜索轨迹.json", attempts)
    if validation["status"] == "PASS":
        # 表格验收完毕后才创建图片目录。
        if best_state is not None:
            base._save_figures(figure_dir, best_state.slices,
                               best_state.assigned, best_state.missions,
                               scheme_label="Q3 scheme 2")
        else:
            old_figures = Path(origin_summary.get("figure_dir") or "")
            if old_figures.is_dir():
                shutil.copytree(old_figures, figure_dir)
            else:
                figure_dir.mkdir()
        (table_dir / "Q3_READY.txt").write_text(
            "Q3 scheme 2 independently verified PASS\n"
            f"source_manifest_sha256={origin_summary['source']['source_manifest_sha256']}\n",
            encoding="utf-8")
        summary["figure_dir"] = str(figure_dir.resolve())
    base.json_dump(table_dir / "Q3_运行摘要.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    origin = args.scheme1_table or _latest_passed_run(
        CODE_DIR / "3_outputs" / "1_optimize")
    origin = origin.resolve()
    if not (origin / "Q3_READY.txt").is_file():
        raise ValueError("方案一目录没有 Q3_READY.txt，不能作为已验收起点")
    origin_summary = json.loads((origin / "Q3_运行摘要.json").read_text(encoding="utf-8"))
    if origin_summary.get("status") != "PASS":
        raise ValueError("方案一起点不是 PASS")
    data_run = (args.data_run or Path(origin_summary["source"]["data_run"])).resolve()
    relay, dem, links, source_meta = base._scene(data_run)
    if source_meta["source_manifest_sha256"] != origin_summary["source"]["source_manifest_sha256"]:
        raise ValueError("方案一与本次原始数据清单不一致")
    base_check = base.validate_official_tables(origin, data_run, links, dem)
    if base_check["status"] != "PASS":
        raise ValueError(f"方案一起点重新验收失败：{base_check['status']}")
    ctx = Context(data_run)
    positions = base._node_positions(data_run)
    relay_rows = base.read_rows(origin / "Q3_中继架次.csv")
    sites = _load_sites(origin, relay_rows, data_run, relay, dem, links, ctx,
                        positions, args.sample_step_s, args.candidate_spacing_m,
                        args.relay_sites)
    access_cache: dict[tuple[Any, ...], bool] = {}
    incumbent, initial_hints = _incumbent_choices(
        ctx, origin, positions, links, args.sample_step_s, sites, access_cache)
    best_metrics = _base_metrics(origin_summary)
    best_solution: dict[str, Any] | None = None
    search_solution: dict[str, Any] | None = None
    best_state: base.JointState | None = None
    attempts: list[dict[str, Any]] = []
    started = time.monotonic()

    # 先确保旧表确实能映射回新的整数模型；否则只保留已验收的原表。
    solution, state, record = _attempt(
        "原解映射", ctx, relay, sites, incumbent, initial_hints,
        initial_hints["fixed_transport"],
        "feasibility", {}, args, data_run, links, dem,
        min(30.0, args.time_budget_s), feasibility_only=True,
        fixed_relays={slot: (site, ready, end)
                      for slot, site, _, _, ready, end in initial_hints["relays"]})
    attempts.append(record)
    if solution is not None and record["validation"] == "PASS":
        search_solution = solution
        if _dominates(_metrics(state), best_metrics):
            best_solution, best_state, best_metrics = solution, state, _metrics(state)

    # 全局重排释放所有运输与中继时刻；原表始终是单独保存的回退解。
    for objective in ("lateness", "makespan", "energy"):
        remaining = args.time_budget_s - (time.monotonic() - started)
        if remaining < 5:
            break
        solution, state, record = _attempt(
            f"整体重排/{objective}", ctx, relay, sites, incumbent,
            initial_hints if search_solution is None else _solution_hints(search_solution),
            None, objective, _limits(best_metrics, {"lateness": "weighted_lateness",
                                                   "makespan": "makespan_s",
                                                   "energy": "energy_j"}[objective]),
            args, data_run, links, dem, min(args.global_seconds, remaining))
        attempts.append(record)
        if state is not None and record["validation"] == "PASS":
            search_solution = solution
            measured = _metrics(state)
            if _dominates(measured, best_metrics):
                best_solution, best_state, best_metrics = solution, state, measured
                print(f"Q3 方案二：接受改进 {best_metrics}", flush=True)
    # 局部改变原方案的组批；固定邻域外的完整时间区间，但中继窗口联合重排。
    deadline = started + args.time_budget_s
    proposed = (_local_route_options(ctx, incumbent, args.max_new_routes, deadline)
                if time.monotonic() < deadline else [])
    print(f"Q3 方案二：{len(proposed)} 个省能的物理新路线候选", flush=True)
    rebatches = _certify_rebatch(ctx, proposed, positions, links,
                                 args.sample_step_s, sites, access_cache, deadline)
    print(f"Q3 方案二：{len(rebatches)} 个新路线通过逐段通信候选认证", flush=True)
    by_pair: dict[tuple[int, int], list[base.JointChoice]] = {}
    for pair, choice, _ in rebatches:
        by_pair.setdefault(pair, []).append(choice)
    pair_order = sorted(by_pair, key=lambda pair: -max(
        saving for p, _, saving in rebatches if p == pair))
    for round_no, pair in enumerate(pair_order[:args.max_neighborhoods], 1):
        remaining = args.time_budget_s - (time.monotonic() - started)
        if remaining < 5:
            break
        current = best_solution if best_solution is not None else search_solution
        if current is None:
            # 尚未取得可映射的整体 CP 解，跳过局部求解，保留方案一。
            attempts.append({"stage": f"邻域{round_no}",
                             "solver_status": "SKIPPED_NO_CERTIFIED_CP_INCUMBENT"})
            break
        chosen_keys = {(current["pool"][i].task.route.visits,
                        current["pool"][i].kind) for i, *_ in current["routes"]}
        released = set(pair)
        related_zones = set(incumbent[pair[0]].task.route.zones) | set(
            incumbent[pair[1]].task.route.zones)
        ranked_neighbors = sorted(
            (i for i in range(len(incumbent)) if i not in released),
            key=lambda i: (not bool(set(incumbent[i].task.route.zones) & related_zones),
                           -float(incumbent[i].option["duration_s"]), i))
        released.update(ranked_neighbors[:2])
        original = {(incumbent[i].task.route.visits, incumbent[i].kind)
                    for i in released}
        if not original <= chosen_keys:
            continue
        fixed = {(current["pool"][i].task.route.visits,
                  current["pool"][i].kind): (start, uav, battery)
                 for i, start, uav, battery in current["routes"]
                 if (current["pool"][i].task.route.visits,
                     current["pool"][i].kind) not in original}
        pool = [current["pool"][i] for i, *_ in current["routes"]]
        known = {(item.task.route.visits, item.kind) for item in pool}
        for candidate_pair, alternatives in by_pair.items():
            if not set(candidate_pair) <= released:
                continue
            for alternative in alternatives:
                key = (alternative.task.route.visits, alternative.kind)
                if key not in known:
                    pool.append(alternative)
                    known.add(key)
        objective = ("energy", "lateness", "makespan", "sorties")[round_no % 4]
        key = {"energy": "energy_j", "lateness": "weighted_lateness",
               "makespan": "makespan_s", "sorties": "transport_sorties"}[objective]
        solution, state, record = _attempt(
            f"邻域{round_no}/{pair}", ctx, relay, sites, pool,
            _solution_hints(current), fixed, objective,
            _limits(best_metrics, key), args, data_run, links, dem,
            min(args.local_seconds, remaining))
        attempts.append(record)
        if state is not None and record["validation"] == "PASS":
            measured = _metrics(state)
            if _dominates(measured, best_metrics):
                best_solution, best_state, best_metrics = solution, state, measured
                print(f"Q3 方案二：邻域{round_no} 接受改进 {best_metrics}", flush=True)
    return _write_result(args, origin, origin_summary, best_state, relay,
                         data_run, links, dem, best_metrics, attempts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme1-table", type=Path, default=None,
                        help="已通过验收的方案一 *_table 目录；默认选最新 PASS")
    parser.add_argument("--data-run", type=Path, default=None)
    parser.add_argument("--output-root", type=Path,
                        default=CODE_DIR / "3_outputs" / "2_optimize")
    parser.add_argument("--time-budget-s", type=float, default=600)
    parser.add_argument("--global-seconds", type=float, default=80)
    parser.add_argument("--local-seconds", type=float, default=18)
    parser.add_argument("--max-neighborhoods", type=int, default=16)
    parser.add_argument("--max-new-routes", type=int, default=36)
    parser.add_argument("--relay-slots", type=int, default=18)
    parser.add_argument("--relay-sites", type=int, default=72)
    parser.add_argument("--horizon-s", type=int, default=21600)
    parser.add_argument("--sample-step-s", type=float, default=20)
    parser.add_argument("--candidate-spacing-m", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if (min(args.time_budget_s, args.global_seconds, args.local_seconds,
            args.sample_step_s) <= 0 or min(args.max_neighborhoods,
            args.max_new_routes, args.relay_slots, args.relay_sites,
            args.horizon_s, args.candidate_spacing_m, args.workers) < 1):
        parser.error("预算、候选数、时域和并行数必须为正")
    summary = run(args)
    print(json.dumps({key: summary[key] for key in
                      ("status", "improved_from_scheme1", "final_metrics",
                       "table_dir", "figure_dir")}, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
