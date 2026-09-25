"""问题三：从方案三少架次 PASS 解出发，优先缩短全部飞机最晚返航时间。

运行环境：code/.venv；默认读取本 code 目录中的方案三和清洗数据。
搜索是限时、有限路线/站点候选的联合优化；只有独立验收 PASS 才写 READY。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import q3_1_optimize as base
import q3_2_optimize as previous
import q3_3_optimize as scheme3
from q2_1_optimize import Context


CODE_DIR = Path(__file__).resolve().parent
SOFT_GRADES = {"G0": 1.0, "G1": 1.05, "G2": 1.10}
MILESTONES_S = (10800, 9000, 8100, 7200)


def _latest_scheme3(root: Path) -> Path:
    for path in sorted(root.glob("q3_opt3_*_sorties_table"), reverse=True):
        summary_file = path / "Q3_运行摘要.json"
        if not (path / "Q3_READY.txt").is_file() or not summary_file.is_file():
            continue
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary.get("status") == "PASS" and summary.get("goal") == "sorties":
            return path
    raise FileNotFoundError("没有找到方案三少架次 PASS 表；请使用 --scheme3-table 指定")


def _data_run(origin: Path, summary: dict[str, Any], supplied: Path | None) -> Path:
    if supplied is not None:
        candidates = [supplied]
    else:
        recorded = Path(summary["source"]["data_run"])
        candidates = [recorded, CODE_DIR / "0_outputs" / recorded.name]
        candidates += sorted((CODE_DIR / "0_outputs").glob("run_*"), reverse=True)
    for candidate in candidates:
        if (candidate / "meta" / "source_manifest.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError("找不到对应的清洗数据；请提供 --data-run")


def _sites_from_current_data(origin: Path, relay: dict[str, Any],
                             dem: base.DemGrid, links: base.LinkEvaluator,
                             home: tuple[float, float, float],
                             site_limit: int,
                             warm_table: Path | None = None) -> list[base.RelayCandidate]:
    """从几何缓存取坐标，在当前 DEM/通信口径下重新生成物理站点。"""
    relay_rows = base.read_rows(origin / "Q3_中继架次.csv")
    if warm_table is not None:
        relay_rows += base.read_rows(warm_table / "Q3_中继架次.csv")
    cached = []
    cache_root = Path(tempfile.gettempdir()) / "huawei_q3_geometry"
    for path in sorted(cache_root.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            candidate = [base.RelayCandidate(**row) for row in record["sites"]]
            if len(candidate) > len(cached):
                cached = candidate
        except (OSError, KeyError, TypeError, ValueError):
            continue
    xy = {(site.x_m, site.y_m) for site in cached}
    for row in relay_rows:
        x, y = dem.to_xy.transform(float(row["悬停经度（°）"]),
                                   float(row["悬停纬度（°）"]))
        xy.add((float(round(x / 500) * 500), float(round(y / 500) * 500)))
    if not xy:
        raise ValueError("方案三没有可重建的中继站点")
    generated, _ = base.generate_candidates(
        dem, links, [], relay, home, extra_xy_points=sorted(xy),
        max_points=max(len(xy), 1))
    original_ids = {site.candidate_id for site in cached}
    selected = [site for site in generated if site.candidate_id in original_ids or
                any(previous._same_site(site, row) for row in relay_rows)]
    if not all(any(previous._same_site(site, row) for site in selected)
               for row in relay_rows):
        raise ValueError("无法由当前 DEM 重建方案三中继站点，数据或采样间距可能不同")
    if len(selected) > site_limit:
        raise ValueError("--max-sites 小于方案三已有站点数")
    print(f"Q3 时间优化：重算并保留 {len(selected)} 个方案三站点", flush=True)
    return selected


def _release_tail(solution: dict[str, Any] | None,
                  current: list[base.JointChoice],
                  source_hints: dict[str, Any], count: int) -> set[int]:
    """先释放最后返航的任务，再向同架飞机/电池的前序任务回溯。"""
    jobs = []
    if solution is not None:
        for index, (pool_index, start, uav, battery) in enumerate(solution["routes"]):
            choice = solution["pool"][pool_index]
            jobs.append((index, start + float(choice.option["duration_s"]),
                         uav, battery, choice.kind))
    else:
        for index, choice in enumerate(current):
            start, uav, battery = source_hints["fixed_transport"][scheme3._route_key(choice)]
            jobs.append((index, start + float(choice.option["duration_s"]),
                         uav, battery, choice.kind))
    tail = sorted(jobs, key=lambda item: -item[1])[:2]
    tail_uavs = {item[2] for item in tail}
    tail_batteries = {item[3] for item in tail}
    ranked = sorted(jobs, key=lambda item: (
        item[0] not in {last[0] for last in tail},
        item[2] not in tail_uavs,
        item[3] not in tail_batteries,
        item[4] == "A",
        -item[1]))
    return {item[0] for item in ranked[:min(count, len(ranked))]}


def _add_fast_relay_sites(geometry: scheme3.GeometryPool,
                          state: base.JointState,
                          limit: int) -> int:
    """在最后两条中继任务附近找更靠近 O01 且能保障其通信段的站点。"""
    if limit <= 0 or len(geometry.sites) >= geometry.max_sites:
        return 0
    tail = sorted(state.missions, key=lambda mission: mission.return_s)[-2:]
    if not tail:
        return 0
    step = geometry.spacing_m
    home = geometry.positions["O01"]
    points: set[tuple[float, float]] = set()
    for mission in tail:
        x, y = mission.candidate.x_m, mission.candidate.y_m
        towards_x = step if home[0] > x else -step
        towards_y = step if home[1] > y else -step
        points.update({(x + towards_x, y), (x, y + towards_y),
                       (x + towards_x, y + towards_y),
                       (x + 2 * towards_x, y), (x, y + 2 * towards_y)})
    generated, _ = base.generate_candidates(
        geometry.dem, geometry.links, [], geometry.relay, home,
        spacing_m=step, extra_xy_points=sorted(points),
        max_points=max(1, len(points)))
    known = {site.candidate_id for site in geometry.sites}
    scores: list[tuple[float, float, base.RelayCandidate]] = []
    for site in generated:
        if site.candidate_id in known:
            continue
        best_fraction = 0.0
        for mission in tail:
            relevant = [item for index, item in enumerate(state.slices)
                        if state.assigned.get(index) == mission.mission_id and
                        not item.direct_available]
            if not relevant:
                continue
            old_travel = (mission.candidate.outward_s +
                          mission.candidate.homeward_s)
            new_travel = site.outward_s + site.homeward_s
            if new_travel >= old_travel - 15:
                continue
            covered = sum(base._access_certified(site, item, geometry.links,
                                                 geometry.access_cache)
                          for item in relevant)
            best_fraction = max(best_fraction, covered / len(relevant))
        if best_fraction >= 0.75:
            scores.append((-best_fraction,
                           site.outward_s + site.homeward_s, site))
    scores.sort(key=lambda item: (item[0], item[1], item[2].candidate_id))
    added = 0
    for _, _, site in scores:
        if added >= limit or len(geometry.sites) >= geometry.max_sites:
            break
        geometry.sites.append(site)
        known.add(site.candidate_id)
        added += 1
    geometry.stats["fast_sites"] = geometry.stats.get("fast_sites", 0) + added
    if added:
        print(f"Q3 时间优化：新增 {added} 个尾段快速中继候选点", flush=True)
    return added


def _soft_limits(reference: dict[str, float | int], grade: str,
                 best_makespan: float) -> dict[str, int]:
    factor = SOFT_GRADES[grade]
    return {
        "energy_j": math.ceil(float(reference["energy_kwh"]) * factor * 3_600_000 + 5000),
        "weighted_lateness": math.ceil(float(reference["weighted_lateness"]) * factor + 5000),
        "makespan_s": math.ceil(best_makespan + 2),
    }


def _acceptable(metrics: dict[str, float | int],
                reference: dict[str, float | int], grade: str,
                best_makespan: float) -> bool:
    factor = SOFT_GRADES[grade]
    return (float(metrics["energy_kwh"]) <=
            float(reference["energy_kwh"]) * factor + 1e-6 and
            float(metrics["weighted_lateness"]) <=
            float(reference["weighted_lateness"]) * factor + 1e-3 and
            float(metrics["makespan_s"]) < best_makespan - 1e-4)


def _save_result(args: argparse.Namespace, origin: Path,
                 source_summary: dict[str, Any], data_run: Path,
                 relay: dict[str, Any], geometry: scheme3.GeometryPool,
                 best: scheme3.Accepted, reference: dict[str, float | int],
                 history: list[dict[str, Any]], candidate_log: list[dict[str, Any]],
                 wall_s: float) -> dict[str, Any]:
    stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    prefix = f"q3_time_{stamp}"
    table = args.output_root / f"{prefix}_table"
    figures = args.output_root / f"{prefix}_figure"
    table.mkdir(parents=True, exist_ok=False)
    if best.state is None:
        for item in (best.origin or origin).iterdir():
            if item.is_file() and item.suffix.lower() in {".csv", ".json"}:
                shutil.copy2(item, table / item.name)
    else:
        base._save_joint_tables(table, best.state, relay)
    validation = base.validate_official_tables(
        table, data_run, geometry.links, geometry.dem)
    base._save_verification(table, validation, scheme_check_id="TIME_TARGET_STATUS")
    measured = (previous._metrics(best.state) if best.state is not None
                else best.metrics)
    comparison = [{"方案": "方案三少架次起点", **reference},
                  {"方案": "时间优化当前最好", **measured}]
    for row in comparison:
        row["总架次"] = int(row["transport_sorties"]) + int(row["relay_sorties"])
        row["距两小时（s）"] = max(0.0, float(row["makespan_s"]) - args.target_s)
    base.write_rows(table / "Q3_时间目标比较.csv", comparison,
                    ["方案", "makespan_s", "距两小时（s）", "energy_kwh",
                     "weighted_lateness", "transport_sorties",
                     "relay_sorties", "总架次"])
    base.json_dump(table / "Q3_时间优化搜索轨迹.json", history)
    base.json_dump(table / "Q3_时间优化候选记录.json", candidate_log)
    attained = [seconds for seconds in MILESTONES_S
                if float(measured["makespan_s"]) <= seconds + 1e-4]
    result = {
        "status": validation["status"],
        "method": "Q3 scheme 3 tail-critical rebatching and joint makespan search",
        "source_scheme3_table": str(origin),
        "warm_table": str(args.warm_table.resolve()) if args.warm_table else None,
        "source_manifest_sha256": source_summary["source_manifest_sha256"],
        "source_data_run": str(data_run),
        "code_sha256": {name: base.sha256(CODE_DIR / name) for name in
                        ("q3_8_time_optimize.py", "q3_3_optimize.py",
                         "q3_2_optimize.py", "q3_1_optimize.py")},
        "reference_metrics": reference,
        "final_metrics": measured,
        "improved_from_scheme3": float(measured["makespan_s"]) <
                                  float(reference["makespan_s"]) - 1e-4,
        "target_s": args.target_s,
        "target_gap_s": max(0.0, float(measured["makespan_s"]) - args.target_s),
        "target_reached": float(measured["makespan_s"]) <= args.target_s + 1e-4,
        "milestones_reached_s": attained,
        "grade": best.grade,
        "search_stage": best.stage,
        "search_wall_s": wall_s,
        "global_optimality_proven": False,
        "candidate_sites": len(geometry.sites),
        "candidate_stats": geometry.stats,
        "table_dir": str(table.resolve()),
        "figure_dir": None,
    }
    if validation["status"] == "PASS":
        if best.state is not None:
            base._save_figures(figures, best.state.slices,
                               best.state.assigned, best.state.missions,
                               scheme_label="Q3 time priority")
        else:
            source_table = best.origin or origin
            source_figures = source_table.with_name(
                source_table.name.replace("_table", "_figure"))
            if source_figures.is_dir():
                shutil.copytree(source_figures, figures)
            else:
                figures.mkdir()
        scheme3._save_resource_figure(figures, table)
        scheme3._save_tradeoff_figure(figures, comparison)
        (table / "Q3_READY.txt").write_text(
            "Q3 time optimization independently verified PASS\n"
            f"source_manifest_sha256={result['source_manifest_sha256']}\n",
            encoding="utf-8")
        result["figure_dir"] = str(figures.resolve())
    base.json_dump(table / "Q3_运行摘要.json", result)
    base.json_dump(args.output_root / f"{prefix}_index.json", result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    origin = (args.scheme3_table or _latest_scheme3(
        CODE_DIR / "3_outputs" / "3_optimize")).resolve()
    source_summary = json.loads((origin / "Q3_运行摘要.json").read_text(encoding="utf-8"))
    if source_summary.get("status") != "PASS" or not (origin / "Q3_READY.txt").is_file():
        raise ValueError("起点必须是方案三少架次 PASS 表")
    data_run = _data_run(origin, source_summary, args.data_run)
    relay, dem, links, meta = base._scene(data_run)
    if meta["source_manifest_sha256"] != source_summary["source_manifest_sha256"]:
        raise ValueError("清洗数据与方案三源清单指纹不一致")
    source_check = base.validate_official_tables(origin, data_run, links, dem)
    if source_check["status"] != "PASS":
        raise ValueError("方案三少架次原表独立验收失败")
    reference = {key: source_summary["final_metrics"][key]
                 for key in scheme3.METRIC_KEYS}
    warm_table = args.warm_table.resolve() if args.warm_table else None
    warm_metrics = None
    if warm_table is not None:
        warm_summary = json.loads((warm_table / "Q3_运行摘要.json").read_text(
            encoding="utf-8"))
        if (warm_summary.get("status") != "PASS" or
                not (warm_table / "Q3_READY.txt").is_file() or
                warm_summary.get("source_manifest_sha256") !=
                source_summary["source_manifest_sha256"]):
            raise ValueError("续跑表必须为同一源数据下的 PASS 结果")
        warm_check = base.validate_official_tables(warm_table, data_run, links, dem)
        if warm_check["status"] != "PASS":
            raise ValueError("续跑表重新验收失败")
        warm_metrics = {key: warm_summary["final_metrics"][key]
                        for key in scheme3.METRIC_KEYS}
        if float(warm_metrics["makespan_s"]) >= float(reference["makespan_s"]):
            raise ValueError("续跑表的最晚返航应优于方案三少架次起点")
    ctx = Context(data_run)
    positions = base._node_positions(data_run)
    sites = _sites_from_current_data(origin, relay, dem, links,
                                     positions["O01"], args.max_sites,
                                     warm_table)
    geometry = scheme3.GeometryPool(
        ctx, relay, dem, links, positions, sites, args.sample_step_s,
        args.candidate_spacing_m, args.max_sites, args.max_site_points)
    incumbent, initial_hints = previous._incumbent_choices(
        ctx, warm_table or origin, positions, links, args.sample_step_s,
        sites, geometry.access_cache)
    best = (scheme3.Accepted(warm_metrics, None, None, warm_table,
                             "已验收续跑起点", "WARM")
            if warm_metrics is not None else
            scheme3.Accepted(reference, None, None, origin,
                             "方案三少架次起点", "REFERENCE"))
    accepted = [best]
    history: list[dict[str, Any]] = []
    candidate_log: list[dict[str, Any]] = []
    global_choices = {scheme3._route_key(c): c for c in incumbent}
    print(f"Q3 时间优化：起点 {best.metrics['makespan_s']:.1f}s；"
          f"目标 {args.target_s}s；搜索预算 {args.time_budget_s:.0f}s", flush=True)

    # 先映射原解；映射失败时仍保留原表，之后可用原表时间作为 CP 提示。
    fixed_relays = {slot: (site, ready, end)
                    for slot, site, _, _, ready, end in initial_hints["relays"]}
    mapping, map_report = base._solve_joint(
        ctx, relay, sites, incumbent, args.horizon_s, args.relay_slots,
        min(args.mapping_seconds, args.time_budget_s), args.workers, False,
        hints=initial_hints,
        fixed_partial_transport=initial_hints["fixed_transport"],
        fixed_relay_schedule=fixed_relays,
        feasibility_only=True, explicit_relay_uavs=True)
    map_record = {"round": 0, "search_stage": "原解映射",
                  "solver_stage": map_report["stage"],
                  **{key: value for key, value in map_report.items()
                     if key != "stage"},
                  "validation": "NO_SOLUTION"}
    if mapping is not None:
        state, check = previous._inspect_solution(
            mapping, ctx, relay, sites, data_run, links, dem)
        map_record["validation"] = check["status"]
        map_record["metrics"] = previous._metrics(state)
        if check["status"] == "PASS":
            mapped = scheme3.Accepted(map_record["metrics"], mapping, state,
                                       None, "原解映射", "REFERENCE")
            accepted.append(mapped)
            if float(mapped.metrics["makespan_s"]) < float(best.metrics["makespan_s"]) - 1e-4:
                best = mapped
    history.append(map_record)
    if mapping is not None and map_record["validation"] == "PASS":
        _add_fast_relay_sites(geometry, state, args.fast_site_additions)

    search_started = time.monotonic()
    deadline = search_started + args.time_budget_s
    try:
        for round_no in range(args.max_rounds):
            remaining = deadline - time.monotonic()
            if remaining < args.min_remaining_s:
                break
            grade = args.grades[round_no % len(args.grades)]
            eligible = [entry for entry in accepted
                        if float(entry.metrics["energy_kwh"]) <=
                        float(reference["energy_kwh"]) * SOFT_GRADES[grade] + 1e-6
                        and float(entry.metrics["weighted_lateness"]) <=
                        float(reference["weighted_lateness"]) * SOFT_GRADES[grade] + 1e-3]
            if not eligible:
                history.append({"round": round_no + 1,
                                "search_stage": "档位起点检查",
                                "grade": grade,
                                "solver_status": "SKIPPED_NO_FEASIBLE_SEED",
                                "validation": "NO_SOLUTION"})
                continue
            seed = min(eligible, key=lambda entry: (
                float(entry.metrics["makespan_s"]), entry.solution is None))
            current = scheme3._selected(seed.solution) if seed.solution else incumbent
            current = [geometry.certify(choice.task.route, choice.kind,
                                        choice.option) or choice
                       for choice in current]
            for choice in current:
                global_choices[scheme3._route_key(choice)] = choice
            full = round_no % args.full_every == args.full_every - 1
            release_count = (6 if round_no < 3 else 10 if round_no < 7 else 16)
            released = (set(range(len(current))) if full else
                        _release_tail(seed.solution, current, initial_hints, release_count))
            print(f"Q3 时间优化第{round_no+1}轮：{grade}，"
                  f"当前最好 {best.metrics['makespan_s']:.1f}s，"
                  f"释放 {len(released)} 批，剩余 {remaining:.0f}s", flush=True)
            proposed, generated = scheme3._propose_routes(
                ctx, current, released, args.max_stops, args.max_packages,
                min(deadline, time.monotonic() + args.candidate_seconds),
                rotation=round_no, priority="makespan")
            fresh, certified = scheme3._certify_packages(
                ctx, proposed, geometry, current, args.max_new_choices,
                min(deadline, time.monotonic() + args.certify_seconds),
                priority="makespan")
            # 同一货箱组增加其他可飞机型，给空闲的 A 型机参与机会。
            new_types = []
            type_deadline = min(deadline, time.monotonic() + args.type_seconds)
            for choice in current:
                if time.monotonic() >= type_deadline:
                    break
                for kind, option in ctx.options(choice.task.route).items():
                    key = choice.task.route.visits, kind
                    if key in global_choices or any(scheme3._route_key(c) == key
                                                   for c in new_types):
                        continue
                    if base._latest_hard_start(ctx, choice.task.route, option) < -base.EPS:
                        continue
                    alternate = geometry.certify(choice.task.route, kind, option)
                    if alternate is not None:
                        new_types.append(alternate)
            for choice in [*fresh, *new_types]:
                global_choices.setdefault(scheme3._route_key(choice), choice)
            protected = {scheme3._route_key(choice) for choice in incumbent}
            for entry in accepted:
                if entry.solution is not None:
                    protected.update(scheme3._route_key(c)
                                     for c in scheme3._selected(entry.solution))
            if len(global_choices) > args.max_active_choices:
                for key in list(global_choices):
                    if len(global_choices) <= args.max_active_choices:
                        break
                    if key not in protected:
                        global_choices.pop(key)
            candidate_log.append({
                "round": round_no + 1, "grade": grade, "released": len(released),
                "full": full, "generated": generated, "certified": certified,
                "new_type_options": len(new_types),
                "active_options": len(global_choices),
                "site_count": len(geometry.sites),
            })
            if full:
                pool = list(global_choices.values())
                fixed = None
            else:
                pool = list(current)
                known = {scheme3._route_key(choice) for choice in pool}
                for choice in [*fresh, *new_types]:
                    key = scheme3._route_key(choice)
                    if key not in known:
                        pool.append(choice)
                        known.add(key)
                fixed = scheme3._fixed_outside(seed.solution, released)
            hints = (previous._solution_hints(seed.solution) if seed.solution is not None
                     else initial_hints)
            seconds = min(args.global_seconds if full else args.local_seconds,
                          deadline - time.monotonic())
            if seconds < args.min_remaining_s:
                break
            limits = _soft_limits(reference, grade,
                                  float(best.metrics["makespan_s"]))
            solution, report = base._solve_joint(
                ctx, relay, geometry.sites, pool, args.horizon_s,
                args.relay_slots, seconds, args.workers, False,
                hints=hints, fixed_partial_transport=fixed,
                objective_mode="scheme2_makespan", metric_limits=limits,
                explicit_relay_uavs=True, search_seed=args.seed + round_no)
            stage = "全程" if full else "拖尾邻域"
            record = {"round": round_no + 1, "search_stage": stage,
                      "solver_stage": report["stage"],
                      "grade": grade, "metric_limits": limits,
                      "released": len(released),
                      **{key: value for key, value in report.items()
                         if key != "stage"},
                      "validation": "NO_SOLUTION"}
            if solution is not None:
                state, check = previous._inspect_solution(
                    solution, ctx, relay, geometry.sites, data_run, links, dem)
                record["validation"] = check["status"]
                metrics = previous._metrics(state)
                record["metrics"] = metrics
                if (check["status"] == "PASS" and
                        _acceptable(metrics, reference, grade,
                                    float(best.metrics["makespan_s"]))):
                    best = scheme3.Accepted(metrics, solution, state, None,
                                            record["search_stage"], grade)
                    accepted.append(best)
                    print(f"Q3 时间优化：验收 PASS，最晚返航 "
                          f"{metrics['makespan_s']:.1f}s，"
                          f"运输 {metrics['transport_sorties']} + "
                          f"中继 {metrics['relay_sorties']}", flush=True)
            history.append(record)
            if (record.get("selected_relay_sorties", 0) >= args.relay_slots - 1
                    and args.relay_slots < args.relay_slots_max):
                args.relay_slots = min(args.relay_slots + 6, args.relay_slots_max)
            if float(best.metrics["makespan_s"]) <= args.target_s + 1e-4:
                print("Q3 时间优化：达到目标，开始保存并独立复核", flush=True)
                break
    except KeyboardInterrupt:
        print("Q3 时间优化：中断后保存已经验收的最好解", flush=True)
    return _save_result(args, origin, source_summary, data_run, relay,
                        geometry, best, reference, history, candidate_log,
                        time.monotonic() - started)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme3-table", type=Path, default=None)
    parser.add_argument("--warm-table", type=Path, default=None,
                        help="同源数据且独立验收 PASS 的上次时间优化表")
    parser.add_argument("--data-run", type=Path, default=None)
    parser.add_argument("--output-root", type=Path,
                        default=CODE_DIR / "3_outputs" / "8_time")
    parser.add_argument("--target-s", type=float, default=7200)
    parser.add_argument("--time-budget-s", type=float, default=1200,
                        help="正式搜索预算；初始化和出表出图另计")
    parser.add_argument("--mapping-seconds", type=float, default=25)
    parser.add_argument("--global-seconds", type=float, default=120)
    parser.add_argument("--local-seconds", type=float, default=50)
    parser.add_argument("--candidate-seconds", type=float, default=18)
    parser.add_argument("--certify-seconds", type=float, default=25)
    parser.add_argument("--type-seconds", type=float, default=20)
    parser.add_argument("--max-rounds", type=int, default=24)
    parser.add_argument("--grades", default="G0,G1,G2",
                        help="搜索档位顺序；续跑可选 G2")
    parser.add_argument("--full-every", type=int, default=3)
    parser.add_argument("--max-packages", type=int, default=60)
    parser.add_argument("--max-new-choices", type=int, default=130)
    parser.add_argument("--max-active-choices", type=int, default=350)
    parser.add_argument("--max-stops", type=int, default=2)
    parser.add_argument("--relay-slots", type=int, default=18)
    parser.add_argument("--relay-slots-max", type=int, default=30)
    parser.add_argument("--max-sites", type=int, default=48)
    parser.add_argument("--fast-site-additions", type=int, default=4,
                        help="尾段中继附近新增的快速站点候选数")
    parser.add_argument("--max-site-points", type=int, default=2400)
    parser.add_argument("--horizon-s", type=int, default=21600)
    parser.add_argument("--sample-step-s", type=float, default=20)
    parser.add_argument("--candidate-spacing-m", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--min-remaining-s", type=float, default=5)
    args = parser.parse_args(argv)
    args.grades = tuple(part.strip() for part in args.grades.split(",") if part.strip())
    if not args.grades or any(grade not in SOFT_GRADES for grade in args.grades):
        parser.error("--grades 只能由 G0,G1,G2 组成")
    times = (args.target_s, args.time_budget_s, args.mapping_seconds,
             args.global_seconds, args.local_seconds, args.candidate_seconds,
             args.certify_seconds, args.type_seconds, args.sample_step_s,
             args.min_remaining_s)
    counts = (args.max_rounds, args.full_every, args.max_packages,
              args.max_new_choices, args.max_active_choices, args.max_stops,
              args.relay_slots, args.max_sites, args.max_site_points,
              args.horizon_s, args.candidate_spacing_m, args.workers)
    if min(times) <= 0 or min(counts) < 1 or args.relay_slots_max < args.relay_slots:
        parser.error("搜索预算、候选数和时域必须为正，中继槽上限不能低于起点")
    if args.fast_site_additions < 0:
        parser.error("快速中继候选点数量不能为负")
    result = run(args)
    print(json.dumps({key: result[key] for key in
                      ("status", "improved_from_scheme3", "target_reached",
                       "target_gap_s", "final_metrics", "table_dir", "figure_dir")},
                     ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
