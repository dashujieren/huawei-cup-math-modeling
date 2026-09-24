"""D题第三问：可回滚的运输与中继联合排程。

复用 q3_1_optimize 的题面物理计算、通信证书和独立验收。一次搜索中的
后续硬时限任务若排不进，会把该任务提前并回退到先前的状态重排；中继
服务窗口还可在能源及资源日历许可时延长。有限候选搜索不证明全局最优。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

import q3_1_optimize as q3
from q2_1_optimize import Context, check_plan


CODE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = CODE_DIR / "3_outputs" / "2_joint"
PREFERRED_GROUPINGS = (
    "balanced", "compact", "neighbor_pair", "hard_split",
    "small_batches", "single_box_fallback",
)


def promoted_order(tasks: list[q3.RouteTask], failed_index: int,
                   target_index: int) -> list[q3.RouteTask]:
    """将失败任务移到回退点；此前缀不动，后缀全部重排。"""
    if not 0 <= target_index < failed_index < len(tasks):
        raise ValueError("回退目标必须位于失败任务之前")
    arranged = list(tasks)
    task = arranged.pop(failed_index)
    arranged.insert(target_index, task)
    if sorted(item.batch_id for item in arranged) != sorted(
            item.batch_id for item in tasks):
        raise AssertionError("重排导致架次丢失或重复")
    return arranged


def pick_diverse(states: list[q3.JointState], beam_width: int
                 ) -> list[q3.JointState]:
    """保留不同机型、执行机以及中继位置的前沿，避免束宽全被同构解占满。"""
    if beam_width < 1:
        raise ValueError("束宽必须为正")
    ordered = sorted(states, key=q3._state_score)
    unique: list[q3.JointState] = []
    seen: set[tuple[Any, ...]] = set()
    for state in ordered:
        last = state.sorties[-1]
        signature = (
            last["type_id"], last["uav_id"],
            tuple(sorted((mission.candidate.candidate_id, mission.uav_id,
                          mission.unit_id) for mission in state.missions)),
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(state)
        if len(unique) >= beam_width:
            return unique
    chosen = {id(state) for state in unique}
    unique.extend(state for state in ordered if id(state) not in chosen)
    return unique[:beam_width]


def validate_state(ctx: Context, state: q3.JointState,
                   links: q3.LinkEvaluator, dem: q3.DemGrid,
                   data_run: Path) -> dict[str, Any]:
    """完整状态在写入官方表前再做一次内存层检查。"""
    transport_bad = [row for row in check_plan(
        ctx, {"sorties": state.sorties, "deliveries": state.deliveries})
        if row["status"] != "PASS"]
    bounds = {row["batch_id"]: (row["start_s"], row["return_s"])
              for row in state.sorties}
    relay_rows = q3.relay_rows_from_missions(state.missions)
    comm = q3.verify_continuous_coverage(
        state.trajectories, links,
        q3.communication_rows(state.slices, state.assigned, bounds),
        relay_rows)
    resources = q3.validate_relay_rows(relay_rows, data_run, links, dem)
    status = ("PASS" if not transport_bad and comm["status"] == "PASS"
              and resources["status"] == "PASS" else "FAILED_VALIDATION")
    return {
        "status": status,
        "transport_failure": transport_bad[0] if transport_bad else None,
        "communication_status": comm["status"],
        "communication_failure": comm["issues"][0] if comm["issues"] else None,
        "resource_status": resources["status"],
        "resource_failure": resources.get("first_failure"),
    }


def schedule_with_rewind(
    ctx: Context, tasks: list[q3.RouteTask],
    candidates: list[q3.RelayCandidate], relay: dict[str, Any],
    links: q3.LinkEvaluator, dem: q3.DemGrid,
    positions: dict[str, tuple[float, float, float]],
    sample_step_s: float, site_tree: cKDTree | None,
    access_cache: dict[tuple[Any, ...], bool], *,
    grouping: str, beam_width: int, candidate_limit: int,
    start_probes: int, max_rewinds: int, max_splits: int,
    soft_horizon_s: float, max_seconds: float,
) -> tuple[q3.JointState, dict[str, Any]]:
    """当前批失败时先让它提前，并从保存的前缀重新安排后续任务。"""
    if not tasks or min(beam_width, candidate_limit, start_probes) < 1:
        raise ValueError("任务、束宽、中继候选数和出发时刻探测数必须为正")
    if min(max_rewinds, max_splits) < 0 or max_seconds < 0:
        raise ValueError("回退、拆批和时间预算不能为负")
    started = perf_counter()
    pending = q3._task_order(ctx, tasks, "hard_slack")
    beam = [q3._empty_joint_state()]
    snapshots: list[list[q3.JointState]] = [beam]
    profiles: dict[tuple[Any, ...], q3.RouteProfile] = {}
    rewind_counts: dict[str, int] = defaultdict(int)
    events: list[dict[str, Any]] = []
    rewinds = 0
    splits = 0
    index = 0
    best_partial = beam[0]
    last_failure: dict[str, Any] = {}
    while index < len(pending):
        if max_seconds and perf_counter() - started >= max_seconds:
            return best_partial, {
                "status": "SEARCH_TIMEOUT", "grouping": grouping,
                "elapsed_s": round(perf_counter() - started, 2),
                "scheduled_boxes": len(best_partial.deliveries),
                "scheduled_sorties": len(best_partial.sorties),
                "rewinds": rewinds, "adaptive_splits": splits,
                "last_failure": last_failure, "events": events,
            }
        task = pending[index]
        successors: list[q3.JointState] = []
        reasons: dict[str, int] = defaultdict(int)
        site_limits = dict.fromkeys((
            min(candidate_limit, len(candidates)),
            min(max(4 * candidate_limit, 64), len(candidates)),
            len(candidates),
        ))
        for widen, site_limit in enumerate(site_limits):
            for state in beam:
                options, failed = q3._insert_task(
                    ctx, state, task, candidates, links, relay,
                    positions, sample_step_s, site_tree,
                    profiles, access_cache,
                    max(3, beam_width // 2), site_limit,
                    start_probes + 4 * widen, soft_horizon_s)
                successors.extend(options)
                for name, count in failed.items():
                    reasons[name] += count
            if successors:
                break
        if successors:
            beam = pick_diverse(successors, beam_width)
            snapshots.append(beam)
            index += 1
            if len(beam[0].deliveries) > len(best_partial.deliveries):
                best_partial = beam[0]
            if index == 1 or index % 5 == 0 or index == len(pending):
                print(f"Q3 {grouping}: {index}/{len(pending)} 批，"
                      f"已交付 {len(beam[0].deliveries)}/80 箱，"
                      f"中继 {len(beam[0].missions)} 架次；"
                      f"回退 {rewinds} 次，耗时 {perf_counter()-started:.1f}s",
                      flush=True)
            continue
        last_failure = {
            "failed_batch_id": task.batch_id,
            "failed_box_ids": list(task.route.box_ids),
            "failed_zone_ids": list(task.route.zones),
            "failure_counts": dict(reasons),
            "scheduled_boxes_before_failure": len(beam[0].deliveries),
        }
        if index > 0 and rewinds < max_rewinds:
            levels = (2, 5, 10, 20, 40, len(pending))
            level = levels[min(rewind_counts[task.batch_id], len(levels) - 1)]
            target = max(0, index - level)
            rewind_counts[task.batch_id] += 1
            pending = promoted_order(pending, index, target)
            beam = snapshots[target]
            snapshots = snapshots[:target + 1]
            index = target
            rewinds += 1
            events.append({"kind": "rewind", "batch_id": task.batch_id,
                           "new_position": target, "window": level,
                           "reason": dict(reasons)})
            print(f"Q3 {grouping}: {task.batch_id} 排不进，回退至第 "
                  f"{target + 1} 批并重排；已回退 {rewinds}/{max_rewinds}",
                  flush=True)
            continue
        parts = (q3._split_current_task(ctx, task, pending)
                 if splits < max_splits else None)
        if parts is not None:
            pending[index:index + 1] = q3._task_order(ctx, parts, "hard_slack")
            splits += 1
            events.append({"kind": "split", "batch_id": task.batch_id,
                           "new_batch_ids": [part.batch_id for part in parts],
                           "reason": dict(reasons)})
            print(f"Q3 {grouping}: 拆分 {task.batch_id} 为 "
                  f"{','.join(part.batch_id for part in parts)}", flush=True)
            continue
        return best_partial, {
            "status": "SEARCH_INCOMPLETE", "grouping": grouping,
            "elapsed_s": round(perf_counter() - started, 2),
            "scheduled_boxes": len(best_partial.deliveries),
            "scheduled_sorties": len(best_partial.sorties),
            "rewinds": rewinds, "adaptive_splits": splits,
            "last_failure": last_failure, "events": events,
        }
    first_invalid: dict[str, Any] | None = None
    for state in beam:
        result = validate_state(ctx, state, links, dem, ctx.data_run)
        if result["status"] == "PASS":
            return state, {
                "status": "PASS", "grouping": grouping,
                "elapsed_s": round(perf_counter() - started, 2),
                "scheduled_boxes": len(state.deliveries),
                "scheduled_sorties": len(state.sorties),
                "rewinds": rewinds, "adaptive_splits": splits,
                "events": events,
            }
        if first_invalid is None:
            first_invalid = result
    return best_partial, {
        "status": "FAILED_VALIDATION", "grouping": grouping,
        "elapsed_s": round(perf_counter() - started, 2),
        "scheduled_boxes": len(best_partial.deliveries),
        "scheduled_sorties": len(best_partial.sorties),
        "rewinds": rewinds, "adaptive_splits": splits,
        "last_failure": first_invalid, "events": events,
    }


def run_joint(args: argparse.Namespace) -> dict[str, Any]:
    data_run = (args.data_run.resolve() if args.data_run else
                q3.find_data_run(CODE_DIR, q3.EXPECTED_MANIFEST))
    if not q3.verify_ready(data_run):
        raise ValueError(f"未经 READY 校验的数据目录：{data_run}")
    relay, dem, links, source = q3._scene(data_run)
    ctx = Context(data_run)
    positions = q3._node_positions(data_run)
    route_sets = q3._raw_route_sets(ctx)
    route_sets.sort(key=lambda pair: PREFERRED_GROUPINGS.index(pair[0])
                    if pair[0] in PREFERRED_GROUPINGS
                    else len(PREFERRED_GROUPINGS))
    source_slices = q3._source_slices_from_raw(
        ctx, positions, links, args.sample_step_s, route_sets)
    candidates, candidate_stats = q3.generate_candidates(
        dem, links, source_slices, relay, positions["O01"],
        spacing_m=args.candidate_spacing_m, max_points=12000)
    site_tree = (cKDTree(np.asarray([item.hover for item in candidates],
                                    dtype=float)) if candidates else None)
    output_root = args.output_root.resolve()
    stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    table_dir = output_root / f"q3_joint_{stamp}_table"
    figure_dir = output_root / f"q3_joint_{stamp}_figure"
    table_dir.mkdir(parents=True, exist_ok=False)
    q3.json_dump(table_dir / "Q3_候选筛选统计.json", candidate_stats)
    access_cache: dict[tuple[Any, ...], bool] = {}
    attempts: list[dict[str, Any]] = []
    selected: q3.JointState | None = None
    best_partial: q3.JointState | None = None
    for name, tasks in route_sets[:args.max_groupings]:
        print(f"Q3 联合排程：组批 {name}，初始 {len(tasks)} 批",
              flush=True)
        state, outcome = schedule_with_rewind(
            ctx, tasks, candidates, relay, links, dem, positions,
            args.sample_step_s, site_tree, access_cache,
            grouping=name, beam_width=args.beam_width,
            candidate_limit=args.candidate_limit,
            start_probes=args.start_probes,
            max_rewinds=args.max_rewinds,
            max_splits=args.max_splits,
            soft_horizon_s=args.soft_horizon_s,
            max_seconds=args.max_seconds_per_grouping)
        attempts.append(outcome)
        q3.json_dump(table_dir / "Q3_搜索进展.json", {
            "status": "RUNNING", "attempts": attempts,
            "source": source, "candidate_stats": candidate_stats})
        if best_partial is None or len(state.deliveries) > len(best_partial.deliveries):
            best_partial = state
        if outcome["status"] == "PASS":
            selected = state
            break
    if selected is None:
        summary = {
            "status": "SEARCH_INCOMPLETE",
            "interpretation": "有限候选及预算内未找到完整方案，不证明题目无解",
            "source": source, "candidate_stats": candidate_stats,
            "attempts": attempts,
            "best_partial_delivered_boxes": (
                len(best_partial.deliveries) if best_partial else 0),
            "best_partial_transport_sorties": (
                len(best_partial.sorties) if best_partial else 0),
            "global_optimality_proven": False,
            "table_dir": str(table_dir), "figure_dir": None,
        }
        q3.json_dump(table_dir / "Q3_运行摘要.json", summary)
        return summary
    q3._save_joint_tables(table_dir, selected, relay)
    validation = q3.validate_official_tables(table_dir, data_run, links, dem)
    q3._save_verification(table_dir, validation)
    transport_energy = sum(row["energy_kwh"] for row in selected.sorties)
    relay_energy = sum(row.energy_kwh for row in selected.missions)
    summary = {
        "status": validation["status"],
        "method": "conflict-guided rewind, adaptive batching, reusable relay windows",
        "interpretation": "从原始80箱组批，冲突时回退并联动重排运输与中继；有限搜索不证明全局最优",
        "assumption": "同一中继架次可同时保障多架运输机，每架运输机每一时刻仅选一种通信方式",
        "source": source, "candidate_stats": candidate_stats,
        "attempts": attempts, "global_optimality_proven": False,
        "transport_sorties": len(selected.sorties),
        "relay_sorties": len(selected.missions),
        "delivered_boxes": len(selected.deliveries),
        "hard_deadline_violations": sum(
            row["hard_deadline_s"] is not None and
            row["complete_s"] > row["hard_deadline_s"] + q3.EPS
            for row in selected.deliveries),
        "transport_last_return_s": max(row["return_s"]
                                       for row in selected.sorties),
        "relay_last_return_s": max(row.return_s for row in selected.missions),
        "joint_makespan_s": max(
            [row["return_s"] for row in selected.sorties] +
            [row.return_s for row in selected.missions]),
        "transport_energy_kwh": transport_energy,
        "relay_energy_kwh": relay_energy,
        "total_energy_kwh": transport_energy + relay_energy,
        "soft_weighted_lateness_s": sum(
            row["priority"] * row["soft_lateness_s"]
            for row in selected.deliveries if row["hard_deadline_s"] is None),
        "communication_status": validation["communication"]["status"],
        "transport_status": validation["transport_status"],
        "relay_resource_status": validation["relay_resources"]["status"],
        "table_dir": str(table_dir),
        "figure_dir": str(figure_dir) if validation["status"] == "PASS" else None,
    }
    q3.json_dump(table_dir / "Q3_运行摘要.json", summary)
    if summary["status"] == "PASS":
        q3._save_figures(figure_dir, selected.slices, selected.assigned,
                         selected.missions)
        (table_dir / "Q3_READY.txt").write_text(
            "Q3 conflict-guided joint search; independently verified PASS\n"
            f"source_manifest_sha256={source['source_manifest_sha256']}\n",
            encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-run", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-groupings", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-rewinds", type=int, default=14)
    parser.add_argument("--max-splits", type=int, default=16)
    parser.add_argument("--candidate-limit", type=int, default=16)
    parser.add_argument("--start-probes", type=int, default=10)
    parser.add_argument("--sample-step-s", type=float, default=20.0)
    parser.add_argument("--candidate-spacing-m", type=int, default=500)
    parser.add_argument("--soft-horizon-s", type=float, default=86400.0)
    parser.add_argument("--max-seconds-per-grouping", type=float, default=600.0,
                        help="每种组批最多搜索的秒数；0 表示不设上限")
    parser.add_argument("--validate-only", type=Path, default=None,
                        metavar="Q3_TABLE_DIR")
    args = parser.parse_args(argv)
    if args.max_groupings < 1 or args.candidate_spacing_m < 1:
        parser.error("组批数和候选点间距必须为正")
    if args.sample_step_s <= 0 or args.soft_horizon_s <= 0:
        parser.error("分段时间和软任务期限必须为正")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_run = (args.data_run.resolve() if args.data_run else
                q3.find_data_run(CODE_DIR, q3.EXPECTED_MANIFEST))
    if args.validate_only is not None:
        _, dem, links, _ = q3._scene(data_run)
        report = q3.validate_official_tables(
            args.validate_only.resolve(), data_run, links, dem)
        print(json.dumps({
            "status": report["status"],
            "transport_status": report["transport_status"],
            "communication_status": report["communication"]["status"],
            "relay_resource_status": report["relay_resources"]["status"],
        }, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "PASS" else 2
    summary = run_joint(args)
    print(json.dumps({
        key: summary.get(key) for key in (
            "status", "delivered_boxes", "transport_sorties",
            "relay_sorties", "hard_deadline_violations",
            "joint_makespan_s", "total_energy_kwh", "table_dir")
    }, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
