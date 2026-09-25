"""问题三方案四：关键任务链驱动的运输与中继协同压缩。

从已验收的方案三出发。先固定组批做两次全程时间优化，再主动增补中继站点，
最后在关键链邻域做机型替换、拆并批和多目标权衡。所有正式输出重新独立验收。
有限候选和限时 CP-SAT 的下界不是原题全局下界。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import q3_1_optimize as base
import q3_2_optimize as previous
import q3_3_optimize as scheme3
from q2_1_optimize import Context


CODE_DIR = Path(__file__).resolve().parent
STAGES = ("fixed_used", "fixed_all", "sites", "rebatch", "tradeoff")
GRADES = {
    "strict": (24, 34, 1.00, 1.00),
    "small": (26, 36, 1.05, 1.00),
    "extended": (28, 38, 1.10, 1.05),
}
GOALS = {"fast": "scheme2_makespan", "energy": "scheme2_energy",
         "lateness": "scheme2_lateness"}


def _source_table(root: Path) -> Path:
    for index_path in sorted(root.glob("q3_opt3_*_index.json"), reverse=True):
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for row in index.get("representatives", []):
            table = Path(row["table_dir"])
            if (row.get("goal") == "sorties" and row.get("status") == "PASS"
                    and (table / "Q3_READY.txt").is_file()):
                return table
    raise FileNotFoundError("缺少已验收的方案三少架次结果；请用 --scheme3-table 指定")


def _min_hard_slack(table: Path) -> float:
    rows = base.read_rows(table / "Q3_逐箱时限检查.csv")
    values = [float(r["hard_slack_s"]) for r in rows
              if r.get("hard_deadline_s") not in ("", None)]
    return min(values) if values else math.inf


def _metrics_from_summary(table: Path) -> dict[str, float | int]:
    summary = json.loads((table / "Q3_运行摘要.json").read_text(encoding="utf-8"))
    return {key: summary["final_metrics"][key] for key in scheme3.METRIC_KEYS}


def _model_caps(reference: dict[str, float | int], grade: str, goal: str,
                ctx: Context, time_cap: float | None = None) -> tuple[dict[str, int], dict[str, float]]:
    nt, total, ef, lf = GRADES[grade]
    # CP 的完成时刻向上取整、期望时刻向下取整；显式记录这个仅供搜索的包络。
    late_mapping = 2 * sum(max(1, round(float(box["priority"])))
                           for box in ctx.boxes.values())
    energy_mapping_j = max(1000, 2 * (total + len(ctx.boxes)))
    caps = {"transport_sorties": nt, "total_sorties": total,
            "energy_j": math.ceil(float(reference["energy_kwh"]) * ef * 3_600_000
                                  + energy_mapping_j),
            "weighted_lateness": math.ceil(float(reference["weighted_lateness"]) * lf
                                               + late_mapping)}
    if goal == "energy":
        caps.pop("energy_j")
    elif goal == "lateness":
        caps.pop("weighted_lateness")
    if time_cap is not None:
        caps["makespan_s"] = math.ceil(time_cap + 1.0)
    return caps, {"lateness_mapping_s": late_mapping,
                  "energy_mapping_j": energy_mapping_j,
                  "makespan_mapping_s": 1.0}


def _actual_within(metrics: dict[str, float | int], reference: dict[str, float | int],
                   grade: str, goal: str, time_cap: float | None,
                   hard_slack: float, margin: float) -> bool:
    nt, total, ef, lf = GRADES[grade]
    if (int(metrics["transport_sorties"]) > nt or
            int(metrics["transport_sorties"]) + int(metrics["relay_sorties"]) > total or
            hard_slack + 1e-6 < margin):
        return False
    if goal != "energy" and float(metrics["energy_kwh"]) > float(reference["energy_kwh"]) * ef + 1e-6:
        return False
    if goal != "lateness" and float(metrics["weighted_lateness"]) > float(reference["weighted_lateness"]) * lf + 1e-3:
        return False
    if time_cap is not None and float(metrics["makespan_s"]) > time_cap + 1e-4:
        return False
    return True


def _union_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    left, right = ordered[0]
    total = 0.0
    for a, b in ordered[1:]:
        if a > right:
            total += right - left
            left, right = a, b
        else:
            right = max(right, b)
    return total + right - left


def _diagnose(table: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    transport = base.read_rows(table / "Q3_运输明细.csv")
    relay = base.read_rows(table / "Q3_中继资源时间线.csv")
    comm = base.read_rows(table / "Q3_通信保障.csv")
    t_by_id = {r["batch_id"]: r for r in transport}
    r_by_id = {r["mission_id"]: r for r in relay}
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    wait_rows: list[dict[str, Any]] = []
    for rows, id_key, resources, ready in (
            (transport, "batch_id", ("uav_id", "battery_id"),
             {"uav_id": "return_s", "battery_id": "charge_end_s"}),
            (relay, "mission_id", ("uav_id", "energy_unit_id"),
             {"uav_id": "uav_next_ready_s", "energy_unit_id": "unit_next_ready_s"})):
        for field in resources:
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                grouped[row[field]].append(row)
            for _, group in grouped.items():
                group.sort(key=lambda r: float(r["start_s"]))
                for prior, after in zip(group, group[1:]):
                    prior_id, after_id = prior[id_key], after[id_key]
                    gap = float(after["start_s"]) - float(prior[ready[field]])
                    adjacency[after_id].append((prior_id, field))
                    wait_rows.append({"前驱": prior_id, "后继": after_id,
                                      "约束": field, "可用后等待秒": gap,
                                      "后继开始秒": float(after["start_s"])})
    coverage: dict[str, list[tuple[float, float]]] = defaultdict(list)
    related: dict[str, set[str]] = defaultdict(set)
    for row in comm:
        rid = row.get("中继架次编号", "")
        tid = row.get("运输架次编号", "")
        if rid and tid and row.get("保障方式") != "直连":
            coverage[rid].append((float(row["开始时刻（s）"]),
                                  float(row["结束时刻（s）"])))
            related[rid].add(tid)
            adjacency[tid].append((rid, "通信保障"))
            adjacency[rid].append((tid, "共享窗口"))
    window_rows = []
    for rid, row in r_by_id.items():
        ready = float(row["link_ready_s"])
        end = float(row["service_end_s"])
        service = end - ready
        covered = _union_length(coverage[rid])
        window_rows.append({"中继架次": rid, "建链完成秒": ready, "服务结束秒": end,
                            "服务窗口秒": service, "通信并集秒": covered,
                            "窗口空闲秒": max(0., service - covered),
                            "关联运输架次": ";".join(sorted(related[rid])),
                            "出入站秒": float(row["return_s"]) - float(row["start_s"]) - service,
                            "返航秒": float(row["return_s"])})
    terminal = sorted([(float(r["return_s"]), r["batch_id"]) for r in transport] +
                      [(float(r["return_s"]), r["mission_id"]) for r in relay],
                      reverse=True)
    queue = deque((name, 0) for _, name in terminal[:3])
    seen = set()
    chosen = []
    while queue and len(chosen) < 12:
        name, depth = queue.popleft()
        if name in seen:
            continue
        seen.add(name)
        if name in t_by_id:
            chosen.append(name)
        if depth < 4:
            for prior, _ in adjacency[name]:
                queue.append((prior, depth + 1))
    # 边缘批次保持进入邻域，避免只有中继前驱而没有足够运输路线。
    for _, name in terminal:
        if name in t_by_id and name not in chosen:
            chosen.append(name)
        if len(chosen) >= 8:
            break
    return wait_rows, window_rows, chosen


def _earliest_shift_hint(table: Path, choices: list[base.JointChoice],
                         source_hints: dict[str, Any]
                         ) -> tuple[dict[str, Any], dict[str, Any]]:
    """固定任务长度/资源顺序/通信分配的最长路提示，绝不当作验收结果。"""
    transport = base.read_rows(table / "Q3_运输明细.csv")
    relay = base.read_rows(table / "Q3_中继资源时间线.csv")
    comm = base.read_rows(table / "Q3_通信保障.csv")
    original = {r["batch_id"]: float(r["start_s"]) for r in transport}
    original.update({r["mission_id"]: float(r["start_s"]) for r in relay})
    edges: dict[tuple[str, str], float] = {}

    def edge(a: str, b: str, weight: float) -> None:
        if a in original and b in original:
            edges[(a, b)] = max(edges.get((a, b), -math.inf), weight)

    for rows, ident, resource_ready in (
            (transport, "batch_id", (("uav_id", "return_s"),
                                     ("battery_id", "charge_end_s"))),
            (relay, "mission_id", (("uav_id", "uav_next_ready_s"),
                                    ("energy_unit_id", "unit_next_ready_s")))):
        for resource, ready in resource_ready:
            groups: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                groups[row[resource]].append(row)
            for group in groups.values():
                group.sort(key=lambda r: float(r["start_s"]))
                for prior, after in zip(group, group[1:]):
                    edge(prior[ident], after[ident],
                         float(prior[ready]) - float(prior["start_s"]))
    relay_by_id = {r["mission_id"]: r for r in relay}
    for row in comm:
        tid, rid = row.get("运输架次编号", ""), row.get("中继架次编号", "")
        if tid not in original or rid not in relay_by_id:
            continue
        mission = relay_by_id[rid]
        t_start, r_start = original[tid], original[rid]
        a = float(row["开始时刻（s）"]) - t_start
        b = float(row["结束时刻（s）"]) - t_start
        ready = float(mission["link_ready_s"]) - r_start
        end = float(mission["service_end_s"]) - r_start
        edge(rid, tid, ready - a)
        edge(tid, rid, b - end)
    earliest = {name: 0.0 for name in original}
    for iteration in range(len(original)):
        changed = False
        for (before, after), weight in edges.items():
            proposed = earliest[before] + weight
            if proposed > earliest[after] + 1e-7:
                earliest[after] = proposed
                changed = True
        if not changed:
            break
    else:
        raise ValueError("固定时序差约束存在正权环；请检查原表窗口与资源边")
    violation = max((earliest[name] - original[name] for name in original),
                    default=0.0)
    if violation > 1e-4:
        raise ValueError(f"前移网络与原表不相容，最大残差 {violation:.6f}s")
    by_boxes = {tuple(sorted(c.task.route.box_ids)): scheme3._route_key(c)
                for c in choices}
    shifted = {"routes": {}, "relays": [],
               "fixed_transport": source_hints["fixed_transport"]}
    for row in transport:
        key = by_boxes[tuple(sorted(row["box_ids"].split(";")))]
        shifted["routes"][key] = max(0, round(earliest[row["batch_id"]]))
    relay_rows = base.read_rows(table / "Q3_中继架次.csv")
    by_slot = {f"R{i+1:03d}": item for i, item in
               enumerate(source_hints["relays"])}
    for row in relay_rows:
        rid = row["中继架次编号"]
        if rid not in by_slot:
            continue
        slot, site, uav, unit, ready, end = by_slot[rid]
        shift = earliest[rid] - original[rid]
        shifted["relays"].append((slot, site, uav, unit,
                                  max(0, round(ready + shift)),
                                  max(0, round(end + shift))))
    transport_c = max((earliest[r["batch_id"]] + float(r["return_s"]) -
                       float(r["start_s"]) for r in transport), default=0.)
    relay_c = max((earliest[r["mission_id"]] + float(r["return_s"]) -
                   float(r["start_s"]) for r in relay), default=0.)
    diagnostic = {"status": "UNVALIDATED_HINT_ONLY", "nodes": len(original),
                  "edges": len(edges), "earliest_fixed_structure_c_s": max(transport_c, relay_c),
                  "original_c_s": max([float(r["return_s"]) for r in transport + relay]),
                  "maximum_original_residual_s": max(0.0, violation),
                  "task_starts": [
                      {"任务": name, "原开始秒": original[name],
                       "前移提示秒": earliest[name],
                       "可前移秒": original[name] - earliest[name]}
                      for name in sorted(original)]}
    return shifted, diagnostic


def _batch_positions(entry: scheme3.Accepted, origin: Path,
                     incumbent: list[base.JointChoice], relay: dict[str, Any]
                     ) -> tuple[list[base.JointChoice], dict[str, int],
                                tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]]:
    if entry.solution is None:
        current = incumbent
        table = entry.origin or origin
        rows = base.read_rows(table / "Q3_运输明细.csv")
        by_boxes = {tuple(sorted(choice.task.route.box_ids)): i
                    for i, choice in enumerate(current)}
        mapping = {r["batch_id"]: by_boxes[tuple(sorted(r["box_ids"].split(";")))]
                   for r in rows}
        return current, mapping, _diagnose(table)
    current = scheme3._selected(entry.solution)
    ordered = sorted(entry.solution["routes"], key=lambda row: (row[1], row[0]))
    key_positions = {scheme3._route_key(choice): i for i, choice in enumerate(current)}
    mapping = {f"T{number:03d}": key_positions[scheme3._route_key(entry.solution["pool"][i])]
               for number, (i, *_rest) in enumerate(ordered, 1)}
    with tempfile.TemporaryDirectory(prefix="q3_opt4_chain_") as tmp:
        table = Path(tmp)
        base._save_joint_tables(table, entry.state, relay)
        diagnostic = _diagnose(table)
    return current, mapping, diagnostic


def _proactive_sites(geometry: scheme3.GeometryPool,
                     choices: list[base.JointChoice], critical: set[int],
                     deadline: float, add_limit: int) -> tuple[int, list[dict[str, Any]]]:
    if len(geometry.sites) >= geometry.max_sites or time.monotonic() >= deadline:
        return 0, []
    blind = [item for i in sorted(critical) for item in choices[i].profile.slices
             if not item.direct_available]
    if not blind:
        return 0, []
    # 局部候选来自关键航段、现有站点周边和站点对中点；原站点不删除。
    stride = max(1, len(blind) // 30)
    sample = blind[::stride][:30]
    xy = set()
    for item in sample:
        x, y, _ = item.midpoint
        xy.update((round(x + dx), round(y + dy)) for dx, dy in
                  ((0, 0), (150, 0), (-150, 0), (0, 150), (0, -150)))
    old = geometry.sites[:]
    for site in old[:16]:
        xy.update((round(site.x_m + dx), round(site.y_m + dy)) for dx, dy in
                  ((0, 0), (150, 0), (0, 150)))
    for a, b in zip(old[:8], old[1:9]):
        xy.add((round((a.x_m + b.x_m) / 2), round((a.y_m + b.y_m) / 2)))
    points = sorted(xy)[:geometry.max_points]
    generated, stats = base.generate_candidates(
        geometry.dem, geometry.links, [], geometry.relay,
        geometry.positions["O01"], spacing_m=geometry.spacing_m,
        max_points=geometry.max_points, extra_xy_points=points)
    known = {s.candidate_id for s in geometry.sites}
    ranked = []
    for site in generated:
        if time.monotonic() >= deadline:
            break
        if site.candidate_id in known:
            continue
        cover = frozenset(i for i, item in enumerate(sample)
                          if base._access_certified(site, item, geometry.links,
                                                    geometry.access_cache))
        if not cover:
            continue
        travel = site.outward_s + site.homeward_s
        ranked.append((site, cover, travel))
    nondominated = []
    for site, cover, travel in ranked:
        if any(cover <= other_cover and other_travel <= travel + 1e-6 and
               other_site.travel_energy_kwh <= site.travel_energy_kwh + 1e-8 and
               other_site.max_service_s >= site.max_service_s - 1e-6
               for other_site, other_cover, other_travel in ranked
               if other_site is not site):
            continue
        nondominated.append((site, cover, travel))
    nondominated.sort(key=lambda row: (-len(row[1]), row[2],
                                       row[0].travel_energy_kwh,
                                       -row[0].max_service_s))
    records = []
    for site, cover, travel in nondominated[:min(add_limit, geometry.max_sites - len(old))]:
        geometry.sites.append(site)
        records.append({"站点": site.candidate_id, "来源": "关键盲段主动增补",
                        "覆盖采样数": len(cover), "往返秒": travel,
                        "往返能耗kWh": site.travel_energy_kwh,
                        "最大服务秒": site.max_service_s,
                        "回传余量dB": site.backhaul_margin_db})
    geometry.stats["new_sites"] += len(records)
    records.append({"站点": "生成统计", "来源": json.dumps(stats, ensure_ascii=False),
                    "覆盖采样数": len(sample)})
    return len(records) - 1, records


def _rebuild(geometry: scheme3.GeometryPool,
             choices: list[base.JointChoice]) -> list[base.JointChoice]:
    rebuilt = []
    for choice in choices:
        new = geometry.certify(choice.task.route, choice.kind, choice.option)
        if new is None:
            raise ValueError("增补站点后原路线认证失败，不能继续使用旧通信覆盖集")
        rebuilt.append(new)
    return rebuilt


def _attempt(stage: str, grade: str, goal: str,
             ctx: Context, relay: dict[str, Any], geometry: scheme3.GeometryPool,
             pool: list[base.JointChoice], hints: dict[str, Any],
             fixed: dict[tuple[Any, str], tuple[int, str, str]] | None,
             reference: dict[str, float | int], data_run: Path,
             args: argparse.Namespace, seconds: float,
             time_cap: float | None = None, margin: float = 0.,
             require_all: bool = False) -> tuple[scheme3.Accepted | None, dict[str, Any]]:
    caps, envelope = _model_caps(reference, grade, goal, ctx, time_cap)
    started = time.monotonic()
    print(f"Q3方案四 {stage}: {goal}/{grade}, 路线机型={len(pool)}, "
          f"站点={len(geometry.sites)}, 任务槽={args.relay_slots}, "
          f"时限={seconds:.0f}s, 余量={margin:.0f}s", flush=True)
    solution, report = base._solve_joint(
        ctx, relay, geometry.sites, pool, args.horizon_s, args.relay_slots,
        seconds, args.workers, False, hints=hints,
        fixed_partial_transport=fixed, require_all_routes=require_all,
        objective_mode=GOALS[goal], metric_limits=caps,
        explicit_relay_uavs=True, search_seed=args.seed + len(pool) + len(stage),
        hard_deadline_buffer_s=margin)
    record = {"stage": stage, "grade": grade, "goal": goal,
              "caps": caps, "mapping_envelope": envelope,
              "hard_margin_s": margin, "model_wall_s": time.monotonic() - started,
              **report, "validation": "NO_SOLUTION"}
    if solution is None:
        return None, record
    state, check = previous._inspect_solution(
        solution, ctx, relay, geometry.sites, data_run, geometry.links, geometry.dem)
    metrics = previous._metrics(state)
    hard_slack = min((float(d["hard_deadline_s"]) - float(d["complete_s"])
                      for d in state.deliveries if d["hard_deadline_s"] is not None),
                     default=math.inf)
    accepted = check["status"] == "PASS" and _actual_within(
        metrics, reference, grade, goal, time_cap, hard_slack, margin)
    record.update({"validation": check["status"], "metrics": metrics,
                   "hard_slack_s": hard_slack,
                   "within_actual_caps": accepted})
    if not accepted:
        return None, record
    return scheme3.Accepted(metrics, solution, state, None, stage, grade), record


def _comparison(origin: Path, scheme3_summary: dict[str, Any],
                archive: list[scheme3.Accepted]) -> list[dict[str, Any]]:
    rows = []
    scheme2_table = Path(scheme3_summary["source_scheme2_table"])
    scheme2 = json.loads((scheme2_table / "Q3_运行摘要.json").read_text(encoding="utf-8"))
    for name, metrics in (("方案一", scheme2.get("baseline_metrics")),
                          ("方案二", scheme2.get("final_metrics"))):
        if metrics:
            rows.append({"方案": name, "验收": "历史PASS", **metrics,
                         "总架次": int(metrics["transport_sorties"]) +
                                  int(metrics["relay_sorties"])})
    source_metrics = scheme3_summary["final_metrics"]
    rows.append({"方案": "方案三-少架次", "验收": "PASS", **source_metrics,
                 "总架次": int(source_metrics["transport_sorties"]) +
                          int(source_metrics["relay_sorties"])})
    for path in sorted(origin.parent.glob("q3_opt3_*_lateness_table"), reverse=True)[:1]:
        if (path / "Q3_READY.txt").is_file():
            summary = json.loads((path / "Q3_运行摘要.json").read_text(encoding="utf-8"))
            if summary.get("status") == "PASS":
                metrics = summary["final_metrics"]
                rows.append({"方案": "方案三-及时性", "验收": "历史PASS", **metrics,
                             "总架次": int(metrics["transport_sorties"]) +
                                      int(metrics["relay_sorties"])})
    for entry in archive:
        if entry.origin is not None:
            continue
        else:
            name = "方案四-" + entry.stage
        rows.append({"方案": name, "验收": "PASS", **entry.metrics,
                     "总架次": int(entry.metrics["transport_sorties"]) +
                              int(entry.metrics["relay_sorties"])})
    return rows


def _save_extra_figures(figure_dir: Path, table_dir: Path,
                        comparison: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    waits, windows, critical = _diagnose(table_dir)
    meaningful = sorted((r for r in waits if r["后继"] in critical),
                        key=lambda r: r["可用后等待秒"], reverse=True)[:16]
    fig, ax = plt.subplots(figsize=(10, max(4, .4 * len(meaningful) + 1.8)))
    if meaningful:
        labels = [f"{r['前驱']}→{r['后继']} {r['约束']}" for r in meaningful]
        ax.barh(range(len(labels)), [r["可用后等待秒"] for r in meaningful],
                color="#3182bd")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.invert_yaxis()
    ax.set_xlabel("Wait after resource becomes available / s")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q3_关键链与等待.{ext}", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, max(4, .36 * len(windows) + 1.7)))
    for i, row in enumerate(windows):
        active = float(row["通信并集秒"])
        idle = float(row["窗口空闲秒"])
        ax.barh(i, active, color="#238b45", label="Covering transport" if i == 0 else "")
        ax.barh(i, idle, left=active, color="#a1d99b",
                label="Window idle" if i == 0 else "")
    ax.set_yticks(range(len(windows)), [r["中继架次"] for r in windows])
    ax.set_xlabel("Relay service window / s")
    ax.grid(axis="x", alpha=.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q3_中继窗口利用.{ext}", dpi=190)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.8, 6))
    for row in comparison:
        label = ({"方案一": "S1", "方案二": "S2",
                  "方案三-少架次": "S3-few", "方案三-及时性": "S3-late"}
                 .get(row["方案"], "S4-" + row["方案"].split("-")[-1]))
        ax.scatter(float(row["makespan_s"]) / 3600,
                   float(row["energy_kwh"]),
                   s=55 + max(0, 33 - int(row["transport_sorties"])) * 16,
                   edgecolors="black", linewidths=.5)
        ax.annotate(f"{label} ({row['transport_sorties']}+{row['relay_sorties']})",
                    (float(row["makespan_s"]) / 3600,
                     float(row["energy_kwh"])), xytext=(5, 5),
                    textcoords="offset points", fontsize=7)
    ax.set_xlabel("Last return to O01 / h")
    ax.set_ylabel("Total transport and relay energy / kWh")
    ax.grid(alpha=.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q3_时间能耗架次权衡.{ext}", dpi=190)
    plt.close(fig)
    scheme3._save_resource_figure(figure_dir, table_dir)


def _choose_representatives(archive: list[scheme3.Accepted],
                            reference: dict[str, float | int]
                            ) -> list[tuple[str, scheme3.Accepted]]:
    metrics = (("fast", lambda e: (float(e.metrics["makespan_s"]),
                                    float(e.metrics["energy_kwh"]))),
               ("energy", lambda e: (float(e.metrics["energy_kwh"]),
                                      float(e.metrics["makespan_s"]))),
               ("lateness", lambda e: (float(e.metrics["weighted_lateness"]),
                                        float(e.metrics["makespan_s"]))),
               ("balanced", lambda e: (sum(float(e.metrics[k]) /
                                            max(1., float(reference[k])) for k in
                                            ("makespan_s", "energy_kwh", "weighted_lateness")),
                                        int(e.metrics["transport_sorties"]) +
                                        int(e.metrics["relay_sorties"]))))
    selected = []
    for goal, key in metrics:
        item = min(archive, key=key)
        if not any(scheme3._same_metrics(item.metrics, old.metrics)
                   for _, old in selected):
            selected.append((goal, item))
    return selected


def _conditional_bounds(origin: Path, ctx: Context,
                        relay: dict[str, Any]) -> dict[str, Any]:
    trows = base.read_rows(origin / "Q3_运输明细.csv")
    rrows = base.read_rows(origin / "Q3_中继资源时间线.csv")
    counts = Counter(uav.type_id for uav in ctx.uavs)
    type_work = defaultdict(float)
    max_duration = 0.0
    for row in trows:
        duration = float(row["return_s"]) - float(row["start_s"])
        type_work[row["type_id"]] += duration
        max_duration = max(max_duration, duration)
    transport_bound = max([max_duration] + [work / counts[kind]
                                          for kind, work in type_work.items()])
    relay_count = len(relay["uav_ids"])
    relay_work = sum(float(row["return_s"]) - float(row["start_s"])
                     for row in rrows)
    relay_bound = (relay_work + max(0, len(rrows) - relay_count) *
                   float(relay["params"]["turnaround_s"])) / relay_count
    return {"fixed_source_transport_workload_bound_s": transport_bound,
            "fixed_source_relay_workload_bound_s": relay_bound,
            "scope": "只对方案三原任务时长、机型、两类实体机库存成立；"
                     "改站点、改组批或机型后不可作为新方案下界",
            "status": "CONDITIONAL_ONLY"}


def _write_outputs(args: argparse.Namespace, origin: Path,
                   scheme3_summary: dict[str, Any], data_run: Path,
                   relay: dict[str, Any], ctx: Context,
                   geometry: scheme3.GeometryPool,
                   archive: list[scheme3.Accepted], attempts: list[dict[str, Any]],
                   station_log: list[dict[str, Any]], phase_log: list[dict[str, Any]],
                   reference: dict[str, float | int], elapsed: float) -> dict[str, Any]:
    stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    args.output_root.mkdir(parents=True, exist_ok=True)
    comparison = _comparison(origin, scheme3_summary, archive)
    conditional = _conditional_bounds(origin, ctx, relay)
    reps = _choose_representatives(archive, reference)
    outputs = []
    archive_rows = [{"编号": i, "阶段": e.stage, "档位": e.grade,
                     **e.metrics, "总架次": int(e.metrics["transport_sorties"]) +
                                         int(e.metrics["relay_sorties"])}
                    for i, e in enumerate(archive, 1)]
    for goal, entry in reps:
        name = f"q3_opt4_{stamp}_{goal}"
        table = args.output_root / (name + "_table")
        figure = args.output_root / (name + "_figure")
        table.mkdir(parents=True, exist_ok=False)
        if entry.state is None:
            source = entry.origin or origin
            for path in source.iterdir():
                if (path.name.startswith("Q3_") and path.suffix == ".csv"
                        and not any(s in path.name for s in
                                    ("方案比较", "非支配", "少架次", "条件下界"))):
                    shutil.copy2(path, table / path.name)
            blind = source / "Q3_待保障区间汇总.json"
            if blind.is_file():
                shutil.copy2(blind, table / blind.name)
        else:
            base._save_joint_tables(table, entry.state, relay)
        validation = base.validate_official_tables(table, data_run,
                                                    geometry.links, geometry.dem)
        base._save_verification(table, validation, scheme_check_id="SCHEME4_STATUS")
        waits, windows, critical = _diagnose(table)
        base.write_rows(table / "Q3_方案比较.csv", comparison,
                        ["方案", "验收", *scheme3.METRIC_KEYS, "总架次"])
        base.write_rows(table / "Q3_非支配候选.csv", archive_rows,
                        ["编号", "阶段", "档位", *scheme3.METRIC_KEYS, "总架次"])
        base.write_rows(table / "Q3_关键链等待.csv", waits,
                        ["前驱", "后继", "约束", "可用后等待秒", "后继开始秒"])
        base.write_rows(table / "Q3_中继窗口利用.csv", windows,
                        ["中继架次", "建链完成秒", "服务结束秒", "服务窗口秒",
                         "通信并集秒", "窗口空闲秒", "关联运输架次", "出入站秒", "返航秒"])
        base.write_rows(table / "Q3_站点筛选.csv", station_log,
                        ["站点", "来源", "覆盖采样数", "往返秒", "往返能耗kWh",
                         "最大服务秒", "回传余量dB"])
        shift = next((row for row in phase_log
                      if row.get("stage") == "time_shift_diagnostic"), {})
        base.write_rows(table / "Q3_时序前移提示.csv", shift.get("task_starts", []),
                        ["任务", "原开始秒", "前移提示秒", "可前移秒"])
        base.json_dump(table / "Q3_搜索轨迹.json", attempts)
        base.json_dump(table / "Q3_分阶段对照.json", phase_log)
        base.json_dump(table / "Q3_关键任务链.json", critical)
        base.json_dump(table / "Q3_条件下界.json", conditional)
        summary = {
            "status": validation["status"],
            "method": "Q3 scheme 4: critical-chain guided joint transport-relay compression",
            "source_scheme3_table": str(origin.resolve()),
            "source": scheme3_summary["source"],
            "source_manifest_sha256": scheme3_summary["source_manifest_sha256"],
            "code_sha256": {name: base.sha256(CODE_DIR / name) for name in
                            ("q3_4_optimize.py", "q3_3_optimize.py",
                             "q3_2_optimize.py", "q3_1_optimize.py",
                             "q2_1_optimize.py")},
            "goal": goal, "stage": entry.stage, "grade": entry.grade,
            "reference_metrics": reference, "final_metrics": entry.metrics,
            "improved_makespan_from_source": float(entry.metrics["makespan_s"]) <
                                             float(reference["makespan_s"]) - 1e-4,
            "hard_deadline_min_slack_s": _min_hard_slack(table),
            "transport_status": validation["transport_status"],
            "communication_status": validation["communication"]["status"],
            "relay_resource_status": validation["relay_resources"]["status"],
            "global_optimality_proven": False,
            "candidate_site_count": len(geometry.sites),
            "candidate_stats": geometry.stats,
            "search_wall_s": elapsed,
            "table_dir": str(table.resolve()), "figure_dir": None,
        }
        if validation["status"] == "PASS":
            if entry.state is not None:
                base._save_figures(figure, entry.state.slices, entry.state.assigned,
                                   entry.state.missions, scheme_label=f"Q3 scheme 4 {goal}")
            else:
                source_summary = json.loads(
                    ((entry.origin or origin) / "Q3_运行摘要.json").read_text(encoding="utf-8"))
                old_name = source_summary.get("figure_dir")
                old = Path(old_name) if old_name else None
                if old is not None and old.is_dir():
                    shutil.copytree(old, figure)
                else:
                    figure.mkdir()
            _save_extra_figures(figure, table, comparison)
            (table / "Q3_READY.txt").write_text(
                "Q3 scheme 4 independently verified PASS\n"
                f"source_manifest_sha256={summary['source_manifest_sha256']}\n",
                encoding="utf-8")
            summary["figure_dir"] = str(figure.resolve())
        base.json_dump(table / "Q3_运行摘要.json", summary)
        outputs.append(summary)
    index = {"status": "PASS" if outputs and all(r["status"] == "PASS" for r in outputs)
             else "FAIL", "source_scheme3_table": str(origin.resolve()),
             "reference_metrics": reference, "representatives": outputs,
             "stage_comparison": phase_log, "search_wall_s": elapsed,
             "global_optimality_proven": False}
    base.json_dump(args.output_root / f"q3_opt4_{stamp}_index.json", index)
    return index


def _hints(entry: scheme3.Accepted, geometry: scheme3.GeometryPool,
           origin_hints: dict[str, Any], origin_sites: list[base.RelayCandidate]
           ) -> dict[str, Any]:
    if entry.solution is None:
        hints = dict(origin_hints)
        site_by_id = {site.candidate_id: i for i, site in enumerate(geometry.sites)}
        hints["relays"] = [
            (slot, site_by_id[origin_sites[old_j].candidate_id], uav, unit, ready, end)
            for slot, old_j, uav, unit, ready, end in origin_hints["relays"]
            if origin_sites[old_j].candidate_id in site_by_id]
        return hints
    hints = previous._solution_hints(entry.solution)
    site_by_id = {site.candidate_id: i for i, site in enumerate(geometry.sites)}
    used = set(entry.solution["block_relay"].values())
    old = sorted((row for row in entry.solution["relays"] if row[0] in used),
                 key=lambda row: row[4])
    mission_by_slot = {row[0]: mission for row, mission in
                       zip(old, entry.state.missions)}
    hints["relays"] = [
        (slot, site_by_id[mission_by_slot[slot].candidate.candidate_id],
         uav, unit, ready, end)
        for slot, _j, uav, unit, ready, end in hints["relays"]
        if slot in mission_by_slot and
        mission_by_slot[slot].candidate.candidate_id in site_by_id]
    return hints


def _alternative_types(ctx: Context, geometry: scheme3.GeometryPool,
                       current: list[base.JointChoice], released: set[int],
                       deadline: float) -> list[base.JointChoice]:
    out = []
    known = {scheme3._route_key(c) for c in current}
    for i in sorted(released):
        if time.monotonic() >= deadline:
            break
        route = current[i].task.route
        for kind, option in ctx.options(route).items():
            if (route.visits, kind) in known or base._latest_hard_start(
                    ctx, route, option) < -base.EPS:
                continue
            choice = geometry.certify(route, kind, option)
            if choice is not None:
                out.append(choice)
                known.add((route.visits, kind))
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    total_started = time.monotonic()
    origin = (args.scheme3_table or _source_table(
        CODE_DIR / "3_outputs" / "3_optimize")).resolve()
    source = json.loads((origin / "Q3_运行摘要.json").read_text(encoding="utf-8"))
    if source.get("status") != "PASS" or not (origin / "Q3_READY.txt").is_file():
        raise ValueError("方案三起点必须具有 PASS 摘要与 Q3_READY.txt")
    data_run = (args.data_run or Path(source["source"]["data_run"])).resolve()
    relay, dem, links, meta = base._scene(data_run)
    if meta["source_manifest_sha256"] != source["source_manifest_sha256"]:
        raise ValueError("数据清单与方案三起点不一致")
    print("Q3方案四：独立复核方案三原表……", flush=True)
    source_check = base.validate_official_tables(origin, data_run, links, dem)
    if source_check["status"] != "PASS":
        raise ValueError("方案三原表独立复核失败")
    ctx = Context(data_run)
    positions = base._node_positions(data_run)
    relay_rows = base.read_rows(origin / "Q3_中继架次.csv")
    sites = previous._load_sites(
        origin, relay_rows, data_run, relay, dem, links, ctx, positions,
        args.sample_step_s, args.candidate_spacing_m, args.max_sites)
    geometry = scheme3.GeometryPool(
        ctx, relay, dem, links, positions, sites, args.sample_step_s,
        args.candidate_spacing_m, args.max_sites, args.max_site_points)
    incumbent, origin_hints = previous._incumbent_choices(
        ctx, origin, positions, links, args.sample_step_s,
        geometry.sites, geometry.access_cache)
    shifted_hints, shift_diagnostic = _earliest_shift_hint(
        origin, incumbent, origin_hints)
    print(f"Q3方案四：固定任务时序诊断 {shift_diagnostic['original_c_s']:.1f}s → "
          f"{shift_diagnostic['earliest_fixed_structure_c_s']:.1f}s；"
          "仅作提示，须经正式求解与独立验收", flush=True)
    reference = {key: source["final_metrics"][key] for key in scheme3.METRIC_KEYS}
    source_entry = scheme3.Accepted(reference, None, None, origin, "S0方案三", "REFERENCE")
    archive = [source_entry]
    # 外部已验收的及时性解仅进入对照/档案，不被冒充为严格档可行提示。
    late_source = next((p for p in sorted(origin.parent.glob("q3_opt3_*_lateness_table"),
                                         reverse=True) if (p / "Q3_READY.txt").is_file()), None)
    if late_source is not None:
        summary = json.loads((late_source / "Q3_运行摘要.json").read_text(encoding="utf-8"))
        if (summary.get("status") == "PASS" and
                summary["source_manifest_sha256"] == source["source_manifest_sha256"] and
                base.validate_official_tables(late_source, data_run, links, dem)["status"] == "PASS"):
            scheme3._archive_add(archive, scheme3.Accepted(
                _metrics_from_summary(late_source), None, None, late_source,
                "方案三及时性对照", "REFERENCE"))
    attempts: list[dict[str, Any]] = []
    station_log: list[dict[str, Any]] = []
    phase_log: list[dict[str, Any]] = [{"stage": "S0", "status": "PASS",
                                      "metrics": reference,
                                      "table": str(origin)},
                                     {"stage": "time_shift_diagnostic",
                                      **shift_diagnostic}]
    started = time.monotonic()
    deadlines = {"fixed_used": started + .175 * args.time_budget_s,
                 "fixed_all": started + .35 * args.time_budget_s,
                 "sites": started + .70 * args.time_budget_s,
                 "rebatch": started + .90 * args.time_budget_s,
                 "tradeoff": started + args.time_budget_s}
    print(f"Q3方案四：站点={len(geometry.sites)}，起点="
          f"{reference['transport_sorties']}+{reference['relay_sorties']}架次，"
          f"C={reference['makespan_s']:.1f}s，预算={args.time_budget_s:.0f}s",
          flush=True)

    def best_fast() -> scheme3.Accepted:
        return min([source_entry, *archive],
                   key=lambda e: (float(e.metrics["makespan_s"]),
                                  float(e.metrics["energy_kwh"])))

    def solve(stage: str, grade: str, goal: str,
              geo: scheme3.GeometryPool, pool: list[base.JointChoice],
              seed: scheme3.Accepted, fixed: dict | None = None,
              cap: float | None = None, margin: float = 0.,
              require_all: bool = False, max_seconds: float | None = None
              ) -> scheme3.Accepted | None:
        remaining = deadlines[stage] - time.monotonic()
        limit = min(remaining, max_seconds or args.global_seconds)
        if limit < args.min_solve_s:
            phase_log.append({"stage": stage, "status": "NOT_RUN_BUDGET",
                              "goal": goal, "grade": grade})
            return None
        hints = _hints(seed, geo, shifted_hints if seed is source_entry
                       else origin_hints, sites)
        candidate, record = _attempt(
            stage, grade, goal, ctx, relay, geo, pool,
            hints, fixed, reference,
            data_run, args, limit, cap, margin, require_all)
        attempts.append(record)
        phase_log.append({"stage": stage, "goal": goal, "grade": grade,
                          "status": record["solver_status"],
                          "validation": record["validation"],
                          "within_actual_caps": record.get("within_actual_caps"),
                          "metrics": record.get("metrics"),
                          "best_bound": record.get("best_bound"),
                          "hard_margin_s": margin,
                          "route_options": len(pool),
                          "site_count": len(geo.sites)})
        if candidate is not None:
            kept = scheme3._archive_add(archive, candidate)
            print(f"Q3方案四：{stage} 验收PASS，{candidate.metrics}，"
                  f"{'进入' if kept else '未进入'}非支配档案", flush=True)
        else:
            print(f"Q3方案四：{stage} {record['solver_status']}，"
                  f"验收={record['validation']}，档位核对="
                  f"{record.get('within_actual_caps', '-')}", flush=True)
        if (int(record.get("selected_relay_sorties", 0)) >= args.relay_slots - 1
                and args.relay_slots < args.relay_slots_max):
            before = args.relay_slots
            args.relay_slots = min(args.relay_slots_max, before + 4)
            print(f"Q3方案四：中继任务槽接近用满，{before}→{args.relay_slots}",
                  flush=True)
        return candidate

    try:
        # A1: 真正放开全部24批的开始时刻、实体资源与中继窗口，仅用原实际8站点。
        used_ids = {sites[j].candidate_id for _, j, *_ in origin_hints["relays"]}
        used_sites = [site for site in sites if site.candidate_id in used_ids]
        used_geo = scheme3.GeometryPool(
            ctx, relay, dem, links, positions, used_sites, args.sample_step_s,
            args.candidate_spacing_m, args.max_sites, args.max_site_points)
        fixed_used = _rebuild(used_geo, incumbent)
        first = solve("fixed_used", "strict", "fast", used_geo, fixed_used,
                      source_entry, require_all=True, max_seconds=args.global_seconds)
        # A2: 同一24批，允许28个已认证候选站点及新的服务窗口。
        second = solve("fixed_all", "strict", "fast", geometry, incumbent,
                       source_entry, require_all=True, max_seconds=args.global_seconds)
        if first is None and second is None:
            solve("fixed_all", "extended", "fast", geometry, incumbent,
                  source_entry, require_all=True, max_seconds=args.global_seconds)

        # B: 即使旧路线已可覆盖，也围绕关键盲段主动增加共同覆盖站点。
        seed = best_fast()
        current, mapping, diagnostic = _batch_positions(seed, origin, incumbent, relay)
        critical = {mapping[name] for name in diagnostic[2] if name in mapping}
        if not critical:
            critical = set(range(max(0, len(current) - 8), len(current)))
        station_time = min(deadlines["sites"],
                           time.monotonic() + args.station_seconds)
        print(f"Q3方案四：主动站点搜索，关键运输任务={len(critical)}，"
              f"当前站点={len(geometry.sites)}", flush=True)
        added, rows = _proactive_sites(geometry, current, critical,
                                      station_time, args.site_additions)
        station_log.extend(rows)
        if added:
            current = _rebuild(geometry, current)
            solve("sites", "strict", "fast", geometry, current, seed,
                  require_all=True, max_seconds=args.global_seconds)
        else:
            phase_log.append({"stage": "sites", "status": "NO_NEW_CERTIFIED_SITE"})

        # C: 从末端返回任务沿实体机、电池/组件与共享通信服务反向释放。
        for number in range(args.rebatch_rounds):
            if deadlines["rebatch"] - time.monotonic() < args.min_solve_s:
                break
            seed = best_fast()
            current, mapping, diagnostic = _batch_positions(seed, origin, incumbent, relay)
            release_order = [mapping[name] for name in diagnostic[2]
                             if name in mapping]
            if not release_order:
                release_order = sorted(scheme3._release(
                    current, number, min(10, len(current))))
            if number % 2:
                release_order.extend(sorted(scheme3._release(
                    current, number, min(4, len(current)))))
            release = set(list(dict.fromkeys(release_order))[:args.release_limit])
            proposal_deadline = min(deadlines["rebatch"],
                                    time.monotonic() + args.candidate_seconds)
            bundles, generated = scheme3._propose_routes(
                ctx, current, release, args.max_stops,
                args.max_packages, proposal_deadline, rotation=number)
            site_count_before = len(geometry.sites)
            fresh, certified = scheme3._certify_packages(
                ctx, bundles, geometry, current, args.max_new_choices,
                min(deadlines["rebatch"], time.monotonic() + args.certify_seconds))
            for site in geometry.sites[site_count_before:]:
                station_log.append({
                    "站点": site.candidate_id, "来源": "新组批盲段覆盖补点",
                    "往返秒": site.outward_s + site.homeward_s,
                    "往返能耗kWh": site.travel_energy_kwh,
                    "最大服务秒": site.max_service_s,
                    "回传余量dB": site.backhaul_margin_db})
            current = _rebuild(geometry, current)
            fresh = _rebuild(geometry, fresh)
            alts = _alternative_types(
                ctx, geometry, current, release,
                min(deadlines["rebatch"], time.monotonic() + args.certify_seconds))
            pool_by_key = {scheme3._route_key(c): c for c in
                           [*current, *fresh, *alts]}
            fixed = scheme3._fixed_outside(seed.solution, release)
            phase_log.append({"stage": f"rebatch_candidates_{number+1}",
                              "released": len(release), "generated": generated,
                              "certified": certified, "type_alternatives": len(alts),
                              "pool_size": len(pool_by_key)})
            solve("rebatch", ("strict", "small", "extended")[number % 3],
                  "fast", geometry, list(pool_by_key.values()), seed, fixed,
                  max_seconds=args.local_seconds)
            # 新的快解立即重新放开全部选中路线的时间与中继资源。
            if (number == 0 or (attempts and attempts[-1].get("within_actual_caps")
                                and best_fast().solution is not None)):
                newest = best_fast()
                if newest.solution is not None and deadlines["rebatch"] - time.monotonic() >= args.min_solve_s:
                    chosen = _rebuild(geometry, scheme3._selected(newest.solution))
                    solve("rebatch", "extended", "fast", geometry, chosen,
                          newest, require_all=True, max_seconds=args.local_seconds)

        # D: 相同原题硬约束下的能耗、软时限及额外硬时限余量敏感性。
        for goal, grade, margin in (("energy", "strict", 0.),
                                    ("lateness", "small", 0.),
                                    ("fast", "extended", 30.),
                                    ("fast", "extended", 60.)):
            if deadlines["tradeoff"] - time.monotonic() < args.min_solve_s:
                break
            seed = best_fast()
            current = _rebuild(geometry,
                               scheme3._selected(seed.solution) if seed.solution else incumbent)
            cap = min(float(reference["makespan_s"]),
                      float(seed.metrics["makespan_s"])) if goal != "fast" else None
            solve("tradeoff", grade, goal, geometry, current, seed,
                  cap=cap, margin=margin, require_all=True,
                  max_seconds=args.local_seconds)
    except KeyboardInterrupt:
        print("Q3方案四：已中断，保存此前独立验收通过的候选。", flush=True)
    return _write_outputs(args, origin, source, data_run, relay, ctx, geometry,
                          archive, attempts, station_log, phase_log,
                          reference, time.monotonic() - total_started)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme3-table", type=Path, default=None,
                        help="方案三已验收少架次 *_table；默认自动读取最新索引")
    parser.add_argument("--data-run", type=Path, default=None)
    parser.add_argument("--output-root", type=Path,
                        default=CODE_DIR / "3_outputs" / "4_optimize")
    parser.add_argument("--time-budget-s", type=float, default=1800,
                        help="搜索预算秒；物理建模和最终出表出图另行计时")
    parser.add_argument("--global-seconds", type=float, default=300)
    parser.add_argument("--local-seconds", type=float, default=65)
    parser.add_argument("--station-seconds", type=float, default=120)
    parser.add_argument("--candidate-seconds", type=float, default=20)
    parser.add_argument("--certify-seconds", type=float, default=28)
    parser.add_argument("--min-solve-s", type=float, default=8)
    parser.add_argument("--rebatch-rounds", type=int, default=5)
    parser.add_argument("--release-limit", type=int, default=12)
    parser.add_argument("--max-packages", type=int, default=26)
    parser.add_argument("--max-new-choices", type=int, default=70)
    parser.add_argument("--max-stops", type=int, default=2)
    parser.add_argument("--relay-slots", type=int, default=18)
    parser.add_argument("--relay-slots-max", type=int, default=26)
    parser.add_argument("--max-sites", type=int, default=64)
    parser.add_argument("--site-additions", type=int, default=16)
    parser.add_argument("--max-site-points", type=int, default=1600)
    parser.add_argument("--horizon-s", type=int, default=21600)
    parser.add_argument("--sample-step-s", type=float, default=20)
    parser.add_argument("--candidate-spacing-m", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args(argv)
    positive = (args.time_budget_s, args.global_seconds, args.local_seconds,
                args.station_seconds, args.candidate_seconds, args.certify_seconds,
                args.min_solve_s, args.sample_step_s)
    integers = (args.rebatch_rounds, args.release_limit, args.max_packages,
                args.max_new_choices, args.max_stops, args.relay_slots,
                args.relay_slots_max,
                args.max_sites, args.site_additions, args.max_site_points,
                args.horizon_s, args.candidate_spacing_m, args.workers)
    if min(positive) <= 0 or min(integers) < 1:
        parser.error("时间、候选数量、站点数、时域和并行数必须为正")
    if args.max_sites < args.site_additions:
        parser.error("max-sites 必须不小于 site-additions")
    if args.relay_slots_max < args.relay_slots:
        parser.error("relay-slots-max 不得小于 relay-slots")
    result = run(args)
    print(json.dumps({"status": result["status"],
                      "search_wall_s": result["search_wall_s"],
                      "representatives": [{"goal": x["goal"],
                                           "metrics": x["final_metrics"],
                                           "table_dir": x["table_dir"],
                                           "figure_dir": x["figure_dir"]}
                                          for x in result["representatives"]]},
                     ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
