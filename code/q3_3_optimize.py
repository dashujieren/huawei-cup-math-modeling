"""问题三方案三：动态重新组批及运输—中继多目标联合排程。

先读取并重新验收方案二；完整结果只在独立复核通过后写入 READY。
有限候选池和限时 CP-SAT 的 OPTIMAL 只适用于本次限定模型。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

import q3_1_optimize as base
import q3_2_optimize as previous
from q2_1_optimize import Context


CODE_DIR = Path(__file__).resolve().parent
METRIC_KEYS = ("transport_sorties", "relay_sorties", "makespan_s",
               "energy_kwh", "weighted_lateness")
GOALS = ("sorties", "makespan", "energy", "lateness")
GOAL_LABEL = {"sorties": "少架次", "makespan": "快返航",
              "energy": "低能耗", "lateness": "及时性"}
CP_OBJECTIVE = {"sorties": "scheme2_sorties",
                "makespan": "scheme2_makespan",
                "energy": "scheme2_energy",
                "lateness": "scheme2_lateness"}
EPSILON_GRADES = {
    "G0": (1.00, 1.00, 1.00),
    "G1": (1.05, 1.05, 1.10),
    "G2": (1.10, 1.10, 1.25),
}


@dataclass(frozen=True)
class RouteBundle:
    """一次重组必须成套提供所有替代路线，避免仅保留换箱的一半。"""

    routes: tuple[Any, ...]
    family: str
    old_count: int
    energy_gain: float
    duration_gain: float

    @property
    def key(self) -> tuple[Any, ...]:
        return tuple(sorted(route.visits for route in self.routes))


@dataclass
class Accepted:
    metrics: dict[str, float | int]
    solution: dict[str, Any] | None
    state: base.JointState | None
    origin: Path | None
    stage: str
    grade: str


def _latest_passed_run(root: Path, prefix: str) -> Path:
    for path in sorted(root.glob(prefix + "*_table"), reverse=True):
        ready = path / "Q3_READY.txt"
        summary = path / "Q3_运行摘要.json"
        if ready.is_file() and summary.is_file():
            if json.loads(summary.read_text(encoding="utf-8")).get("status") == "PASS":
                return path
    raise FileNotFoundError(f"没有已验收的 {prefix} 输出；请指定 --scheme2-table")


def _selected(solution: dict[str, Any]) -> list[base.JointChoice]:
    return [solution["pool"][i] for i, *_ in solution["routes"]]


def _route_key(choice: base.JointChoice) -> tuple[Any, str]:
    return choice.task.route.visits, choice.kind


def _rounded_caps(reference: dict[str, float | int], goal: str,
                  grade: str, target: int | None) -> dict[str, int]:
    c_factor, e_factor, l_factor = EPSILON_GRADES[grade]
    caps = {
        "makespan_s": math.ceil(float(reference["makespan_s"]) * c_factor + 2),
        "energy_j": math.ceil(float(reference["energy_kwh"]) * e_factor * 3_600_000 + 5000),
        "weighted_lateness": math.ceil(
            float(reference["weighted_lateness"]) * l_factor + 5000),
    }
    caps.pop({"sorties": "transport_sorties", "makespan": "makespan_s",
              "energy": "energy_j", "lateness": "weighted_lateness"}[goal], None)
    if target is not None:
        caps["transport_sorties"] = target
    return caps


def _within_grade(metrics: dict[str, float | int],
                  reference: dict[str, float | int], goal: str,
                  grade: str, target: int | None) -> bool:
    c_factor, e_factor, l_factor = EPSILON_GRADES[grade]
    limits = {
        "makespan": ("makespan_s", c_factor, 1e-4),
        "energy": ("energy_kwh", e_factor, 1e-6),
        "lateness": ("weighted_lateness", l_factor, 1e-3),
    }
    if target is not None and metrics["transport_sorties"] > target:
        return False
    return all(float(metrics[key]) <= float(reference[key]) * factor + tol
               for metric, (key, factor, tol) in limits.items() if metric != goal)


def _dominates(a: dict[str, float | int], b: dict[str, float | int]) -> bool:
    tol = {"transport_sorties": 0, "relay_sorties": 0,
           "makespan_s": 1e-4, "energy_kwh": 1e-6,
           "weighted_lateness": 1e-3}
    return (all(float(a[key]) <= float(b[key]) + tol[key] for key in METRIC_KEYS)
            and any(float(a[key]) < float(b[key]) - tol[key]
                    for key in METRIC_KEYS))


def _same_metrics(a: dict[str, float | int], b: dict[str, float | int]) -> bool:
    return (int(a["transport_sorties"]) == int(b["transport_sorties"])
            and int(a["relay_sorties"]) == int(b["relay_sorties"])
            and abs(float(a["makespan_s"]) - float(b["makespan_s"])) < 1e-4
            and abs(float(a["energy_kwh"]) - float(b["energy_kwh"])) < 1e-6
            and abs(float(a["weighted_lateness"]) - float(b["weighted_lateness"])) < 1e-3)


def _archive_add(archive: list[Accepted], item: Accepted) -> bool:
    if any(_same_metrics(old.metrics, item.metrics)
           or _dominates(old.metrics, item.metrics) for old in archive):
        return False
    archive[:] = [old for old in archive if not _dominates(item.metrics, old.metrics)]
    archive.append(item)
    return True


def _route_forms(ctx: Context, box_ids: list[str],
                 max_stops: int) -> list[Any]:
    by_zone: dict[str, list[str]] = {}
    for box_id in dict.fromkeys(box_ids):
        by_zone.setdefault(str(ctx.boxes[box_id]["zone_id"]), []).append(box_id)
    if not 1 <= len(by_zone) <= max_stops:
        return []
    return [ctx.route([(zone, by_zone[zone]) for zone in order])
            for order in itertools.permutations(sorted(by_zone))]


def _new_bundle(ctx: Context, routes: tuple[Any, ...], family: str,
                old: tuple[base.JointChoice, ...],
                seen: set[tuple[Any, ...]]) -> RouteBundle | None:
    if len(routes) >= len(old) and {r.visits for r in routes} == {
            c.task.route.visits for c in old}:
        return None
    box_ids = [bid for route in routes for bid in route.box_ids]
    source_ids = [bid for choice in old for bid in choice.task.route.box_ids]
    split = family == "split_1_to_2" and len(old) == 1 and len(routes) == 2
    if (len(box_ids) != len(set(box_ids)) or set(box_ids) != set(source_ids)
            or (len(routes) > len(old) and not split)):
        return None
    key = tuple(sorted(route.visits for route in routes))
    if key in seen:
        return None
    valid_options = [
        [option for kind, option in ctx.options(route).items()
         if base._latest_hard_start(ctx, route, option) >= -base.EPS]
        for route in routes
    ]
    if any(not options for options in valid_options):
        return None
    seen.add(key)
    new_energy = sum(min(float(o["energy_kwh"]) for o in opts)
                     for opts in valid_options)
    new_duration = (max(min(float(o["duration_s"]) for o in opts)
                        for opts in valid_options) if split else
                    sum(min(float(o["duration_s"]) for o in opts)
                        for opts in valid_options))
    return RouteBundle(routes, family, len(old),
                       sum(float(c.option["energy_kwh"]) for c in old) - new_energy,
                       sum(float(c.option["duration_s"]) for c in old) - new_duration)


def _propose_routes(ctx: Context, current: list[base.JointChoice],
                    released: set[int], max_stops: int,
                    package_limit: int, deadline: float,
                    rotation: int = 0, priority: str = "sorties") -> tuple[
                        list[RouteBundle], dict[str, int]]:
    """按路线包生成，先去重再按四种用途分配认证预算。"""
    produced: list[RouteBundle] = []
    seen: set[tuple[Any, ...]] = set()
    reasons = {"physical_or_deadline": 0, "duplicate": 0,
               "generated": 0, "budget_truncated": 0}

    def add(routes: tuple[Any, ...], family: str,
            old: tuple[base.JointChoice, ...]) -> None:
        reasons["generated"] += 1
        prior = len(seen)
        bundle = _new_bundle(ctx, routes, family, old, seen)
        if bundle is not None:
            produced.append(bundle)
        else:
            reasons["duplicate" if len(seen) == prior and tuple(sorted(
                r.visits for r in routes)) in seen else "physical_or_deadline"] += 1

    pairs = list(itertools.combinations(sorted(released), 2))
    pairs.sort(key=lambda pair: (
        not bool(set(current[pair[0]].task.route.zones)
                 & set(current[pair[1]].task.route.zones)),
        sum(len(current[i].task.route.zones) for i in pair), pair))
    if len(pairs) > max(12, package_limit):
        offset = (rotation * max(1, package_limit // 2)) % len(pairs)
        pairs = (pairs[offset:] + pairs[:offset])[:max(12, package_limit)]
    for i, j in pairs:
        if time.monotonic() >= deadline:
            break
        a, b = current[i], current[j]
        old = (a, b)
        a_ids, b_ids = list(a.task.route.box_ids), list(b.task.route.box_ids)
        for route in _route_forms(ctx, a_ids + b_ids, max_stops):
            add((route,), "merge_2_to_1", old)
        for donor_ids, receiver_ids in ((a_ids, b_ids), (b_ids, a_ids)):
            if time.monotonic() >= deadline:
                break
            if len(donor_ids) < 2:
                continue
            for count in (1, 2, 3):
                if len(donor_ids) <= count:
                    continue
                for moved in itertools.combinations(donor_ids, count):
                    if time.monotonic() >= deadline:
                        break
                    remain = [bid for bid in donor_ids if bid not in moved]
                    for left in _route_forms(ctx, remain, max_stops):
                        for right in _route_forms(
                                ctx, receiver_ids + list(moved), max_stops):
                            add((left, right), "move_boxes", old)
        for left_id in a_ids:
            if time.monotonic() >= deadline:
                break
            for right_id in b_ids:
                for left in _route_forms(
                        ctx, [bid for bid in a_ids if bid != left_id] + [right_id],
                        max_stops):
                    for right in _route_forms(
                            ctx, [bid for bid in b_ids if bid != right_id] + [left_id],
                            max_stops):
                        add((left, right), "swap_boxes", old)

    for i in sorted(released):
        if time.monotonic() >= deadline:
            break
        original = current[i]
        ids = list(original.task.route.box_ids)
        if len(ids) < 2:
            continue
        for count in range(1, min(len(ids) // 2, 3) + 1):
            for subset in itertools.combinations(ids, count):
                if time.monotonic() >= deadline:
                    break
                rest = [bid for bid in ids if bid not in subset]
                for left in _route_forms(ctx, list(subset), max_stops):
                    for right in _route_forms(ctx, rest, max_stops):
                        add((left, right), "split_1_to_2", (original,))

    triples = list(itertools.combinations(sorted(released), 3))
    if len(triples) > max(8, package_limit // 2):
        offset = (rotation * max(1, package_limit // 4)) % len(triples)
        triples = (triples[offset:] + triples[:offset])[
            :max(8, package_limit // 2)]
    for triple in triples:
        if time.monotonic() >= deadline:
            break
        old = tuple(current[i] for i in triple)
        ids = list(dict.fromkeys(bid for c in old for bid in c.task.route.box_ids))
        if len(ids) > 14 or len(ids) < 3:
            continue
        for route in _route_forms(ctx, ids, max_stops):
            add((route,), "merge_3_to_1", old)
        # 三批合两批：小规模枚举互补箱集合，两条新路线始终成套进入候选。
        for count in range(1, min(len(ids) // 2, 4) + 1):
            for subset in itertools.combinations(ids, count):
                if time.monotonic() >= deadline:
                    break
                complement = [bid for bid in ids if bid not in subset]
                for first in _route_forms(ctx, list(subset), max_stops):
                    for second in _route_forms(ctx, complement, max_stops):
                        add((first, second), "merge_3_to_2", old)

    buckets = {
        "few": sorted(produced, key=lambda x: (
            len(x.routes) - x.old_count, -x.energy_gain, -x.duration_gain)),
        "energy": sorted(produced, key=lambda x: (
            -x.energy_gain, len(x.routes) - x.old_count, -x.duration_gain)),
        "fast": sorted(produced, key=lambda x: (
            -x.duration_gain, len(x.routes) - x.old_count, -x.energy_gain)),
        "coverage": sorted(produced, key=lambda x: (
            sum(len(r.zones) - 1 for r in x.routes),
            len(x.routes) - x.old_count, -x.energy_gain)),
        "parallel": sorted(
            (x for x in produced if x.family == "split_1_to_2"),
            key=lambda x: (-x.duration_gain, -x.energy_gain)),
    }
    selected: list[RouteBundle] = []
    chosen: set[tuple[Any, ...]] = set()
    quota = max(1, package_limit // len(buckets))
    if priority not in {"sorties", "makespan"}:
        raise ValueError("候选路线优先级只能是 sorties 或 makespan")
    bucket_order = (("parallel", "fast", "coverage", "energy", "few")
                    if priority == "makespan" else tuple(buckets))
    for bucket_name in bucket_order:
        ranked = buckets[bucket_name]
        added = 0
        for bundle in ranked:
            if bundle.key in chosen:
                continue
            selected.append(bundle)
            chosen.add(bundle.key)
            added += 1
            if added >= quota or len(selected) >= package_limit:
                break
        if len(selected) >= package_limit:
            break
    if len(selected) < package_limit:
        for bundle in buckets["fast" if priority == "makespan" else "few"]:
            if bundle.key not in chosen:
                selected.append(bundle)
                chosen.add(bundle.key)
            if len(selected) >= package_limit:
                break
    reasons["budget_truncated"] = len(produced) - len(selected)
    return selected, reasons


class GeometryPool:
    def __init__(self, ctx: Context, relay: dict[str, Any],
                 dem: base.DemGrid, links: base.LinkEvaluator,
                 positions: dict[str, tuple[float, float, float]],
                 sites: list[base.RelayCandidate], step_s: float,
                 spacing_m: int, max_sites: int, max_points: int):
        self.ctx, self.relay, self.dem, self.links = ctx, relay, dem, links
        self.positions, self.sites = positions, sites
        self.step_s, self.spacing_m = step_s, spacing_m
        self.max_sites, self.max_points = max_sites, max_points
        self.access_cache: dict[tuple[Any, ...], bool] = {}
        self.choice_cache: dict[tuple[Any, str, int], base.JointChoice | None] = {}
        self.uncovered: list[base.TimeSlice] = []
        self.stats: dict[str, int] = {
            "certified": 0, "uncovered_routes": 0, "new_sites": 0,
            "site_budget_reached": 0, "site_grid_exceeded": 0,
        }

    def certify(self, route: Any, kind: str,
                option: dict[str, Any]) -> base.JointChoice | None:
        key = route.visits, kind, len(self.sites)
        if key in self.choice_cache:
            return self.choice_cache[key]
        task = base.RouteTask(f"P{len(self.choice_cache)+1:06d}", route)
        tree = (cKDTree(np.asarray([site.hover for site in self.sites], dtype=float))
                if self.sites else None)
        profile = base._route_profile(
            self.ctx, task, kind, option, self.positions, self.links,
            self.step_s, self.sites, tree, self.access_cache)
        blocks = base._blind_blocks(profile, self.sites,
                                    split_on_site_change=True)
        if profile.uncovered_slices or blocks is None:
            self.stats["uncovered_routes"] += 1
            missing = set(profile.uncovered_slices)
            if not missing and blocks is None:
                missing = {item.slice_id for item in profile.slices
                           if not item.direct_available}
            self.uncovered.extend(item for item in profile.slices
                                  if item.slice_id in missing)
            self.choice_cache[key] = None
        else:
            self.stats["certified"] += 1
            self.choice_cache[key] = base.JointChoice(
                task, kind, option, profile, blocks)
        return self.choice_cache[key]

    def expand_sites(self, deadline: float, additions: int = 4) -> int:
        if not self.uncovered or time.monotonic() >= deadline:
            return 0
        if len(self.sites) >= self.max_sites:
            self.stats["site_budget_reached"] += 1
            self.uncovered.clear()
            return 0
        # 只从未覆盖航段提议站点；相同位置仍必须重新验证接入/回传/能量。
        pending = self.uncovered[:24]
        self.uncovered.clear()
        try:
            generated, _ = base.generate_candidates(
                self.dem, self.links, pending, self.relay,
                self.positions["O01"], spacing_m=self.spacing_m,
                max_points=self.max_points)
        except ValueError as exc:
            if "超出显式预算" not in str(exc):
                raise
            self.stats["site_grid_exceeded"] += 1
            return 0
        known = {site.candidate_id for site in self.sites}
        scores: list[tuple[int, float, base.RelayCandidate]] = []
        for site in generated:
            if site.candidate_id in known:
                continue
            cover = sum(base._access_certified(site, item, self.links,
                                                self.access_cache)
                        for item in pending)
            if cover:
                scores.append((cover, site.travel_energy_kwh, site))
        scores.sort(key=lambda item: (-item[0], item[1],
                                      item[2].candidate_id))
        added = 0
        for _, _, site in scores:
            if len(self.sites) >= self.max_sites or added >= additions:
                break
            self.sites.append(site)
            known.add(site.candidate_id)
            added += 1
        self.stats["new_sites"] += added
        if added:
            print(f"Q3 方案三：针对新航段增加 {added} 个认证站点，"
                  f"当前 {len(self.sites)} 个", flush=True)
        return added


def _certify_packages(ctx: Context, bundles: list[RouteBundle],
                      geometry: GeometryPool, old: list[base.JointChoice],
                      max_new_choices: int, deadline: float,
                      priority: str = "sorties"
                      ) -> tuple[list[base.JointChoice], dict[str, int]]:
    """包内所有路线都能获至少一个机型时，才把该包加入活动池。"""
    accepted: list[base.JointChoice] = []
    seen = {_route_key(choice) for choice in old}
    counts = {"packages": len(bundles), "complete": 0, "coverage_rejected": 0,
              "choice_budget": 0, "new_choices": 0}
    pending: list[RouteBundle] = []

    def certify_bundle(bundle: RouteBundle) -> list[base.JointChoice] | None:
        package: list[base.JointChoice] = []
        for route in bundle.routes:
            alternatives = []
            for kind, option in ctx.options(route).items():
                if base._latest_hard_start(ctx, route, option) < -base.EPS:
                    continue
                choice = geometry.certify(route, kind, option)
                if choice is not None:
                    alternatives.append(choice)
            if not alternatives:
                return None
            package.extend(alternatives)
        return package

    def include(package: list[base.JointChoice]) -> bool:
        fresh = [choice for choice in package
                 if _route_key(choice) not in seen]
        if len(accepted) + len(fresh) > max_new_choices:
            counts["choice_budget"] += 1
            return False
        for choice in fresh:
            seen.add(_route_key(choice))
        accepted.extend(fresh)
        counts["complete"] += 1
        return True

    if priority not in {"sorties", "makespan"}:
        raise ValueError("候选认证优先级只能是 sorties 或 makespan")
    family_rank = ({"split_1_to_2": 0, "move_boxes": 1,
                    "swap_boxes": 2, "merge_3_to_2": 3,
                    "merge_2_to_1": 4, "merge_3_to_1": 5}
                   if priority == "makespan" else
                   {"merge_3_to_1": 0, "merge_2_to_1": 0,
                    "merge_3_to_2": 1, "split_1_to_2": 2,
                    "move_boxes": 3, "swap_boxes": 4})
    ranked_bundles = sorted(
        bundles,
        key=lambda bundle: (
            family_rank.get(bundle.family, 6),
            -bundle.duration_gain if priority == "makespan" else
            -bundle.energy_gain))
    for number, bundle in enumerate(ranked_bundles, 1):
        if time.monotonic() >= deadline:
            break
        if number == 1 or number % 8 == 0:
            print(f"Q3 方案三：通信认证 {number}/{len(bundles)}，"
                  f"新增路线/机型 {len(accepted)}", flush=True)
        package = certify_bundle(bundle)
        if package is None:
            pending.append(bundle)
        else:
            include(package)
    if pending and time.monotonic() < deadline:
        if geometry.expand_sites(deadline):
            for bundle in pending:
                if time.monotonic() >= deadline:
                    break
                package = certify_bundle(bundle)
                if package is not None:
                    include(package)
                else:
                    counts["coverage_rejected"] += 1
        else:
            counts["coverage_rejected"] += len(pending)
    counts["new_choices"] = len(accepted)
    return accepted, counts


def _rank_pairs(current: list[base.JointChoice]) -> list[tuple[int, int]]:
    pairs = list(itertools.combinations(range(len(current)), 2))
    return sorted(pairs, key=lambda pair: (
        not bool(set(current[pair[0]].task.route.zones) &
                 set(current[pair[1]].task.route.zones)),
        sum(len(current[i].task.route.zones) for i in pair),
        sum(len(current[i].task.route.box_ids) for i in pair), pair))


def _release(current: list[base.JointChoice], round_no: int,
             count: int) -> set[int]:
    pairs = _rank_pairs(current)
    if not pairs:
        return set(range(len(current)))
    # 先遍历同区/邻近双批，再把相关运输任务扩大为 6、8、12 批。
    first = pairs[(round_no * 7) % len(pairs)]
    selected = set(first)
    zones = set().union(*(set(current[i].task.route.zones) for i in first))
    rest = sorted((i for i in range(len(current)) if i not in selected),
                  key=lambda i: (
                      not bool(set(current[i].task.route.zones) & zones),
                      len(current[i].task.route.box_ids),
                      -float(current[i].option["duration_s"]), i))
    selected.update(rest[:max(0, count - len(selected))])
    return selected


def _fixed_outside(solution: dict[str, Any] | None,
                   released: set[int]) -> dict[tuple[Any, str], tuple[int, str, str]] | None:
    if solution is None:
        return None
    fixed: dict[tuple[Any, str], tuple[int, str, str]] = {}
    for position, (pool_index, start, uav, battery) in enumerate(
            solution["routes"]):
        if position not in released:
            choice = solution["pool"][pool_index]
            fixed[_route_key(choice)] = (start, uav, battery)
    return fixed


def _attempt(label: str, grade: str, goal: str, target: int | None,
             reference: dict[str, float | int], ctx: Context,
             relay: dict[str, Any], geometry: GeometryPool,
             pool: list[base.JointChoice], hints: dict[str, Any],
             fixed: dict[tuple[Any, str], tuple[int, str, str]] | None,
             args: argparse.Namespace, data_run: Path,
             seconds: float) -> tuple[dict[str, Any] | None,
                                       base.JointState | None,
                                       dict[str, Any]]:
    limits = _rounded_caps(reference, goal, grade, target)
    print(f"Q3 方案三 {label}：{len(pool)} 个路线/机型、"
          f"{len(geometry.sites)} 个中继点、{args.relay_slots} 个任务槽；"
          f"目标={goal}、档位={grade}、架次上限={target or '无'}、"
          f"求解预算={seconds:.0f}s", flush=True)
    started = time.monotonic()
    solution, solver_report = base._solve_joint(
        ctx, relay, geometry.sites, pool, args.horizon_s,
        args.relay_slots, seconds, args.workers, False,
        hints=hints, fixed_partial_transport=fixed,
        objective_mode=CP_OBJECTIVE[goal],
        metric_limits=limits, explicit_relay_uavs=True,
        search_seed=args.seed + len(pool) + len(label))
    record = {"search_stage": label, "solver_stage": solver_report["stage"],
              "goal": goal, "grade": grade, "target_transport_sorties": target,
              "metric_limits": limits, **{k: v for k, v in solver_report.items()
                                           if k != "stage"},
              "elapsed_with_model_s": time.monotonic() - started,
              "validation": "NO_SOLUTION"}
    if solution is None:
        return None, None, record
    state, validation = previous._inspect_solution(
        solution, ctx, relay, geometry.sites, data_run,
        geometry.links, geometry.dem)
    record["validation"] = validation["status"]
    record["metrics"] = previous._metrics(state)
    record["within_grade"] = _within_grade(
        record["metrics"], reference, goal, grade, target)
    if validation["status"] != "PASS" or not record["within_grade"]:
        return None, None, record
    return solution, state, record


def _goal_value(entry: Accepted, goal: str) -> tuple[float, ...]:
    metrics = entry.metrics
    if goal == "sorties":
        return (float(metrics["transport_sorties"]), float(metrics["relay_sorties"]),
                float(metrics["makespan_s"]), float(metrics["energy_kwh"]))
    if goal == "makespan":
        return (float(metrics["makespan_s"]), float(metrics["energy_kwh"]),
                float(metrics["transport_sorties"]))
    if goal == "energy":
        return (float(metrics["energy_kwh"]), float(metrics["makespan_s"]),
                float(metrics["transport_sorties"]))
    return (float(metrics["weighted_lateness"]), float(metrics["makespan_s"]),
            float(metrics["energy_kwh"]))


def _save_tradeoff_figure(figure_dir: Path,
                          comparison: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6))
    figure_labels = {"方案一": "S1", "方案二": "S2",
                     "方案三-少架次": "S3-sorties",
                     "方案三-快返航": "S3-fast",
                     "方案三-低能耗": "S3-energy",
                     "方案三-及时性": "S3-timely"}
    for row in comparison:
        x, y = float(row["makespan_s"]) / 3600, float(row["energy_kwh"])
        count = int(row["transport_sorties"])
        ax.scatter(x, y, s=55 + max(0, 33 - count) * 22,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(f"{figure_labels.get(row['方案'], 'S3')} "
                    f"({count}+{row['relay_sorties']})",
                    (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=8)
    ax.set_xlabel("Last return to O01 / h")
    ax.set_ylabel("Total transport and relay energy / kWh")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q3_时间能耗架次权衡.{ext}", dpi=190)
    plt.close(fig)


def _save_resource_figure(figure_dir: Path, table_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = base.read_rows(table_dir / "Q3_运输明细.csv")
    uavs = sorted({row["uav_id"] for row in rows})
    batteries = sorted({row["battery_id"] for row in rows})
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for row in rows:
        start = float(row["start_s"]) / 3600
        returned = float(row["return_s"]) / 3600
        charged = float(row["charge_end_s"]) / 3600
        ui = uavs.index(row["uav_id"])
        bi = batteries.index(row["battery_id"])
        axes[0].broken_barh([(start, returned - start)],
                            (ui - 0.35, 0.7), facecolors="#3182bd")
        axes[1].broken_barh([(start, returned - start)],
                            (bi - 0.35, 0.7), facecolors="#41ab5d")
        if charged > returned:
            axes[1].broken_barh([(returned, charged - returned)],
                                (bi - 0.35, 0.7), facecolors="#a1d99b")
    axes[0].set_yticks(range(len(uavs)), uavs)
    axes[1].set_yticks(range(len(batteries)), batteries)
    axes[0].set_ylabel("Transport aircraft")
    axes[1].set_ylabel("Shared batteries")
    axes[1].set_xlabel("Time since start / h")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q3_运输机电池甘特.{ext}", dpi=190)
    plt.close(fig)


def _representatives(archive: list[Accepted]) -> list[tuple[str, Accepted]]:
    selected: list[tuple[str, Accepted]] = []
    for goal in GOALS:
        item = min(archive, key=lambda x: _goal_value(x, goal))
        if not any(_same_metrics(item.metrics, previous_item.metrics)
                   for _, previous_item in selected):
            selected.append((goal, item))
    return selected


def _comparison_rows(source_summary: dict[str, Any],
                     chosen: list[tuple[str, Accepted]]) -> list[dict[str, Any]]:
    rows = []
    for label, metrics in (
            ("方案一", source_summary.get("baseline_metrics")),
            ("方案二", source_summary["final_metrics"])):
        if metrics:
            rows.append({"方案": label, **metrics,
                         "总架次": int(metrics["transport_sorties"]) +
                         int(metrics["relay_sorties"])})
    for goal, item in chosen:
        metrics = item.metrics
        rows.append({"方案": f"方案三-{GOAL_LABEL[goal]}", **metrics,
                     "总架次": int(metrics["transport_sorties"]) +
                     int(metrics["relay_sorties"])})
    return rows


def _write_outputs(args: argparse.Namespace, origin: Path,
                   source_summary: dict[str, Any], data_run: Path,
                   relay: dict[str, Any], geometry: GeometryPool,
                   archive: list[Accepted], attempts: list[dict[str, Any]],
                   candidate_log: list[dict[str, Any]],
                   run_seconds: float) -> dict[str, Any]:
    chosen = _representatives(archive)
    comparison = _comparison_rows(source_summary, chosen)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    archive_rows = []
    for index, entry in enumerate(archive, 1):
        archive_rows.append({"编号": index, "阶段": entry.stage, "档位": entry.grade,
                             **entry.metrics,
                             "总架次": int(entry.metrics["transport_sorties"]) +
                             int(entry.metrics["relay_sorties"])})
    sortielimit_rows = [{
        "搜索阶段": item["search_stage"], "档位": item["grade"],
        "目标架次上限": item["target_transport_sorties"],
        "状态": item["solver_status"], "验收": item["validation"],
        "限制模型下界": item.get("best_bound", ""),
        "路线候选": item.get("route_options", ""),
        "站点候选": item.get("relay_sites", ""),
        "中继任务槽": item.get("relay_slots", ""),
    } for item in attempts if item.get("goal") == "sorties"]
    for goal, entry in chosen:
        prefix = f"q3_opt3_{stamp}_{goal}"
        table_dir = args.output_root / (prefix + "_table")
        figure_dir = args.output_root / (prefix + "_figure")
        table_dir.mkdir(parents=True, exist_ok=False)
        if entry.state is None:
            for item in origin.iterdir():
                if item.is_file() and item.suffix.lower() in {".csv", ".json"}:
                    shutil.copy2(item, table_dir / item.name)
        else:
            base._save_joint_tables(table_dir, entry.state, relay)
        validation = base.validate_official_tables(
            table_dir, data_run, geometry.links, geometry.dem)
        base._save_verification(table_dir, validation,
                                scheme_check_id="SCHEME3_STATUS")
        base.write_rows(table_dir / "Q3_方案比较.csv", comparison,
                        ["方案", "weighted_lateness", "makespan_s", "energy_kwh",
                         "transport_sorties", "relay_sorties", "总架次"])
        base.write_rows(table_dir / "Q3_非支配候选.csv", archive_rows,
                        ["编号", "阶段", "档位", *METRIC_KEYS, "总架次"])
        base.write_rows(table_dir / "Q3_少架次检验.csv", sortielimit_rows,
                        ["搜索阶段", "档位", "目标架次上限", "状态", "验收",
                         "限制模型下界", "路线候选", "站点候选", "中继任务槽"])
        base.json_dump(table_dir / "Q3_方案三搜索轨迹.json", attempts)
        base.json_dump(table_dir / "Q3_候选筛选记录.json", candidate_log)
        summary = {
            "status": validation["status"],
            "method": "Q3 scheme 3: dynamic rebatching, adaptive joint scheduling, Pareto archive",
            "source_scheme2_table": str(origin.resolve()),
            "source": source_summary["source"],
            "source_manifest_sha256": source_summary["source"]["source_manifest_sha256"],
            "code_sha256": {
                name: base.sha256(CODE_DIR / name) for name in (
                    "q3_3_optimize.py", "q3_2_optimize.py",
                    "q3_1_optimize.py", "q2_1_optimize.py")},
            "goal": goal, "grade": entry.grade, "stage": entry.stage,
            "global_optimality_proven": False,
            "reference_metrics": source_summary["final_metrics"],
            "final_metrics": entry.metrics,
            "representative_count": len(chosen),
            "archive_count": len(archive),
            "transport_status": validation["transport_status"],
            "communication_status": validation["communication"]["status"],
            "relay_resource_status": validation["relay_resources"]["status"],
            "search_wall_s": run_seconds,
            "candidate_stats": geometry.stats,
            "candidate_sites": len(geometry.sites),
            "candidate_site_limit": args.max_sites,
            "candidate_stop_limit": args.max_stops,
            "relay_slot_limit": args.relay_slots,
            "table_dir": str(table_dir.resolve()),
            "figure_dir": None,
        }
        if validation["status"] == "PASS":
            if entry.state is not None:
                base._save_figures(figure_dir, entry.state.slices,
                                   entry.state.assigned, entry.state.missions,
                                   scheme_label=f"Q3 scheme 3 {goal}")
            else:
                old_figures = Path(source_summary.get("figure_dir") or "")
                if old_figures.is_dir():
                    shutil.copytree(old_figures, figure_dir)
                else:
                    figure_dir.mkdir()
            _save_tradeoff_figure(figure_dir, comparison)
            _save_resource_figure(figure_dir, table_dir)
            (table_dir / "Q3_READY.txt").write_text(
                "Q3 scheme 3 independently verified PASS\n"
                f"source_manifest_sha256={summary['source_manifest_sha256']}\n",
                encoding="utf-8")
            summary["figure_dir"] = str(figure_dir.resolve())
        base.json_dump(table_dir / "Q3_运行摘要.json", summary)
        outputs.append(summary)
    index = {"status": "PASS" if all(out["status"] == "PASS" for out in outputs)
             else "FAIL",
             "source_scheme2_table": str(origin.resolve()),
             "representatives": outputs,
             "search_wall_s": run_seconds,
             "global_optimality_proven": False}
    base.json_dump(args.output_root / f"q3_opt3_{stamp}_index.json", index)
    return index


def run(args: argparse.Namespace) -> dict[str, Any]:
    total_started = time.monotonic()
    origin = (args.scheme2_table or _latest_passed_run(
        CODE_DIR / "3_outputs" / "2_optimize", "q3_opt2_")).resolve()
    source_summary = json.loads(
        (origin / "Q3_运行摘要.json").read_text(encoding="utf-8"))
    if source_summary.get("status") != "PASS" or not (
            origin / "Q3_READY.txt").is_file():
        raise ValueError("方案二起点须有 PASS 摘要及 Q3_READY.txt")
    data_run = (args.data_run or Path(
        source_summary["source"]["data_run"])).resolve()
    relay, dem, links, meta = base._scene(data_run)
    if meta["source_manifest_sha256"] != source_summary[
            "source"]["source_manifest_sha256"]:
        raise ValueError("方案二与当前数据清单不一致")
    print("Q3 方案三：重新验收方案二完整运输、资源和连续通信结果……",
          flush=True)
    source_check = base.validate_official_tables(origin, data_run, links, dem)
    if source_check["status"] != "PASS":
        raise ValueError("方案二重新验收失败，停止使用该起点")
    ctx = Context(data_run)
    positions = base._node_positions(data_run)
    relay_rows = base.read_rows(origin / "Q3_中继架次.csv")
    sites = previous._load_sites(
        origin, relay_rows, data_run, relay, dem, links, ctx,
        positions, args.sample_step_s, args.candidate_spacing_m,
        args.max_sites)
    geometry = GeometryPool(
        ctx, relay, dem, links, positions, sites,
        args.sample_step_s, args.candidate_spacing_m,
        args.max_sites, args.max_site_points)
    incumbent, origin_hints = previous._incumbent_choices(
        ctx, origin, positions, links, args.sample_step_s,
        sites, geometry.access_cache)
    reference = {key: source_summary["final_metrics"][key]
                 for key in METRIC_KEYS}
    source_entry = Accepted(reference, None, None, origin,
                            "方案二已验收起点", "REFERENCE")
    archive = [source_entry]
    attempts: list[dict[str, Any]] = []
    candidate_log: list[dict[str, Any]] = []
    global_choices: dict[tuple[Any, str], base.JointChoice] = {
        _route_key(choice): choice for choice in incumbent}
    global_batches: list[list[tuple[Any, str]]] = []
    print("Q3 方案三：验证旧方案可映射到当前整数排程，以便作为搜索起点……",
          flush=True)
    fixed_relays = {slot: (site, ready, end)
                    for slot, site, _, _, ready, end in origin_hints["relays"]}
    mapping, map_report = base._solve_joint(
        ctx, relay, sites, incumbent, args.horizon_s,
        args.relay_slots, min(30, args.global_seconds), args.workers, False,
        hints=origin_hints,
        fixed_partial_transport=origin_hints["fixed_transport"],
        fixed_relay_schedule=fixed_relays,
        feasibility_only=True, explicit_relay_uavs=True)
    map_record = {"search_stage": "原解映射",
                  "solver_stage": map_report["stage"],
                  **{k: v for k, v in map_report.items() if k != "stage"},
                  "validation": "NO_SOLUTION"}
    mapping_entry: Accepted | None = None
    if mapping is not None:
        map_state, check = previous._inspect_solution(
            mapping, ctx, relay, sites, data_run, links, dem)
        map_record["validation"] = check["status"]
        map_record["metrics"] = previous._metrics(map_state)
        if check["status"] == "PASS":
            mapping_entry = Accepted(
                map_record["metrics"], mapping, map_state, None,
                "原解映射", "REFERENCE")
            if _same_metrics(mapping_entry.metrics, reference):
                source_entry.solution, source_entry.state = mapping, map_state
            else:
                _archive_add(archive, mapping_entry)
    attempts.append(map_record)
    if mapping_entry is None:
        print("Q3 方案三：旧表未能映射；仍保留独立验收过的原表，"
              "从整体联合重排继续搜索", flush=True)
    search_started = time.monotonic()
    search_deadline = search_started + args.time_budget_s
    print(f"Q3 方案三：搜索预算 {args.time_budget_s:.0f}s，"
          f"当前参考 {reference['transport_sorties']} 运输 + "
          f"{reference['relay_sorties']} 中继架次", flush=True)
    try:
        for round_no in range(args.max_rounds):
            remaining = search_deadline - time.monotonic()
            if remaining < args.min_remaining_s:
                break
            goal = ("sorties" if round_no < args.sortie_rounds else
                    ("makespan", "energy", "lateness", "sorties")[
                        (round_no - args.sortie_rounds) % 4])
            grade = ("G0", "G1", "G2")[round_no % 3]
            target = (min(int(entry.metrics["transport_sorties"])
                          for entry in archive) - 1 if goal == "sorties" else None)
            if target is not None and target < 1:
                continue
            seed = min(archive, key=lambda item: _goal_value(item, goal))
            if seed.solution is None and mapping_entry is not None:
                seed = mapping_entry
            current = _selected(seed.solution) if seed.solution else incumbent
            release_count = (6 if round_no < 3 else
                             8 if round_no < 7 else 12)
            released = _release(current, round_no, release_count)
            full = seed.solution is None or round_no % args.full_every == (
                args.full_every - 1)
            if full:
                released = set(range(len(current)))
            print(f"Q3 方案三第 {round_no+1}/{args.max_rounds} 轮："
                  f"主目标={GOAL_LABEL[goal]}、{grade}、"
                  f"当前少架次目标={target if target else '-'}、"
                  f"释放={len(released)} 批、剩余={remaining:.0f}s", flush=True)
            proposals, generated = _propose_routes(
                ctx, current, released, args.max_stops,
                args.max_packages,
                min(search_deadline, time.monotonic() +
                    args.candidate_seconds),
                rotation=round_no)
            fresh, certified = _certify_packages(
                ctx, proposals, geometry, current,
                args.max_new_choices,
                min(search_deadline, time.monotonic() +
                    args.certify_seconds))
            batch = []
            for choice in fresh:
                key = _route_key(choice)
                if key not in global_choices:
                    global_choices[key] = choice
                    batch.append(key)
            if batch:
                global_batches.append(batch)
            candidate_log.append({
                "round": round_no + 1, "goal": goal, "grade": grade,
                "released": len(released), "full_reoptimization": full,
                "generated": generated, "certified": certified,
                "site_count": len(geometry.sites)})
            selected_keys = {_route_key(choice) for choice in current}
            protected = {_route_key(choice) for choice in incumbent}
            protected.update(selected_keys)
            for entry in archive:
                if entry.solution is not None:
                    protected.update(_route_key(choice)
                                     for choice in _selected(entry.solution))
            if len(global_choices) > args.max_active_choices:
                for batch in list(global_batches):
                    if len(global_choices) <= args.max_active_choices:
                        break
                    if any(key in protected for key in batch):
                        continue
                    for key in batch:
                        global_choices.pop(key, None)
                    global_batches.remove(batch)
            candidate_log[-1]["global_active_choices"] = len(global_choices)
            candidate_log[-1]["active_budget_exceeded"] = (
                len(global_choices) > args.max_active_choices)
            if full:
                pool = list(global_choices.values())
                for choice in current:
                    if _route_key(choice) not in global_choices:
                        pool.append(choice)
                fixed = None
            else:
                pool = list(current)
                known = selected_keys.copy()
                for choice in fresh:
                    if _route_key(choice) not in known:
                        pool.append(choice)
                        known.add(_route_key(choice))
                fixed = _fixed_outside(seed.solution, released)
            if seed.solution is not None:
                hints = previous._solution_hints(seed.solution)
            else:
                hints = origin_hints
            seconds = min(
                args.global_seconds if full else args.local_seconds,
                search_deadline - time.monotonic())
            if seconds < args.min_remaining_s:
                break
            label = f"第{round_no+1}轮/" + ("全局" if full else "邻域")
            solution, state, record = _attempt(
                label, grade, goal, target, reference, ctx, relay,
                geometry, pool, hints, fixed, args, data_run, seconds)
            record["released_transport_tasks"] = len(released)
            record["candidate_route_count"] = len(pool)
            attempts.append(record)
            print(f"Q3 方案三：{record['solver_status']}，"
                  f"独立验收={record['validation']}，"
                  f"已交付方案={len(archive)} 个非支配候选", flush=True)
            if solution is not None and state is not None:
                item = Accepted(record["metrics"], solution, state, None,
                                label + "/" + goal, grade)
                kept = _archive_add(archive, item)
                if kept:
                    print(f"Q3 方案三：收录可行解 {item.metrics}", flush=True)
                if (record.get("selected_relay_sorties", 0) >=
                        args.relay_slots - 1 and
                        args.relay_slots < args.relay_slots_max):
                    old = args.relay_slots
                    args.relay_slots = min(args.relay_slots_max, old + 6)
                    print(f"Q3 方案三：中继任务槽接近用满，"
                          f"从 {old} 增至 {args.relay_slots}", flush=True)
    except KeyboardInterrupt:
        print("Q3 方案三：已收到中断，保存已独立验收的候选。", flush=True)
    return _write_outputs(
        args, origin, source_summary, data_run, relay, geometry,
        archive, attempts, candidate_log, time.monotonic() - total_started)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme2-table", type=Path, default=None,
                        help="方案二已验收的 *_table；默认取最新 PASS")
    parser.add_argument("--data-run", type=Path, default=None)
    parser.add_argument("--output-root", type=Path,
                        default=CODE_DIR / "3_outputs" / "3_optimize")
    parser.add_argument("--time-budget-s", type=float, default=1200,
                        help="搜索预算；几何准备和最终出表出图另行计时")
    parser.add_argument("--global-seconds", type=float, default=120)
    parser.add_argument("--local-seconds", type=float, default=45)
    parser.add_argument("--candidate-seconds", type=float, default=22)
    parser.add_argument("--certify-seconds", type=float, default=28)
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--sortie-rounds", type=int, default=8)
    parser.add_argument("--full-every", type=int, default=4)
    parser.add_argument("--max-packages", type=int, default=36)
    parser.add_argument("--max-new-choices", type=int, default=96)
    parser.add_argument("--max-active-choices", type=int, default=260)
    parser.add_argument("--max-stops", type=int, default=2,
                        help="候选路线访问区数的搜索上限，不是题目物理限制")
    parser.add_argument("--relay-slots", type=int, default=18)
    parser.add_argument("--relay-slots-max", type=int, default=24)
    parser.add_argument("--max-sites", type=int, default=48)
    parser.add_argument("--max-site-points", type=int, default=2400)
    parser.add_argument("--horizon-s", type=int, default=21600)
    parser.add_argument("--sample-step-s", type=float, default=20)
    parser.add_argument("--candidate-spacing-m", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--min-remaining-s", type=float, default=5)
    args = parser.parse_args(argv)
    positive = (args.time_budget_s, args.global_seconds, args.local_seconds,
                args.candidate_seconds, args.certify_seconds,
                args.sample_step_s, args.min_remaining_s)
    integer_positive = (args.max_rounds, args.sortie_rounds,
                        args.full_every, args.max_packages,
                        args.max_new_choices, args.max_active_choices,
                        args.max_stops, args.relay_slots, args.max_sites,
                        args.max_site_points, args.horizon_s,
                        args.candidate_spacing_m, args.workers)
    if min(positive) <= 0 or min(integer_positive) < 1:
        parser.error("搜索时间、候选数、站点数、时域与并行数须为正")
    if args.relay_slots_max < args.relay_slots or args.max_sites < 2:
        parser.error("扩容上限须不小于初始中继任务槽，且至少允许两个站点")
    result = run(args)
    print(json.dumps({
        "status": result["status"], "search_wall_s": result["search_wall_s"],
        "representatives": [{
            "goal": row["goal"], "metrics": row["final_metrics"],
            "table_dir": row["table_dir"], "figure_dir": row["figure_dir"]}
            for row in result["representatives"]]
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
