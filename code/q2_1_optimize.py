"""D 题问题二：逐箱组批、多区路线、实体机与共享电池联合优化。

这是可行性优先的随机邻域搜索，不声称全局最优。单次运行先保存表格与
独立验收，再从同一方案生成 PNG/PDF。输入始终是已经验收的 0_outputs。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from _0_pipeline import sha256, verify_ready
from q1_0_baseline import (
    CODE_DIR, _read_inputs, find_latest_validated_run, leg_energy_kwh, leg_time_s,
)
from q2_0_baseline import (
    DELIVERY_COLUMNS, SORTIE_COLUMNS, charge_to_full_s, optional_seconds,
    read_resources, required_seconds, save_delivery_figures, save_route_figure,
    validate_template, write_csv,
)


TOL = 1e-6
MAX_SEARCH_STOPS = 3  # 搜索邻域大小，不是题目给出的物理上限


@dataclass(frozen=True)
class Route:
    """每站只出现一次；站内箱序保留，以便逐箱计算交付时刻。"""

    visits: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def box_ids(self) -> tuple[str, ...]:
        return tuple(box_id for _, group in self.visits for box_id in group)

    @property
    def zones(self) -> tuple[str, ...]:
        return tuple(zone for zone, _ in self.visits)


def box_order(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(boxes, key=lambda box: (
        optional_seconds(box["hard_deadline_s"])
        if optional_seconds(box["hard_deadline_s"]) is not None else math.inf,
        required_seconds(box["expected_s"], str(box["box_id"])),
        -float(box["priority"]), str(box["box_id"]),
    ))


class Context:
    def __init__(self, data_run: Path):
        self.data_run = data_run
        boxes, self.models, self.arcs = _read_inputs(data_run)
        self.boxes = {str(box["box_id"]): box for box in boxes}
        if len(boxes) != 80 or len(self.boxes) != 80:
            raise ValueError("问题二要求 80 个编号唯一的货箱")
        self.uavs, self.batteries = read_resources(data_run)
        self.options_cache: dict[Route, dict[str, dict[str, Any]]] = {}
        for box in boxes:
            if (str(box["medical_bool"]).lower() in {"true", "1"} or
                    str(box["is_first_batch"]).lower() in {"true", "1"}):
                if optional_seconds(box["hard_deadline_s"]) is None:
                    raise ValueError(f"{box['box_id']} 缺少硬截止")

    def route(self, visits: list[tuple[str, list[str] | tuple[str, ...]]]) -> Route:
        normalized = []
        seen_zones: set[str] = set()
        for zone, box_ids in visits:
            if not box_ids:
                continue
            if zone in seen_zones:
                raise ValueError(f"路线重复访问 {zone}")
            seen_zones.add(zone)
            ids = tuple(str(box["box_id"]) for box in
                        box_order([self.boxes[box_id] for box_id in box_ids]))
            if len(set(ids)) != len(ids) or any(self.boxes[bid]["zone_id"] != zone for bid in ids):
                raise ValueError(f"{zone} 的箱号重复或服务区不符")
            normalized.append((zone, ids))
        if not normalized:
            raise ValueError("空路线")
        return Route(tuple(normalized))

    def options(self, route: Route) -> dict[str, dict[str, Any]]:
        if route not in self.options_cache:
            self.options_cache[route] = {
                kind: option for kind in sorted(self.models)
                if (option := self.evaluate(route, kind)) is not None
            }
        return self.options_cache[route]

    def evaluate(self, route: Route, kind: str) -> dict[str, Any] | None:
        """从 O01 出发逐段按剩余载荷算能耗，交付后才减载。"""
        model = self.models[kind]
        boxes = [self.boxes[bid] for bid in route.box_ids]
        total_mass = sum(float(box["mass_kg"]) for box in boxes)
        total_volume = sum(float(box["volume_m3"]) for box in boxes)
        if (total_mass > float(model["max_payload_kg"]) + TOL or
                total_volume > float(model["capacity_m3"]) + TOL):
            return None
        elapsed = float(model["prep_s"]) + len(boxes) * float(model["load_per_box_s"])
        launch_offset = elapsed
        remaining_mass = total_mass
        energy = 0.0
        node = "O01"
        legs = []
        visit_times = []
        completion_offsets = {}
        sequence = {}
        count = 0
        for zone, ids in (*route.visits, ("O01", ())):
            arc = self.arcs.get((node, zone))
            if arc is None or str(arc["valid_flag"]).lower() not in {"true", "1"}:
                return None
            try:
                leg_energy = leg_energy_kwh(model, arc, remaining_mass)["total_kwh"]
                leg_time = leg_time_s(model, arc)
            except ValueError:
                return None
            energy += leg_energy
            elapsed += leg_time
            legs.append({
                "from_id": node, "to_id": zone, "load_kg": remaining_mass,
                "energy_kwh": leg_energy, "flight_s": leg_time,
                "arrival_offset_s": elapsed,
            })
            if zone != "O01":
                arrival = elapsed
                elapsed += float(model["handoff_base_s"])
                for box_id in ids:
                    elapsed += float(model["handoff_per_box_s"])
                    completion_offsets[box_id] = elapsed
                    count += 1
                    sequence[box_id] = count
                visit_times.append({
                    "zone_id": zone, "arrival_offset_s": arrival,
                    "handoff_end_offset_s": elapsed,
                })
                remaining_mass -= sum(float(self.boxes[bid]["mass_kg"]) for bid in ids)
                if remaining_mass < -TOL:
                    raise AssertionError("货物质量守恒失败")
            node = zone
        limit = (1.0 - float(model["return_soc_min"])) * float(model["energy_use_kwh"])
        if energy > limit + TOL:
            return None
        return {
            "type_id": kind, "duration_s": elapsed, "launch_offset_s": launch_offset,
            "energy_kwh": energy, "return_soc": 1.0 - energy / float(model["energy_use_kwh"]),
            "mass_kg": total_mass, "volume_m3": total_volume,
            "completion_offsets": completion_offsets, "sequence": sequence,
            "legs": legs, "visits": visit_times,
        }


def initial_routes(ctx: Context) -> list[Route]:
    """硬截止箱先独立组批；软箱在各区用物理可行性检查做首次装箱。"""
    by_zone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for box in ctx.boxes.values():
        by_zone[str(box["zone_id"])].append(box)
    urgent: list[Route] = []
    soft: list[Route] = []
    for zone in sorted(by_zone):
        hard = box_order([b for b in by_zone[zone]
                          if optional_seconds(b["hard_deadline_s"]) is not None])
        later = sorted((b for b in by_zone[zone]
                        if optional_seconds(b["hard_deadline_s"]) is None),
                       key=lambda b: (-float(b["mass_kg"]), -float(b["volume_m3"]),
                                      str(b["box_id"])))
        if hard:
            route = ctx.route([(zone, [str(b["box_id"]) for b in hard])])
            if not ctx.options(route):
                raise ValueError(f"{zone} 的硬截止箱同批对全部机型不可行；需要拆批")
            urgent.append(route)
        bins: list[list[str]] = []
        for box in later:
            box_id = str(box["box_id"])
            placed = False
            for group in bins:
                trial = ctx.route([(zone, [*group, box_id])])
                if ctx.options(trial):
                    group.append(box_id)
                    placed = True
                    break
            if not placed:
                trial = ctx.route([(zone, [box_id])])
                if not ctx.options(trial):
                    raise ValueError(f"货箱 {box_id} 无可用运输机型")
                bins.append([box_id])
        soft.extend(ctx.route([(zone, group)]) for group in bins)

    def priority(route: Route) -> tuple[float, float, str]:
        boxes = [ctx.boxes[bid] for bid in route.box_ids]
        hard = min((d for b in boxes
                    if (d := optional_seconds(b["hard_deadline_s"])) is not None),
                   default=math.inf)
        due = min(required_seconds(b["expected_s"], str(b["box_id"])) for b in boxes)
        return hard, due, route.box_ids[0]

    return sorted(urgent, key=priority) + sorted(soft, key=priority)


def simulate(ctx: Context, routes: list[Route]) -> dict[str, Any] | None:
    """给定组批与路线顺序，联合选择机型、实体机、电池及最早可行开始时刻。"""
    uav_available = {u.uav_id: 0.0 for u in ctx.uavs}
    battery_available = {b.battery_id: 0.0 for b in ctx.batteries}
    sorties = []
    deliveries = []
    for index, route in enumerate(routes, 1):
        options = ctx.options(route)
        if not options:
            return None
        choices = []
        for uav in ctx.uavs:
            if uav.type_id not in options:
                continue
            option = options[uav.type_id]
            for battery in ctx.batteries:
                if battery.type_id != uav.type_id:
                    continue
                start = max(uav_available[uav.uav_id],
                            battery_available[battery.battery_id])
                hard_late = []
                soft_late = 0.0
                for box_id, offset in option["completion_offsets"].items():
                    box = ctx.boxes[box_id]
                    finish = start + offset
                    deadline = optional_seconds(box["hard_deadline_s"])
                    if deadline is not None and finish > deadline + TOL:
                        hard_late.append(finish - deadline)
                    if deadline is None:
                        soft_late += float(box["priority"]) * max(
                            0.0, finish - required_seconds(box["expected_s"], box_id))
                rank = (
                    len(hard_late), sum(hard_late), soft_late,
                    start + option["duration_s"], option["energy_kwh"],
                    uav.uav_id, battery.battery_id,
                )
                choices.append((rank, uav, battery, option, start))
        if not choices:
            return None
        _, uav, battery, option, start = min(choices, key=lambda item: item[0])
        batch_id = f"Q{index:03d}"
        end = start + option["duration_s"]
        charge_s = charge_to_full_s(option["return_soc"], battery.full_charge_s)
        sortie = {
            "batch_id": batch_id, "type_id": uav.type_id,
            "uav_id": uav.uav_id, "battery_id": battery.battery_id,
            "route": route, "visit_order": ">".join(route.zones),
            "box_ids": list(route.box_ids), "start_s": start,
            "launch_s": start + option["launch_offset_s"],
            "return_s": end, "charge_start_s": end,
            "charge_end_s": end + charge_s, "charge_s": charge_s,
            "mass_kg": option["mass_kg"], "volume_m3": option["volume_m3"],
            "energy_kwh": option["energy_kwh"], "return_soc": option["return_soc"],
            "legs": option["legs"], "visits": option["visits"],
        }
        sorties.append(sortie)
        for box_id in route.box_ids:
            box = ctx.boxes[box_id]
            finish = start + option["completion_offsets"][box_id]
            hard = optional_seconds(box["hard_deadline_s"])
            expected = required_seconds(box["expected_s"], box_id)
            deliveries.append({
                "box_id": box_id, "batch_id": batch_id,
                "zone_id": str(box["zone_id"]), "sequence": option["sequence"][box_id],
                "complete_s": finish, "hard_deadline_s": hard, "expected_s": expected,
                "priority": float(box["priority"]),
                "hard_slack_s": None if hard is None else hard - finish,
                "soft_lateness_s": max(0.0, finish - expected),
            })
        uav_available[uav.uav_id] = end
        battery_available[battery.battery_id] = end + charge_s
    sorties.sort(key=lambda s: (s["start_s"], s["batch_id"]))
    deliveries.sort(key=lambda d: d["box_id"])
    hard_fail = [d for d in deliveries
                 if d["hard_deadline_s"] is not None and d["hard_slack_s"] < -TOL]
    hard_delay = sum(-d["hard_slack_s"] for d in hard_fail)
    soft_late = sum(d["priority"] * d["soft_lateness_s"] for d in deliveries
                    if d["hard_deadline_s"] is None)
    score = (
        len(hard_fail), hard_delay, soft_late,
        max(s["return_s"] for s in sorties),
        sum(s["energy_kwh"] for s in sorties), len(sorties),
    )
    return {"routes": routes, "sorties": sorties, "deliveries": deliveries, "score": score}


def search_cost(score: tuple[float, ...]) -> float:
    failures, hard_delay, soft_delay, makespan, energy, sorties = score
    return (failures * 1e10 + hard_delay * 1e6 + soft_delay +
            0.2 * makespan + 10 * energy + 20 * sorties)


def mutate(ctx: Context, current: dict[str, Any],
           rng: random.Random) -> list[Route] | None:
    routes = list(current["routes"])
    if len(routes) < 2:
        return None
    move = rng.choices(
        ["relocate", "merge", "split", "swap", "reverse"],
        weights=[44, 25, 12, 13, 6],
    )[0]
    if move == "swap":
        i, j = rng.sample(range(len(routes)), 2)
        routes[i], routes[j] = routes[j], routes[i]
        return routes
    if move == "reverse":
        choices = [i for i, route in enumerate(routes) if len(route.visits) > 1]
        if not choices:
            return None
        i = rng.choice(choices)
        routes[i] = Route(tuple(reversed(routes[i].visits)))
        return routes
    if move == "merge":
        i, j = sorted(rng.sample(range(len(routes)), 2))
        merged: dict[str, list[str]] = {}
        for route in (routes[i], routes[j]):
            for zone, ids in route.visits:
                merged.setdefault(zone, []).extend(ids)
        if len(merged) > MAX_SEARCH_STOPS:
            return None
        combined = ctx.route([(zone, ids) for zone, ids in merged.items()])
        if not ctx.options(combined):
            return None
        routes[i] = combined
        routes.pop(j)
        return routes
    if move == "split":
        choices = [i for i, route in enumerate(routes) if len(route.box_ids) > 1]
        if not choices:
            return None
        i = rng.choice(choices)
        donor = routes[i]
        box_id = rng.choice(donor.box_ids)
        zone = str(ctx.boxes[box_id]["zone_id"])
        remaining = [(z, [bid for bid in ids if bid != box_id]) for z, ids in donor.visits]
        routes[i] = ctx.route(remaining)
        routes.insert(i, ctx.route([(zone, [box_id])]))
        return routes
    # 最迟交付箱更常被选来做重插入，仍允许探索其他箱。
    late = [d["box_id"] for d in current["deliveries"]
            if d["soft_lateness_s"] > TOL]
    box_id = (rng.choice(late) if late and rng.random() < 0.65
              else rng.choice(rng.choice(routes).box_ids))
    donor_i = next(i for i, route in enumerate(routes) if box_id in route.box_ids)
    receiver_i = rng.choice([i for i in range(len(routes)) if i != donor_i])
    donor = routes[donor_i]
    receiver = routes[receiver_i]
    zone = str(ctx.boxes[box_id]["zone_id"])
    new_donor = [(z, [bid for bid in ids if bid != box_id]) for z, ids in donor.visits]
    new_receiver = [(z, list(ids)) for z, ids in receiver.visits]
    if zone in receiver.zones:
        for z, ids in new_receiver:
            if z == zone:
                ids.append(box_id)
                break
    else:
        if len(new_receiver) >= MAX_SEARCH_STOPS:
            return None
        new_receiver.insert(rng.randrange(len(new_receiver) + 1), (zone, [box_id]))
    changed = ctx.route(new_receiver)
    if not ctx.options(changed):
        return None
    routes[receiver_i] = changed
    if any(ids for _, ids in new_donor):
        routes[donor_i] = ctx.route(new_donor)
    else:
        routes.pop(donor_i)
    return routes


def optimize(ctx: Context, iterations: int, time_limit_s: float,
             seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)
    seed_plan = simulate(ctx, initial_routes(ctx))
    if seed_plan is None:
        raise RuntimeError("初始路线无法排程")
    best = current = seed_plan
    start = time.perf_counter()
    tried = accepted = improvements = 0
    for step in range(iterations):
        if time.perf_counter() - start >= time_limit_s:
            break
        proposal = mutate(ctx, current, rng)
        if proposal is None:
            continue
        tried += 1
        trial = simulate(ctx, proposal)
        if trial is None:
            continue
        if trial["score"] < best["score"]:
            best = trial
            improvements += 1
            if improvements == 1 or improvements % 20 == 0:
                print(f"improved {improvements}: hard={best['score'][0]}, "
                      f"late={best['score'][2]:.1f}, "
                      f"makespan={best['score'][3]:.1f}s, "
                      f"energy={best['score'][4]:.3f}kWh", flush=True)
        temperature = max(100.0, 10000.0 * (1.0 - step / max(1, iterations)) ** 2)
        delta = search_cost(trial["score"]) - search_cost(current["score"])
        if delta <= 0 or rng.random() < math.exp(-min(delta / temperature, 700)):
            current = trial
            accepted += 1
        if step and step % 350 == 0:
            current = best
    return best, seed_plan, {
        "iterations_completed": step + 1 if iterations else 0,
        "proposals": tried, "accepted": accepted,
        "improvements": improvements,
        "elapsed_s": time.perf_counter() - start,
        "seed": seed, "global_optimality_proven": False,
        "search_stops_limit": MAX_SEARCH_STOPS,
    }


def check_plan(ctx: Context, plan: dict[str, Any]) -> list[dict[str, str]]:
    """独立重放每段载荷/飞行/交付，再检查实体资源日历。"""
    sorties, deliveries = plan["sorties"], plan["deliveries"]
    checks: list[dict[str, str]] = []

    def add(check_id: str, good: bool, detail: str = "") -> None:
        checks.append({
            "check_id": check_id,
            "status": "PASS" if good else "FAIL",
            "detail": detail,
        })

    add("all_80_boxes_once",
        Counter(d["box_id"] for d in deliveries) == Counter(ctx.boxes.keys()),
        f"rows={len(deliveries)}, unique={len({d['box_id'] for d in deliveries})}")
    add("route_delivery_consistency",
        Counter(bid for s in sorties for bid in s["box_ids"]) ==
        Counter(d["box_id"] for d in deliveries))
    uavs = {u.uav_id: u for u in ctx.uavs}
    batteries = {b.battery_id: b for b in ctx.batteries}
    by_box = {d["box_id"]: d for d in deliveries}
    for sortie in sorties:
        batch_id = sortie["batch_id"]
        route = sortie["route"]
        model = ctx.models[sortie["type_id"]]
        add(f"{batch_id}:resource_type",
            uavs[sortie["uav_id"]].type_id ==
            batteries[sortie["battery_id"]].type_id == sortie["type_id"])
        add(f"{batch_id}:known_route",
            bool(route.visits) and len(route.zones) == len(set(route.zones)) and
            all(all(ctx.boxes[bid]["zone_id"] == zone for bid in ids)
                for zone, ids in route.visits))
        mass = sum(float(ctx.boxes[bid]["mass_kg"]) for bid in route.box_ids)
        volume = sum(float(ctx.boxes[bid]["volume_m3"]) for bid in route.box_ids)
        add(f"{batch_id}:capacity",
            mass <= float(model["max_payload_kg"]) + TOL and
            volume <= float(model["capacity_m3"]) + TOL)
        elapsed = sortie["start_s"] + float(model["prep_s"]) + (
            len(route.box_ids) * float(model["load_per_box_s"]))
        good = math.isclose(elapsed, sortie["launch_s"], abs_tol=TOL)
        remaining = mass
        energy = 0.0
        origin = "O01"
        route_legs = []
        for zone, ids in (*route.visits, ("O01", ())):
            arc = ctx.arcs.get((origin, zone))
            if arc is None or str(arc["valid_flag"]).lower() not in {"true", "1"}:
                good = False
                break
            value = leg_energy_kwh(model, arc, remaining)["total_kwh"]
            duration = leg_time_s(model, arc)
            energy += value
            elapsed += duration
            route_legs.append((origin, zone, remaining, value, duration))
            if zone != "O01":
                elapsed += float(model["handoff_base_s"])
                for bid in ids:
                    elapsed += float(model["handoff_per_box_s"])
                    good &= (bid in by_box and
                             math.isclose(elapsed, by_box[bid]["complete_s"], abs_tol=TOL))
                remaining -= sum(float(ctx.boxes[bid]["mass_kg"]) for bid in ids)
            origin = zone
        good &= math.isclose(remaining, 0.0, abs_tol=TOL)
        good &= math.isclose(elapsed, sortie["return_s"], abs_tol=TOL)
        good &= math.isclose(energy, sortie["energy_kwh"], abs_tol=TOL)
        add(f"{batch_id}:physical_replay", good)
        add(f"{batch_id}:return_soc",
            energy <= (1.0 - float(model["return_soc_min"])) *
            float(model["energy_use_kwh"]) + TOL and
            math.isclose(sortie["return_soc"],
                         1.0 - energy / float(model["energy_use_kwh"]), abs_tol=TOL))
        add(f"{batch_id}:leg_load_energy",
            len(route_legs) == len(sortie["legs"]) and all(
                a == row["from_id"] and b == row["to_id"] and
                math.isclose(load, row["load_kg"], abs_tol=TOL) and
                math.isclose(e, row["energy_kwh"], abs_tol=TOL) and
                math.isclose(t, row["flight_s"], abs_tol=TOL)
                for (a, b, load, e, t), row in zip(route_legs, sortie["legs"])))
        soc = 1.0 - energy / float(model["energy_use_kwh"])
        full = batteries[sortie["battery_id"]].full_charge_s
        charging = (full * (0.65 * (0.9 - soc) / 0.9 + 0.35)
                    if soc < 0.9 else full * 0.35 * (1.0 - soc) / 0.1)
        add(f"{batch_id}:charging_replay",
            math.isclose(sortie["charge_end_s"] - sortie["return_s"],
                         charging, abs_tol=TOL))
        add(f"{batch_id}:time_order",
            0 <= sortie["start_s"] <= sortie["launch_s"] <=
            sortie["return_s"] <= sortie["charge_end_s"])
    for delivery in deliveries:
        deadline = delivery["hard_deadline_s"]
        if deadline is not None:
            add(f"{delivery['box_id']}:hard_deadline",
                delivery["complete_s"] <= deadline + TOL,
                f"delivered={delivery['complete_s']:.3f}s, "
                f"deadline={deadline:.3f}s")
    for uav in ctx.uavs:
        assigned = sorted((s for s in sorties if s["uav_id"] == uav.uav_id),
                          key=lambda s: s["start_s"])
        add(f"{uav.uav_id}:no_overlap",
            all(a["return_s"] <= b["start_s"] + TOL
                for a, b in zip(assigned, assigned[1:])))
    for battery in ctx.batteries:
        assigned = sorted((s for s in sorties if s["battery_id"] == battery.battery_id),
                          key=lambda s: s["start_s"])
        add(f"{battery.battery_id}:charged_before_reuse",
            all(a["charge_end_s"] <= b["start_s"] + TOL
                for a, b in zip(assigned, assigned[1:])))
    return checks


def baseline_comparison(output_root: Path, data_hash: str) -> dict[str, Any] | None:
    directories = [*(output_root / "0_baseline").glob("q2_base_*_table"),
                   *output_root.glob("q2_base_*_table")]
    for directory in sorted(directories, key=lambda path: path.stat().st_mtime,
                            reverse=True):
        summary_path = directory / "Q2_运行摘要.json"
        if not summary_path.is_file():
            continue
        baseline = json.loads(summary_path.read_text(encoding="utf-8"))
        if baseline.get("source_manifest_sha256") == data_hash:
            return baseline
    return None


def save_tables(table_dir: Path, ctx: Context, plan: dict[str, Any],
                seed_plan: dict[str, Any], search: dict[str, Any],
                checks: list[dict[str, str]], output_root: Path) -> dict[str, Any]:
    if table_dir.exists():
        raise FileExistsError(f"不覆盖已有结果：{table_dir}")
    validate_template()
    table_dir.mkdir(parents=True)
    sorties, deliveries = plan["sorties"], plan["deliveries"]
    official_sorties = [{
        "架次编号": s["batch_id"], "无人机编号": s["uav_id"],
        "机型编号": s["type_id"], "电池编号": s["battery_id"],
        "开始时刻（s）": s["start_s"], "访问服务区顺序": s["visit_order"],
        "返回O01时刻（s）": s["return_s"], "架次能耗（kWh）": s["energy_kwh"],
    } for s in sorties]
    official_delivery = [{
        "货箱编号": d["box_id"], "架次编号": d["batch_id"],
        "服务区编号": d["zone_id"], "交付完成时刻（s）": d["complete_s"],
    } for d in deliveries]
    write_csv(table_dir / "Q2_运输架次.csv", official_sorties, SORTIE_COLUMNS)
    write_csv(table_dir / "Q2_逐箱交付.csv", official_delivery, DELIVERY_COLUMNS)
    detail = [{
        key: value for key, value in s.items() if key not in {"route", "legs", "visits"}
    } for s in sorties]
    for row in detail:
        row["box_ids"] = ";".join(row["box_ids"])
    write_csv(table_dir / "Q2_架次与电池周转.csv", detail, list(detail[0]))
    write_csv(table_dir / "Q2_逐箱时限检查.csv", deliveries, list(deliveries[0]))
    write_csv(table_dir / "Q2_验收检查.csv", checks, ["check_id", "status", "detail"])
    leg_rows = []
    for s in sorties:
        for index, leg in enumerate(s["legs"], 1):
            leg_rows.append({
                "架次编号": s["batch_id"], "航段序号": index,
                "起点": leg["from_id"], "终点": leg["to_id"],
                "离开时载货质量（kg）": leg["load_kg"],
                "飞行时间（s）": leg["flight_s"],
                "航段能耗（kWh）": leg["energy_kwh"],
                "抵达时刻（s）": s["start_s"] + leg["arrival_offset_s"],
            })
    write_csv(table_dir / "Q2_路线航段.csv", leg_rows, list(leg_rows[0]))
    failed = [check for check in checks if check["status"] == "FAIL"]
    hard_failed = [d for d in deliveries if d["hard_slack_s"] is not None and
                   d["hard_slack_s"] < -TOL]
    data_hash = sha256(ctx.data_run / "meta" / "validated_artifacts.json")
    summary = {
        "status": "PASS" if not failed else "FAIL",
        "method": "feasibility-first multi-zone route and resource neighborhood search",
        "global_optimality_proven": False,
        "source_data_run": str(ctx.data_run),
        "source_manifest_sha256": data_hash,
        "sorties": len(sorties), "delivered_boxes": len(deliveries),
        "multi_zone_sorties": sum(len(s["route"].visits) > 1 for s in sorties),
        "uavs_available": len(ctx.uavs), "batteries_available": len(ctx.batteries),
        "uavs_used": len({s["uav_id"] for s in sorties}),
        "batteries_used": len({s["battery_id"] for s in sorties}),
        "hard_deadline_boxes": sum(d["hard_deadline_s"] is not None for d in deliveries),
        "hard_deadline_violations": len(hard_failed),
        "first_failure": failed[0] if failed else None,
        "makespan_s": plan["score"][3],
        "total_energy_kwh": plan["score"][4],
        "soft_weighted_lateness_s": plan["score"][2],
        "seed_plan_metrics": {
            "hard_deadline_violations": seed_plan["score"][0],
            "soft_weighted_lateness_s": seed_plan["score"][2],
            "makespan_s": seed_plan["score"][3],
            "total_energy_kwh": seed_plan["score"][4],
            "sorties": seed_plan["score"][5],
        },
        "search": search,
        "battery_rule": "same-type pool, initially full, recharge to 100% before reuse",
        "note": "PASS is a verified feasible heuristic solution, not a proof of global optimum.",
    }
    baseline = baseline_comparison(output_root, data_hash)
    comparison = []
    if baseline:
        comparison.append({
            "方案": "固定FFD基线", "状态": baseline["status"],
            "硬时限违约箱数": baseline["hard_deadline_violations"],
            "架次数": baseline["sorties"],
            "最晚返航（s）": baseline["makespan_s"],
            "运输总能耗（kWh）": baseline["total_energy_kwh"],
            "软箱加权迟到（s）": baseline["soft_weighted_lateness_s"],
        })
    comparison.append({
        "方案": "路线与资源联合搜索", "状态": summary["status"],
        "硬时限违约箱数": summary["hard_deadline_violations"],
        "架次数": summary["sorties"],
        "最晚返航（s）": summary["makespan_s"],
        "运输总能耗（kWh）": summary["total_energy_kwh"],
        "软箱加权迟到（s）": summary["soft_weighted_lateness_s"],
    })
    write_csv(table_dir / "Q2_方案对照.csv", comparison, list(comparison[0]))
    (table_dir / "Q2_运行摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    (table_dir / "Q2_优化状态.txt").write_text(
        f"{summary['status']}\n"
        f"hard_deadline_violations={summary['hard_deadline_violations']}\n"
        f"first_failure={summary['first_failure']}\n"
        f"global_optimality_proven=False\n", encoding="utf-8")
    (table_dir / "Q2_运行说明.txt").write_text(
        "本方案从已验收原始箱单重新组批，并搜索单区/多区路线及机、电池排程。\n"
        "每段能耗按离开时剩余载荷计算；抵达后先基础交接，再逐箱交接并减载。\n"
        "同型电池初始满电，返航后按两阶段充电模型充至100%才能复用。\n"
        "方案状态以独立验收表为准；PASS仅表示当前启发式方案可行，不证明全局最优。\n"
        "固定FFD基线若为FAIL，方案对照仅作诊断，不能计算有效改进百分比。\n",
        encoding="utf-8")
    if summary["status"] == "PASS":
        (table_dir / "Q2_READY.txt").write_text(
            "Q2 verified feasible; review official CSVs, assumptions and figures before use.\n",
            encoding="utf-8")
    for filename, columns, count in (
        ("Q2_运输架次.csv", SORTIE_COLUMNS, len(sorties)),
        ("Q2_逐箱交付.csv", DELIVERY_COLUMNS, len(deliveries)),
    ):
        saved = pd.read_csv(table_dir / filename, encoding="utf-8-sig")
        if list(saved.columns) != columns or len(saved) != count:
            raise ValueError(f"官方格式表导出复核失败：{filename}")
    return summary


def save_figures(figure_dir: Path, ctx: Context, plan: dict[str, Any],
                 summary: dict[str, Any]) -> None:
    if figure_dir.exists():
        raise FileExistsError(f"不覆盖已有结果：{figure_dir}")
    os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIR / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    figure_dir.mkdir(parents=True)
    sorties, deliveries = plan["sorties"], plan["deliveries"]
    colors = {"A": "#3A80C1", "B": "#27A275", "C": "#DB873F"}
    uav_ids = sorted(u.uav_id for u in ctx.uavs)
    battery_ids = sorted(b.battery_id for b in ctx.batteries)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for s in sorties:
        kind = s["type_id"]
        axes[0].barh(uav_ids.index(s["uav_id"]),
                     (s["return_s"] - s["start_s"]) / 3600,
                     left=s["start_s"] / 3600, color=colors[kind], height=0.65)
        row = battery_ids.index(s["battery_id"])
        axes[1].barh(row, (s["return_s"] - s["start_s"]) / 3600,
                     left=s["start_s"] / 3600, color=colors[kind], height=0.65)
        axes[1].barh(row, s["charge_s"] / 3600,
                     left=s["return_s"] / 3600, color="#ACB4BB", height=0.65)
    for ax, labels in zip(axes, (uav_ids, battery_ids)):
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Transport UAV")
    axes[1].set_ylabel("Battery")
    axes[1].set_xlabel("Time since start (h)")
    fig.legend(handles=[Patch(facecolor=color, label=f"Type {kind}")
                        for kind, color in colors.items()] +
               [Patch(facecolor="#ACB4BB", label="Charging")],
               loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.11, left=0.13, right=0.98, top=0.97, hspace=0.22)
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q2_机电池时间线.{extension}", dpi=220)
    plt.close(fig)

    hard = sorted((d for d in deliveries if d["hard_slack_s"] is not None),
                  key=lambda d: (d["hard_slack_s"], d["box_id"]))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(hard)), [d["hard_slack_s"] / 60 for d in hard],
           color=["#C64F4F" if d["hard_slack_s"] < -TOL else "#3A80C1"
                  for d in hard])
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xlabel("Hard-deadline boxes ordered by slack")
    ax.set_ylabel("Deadline slack (min)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q2_硬时限余量.{extension}", dpi=220)
    plt.close(fig)
    save_delivery_figures(figure_dir, deliveries, plt)
    save_route_figure(figure_dir, ctx.data_run, sorties, plt)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for kind in sorted(colors):
        chosen = [s for s in sorties if s["type_id"] == kind]
        ax.scatter([s["return_s"] / 3600 for s in chosen],
                   [100 * s["return_soc"] for s in chosen],
                   label=f"Type {kind}", color=colors[kind], s=35)
    ax.axhline(20, ls="--", color="#444444", lw=0.9, label="20% minimum")
    ax.set_xlabel("Return time (h)")
    ax.set_ylabel("Return SOC (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q2_返航SOC.{extension}", dpi=220)
    plt.close(fig)
    (figure_dir / "图表说明.txt").write_text(
        "机电池时间线：彩色为架次占用，灰色为电池返航后的充电。\n"
        "硬时限余量：红色表示超时；零线以上才满足截止。\n"
        "逐箱交付进度：按逐箱交接完成时刻累计，虚线为硬截止。\n"
        "交付硬时限对照：对角线下方满足硬截止。\n"
        "运输路线图：投影坐标中的箭头为飞行方向，多区路线数字为访问顺序；重复路线合并。\n"
        "返航SOC：每架次返航时电量，虚线为20%下限。\n"
        f"本次独立验收状态：{summary['status']}。启发式搜索不证明全局最优。\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="问题二：多区路线与机电池联合启发式优化")
    parser.add_argument("--data-run", type=Path, help="已验收数据运行目录；默认最新一次")
    parser.add_argument("--output-root", type=Path, help="默认 code/2_outputs")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--time-limit-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    if args.iterations < 0 or args.time_limit_s <= 0:
        parser.error("iterations 不可为负数，time-limit-s 必须为正")
    data_run = args.data_run.resolve() if args.data_run else find_latest_validated_run()
    if not verify_ready(data_run):
        raise ValueError(f"输入数据未通过哈希验收：{data_run}")
    ctx = Context(data_run)
    plan, seed_plan, search = optimize(ctx, args.iterations, args.time_limit_s, args.seed)
    checks = check_plan(ctx, plan)
    if not verify_ready(data_run):
        raise ValueError("求解期间输入数据发生改变")
    output_root = args.output_root.resolve() if args.output_root else CODE_DIR / "2_outputs"
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    method_dir = output_root / "1_optimize"
    table_dir = method_dir / f"q2_opt_{stamp}_table"
    figure_dir = method_dir / f"q2_opt_{stamp}_figure"
    if table_dir.exists() or figure_dir.exists():
        raise FileExistsError("结果目录已存在，请稍后重试或指定新的 --output-root")
    summary = save_tables(table_dir, ctx, plan, seed_plan, search, checks, output_root)
    print(f"tables: {table_dir}", flush=True)
    save_figures(figure_dir, ctx, plan, summary)
    print(f"figures: {figure_dir}", flush=True)
    print(f"Q2 optimization {summary['status']}: {summary['sorties']} sorties, "
          f"{summary['delivered_boxes']} boxes, "
          f"{summary['hard_deadline_violations']} hard-deadline violations")
    print(f"makespan={summary['makespan_s']:.3f}s, "
          f"energy={summary['total_energy_kwh']:.6f}kWh; "
          "heuristic solution, global optimum not proven")
    if summary["first_failure"]:
        print(f"first failure: {summary['first_failure']}")


if __name__ == "__main__":
    main()
