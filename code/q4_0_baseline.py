"""问题四基线：冻结 Q3 任务，原子块贪心分组并核算独立资源峰值。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from _0_pipeline import verify_ready


CODE_DIR = Path(__file__).resolve().parent
ZONE_PATTERN = re.compile(r"S\d{3}")
RESOURCE_ORDER = (
    "UAV_A", "UAV_B", "UAV_C",
    "BAT_A", "BAT_B", "BAT_C",
    "RELAY_UAV", "RELAY_ENERGY",
)
Q4_COLUMNS = (
    "K（2或3）", "任务组编号", "服务区列表",
    "A型运输无人机数", "B型运输无人机数", "C型运输无人机数",
    "A型电池组数", "B型电池组数", "C型电池组数",
    "中继无人机数", "中继能源组件数",
)
Q3_FILES = (
    "Q3_运输架次.csv", "Q3_逐箱交付.csv",
    "Q3_中继架次.csv", "Q3_通信保障.csv",
)


@dataclass(frozen=True)
class Transport:
    task_id: str
    type_id: str
    start_s: float
    return_s: float
    energy_kwh: float
    route: tuple[str, ...]
    uav_id: str = ""
    battery_id: str = ""


@dataclass(frozen=True)
class Relay:
    task_id: str
    start_s: float
    link_ready_s: float
    service_end_s: float
    return_s: float
    energy_kwh: float
    uav_id: str = ""
    energy_unit_id: str = ""


@dataclass(frozen=True)
class Occupancy:
    kind: str
    task_id: str
    resource: str
    start_s: float
    end_s: float
    end_soc: float | None = None
    charge_s: float = 0.0


class UnionFind:
    def __init__(self, zones: set[str]) -> None:
        self.parent = {zone: zone for zone in zones}

    def find(self, zone: str) -> str:
        parent = self.parent[zone]
        if parent != zone:
            self.parent[zone] = self.find(parent)
        return self.parent[zone]

    def union_many(self, zones: set[str]) -> None:
        ordered = sorted(zones)
        for zone in ordered[1:]:
            self.parent[self.find(zone)] = self.find(ordered[0])


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少输入文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} 缺少列：{sorted(missing)}")
        return [
            {key: (value or "").strip() for key, value in row.items() if key is not None}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]


def number(value: str, label: str, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 不是数值：{value!r}") from exc
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{label} 不在有效范围：{value!r}")
    return parsed


def route_zones(value: str, known_zones: set[str], task_id: str) -> tuple[str, ...]:
    route = tuple(ZONE_PATTERN.findall(value.upper()))
    if not route:
        raise ValueError(f"运输架次 {task_id} 没有可识别的服务区访问顺序")
    unknown = set(route) - known_zones
    if unknown:
        raise ValueError(f"运输架次 {task_id} 含未知服务区：{sorted(unknown)}")
    return route


def charge_time_s(end_soc: float, full_charge_s: float) -> float:
    """附件的两段充电规则，按返航 SOC 计算补至 100% 的时间。"""
    if not -1e-9 <= end_soc <= 1 + 1e-9:
        raise ValueError(f"返航 SOC 越界：{end_soc}")
    soc = min(1.0, max(0.0, end_soc))
    if soc < 0.9:
        return full_charge_s * (0.65 * (0.9 - soc) / 0.9 + 0.35)
    return full_charge_s * 0.35 * (1.0 - soc) / 0.1


def peak_occupancy(intervals: list[Occupancy]) -> int:
    """半开区间 [start,end)；同一时刻释放的资源可立即再用。"""
    events: dict[float, int] = defaultdict(int)
    for item in intervals:
        if item.end_s <= item.start_s:
            raise ValueError(f"{item.task_id} 的 {item.resource} 占用区间无效")
        events[item.start_s] += 1
        events[item.end_s] -= 1
    active = peak = 0
    for moment in sorted(events):
        active += events[moment]
        if active < 0:
            raise ValueError("资源事件扫描出现负占用")
        peak = max(peak, active)
    if active:
        raise ValueError("资源事件扫描结束后仍有占用")
    return peak


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_data_run() -> Path:
    for path in sorted((CODE_DIR / "0_outputs").glob("run_*"), reverse=True):
        if verify_ready(path):
            return path
    raise FileNotFoundError("没有已验收的数据目录；请先运行 0_0_run_all.py")


def load_clean_data(data_run: Path) -> dict:
    if not verify_ready(data_run):
        raise ValueError(f"数据目录未通过验收：{data_run}")
    clean = data_run / "clean"
    nodes = read_csv(clean / "nodes.csv", ("node_id",))
    zones = {row["node_id"] for row in nodes if row["node_id"].startswith("S")}
    boxes_rows = read_csv(clean / "boxes.csv", ("box_id", "zone_id", "mass_kg", "priority"))
    boxes = {row["box_id"]: row for row in boxes_rows}
    if len(zones) != 15 or len(boxes_rows) != 80 or len(boxes) != 80:
        raise ValueError("Q4 固定场景必须包含 15 个服务区和 80 个唯一货箱")
    if any(row["zone_id"] not in zones for row in boxes_rows):
        raise ValueError("货箱引用了未知服务区")
    types = {
        row["type_id"]: row
        for row in read_csv(clean / "transport_types.csv",
                            ("type_id", "energy_use_kwh", "return_soc_min"))
    }
    if set(types) != {"A", "B", "C"}:
        raise ValueError("运输机型应为 A/B/C")
    uavs = read_csv(clean / "transport_uavs.csv", ("uav_id", "type_id"))
    battery_units = read_csv(clean / "battery_units.csv", ("battery_id", "type_id"))
    batteries = read_csv(clean / "battery_pool.csv",
                         ("type_id", "stock_qty", "full_charge_s"))
    relay_uavs = read_csv(clean / "relay_uavs.csv", ("uav_id",))
    relay_units = read_csv(clean / "relay_energy_units.csv", ("unit_id",))
    relay_pool = read_csv(clean / "relay_energy_pool.csv",
                          ("stock_qty", "full_charge_s"))
    relay_types = read_csv(clean / "relay_type.csv",
                           ("energy_use_kwh", "return_soc_min", "turnaround_s"))
    if len(relay_pool) != 1 or len(relay_types) != 1:
        raise ValueError("中继资源表应各只有一行")
    battery_by_type = {row["type_id"]: row for row in batteries}
    if set(battery_by_type) != set(types):
        raise ValueError("电池池机型与运输机型不一致")
    stock = {
        **{f"UAV_{type_id}": sum(u["type_id"] == type_id for u in uavs)
           for type_id in types},
        **{f"BAT_{type_id}": int(number(battery_by_type[type_id]["stock_qty"],
                                       f"{type_id} 电池库存")) for type_id in types},
        "RELAY_UAV": len(relay_uavs),
        "RELAY_ENERGY": int(number(relay_pool[0]["stock_qty"], "中继组件库存")),
    }
    return {
        "zones": zones, "boxes": boxes, "types": types,
        "battery_by_type": battery_by_type,
        "relay_type": relay_types[0], "relay_pool": relay_pool[0],
        "uav_type_by_id": {row["uav_id"]: row["type_id"] for row in uavs},
        "battery_type_by_id": {
            row["battery_id"]: row["type_id"] for row in battery_units
        },
        "relay_uav_ids": {row["uav_id"] for row in relay_uavs},
        "relay_energy_ids": {row["unit_id"] for row in relay_units},
        "stock": stock,
    }


def load_q3(q3_dir: Path, clean: dict) -> tuple[dict[str, Transport], dict[str, Relay],
                                                dict[str, set[str]]]:
    zones = clean["zones"]
    t_rows = read_csv(q3_dir / Q3_FILES[0],
                      ("架次编号", "机型编号", "开始时刻（s）", "访问服务区顺序",
                       "返回O01时刻（s）", "架次能耗（kWh）"))
    d_rows = read_csv(q3_dir / Q3_FILES[1],
                      ("货箱编号", "架次编号", "服务区编号", "交付完成时刻（s）"))
    r_rows = read_csv(q3_dir / Q3_FILES[2],
                      ("中继架次编号", "开始时刻（s）", "建链完成时刻（s）",
                       "服务结束时刻（s）", "返回O01时刻（s）", "架次能耗（kWh）"))
    c_rows = read_csv(q3_dir / Q3_FILES[3],
                      ("运输架次编号", "开始时刻（s）", "结束时刻（s）", "中继架次编号"))
    transports: dict[str, Transport] = {}
    for row in t_rows:
        task_id = row["架次编号"]
        if not task_id or task_id in transports:
            raise ValueError(f"运输架次编号空缺或重复：{task_id!r}")
        type_id = row["机型编号"]
        if type_id not in clean["types"]:
            raise ValueError(f"{task_id} 的机型未知：{type_id}")
        start = number(row["开始时刻（s）"], f"{task_id} 开始时刻")
        end = number(row["返回O01时刻（s）"], f"{task_id} 返回时刻")
        energy = number(row["架次能耗（kWh）"], f"{task_id} 能耗")
        if end <= start:
            raise ValueError(f"{task_id} 返回不晚于开始")
        route = route_zones(row["访问服务区顺序"], zones, task_id)
        uav_id = row.get("无人机编号", "")
        battery_id = row.get("电池编号", "")
        if uav_id and uav_id not in clean["uav_type_by_id"]:
            raise ValueError(f"{task_id} 引用未知运输机编号：{uav_id}")
        if uav_id and clean["uav_type_by_id"][uav_id] != type_id:
            raise ValueError(f"{task_id} 实体机型号与机型编号不一致")
        if battery_id and battery_id not in clean["battery_type_by_id"]:
            raise ValueError(f"{task_id} 引用未知电池编号：{battery_id}")
        if battery_id and clean["battery_type_by_id"][battery_id] != type_id:
            raise ValueError(f"{task_id} 电池型号与机型编号不一致")
        transports[task_id] = Transport(
            task_id, type_id, start, end, energy, route, uav_id, battery_id
        )
    if not transports:
        raise ValueError("Q3 运输架次表为空")

    delivered: dict[str, set[str]] = defaultdict(set)
    seen_boxes: set[str] = set()
    for row in d_rows:
        box_id, task_id, zone = row["货箱编号"], row["架次编号"], row["服务区编号"]
        if box_id not in clean["boxes"] or box_id in seen_boxes:
            raise ValueError(f"货箱不存在或重复交付：{box_id}")
        if task_id not in transports:
            raise ValueError(f"货箱 {box_id} 指向未知运输架次 {task_id}")
        if zone != clean["boxes"][box_id]["zone_id"] or zone not in transports[task_id].route:
            raise ValueError(f"货箱 {box_id} 的服务区与原始数据或运输路线不一致")
        delivered_at = number(row["交付完成时刻（s）"], f"{box_id} 交付时刻")
        task = transports[task_id]
        if not task.start_s <= delivered_at <= task.return_s:
            raise ValueError(f"货箱 {box_id} 的交付时刻超出架次区间")
        seen_boxes.add(box_id)
        delivered[task_id].add(zone)
    if seen_boxes != set(clean["boxes"]):
        raise ValueError(f"Q3 交付表未恰好覆盖 80 箱；缺失 {len(set(clean['boxes']) - seen_boxes)} 箱")
    for task_id, task in transports.items():
        if delivered[task_id] != set(task.route):
            raise ValueError(f"{task_id} 的访问服务区与实际交付服务区不一致")

    relays: dict[str, Relay] = {}
    for row in r_rows:
        task_id = row["中继架次编号"]
        if not task_id or task_id in relays:
            raise ValueError(f"中继架次编号空缺或重复：{task_id!r}")
        start = number(row["开始时刻（s）"], f"{task_id} 开始时刻")
        ready = number(row["建链完成时刻（s）"], f"{task_id} 建链时刻")
        service_end = number(row["服务结束时刻（s）"], f"{task_id} 服务结束时刻")
        end = number(row["返回O01时刻（s）"], f"{task_id} 返回时刻")
        if not start < ready <= service_end <= end:
            raise ValueError(f"{task_id} 的建链、服务、返航时序无效")
        uav_id = row.get("中继无人机编号", "")
        energy_unit_id = row.get("能源组件编号", "")
        if uav_id and uav_id not in clean["relay_uav_ids"]:
            raise ValueError(f"{task_id} 引用未知中继无人机编号：{uav_id}")
        if energy_unit_id and energy_unit_id not in clean["relay_energy_ids"]:
            raise ValueError(f"{task_id} 引用未知中继能源组件编号：{energy_unit_id}")
        relays[task_id] = Relay(
            task_id, start, ready, service_end, end,
            number(row["架次能耗（kWh）"], f"{task_id} 能耗"),
            uav_id, energy_unit_id,
        )
    if set(transports) & set(relays):
        raise ValueError("运输架次与中继架次编号不能重名")

    relay_to_transports: dict[str, set[str]] = defaultdict(set)
    comm_seen: set[str] = set()
    for row in c_rows:
        t_id, r_id = row["运输架次编号"], row["中继架次编号"]
        if r_id in {"-", "—", "/", "无", "直连"}:
            r_id = ""
        if t_id not in transports:
            raise ValueError(f"通信保障引用未知运输架次：{t_id}")
        if "中继" in row.get("保障方式", "") and not r_id:
            raise ValueError(f"{t_id} 标为中继保障但缺少中继架次编号")
        comm_seen.add(t_id)
        start = number(row["开始时刻（s）"], f"{t_id} 通信开始")
        end = number(row["结束时刻（s）"], f"{t_id} 通信结束")
        t = transports[t_id]
        if not t.start_s <= start < end <= t.return_s:
            raise ValueError(f"{t_id} 的通信区间超出运输架次")
        if r_id:
            if r_id not in relays:
                raise ValueError(f"通信保障引用未知中继架次：{r_id}")
            relay = relays[r_id]
            if not relay.link_ready_s <= start < end <= relay.service_end_s:
                raise ValueError(f"{r_id} 未覆盖 {t_id} 声明的通信区间")
            relay_to_transports[r_id].add(t_id)
    if comm_seen != set(transports):
        raise ValueError(f"通信保障缺少 {len(set(transports) - comm_seen)} 个运输架次")
    for task_id in relays:
        if not relay_to_transports[task_id]:
            raise ValueError(f"中继架次 {task_id} 未关联任何运输架次，无法分组")
    return transports, relays, relay_to_transports


def atomic_blocks(zones: set[str], boxes: dict[str, dict[str, str]],
                  transports: dict[str, Transport], relays: dict[str, Relay],
                  relay_to_transports: dict[str, set[str]]) -> tuple[list[dict], dict[str, str]]:
    union = UnionFind(zones)
    for task in transports.values():
        union.union_many(set(task.route))
    for t_ids in relay_to_transports.values():
        union.union_many({zone for t_id in t_ids for zone in transports[t_id].route})
    members: dict[str, set[str]] = defaultdict(set)
    for zone in zones:
        members[union.find(zone)].add(zone)
    blocks = []
    zone_to_block = {}
    for index, zone_set in enumerate(sorted(members.values(), key=lambda x: min(x)), 1):
        block_id = f"B{index:02d}"
        zone_to_block.update({zone: block_id for zone in zone_set})
        block_boxes = [row for row in boxes.values() if row["zone_id"] in zone_set]
        blocks.append({
            "block_id": block_id, "zones": sorted(zone_set),
            "box_count": len(block_boxes),
            "mass_kg": sum(number(row["mass_kg"], "货箱质量") for row in block_boxes),
            "urgent_box_count": sum(
                row.get("medical_bool") == "True" or row.get("is_first_batch") == "True"
                for row in block_boxes
            ),
            "priority_sum": sum(number(row["priority"], "货箱优先系数")
                                for row in block_boxes),
            "transport_ids": sorted(t.task_id for t in transports.values()
                                    if set(t.route) & zone_set),
            "relay_ids": sorted(r_id for r_id in relays
                                if any(set(transports[t_id].route) & zone_set
                                       for t_id in relay_to_transports[r_id])),
        })
    return blocks, zone_to_block


def greedy_partition(blocks: list[dict], k: int) -> dict[str, str]:
    if len(blocks) < k:
        raise ValueError(
            f"不可拆原子块只有 {len(blocks)} 个，无法划为 {k} 个非空组；"
            "须返回问题三选择可分的联合方案"
        )
    ordered = sorted(blocks, key=lambda b: (-b["box_count"], b["zones"][0]))
    assignment: dict[str, str] = {}
    totals = {f"G{i}": 0 for i in range(1, k + 1)}
    for index, block in enumerate(ordered):
        group = f"G{index + 1}" if index < k else min(
            totals, key=lambda key: (totals[key], int(key[1:]))
        )
        assignment[block["block_id"]] = group
        totals[group] += block["box_count"]
    return assignment


def occupancies(clean: dict, transports: dict[str, Transport],
                relays: dict[str, Relay]) -> list[Occupancy]:
    intervals = []
    for task in transports.values():
        type_row = clean["types"][task.type_id]
        capacity = number(type_row["energy_use_kwh"], "运输电池容量", 1e-12)
        soc = 1.0 - task.energy_kwh / capacity
        reserve = number(type_row["return_soc_min"], "运输返航余量")
        if soc + 1e-8 < reserve:
            raise ValueError(f"{task.task_id} 返航 SOC {soc:.4f} 低于机型余量 {reserve:.4f}")
        full_s = number(clean["battery_by_type"][task.type_id]["full_charge_s"],
                        "运输电池满充时间")
        charge_s = charge_time_s(soc, full_s)
        intervals.extend((
            Occupancy("transport", task.task_id, f"UAV_{task.type_id}",
                      task.start_s, task.return_s),
            Occupancy("transport", task.task_id, f"BAT_{task.type_id}",
                      task.start_s, task.return_s + charge_s, soc, charge_s),
        ))
    relay_type = clean["relay_type"]
    relay_capacity = number(relay_type["energy_use_kwh"], "中继组件容量", 1e-12)
    relay_reserve = number(relay_type["return_soc_min"], "中继返航余量")
    relay_charge_full = number(clean["relay_pool"]["full_charge_s"], "中继组件满充时间")
    turnaround = number(relay_type["turnaround_s"], "中继机周转时间")
    for task in relays.values():
        soc = 1.0 - task.energy_kwh / relay_capacity
        if soc + 1e-8 < relay_reserve:
            raise ValueError(f"{task.task_id} 返航 SOC {soc:.4f} 低于中继余量 {relay_reserve:.4f}")
        charge_s = charge_time_s(soc, relay_charge_full)
        intervals.extend((
            Occupancy("relay", task.task_id, "RELAY_UAV",
                      task.start_s, task.return_s + turnaround),
            Occupancy("relay", task.task_id, "RELAY_ENERGY",
                      task.start_s, task.return_s + charge_s, soc, charge_s),
        ))
    return intervals


def calculate(clean: dict, transports: dict[str, Transport],
              relays: dict[str, Relay], relay_to_transports: dict[str, set[str]]) -> dict:
    blocks, zone_to_block = atomic_blocks(
        clean["zones"], clean["boxes"], transports, relays, relay_to_transports
    )
    intervals = occupancies(clean, transports, relays)
    ids = {
        **{
            (task.task_id, f"UAV_{task.type_id}"): task.uav_id
            for task in transports.values()
        },
        **{
            (task.task_id, f"BAT_{task.type_id}"): task.battery_id
            for task in transports.values()
        },
        **{(task.task_id, "RELAY_UAV"): task.uav_id for task in relays.values()},
        **{
            (task.task_id, "RELAY_ENERGY"): task.energy_unit_id
            for task in relays.values()
        },
    }
    id_values = list(ids.values())
    if any(id_values) and not all(id_values):
        raise ValueError("Q3 实体机、电池和组件编号只能全部提供或全部省略")
    has_original_ids = bool(id_values) and all(id_values)
    if has_original_ids:
        by_original_id: dict[tuple[str, str], list[Occupancy]] = defaultdict(list)
        for item in intervals:
            by_original_id[(item.resource, ids[item.task_id, item.resource])].append(item)
        for (resource, unit_id), uses in by_original_id.items():
            if peak_occupancy(uses) > 1:
                raise ValueError(f"Q3 原编号资源占用冲突：{resource}/{unit_id}")
    global_peak = {
        resource: peak_occupancy([x for x in intervals if x.resource == resource])
        for resource in RESOURCE_ORDER
    }
    solutions = {}
    for k in (2, 3):
        assignment = greedy_partition(blocks, k)
        zone_group = {zone: assignment[zone_to_block[zone]] for zone in clean["zones"]}
        transport_group = {}
        for t_id, task in transports.items():
            groups = {zone_group[zone] for zone in task.route}
            if len(groups) != 1:
                raise ValueError(f"{t_id} 跨任务组")
            transport_group[t_id] = groups.pop()
        relay_group = {}
        for r_id, t_ids in relay_to_transports.items():
            groups = {transport_group[t_id] for t_id in t_ids}
            if len(groups) != 1:
                raise ValueError(f"{r_id} 跨任务组；保守主口径不复制中继架次")
            relay_group[r_id] = groups.pop()
        task_group = {**transport_group, **relay_group}
        group_zones = {
            f"G{i}": sorted(zone for zone, group in zone_group.items()
                            if group == f"G{i}")
            for i in range(1, k + 1)
        }
        if any(not zone_set for zone_set in group_zones.values()):
            raise ValueError(f"K={k} 出现空任务组")
        demand = {
            group: {
                resource: peak_occupancy([
                    item for item in intervals
                    if item.resource == resource and task_group[item.task_id] == group
                ])
                for resource in RESOURCE_ORDER
            }
            for group in group_zones
        }
        totals = {resource: sum(demand[group][resource] for group in group_zones)
                  for resource in RESOURCE_ORDER}
        original_id_demand = None
        if has_original_ids:
            original_id_demand = {
                group: {
                    resource: len({
                        ids[item.task_id, item.resource]
                        for item in intervals
                        if item.resource == resource and task_group[item.task_id] == group
                    })
                    for resource in RESOURCE_ORDER
                }
                for group in group_zones
            }
        group_work = {}
        for group, zone_set in group_zones.items():
            zone_set_ = set(zone_set)
            group_boxes = [row for row in clean["boxes"].values()
                           if row["zone_id"] in zone_set_]
            t_tasks = [task for task in transports.values()
                       if transport_group[task.task_id] == group]
            r_tasks = [task for task in relays.values()
                       if relay_group[task.task_id] == group]
            group_work[group] = {
                "box_count": len(group_boxes),
                "mass_kg": sum(number(row["mass_kg"], "货箱质量")
                               for row in group_boxes),
                "urgent_box_count": sum(
                    row.get("medical_bool") == "True" or row.get("is_first_batch") == "True"
                    for row in group_boxes
                ),
                "priority_sum": sum(number(row["priority"], "货箱优先系数")
                                    for row in group_boxes),
                "transport_sorties": len(t_tasks),
                "relay_sorties": len(r_tasks),
                "transport_task_s": sum(t.return_s - t.start_s for t in t_tasks),
                "relay_task_s": sum(r.return_s - r.start_s for r in r_tasks),
            }
        solutions[k] = {
            "assignment": assignment, "zone_group": zone_group,
            "transport_group": transport_group, "relay_group": relay_group,
            "task_group": task_group, "group_zones": group_zones,
            "demand": demand, "totals": totals, "work": group_work,
            "deficit": {r: max(0, totals[r] - clean["stock"][r])
                        for r in RESOURCE_ORDER},
            "redundancy": {r: max(0, clean["stock"][r] - totals[r])
                           for r in RESOURCE_ORDER},
            "original_id_demand": original_id_demand,
        }
    return {
        "blocks": blocks, "intervals": intervals,
        "global_peak": global_peak, "solutions": solutions,
        "has_original_ids": has_original_ids,
    }


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_results(output_dir: Path, q3_dir: Path, data_run: Path, clean: dict,
                 transports: dict[str, Transport], relays: dict[str, Relay],
                 relay_to_transports: dict[str, set[str]], result: dict) -> None:
    blocks, intervals, solutions = (
        result["blocks"], result["intervals"], result["solutions"]
    )
    block_rows = [
        {
            "原子块编号": b["block_id"], "服务区列表": ",".join(b["zones"]),
            "箱数": b["box_count"], "货物质量（kg）": b["mass_kg"],
            "紧急箱数": b["urgent_box_count"], "优先系数合计": b["priority_sum"],
            "运输架次列表": ",".join(b["transport_ids"]),
            "中继架次列表": ",".join(b["relay_ids"]),
            **{
                resource: peak_occupancy([
                    item for item in intervals
                    if item.resource == resource
                    and item.task_id in set(b["transport_ids"] + b["relay_ids"])
                ])
                for resource in RESOURCE_ORDER
            },
        }
        for b in blocks
    ]
    write_csv(output_dir / "Q4_原子块.csv",
              ("原子块编号", "服务区列表", "箱数", "货物质量（kg）",
               "紧急箱数", "优先系数合计", "运输架次列表", "中继架次列表",
               *RESOURCE_ORDER),
              block_rows)
    config_rows, summary_rows, work_rows, task_rows, interval_rows = [], [], [], [], []
    original_id_rows = []
    for k, solution in solutions.items():
        for group, zones in solution["group_zones"].items():
            d = solution["demand"][group]
            config_rows.append(dict(zip(Q4_COLUMNS, (
                k, group, ",".join(zones),
                d["UAV_A"], d["UAV_B"], d["UAV_C"],
                d["BAT_A"], d["BAT_B"], d["BAT_C"],
                d["RELAY_UAV"], d["RELAY_ENERGY"],
            ))))
            if solution["original_id_demand"] is not None:
                original = solution["original_id_demand"][group]
                original_id_rows.append(dict(zip(Q4_COLUMNS, (
                    k, group, ",".join(zones),
                    original["UAV_A"], original["UAV_B"], original["UAV_C"],
                    original["BAT_A"], original["BAT_B"], original["BAT_C"],
                    original["RELAY_UAV"], original["RELAY_ENERGY"],
                ))))
            work_rows.append({
                "K": k, "任务组编号": group, "服务区列表": ",".join(zones),
                **solution["work"][group],
            })
        for resource in RESOURCE_ORDER:
            stock = clean["stock"][resource]
            global_peak = result["global_peak"][resource]
            total = solution["totals"][resource]
            if total <= stock:
                reason = "无缺口"
            elif global_peak > stock and total > global_peak:
                reason = "全局任务峰值已超库存，分组独立又增加需求"
            elif global_peak > stock:
                reason = "全局任务峰值已超库存"
            else:
                reason = "分组独立执行使各组峰值之和超过库存"
            summary_rows.append({
                "K": k, "资源类型": resource, "现有库存": stock,
                "全局共享峰值": global_peak,
                "分组独立需求": total,
                "分组额外需求": total - global_peak,
                "库存缺口": solution["deficit"][resource],
                "库存冗余": solution["redundancy"][resource],
                "缺口原因": reason,
            })
        for task_id, group in sorted(solution["task_group"].items()):
            task = transports.get(task_id) or relays[task_id]
            task_rows.append({
                "K": k, "任务类型": "运输" if task_id in transports else "中继",
                "任务编号": task_id, "任务组编号": group,
                "开始时刻（s）": task.start_s, "返回O01时刻（s）": task.return_s,
                "原无人机编号": task.uav_id,
                "原电池或组件编号": (
                    task.battery_id if task_id in transports else task.energy_unit_id
                ),
                "访问服务区或保障运输架次": (
                    ",".join(task.route) if task_id in transports
                    else ",".join(sorted(relay_to_transports[task_id]))
                ),
            })
        for item in intervals:
            interval_rows.append({
                "K": k, "任务组编号": solution["task_group"][item.task_id],
                "任务类型": item.kind, "任务编号": item.task_id,
                "资源类型": item.resource, "占用开始（s）": item.start_s,
                "占用结束（s）": item.end_s,
                "返航SOC": "" if item.end_soc is None else round(item.end_soc, 10),
                "充满时间（s）": item.charge_s,
            })
    write_csv(output_dir / "Q4_分区配置_基线.csv", Q4_COLUMNS, config_rows)
    if original_id_rows:
        write_csv(output_dir / "Q4_原编号保守配置_基线.csv",
                  Q4_COLUMNS, original_id_rows)
    write_csv(output_dir / "Q4_资源汇总_基线.csv",
              ("K", "资源类型", "现有库存", "全局共享峰值", "分组独立需求",
               "分组额外需求", "库存缺口", "库存冗余", "缺口原因"), summary_rows)
    write_csv(output_dir / "Q4_组间工作量_基线.csv",
              ("K", "任务组编号", "服务区列表", "box_count", "mass_kg",
               "urgent_box_count", "priority_sum", "transport_sorties",
               "relay_sorties", "transport_task_s", "relay_task_s"), work_rows)
    write_csv(output_dir / "Q4_任务归组_基线.csv",
              ("K", "任务类型", "任务编号", "任务组编号", "开始时刻（s）",
               "返回O01时刻（s）", "原无人机编号", "原电池或组件编号",
               "访问服务区或保障运输架次"), task_rows)
    write_csv(output_dir / "Q4_资源占用区间_基线.csv",
              ("K", "任务组编号", "任务类型", "任务编号", "资源类型",
               "占用开始（s）", "占用结束（s）", "返航SOC", "充满时间（s）"),
              interval_rows)
    checks = [
        {"检查项": "15个服务区各归一组", "状态": "PASS",
         "证据": f"K={k}: {len(solution['zone_group'])} 个唯一服务区"}
        for k, solution in solutions.items()
    ] + [
        {"检查项": "全部运输与中继任务各归一组", "状态": "PASS",
         "证据": f"K={k}: {len(solution['task_group'])} 个任务"}
        for k, solution in solutions.items()
    ] + [
        {"检查项": "80箱唯一交付且 Q3 任务不改时刻", "状态": "PASS",
         "证据": f"运输 {len(transports)} 架次，中继 {len(relays)} 架次；仅分组"}
    ]
    write_csv(output_dir / "Q4_验收检查.csv", ("检查项", "状态", "证据"), checks)
    source_ready = (q3_dir / "Q3_READY.txt").is_file()
    summary = {
        "status": "PASS" if source_ready else "Q3_SOURCE_NOT_MARKED_READY",
        "scope": "问题四基线；固定 Q3 任务，不重排任务时刻或复制中继架次",
        "q3_source_dir": str(q3_dir.resolve()),
        "q3_ready_marker_present": source_ready,
        "data_run": str(data_run.resolve()),
        "source_sha256": {name: sha256(q3_dir / name) for name in Q3_FILES},
        "atomic_block_count": len(blocks),
        "transport_sorties": len(transports),
        "relay_sorties": len(relays),
        "global_stock": clean["stock"],
        "global_shared_peak": result["global_peak"],
        "original_ids_complete": result["has_original_ids"],
        "by_k": {
            str(k): {
                "groups": solution["group_zones"],
                "independent_demand": solution["totals"],
                "stock_deficit": solution["deficit"],
                "stock_redundancy": solution["redundancy"],
                "box_counts": {
                    group: work["box_count"] for group, work in solution["work"].items()
                },
                "stock_feasible": not any(solution["deficit"].values()),
                "original_id_copy_demand": (
                    {
                        resource: sum(
                            solution["original_id_demand"][group][resource]
                            for group in solution["group_zones"]
                        )
                        for resource in RESOURCE_ORDER
                    }
                    if solution["original_id_demand"] is not None else None
                ),
            }
            for k, solution in solutions.items()
        },
        "limitation": (
            "本程序复核 Q3 编号、交付和关联时段；Q3 物理飞行与连续通信须由 Q3 独立验证器提供。"
        ),
    }
    (output_dir / "Q4_运行摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if source_ready:
        (output_dir / "Q4_BASE_READY.txt").write_text(
            "PASS: K=2/3 baseline partitions and resource peaks verified; "
            "stock gaps, if any, are reported separately.\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q3-dir", required=True, type=Path,
                        help="已固定的 Q3 四张 CSV 所在目录")
    parser.add_argument("--data-run", type=Path,
                        help="已验收的数据处理 run 目录；默认最新一次")
    parser.add_argument("--output-dir", type=Path,
                        help="基线输出目录；默认 code/4_outputs/q4_base_时间戳_table")
    args = parser.parse_args()
    q3_dir = args.q3_dir.resolve()
    data_run = args.data_run.resolve() if args.data_run else latest_data_run()
    output_dir = args.output_dir.resolve() if args.output_dir else (
        CODE_DIR / "4_outputs" / datetime.now().strftime("q4_base_%y%m%d_%H%M%S_table")
    )
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    clean = load_clean_data(data_run)
    transports, relays, relay_to_transports = load_q3(q3_dir, clean)
    output_dir.mkdir(parents=True)
    try:
        result = calculate(clean, transports, relays, relay_to_transports)
        save_results(output_dir, q3_dir, data_run, clean, transports, relays,
                     relay_to_transports, result)
    except Exception as exc:
        (output_dir / "Q4_失败诊断.json").write_text(
            json.dumps({"status": "FAILED", "reason": str(exc)},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    print(f"Q4 baseline PASS: {len(result['blocks'])} atomic blocks; "
          f"K=2/3 completed; output={output_dir}")
    if not (q3_dir / "Q3_READY.txt").is_file():
        print("注意：Q3_READY.txt 缺失，结果未标记为正式可用。")


if __name__ == "__main__":
    main()
