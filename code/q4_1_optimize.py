"""问题四方案 1：固定问题三方案 1，精确枚举 2/3 个任务组。

运行方式：python code/q4_1_optimize.py
可用 --q3-table 指定已通过验收的问题三方案一表格目录。

保持原货箱组批、路线、绝对时刻及通信保障关系。只允许在各组内部
重新编号同型实体机、电池和中继能源组件；另输出原编号完全保留的
保守资源需求。资源配置数是固定区间的最大并发数，未重新优化问题三。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from heapq import heappop, heappush
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
RESOURCE_NAMES = {
    "UA": "A型运输无人机", "UB": "B型运输无人机", "UC": "C型运输无人机",
    "BA": "A型共享电池", "BB": "B型共享电池", "BC": "C型共享电池",
    "RU": "中继无人机", "RE": "中继能源组件",
}
RESOURCE_KEYS = tuple(RESOURCE_NAMES)
TIME_EPS = 1e-7


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少输入表：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_by(rows: list[dict[str, str]], key: str, label: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        item_id = row[key].strip()
        check(bool(item_id), f"{label}存在空编号")
        check(item_id not in result, f"{label}编号重复：{item_id}")
        result[item_id] = row
    return result


def number(row: dict[str, str], key: str, label: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{label}的 {key} 缺失或不是数值") from exc
    check(math.isfinite(value), f"{label}的 {key} 不是有限数")
    return value


def latest_q3_table() -> Path:
    base = CODE_DIR / "3_outputs" / "1_optimize"
    candidates = sorted(base.glob("q3_opt1_*_table"), reverse=True)
    for path in candidates:
        if (path / "Q3_READY.txt").is_file() and (path / "Q3_运行摘要.json").is_file():
            try:
                summary = json.loads((path / "Q3_运行摘要.json").read_text(encoding="utf-8-sig"))
            except (ValueError, OSError):
                continue
            if summary.get("status") == "PASS":
                return path
    raise FileNotFoundError("3_outputs/1_optimize 下没有通过验收的方案一结果；请用 --q3-table 指定")


def load_source(table_dir: Path) -> dict[str, Any]:
    table_dir = table_dir.resolve()
    check(table_dir.is_dir(), f"问题三表格目录不存在：{table_dir}")
    ready = table_dir / "Q3_READY.txt"
    check(ready.is_file() and "PASS" in ready.read_text(encoding="utf-8-sig"),
          "问题三结果缺少 Q3_READY.txt 或未标记 PASS")
    summary = json.loads((table_dir / "Q3_运行摘要.json").read_text(encoding="utf-8-sig"))
    check(summary.get("status") == "PASS", "问题三运行摘要未通过验收")
    for key in ("transport_status", "relay_resource_status", "communication_status"):
        check(summary.get(key) == "PASS", f"问题三 {key} 未通过验收")
    check(summary.get("delivered_boxes") == 80, "问题三不是完整 80 箱方案")
    checks = read_csv(table_dir / "Q3_验收检查.csv")
    check(bool(checks) and all(row["status"] == "PASS" for row in checks),
          "问题三验收检查表含未通过项")
    data_run = Path(summary["source"]["data_run"])
    if not (data_run / "clean").is_dir():
        wanted = summary["source"].get("source_manifest_sha256")
        for path in sorted((CODE_DIR / "0_outputs").glob("run_*"), reverse=True):
            meta = path / "meta" / "READY.txt"
            if meta.is_file() and f"manifest_sha256={wanted}" in meta.read_text(encoding="utf-8-sig"):
                data_run = path
                break
    check((data_run / "clean").is_dir(), "找不到问题三对应的 0_outputs 清洗数据")
    source_ready = data_run / "meta" / "READY.txt"
    check(source_ready.is_file(), "原始清洗数据缺少 READY.txt")
    marker = f"manifest_sha256={summary['source']['source_manifest_sha256']}"
    check(marker in source_ready.read_text(encoding="utf-8-sig"),
          "问题三与清洗数据的清单哈希不一致")
    files = (
        "Q3_运输明细.csv", "Q3_逐箱交付.csv", "Q3_通信保障.csv",
        "Q3_中继架次.csv", "Q3_中继资源时间线.csv",
    )
    hashes = {name: hashlib.sha256((table_dir / name).read_bytes()).hexdigest()
              for name in files}
    return {
        "table_dir": table_dir, "data_run": data_run, "summary": summary,
        "hashes": hashes,
        "transport": read_csv(table_dir / files[0]),
        "deliveries": read_csv(table_dir / files[1]),
        "communication": read_csv(table_dir / files[2]),
        "relay": read_csv(table_dir / files[3]),
        "relay_timeline": read_csv(table_dir / files[4]),
    }


def load_inventory(data_run: Path) -> tuple[dict[str, int], dict[str, dict[str, str]]]:
    clean = data_run / "clean"
    source_files = {
        "transport_uav": ("transport_uavs.csv", "uav_id"),
        "transport_battery": ("battery_units.csv", "battery_id"),
        "relay_uav": ("relay_uavs.csv", "uav_id"),
        "relay_unit": ("relay_energy_units.csv", "unit_id"),
    }
    ids: dict[str, dict[str, str]] = {}
    for group, (file_name, id_column) in source_files.items():
        rows = read_csv(clean / file_name)
        indexed = unique_by(rows, id_column, file_name)
        ids[group] = {key: row["type_id"] for key, row in indexed.items()}
    stock = {key: 0 for key in RESOURCE_KEYS}
    for kind in ids["transport_uav"].values():
        check(kind in ("A", "B", "C"), f"未知运输机型：{kind}")
        stock["U" + kind] += 1
    for kind in ids["transport_battery"].values():
        check(kind in ("A", "B", "C"), f"未知电池型号：{kind}")
        stock["B" + kind] += 1
    stock["RU"] = len(ids["relay_uav"])
    stock["RE"] = len(ids["relay_unit"])
    return stock, ids


def build_tasks(source: dict[str, Any], ids: dict[str, dict[str, str]]) -> dict[str, Any]:
    boxes = unique_by(read_csv(source["data_run"] / "clean" / "boxes.csv"),
                      "box_id", "原始货箱")
    zones = sorted({row["zone_id"] for row in boxes.values()})
    check(len(zones) == 15, f"服务区应有 15 个，实为 {len(zones)} 个")
    transport = unique_by(source["transport"], "batch_id", "运输架次")
    relay = unique_by(source["relay"], "中继架次编号", "中继架次")
    timeline = unique_by(source["relay_timeline"], "mission_id", "中继资源时间线")
    check(set(relay) == set(timeline), "中继架次与资源时间线编号不一致")
    check(len(transport) == source["summary"]["transport_sorties"], "运输架次数与摘要不一致")
    check(len(relay) == source["summary"]["relay_sorties"], "中继架次数与摘要不一致")

    sortie_zones: dict[str, set[str]] = defaultdict(set)
    seen_boxes: set[str] = set()
    for row in source["deliveries"]:
        box_id, task_id, zone = row["货箱编号"], row["架次编号"], row["服务区编号"]
        check(box_id in boxes, f"问题三交付表存在未知货箱：{box_id}")
        check(box_id not in seen_boxes, f"问题三交付表货箱重复：{box_id}")
        check(task_id in transport, f"问题三交付表存在未知运输架次：{task_id}")
        check(boxes[box_id]["zone_id"] == zone, f"货箱 {box_id} 的服务区不一致")
        seen_boxes.add(box_id)
        sortie_zones[task_id].add(zone)
    check(seen_boxes == set(boxes), "问题三交付货箱未完整覆盖原始清单")
    check(len(boxes) == source["summary"]["delivered_boxes"],
          "问题三摘要箱数与原始清单不一致")
    check(set(sortie_zones) == set(transport), "运输架次中存在没有交付货箱的任务")

    intervals: dict[str, dict[str, Any]] = {}
    for task_id, row in transport.items():
        kind = row["type_id"]
        check(kind in ("A", "B", "C"), f"架次 {task_id} 机型未知")
        check(ids["transport_uav"].get(row["uav_id"]) == kind,
              f"架次 {task_id} 实体机型号不匹配")
        check(ids["transport_battery"].get(row["battery_id"]) == kind,
              f"架次 {task_id} 电池型号不匹配")
        start = number(row, "start_s", task_id)
        returned = number(row, "return_s", task_id)
        charged = number(row, "charge_end_s", task_id)
        check(0 <= start + TIME_EPS < returned <= charged + TIME_EPS,
              f"架次 {task_id} 运输或充电时间顺序异常")
        listed = {part for part in row["box_ids"].split(";") if part}
        actual = {r["货箱编号"] for r in source["deliveries"] if r["架次编号"] == task_id}
        check(listed == actual, f"架次 {task_id} 的箱清单与逐箱交付不符")
        intervals[task_id] = {"kind": "transport", "zones": sortie_zones[task_id],
                              "resources": (("U" + kind, start, returned, row["uav_id"]),
                                            ("B" + kind, start, charged, row["battery_id"])),
                              "start_s": start, "return_s": returned,
                              "box_ids": sorted(actual),
                              "energy_kwh": number(row, "energy_kwh", task_id),
                              "visit_order": row["visit_order"]}

    relay_zones: dict[str, set[str]] = defaultdict(set)
    for row in source["communication"]:
        task_id, method = row["运输架次编号"], row["保障方式"]
        check(task_id in transport, f"通信保障表存在未知运输架次：{task_id}")
        relay_id = row["中继架次编号"].strip()
        if method == "中继":
            check(relay_id in relay, f"运输 {task_id} 引用了未知中继 {relay_id}")
            relay_zones[relay_id].update(sortie_zones[task_id])
        else:
            check(method == "直连" and not relay_id,
                  f"运输 {task_id} 存在未覆盖或不一致的通信保障记录")
    check(set(relay_zones) == set(relay), "存在未用于通信保障的中继任务；无法按组归属")

    for mission_id, row in relay.items():
        line = timeline[mission_id]
        check(row["中继无人机编号"] == line["uav_id"] and
              row["能源组件编号"] == line["energy_unit_id"],
              f"中继 {mission_id} 的资源编号在两表中不一致")
        check(line["uav_id"] in ids["relay_uav"] and
              line["energy_unit_id"] in ids["relay_unit"],
              f"中继 {mission_id} 使用了库存之外的资源")
        start = number(line, "start_s", mission_id)
        returned = number(line, "return_s", mission_id)
        uav_ready = number(line, "uav_next_ready_s", mission_id)
        unit_ready = number(line, "unit_next_ready_s", mission_id)
        check(0 <= start + TIME_EPS < returned <= min(uav_ready, unit_ready) + TIME_EPS,
              f"中继 {mission_id} 的返航或周转时间顺序异常")
        check(abs(start - number(row, "开始时刻（s）", mission_id)) < 2e-5 and
              abs(returned - number(row, "返回O01时刻（s）", mission_id)) < 2e-5,
              f"中继 {mission_id} 的两表时间不一致")
        intervals[mission_id] = {"kind": "relay", "zones": relay_zones[mission_id],
                                 "resources": (("RU", start, uav_ready, line["uav_id"]),
                                               ("RE", start, unit_ready, line["energy_unit_id"])),
                                 "start_s": start, "return_s": returned,
                                 "box_ids": [],
                                 "energy_kwh": number(row, "架次能耗（kWh）", mission_id),
                                 "visit_order": ""}

    return {"boxes": boxes, "zones": zones, "intervals": intervals,
            "transport": transport, "relay": relay}


def components(tasks: dict[str, Any]) -> list[dict[str, Any]]:
    zones = tasks["zones"]
    parent = {zone: zone for zone in zones}

    def find(zone: str) -> str:
        while parent[zone] != zone:
            parent[zone] = parent[parent[zone]]
            zone = parent[zone]
        return zone

    for task in tasks["intervals"].values():
        members = sorted(task["zones"])
        check(bool(members), "存在未归属服务区的任务")
        for member in members[1:]:
            parent[find(member)] = find(members[0])
    grouped: dict[str, set[str]] = defaultdict(set)
    for zone in zones:
        grouped[find(zone)].add(zone)
    box_counts = Counter(row["zone_id"] for row in tasks["boxes"].values())
    parts = sorted(grouped.values(), key=lambda z: (-len(z),
                    -sum(box_counts[x] for x in z), sorted(z)))
    return [{"id": f"C{i:02d}", "zones": sorted(part),
             "boxes": sum(box_counts[z] for z in part)}
            for i, part in enumerate(parts, 1)]


def partitions(count: int, groups: int) -> list[tuple[int, ...]]:
    if groups > count:
        return []
    answers: list[tuple[int, ...]] = []

    def visit(labels: list[int], greatest: int) -> None:
        if len(labels) == count:
            if greatest + 1 == groups:
                answers.append(tuple(labels))
            return
        for label in range(min(greatest + 1, groups - 1) + 1):
            labels.append(label)
            visit(labels, max(greatest, label))
            labels.pop()

    visit([0], 0)
    return answers


def allocate(intervals: list[tuple[str, float, float, str]]) -> tuple[int, dict[str, int]]:
    """同型区间着色；结束时刻等于开始时刻时允许复用。"""
    busy: list[tuple[float, int]] = []
    free: list[int] = []
    assigned: dict[str, int] = {}
    created = 0
    for task_id, start, end, _old_id in sorted(intervals,
                                                key=lambda x: (x[1], x[2], x[0])):
        check(end > start + TIME_EPS, f"任务 {task_id} 资源占用非正")
        while busy and busy[0][0] <= start + TIME_EPS:
            _, index = heappop(busy)
            heappush(free, index)
        if free:
            index = heappop(free)
        else:
            created += 1
            index = created
        assigned[task_id] = index
        heappush(busy, (end, index))
    # 独立回代所有同号区间，避免只有峰值数字而没有可执行的编号方案。
    by_index: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    for task_id, start, end, _old_id in intervals:
        by_index[assigned[task_id]].append((start, end, task_id))
    for index, rows in by_index.items():
        rows.sort()
        for left, right in zip(rows, rows[1:]):
            check(left[1] <= right[0] + TIME_EPS,
                  f"新资源编号 {index} 的任务 {left[2]}、{right[2]} 时间重叠")
    return created, assigned


def evaluate(labels: tuple[int, ...], parts: list[dict[str, Any]],
             tasks: dict[str, Any], stock: dict[str, int]) -> dict[str, Any]:
    groups = max(labels) + 1
    zone_group = {zone: labels[i] + 1 for i, part in enumerate(parts)
                  for zone in part["zones"]}
    check(set(zone_group) == set(tasks["zones"]), "分区未完整覆盖 15 个服务区")
    group_boxes = Counter(zone_group[row["zone_id"]] for row in tasks["boxes"].values())
    group_task_ids: dict[int, list[str]] = defaultdict(list)
    for task_id, task in tasks["intervals"].items():
        owner = {zone_group[zone] for zone in task["zones"]}
        check(len(owner) == 1, f"分区拆开了问题三的运输或中继任务 {task_id}")
        group_task_ids[owner.pop()].append(task_id)
    check(len(group_task_ids) == groups and len(group_boxes) == groups,
          "分区含空组或空任务组")

    group_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    total = Counter({key: 0 for key in RESOURCE_KEYS})
    fixed_total = Counter({key: 0 for key in RESOURCE_KEYS})
    for group_id in range(1, groups + 1):
        task_ids = sorted(group_task_ids[group_id])
        by_resource: dict[str, list[tuple[str, float, float, str]]] = defaultdict(list)
        for task_id in task_ids:
            for key, start, end, old_id in tasks["intervals"][task_id]["resources"]:
                by_resource[key].append((task_id, start, end, old_id))
        need: dict[str, int] = {}
        fixed: dict[str, int] = {}
        for key in RESOURCE_KEYS:
            occupied = by_resource[key]
            need[key], assigned = allocate(occupied)
            fixed[key] = len({row[3] for row in occupied})
            total[key] += need[key]
            fixed_total[key] += fixed[key]
            for task_id, start, end, old_id in occupied:
                allocation_rows.append({
                    "组号": group_id, "任务编号": task_id,
                    "任务类别": tasks["intervals"][task_id]["kind"],
                    "资源类别": RESOURCE_NAMES[key], "资源代码": key,
                    "新资源编号": f"G{group_id}-{key}{assigned[task_id]:02d}",
                    "原资源编号": old_id, "开始时刻_s": start,
                    "下次可用时刻_s": end,
                })
        group_rows.append({
            "组号": group_id,
            "服务区": sorted(zone for zone, owner in zone_group.items() if owner == group_id),
            "箱数": group_boxes[group_id],
            "运输架次": sum(tasks["intervals"][x]["kind"] == "transport" for x in task_ids),
            "中继架次": sum(tasks["intervals"][x]["kind"] == "relay" for x in task_ids),
            "运输能耗_kWh": sum(tasks["intervals"][x]["energy_kwh"] for x in task_ids
                                  if tasks["intervals"][x]["kind"] == "transport"),
            "中继能耗_kWh": sum(tasks["intervals"][x]["energy_kwh"] for x in task_ids
                                  if tasks["intervals"][x]["kind"] == "relay"),
            "运输机占用_h": sum(end - start for key in ("UA", "UB", "UC")
                                for _, start, end, _ in by_resource[key]) / 3600.0,
            "电池占用_h": sum(end - start for key in ("BA", "BB", "BC")
                               for _, start, end, _ in by_resource[key]) / 3600.0,
            "中继机占用_h": sum(end - start for _, start, end, _ in by_resource["RU"]) / 3600.0,
            "能源组件占用_h": sum(end - start for _, start, end, _ in by_resource["RE"]) / 3600.0,
            "资源最少配置": need, "原编号配置": fixed,
            "任务编号": task_ids,
        })
    check(sum(row["箱数"] for row in group_rows) == len(tasks["boxes"]), "分组箱数合计不正确")
    check(sum(row["运输架次"] for row in group_rows) == len(tasks["transport"]),
          "运输任务归属数量不正确")
    check(sum(row["中继架次"] for row in group_rows) == len(tasks["relay"]),
          "中继任务归属数量不正确")
    gap = {key: max(0, total[key] - stock[key]) for key in RESOURCE_KEYS}
    spare = {key: max(0, stock[key] - total[key]) for key in RESOURCE_KEYS}
    fixed_gap = {key: max(0, fixed_total[key] - stock[key]) for key in RESOURCE_KEYS}
    boxes = [row["箱数"] for row in group_rows]
    transports = [row["运输架次"] for row in group_rows]
    relays = [row["中继架次"] for row in group_rows]
    return {
        "groups": groups, "labels": labels, "zone_group": zone_group,
        "group_rows": group_rows, "allocation_rows": allocation_rows,
        "total": dict(total), "fixed_total": dict(fixed_total),
        "gap": gap, "spare": spare, "fixed_gap": fixed_gap,
        "gap_sum": sum(gap.values()), "fixed_gap_sum": sum(fixed_gap.values()),
        "box_spread": max(boxes) - min(boxes),
        "transport_spread": max(transports) - min(transports),
        "relay_spread": max(relays) - min(relays),
    }


def frontier_flags(schemes: list[dict[str, Any]]) -> None:
    for candidate in schemes:
        score = (candidate["gap_sum"], candidate["box_spread"],
                 candidate["redundancy_sum"])
        candidate["pareto"] = not any(
            other is not candidate and
            (other["gap_sum"], other["box_spread"], other["redundancy_sum"]) != score and
            all(a <= b for a, b in zip(
                (other["gap_sum"], other["box_spread"], other["redundancy_sum"]), score))
            for other in schemes if other["groups"] == candidate["groups"])


def compact_resources(values: dict[str, int]) -> str:
    return "; ".join(f"{key}:{values[key]}" for key in RESOURCE_KEYS if values[key]) or "无"


def create_figures(figure_dir: Path, schemes: list[dict[str, Any]],
                   representatives: list[tuple[str, dict[str, Any]]],
                   stock: dict[str, int]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIR / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False})
    figure_dir.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, count in zip(axes, (2, 3)):
        subset = [row for row in schemes if row["groups"] == count]
        point = ax.scatter([x["box_spread"] for x in subset],
                           [x["gap_sum"] for x in subset],
                           c=[x["redundancy_sum"] for x in subset],
                           cmap="viridis", s=70, edgecolors="black", linewidths=0.3)
        for role, selected in representatives:
            if selected["groups"] == count:
                ax.annotate(role, (selected["box_spread"], selected["gap_sum"]),
                            xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set(xlabel="组间箱数差", ylabel="库存缺口总数", title=f"{count}组：{len(subset)}种完整分区")
        ax.grid(alpha=0.2)
        fig.colorbar(point, ax=ax, label="相对共用资源的新增配置")
    for ext in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q4_库存缺口与均衡权衡.{ext}", dpi=200)
    plt.close(fig)

    # 每个代表方案逐类型展示配置与现有库存，不把异型资源混作一种。
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    for ax, (role, selected) in zip(axes.flat, representatives):
        x = list(range(len(RESOURCE_KEYS)))
        ax.bar([i - 0.2 for i in x], [stock[key] for key in RESOURCE_KEYS],
               width=0.4, label="现有库存", color="#5470a8")
        ax.bar([i + 0.2 for i in x], [selected["total"][key] for key in RESOURCE_KEYS],
               width=0.4, label="分组最少配置", color="#d8894c")
        ax.set_xticks(x, RESOURCE_KEYS)
        ax.set_title(f"{selected['groups']}组 {role}（缺口{selected['gap_sum']}）")
        ax.set_ylabel("架 / 组")
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q4_代表方案资源配置.{ext}", dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q3-table", type=Path,
                        help="问题三方案一中含 Q3_READY.txt 的表格目录；默认最新 PASS 目录")
    parser.add_argument("--output-root", type=Path,
                        default=CODE_DIR / "4_outputs" / "1_optimize",
                        help="第四问方案一输出根目录")
    args = parser.parse_args(argv)
    source = load_source(args.q3_table or latest_q3_table())
    stock, ids = load_inventory(source["data_run"])
    tasks = build_tasks(source, ids)
    parts = components(tasks)
    print(f"Q4: 读取问题三方案一 PASS：{source['table_dir']}", flush=True)
    print(f"Q4: 15个服务区形成 {len(parts)} 个不可拆单元", flush=True)
    if len(parts) < 3:
        raise ValueError("保持当前问题三联合调度不变时，原子单元不足 3 个；问题四三组不可行")

    pooled = evaluate(tuple(0 for _ in parts), parts, tasks, stock)["total"]
    schemes: list[dict[str, Any]] = []
    for group_count in (2, 3):
        labels_list = partitions(len(parts), group_count)
        print(f"Q4: 枚举 {group_count} 组的 {len(labels_list)} 种无标签分区", flush=True)
        for labels in labels_list:
            result = evaluate(labels, parts, tasks, stock)
            result["redundancy"] = {key: result["total"][key] - pooled[key]
                                    for key in RESOURCE_KEYS}
            check(all(value >= 0 for value in result["redundancy"].values()),
                  "分区资源需求小于合并执行需求，区间核算错误")
            result["redundancy_sum"] = sum(result["redundancy"].values())
            result["id"] = f"G{group_count}-{len([r for r in schemes if r['groups'] == group_count]) + 1:03d}"
            schemes.append(result)
    frontier_flags(schemes)
    representatives: list[tuple[str, dict[str, Any]]] = []
    for count in (2, 3):
        subset = [row for row in schemes if row["groups"] == count]
        minimum_gap = min(subset, key=lambda x: (x["gap_sum"], x["box_spread"],
                                                  x["redundancy_sum"], x["id"]))
        balanced = min(subset, key=lambda x: (x["box_spread"], x["gap_sum"],
                                               x["redundancy_sum"], x["id"]))
        representatives.extend((("缺口优先", minimum_gap), ("箱数均衡优先", balanced)))
        print(f"Q4: {count}组 缺口优先={minimum_gap['gap_sum']}件、箱数差={minimum_gap['box_spread']}；"
              f"均衡优先={balanced['gap_sum']}件、箱数差={balanced['box_spread']}", flush=True)

    stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    table_dir = root / f"q4_opt1_{stamp}_table"
    figure_dir = root / f"q4_opt1_{stamp}_figure"
    table_dir.mkdir(parents=True, exist_ok=False)
    box_total = len(tasks["boxes"])
    scheme_rows = []
    for row in schemes:
        scheme_rows.append({
            "方案编号": row["id"], "组数": row["groups"],
            "单元分组": "; ".join(f"{part['id']}→{row['labels'][i]+1}"
                                  for i, part in enumerate(parts)),
            "各组箱数": "/".join(str(g["箱数"]) for g in row["group_rows"]),
            "各组运输架次": "/".join(str(g["运输架次"]) for g in row["group_rows"]),
            "各组中继架次": "/".join(str(g["中继架次"]) for g in row["group_rows"]),
            "箱数差": row["box_spread"], "运输架次差": row["transport_spread"],
            "中继架次差": row["relay_spread"],
            "库存缺口总数": row["gap_sum"], "各类库存缺口": compact_resources(row["gap"]),
            "库存未用余量": compact_resources(row["spare"]),
            "分组配置总数": sum(row["total"].values()),
            "各类最少配置": compact_resources(row["total"]),
            "相对共用新增配置": row["redundancy_sum"],
            "原编号保留缺口总数": row["fixed_gap_sum"],
            "原编号保留缺口明细": compact_resources(row["fixed_gap"]),
            "Pareto非支配": "是" if row["pareto"] else "否",
        })
    write_csv(table_dir / "Q4_全部分区.csv", scheme_rows, list(scheme_rows[0]))
    representative_rows = []
    group_rows = []
    zone_rows = []
    task_rows = []
    delivery_rows = []
    resource_rows = []
    for role, selected in representatives:
        representative_rows.append({"取向": role, **next(x for x in scheme_rows
                                                    if x["方案编号"] == selected["id"]),
                                    "各类最少配置": compact_resources(selected["total"]),
                                    "各类原编号配置": compact_resources(selected["fixed_total"])})
        for group in selected["group_rows"]:
            group_rows.append({
                "方案编号": selected["id"], "取向": role, "组号": group["组号"],
                "服务区": ";".join(group["服务区"]), "箱数": group["箱数"],
                "运输架次": group["运输架次"], "中继架次": group["中继架次"],
                "运输能耗_kWh": group["运输能耗_kWh"],
                "中继能耗_kWh": group["中继能耗_kWh"],
                "运输机占用_h": group["运输机占用_h"],
                "电池占用_h": group["电池占用_h"],
                "中继机占用_h": group["中继机占用_h"],
                "能源组件占用_h": group["能源组件占用_h"],
                **{RESOURCE_NAMES[key]: group["资源最少配置"][key]
                   for key in RESOURCE_KEYS},
            })
        for zone, group_id in sorted(selected["zone_group"].items()):
            zone_rows.append({"方案编号": selected["id"], "取向": role,
                              "服务区": zone, "组号": group_id,
                              "货箱数": sum(x["zone_id"] == zone
                                           for x in tasks["boxes"].values())})
        for group in selected["group_rows"]:
            for task_id in group["任务编号"]:
                task = tasks["intervals"][task_id]
                task_rows.append({"方案编号": selected["id"], "取向": role,
                                  "组号": group["组号"], "任务编号": task_id,
                                  "类别": task["kind"],
                                  "服务区": ";".join(sorted(task["zones"])),
                                  "访问顺序": task["visit_order"],
                                  "货箱编号": ";".join(task["box_ids"]),
                                  "开始时刻_s": task["start_s"],
                                  "返航时刻_s": task["return_s"],
                                  "能耗_kWh": task["energy_kwh"]})
        for item in source["deliveries"]:
            delivery_rows.append({"方案编号": selected["id"], "取向": role,
                                  "组号": selected["zone_group"][item["服务区编号"]],
                                  **item})
        for item in selected["allocation_rows"]:
            resource_rows.append({"方案编号": selected["id"], "取向": role, **item})
    write_csv(table_dir / "Q4_代表方案.csv", representative_rows,
              list(representative_rows[0]))
    write_csv(table_dir / "Q4_各组资源配置.csv", group_rows, list(group_rows[0]))
    write_csv(table_dir / "Q4_服务区分组.csv", zone_rows, list(zone_rows[0]))
    write_csv(table_dir / "Q4_固定任务归属.csv", task_rows, list(task_rows[0]))
    write_csv(table_dir / "Q4_逐箱交付分组.csv", delivery_rows, list(delivery_rows[0]))
    write_csv(table_dir / "Q4_组内资源编号.csv", resource_rows, list(resource_rows[0]))
    write_csv(table_dir / "Q4_库存与共用峰值.csv", [
        {"资源代码": key, "资源类别": RESOURCE_NAMES[key], "现有库存": stock[key],
         "不分组共用峰值": pooled[key]} for key in RESOURCE_KEYS],
        ["资源代码", "资源类别", "现有库存", "不分组共用峰值"])
    write_csv(table_dir / "Q4_原子单元.csv", [
        {"单元编号": part["id"], "服务区": ";".join(part["zones"]),
         "服务区数": len(part["zones"]), "货箱数": part["boxes"]}
        for part in parts], ["单元编号", "服务区", "服务区数", "货箱数"])
    checks = [
        {"检查项": "问题三源结果", "状态": "PASS", "说明": "Q3_READY、摘要和验收表均为PASS"},
        {"检查项": "服务区互斥完整", "状态": "PASS", "说明": "15区每区恰归一组；每组非空"},
        {"检查项": "货箱与任务完整", "状态": "PASS",
         "说明": f"{box_total}箱、{len(tasks['transport'])}运输架次、{len(tasks['relay'])}中继架次原样归组"},
        {"检查项": "共享任务不可拆", "状态": "PASS", "说明": "运输和中继共享关系先合并为连通分量"},
        {"检查项": "组内资源时序", "状态": "PASS", "说明": "区间着色编号已逐资源复核无重叠"},
    ]
    write_csv(table_dir / "Q4_验收检查.csv", checks, ["检查项", "状态", "说明"])
    write_json(table_dir / "Q4_运行摘要.json", {
        "status": "PASS", "interpretation": "固定Q3方案1全部任务与绝对时刻；组内同型资源可重新编号；精确枚举分区",
        "global_optimality": "仅对固定问题三任务时刻与现有原子单元的2组/3组分区完整枚举，不是问题三与问题四联合全局最优",
        "source_q3_table": str(source["table_dir"]),
        "source_q3_manifest_sha256": source["summary"]["source"]["source_manifest_sha256"],
        "source_q3_file_sha256": source["hashes"],
        "source_data_run": str(source["data_run"]),
        "stock": stock, "pooled_peak_need": pooled,
        "components": parts, "partition_count": {str(k): sum(x["groups"] == k for x in schemes)
                                                   for k in (2, 3)},
        "representatives": [
            {"groups": selected["groups"], "role": role, "scheme_id": selected["id"],
             "group_boxes": [group["箱数"] for group in selected["group_rows"]],
             "minimum_resource_need": selected["total"],
             "inventory_gap": selected["gap"],
             "fixed_original_id_gap": selected["fixed_gap"]}
            for role, selected in representatives],
        "table_dir": str(table_dir), "figure_dir": str(figure_dir),
    })
    create_figures(figure_dir, schemes, representatives, stock)
    (table_dir / "Q4_READY.txt").write_text(
        "Q4 scheme 1 PASS: frozen Q3 scheme 1; all 2/3-group partitions and resource timelines checked\n",
        encoding="utf-8")
    print(f"Q4 PASS: 表格 {table_dir}", flush=True)
    print(f"Q4 PASS: 图片 {figure_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
