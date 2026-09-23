"""D 题问题二基线：复用 Q1 同区 FFD 批次，排 8 架实体机与共享电池。

单次运行先写 Q2 官方格式表和验收表，再画实体机/电池及交付时限图。
这是截止优先的事件排程基线，不是跨区联合优化；超时会明确标记为失败。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook

from _0_pipeline import sha256, verify_ready
from q1_0_baseline import CODE_DIR, PROJECT, _read_inputs, find_latest_validated_run, run_model, sortie_metrics


TOL = 1e-7
SORTIE_COLUMNS = [
    "架次编号", "无人机编号", "机型编号", "电池编号", "开始时刻（s）",
    "访问服务区顺序", "返回O01时刻（s）", "架次能耗（kWh）",
]
DELIVERY_COLUMNS = ["货箱编号", "架次编号", "服务区编号", "交付完成时刻（s）"]


@dataclass
class UAV:
    uav_id: str
    type_id: str
    available_s: float = 0.0


@dataclass
class Battery:
    battery_id: str
    type_id: str
    full_charge_s: float
    available_s: float = 0.0


def optional_seconds(value: Any) -> float | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"非法时刻：{value}")
    return seconds


def required_seconds(value: Any, label: str) -> float:
    seconds = optional_seconds(value)
    if seconds is None:
        raise ValueError(f"缺少{label}")
    return seconds


def source_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"true", "1", "是"}:
        return True
    if str(value).strip().lower() in {"false", "0", "否"}:
        return False
    raise ValueError(f"非法布尔值：{value}")


def charge_to_full_s(return_soc: float, full_charge_s: float) -> float:
    """官方两阶段模型：0–90% 用 65% 总时长，90–100% 用 35%。"""
    if not 0 <= return_soc <= 1 + TOL or full_charge_s <= 0:
        raise ValueError("电池 SOC 或等效充满时间超范围")
    soc = min(return_soc, 1.0)
    if soc < 0.9:
        return full_charge_s * (0.65 * (0.9 - soc) / 0.9 + 0.35)
    return full_charge_s * 0.35 * (1.0 - soc) / 0.1


def read_resources(data_run: Path) -> tuple[list[UAV], list[Battery]]:
    clean = data_run / "clean"
    uav_rows = pd.read_csv(clean / "transport_uavs.csv", dtype=str).to_dict("records")
    battery_rows = pd.read_csv(clean / "battery_units.csv", dtype=str).to_dict("records")
    pool_rows = pd.read_csv(clean / "battery_pool.csv", dtype=str).to_dict("records")
    if len(uav_rows) != 8 or len(battery_rows) != 14:
        raise ValueError("问题二要求 8 架实体运输机和总计 14 组电池")
    uavs = [UAV(str(row["uav_id"]), str(row["type_id"])) for row in uav_rows]
    if len({u.uav_id for u in uavs}) != 8 or any(row["initial_site"] != "O01" for row in uav_rows):
        raise ValueError("运输机编号不唯一或初始位置不是 O01")
    pool: dict[str, tuple[int, float]] = {}
    for row in pool_rows:
        type_id = str(row["type_id"])
        if type_id in pool:
            raise ValueError(f"重复的电池机型：{type_id}")
        pool[type_id] = (int(row["stock_qty"]), float(row["full_charge_s"]))
    if {kind: qty for kind, (qty, _) in pool.items()} != {"A": 6, "B": 4, "C": 4}:
        raise ValueError("A/B/C 电池库存与附件 6/4/4 不一致")
    batteries = []
    for row in battery_rows:
        type_id = str(row["type_id"])
        if type_id not in pool or not math.isclose(float(row["initial_soc"]), 1.0, abs_tol=TOL):
            raise ValueError("电池机型或初始 100% SOC 不符合附件")
        batteries.append(Battery(str(row["battery_id"]), type_id, pool[type_id][1]))
    if len({b.battery_id for b in batteries}) != 14:
        raise ValueError("电池编号不唯一")
    if Counter(u.type_id for u in uavs) != Counter({"A": 4, "B": 2, "C": 2}):
        raise ValueError("实体机 A/B/C 数量与附件 4/2/2 不一致")
    if Counter(b.type_id for b in batteries) != Counter({"A": 6, "B": 4, "C": 4}):
        raise ValueError("电池明细与库存不一致，不能另加初装电池")
    return sorted(uavs, key=lambda u: u.uav_id), sorted(batteries, key=lambda b: b.battery_id)


def ordered_boxes(batch: dict[str, Any], boxes_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    boxes = [boxes_by_id[box_id] for box_id in batch["box_ids"]]
    # 同一站点先交付硬时限紧的箱，接着按期望时刻与优先系数排序。
    return sorted(boxes, key=lambda b: (
        optional_seconds(b["hard_deadline_s"]) if optional_seconds(b["hard_deadline_s"]) is not None else math.inf,
        required_seconds(b["expected_s"], f"{b['box_id']} 的期望时刻"),
        -float(b["priority"]), str(b["box_id"]),
    ))


def make_tasks(data_run: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    q1 = run_model(data_run)
    if any(check["status"] != "PASS" for check in q1["checks"]):
        raise ValueError("问题一 FFD 输入未通过验收")
    boxes, models, arcs = _read_inputs(data_run)
    boxes_by_id = {str(box["box_id"]): box for box in boxes}
    if len(boxes_by_id) != 80:
        raise ValueError("货箱编号不唯一")
    tasks = []
    seen: Counter[str] = Counter()
    for batch in q1["ffd"]:
        batch_id, zone, type_id = batch["batch_id"], batch["zone_id"], batch["type_id"]
        if any(box_id not in boxes_by_id for box_id in batch["box_ids"]):
            raise ValueError(f"{batch_id} 引用了未知货箱")
        delivery_order = ordered_boxes(batch, boxes_by_id)
        if any(str(box["zone_id"]) != zone for box in delivery_order):
            raise ValueError(f"{batch_id} 混入其他服务区货箱")
        seen.update(batch["box_ids"])
        physical = sortie_metrics(models[type_id], arcs[("O01", zone)], arcs[(zone, "O01")], delivery_order)
        if not physical["feasible"]:
            raise ValueError(f"{batch_id} 不满足载重、体积或返航余量：{physical['failure_reasons']}")
        for key in ("mass_kg", "volume_m3", "energy_kwh", "operation_s"):
            if not math.isclose(float(physical[key]), float(batch[key]), rel_tol=0, abs_tol=TOL):
                raise ValueError(f"{batch_id} 与问题一批次的 {key} 不一致")
        for box in delivery_order:
            if source_bool(box["medical_bool"]) or source_bool(box["is_first_batch"]):
                if optional_seconds(box["hard_deadline_s"]) is None:
                    raise ValueError(f"{box['box_id']} 缺少医疗或首批硬时限")
        deadlines = [optional_seconds(box["hard_deadline_s"]) for box in delivery_order]
        tasks.append({
            "batch_id": batch_id, "zone_id": zone, "type_id": type_id,
            "boxes": delivery_order, "physical": physical,
            "handoff_base_s": float(models[type_id]["handoff_base_s"]),
            "handoff_per_box_s": float(models[type_id]["handoff_per_box_s"]),
            "return_soc_min": float(models[type_id]["return_soc_min"]),
            "nearest_hard_s": min((d for d in deadlines if d is not None), default=math.inf),
            "nearest_expected_s": min(required_seconds(b["expected_s"], "期望时刻") for b in delivery_order),
        })
    if seen != Counter(boxes_by_id.keys()):
        raise ValueError("问题一 FFD 未将 80 个货箱恰好分配一次")
    return sorted(tasks, key=lambda t: (t["nearest_hard_s"], t["nearest_expected_s"], t["batch_id"])), boxes_by_id, q1


def schedule_baseline(tasks: list[dict[str, Any]], uavs: list[UAV],
                      batteries: list[Battery]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sorties: list[dict[str, Any]] = []
    deliveries: list[dict[str, Any]] = []
    for task in tasks:
        kind = task["type_id"]
        physical = task["physical"]
        choices = [
            (max(uav.available_s, battery.available_s) + physical["operation_s"],
             max(uav.available_s, battery.available_s), uav.uav_id, battery.battery_id, uav, battery)
            for uav in uavs if uav.type_id == kind
            for battery in batteries if battery.type_id == kind
        ]
        if not choices:
            raise ValueError(f"{kind} 型缺少可匹配实体机或电池")
        _, start_s, _, _, uav, battery = min(choices, key=lambda x: x[:4])
        prep_end_s = start_s + physical["prep_s"]
        launch_s = prep_end_s + physical["loading_s"]
        arrival_s = launch_s + physical["flight_out_s"]
        handoff_end_s = arrival_s + physical["handoff_s"]
        return_s = handoff_end_s + physical["flight_back_s"]
        soc = float(physical["return_soc"])
        charge_s = charge_to_full_s(soc, battery.full_charge_s)
        charge_end_s = return_s + charge_s
        if not math.isclose(return_s, start_s + physical["operation_s"], rel_tol=0, abs_tol=TOL):
            raise AssertionError("排程的返回时刻与问题一作业时间不一致")
        sorties.append({
            "batch_id": task["batch_id"], "zone_id": task["zone_id"], "type_id": kind,
            "uav_id": uav.uav_id, "battery_id": battery.battery_id,
            "box_ids": [str(box["box_id"]) for box in task["boxes"]],
            "start_s": start_s, "prep_end_s": prep_end_s, "launch_s": launch_s,
            "arrival_s": arrival_s, "handoff_end_s": handoff_end_s, "return_s": return_s,
            "charge_start_s": return_s, "charge_end_s": charge_end_s, "charge_s": charge_s,
            "mass_kg": physical["mass_kg"], "volume_m3": physical["volume_m3"],
            "energy_kwh": physical["energy_kwh"], "return_soc": soc,
        })
        for index, box in enumerate(task["boxes"], 1):
            # 基础交接只收取一次；逐箱交接时间由对应机型参数给定。
            complete_s = arrival_s + task["handoff_base_s"] + index * task["handoff_per_box_s"]
            hard_s = optional_seconds(box["hard_deadline_s"])
            expected_s = required_seconds(box["expected_s"], f"{box['box_id']} 的期望时刻")
            deliveries.append({
                "box_id": str(box["box_id"]), "batch_id": task["batch_id"],
                "zone_id": task["zone_id"], "sequence": index, "complete_s": complete_s,
                "hard_deadline_s": hard_s, "expected_s": expected_s,
                "priority": float(box["priority"]),
                "hard_slack_s": None if hard_s is None else hard_s - complete_s,
                "soft_lateness_s": max(0.0, complete_s - expected_s),
            })
        uav.available_s = return_s
        battery.available_s = charge_end_s
    return sorted(sorties, key=lambda s: (s["start_s"], s["batch_id"])), sorted(deliveries, key=lambda d: d["box_id"])


def check_plan(sorties: list[dict[str, Any]], deliveries: list[dict[str, Any]],
               tasks: list[dict[str, Any]], uavs: list[UAV], batteries: list[Battery],
               boxes_by_id: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    def add(check_id: str, good: bool, detail: str = "") -> None:
        checks.append({"check_id": check_id, "status": "PASS" if good else "FAIL", "detail": detail})

    add("all_80_boxes_once", Counter(d["box_id"] for d in deliveries) == Counter(boxes_by_id.keys()),
        f"deliveries={len(deliveries)}, unique={len({d['box_id'] for d in deliveries})}")
    add("all_q1_batches_once", Counter(s["batch_id"] for s in sorties) ==
        Counter(t["batch_id"] for t in tasks), f"sorties={len(sorties)}")
    uav_types = {u.uav_id: u.type_id for u in uavs}
    battery_types = {b.battery_id: b.type_id for b in batteries}
    battery_full_s = {b.battery_id: b.full_charge_s for b in batteries}
    tasks_by_id = {t["batch_id"]: t for t in tasks}
    sorties_by_id = {s["batch_id"]: s for s in sorties}
    for sortie in sorties:
        batch_id = sortie["batch_id"]
        add(f"{batch_id}:same_type", uav_types[sortie["uav_id"]] ==
            battery_types[sortie["battery_id"]] == sortie["type_id"])
        add(f"{batch_id}:return_soc",
            sortie["return_soc"] >= tasks_by_id[batch_id]["return_soc_min"] - TOL,
            f"SOC={sortie['return_soc']:.9f}, minimum={tasks_by_id[batch_id]['return_soc_min']:.9f}")
        add(f"{batch_id}:time_order", 0 <= sortie["start_s"] <= sortie["launch_s"] <=
            sortie["arrival_s"] <= sortie["handoff_end_s"] <= sortie["return_s"] <=
            sortie["charge_end_s"])
        add(f"{batch_id}:one_zone", all(boxes_by_id[box_id]["zone_id"] == sortie["zone_id"]
                                       for box_id in sortie["box_ids"]))
        expected = tasks_by_id[batch_id]["physical"]
        add(f"{batch_id}:physical_replay", math.isclose(sortie["energy_kwh"],
            expected["energy_kwh"], rel_tol=0, abs_tol=TOL) and
            math.isclose(sortie["return_s"] - sortie["start_s"],
                         expected["operation_s"], rel_tol=0, abs_tol=TOL))
        expected_charge_s = charge_to_full_s(
            sortie["return_soc"], battery_full_s[sortie["battery_id"]])
        add(f"{batch_id}:charging_replay", math.isclose(
            sortie["charge_end_s"] - sortie["return_s"], expected_charge_s,
            rel_tol=0, abs_tol=TOL))
    for delivery in deliveries:
        task = tasks_by_id[delivery["batch_id"]]
        sortie = sorties_by_id[delivery["batch_id"]]
        expected_delivery_s = (sortie["arrival_s"] + task["handoff_base_s"] +
                               delivery["sequence"] * task["handoff_per_box_s"])
        add(f"{delivery['box_id']}:delivery_replay", math.isclose(
            delivery["complete_s"], expected_delivery_s, rel_tol=0, abs_tol=TOL) and
            delivery["complete_s"] <= sortie["handoff_end_s"] + TOL)
        hard = delivery["hard_deadline_s"]
        if hard is not None:
            add(f"{delivery['box_id']}:hard_deadline", delivery["complete_s"] <= hard + TOL,
                f"delivered={delivery['complete_s']:.3f}s, deadline={hard:.3f}s, "
                f"slack={delivery['hard_slack_s']:.3f}s")
    for uav in uavs:
        uses = sorted((s for s in sorties if s["uav_id"] == uav.uav_id), key=lambda s: s["start_s"])
        add(f"{uav.uav_id}:no_overlap", all(a["return_s"] <= b["start_s"] + TOL
                                            for a, b in zip(uses, uses[1:])))
    for battery in batteries:
        uses = sorted((s for s in sorties if s["battery_id"] == battery.battery_id),
                      key=lambda s: s["start_s"])
        add(f"{battery.battery_id}:charged_before_reuse",
            all(a["charge_end_s"] <= b["start_s"] + TOL for a, b in zip(uses, uses[1:])))
    return checks


def validate_template() -> None:
    template = load_workbook(PROJECT / "0_3_数据" / "结果提交模板.xlsx", read_only=True, data_only=True)
    try:
        for sheet_name, expected in (("Q2_运输架次", SORTIE_COLUMNS),
                                     ("Q2_逐箱交付", DELIVERY_COLUMNS)):
            actual = [template[sheet_name].cell(1, col).value for col in range(1, len(expected) + 1)]
            if actual != expected:
                raise ValueError(f"官方模板 {sheet_name} 列名变化，停止导出：{actual}")
    finally:
        template.close()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def save_tables(table_dir: Path, data_run: Path, sorties: list[dict[str, Any]],
                deliveries: list[dict[str, Any]], checks: list[dict[str, str]],
                q1: dict[str, Any]) -> dict[str, Any]:
    if table_dir.exists():
        raise FileExistsError(f"不覆盖已有结果：{table_dir}")
    validate_template()
    table_dir.mkdir(parents=True)
    sortie_rows = [{
        "架次编号": s["batch_id"], "无人机编号": s["uav_id"],
        "机型编号": s["type_id"], "电池编号": s["battery_id"],
        "开始时刻（s）": s["start_s"], "访问服务区顺序": s["zone_id"],
        "返回O01时刻（s）": s["return_s"], "架次能耗（kWh）": s["energy_kwh"],
    } for s in sorties]
    delivery_rows = [{
        "货箱编号": d["box_id"], "架次编号": d["batch_id"],
        "服务区编号": d["zone_id"], "交付完成时刻（s）": d["complete_s"],
    } for d in deliveries]
    write_csv(table_dir / "Q2_运输架次.csv", sortie_rows, SORTIE_COLUMNS)
    write_csv(table_dir / "Q2_逐箱交付.csv", delivery_rows, DELIVERY_COLUMNS)
    detailed_sorties = [{**s, "box_ids": ";".join(s["box_ids"])} for s in sorties]
    write_csv(table_dir / "Q2_架次与电池周转.csv", detailed_sorties, list(detailed_sorties[0]))
    write_csv(table_dir / "Q2_逐箱时限检查.csv", deliveries, list(deliveries[0]))
    write_csv(table_dir / "Q2_验收检查.csv", checks, ["check_id", "status", "detail"])
    failed = [check for check in checks if check["status"] == "FAIL"]
    hard_failed = [check for check in failed if check["check_id"].endswith(":hard_deadline")]
    hard_conflicts = sorted((d for d in deliveries if d["hard_slack_s"] is not None and
                             d["hard_slack_s"] < -TOL),
                            key=lambda d: (d["complete_s"], d["box_id"]))
    summary = {
        "status": "PASS" if not failed else "FAIL",
        "method": "Q1 same-zone FFD batches, hard-deadline-first list scheduling",
        "baseline_not_optimized": True,
        "source_data_run": str(data_run),
        "source_manifest_sha256": sha256(data_run / "meta" / "validated_artifacts.json"),
        "q1_batch_count": len(q1["ffd"]),
        "sorties": len(sorties), "delivered_boxes": len(deliveries),
        "uavs": 8, "batteries": 14,
        "makespan_s": max(s["return_s"] for s in sorties),
        "total_energy_kwh": sum(s["energy_kwh"] for s in sorties),
        "hard_deadline_boxes": sum(d["hard_deadline_s"] is not None for d in deliveries),
        "hard_deadline_violations": len(hard_failed),
        "first_failure": failed[0] if failed else None,
        "first_hard_conflict": ({"box_id": hard_conflicts[0]["box_id"],
                                  "delivered_s": hard_conflicts[0]["complete_s"],
                                  "deadline_s": hard_conflicts[0]["hard_deadline_s"]}
                                 if hard_conflicts else None),
        "soft_weighted_lateness_s": sum(
            d["priority"] * d["soft_lateness_s"] for d in deliveries
            if d["hard_deadline_s"] is None
        ),
        "battery_rule": "same-type shared pool; starts full; each used battery recharges to 100% before reuse",
        "charging_model": "two-stage 0-90% takes 65% of T_full; 90-100% takes 35%; parallel charging allowed",
        "note": "A failed hard deadline means this fixed Q1 FFD scheduling baseline failed; it does not prove Q2 infeasible.",
    }
    (table_dir / "Q2_运行摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (table_dir / "Q2_运行说明.txt").write_text(
        "问题二单区批次排程基线：复用 Q1 FFD 的批次和物理计算器，不做跨区路线优化。\n"
        "按最紧硬时限、期望时刻、批次号排序；同机型中选最早返航的实体机和满电池。\n"
        "Q2_运输架次.csv 与 Q2_逐箱交付.csv 的列名与官方模板一致。\n"
        "逐箱交付时刻包含准备、装载、去程飞行、基础交接及本箱之前全部逐箱交接时间。\n"
        "电池返航后按官方两阶段规则充至 100% 才可复用；库存 6/4/4 已含初装。\n"
        "Q2_验收检查.csv 中如有 FAIL，本次基线不可行，但不能据此断定问题二无解。\n",
        encoding="utf-8")
    for filename, expected_columns, expected_count in (
        ("Q2_运输架次.csv", SORTIE_COLUMNS, len(sorties)),
        ("Q2_逐箱交付.csv", DELIVERY_COLUMNS, len(deliveries)),
    ):
        saved = pd.read_csv(table_dir / filename, encoding="utf-8-sig")
        if list(saved.columns) != expected_columns or len(saved) != expected_count:
            raise ValueError(f"官方格式表导出复核失败：{filename}")
    return summary


def save_figures(figure_dir: Path, sorties: list[dict[str, Any]],
                 deliveries: list[dict[str, Any]], uavs: list[UAV],
                 batteries: list[Battery], summary: dict[str, Any]) -> None:
    if figure_dir.exists():
        raise FileExistsError(f"不覆盖已有结果：{figure_dir}")
    os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIR / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure_dir.mkdir(parents=True)
    uav_ids = sorted(u.uav_id for u in uavs)
    battery_ids = sorted(b.battery_id for b in batteries)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [len(uav_ids), len(battery_ids)]})
    kind_colors = {"A": "#3A80C1", "B": "#23A67A", "C": "#DA884A"}
    for s in sorties:
        color = kind_colors[s["type_id"]]
        axes[0].barh(uav_ids.index(s["uav_id"]), (s["return_s"] - s["start_s"]) / 3600,
                     left=s["start_s"] / 3600, height=0.65, color=color, edgecolor="white")
        y = battery_ids.index(s["battery_id"])
        axes[1].barh(y, (s["return_s"] - s["start_s"]) / 3600,
                     left=s["start_s"] / 3600, height=0.65, color=color, edgecolor="white")
        axes[1].barh(y, s["charge_s"] / 3600, left=s["return_s"] / 3600,
                     height=0.65, color="#AEB5BD", edgecolor="white")
    for ax, labels, title in ((axes[0], uav_ids, "Transport UAVs"),
                              (axes[1], battery_ids, "Shared batteries: use and recharge")):
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    axes[1].set_xlabel("Time since start (h)")
    fig.legend(handles=[Patch(facecolor=color, label=f"Type {kind}")
                        for kind, color in kind_colors.items()] +
               [Patch(facecolor="#AEB5BD", label="Charging")],
               loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.10, hspace=0.32, left=0.14, right=0.98, top=0.94)
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q2_机电池时间线.{extension}", dpi=220)
    plt.close(fig)

    hard = sorted((d for d in deliveries if d["hard_deadline_s"] is not None),
                  key=lambda d: (d["hard_slack_s"], d["box_id"]))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(range(len(hard)), [d["hard_slack_s"] / 60 for d in hard],
           color=["#C84D4D" if d["hard_slack_s"] < -TOL else "#3A80C1" for d in hard])
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel("Hard-deadline boxes ordered by slack")
    ax.set_ylabel("Deadline slack (min)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q2_硬时限余量.{extension}", dpi=220)
    plt.close(fig)
    (figure_dir / "图表说明.txt").write_text(
        "Q2_机电池时间线：上图为 8 架实体机占用，下图为共享电池占用与充电。\n"
        "Q2_硬时限余量：每个有硬截止的箱子以截止时刻减实际交付时刻作图；红色为超时。\n"
        f"本次基线状态：{summary['status']}；如为 FAIL，图仅用于定位冲突，不能作为可行方案。\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="D 题问题二：同区批次、实体机和共享电池排程基线")
    parser.add_argument("--data-run", type=Path, help="已验收的数据运行目录；默认最新一次")
    parser.add_argument("--output-root", type=Path, help="结果根目录；默认 code/2_outputs")
    args = parser.parse_args()
    data_run = args.data_run.resolve() if args.data_run else find_latest_validated_run()
    if not verify_ready(data_run):
        raise ValueError(f"数据目录未通过验收：{data_run}")
    tasks, boxes_by_id, q1 = make_tasks(data_run)
    uavs, batteries = read_resources(data_run)
    sorties, deliveries = schedule_baseline(tasks, uavs, batteries)
    checks = check_plan(sorties, deliveries, tasks, uavs, batteries, boxes_by_id)
    if not verify_ready(data_run):
        raise ValueError("排程期间输入数据发生改变")
    output_root = args.output_root.resolve() if args.output_root else CODE_DIR / "2_outputs"
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    table_dir = output_root / f"q2_base_{stamp}_table"
    figure_dir = output_root / f"q2_base_{stamp}_figure"
    if table_dir.exists() or figure_dir.exists():
        raise FileExistsError("本次结果目录已存在；请稍后重试或指定新的 --output-root")
    summary = save_tables(table_dir, data_run, sorties, deliveries, checks, q1)
    save_figures(figure_dir, sorties, deliveries, uavs, batteries, summary)
    (table_dir / "Q2_基线状态.txt").write_text(
        f"{summary['status']}\nfirst_failure={summary['first_failure']}\n"
        f"first_hard_conflict={summary['first_hard_conflict']}\n", encoding="utf-8")
    print(f"Q2 baseline {summary['status']}: {summary['sorties']} sorties, "
          f"{summary['delivered_boxes']} boxes, {summary['hard_deadline_violations']} hard-deadline violations")
    print(f"makespan={summary['makespan_s']:.3f} s, total_energy={summary['total_energy_kwh']:.6f} kWh")
    if summary["first_failure"]:
        print(f"first failure: {summary['first_failure']}")
    if summary["first_hard_conflict"]:
        print(f"first hard conflict: {summary['first_hard_conflict']}")
    print(f"tables: {table_dir}")
    print(f"figures: {figure_dir}")


if __name__ == "__main__":
    main()
