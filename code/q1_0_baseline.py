"""D 题第一问可行基线：安全载荷、单箱保底、同区 FFD 与回代验收。

注意：水平与爬升能耗的具体分项公式来自组内执行思路，是建模假设，
而非官方题面直接给出的公式。这里不求解集合划分优化，也不排实体机/电池。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook

from _0_pipeline import sha256, verify_ready


CODE_DIR = Path(__file__).resolve().parent
PROJECT = CODE_DIR.parent
GRAVITY = 9.81
KWH_JOULES = 3_600_000.0
TOL = 1e-9
TEMPLATE_COLUMNS = [
    "架次编号", "服务区编号", "机型编号", "货箱编号列表", "总质量（kg）",
    "总体积（m³）", "往返时间（s）", "架次能耗（kWh）", "返航SOC（%）",
]


def equivalent_range_m(model: dict[str, Any], load_kg: float) -> float:
    capacity = float(model["max_payload_kg"])
    if capacity <= 0 or load_kg < -TOL or load_kg > capacity + TOL:
        raise ValueError(f"非法载荷：{load_kg} kg；额定载重 {capacity} kg")
    q = max(0.0, min(float(load_kg), capacity))
    result = float(model["range_empty_m"]) - (
        float(model["range_empty_m"]) - float(model["range_full_m"])
    ) * (q / capacity) ** 1.5
    if result <= 0:
        raise ValueError("等效航程必须为正")
    return result


def leg_energy_kwh(model: dict[str, Any], arc: dict[str, Any], load_kg: float) -> dict[str, float]:
    horizontal = (float(model["energy_use_kwh"]) * float(arc["distance_m"]) /
                  equivalent_range_m(model, load_kg))
    climb = ((float(model["empty_mass_kg"]) + load_kg) * GRAVITY *
             float(arc["climb_m"]) /
             (KWH_JOULES * float(model["climb_efficiency"])))
    return {"horizontal_kwh": horizontal, "climb_kwh": climb,
            "total_kwh": horizontal + climb}


def leg_time_s(model: dict[str, Any], arc: dict[str, Any]) -> float:
    return (float(arc["climb_m"]) / float(model["climb_speed_mps"]) +
            float(arc["distance_m"]) / float(model["cruise_speed_mps"]) +
            float(arc["descent_m"]) / float(model["descent_speed_mps"]))


def roundtrip_energy_kwh(model: dict[str, Any], outward: dict[str, Any],
                         homeward: dict[str, Any], load_kg: float) -> float:
    # 返程货物已交付，载荷始终为 0；使用反向航段自身的爬升高度。
    return (leg_energy_kwh(model, outward, load_kg)["total_kwh"] +
            leg_energy_kwh(model, homeward, 0.0)["total_kwh"])


def max_safe_payload_kg(model: dict[str, Any], outward: dict[str, Any],
                        homeward: dict[str, Any]) -> dict[str, Any]:
    rho = float(model["return_soc_min"])
    if not 0 <= rho < 1:
        raise ValueError(f"返航余量比例超范围：{rho}")
    rated = float(model["max_payload_kg"])
    limit = (1.0 - rho) * float(model["energy_use_kwh"])
    empty_energy = roundtrip_energy_kwh(model, outward, homeward, 0.0)
    result: dict[str, Any] = {
        "energy_limit_kwh": limit, "energy_empty_kwh": empty_energy,
        "rated_payload_kg": rated,
    }
    if empty_energy > limit + TOL:
        return dict(result, status="unreachable", q_max_kg=None,
                    energy_at_qmax_kwh=None, soc_at_qmax_pct=None)
    rated_energy = roundtrip_energy_kwh(model, outward, homeward, rated)
    if rated_energy <= limit + TOL:
        safe_load, status = rated, "rated_payload"
    else:
        low, high = 0.0, rated
        for _ in range(80):
            mid = (low + high) / 2.0
            if roundtrip_energy_kwh(model, outward, homeward, mid) <= limit:
                low = mid
            else:
                high = mid
        # 对外保留 6 位小数并向下截断，避免打印舍入越过能量边界。
        safe_load, status = math.floor(low * 1_000_000) / 1_000_000, "energy_limited"
    energy = roundtrip_energy_kwh(model, outward, homeward, safe_load)
    if energy > limit + TOL:
        raise AssertionError("二分的安全载荷未通过能量回代")
    return dict(result, status=status, q_max_kg=safe_load,
                energy_at_qmax_kwh=energy,
                soc_at_qmax_pct=100.0 * (1.0 - energy / float(model["energy_use_kwh"])))


def sortie_metrics(model: dict[str, Any], outward: dict[str, Any],
                   homeward: dict[str, Any], boxes: list[dict[str, Any]]) -> dict[str, Any]:
    if not boxes:
        raise ValueError("架次不能没有货箱")
    mass = sum(float(box["mass_kg"]) for box in boxes)
    volume = sum(float(box["volume_m3"]) for box in boxes)
    outbound = leg_energy_kwh(model, outward, mass)
    inbound = leg_energy_kwh(model, homeward, 0.0)
    energy = outbound["total_kwh"] + inbound["total_kwh"]
    energy_use = float(model["energy_use_kwh"])
    energy_limit = (1.0 - float(model["return_soc_min"])) * energy_use
    flight_out = leg_time_s(model, outward)
    flight_back = leg_time_s(model, homeward)
    prep = float(model["prep_s"])
    loading = len(boxes) * float(model["load_per_box_s"])
    handoff = (float(model["handoff_base_s"]) +
               len(boxes) * float(model["handoff_per_box_s"]))
    failures = []
    if mass > float(model["max_payload_kg"]) + TOL:
        failures.append("rated_mass")
    if volume > float(model["capacity_m3"]) + TOL:
        failures.append("volume")
    if energy > energy_limit + TOL:
        failures.append("return_energy")
    return {
        "feasible": not failures, "failure_reasons": ";".join(failures),
        "mass_kg": mass, "volume_m3": volume, "box_count": len(boxes),
        "energy_out_kwh": outbound["total_kwh"], "energy_back_kwh": inbound["total_kwh"],
        "energy_kwh": energy, "energy_limit_kwh": energy_limit,
        "return_soc": 1.0 - energy / energy_use,
        "flight_out_s": flight_out, "flight_back_s": flight_back,
        "flight_s": flight_out + flight_back,
        "prep_s": prep, "loading_s": loading, "handoff_s": handoff,
        "operation_s": prep + loading + flight_out + handoff + flight_back,
    }


def best_type_for_batch(boxes: list[dict[str, Any]], models: dict[str, dict[str, Any]],
                        outward: dict[str, Any], homeward: dict[str, Any]) -> dict[str, Any] | None:
    best = None
    for type_id in sorted(models):
        model = models[type_id]
        mass = sum(float(box["mass_kg"]) for box in boxes)
        volume = sum(float(box["volume_m3"]) for box in boxes)
        if (mass > float(model["max_payload_kg"]) + TOL or
                volume > float(model["capacity_m3"]) + TOL):
            continue
        result = sortie_metrics(model, outward, homeward, boxes)
        if not result["feasible"]:
            continue
        candidate = dict(result, type_id=type_id)
        if best is None or candidate["energy_kwh"] < best["energy_kwh"] - 1e-12:
            best = candidate
        # 已按型号排序；能耗平局时保留先遇到的较小型号。
    return best


def _read_inputs(data_run: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]],
                                            dict[tuple[str, str], dict[str, Any]]]:
    if not verify_ready(data_run):
        raise ValueError(f"输入数据未通过哈希验收，不可建模：{data_run}")
    clean = data_run / "clean"
    boxes_frame = pd.read_csv(clean / "boxes.csv", dtype={"box_id": str, "zone_id": str})
    types_frame = pd.read_csv(clean / "transport_types.csv", dtype={"type_id": str})
    arcs_frame = pd.read_csv(data_run / "derived" / "arc_geometry.csv",
                             dtype={"from_id": str, "to_id": str})
    if len(boxes_frame) != 80 or len(types_frame) != 3 or len(arcs_frame) != 240:
        raise ValueError("输入表规模与正式附件不符")
    boxes = boxes_frame.to_dict("records")
    models = {str(row["type_id"]): row for row in types_frame.to_dict("records")}
    arcs = {(str(row["from_id"]), str(row["to_id"])): row
            for row in arcs_frame.to_dict("records")}
    return boxes, models, arcs


def _plan_from_groups(prefix: str, groups: list[dict[str, Any]],
                      models: dict[str, dict[str, Any]],
                      arcs: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    plan = []
    for sequence, group in enumerate(groups, 1):
        zone = group["zone_id"]
        batch_boxes = group["boxes"]
        chosen = best_type_for_batch(batch_boxes, models, arcs[("O01", zone)], arcs[(zone, "O01")])
        if chosen is None:
            raise ValueError(f"{zone} 的货箱组不可由任一机型安全往返：{[b['box_id'] for b in batch_boxes]}")
        plan.append(dict(chosen, batch_id=f"{prefix}{sequence:03d}", zone_id=zone,
                         box_ids=[str(box["box_id"]) for box in batch_boxes]))
    return plan


def _validate_plan(label: str, plan: list[dict[str, Any]],
                   source_boxes: list[dict[str, Any]], models: dict[str, dict[str, Any]],
                   arcs: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    source = {str(box["box_id"]): box for box in source_boxes}
    seen = Counter(box_id for batch in plan for box_id in batch["box_ids"])

    def check(name: str, good: bool, detail: str = "") -> None:
        checks.append({"check_id": f"{label}:{name}",
                       "status": "PASS" if good else "FAIL", "detail": detail})

    check("all_boxes_exactly_once", seen == Counter(source.keys()),
          f"assigned={sum(seen.values())}, unique={len(seen)}, expected={len(source)}")
    for batch in plan:
        batch_id, zone, type_id = batch["batch_id"], batch["zone_id"], batch["type_id"]
        if any(box_id not in source for box_id in batch["box_ids"]):
            check(f"{batch_id}:known_box_ids", False)
            continue
        group = [source[box_id] for box_id in batch["box_ids"]]
        check(f"{batch_id}:one_zone", all(box["zone_id"] == zone for box in group))
        recalculated = sortie_metrics(models[type_id], arcs[("O01", zone)],
                                      arcs[(zone, "O01")], group)
        check(f"{batch_id}:mass_volume_energy", recalculated["feasible"],
              recalculated["failure_reasons"])
        fields = ("mass_kg", "volume_m3", "energy_kwh", "return_soc", "flight_s", "operation_s")
        check(f"{batch_id}:replay", all(math.isclose(batch[field], recalculated[field],
                                                     rel_tol=0, abs_tol=1e-8) for field in fields))
    return checks


def _metrics(label: str, plan: list[dict[str, Any]]) -> dict[str, Any]:
    return {"baseline": label, "sorties": len(plan),
            "delivered_boxes": sum(len(batch["box_ids"]) for batch in plan),
            "total_mass_kg": sum(batch["mass_kg"] for batch in plan),
            "total_volume_m3": sum(batch["volume_m3"] for batch in plan),
            "total_energy_kwh": sum(batch["energy_kwh"] for batch in plan),
            "cumulative_operation_s": sum(batch["operation_s"] for batch in plan),
            "cumulative_flight_s": sum(batch["flight_s"] for batch in plan),
            "min_return_soc_pct": min(batch["return_soc"] for batch in plan) * 100}


def run_model(data_run: Path) -> dict[str, Any]:
    data_run = data_run.resolve()
    boxes, models, arcs = _read_inputs(data_run)
    zones = sorted({str(box["zone_id"]) for box in boxes})
    if len(zones) != 15 or set(models) != {"A", "B", "C"}:
        raise ValueError("第一问要求 15 个服务区与 A/B/C 三种机型")
    for zone in zones:
        for leg in (("O01", zone), (zone, "O01")):
            if leg not in arcs or not bool(arcs[leg]["valid_flag"]):
                raise ValueError(f"第一问必要航段无有效地形数据：{leg}")
    max_payloads = []
    checks: list[dict[str, str]] = []
    for type_id in sorted(models):
        model = models[type_id]
        for zone in zones:
            outward, homeward = arcs[("O01", zone)], arcs[(zone, "O01")]
            result = max_safe_payload_kg(model, outward, homeward)
            max_payloads.append(dict(type_id=type_id, zone_id=zone,
                                     out_distance_m=outward["distance_m"],
                                     back_distance_m=homeward["distance_m"], **result))
            if result["status"] == "unreachable":
                good = (result["q_max_kg"] is None and
                        result["energy_empty_kwh"] > result["energy_limit_kwh"] + TOL)
            else:
                good = result["energy_at_qmax_kwh"] <= result["energy_limit_kwh"] + TOL
            checks.append({"check_id": f"max_payload:{type_id}:{zone}",
                           "status": "PASS" if good else "FAIL",
                           "detail": result["status"]})
    checks.append({"check_id": "max_payloads:45_rows", "status": "PASS" if len(max_payloads) == 45 else "FAIL",
                   "detail": str(len(max_payloads))})

    single_groups = []
    ffd_groups = []
    for zone in zones:
        local = sorted((box for box in boxes if box["zone_id"] == zone), key=lambda b: str(b["box_id"]))
        for box in local:
            if best_type_for_batch([box], models, arcs[("O01", zone)], arcs[(zone, "O01")]) is None:
                raise ValueError(f"单箱无可行机型：{box['box_id']}")
            single_groups.append({"zone_id": zone, "boxes": [box]})
        ordered = sorted(local, key=lambda b: (-float(b["mass_kg"]), -float(b["volume_m3"]), str(b["box_id"])))
        local_groups: list[dict[str, Any]] = []
        for box in ordered:
            placed = False
            for group in local_groups:
                candidate = group["boxes"] + [box]
                if best_type_for_batch(candidate, models, arcs[("O01", zone)], arcs[(zone, "O01")]) is not None:
                    group["boxes"].append(box)
                    placed = True
                    break
            if not placed:
                local_groups.append({"zone_id": zone, "boxes": [box]})
        ffd_groups.extend(local_groups)

    single = _plan_from_groups("S", single_groups, models, arcs)
    ffd = _plan_from_groups("F", ffd_groups, models, arcs)
    checks.extend(_validate_plan("single_box", single, boxes, models, arcs))
    checks.extend(_validate_plan("ffd", ffd, boxes, models, arcs))
    checks.append({"check_id": "ffd:sorties_not_above_single_box",
                   "status": "PASS" if len(ffd) <= len(single) else "FAIL",
                   "detail": f"ffd={len(ffd)}, single={len(single)}"})
    metrics = [_metrics("single_box", single), _metrics("ffd", ffd)]
    for metric in metrics:
        checks.append({"check_id": f"{metric['baseline']}:reconcile_totals",
                       "status": "PASS" if (metric["delivered_boxes"] == 80 and
                           math.isclose(metric["total_mass_kg"], 758, abs_tol=1e-8) and
                           math.isclose(metric["total_volume_m3"], 2.011, abs_tol=1e-8)) else "FAIL",
                       "detail": str(metric["delivered_boxes"])})
    return {"data_run": data_run, "max_payloads": max_payloads,
            "single_box": single, "ffd": ffd, "metrics": metrics, "checks": checks}


def _template_rows(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "架次编号": batch["batch_id"], "服务区编号": batch["zone_id"],
        "机型编号": batch["type_id"], "货箱编号列表": ";".join(batch["box_ids"]),
        "总质量（kg）": batch["mass_kg"], "总体积（m³）": batch["volume_m3"],
        "往返时间（s）": batch["operation_s"], "架次能耗（kWh）": batch["energy_kwh"],
        "返航SOC（%）": batch["return_soc"] * 100,
    } for batch in plan]


def _detail_rows(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("batch_id", "zone_id", "type_id", "box_ids", "box_count", "mass_kg",
            "volume_m3", "energy_out_kwh", "energy_back_kwh", "energy_kwh",
            "energy_limit_kwh", "return_soc", "flight_out_s", "flight_back_s",
            "flight_s", "prep_s", "loading_s", "handoff_s", "operation_s")
    return [{key: ";".join(batch[key]) if key == "box_ids" else batch[key] for key in keys}
            for batch in plan]


def _save_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def _verify_written_outputs(result: dict[str, Any], output_dir: Path) -> None:
    """复读落盘结果；写出过程有错时不得留下 Q1_READY 标记。"""
    for filename, plan in (("Q1_单箱保底.csv", result["single_box"]),
                           ("Q1_单点组批.csv", result["ffd"])):
        frame = pd.read_csv(output_dir / filename, dtype={column: str for column in
                            TEMPLATE_COLUMNS[:4]}, encoding="utf-8-sig")
        expected = _template_rows(plan)
        if list(frame.columns) != TEMPLATE_COLUMNS or len(frame) != len(expected):
            raise ValueError(f"导出复核失败：{filename} 的列名或行数不符")
        for index, row in enumerate(expected):
            actual = frame.iloc[index]
            for column in TEMPLATE_COLUMNS[:4]:
                if actual[column] != row[column]:
                    raise ValueError(f"导出复核失败：{filename} 第 {index + 2} 行 {column} 不符")
            for column in TEMPLATE_COLUMNS[4:]:
                if not math.isclose(float(actual[column]), float(row[column]),
                                    rel_tol=1e-11, abs_tol=1e-8):
                    raise ValueError(f"导出复核失败：{filename} 第 {index + 2} 行 {column} 不符")

    payloads = pd.read_csv(output_dir / "Q1_安全载荷.csv", encoding="utf-8-sig")
    expected_pairs = {(row["type_id"], row["zone_id"]) for row in result["max_payloads"]}
    actual_pairs = set(zip(payloads["type_id"], payloads["zone_id"]))
    if len(payloads) != 45 or actual_pairs != expected_pairs:
        raise ValueError("导出复核失败：安全载荷矩阵不是预期的 45 个机型-服务区组合")
    for label, filename in (("指标", "Q1_指标对照.csv"), ("验收", "Q1_验收检查.csv")):
        frame = pd.read_csv(output_dir / filename, encoding="utf-8-sig")
        expected_count = len(result["metrics"] if label == "指标" else result["checks"])
        if len(frame) != expected_count:
            raise ValueError(f"导出复核失败：{label}行数不符")
        if label == "验收" and not frame["status"].eq("PASS").all():
            raise ValueError("导出复核失败：验收文件存在失败项")


def save_results(result: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"不覆盖既有结果：{output_dir}")
    if any(check["status"] != "PASS" for check in result["checks"]):
        raise ValueError("第一问回代检查未全通过，结果不发布")
    if not verify_ready(result["data_run"]):
        raise ValueError("输入数据在求解期间发生改变")
    template = load_workbook(PROJECT / "0_3_数据" / "结果提交模板.xlsx", read_only=True, data_only=True)
    actual_header = [template["Q1_单点组批"].cell(1, col).value for col in range(1, 10)]
    template.close()
    if actual_header != TEMPLATE_COLUMNS:
        raise ValueError("官方 Q1 模板列名变化，停止导出以免错列")
    output_dir.mkdir(parents=True)
    _save_csv(output_dir / "Q1_安全载荷.csv", result["max_payloads"])
    _save_csv(output_dir / "Q1_单箱保底.csv", _template_rows(result["single_box"]), TEMPLATE_COLUMNS)
    _save_csv(output_dir / "Q1_单点组批.csv", _template_rows(result["ffd"]), TEMPLATE_COLUMNS)
    _save_csv(output_dir / "Q1_批次明细.csv", _detail_rows(result["ffd"]))
    _save_csv(output_dir / "Q1_指标对照.csv", result["metrics"])
    _save_csv(output_dir / "Q1_验收检查.csv", result["checks"])
    assumptions = {
        "source_data_run": str(result["data_run"]),
        "source_validated_artifacts_sha256": sha256(result["data_run"] / "meta" / "validated_artifacts.json"),
        "method": "single-box fallback and same-zone first-fit-decreasing; not optimized",
        "energy_formula_status": "modeling assumption from execution guide page 6, not an explicit official formula",
        "energy_horizontal": "E_use * distance_m / [L0 - (L0-LF)*(q/Q)^1.5]",
        "energy_climb": "(M_empty+q)*9.81*climb_m/(3600000*climb_efficiency)",
        "return_safety": "sortie_energy <= (1-rho)*E_use; return_SOC = 1-sortie_energy/E_use",
        "route": "O01 -> one service zone -> O01; outbound actual payload; return zero payload",
        "time_in_template": "full operation time = prep + per-box loading + both flight legs + one base handoff + per-box handoff",
        "cumulative_operation_time": "sum of full operation time over all sorties; not makespan",
        "excluded": ["entity-UAV scheduling", "battery scheduling", "delivery deadlines",
                     "exact set partitioning optimization", "return-margin sensitivity"],
        "metrics": result["metrics"],
        "checks_total": len(result["checks"]),
        "checks_passed": sum(check["status"] == "PASS" for check in result["checks"]),
    }
    (output_dir / "Q1_运行摘要.json").write_text(json.dumps(assumptions, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    explanation = (
        "D题第一问：可行基线，不是最优解。\n"
        "按官方题面只做单服务区 O01→Si→O01，不排实体无人机、共享电池和时限。\n"
        "Q1_安全载荷.csv：A/B/C×15区的45个安全载荷边界。\n"
        "Q1_单箱保底.csv：80箱各占一架次的保底方案。\n"
        "Q1_单点组批.csv：FFD组批，列名与官方Q1模板A:I完全一致。\n"
        "Q1_批次明细.csv：把飞行、准备、装载、交接、能耗分项列开，便于复算。\n"
        "模板的“往返时间”采用完整架次作业时间；纯飞行时间见批次明细。\n"
        "水平/爬升具体能耗公式是执行思路的建模假设，不是官方直接给出的公式。\n"
        "不包含集合划分优化、返航余量敏感性；这两项是第一问后续工作。\n"
    )
    (output_dir / "Q1_运行说明.txt").write_text(explanation, encoding="utf-8")
    _verify_written_outputs(result, output_dir)
    (output_dir / "Q1_READY.txt").write_text(
        f"PASS; {len(result['checks'])} checks; source={result['data_run']}\n", encoding="utf-8")


def find_latest_validated_run() -> Path:
    for path in sorted((CODE_DIR / "0_outputs").glob("run_*"), reverse=True):
        if path.is_dir() and verify_ready(path):
            return path
    raise FileNotFoundError("找不到通过验收的数据目录，请先运行 0_0_run_all.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="D题第一问单点往返基线")
    parser.add_argument("--data-run", type=Path, help="指定已验收数据运行目录；默认最新通过验收的一次")
    parser.add_argument("--output-dir", type=Path, help="指定新的结果目录；默认自动生成")
    args = parser.parse_args()
    data_run = args.data_run.resolve() if args.data_run else find_latest_validated_run()
    result = run_model(data_run)
    output_dir = args.output_dir.resolve() if args.output_dir else (
        CODE_DIR / "1_outputs" / datetime.now().strftime("q1_base_%y%m%d_%H%M%S_table"))
    save_results(result, output_dir)
    single, ffd = result["metrics"]
    print(f"Q1 baseline PASS: 45 safe loads, 80 boxes, {len(result['checks'])} checks")
    print(f"single-box: {single['sorties']} sorties, {single['total_energy_kwh']:.6f} kWh, {single['cumulative_operation_s']:.3f} s")
    print(f"FFD: {ffd['sorties']} sorties, {ffd['total_energy_kwh']:.6f} kWh, {ffd['cumulative_operation_s']:.3f} s")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
