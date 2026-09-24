"""D题第一问：基于同一物理计算器的精确集合划分组批优化。"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csr_array, vstack

from q1_0_baseline import (
    CODE_DIR, TEMPLATE_COLUMNS, _metrics, _read_inputs, _save_csv, _template_rows,
    _validate_plan, configure_plot_style, figure_dir_for, find_latest_validated_run, leg_time_s,
    roundtrip_energy_kwh, run_model, sortie_metrics,
)
from _0_pipeline import sha256, verify_ready
import matplotlib
import matplotlib.pyplot as plt


TOL = 1e-9


@dataclass(frozen=True)
class Candidate:
    zone_id: str
    box_ids: tuple[str, ...]
    type_id: str
    mass_kg: float
    volume_m3: float
    energy_kwh: float
    operation_s: float


def _nondominated(options: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """同一箱组内删除能耗与时间均不优的机型，保留型号较小的平局项。"""
    kept = []
    for type_id, energy, operation in options:
        dominated = any(
            (other_energy <= energy + 1e-10 and other_time <= operation + 1e-8 and
             (other_energy < energy - 1e-10 or other_time < operation - 1e-8 or
              other_id < type_id))
            for other_id, other_energy, other_time in options if other_id != type_id
        )
        if not dominated:
            kept.append((type_id, energy, operation))
    return kept


def enumerate_candidates(boxes: list[dict[str, Any]], models: dict[str, dict[str, Any]],
                         arcs: dict[tuple[str, str], dict[str, Any]]) -> list[Candidate]:
    """逐区列举不可拆箱组；每个候选架次单独验载重、体积和返航能量。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for box in boxes:
        grouped[str(box["zone_id"])].append(box)
    candidates: list[Candidate] = []
    max_mass = max(float(model["max_payload_kg"]) for model in models.values())
    max_volume = max(float(model["capacity_m3"]) for model in models.values())
    for zone in sorted(grouped):
        local = sorted(grouped[zone], key=lambda box: str(box["box_id"]))
        n = len(local)
        if n > 20:
            raise ValueError(f"{zone} 有 {n} 箱，完整子集枚举超出安全规模")
        outward, homeward = arcs[("O01", zone)], arcs[(zone, "O01")]
        base_time = {
            type_id: (leg_time_s(model, outward) + leg_time_s(model, homeward) +
                      float(model["prep_s"]) + float(model["handoff_base_s"]))
            for type_id, model in models.items()
        }
        size = 1 << n
        masses = np.zeros(size, dtype=float)
        volumes = np.zeros(size, dtype=float)
        counts = np.zeros(size, dtype=np.int16)
        ids = [str(box["box_id"]) for box in local]
        for mask in range(1, size):
            bit = mask & -mask
            index = bit.bit_length() - 1
            previous = mask ^ bit
            masses[mask] = masses[previous] + float(local[index]["mass_kg"])
            volumes[mask] = volumes[previous] + float(local[index]["volume_m3"])
            counts[mask] = counts[previous] + 1
            mass, volume = masses[mask], volumes[mask]
            if mass > max_mass + TOL or volume > max_volume + TOL:
                continue
            options: list[tuple[str, float, float]] = []
            for type_id in sorted(models):
                model = models[type_id]
                if (mass > float(model["max_payload_kg"]) + TOL or
                        volume > float(model["capacity_m3"]) + TOL):
                    continue
                energy = roundtrip_energy_kwh(model, outward, homeward, mass)
                limit = (1.0 - float(model["return_soc_min"])) * float(model["energy_use_kwh"])
                if energy > limit + TOL:
                    continue
                operation = (base_time[type_id] + int(counts[mask]) *
                             (float(model["load_per_box_s"]) +
                              float(model["handoff_per_box_s"])))
                options.append((type_id, energy, operation))
            if not options:
                continue
            box_ids = tuple(ids[index] for index in range(n) if mask & (1 << index))
            for type_id, energy, operation in _nondominated(options):
                candidates.append(Candidate(zone, box_ids, type_id, mass, volume,
                                            energy, operation))
    source_ids = {str(box["box_id"]) for box in boxes}
    covered_ids = {box_id for item in candidates for box_id in item.box_ids}
    if covered_ids != source_ids:
        raise ValueError(f"候选架次未覆盖全部货箱：{sorted(source_ids - covered_ids)}")
    return candidates


def solve_partition(candidates: list[Candidate], box_ids: list[str], *,
                    objective: str, exact_sorties: int | None = None,
                    time_cap_s: float | None = None,
                    time_limit_s: float = 120.0) -> dict[str, Any]:
    """稀疏 0-1 集合划分；仅 status=optimal 才声明当前目标被证明最优。"""
    if objective not in {"sorties", "energy", "time"}:
        raise ValueError(f"未知目标：{objective}")
    unique_ids = sorted(set(box_ids))
    if len(unique_ids) != len(box_ids) or not candidates:
        raise ValueError("货箱编号重复或没有候选架次")
    index = {box_id: position for position, box_id in enumerate(unique_ids)}
    row_indices, col_indices = [], []
    for column, candidate in enumerate(candidates):
        for box_id in candidate.box_ids:
            if box_id not in index:
                raise ValueError(f"候选架次包含未知箱号：{box_id}")
            row_indices.append(index[box_id])
            col_indices.append(column)
    n = len(candidates)
    coverage = coo_array((np.ones(len(row_indices)),
                          (np.asarray(row_indices), np.asarray(col_indices))),
                         shape=(len(unique_ids), n)).tocsr()
    parts = [coverage]
    lower = [np.ones(len(unique_ids))]
    upper = [np.ones(len(unique_ids))]
    if exact_sorties is not None:
        parts.append(csr_array(np.ones((1, n))))
        lower.append(np.array([float(exact_sorties)]))
        upper.append(np.array([float(exact_sorties)]))
    if time_cap_s is not None:
        parts.append(csr_array(np.array([[item.operation_s for item in candidates]])))
        lower.append(np.array([-np.inf]))
        upper.append(np.array([float(time_cap_s)]))
    matrix = vstack(parts, format="csc")
    costs = np.array([
        1.0 if objective == "sorties" else
        item.energy_kwh if objective == "energy" else item.operation_s
        for item in candidates
    ], dtype=float)
    started = time.perf_counter()
    solution = milp(
        c=costs, integrality=np.ones(n, dtype=np.int8), bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, np.concatenate(lower), np.concatenate(upper)),
        options={"time_limit": float(time_limit_s), "mip_rel_gap": 0.0},
    )
    elapsed = time.perf_counter() - started
    chosen: list[Candidate] = []
    if solution.x is not None:
        chosen = [item for item, value in zip(candidates, solution.x) if value > 0.5]
        seen = Counter(box_id for item in chosen for box_id in item.box_ids)
        valid = (seen == Counter(unique_ids) and
                 (exact_sorties is None or len(chosen) == exact_sorties) and
                 (time_cap_s is None or sum(item.operation_s for item in chosen) <= time_cap_s + 1e-5))
        if not valid:
            raise AssertionError("整数规划返回的方案未通过独立覆盖/架次/时限检查")
    label = {0: "optimal", 1: "time_or_node_limit", 2: "infeasible",
             3: "unbounded", 4: "solver_error"}.get(int(solution.status), "unknown")
    return {
        "status": label, "selected": chosen, "objective": objective,
        "exact_sorties": exact_sorties, "time_cap_s": time_cap_s,
        "objective_value": None if solution.fun is None else float(solution.fun),
        "dual_bound": None if getattr(solution, "mip_dual_bound", None) is None else
                      float(solution.mip_dual_bound),
        "mip_gap": None if getattr(solution, "mip_gap", None) is None else float(solution.mip_gap),
        "node_count": None if getattr(solution, "mip_node_count", None) is None else
                      int(solution.mip_node_count),
        "elapsed_s": elapsed, "message": str(solution.message),
    }


def replay_selection(selected: list[Candidate], source_boxes: list[dict[str, Any]],
                     models: dict[str, dict[str, Any]],
                     arcs: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    source = {str(box["box_id"]): box for box in source_boxes}
    seen = Counter(box_id for item in selected for box_id in item.box_ids)
    if seen != Counter(source.keys()):
        raise ValueError("优化架次未将每箱恰好配送一次")
    plan = []
    for sequence, item in enumerate(sorted(selected,
                                           key=lambda part: (part.zone_id, part.box_ids, part.type_id)), 1):
        boxes = [source[box_id] for box_id in item.box_ids]
        if any(str(box["zone_id"]) != item.zone_id for box in boxes):
            raise ValueError("优化架次跨越多个服务区")
        result = sortie_metrics(models[item.type_id], arcs[("O01", item.zone_id)],
                                arcs[(item.zone_id, "O01")], boxes)
        if not result["feasible"]:
            raise ValueError(f"优化架次能量/容量回代失败：{item.zone_id}/{item.box_ids}")
        for key, expected in (("mass_kg", item.mass_kg), ("volume_m3", item.volume_m3),
                              ("energy_kwh", item.energy_kwh),
                              ("operation_s", item.operation_s)):
            if not math.isclose(float(result[key]), expected, abs_tol=1e-7):
                raise ValueError(f"优化候选计算与物理回代不符：{key}")
        plan.append(dict(result, batch_id=f"O{sequence:03d}", zone_id=item.zone_id,
                         type_id=item.type_id, box_ids=list(item.box_ids)))
    return plan


def zone_lower_bounds(boxes: list[dict[str, Any]], models: dict[str, dict[str, Any]],
                      payload_rows: list[dict[str, Any]],
                      ffd_plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """质量与体积给每个服务区的架次下界；与可行 FFD 相等则直接证最少架次。"""
    box_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for box in boxes:
        box_groups[str(box["zone_id"])].append(box)
    ffd_counts = Counter(batch["zone_id"] for batch in ffd_plan)
    volume_max = max(float(model["capacity_m3"]) for model in models.values())
    rows = []
    for zone in sorted(box_groups):
        local = box_groups[zone]
        mass = sum(float(box["mass_kg"]) for box in local)
        volume = sum(float(box["volume_m3"]) for box in local)
        zone_payloads = [float(row["q_max_kg"]) for row in payload_rows
                         if row["zone_id"] == zone and row["q_max_kg"] is not None]
        if not zone_payloads:
            raise ValueError(f"{zone} 对所有机型均不可达")
        max_safe = max(zone_payloads)
        lb_mass = math.ceil((mass - TOL) / max_safe)
        lb_volume = math.ceil((volume - TOL) / volume_max)
        lower_bound = max(1, lb_mass, lb_volume)
        ffd = ffd_counts[zone]
        if ffd < lower_bound:
            raise AssertionError(f"{zone} 的基线架次少于物理下界")
        rows.append({"zone_id": zone, "box_count": len(local), "mass_kg": mass,
                     "volume_m3": volume, "max_safe_payload_kg": max_safe,
                     "mass_lower_bound": lb_mass, "volume_lower_bound": lb_volume,
                     "sortie_lower_bound": lower_bound, "ffd_sorties": ffd,
                     "matched_lower_bound": ffd == lower_bound})
    return rows


def _solution_record(name: str, solution: dict[str, Any],
                     plan: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = _metrics(name, plan)
    return {"name": name, "solver": solution, "plan": plan, "metrics": metrics}


def _require_solution(name: str, solution: dict[str, Any],
                      boxes: list[dict[str, Any]], models: dict[str, dict[str, Any]],
                      arcs: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    if solution["status"] != "optimal":
        raise RuntimeError(f"{name} 未证明最优：{solution['status']}；{solution['message']}")
    plan = replay_selection(solution["selected"], boxes, models, arcs)
    return _solution_record(name, solution, plan)


def optimize_q1(data_run: Path, *, time_limit_s: float = 120,
                explore_extra_sortie: bool = True) -> dict[str, Any]:
    """逐区可行候选 + 全局集合划分，输出已证明最优的极值和代表性权衡点。"""
    data_run = data_run.resolve()
    baseline = run_model(data_run)
    boxes, models, arcs = _read_inputs(data_run)
    if any(check["status"] != "PASS" for check in baseline["checks"]):
        raise ValueError("基础组批未通过验收")
    lower_bounds = zone_lower_bounds(boxes, models, baseline["max_payloads"], baseline["ffd"])
    candidates = enumerate_candidates(boxes, models, arcs)
    ids = [str(box["box_id"]) for box in boxes]
    lower_total = sum(row["sortie_lower_bound"] for row in lower_bounds)
    if all(row["matched_lower_bound"] for row in lower_bounds):
        minimum_sorties = lower_total
        minimum_proof = "zone_mass_volume_lower_bound_equals_feasible_ffd"
        minimum_solver = None
    else:
        minimum_solver = solve_partition(candidates, ids, objective="sorties",
                                         time_limit_s=time_limit_s)
        if minimum_solver["status"] != "optimal":
            raise RuntimeError("最少架次尚未证明，停止输出优化结论")
        minimum_sorties = len(minimum_solver["selected"])
        minimum_proof = "scipy_milp_optimal"

    results: list[dict[str, Any]] = []
    for name, objective in ((f"{minimum_sorties}架_能耗最低", "energy"),
                            (f"{minimum_sorties}架_时间最短", "time")):
        solved = solve_partition(candidates, ids, objective=objective,
                                 exact_sorties=minimum_sorties, time_limit_s=time_limit_s)
        results.append(_require_solution(name, solved, boxes, models, arcs))
    # 代表性 ε 约束：多一架次时，在两个极值时间的中点限制内再最小化能耗。
    if explore_extra_sortie:
        extra = minimum_sorties + 1
        for name, objective in ((f"{extra}架_能耗最低", "energy"),
                                (f"{extra}架_时间最短", "time")):
            solved = solve_partition(candidates, ids, objective=objective,
                                     exact_sorties=extra, time_limit_s=time_limit_s)
            results.append(_require_solution(name, solved, boxes, models, arcs))
        extra_energy = results[-2]["metrics"]
        extra_fast = results[-1]["metrics"]
        if extra_energy["cumulative_operation_s"] > extra_fast["cumulative_operation_s"] + 1e-5:
            cap = (extra_energy["cumulative_operation_s"] +
                   extra_fast["cumulative_operation_s"]) / 2.0
            solved = solve_partition(candidates, ids, objective="energy", exact_sorties=extra,
                                     time_cap_s=cap, time_limit_s=time_limit_s)
            results.append(_require_solution(f"{extra}架_时间约束折中", solved, boxes, models, arcs))

    checks = list(baseline["checks"])
    checks.append({"check_id": "optimization:minimum_sorties_proven",
                   "status": "PASS" if minimum_sorties >= lower_total else "FAIL",
                   "detail": f"lower={lower_total}; minimum={minimum_sorties}; proof={minimum_proof}"})
    for item in results:
        checks.extend(_validate_plan(item["name"], item["plan"], boxes, models, arcs))
        checks.append({"check_id": item["name"] + ":solver_optimal",
                       "status": "PASS" if item["solver"]["status"] == "optimal" else "FAIL",
                       "detail": str(item["solver"]["mip_gap"])})
    if any(check["status"] != "PASS" for check in checks):
        raise ValueError("优化方案物理回代或求解证明未通过")
    return {"data_run": data_run, "baseline": baseline, "candidates": candidates,
            "lower_bounds": lower_bounds, "minimum_sorties": minimum_sorties,
            "minimum_proof": minimum_proof, "minimum_solver": minimum_solver,
            "solutions": results, "checks": checks}


def _result_metrics_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for baseline_metric in result["baseline"]["metrics"]:
        rows.append(dict(baseline_metric, solver_status="feasible_baseline", mip_gap=None,
                         compared_to_ffd_energy_pct=None, compared_to_ffd_time_pct=None))
    ffd = result["baseline"]["metrics"][1]
    for item in result["solutions"]:
        metric = item["metrics"]
        rows.append(dict(metric, solver_status=item["solver"]["status"],
                         mip_gap=item["solver"]["mip_gap"],
                         compared_to_ffd_energy_pct=100 * (1 - metric["total_energy_kwh"] /
                                                         ffd["total_energy_kwh"]),
                         compared_to_ffd_time_pct=100 * (1 - metric["cumulative_operation_s"] /
                                                       ffd["cumulative_operation_s"])))
    for row in rows:
        row["nondominated_among_reported"] = not any(
            (other is not row and other["sorties"] <= row["sorties"] and
             other["total_energy_kwh"] <= row["total_energy_kwh"] + 1e-8 and
             other["cumulative_operation_s"] <= row["cumulative_operation_s"] + 1e-5 and
             (other["sorties"] < row["sorties"] or
              other["total_energy_kwh"] < row["total_energy_kwh"] - 1e-8 or
              other["cumulative_operation_s"] < row["cumulative_operation_s"] - 1e-5))
            for other in rows
        )
    return rows


def save_optimization(result: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"不覆盖已有优化结果：{output_dir}")
    if not verify_ready(result["data_run"]):
        raise ValueError("源数据在优化期间发生变化")
    output_dir.mkdir(parents=True)
    recommended = result["solutions"][0]
    recommended_filename = f"Q1_单点组批_推荐{result['minimum_sorties']}架.csv"
    _save_csv(output_dir / recommended_filename,
              _template_rows(recommended["plan"]), TEMPLATE_COLUMNS)
    for item in result["solutions"][1:]:
        _save_csv(output_dir / f"Q1_单点组批_{item['name']}.csv",
                  _template_rows(item["plan"]), TEMPLATE_COLUMNS)
    _save_csv(output_dir / "Q1_安全载荷.csv", result["baseline"]["max_payloads"])
    _save_csv(output_dir / "Q1_最少架次下界.csv", result["lower_bounds"])
    metrics_rows = _result_metrics_rows(result)
    _save_csv(output_dir / "Q1_基线优化指标对照.csv", metrics_rows)
    _save_csv(output_dir / "Q1_验收检查.csv", result["checks"])
    solver_rows = []
    for item in result["solutions"]:
        solver = item["solver"]
        solver_rows.append({key: value for key, value in solver.items() if key != "selected"} |
                           {"scenario": item["name"], "selected_sorties": len(item["plan"])})
    _save_csv(output_dir / "Q1_求解记录.csv", solver_rows)
    summary = {
        "source_data_run": str(result["data_run"]),
        "source_validated_artifacts_sha256": sha256(result["data_run"] / "meta" / "validated_artifacts.json"),
        "method": "exact set partitioning over feasible same-zone box subsets and model types",
        "energy_formula_status": "modeling assumption from execution guide, not an explicit official formula",
        "minimum_sorties": result["minimum_sorties"],
        "minimum_proof": result["minimum_proof"],
        "candidate_count": len(result["candidates"]),
        "candidate_count_by_zone": dict(sorted(Counter(c.zone_id for c in result["candidates"]).items())),
        "baseline_ffd": result["baseline"]["metrics"][1],
        "optimization_metrics": [{"name": item["name"], "metrics": item["metrics"],
                                  "solver_status": item["solver"]["status"],
                                  "mip_gap": item["solver"]["mip_gap"]}
                                 for item in result["solutions"]],
        "nondominated_among_reported": [row["baseline"] for row in metrics_rows
                                         if row["nondominated_among_reported"]],
        "checks_total": len(result["checks"]),
        "checks_passed": sum(check["status"] == "PASS" for check in result["checks"]),
        "excluded": ["entity-UAV scheduling", "battery scheduling", "delivery deadlines",
                     "return-margin sensitivity"],
    }
    (output_dir / "Q1_优化摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "Q1_运行说明.txt").write_text(
        "D题第一问精确组批。先用逐区质量/体积下界证明最少架次，再用0-1集合划分求能耗和累计作业时间极值。\n"
        "推荐方案为最少架次下的最低能耗解；其他CSV是时间极值或多一架次的权衡方案，不是最终推荐。\n"
        "每箱恰好一次；每架只去一个区；容量、体积、能耗和返航SOC均用基线物理计算器回代。\n"
        "返航余量敏感性尚未进行；不涉及实体无人机、电池与逐箱时限。\n",
        encoding="utf-8")
    # 先从实际导出的 CSV 复读，避免标记发布损坏或错列文件。
    for filename, expected_plan in (
        [(recommended_filename, recommended["plan"])] +
        [(f"Q1_单点组批_{item['name']}.csv", item["plan"])
         for item in result["solutions"][1:]]
    ):
        frame = pd.read_csv(output_dir / filename, encoding="utf-8-sig",
                            dtype={column: str for column in TEMPLATE_COLUMNS[:4]})
        if list(frame.columns) != TEMPLATE_COLUMNS or len(frame) != len(expected_plan):
            raise ValueError("优化方案导出列名或行数不正确")
        ids = [box_id for cell in frame["货箱编号列表"] for box_id in cell.split(";")]
        if len(ids) != 80 or len(set(ids)) != 80:
            raise ValueError("优化方案导出后不能覆盖全部80箱")
        expected_energy = sum(batch["energy_kwh"] for batch in expected_plan)
        if not math.isclose(frame["架次能耗（kWh）"].sum(), expected_energy, abs_tol=1e-6):
            raise ValueError("优化方案导出后能耗总计不一致")
    (output_dir / "Q1_OPT_READY.txt").write_text(
        f"PASS; {len(result['checks'])} checks; source={result['data_run']}\n", encoding="utf-8")


def make_figure(result_dir: Path, output_dir: Path) -> None:
    result_dir = result_dir.resolve()
    if not (result_dir / "Q1_OPT_READY.txt").is_file():
        raise ValueError("优化结果缺少可用标记")
    if output_dir.exists():
        raise FileExistsError(f"不覆盖已有图表：{output_dir}")
    source = result_dir / "Q1_基线优化指标对照.csv"
    checksum = sha256(source)
    data = pd.read_csv(source, encoding="utf-8-sig")
    qa = pd.read_csv(result_dir / "Q1_验收检查.csv", encoding="utf-8-sig")
    if qa.empty or not qa["status"].eq("PASS").all():
        raise ValueError("优化验收未全通过，不应绘图")
    needed = {"ffd", "18架_能耗最低", "18架_时间最短",
              "19架_能耗最低", "19架_时间最短", "19架_时间约束折中"}
    if not needed.issubset(set(data["baseline"])):
        raise ValueError("对照表缺少必要方案")
    if not data.loc[data["baseline"] != "single_box", "solver_status"].isin(
        ["optimal", "feasible_baseline"]
    ).all():
        raise ValueError("图表包含未证明或未验收的方案")
    displayed = data[data["baseline"] != "single_box"].copy()
    displayed["operation_h"] = displayed["cumulative_operation_s"] / 3600.0
    ffd = displayed.set_index("baseline").loc["ffd"]
    recommended = displayed.set_index("baseline").loc["18架_能耗最低"]
    if recommended["sorties"] != 18 or recommended["total_energy_kwh"] >= ffd["total_energy_kwh"]:
        raise ValueError("18架推荐解的指标与预期不符")

    configure_plot_style()
    fig = plt.figure(figsize=(9.2, 5.1), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[14, 1.3])
    ax = fig.add_subplot(grid[0])
    note = fig.add_subplot(grid[1])
    note.axis("off")
    dominated = displayed[(~displayed["nondominated_among_reported"]) &
                          (displayed["baseline"] != "ffd")]
    ax.scatter(dominated["operation_h"], dominated["total_energy_kwh"],
               marker="x", s=90, linewidths=1.8, color="#8A97A5",
               label="其他已求解方案（被支配）", zorder=3)
    markers = [
        ("ffd", "FFD基线 · 18架", "#526174", "s", 125),
        ("18架_能耗最低", "推荐优化 · 18架", "#007D82", "o", 170),
        ("19架_能耗最低", "最低能耗 · 19架", "#C66A1C", "^", 160),
    ]
    table = displayed.set_index("baseline")
    for name, label, color, marker, size in markers:
        row = table.loc[name]
        ax.scatter([row["operation_h"]], [row["total_energy_kwh"]],
                   marker=marker, s=size, color=color, edgecolors="white",
                   linewidths=1.1, label=label, zorder=5)
    for name, offset, align in [
        ("ffd", (8, 10), "left"),
        ("18架_能耗最低", (8, -19), "left"),
        ("19架_能耗最低", (-8, -22), "right"),
    ]:
        row = table.loc[name]
        short_name = {"ffd": "FFD基线", "18架_能耗最低": "18架推荐",
                      "19架_能耗最低": "19架节能"}[name]
        label = f"{short_name}：{row['total_energy_kwh']:.3f} kWh，{row['operation_h']:.3f} h"
        ax.annotate(label, (row["operation_h"], row["total_energy_kwh"]),
                    xytext=offset, textcoords="offset points", fontsize=9,
                    ha=align, va="center", color="#22313C")
    xs, ys = displayed["operation_h"], displayed["total_energy_kwh"]
    ax.set_xlim(xs.min() - 0.06, xs.max() + 0.22)
    ax.set_ylim(ys.min() - 0.12, ys.max() + 0.17)
    ax.set_xlabel("累计作业时间（小时，各架次时间之和）")
    ax.set_ylabel("总运输能耗（kWh）")
    ax.grid(color="#D8DEE5", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    note.text(0, 0.5,
              "坐标轴展示局部区间；精确数值见随图CSV。灰叉为已求解但被支配的方案；本问不含实体机/电池排程。",
              ha="left", va="center", fontsize=8.8, color="#364451")

    output_dir.mkdir(parents=True)
    stem = output_dir / "图3_第一问组批优化权衡"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    if sha256(source) != checksum:
        raise ValueError("作图期间源指标表发生改变")
    for extension in ("pdf", "png"):
        if stem.with_suffix("." + extension).stat().st_size < 10_000:
            raise ValueError("图表导出文件异常过小")
    shutil.copy2(source, output_dir / source.name)
    provenance = {
        "source_result_dir": str(result_dir),
        "source_metrics_sha256": checksum,
        "source_qa_passed": len(qa),
        "transformation": "cumulative_operation_s / 3600 -> hours; no filtering except excluding the 80-sortie fallback from the local-scale chart",
        "shown_scenarios": displayed["baseline"].tolist(),
        "excluded_from_figure": "single_box is 80 sorties and 138 kWh; excluded from local-scale optimizer comparison but retained in underlying CSV",
        "axes": "local point-plot ranges, not zero-based bars",
        "matplotlib_version": matplotlib.__version__,
    }
    (output_dir / "图表数据来源.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "图表说明.txt").write_text(
        "第一问精确组批：18架次已达到理论最少架次。\n"
        "同为18架，推荐方案在所列方案中同时改善能耗和累计作业时间。\n"
        "19架次可以再降低少量能耗，但累计作业时间增加；灰叉方案被其他方案支配。\n"
        "坐标轴是为观察权衡而截取的局部区间，比较时以附带CSV的准确数值为准。\n"
        "PNG可直接发消息，PDF可用于后续论文排版。\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="D题第一问精确组批优化")
    parser.add_argument("--data-run", type=Path, help="已验收数据目录；默认最新")
    parser.add_argument("--output-dir", type=Path, help="新的优化结果目录；默认自动命名")
    parser.add_argument("--figure-dir", type=Path, help="新的图表目录；默认与表格目录同名时间戳")
    parser.add_argument("--time-limit", type=float, default=120,
                        help="每个整数规划的时间上限（秒）")
    parser.add_argument("--skip-extra-sortie", action="store_true",
                        help="只求最少架次下的能耗和时间极值")
    args = parser.parse_args()
    data_run = args.data_run.resolve() if args.data_run else find_latest_validated_run()
    result = optimize_q1(data_run, time_limit_s=args.time_limit,
                         explore_extra_sortie=not args.skip_extra_sortie)
    output_dir = args.output_dir.resolve() if args.output_dir else (
        CODE_DIR / "1_outputs" / datetime.now().strftime("q1_opt_%y%m%d_%H%M%S_table"))
    figure_dir = args.figure_dir.resolve() if args.figure_dir else figure_dir_for(output_dir)
    if figure_dir.exists():
        raise FileExistsError(f"不覆盖既有图表：{figure_dir}")
    save_optimization(result, output_dir)
    if args.skip_extra_sortie:
        print("图表未生成：--skip-extra-sortie 未求解图3所需的19架方案")
    else:
        make_figure(output_dir, figure_dir)
    ffd = result["baseline"]["metrics"][1]
    recommended = result["solutions"][0]["metrics"]
    print(f"Q1 exact optimization PASS: {result['minimum_sorties']} minimum sorties, "
          f"{len(result['candidates'])} feasible candidates, {len(result['checks'])} checks")
    print(f"FFD: {ffd['total_energy_kwh']:.6f} kWh, {ffd['cumulative_operation_s']:.3f} s")
    print(f"Recommended: {recommended['total_energy_kwh']:.6f} kWh, "
          f"{recommended['cumulative_operation_s']:.3f} s")
    print(f"output: {output_dir}")
    if not args.skip_extra_sortie:
        print(f"figure: {figure_dir}")


if __name__ == "__main__":
    main()
