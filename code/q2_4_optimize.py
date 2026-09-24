"""问题二优化方案 4：逐箱换位、互换与排程顺序的可行邻域搜索。

从同一已验收数据的最佳零迟到方案热启动。严格保持全部硬截止、
80 箱覆盖和零软迟到，目标是在这些约束内进一步降低运输能耗。
算法为启发式，结果须以独立验收表为准，不声称全局最优。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from _0_pipeline import sha256, verify_ready
from q1_0_baseline import CODE_DIR, find_latest_validated_run
from q2_0_baseline import write_csv
from q2_1_optimize import (
    TOL, Context, Route, check_plan, save_figures, save_tables, simulate,
)
from q2_3_optimize import initial_solution, insertion_options, is_admissible, merge_step, zone_distance


def quality(plan: dict[str, Any]) -> float:
    """能耗为主，最晚返航仅用于能耗近似相同时打破平局。"""
    return float(plan["score"][4]) + 0.00001 * float(plan["score"][3])


def improves(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    candidate_energy = float(candidate["score"][4])
    incumbent_energy = float(incumbent["score"][4])
    return (candidate_energy < incumbent_energy - TOL or
            (abs(candidate_energy - incumbent_energy) <= TOL and
             candidate["score"][3] < incumbent["score"][3] - TOL))


def without_box(ctx: Context, route: Route, box_id: str) -> Route | None:
    visits = [(zone, [bid for bid in ids if bid != box_id])
              for zone, ids in route.visits]
    visits = [(zone, ids) for zone, ids in visits if ids]
    return ctx.route(visits) if visits else None


def routes_after_relocation(ctx: Context, routes: list[Route], donor: int,
                            target: int, box_id: str) -> list[list[Route]]:
    remainder = without_box(ctx, routes[donor], box_id)
    output = []
    for inserted in insertion_options(ctx, routes[target], box_id):
        candidate = list(routes)
        candidate[target] = inserted
        if remainder is None:
            candidate.pop(donor)
        else:
            candidate[donor] = remainder
        if remainder is None or ctx.options(remainder):
            output.append(candidate)
    return output


def best_relocation(ctx: Context, incumbent: dict[str, Any],
                    deadline: float, max_targets: int) -> tuple[dict[str, Any], int]:
    """系统扫描逐箱挪动；优先考察单位货箱能耗较高的架次。"""
    routes = incumbent["routes"]
    energy = {s["batch_id"]: float(s["energy_kwh"])
              for s in incumbent["sorties"]}
    donors = sorted(range(len(routes)),
                    key=lambda i: energy[f"Q{i + 1:03d}"] / len(routes[i].box_ids),
                    reverse=True)
    best = incumbent
    checked = 0
    for donor in donors:
        if time.perf_counter() >= deadline:
            break
        for box_id in routes[donor].box_ids:
            zone = str(ctx.boxes[box_id]["zone_id"])
            targets = sorted((i for i in range(len(routes)) if i != donor),
                             key=lambda i: (zone_distance(ctx, zone, routes[i]), i))
            for target in targets[:max_targets]:
                if time.perf_counter() >= deadline:
                    break
                for candidate in routes_after_relocation(ctx, routes, donor, target, box_id):
                    trial = simulate(ctx, candidate)
                    checked += 1
                    if is_admissible(trial, 0.0) and improves(trial, best):
                        best = trial
    return best, checked


def best_order(ctx: Context, incumbent: dict[str, Any],
               deadline: float) -> tuple[dict[str, Any], int]:
    """路线保持不变，逐一试探任务先后顺序。"""
    routes = incumbent["routes"]
    best = incumbent
    checked = 0
    for source in range(len(routes)):
        if time.perf_counter() >= deadline:
            break
        for destination in range(len(routes)):
            if source == destination or time.perf_counter() >= deadline:
                continue
            candidate = list(routes)
            candidate.insert(destination, candidate.pop(source))
            trial = simulate(ctx, candidate)
            checked += 1
            if is_admissible(trial, 0.0) and improves(trial, best):
                best = trial
    return best, checked


def random_swap(ctx: Context, routes: list[Route],
                rng: random.Random) -> list[Route] | None:
    if len(routes) < 2:
        return None
    first, second = rng.sample(range(len(routes)), 2)
    box_a = rng.choice(routes[first].box_ids)
    box_b = rng.choice(routes[second].box_ids)
    reduced_a = without_box(ctx, routes[first], box_a)
    reduced_b = without_box(ctx, routes[second], box_b)
    options_a = (insertion_options(ctx, reduced_a, box_b) if reduced_a
                 else [ctx.route([(str(ctx.boxes[box_b]["zone_id"]), [box_b])])])
    options_b = (insertion_options(ctx, reduced_b, box_a) if reduced_b
                 else [ctx.route([(str(ctx.boxes[box_a]["zone_id"]), [box_a])])])
    choices = [(a, b) for a, b in itertools.product(options_a, options_b)
               if ctx.options(a) and ctx.options(b)]
    if not choices:
        return None
    changed = list(routes)
    changed[first], changed[second] = rng.choice(choices)
    return changed


def random_neighbor(ctx: Context, current: dict[str, Any],
                    rng: random.Random, max_targets: int) -> list[Route] | None:
    routes = current["routes"]
    if len(routes) < 2:
        return None
    kind = rng.choices(["relocate", "swap", "order", "zone_order"],
                       weights=[38, 28, 26, 8])[0]
    if kind == "relocate":
        donor = rng.randrange(len(routes))
        box_id = rng.choice(routes[donor].box_ids)
        zone = str(ctx.boxes[box_id]["zone_id"])
        targets = sorted((i for i in range(len(routes)) if i != donor),
                         key=lambda i: (zone_distance(ctx, zone, routes[i]), i))
        if not targets:
            return None
        target = rng.choice(targets[:max_targets])
        choices = routes_after_relocation(ctx, routes, donor, target, box_id)
        return rng.choice(choices) if choices else None
    if kind == "swap":
        return random_swap(ctx, routes, rng)
    if kind == "order":
        source, destination = rng.sample(range(len(routes)), 2)
        changed = list(routes)
        changed.insert(destination, changed.pop(source))
        return changed
    choices = [i for i, route in enumerate(routes) if len(route.visits) > 1]
    if not choices:
        return None
    index = rng.choice(choices)
    route = routes[index]
    order = list(route.visits)
    rng.shuffle(order)
    changed_route = Route(tuple(order))
    if changed_route == route or not ctx.options(changed_route):
        return None
    changed = list(routes)
    changed[index] = changed_route
    return changed


def search(ctx: Context, start: dict[str, Any], seconds: float, seed: int,
           max_targets: int) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, float]]]:
    rng = random.Random(seed)
    start_clock = time.perf_counter()
    deadline = start_clock + seconds
    best = current = start
    proposals = feasible = improvements = 0
    history = [{"elapsed_s": 0.0, "energy_kwh": best["score"][4],
                "sorties": best["score"][5]}]
    # 先对方案 3 较少覆盖的逐箱挪动和路线顺序做系统下降。
    for scan in (best_relocation, best_order):
        if time.perf_counter() >= deadline:
            break
        budget_end = min(deadline, time.perf_counter() + seconds * 0.22)
        if scan is best_relocation:
            trial, count = scan(ctx, best, budget_end, max_targets)
        else:
            trial, count = scan(ctx, best, budget_end)
        proposals += count
        if improves(trial, best):
            best = current = trial
            improvements += 1
            history.append({"elapsed_s": time.perf_counter() - start_clock,
                            "energy_kwh": best["score"][4],
                            "sorties": best["score"][5]})
            print(f"improved: {best['score'][4]:.3f} kWh, "
                  f"{best['score'][5]} sorties", flush=True)
    # 再以可行性过滤的随机换箱与任务顺序扰动逃出局部最优。
    while time.perf_counter() < deadline:
        candidate = random_neighbor(ctx, current, rng, max_targets)
        if candidate is None:
            continue
        trial = simulate(ctx, candidate)
        proposals += 1
        if not is_admissible(trial, 0.0):
            continue
        feasible += 1
        if improves(trial, best):
            best = trial
            improvements += 1
            history.append({"elapsed_s": time.perf_counter() - start_clock,
                            "energy_kwh": best["score"][4],
                            "sorties": best["score"][5]})
            print(f"improved: {best['score'][4]:.3f} kWh, "
                  f"{best['score'][5]} sorties", flush=True)
        delta = quality(trial) - quality(current)
        remaining = max(0.0, (deadline - time.perf_counter()) / seconds)
        temperature = 0.03 + 0.65 * remaining
        if delta <= 0 or rng.random() < math.exp(-min(delta / temperature, 700)):
            current = trial
        if proposals % 180 == 0:
            current = best
            merged = merge_step(ctx, best, 0.0, 0.00001, 0.0, deadline)
            if improves(merged, best):
                best = current = merged
                improvements += 1
                history.append({"elapsed_s": time.perf_counter() - start_clock,
                                "energy_kwh": best["score"][4],
                                "sorties": best["score"][5]})
    elapsed = time.perf_counter() - start_clock
    history.append({"elapsed_s": elapsed, "energy_kwh": best["score"][4],
                    "sorties": best["score"][5]})
    return best, {"seconds": elapsed,
                  "proposals": proposals, "feasible_neighbors": feasible,
                  "improvements": improvements, "seed": seed,
                  "global_optimality_proven": False}, history


def compare_methods(output_root: Path, source_hash: str,
                    latest: dict[str, Any], latest_sorties: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for folder, pattern, label in (
        ("0_baseline", "q2_base_*_table", "固定FFD基线"),
        ("1_optimize", "q2_opt_*_table", "优化方案1"),
        ("2_optimize", "q2_opt2_*_table", "优化方案2"),
        ("3_optimize", "q2_opt3_*_table", "优化方案3（均衡推荐）"),
    ):
        directories = [*(output_root / folder).glob(pattern),
                       *output_root.glob(pattern)]
        for directory in sorted(directories, key=lambda path: path.stat().st_mtime,
                                reverse=True):
            summary_file = directory / "Q2_运行摘要.json"
            sortie_file = directory / "Q2_运输架次.csv"
            if not summary_file.is_file() or not sortie_file.is_file():
                continue
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            if (summary.get("source_manifest_sha256") != source_hash or
                    (folder == "0_baseline" and summary.get("baseline_method_id") !=
                     "q2_baseline_abc_single_zone_v2")):
                continue
            with sortie_file.open("r", encoding="utf-8-sig", newline="") as stream:
                type_counts = Counter(row["机型编号"] for row in csv.DictReader(stream))
            rows.append({"方案": label, "状态": summary["status"],
                         "硬时限违约箱数": summary["hard_deadline_violations"],
                         "软箱加权迟到（s）": summary["soft_weighted_lateness_s"],
                         "最晚返航（s）": summary["makespan_s"],
                         "运输总能耗（kWh）": summary["total_energy_kwh"],
                         "架次数": summary["sorties"],
                         **{f"{kind}型架次": type_counts[kind] for kind in "ABC"}})
            break
    latest_counts = Counter(sortie["type_id"] for sortie in latest_sorties)
    rows.append({"方案": "优化方案4", "状态": latest["status"],
                 "硬时限违约箱数": latest["hard_deadline_violations"],
                 "软箱加权迟到（s）": latest["soft_weighted_lateness_s"],
                 "最晚返航（s）": latest["makespan_s"],
                 "运输总能耗（kWh）": latest["total_energy_kwh"],
                 "架次数": latest["sorties"],
                 **{f"{kind}型架次": latest_counts[kind] for kind in "ABC"}})
    return rows


def save_comparison_figures(figure_dir: Path, comparison: list[dict[str, Any]]) -> None:
    """从同源结果生成五方案权衡和三机型架次图，不重算任何方案。"""
    note_file = figure_dir / "图表说明.txt"
    if len(comparison) != 5:
        with note_file.open("a", encoding="utf-8") as stream:
            stream.write("五方案对照图未生成：缺少同一已验收数据下某个方法的结果。\n")
        return
    os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIR / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = ["基线", "优化1", "优化2", "优化3", "优化4"]
    colors = ["#777777", "#3A80C1", "#27A275", "#8361AA", "#DB873F"]
    offsets = [(-8, 10), (8, -16), (8, 9), (8, 9), (8, 10)]
    fig, ax = plt.subplots(figsize=(9, 6))
    for index, (label, row) in enumerate(zip(labels, comparison)):
        x = float(row["最晚返航（s）"]) / 3600
        y = float(row["运输总能耗（kWh）"])
        valid = row["状态"] == "PASS" and int(row["硬时限违约箱数"]) == 0
        soft_late = float(row["软箱加权迟到（s）"]) > TOL
        ax.scatter(x, y, s=8 * int(row["架次数"]),
                   color=colors[index] if valid else "#C64F4F",
                   marker="o" if valid else "X",
                   edgecolors="#B43838" if soft_late else "white",
                   linewidths=2 if soft_late else 0.8, zorder=3)
        ax.annotate(f"{label} · {row['架次数']}架次", (x, y),
                    xytext=offsets[index], textcoords="offset points", fontsize=9,
                    ha="right" if index == 0 else "left")
    ax.set(xlabel="最晚返航（小时）", ylabel="运输总能耗（kWh）")
    ax.margins(x=0.11, y=0.13)
    ax.grid(alpha=0.23)
    ax.text(0.02, 0.98, "点面积表示架次数；红边表示存在软箱迟到",
            transform=ax.transAxes, fontsize=8, color="#555555", va="top")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q2_五方案权衡.{extension}", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = [0] * len(comparison)
    for kind, color in (("A", "#3A80C1"), ("B", "#27A275"), ("C", "#DB873F")):
        counts = [int(row[f"{kind}型架次"]) for row in comparison]
        bars = ax.bar(labels, counts, bottom=bottoms, color=color, label=f"{kind}型",
                      edgecolor="white", width=0.65)
        for bar, count, bottom in zip(bars, counts, bottoms):
            if count:
                ax.text(bar.get_x() + bar.get_width() / 2, bottom + count / 2,
                        str(count), ha="center", va="center", fontsize=9, color="white")
        bottoms = [bottom + count for bottom, count in zip(bottoms, counts)]
    ax.set(ylabel="架次数", ylim=(0, max(bottoms) + 4))
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q2_五方案机型构成.{extension}", dpi=220)
    plt.close(fig)
    with note_file.open("a", encoding="utf-8") as stream:
        stream.write("五方案权衡：横轴为最晚返航、纵轴为总能耗，点面积为架次数；不是全局帕累托前沿。\n"
                     "五方案机型构成：同一数据哈希下各方法的 A/B/C 实际架次数。\n")


def save_progress_figure(figure_dir: Path, history: list[dict[str, float]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIR / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = [row["elapsed_s"] for row in history]
    y = [row["energy_kwh"] for row in history]
    ax.step(x, y, where="post", color="#2879B8", lw=1.8)
    ax.scatter(x[-1], y[-1], color="#C75849", zorder=3)
    ax.set_xlabel("Search time (s)")
    ax.set_ylabel("Best feasible energy (kWh)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q2_能耗搜索过程.{suffix}", dpi=220)
    plt.close(fig)
    note_file = figure_dir / "图表说明.txt"
    note = "能耗搜索过程：每次下降表示找到更低能耗的可行方案。\n"
    if note not in note_file.read_text(encoding="utf-8"):
        with note_file.open("a", encoding="utf-8") as file:
            file.write(note)


def main() -> None:
    parser = argparse.ArgumentParser(description="问题二优化方案4：逐箱挪动、互换与排程微调")
    parser.add_argument("--data-run", type=Path, help="已验收数据目录；默认最新")
    parser.add_argument("--output-root", type=Path, help="默认 code/2_outputs")
    parser.add_argument("--time-limit-s", type=float, default=180.0,
                        help="搜索总时间，默认180秒")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--targets", type=int, default=8,
                        help="逐箱挪动时优先考察的目标架次数")
    args = parser.parse_args()
    if args.time_limit_s <= 0 or args.targets < 1:
        parser.error("time-limit-s 和 targets 必须为正")
    data_run = args.data_run.resolve() if args.data_run else find_latest_validated_run()
    if not verify_ready(data_run):
        raise ValueError(f"输入数据未通过哈希验收：{data_run}")
    ctx = Context(data_run)
    output_root = args.output_root.resolve() if args.output_root else CODE_DIR / "2_outputs"
    start, origin = initial_solution(ctx, output_root, ("opt1", "opt2", "opt3"))
    if not all(check["status"] == "PASS" for check in check_plan(ctx, start)):
        raise ValueError("热启动方案未通过独立验收")
    print(f"warm start: {origin}; energy={start['score'][4]:.3f} kWh, "
          f"sorties={start['score'][5]}", flush=True)
    best, stats, history = search(ctx, start, args.time_limit_s, args.seed,
                                  args.targets)
    checks = check_plan(ctx, best)
    if not verify_ready(data_run):
        raise ValueError("搜索期间输入数据发生改变")
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    method_dir = output_root / "4_optimize"
    table_dir = method_dir / f"q2_opt4_{stamp}_table"
    figure_dir = method_dir / f"q2_opt4_{stamp}_figure"
    if table_dir.exists() or figure_dir.exists():
        raise FileExistsError("本次结果目录已存在；请稍后重试")
    summary = save_tables(table_dir, ctx, best, start, stats, checks, output_root)
    summary.update({
        "method": "zero-lateness variable-neighborhood energy search",
        "warm_start": origin,
        "starting_energy_kwh": start["score"][4],
        "energy_change_kwh": best["score"][4] - start["score"][4],
        "global_optimality_proven": False,
    })
    source_hash = sha256(data_run / "meta" / "validated_artifacts.json")
    comparison = compare_methods(output_root, source_hash, summary, best["sorties"])
    write_csv(table_dir / "Q2_方案对照.csv", comparison, list(comparison[0]))
    write_csv(table_dir / "Q2_搜索过程.csv", history, list(history[0]))
    (table_dir / "Q2_运行摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    (table_dir / "Q2_运行说明.txt").write_text(
        "方案4从同一份已验收数据的最低能耗零迟到方案热启动。\n"
        "逐箱挪动、两路线换箱、路线先后和区间先后均经过同一机/电池模拟器。\n"
        "官方两张Q2表及逐项验收表在本目录；图在同方法的figure目录。\n"
        "即使PASS，也只说明当前启发式方案可行，不是全局最优证明。\n",
        encoding="utf-8")
    print(f"tables: {table_dir}", flush=True)
    save_figures(figure_dir, ctx, best, summary)
    save_progress_figure(figure_dir, history)
    save_comparison_figures(figure_dir, comparison)
    print(f"figures: {figure_dir}", flush=True)
    print(f"Q2 method 4 {summary['status']}: energy={best['score'][4]:.6f} kWh, "
          f"sorties={best['score'][5]}, hard={best['score'][0]}, "
          f"soft={best['score'][2]:.1f}; global optimum not proven")


if __name__ == "__main__":
    main()
