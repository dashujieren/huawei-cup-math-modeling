"""从已验收的第一问优化结果绘制能耗-时间权衡图。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from plot_q1_results import CODE_DIR, FIGURES, configure_plot_style

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def latest_optimized() -> Path:
    for directory in sorted((CODE_DIR / "outputs").glob("q1_optimized_*"), reverse=True):
        if directory.is_dir() and (directory / "Q1_OPT_READY.txt").is_file():
            return directory
    raise FileNotFoundError("找不到已验收的第一问优化结果，请先运行 q1_optimize.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    (output_dir / "发给师兄的图表说明.txt").write_text(
        "第一问精确组批：18架次已达到理论最少架次。\n"
        "同为18架，推荐方案在所列方案中同时改善能耗和累计作业时间。\n"
        "19架次可以再降低少量能耗，但累计作业时间增加；灰叉方案被其他方案支配。\n"
        "坐标轴是为观察权衡而截取的局部区间，比较时以附带CSV的准确数值为准。\n"
        "PNG可直接发消息，PDF可用于后续论文排版。\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制第一问精确组批的能耗-时间权衡图")
    parser.add_argument("--result-dir", type=Path, help="已验收优化结果目录，默认最新")
    parser.add_argument("--output-dir", type=Path, help="新的图表目录，默认在 figures 下自动命名")
    args = parser.parse_args()
    result_dir = args.result_dir.resolve() if args.result_dir else latest_optimized()
    output_dir = args.output_dir.resolve() if args.output_dir else (
        FIGURES / datetime.now().strftime("q1_optimization_%Y%m%d_%H%M%S_%f"))
    make_figure(result_dir, output_dir)
    print(f"Q1 optimization figure PASS; source={result_dir}; output={output_dir}")


if __name__ == "__main__":
    main()
