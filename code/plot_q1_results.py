"""把第一问已验收的可行基线结果绘成可发送 PNG 与论文用 PDF。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / ".mplcache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, font_manager


CODE_DIR = Path(__file__).resolve().parent
PROJECT = CODE_DIR.parent
OUTPUTS = CODE_DIR / "outputs"
FIGURES = PROJECT / "figures"
TEMPLATE_COLUMNS = [
    "架次编号", "服务区编号", "机型编号", "货箱编号列表", "总质量（kg）",
    "总体积（m³）", "往返时间（s）", "架次能耗（kWh）", "返航SOC（%）",
]
SOURCE_FILES = ["Q1_安全载荷.csv", "Q1_单箱保底.csv", "Q1_单点组批.csv",
                "Q1_指标对照.csv", "Q1_验收检查.csv", "Q1_运行摘要.json", "Q1_READY.txt"]
GRAY = "#526174"
TEAL = "#007D82"
GRID = "#D8DEE5"


def latest_result_dir() -> Path:
    for path in sorted(OUTPUTS.glob("q1_baseline_*"), reverse=True):
        if path.is_dir() and (path / "Q1_READY.txt").is_file():
            return path
    raise FileNotFoundError("找不到有 Q1_READY.txt 的第一问结果；请先运行 q1_baseline.py")


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少图表数据：{path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def _box_ids(frame: pd.DataFrame) -> list[str]:
    return [box_id for cell in frame["货箱编号列表"].astype(str)
            for box_id in cell.split(";") if box_id]


def load_results(result_dir: Path) -> dict[str, object]:
    result_dir = result_dir.resolve()
    if not (result_dir / "Q1_READY.txt").is_file():
        raise ValueError("结果缺少 Q1_READY.txt，不应作图")
    payload = _read(result_dir / "Q1_安全载荷.csv")
    single = _read(result_dir / "Q1_单箱保底.csv")
    ffd = _read(result_dir / "Q1_单点组批.csv")
    metrics = _read(result_dir / "Q1_指标对照.csv")
    qa = _read(result_dir / "Q1_验收检查.csv")
    for name, frame in (("单箱保底", single), ("同区FFD", ffd)):
        if list(frame.columns) != TEMPLATE_COLUMNS:
            raise ValueError(f"{name} 的列名或顺序与第一问模板不一致")
    if qa.empty or not qa["status"].eq("PASS").all():
        raise ValueError("第一问验收表中存在失败项，不应作图")
    if set(metrics["baseline"]) != {"single_box", "ffd"} or len(metrics) != 2:
        raise ValueError("指标对照表应恰有两种基线")
    metric_by_name = metrics.set_index("baseline")
    for name, frame in (("single_box", single), ("ffd", ffd)):
        metric = metric_by_name.loc[name]
        if len(frame) != int(metric["sorties"]):
            raise ValueError(f"{name} 的架次行数与指标不一致")
        for column, expected in (("总质量（kg）", "total_mass_kg"),
                                 ("总体积（m³）", "total_volume_m3"),
                                 ("往返时间（s）", "cumulative_operation_s"),
                                 ("架次能耗（kWh）", "total_energy_kwh")):
            if not math.isclose(float(frame[column].sum()), float(metric[expected]),
                                rel_tol=1e-9, abs_tol=1e-5):
                raise ValueError(f"{name} 的 {column} 与指标对照表不一致")
        ids = _box_ids(frame)
        if len(ids) != 80 or len(set(ids)) != 80:
            raise ValueError(f"{name} 未将 80 箱恰好安排一次")
    if set(_box_ids(single)) != set(_box_ids(ffd)):
        raise ValueError("两个基线的货箱集合不相同")
    expected_pairs = {(kind, f"S{zone:03d}") for kind in "ABC" for zone in range(1, 16)}
    actual_pairs = set(zip(payload["type_id"], payload["zone_id"]))
    if len(payload) != 45 or actual_pairs != expected_pairs:
        raise ValueError("安全载荷表不包含完整的 3×15 组合")
    allowed = {"rated_payload", "energy_limited", "unreachable"}
    if not set(payload["status"]).issubset(allowed):
        raise ValueError("安全载荷状态存在未知值")
    reachable = payload["status"] != "unreachable"
    if (payload.loc[reachable, "q_max_kg"].isna().any() or
            (payload.loc[reachable, "q_max_kg"] < 0).any() or
            (payload.loc[reachable, "q_max_kg"] >
             payload.loc[reachable, "rated_payload_kg"] + 1e-8).any() or
            payload.loc[~reachable, "q_max_kg"].notna().any()):
        raise ValueError("安全载荷数值与状态不一致")
    summary = json.loads((result_dir / "Q1_运行摘要.json").read_text(encoding="utf-8"))
    if int(summary["checks_passed"]) != len(qa):
        raise ValueError("运行摘要与验收表不一致")
    return {"directory": result_dir, "payload": payload, "single": single,
            "ffd": ffd, "metrics": metric_by_name, "qa_count": len(qa),
            "summary": summary}


def configure_plot_style() -> str:
    candidates = [Path("C:/Windows/Fonts/simhei.ttf"),
                  Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    for font_path in candidates:
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            break
    else:
        try:
            font_name = font_manager.FontProperties(
                fname=font_manager.findfont("Noto Sans CJK SC", fallback_to_default=False)
            ).get_name()
        except ValueError as error:
            raise RuntimeError("找不到中文字体，无法安全输出中文图表") from error
    plt.rcParams.update({
        "font.family": font_name, "font.size": 10, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    return font_name


def _save_figure(fig: plt.Figure, destination: Path) -> None:
    fig.savefig(destination.with_suffix(".pdf"), dpi=300)
    fig.savefig(destination.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_comparison(data: dict[str, object], destination: Path) -> dict[str, float]:
    metrics = data["metrics"]
    single, ffd = metrics.loc["single_box"], metrics.loc["ffd"]
    definitions = [
        ("架次数", "架次", float(single["sorties"]), float(ffd["sorties"]), "{:.0f}"),
        ("总运输能耗", "kWh", float(single["total_energy_kwh"]),
         float(ffd["total_energy_kwh"]), "{:.1f}"),
        ("累计作业时间", "小时", float(single["cumulative_operation_s"]) / 3600,
         float(ffd["cumulative_operation_s"]) / 3600, "{:.2f}"),
    ]
    fig = plt.figure(figsize=(10.8, 4.2), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[14, 1.2])
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    note = fig.add_subplot(grid[1, :])
    note.axis("off")
    reductions: dict[str, float] = {}
    for ax, (label, unit, initial, batched, fmt) in zip(axes, definitions):
        reduction = 100 * (1 - batched / initial)
        reductions[label] = reduction
        bars = ax.bar([0, 1], [initial, batched], width=0.56,
                      color=[GRAY, TEAL], edgecolor="#283844", linewidth=0.8,
                      hatch=["///", ""])
        ax.set_xticks([0, 1], ["单箱保底", "同区FFD"])
        ax.set_ylabel(f"{label}（{unit}）")
        ax.set_ylim(0, max(initial, batched) * 1.32)
        ax.grid(axis="y", color=GRID, linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, (initial, batched)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() +
                    max(initial, batched) * 0.025, fmt.format(value),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(0.98, 0.97, f"减少 {reduction:.1f}%", transform=ax.transAxes,
                ha="right", va="top", color=TEAL, fontsize=9)
    note.text(0, 0.45,
              "注：两方案均为可行基线，非最优解；累计作业时间为各架次时间之和，不是任务完工时刻。",
              ha="left", va="center", fontsize=9, color="#364451")
    _save_figure(fig, destination)
    return reductions


def plot_safe_payload(data: dict[str, object], destination: Path) -> dict[str, object]:
    source = data["payload"].set_index(["type_id", "zone_id"])
    zones = [f"S{i:03d}" for i in range(1, 16)]
    values = np.full((3, 15), np.nan, dtype=float)
    labels: list[list[str]] = []
    limited: list[str] = []
    for row, kind in enumerate("ABC"):
        row_labels = []
        for column, zone in enumerate(zones):
            item = source.loc[(kind, zone)]
            if item["status"] == "unreachable":
                row_labels.append("不可达")
                continue
            qmax, rated = float(item["q_max_kg"]), float(item["rated_payload_kg"])
            values[row, column] = qmax / rated
            if item["status"] == "energy_limited":
                limited.append(f"{kind}-{zone}")
            shown = f"{qmax:.1f}" if not math.isclose(qmax, rated, abs_tol=1e-6) else f"{rated:.0f}"
            row_labels.append(shown + ("*" if item["status"] == "energy_limited" else ""))
        labels.append(row_labels)
    cmap = matplotlib.colormaps["YlGnBu"].copy()
    cmap.set_bad("#E6E9ED")
    fig = plt.figure(figsize=(12.2, 4.0), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[12, 1.2])
    ax = fig.add_subplot(grid[0])
    note = fig.add_subplot(grid[1])
    note.axis("off")
    picture = ax.imshow(np.ma.masked_invalid(values), cmap=cmap,
                        norm=colors.Normalize(vmin=0, vmax=1), aspect="auto")
    ax.set_xticks(range(15), zones)
    ax.set_yticks(range(3), list("ABC"))
    ax.set_xlabel("服务区")
    ax.set_ylabel("运输机型")
    ax.set_xticks(np.arange(-0.5, 15, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(3):
        for column in range(15):
            ratio = values[row, column]
            ax.text(column, row, labels[row][column], ha="center", va="center",
                    fontsize=8.7, fontweight="bold" if "*" in labels[row][column] else "normal",
                    color="white" if math.isfinite(ratio) and ratio > 0.66 else "#17242F")
    colorbar = fig.colorbar(picture, ax=ax, fraction=0.032, pad=0.018)
    colorbar.set_label("安全载荷 / 额定载荷")
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.set_ticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    note.text(0, 0.5, "单元格数字：最大安全载荷（kg）；* 表示受返航能量约束，未标星表示达到额定载荷上限。",
              ha="left", va="center", fontsize=9, color="#364451")
    _save_figure(fig, destination)
    return {"energy_limited_cells": limited, "rated_cells": 45 - len(limited) -
            int(np.isnan(values).sum()), "unreachable_cells": int(np.isnan(values).sum())}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_figures(result_dir: Path, output_dir: Path) -> dict[str, object]:
    data = load_results(result_dir)
    if output_dir.exists():
        raise FileExistsError(f"不覆盖已有图表目录：{output_dir}")
    font_name = configure_plot_style()
    output_dir.mkdir(parents=True)
    comparison = output_dir / "图1_两种可行基线对比"
    payload = output_dir / "图2_各区安全载荷边界"
    reductions = plot_comparison(data, comparison)
    payload_summary = plot_safe_payload(data, payload)
    artifact_names = [f"{stem.name}.{extension}" for stem in (comparison, payload)
                      for extension in ("pdf", "png")]
    for name in artifact_names:
        if (output_dir / name).stat().st_size < 10_000:
            raise ValueError(f"图表文件过小，可能导出失败：{name}")
    provenance = {
        "source_result_dir": str(result_dir.resolve()),
        "source_sha256": {name: _sha256(result_dir / name) for name in SOURCE_FILES},
        "source_qa_pass_count": data["qa_count"],
        "model_status": "feasible baseline, not optimized",
        "transformations": ["cumulative_operation_s / 3600 -> hours",
                            "percentage decrease = 100*(1-FFD/single_box)",
                            "heatmap ratio = q_max_kg / rated_payload_kg",
                            "no filtering or smoothing"],
        "comparison_reduction_pct": reductions,
        "payload_summary": payload_summary,
        "font": font_name,
        "matplotlib_version": matplotlib.__version__,
        "artifacts": artifact_names,
    }
    (output_dir / "图表数据来源.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "发给师兄的图表说明.txt").write_text(
        "第一问目前已有两张结果图。\n"
        "图1比较单箱保底与同区FFD组批在架次、能耗和累计作业时间上的差异；\n"
        "图2展示A/B/C三种机型在15个服务区的最大安全载荷，星号代表受返航能量约束。\n"
        "这些是满足当前建模假设的可行基线，不是第一问最终最优解。\n"
        "PNG可直接发微信；PDF为矢量图，可用于论文排版。\n",
        encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="D题第一问结果图")
    parser.add_argument("--result-dir", type=Path, help="已验收的 Q1 基线目录；默认最新一次")
    parser.add_argument("--output-dir", type=Path, help="新的出图目录；默认在 D题/figures 下自动命名")
    args = parser.parse_args()
    result_dir = args.result_dir.resolve() if args.result_dir else latest_result_dir()
    output_dir = args.output_dir.resolve() if args.output_dir else (
        FIGURES / datetime.now().strftime("q1_figures_%Y%m%d_%H%M%S_%f"))
    report = create_figures(result_dir, output_dir)
    print(f"Q1 figures PASS: {len(report['artifacts'])} PDF/PNG artifacts, "
          f"{report['source_qa_pass_count']} upstream checks")
    print(f"source: {result_dir}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
