"""从已验收的第三问方案三比较表生成两张中文论文图。

直接在 PyCharm 运行即可；也可用 --comparison-csv 指定另一轮方案三结果。
本脚本只读取已保存的结果，不重新求解或修改任何原始表格。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
SCHEME_NAMES = ("方案一", "方案二", "方案三-少架次", "方案三-及时性")
METRIC_FIELDS = ("weighted_lateness", "makespan_s", "energy_kwh", "总架次")
COLORS = {
    "方案一": "#8b95a1",
    "方案二": "#4b729c",
    "方案三-少架次": "#177e74",
    "方案三-及时性": "#db8737",
}


@dataclass(frozen=True)
class Scheme:
    name: str
    weighted_lateness: float
    makespan_s: float
    energy_kwh: float
    transport_sorties: int
    relay_sorties: int
    total_sorties: int

    def metric(self, field: str) -> float:
        return float(getattr(self, "total_sorties" if field == "总架次" else field))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_comparison() -> Path:
    candidates = sorted((CODE_DIR / "3_outputs" / "3_optimize").glob(
        "q3_opt3_*_sorties_table/Q3_方案比较.csv"
    ))
    if len(candidates) != 1:
        raise ValueError(
            f"找到 {len(candidates)} 份方案三比较表；请用 --comparison-csv 指定本次要画的文件。"
        )
    return candidates[0]


def _read_schemes(path: Path) -> dict[str, Scheme]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "方案", "weighted_lateness", "makespan_s", "energy_kwh",
            "transport_sorties", "relay_sorties", "总架次",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"比较表缺列：{sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    if len(rows) != len(SCHEME_NAMES) or {row["方案"] for row in rows} != set(SCHEME_NAMES):
        raise ValueError("比较表应恰好包含方案一、方案二及方案三两种代表解各一行。")

    schemes: dict[str, Scheme] = {}
    for row in rows:
        name = row["方案"]
        values = [float(row[key]) for key in METRIC_FIELDS[:3]]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError(f"{name} 有非正数或非有限的目标值。")
        transport = int(row["transport_sorties"])
        relay = int(row["relay_sorties"])
        total = int(row["总架次"])
        if min(transport, relay) < 0 or transport + relay != total:
            raise ValueError(f"{name} 的运输、中继与总架次不一致。")
        schemes[name] = Scheme(name, *values, transport, relay, total)
    return schemes


def _close(actual: float, expected: Any) -> bool:
    return math.isclose(actual, float(expected), rel_tol=1e-8, abs_tol=1e-6)


def _validate_record(comparison_path: Path, schemes: dict[str, Scheme]) -> dict[str, Any]:
    table_name = comparison_path.parent.name
    if not table_name.endswith("_sorties_table"):
        raise ValueError("请使用方案三的 _sorties_table 比较表，以定位同轮验收文件。")
    run_id = table_name.removesuffix("_sorties_table")
    result_root = comparison_path.parent.parent
    index_path = result_root / f"{run_id}_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("status") != "PASS":
        raise ValueError("方案三索引没有 PASS，不生成论文图。")
    representatives = {item.get("goal"): item for item in index.get("representatives", [])}
    if set(representatives) != {"sorties", "lateness"}:
        raise ValueError("方案三索引没有两种预期的代表解。")

    baseline = schemes["方案二"]
    for field in METRIC_FIELDS[:3]:
        if not _close(baseline.metric(field), representatives["sorties"]["reference_metrics"][field]):
            raise ValueError(f"方案二 {field} 与索引参考值不符。")
    for field in ("transport_sorties", "relay_sorties"):
        if getattr(baseline, field) != int(representatives["sorties"]["reference_metrics"][field]):
            raise ValueError(f"方案二 {field} 与索引参考值不符。")

    checked: dict[str, str] = {}
    for goal, name in (("sorties", "方案三-少架次"), ("lateness", "方案三-及时性")):
        record = representatives[goal]
        table_dir = result_root / f"{run_id}_{goal}_table"
        if record.get("status") != "PASS" or not (table_dir / "Q3_READY.txt").exists():
            raise ValueError(f"{name} 缺少 PASS 或 READY 标记。")
        with (table_dir / "Q3_验收检查.csv").open("r", encoding="utf-8-sig", newline="") as stream:
            checks = list(csv.DictReader(stream))
        if not checks or any(row.get("status") != "PASS" for row in checks):
            raise ValueError(f"{name} 的验收检查并非全部 PASS。")
        final = record["final_metrics"]
        scheme = schemes[name]
        for field in METRIC_FIELDS[:3]:
            if not _close(scheme.metric(field), final[field]):
                raise ValueError(f"{name} 的 {field} 与索引不符。")
        for field in ("transport_sorties", "relay_sorties"):
            if getattr(scheme, field) != int(final[field]):
                raise ValueError(f"{name} 的 {field} 与索引不符。")
        checked[goal] = ",".join(row["check_id"] for row in checks)

    other_comparison = result_root / f"{run_id}_lateness_table" / "Q3_方案比较.csv"
    if _sha256(other_comparison) != _sha256(comparison_path):
        raise ValueError("同轮两份方案比较表不一致，停止绘图。")

    recorded = representatives["sorties"].get("code_sha256", {}).get("q3_1_optimize.py")
    current_source = CODE_DIR / "q3_1_optimize.py"
    current = _sha256(current_source) if current_source.exists() else None
    return {
        "run_id": run_id,
        "comparison_csv_sha256": _sha256(comparison_path),
        "index_sha256": _sha256(index_path),
        "representative_checks": checked,
        "q3_1_recorded_sha256": recorded,
        "q3_1_current_sha256": current,
        "q3_1_matches_recorded": recorded == current,
        "global_optimality_proven": bool(index.get("global_optimality_proven", False)),
    }


def _configure_matplotlib() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "huawei_q3_paper_mplconfig"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    font_names = {entry.name for entry in font_manager.fontManager.ttflist}
    font = next(
        (name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if name in font_names),
        None,
    )
    if font is None:
        raise RuntimeError("缺少中文字体；请安装微软雅黑、黑体或 Noto Sans CJK SC。")
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })


def _save_figure(fig: Any, stem: Path, qa_scripts: Path | None) -> None:
    if qa_scripts is not None:
        sys.path.insert(0, str(qa_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(
            fig,
            json_out=f"{stem}.alignment.json",
            overlay_svg=f"{stem}.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            strict=True,
        )
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")


def _make_metric_figure(schemes: dict[str, Scheme], output: Path, qa_scripts: Path | None) -> None:
    import matplotlib.pyplot as plt

    baseline = schemes["方案二"]
    metrics = (
        ("weighted_lateness", "加权迟到"),
        ("makespan_s", "最晚返航"),
        ("energy_kwh", "运输与中继能耗"),
        ("总架次", "总架次"),
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.7), layout="constrained")
    y_positions = list(reversed(range(len(metrics))))
    for name, offset in (("方案三-少架次", 0.16), ("方案三-及时性", -0.16)):
        scheme = schemes[name]
        ratios = [100 * scheme.metric(field) / baseline.metric(field) for field, _ in metrics]
        ax.barh(
            [y + offset for y in y_positions], ratios, height=0.25,
            color=COLORS[name], label=name.replace("-", "·"), zorder=2,
        )
        for y, ratio in zip(y_positions, ratios):
            ax.text(ratio - 1.4, y + offset, f"{ratio:.1f}%", va="center",
                    ha="right", color="white", fontsize=8, zorder=4)
    ax.axvline(100, color="#66717d", linewidth=1, linestyle="--", zorder=1)
    ax.set_yticks(y_positions, [label for _, label in metrics])
    ax.set_xlim(0, 117)
    ax.set_xlabel("相对方案二的比例（%，越低越好）")
    ax.set_xticks(range(0, 101, 20))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
    _save_figure(fig, output / "Q3_四指标相对方案二对比", qa_scripts)
    plt.close(fig)


def _make_tradeoff_figure(schemes: dict[str, Scheme], output: Path, qa_scripts: Path | None) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.3), layout="constrained")
    offsets = {
        "方案一": (-8, -15),
        "方案二": (8, 12),
        "方案三-少架次": (8, 12),
        "方案三-及时性": (8, -20),
    }
    markers = {"方案一": "o", "方案二": "o", "方案三-少架次": "s", "方案三-及时性": "D"}
    for name in SCHEME_NAMES:
        scheme = schemes[name]
        x, y = scheme.makespan_s / 3600, scheme.energy_kwh
        ax.scatter(x, y, s=100, marker=markers[name], c=COLORS[name],
                   edgecolors="#25313d", linewidths=0.6, zorder=3)
        label = f"{name.replace('-', '·')}（{scheme.transport_sorties}+{scheme.relay_sorties} 架次）"
        dx, dy = offsets[name]
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="right" if name == "方案一" else "left", va="center", fontsize=8)
    ax.set_xlim(3.30, 6.70)
    ax.set_ylim(72.5, 101.5)
    ax.set_xlabel("全部架次最晚返航时间（h）")
    ax.set_ylabel("运输与中继总能耗（kWh）")
    _save_figure(fig, output / "Q3_返航能耗架次权衡_中文版", qa_scripts)
    plt.close(fig)


def _write_plot_data(path: Path, schemes: dict[str, Scheme]) -> None:
    baseline = schemes["方案二"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("方案", "加权迟到_s", "最晚返航_s", "总能耗_kWh",
                         "运输架次", "中继架次", "总架次", "迟到_相对方案二_pct",
                         "返航_相对方案二_pct", "能耗_相对方案二_pct", "架次_相对方案二_pct"))
        for name in SCHEME_NAMES:
            scheme = schemes[name]
            writer.writerow((name, scheme.weighted_lateness, scheme.makespan_s,
                             scheme.energy_kwh, scheme.transport_sorties,
                             scheme.relay_sorties, scheme.total_sorties,
                             *((100.0 if name == "方案二" else
                                100 * scheme.metric(field) / baseline.metric(field))
                               for field in METRIC_FIELDS)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", type=Path, help="方案三的 Q3_方案比较.csv")
    parser.add_argument("--output-dir", type=Path, help="论文图输出文件夹")
    parser.add_argument("--qa-scripts", type=Path, help="可选：绘图 QA 脚本目录")
    args = parser.parse_args()
    comparison = (args.comparison_csv or _find_comparison()).resolve()
    schemes = _read_schemes(comparison)
    record = _validate_record(comparison, schemes)
    output = (args.output_dir or comparison.parent.parent /
              f"{record['run_id']}_paper_figure").resolve()
    output.mkdir(parents=True, exist_ok=True)

    _configure_matplotlib()
    _make_metric_figure(schemes, output, args.qa_scripts)
    _make_tradeoff_figure(schemes, output, args.qa_scripts)
    _write_plot_data(output / "Q3_论文图数据.csv", schemes)
    try:
        source_label = str(comparison.relative_to(CODE_DIR))
    except ValueError:
        source_label = comparison.name
    manifest = {
        "status": "FIGURES_FROM_RECORDED_PASS_RESULTS",
        "claim": "方案三两种代表解相对方案二在四项指标上的变化及内部权衡",
        "normalization": "方案二逐指标=100%；只为展示，不表示方案三的搜索权重",
        "source_csv": source_label,
        "source_record": record,
        "figure_archetype": "quantitative grid (two separate single-panel figures)",
        "backend": "Python/matplotlib",
        "exclusion": "四指标图不画方案一，因其问题是方案二与方案三两代表解的直接比较；散点图保留全部四行。",
        "statistics": "确定性求解结果；非重复样本，无误差条或显著性检验。",
        "note": "图可由保存的 CSV 复现；求解器源码指纹不一致时，不可声称完整求解已复现。",
    }
    (output / "Q3_论文图来源.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"图已保存：{output}")
    if not record["q3_1_matches_recorded"]:
        print("注意：当前 q3_1_optimize.py 与结果记录中的源码指纹不一致；仅图表由存档数据复现。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
