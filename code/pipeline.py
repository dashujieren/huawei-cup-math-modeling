"""D 题原始附件的只读抽取、标准化、空间派生与独立硬验收。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import tifffile
from pyproj import Transformer


DATA_DIR = Path("数据/无人机应急物资运输基础数据")
DEM_DIR = Path("数据/镇龙乡地理空间数据/镇龙乡及周边地理数据/数字高程模型数据（DEM）")

# Excel 行号是原始附件中的 1-based 行号；不使用自动识别表头。
SEGMENTS = {
    "center": ("调度中心与服务区.xlsx", "数据", 2, 1, "A:E"),
    "zones": ("调度中心与服务区.xlsx", "数据", 6, 15, "A:F"),
    "demand_summary": ("物资需求与配送时限.xlsx", "数据", 1, 53, "A:I"),
    "boxes": ("物资需求与配送时限.xlsx", "逐箱货箱清单", 1, 80, "A:I"),
    "transport_types": ("运输无人机数据.xlsx", "数据", 2, 3, "A:R"),
    "transport_uavs": ("运输无人机数据.xlsx", "数据", 8, 8, "A:C"),
    "battery_pool": ("运输无人机数据.xlsx", "数据", 19, 3, "A:C"),
    "relay_type": ("中继无人机数据.xlsx", "数据", 2, 1, "A:S"),
    "relay_uavs": ("中继无人机数据.xlsx", "数据", 6, 2, "A:C"),
    "relay_energy_pool": ("中继无人机数据.xlsx", "数据", 11, 1, "A:C"),
    "comm_params": ("通信链路参数.xlsx", "数据", 2, 14, "A:B,D:E"),
}
FIELD_ORIGINS: dict[str, dict[str, str]] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def as_number(series: pd.Series, label: str, nullable: bool = False) -> pd.Series:
    raw = clean_text(series).replace("", np.nan)
    try:
        result = pd.to_numeric(raw, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} 存在无法转换的数值：{exc}") from exc
    if not nullable and result.isna().any():
        raise ValueError(f"{label} 存在空值，原始行：{series.index[result.isna()].tolist()}")
    return result.astype("float64")


def as_bool(series: pd.Series, label: str) -> pd.Series:
    values = clean_text(series)
    bad = ~values.isin(["是", "否"])
    if bad.any():
        raise ValueError(f"{label} 只能为“是/否”：{values[bad].tolist()}")
    return values.eq("是")


def pick(raw: pd.DataFrame, mapping: dict[str, str], source_key: str) -> pd.DataFrame:
    missing = set(mapping) - set(raw.columns)
    if missing:
        raise ValueError(f"表段缺少字段：{sorted(missing)}")
    FIELD_ORIGINS[source_key] = {normalized: original for original, normalized in mapping.items()}
    out = raw[list(mapping)].rename(columns=mapping).copy()
    out["source_file"] = clean_text(raw["source_file"])
    out["source_sheet"] = clean_text(raw["source_sheet"])
    out["source_row"] = as_number(raw["source_row"], "source_row").astype(int)
    return out


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", float_format="%.15g")


def revoke_ready(run_dir: Path) -> None:
    ready = run_dir / "meta" / "READY.txt"
    if ready.exists():
        ready.unlink()


def read_stage(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / "staging" / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少抽取表段：{path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_clean(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / "clean" / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少标准表：{path}")
    return pd.read_csv(path, dtype={
        "node_id": str, "box_id": str, "zone_id": str, "type_id": str,
        "uav_id": str, "battery_id": str, "unit_id": str,
    })


def extract(project: Path, run_dir: Path) -> None:
    revoke_ready(run_dir)
    if (run_dir / "staging").exists():
        raise FileExistsError(f"抽取目录已存在，不覆盖：{run_dir / 'staging'}")
    (run_dir / "staging").mkdir(parents=True)
    manifest: dict[str, object] = {"source_files": {}, "segments": {}}
    source_files = sorted({spec[0] for spec in SEGMENTS.values()})
    for filename in source_files:
        path = project / DATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest["source_files"][str(path.relative_to(project)).replace("\\", "/")] = sha256(path)
    mat = next((project / DEM_DIR).glob("*.mat"))
    tif = next((project / DEM_DIR).glob("*.tif"))
    for path in (mat, tif):
        manifest["source_files"][str(path.relative_to(project)).replace("\\", "/")] = sha256(path)
    for name, (filename, sheet, header_row, nrows, usecols) in SEGMENTS.items():
        path = project / DATA_DIR / filename
        df = pd.read_excel(path, sheet_name=sheet, header=header_row - 1,
                           nrows=nrows, usecols=usecols, engine="openpyxl")
        if len(df) != nrows:
            raise ValueError(f"{filename}/{sheet} 的 {name} 应有 {nrows} 行，实际 {len(df)} 行")
        df["source_file"] = filename
        df["source_sheet"] = sheet
        df["source_row"] = np.arange(header_row + 1, header_row + 1 + len(df))
        write_csv(run_dir / "staging" / f"{name}.csv", df)
        manifest["segments"][name] = {
            "source_file": filename, "sheet": sheet, "header_row": header_row,
            "first_data_row": header_row + 1, "last_data_row": header_row + nrows,
            "rows": len(df), "columns": [str(c) for c in df.columns if not str(c).startswith("Unnamed:")],
        }
    dump_json(run_dir / "meta" / "source_manifest.json", manifest)
    print(f"01 抽取完成：{len(SEGMENTS)} 个表段，原始附件未修改")


def _numeric_columns(df: pd.DataFrame, cols: list[str], label: str,
                     nullable: set[str] | None = None) -> None:
    nullable = nullable or set()
    for col in cols:
        df[col] = as_number(df[col], f"{label}.{col}", col in nullable)


def require_integer(series: pd.Series, label: str, minimum: int = 0) -> None:
    values = series.to_numpy(dtype=float)
    bad = (~np.isfinite(values) | (values < minimum) |
           ~np.isclose(values, np.rint(values), rtol=0, atol=1e-9))
    if np.any(bad):
        raise ValueError(f"{label} 必须是大于等于 {minimum} 的整数，异常值：{values[bad].tolist()}")


def field_unit(name: str) -> str:
    for suffix, unit in (("_mps", "m/s"), ("_m3", "m³"), ("_kwh", "kWh"),
                         ("_kw", "kW"), ("_kg", "kg"), ("_deg", "degree"),
                         ("_pct", "%"), ("_s", "s"), ("_m", "m")):
        if name.endswith(suffix):
            return unit
    if name in {"population", "stock_qty", "total_boxes", "first_boxes", "pixels_traversed"}:
        return "count"
    if name in {"initial_soc", "return_soc_min", "priority"}:
        return "dimensionless"
    if name == "value":
        return "see unit column"
    return ""


def add_field_schema(schema: dict[str, object], run_dir: Path) -> None:
    node_origins = {}
    for label in ("center", "zones"):
        for normalized, original in FIELD_ORIGINS[label].items():
            node_origins.setdefault(normalized, [])
            if original not in node_origins[normalized]:
                node_origins[normalized].append(original)
    int_fields = {"source_row", "population", "stock_qty", "total_boxes", "first_boxes"}
    nullable = {("nodes", "population"), ("boxes", "first_deadline_s"),
                ("boxes", "hard_deadline_s"), ("demand_summary", "first_deadline_s")}
    bool_fields = {"is_first_batch", "medical_bool", "valid_flag"}
    for name in ("nodes", "boxes", "demand_summary", "transport_types", "transport_uavs",
                 "battery_pool", "battery_units", "relay_type", "relay_uavs",
                 "relay_energy_pool", "relay_energy_units", "comm_params"):
        frame = read_clean(run_dir, name)
        origins = node_origins if name == "nodes" else FIELD_ORIGINS.get(name, {})
        if name == "battery_units":
            origins = {"type_id": FIELD_ORIGINS["battery_pool"]["type_id"]}
        if name == "relay_energy_units":
            origins = {"type_id": FIELD_ORIGINS["relay_energy_pool"]["type_id"]}
        fields = {}
        for col in frame.columns:
            if col in bool_fields:
                dtype = "boolean"
            elif col in int_fields:
                dtype = "integer"
            elif pd.api.types.is_numeric_dtype(frame[col]):
                dtype = "number"
            else:
                dtype = "string"
            source_col = origins.get(col)
            fields[col] = {
                "type": dtype, "unit": field_unit(col), "nullable": (name, col) in nullable,
                "source_column": source_col if source_col else
                ("Excel provenance" if col.startswith("source_") else "derived"),
            }
        schema["files"][f"{name}.csv"]["fields"] = fields
    arc_types = {"from_id": "string", "to_id": "string", "valid_flag": "boolean",
                 "invalid_reason": "string", "pixels_traversed": "integer"}
    arc_columns = ("from_id", "to_id", "distance_m", "start_alt_m", "end_alt_m",
                   "max_dsm_m", "cruise_alt_m", "climb_m", "descent_m",
                   "pixels_traversed", "valid_flag", "invalid_reason")
    schema["files"]["arc_geometry.csv"]["fields"] = {
        col: {"type": arc_types.get(col, "number"), "unit": field_unit(col),
              "nullable": col in {"max_dsm_m", "cruise_alt_m", "climb_m", "descent_m", "invalid_reason"},
              "source_column": "derived from nodes.csv and official DEM.mat"}
        for col in arc_columns
    }


def normalize(run_dir: Path) -> None:
    revoke_ready(run_dir)
    if (run_dir / "clean").exists():
        raise FileExistsError(f"标准表目录已存在，不覆盖：{run_dir / 'clean'}")
    clean = run_dir / "clean"
    clean.mkdir(parents=True)

    center = pick(read_stage(run_dir, "center"), {
        "调度中心编号": "node_id", "调度中心名称": "name", "经度（°）": "lon_deg",
        "纬度（°）": "lat_deg", "海拔（m）": "ground_elev_m",
    }, "center")
    center["kind"] = "center"
    center["population"] = np.nan
    zones = pick(read_stage(run_dir, "zones"), {
        "服务区编号": "node_id", "服务区名称": "name", "经度（°）": "lon_deg",
        "纬度（°）": "lat_deg", "海拔（m）": "ground_elev_m",
        "本次需保障人口（人）": "population",
    }, "zones")
    zones["kind"] = "zone"
    nodes = pd.concat([center, zones], ignore_index=True)
    nodes["node_id"] = clean_text(nodes["node_id"])
    nodes["name"] = clean_text(nodes["name"])
    _numeric_columns(nodes, ["lon_deg", "lat_deg", "ground_elev_m"], "nodes")
    nodes["population"] = as_number(nodes["population"], "nodes.population", nullable=True)
    require_integer(nodes.loc[nodes["kind"].eq("zone"), "population"], "zones.population", 1)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    nodes["x_m"], nodes["y_m"] = transformer.transform(nodes["lon_deg"].to_numpy(), nodes["lat_deg"].to_numpy())
    nodes["operation_alt_m"] = nodes["ground_elev_m"] + np.where(nodes["kind"].eq("zone"), 30, 0)
    nodes = nodes[["node_id", "kind", "name", "lon_deg", "lat_deg", "ground_elev_m",
                   "x_m", "y_m", "operation_alt_m", "population",
                   "source_file", "source_sheet", "source_row"]]
    write_csv(clean / "nodes.csv", nodes)

    boxes = pick(read_stage(run_dir, "boxes"), {
        "货箱编号": "box_id", "服务区编号": "zone_id", "物资类型": "item_type",
        "单箱质量（kg）": "mass_kg", "单箱体积（m³）": "volume_m3",
        "是否首批保障": "is_first_batch", "首批截止时间（s）": "first_deadline_s",
        "期望送达时间（s）": "expected_s", "应急优先系数": "priority",
    }, "boxes")
    for col in ("box_id", "zone_id", "item_type"):
        boxes[col] = clean_text(boxes[col])
    boxes["is_first_batch"] = as_bool(boxes["is_first_batch"], "boxes.is_first_batch")
    _numeric_columns(boxes, ["mass_kg", "volume_m3", "first_deadline_s", "expected_s", "priority"],
                     "boxes", {"first_deadline_s"})
    boxes["medical_bool"] = boxes["item_type"].eq("医疗物资")
    medical_deadline = boxes["expected_s"].where(boxes["medical_bool"])
    first_deadline = boxes["first_deadline_s"].where(boxes["is_first_batch"])
    boxes["hard_deadline_s"] = pd.concat([medical_deadline, first_deadline], axis=1).min(axis=1)
    write_csv(clean / "boxes.csv", boxes)

    summary = pick(read_stage(run_dir, "demand_summary"), {
        "服务区编号": "zone_id", "物资类型": "item_type", "总需求箱数": "total_boxes",
        "首批必须送达箱数": "first_boxes", "单箱质量（kg）": "unit_mass_kg",
        "单箱体积（m³）": "unit_volume_m3", "应急优先系数": "priority",
        "首批截止时间（s）": "first_deadline_s", "期望送达时间（s）": "expected_s",
    }, "demand_summary")
    summary["zone_id"] = clean_text(summary["zone_id"])
    summary["item_type"] = clean_text(summary["item_type"])
    _numeric_columns(summary, ["total_boxes", "first_boxes", "unit_mass_kg", "unit_volume_m3",
                               "priority", "first_deadline_s", "expected_s"],
                     "demand_summary", {"first_deadline_s"})
    require_integer(summary["total_boxes"], "demand_summary.total_boxes", 1)
    require_integer(summary["first_boxes"], "demand_summary.first_boxes")
    write_csv(clean / "demand_summary.csv", summary)

    transport_types = pick(read_stage(run_dir, "transport_types"), {
        "机型编号": "type_id", "机型名称": "name", "含电池空载总质量（kg）": "empty_mass_kg",
        "最大载货质量（kg）": "max_payload_kg", "可用装载体积（m³）": "capacity_m3",
        "计划巡航速度（m/s）": "cruise_speed_mps", "空载标准航程（m）": "range_empty_m",
        "满载标准航程（m）": "range_full_m", "电池可用能量（kWh）": "energy_use_kwh",
        "返航电量下限（%）": "return_soc_min_pct", "工位固定准备时间（s）": "prep_s",
        "每箱装载时间（s）": "load_per_box_s", "接收点基础交接时间（s）": "handoff_base_s",
        "每箱增加交接时间（s）": "handoff_per_box_s", "最大爬升速度（m/s）": "climb_speed_mps",
        "最大下降速度（m/s）": "descent_speed_mps", "爬升能耗效率": "climb_efficiency",
        "下降能耗效率": "descent_efficiency",
    }, "transport_types")
    transport_types["type_id"] = clean_text(transport_types["type_id"])
    transport_types["name"] = clean_text(transport_types["name"])
    _numeric_columns(transport_types, [c for c in transport_types if c not in {"type_id", "name", "source_file", "source_sheet", "source_row"}], "transport_types")
    transport_types["return_soc_min"] = transport_types["return_soc_min_pct"] / 100
    write_csv(clean / "transport_types.csv", transport_types)

    transport_uavs = pick(read_stage(run_dir, "transport_uavs"), {
        "无人机编号": "uav_id", "机型编号": "type_id", "初始位置": "initial_site",
    }, "transport_uavs")
    for col in ("uav_id", "type_id", "initial_site"):
        transport_uavs[col] = clean_text(transport_uavs[col])
    write_csv(clean / "transport_uavs.csv", transport_uavs)

    battery_pool = pick(read_stage(run_dir, "battery_pool"), {
        "机型编号": "type_id", "共享电池组总数（组）": "stock_qty",
        "等效完全充电时间（s）": "full_charge_s",
    }, "battery_pool")
    battery_pool["type_id"] = clean_text(battery_pool["type_id"])
    _numeric_columns(battery_pool, ["stock_qty", "full_charge_s"], "battery_pool")
    require_integer(battery_pool["stock_qty"], "battery_pool.stock_qty", 1)
    write_csv(clean / "battery_pool.csv", battery_pool)
    battery_units = pd.DataFrame([
        {"battery_id": f"{row.type_id}-B{i:02d}", "type_id": row.type_id, "initial_soc": 1.0,
         "source_file": row.source_file, "source_sheet": row.source_sheet, "source_row": row.source_row}
        for row in battery_pool.itertuples() for i in range(1, int(row.stock_qty) + 1)
    ])
    write_csv(clean / "battery_units.csv", battery_units)

    relay_type = pick(read_stage(run_dir, "relay_type"), {
        "机型编号": "type_id", "机型名称": "name",
        "含能源组件空载总质量（kg）": "empty_mass_kg", "中继通信模块质量（kg）": "module_mass_kg",
        "计划起飞总质量（kg）": "takeoff_mass_kg", "计划巡航速度（m/s）": "cruise_speed_mps",
        "巡航功率（kW）": "cruise_power_kw", "能源组件可用能量（kWh）": "energy_use_kwh",
        "返航电量下限（%）": "return_soc_min_pct", "工位固定准备时间（s）": "prep_s",
        "建链时间（s）": "link_setup_s", "架次周转时间（s）": "turnaround_s",
        "最大爬升速度（m/s）": "climb_speed_mps", "最大下降速度（m/s）": "descent_speed_mps",
        "爬升能耗效率": "climb_efficiency", "下降能耗效率": "descent_efficiency",
        "悬停功率（kW）": "hover_power_kw", "通信附加功率（kW）": "comm_extra_power_kw",
        "最大悬停离地高度（m）": "max_hover_agl_m",
    }, "relay_type")
    relay_type["type_id"] = clean_text(relay_type["type_id"])
    relay_type["name"] = clean_text(relay_type["name"])
    _numeric_columns(relay_type, [c for c in relay_type if c not in {"type_id", "name", "source_file", "source_sheet", "source_row"}], "relay_type")
    relay_type["return_soc_min"] = relay_type["return_soc_min_pct"] / 100
    write_csv(clean / "relay_type.csv", relay_type)

    relay_uavs = pick(read_stage(run_dir, "relay_uavs"), {
        "中继无人机编号": "uav_id", "机型编号": "type_id", "初始位置": "initial_site",
    }, "relay_uavs")
    for col in ("uav_id", "type_id", "initial_site"):
        relay_uavs[col] = clean_text(relay_uavs[col])
    write_csv(clean / "relay_uavs.csv", relay_uavs)

    relay_energy_pool = pick(read_stage(run_dir, "relay_energy_pool"), {
        "机型编号": "type_id", "共享能源组件总数（组）": "stock_qty",
        "等效完全充电时间（s）": "full_charge_s",
    }, "relay_energy_pool")
    relay_energy_pool["type_id"] = clean_text(relay_energy_pool["type_id"])
    _numeric_columns(relay_energy_pool, ["stock_qty", "full_charge_s"], "relay_energy_pool")
    require_integer(relay_energy_pool["stock_qty"], "relay_energy_pool.stock_qty", 1)
    write_csv(clean / "relay_energy_pool.csv", relay_energy_pool)
    relay_energy_units = pd.DataFrame([
        {"unit_id": f"{row.type_id}-E{i:02d}", "type_id": row.type_id, "initial_soc": 1.0,
         "source_file": row.source_file, "source_sheet": row.source_sheet, "source_row": row.source_row}
        for row in relay_energy_pool.itertuples() for i in range(1, int(row.stock_qty) + 1)
    ])
    write_csv(clean / "relay_energy_units.csv", relay_energy_units)

    comm = pick(read_stage(run_dir, "comm_params"), {
        "参数类别": "category", "参数名称": "parameter_name", "符号": "symbol", "参数值": "value",
    }, "comm_params")
    for col in ("category", "parameter_name", "symbol"):
        comm[col] = clean_text(comm[col])
    comm["value"] = as_number(comm["value"], "comm_params.value")
    comm["unit"] = comm["parameter_name"].map(lambda s: re.search(r"（([^）]+)）", s).group(1) if re.search(r"（([^）]+)）", s) else "")
    write_csv(clean / "comm_params.csv", comm)
    gateway_height = comm.loc[(comm["category"] == "固定网关 G01") &
                              (comm["symbol"] == "hG"), "value"]
    if len(gateway_height) != 1:
        raise ValueError("固定网关 G01 的天线离地高度应恰有一条")
    center_row = nodes.loc[nodes["node_id"] == "O01"].iloc[0]
    gateway = {
        "gateway_id": "G01", "base_node_id": "O01",
        "lon_deg": float(center_row.lon_deg), "lat_deg": float(center_row.lat_deg),
        "x_m": float(center_row.x_m), "y_m": float(center_row.y_m),
        "ground_elev_m": float(center_row.ground_elev_m),
        "antenna_agl_m": float(gateway_height.iloc[0]),
        "antenna_abs_m": float(center_row.ground_elev_m + gateway_height.iloc[0]),
        "source": "O01 coordinates/elevation plus communication parameter hG",
    }
    dump_json(run_dir / "meta" / "gateway.json", gateway)

    schema = {
        "format": "UTF-8 CSV; empty field means not applicable, not zero",
        "coordinate_systems": {"lon_deg/lat_deg": "EPSG:4326", "x_m/y_m": "EPSG:32649"},
        "time_origin": "all deadline and schedule values are seconds after common t=0",
        "soc": "0-1 internally; *_pct fields preserve original percent values",
        "source": "official D-problem attachments; source_row is original Excel row",
        "files": {
            "nodes.csv": {"key": "node_id", "nullable": ["population (O01 only)"], "units": "lon/lat degree; x/y/elevation m"},
            "boxes.csv": {"key": "box_id", "foreign_key": "zone_id -> nodes.node_id", "nullable": ["first_deadline_s", "hard_deadline_s"], "units": "mass kg; volume m3; time s"},
            "demand_summary.csv": {"key": ["zone_id", "item_type"], "nullable": ["first_deadline_s"], "units": "mass kg; volume m3; time s"},
            "transport_types.csv": {"key": "type_id", "units": "mass kg; volume m3; speed m/s; range m; energy kWh; time s"},
            "transport_uavs.csv": {"key": "uav_id", "foreign_key": "type_id -> transport_types.type_id"},
            "battery_pool.csv": {"key": "type_id", "units": "stock count; charge time s"},
            "battery_units.csv": {"key": "battery_id", "foreign_key": "type_id -> battery_pool.type_id"},
            "relay_type.csv": {"key": "type_id", "units": "mass kg; speed m/s; power kW; energy kWh; time s"},
            "relay_uavs.csv": {"key": "uav_id", "foreign_key": "type_id -> relay_type.type_id"},
            "relay_energy_pool.csv": {"key": "type_id", "units": "stock count; charge time s"},
            "relay_energy_units.csv": {"key": "unit_id", "foreign_key": "type_id -> relay_energy_pool.type_id"},
            "comm_params.csv": {"key": ["category", "parameter_name"], "units": "per-row unit field"},
            "arc_geometry.csv": {"key": ["from_id", "to_id"], "units": "distance/elevation m", "note": "no fixed-load energy"},
        },
    }
    add_field_schema(schema, run_dir)
    schema["geometry_convention"] = (
        "Raster crossing follows a straight line in source lon/lat; distance_m is the "
        "Euclidean distance between UTM EPSG:32649 projected endpoints."
    )
    schema["gateway_config"] = "meta/gateway.json: G01 is fixed at O01 with hG antenna height"
    dump_json(run_dir / "meta" / "schema.json", schema)
    print("02 标准化完成：12 张 clean 表，保留原始行号与合法空值")


def _line_cells(x0: float, y0: float, x1: float, y1: float,
                width: int, height: int) -> set[tuple[int, int]]:
    """精确列举线段触及的栅格；穿过角点时保守计入相邻像元。"""
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        raise ValueError("航段含非有限栅格坐标")
    if not all(0 <= x < width and 0 <= y < height for x, y in ((x0, y0), (x1, y1))):
        raise ValueError("航段端点超出 DEM 范围")
    dx, dy = x1 - x0, y1 - y0
    events = [0.0, 1.0]
    if dx:
        for grid_x in range(math.floor(min(x0, x1)) + 1, math.ceil(max(x0, x1))):
            t = (grid_x - x0) / dx
            if 0 < t < 1:
                events.append(t)
    if dy:
        for grid_y in range(math.floor(min(y0, y1)) + 1, math.ceil(max(y0, y1))):
            t = (grid_y - y0) / dy
            if 0 < t < 1:
                events.append(t)
    events.sort()
    cells: set[tuple[int, int]] = set()

    def include(t: float) -> None:
        x, y = x0 + t * dx, y0 + t * dy
        # 双侧微扰可涵盖恰好沿格线或穿格点的情况。
        for epsilon_x in (-1e-9, 1e-9):
            for epsilon_y in (-1e-9, 1e-9):
                col, row = math.floor(x + epsilon_x), math.floor(y + epsilon_y)
                if 0 <= col < width and 0 <= row < height:
                    cells.add((row, col))

    for idx, t in enumerate(events):
        include(t)
        if idx + 1 < len(events) and events[idx + 1] > t:
            include((t + events[idx + 1]) / 2)
    return cells


def geo(project: Path, run_dir: Path) -> None:
    revoke_ready(run_dir)
    derived = run_dir / "derived"
    if derived.exists():
        raise FileExistsError(f"派生目录已存在，不覆盖：{derived}")
    derived.mkdir(parents=True)
    dem_folder = project / DEM_DIR
    mat_path = next(dem_folder.glob("*.mat"))
    tif_path = next(dem_folder.glob("*.tif"))
    mat = scipy.io.loadmat(mat_path)
    dem = np.asarray(mat["dem"])
    tif_dem = tifffile.imread(tif_path)
    if dem.ndim != 2 or not np.array_equal(dem, tif_dem):
        raise ValueError("DEM.mat 与 GeoTIFF 不逐像元一致，停止空间派生")
    lats = np.asarray(mat["latitude"]).ravel()
    lons = np.asarray(mat["longitude"]).ravel()
    nodata = float(np.asarray(mat["nodata"]).item())
    epsg = int(np.asarray(mat["epsg_code"]).item())
    if dem.shape != (len(lats), len(lons)) or epsg != 4326:
        raise ValueError("DEM 坐标数组、矩阵尺寸或 EPSG 不符合预期")
    lon_step, lat_step = float(lons[1] - lons[0]), float(lats[0] - lats[1])
    if (lon_step <= 0 or lat_step <= 0 or
            not np.allclose(np.diff(lons), lon_step, rtol=0, atol=1e-10) or
            not np.allclose(np.diff(lats), -lat_step, rtol=0, atol=1e-10)):
        raise ValueError("DEM 坐标轴不是规则经纬度栅格")
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        scale = page.tags["ModelPixelScaleTag"].value
        tie = page.tags["ModelTiepointTag"].value
        geokey = page.tags["GeoKeyDirectoryTag"].value
    if (not np.allclose(scale[:2], [lon_step, lat_step], rtol=0, atol=1e-10) or
            not np.allclose(tie[3:5], [lons[0], lats[0]], rtol=0, atol=1e-9) or
            4326 not in geokey):
        raise ValueError("GeoTIFF 仿射坐标或地理参考与 DEM.mat 不一致")
    dem_meta = {
        "source_mat": str(mat_path.relative_to(project)).replace("\\", "/"),
        "source_tif": str(tif_path.relative_to(project)).replace("\\", "/"),
        "shape_rows_cols": list(dem.shape), "epsg": epsg, "nodata": nodata,
        "lon_first_center": float(lons[0]), "lon_last_center": float(lons[-1]),
        "lat_first_center": float(lats[0]), "lat_last_center": float(lats[-1]),
        "lon_step_deg": lon_step, "lat_step_deg": lat_step,
        "tif_pixel_scale_deg": [float(scale[0]), float(scale[1])],
        "tif_tiepoint_lon_lat": [float(tie[3]), float(tie[4])],
        "valid_min_m": float(np.min(dem[dem != nodata])),
        "valid_max_m": float(np.max(dem[dem != nodata])),
        "nodata_count": int(np.sum(dem == nodata)), "mat_tif_pixel_equal": True,
    }
    dump_json(run_dir / "meta" / "dem_meta.json", dem_meta)

    nodes = read_clean(run_dir, "nodes")
    positions = {}
    for row in nodes.itertuples():
        x = (row.lon_deg - lons[0]) / lon_step + 0.5
        y = (lats[0] - row.lat_deg) / lat_step + 0.5
        positions[row.node_id] = (float(x), float(y))
    records = []
    for origin in nodes.itertuples():
        for dest in nodes.itertuples():
            if origin.node_id == dest.node_id:
                continue
            record = {
                "from_id": origin.node_id, "to_id": dest.node_id,
                "distance_m": float(math.hypot(dest.x_m - origin.x_m, dest.y_m - origin.y_m)),
                "start_alt_m": float(origin.operation_alt_m),
                "end_alt_m": float(dest.operation_alt_m),
                "max_dsm_m": np.nan, "cruise_alt_m": np.nan,
                "climb_m": np.nan, "descent_m": np.nan,
                "pixels_traversed": 0, "valid_flag": False, "invalid_reason": "",
            }
            try:
                cells = _line_cells(*positions[origin.node_id], *positions[dest.node_id],
                                    dem.shape[1], dem.shape[0])
                heights = np.array([dem[r, c] for r, c in cells])
                if not cells or np.any(~np.isfinite(heights)) or np.any(heights == nodata):
                    raise ValueError("航段经过空值或 NoData 像元")
                max_height = float(np.max(heights))
                cruise = max_height + 50.0
                record.update({
                    "max_dsm_m": max_height, "cruise_alt_m": cruise,
                    "climb_m": max(0.0, cruise - origin.operation_alt_m),
                    "descent_m": max(0.0, cruise - dest.operation_alt_m),
                    "pixels_traversed": len(cells), "valid_flag": True,
                })
            except ValueError as exc:
                record["invalid_reason"] = str(exc)
            records.append(record)
    write_csv(derived / "arc_geometry.csv", pd.DataFrame.from_records(records))
    print(f"03 地理派生完成：{len(records)} 条有向弧；MAT/TIFF 逐像元一致")


def _validate_impl(project: Path, run_dir: Path) -> bool:
    checks: list[dict[str, object]] = []

    def check(code: str, actual: object, expected: object, passed: bool | None = None,
              detail: str = "") -> None:
        good = actual == expected if passed is None else bool(passed)
        checks.append({"check_id": code, "status": "PASS" if good else "FAIL",
                       "expected": str(expected), "actual": str(actual), "detail": detail})

    manifest = json.loads((run_dir / "meta" / "source_manifest.json").read_text(encoding="utf-8"))
    for relative, old_hash in manifest["source_files"].items():
        path = project / relative
        current = sha256(path) if path.is_file() else "MISSING"
        check(f"source_sha256:{relative}", current, old_hash)

    names = ["nodes", "boxes", "demand_summary", "transport_types", "transport_uavs",
             "battery_pool", "battery_units", "relay_type", "relay_uavs", "relay_energy_pool",
             "relay_energy_units", "comm_params"]
    tables = {name: read_clean(run_dir, name) for name in names}
    arcs = pd.read_csv(run_dir / "derived" / "arc_geometry.csv",
                       dtype={"from_id": str, "to_id": str})
    nodes, boxes, summary = tables["nodes"], tables["boxes"], tables["demand_summary"]
    counts = {"nodes": 16, "boxes": 80, "demand_summary": 53, "transport_types": 3,
              "transport_uavs": 8, "battery_pool": 3, "battery_units": 14,
              "relay_type": 1, "relay_uavs": 2, "relay_energy_pool": 1,
              "relay_energy_units": 6, "comm_params": 14}
    for name, expected in counts.items():
        check(f"row_count:{name}", len(tables[name]), expected)
    check("zones", int(nodes["kind"].eq("zone").sum()), 15)
    check("center", nodes.loc[nodes["kind"].eq("center"), "node_id"].tolist(), ["O01"])
    check("zone_ids", sorted(nodes.loc[nodes.kind.eq("zone"), "node_id"]),
          [f"S{i:03d}" for i in range(1, 16)])
    check("first_batch", int(boxes["is_first_batch"].sum()), 30)
    check("medical", int(boxes["medical_bool"].sum()), 16)
    check("medical_flag_matches_type",
          int((boxes["medical_bool"] != boxes["item_type"].eq("医疗物资")).sum()), 0)
    check("mass_kg", float(boxes["mass_kg"].sum()), 758.0,
          bool(np.isclose(boxes["mass_kg"].sum(), 758.0)))
    check("volume_m3", float(boxes["volume_m3"].sum()), 2.011,
          bool(np.isclose(boxes["volume_m3"].sum(), 2.011, atol=1e-9)))

    keys = {"nodes": ["node_id"], "boxes": ["box_id"],
            "demand_summary": ["zone_id", "item_type"],
            "transport_types": ["type_id"], "transport_uavs": ["uav_id"],
            "battery_pool": ["type_id"], "battery_units": ["battery_id"],
            "relay_type": ["type_id"], "relay_uavs": ["uav_id"],
            "relay_energy_pool": ["type_id"], "relay_energy_units": ["unit_id"],
            "comm_params": ["category", "parameter_name"]}
    for name, cols in keys.items():
        table = tables[name]
        blank_key = table[cols].isna().any(axis=1)
        for col in cols:
            if table[col].dtype == object:
                blank_key |= table[col].astype(str).str.strip().eq("")
        invalid = int(table.duplicated(cols).sum() + blank_key.sum())
        check(f"key:{name}", invalid, 0)
        invalid_origin = (table[["source_file", "source_sheet"]].isna().any(axis=1) |
                          table["source_file"].eq("") | table["source_sheet"].eq("") |
                          table["source_row"].isna() | (table["source_row"] <= 0))
        check(f"provenance:{name}", int(invalid_origin.sum()), 0)
        numerics = table.select_dtypes(include=[np.number])
        check(f"infinite_numeric:{name}", int(np.isinf(numerics.to_numpy(dtype=float)).sum()), 0)
    zone_ids = set(nodes.loc[nodes.kind.eq("zone"), "node_id"])
    type_ids = set(tables["transport_types"]["type_id"])
    relay_type_ids = set(tables["relay_type"]["type_id"])
    fk = {
        "boxes.zone_id": (set(boxes["zone_id"]), zone_ids),
        "summary.zone_id": (set(summary["zone_id"]), zone_ids),
        "transport_uavs.type_id": (set(tables["transport_uavs"]["type_id"]), type_ids),
        "battery_pool.type_id": (set(tables["battery_pool"]["type_id"]), type_ids),
        "battery_units.type_id": (set(tables["battery_units"]["type_id"]), type_ids),
        "relay_uavs.type_id": (set(tables["relay_uavs"]["type_id"]), relay_type_ids),
        "relay_energy_pool.type_id": (set(tables["relay_energy_pool"]["type_id"]), relay_type_ids),
        "relay_energy_units.type_id": (set(tables["relay_energy_units"]["type_id"]), relay_type_ids),
    }
    for name, (values, allowed) in fk.items():
        missing = sorted(values - allowed)
        check(f"foreign_key:{name}", missing, [])
    gateway = json.loads((run_dir / "meta" / "gateway.json").read_text(encoding="utf-8"))
    center_row = nodes.loc[nodes.node_id.eq("O01")].iloc[0]
    h_g = tables["comm_params"].loc[
        tables["comm_params"]["category"].eq("固定网关 G01") &
        tables["comm_params"]["symbol"].eq("hG"), "value"]
    check("gateway_unique_height", len(h_g), 1)
    check("gateway_id_and_base", [gateway["gateway_id"], gateway["base_node_id"]], ["G01", "O01"])
    if len(h_g) == 1:
        gateway_expected = [center_row.lon_deg, center_row.lat_deg, center_row.x_m,
                            center_row.y_m, center_row.ground_elev_m, h_g.iloc[0],
                            center_row.ground_elev_m + h_g.iloc[0]]
        gateway_actual = [gateway[key] for key in ("lon_deg", "lat_deg", "x_m", "y_m",
                                                   "ground_elev_m", "antenna_agl_m", "antenna_abs_m")]
        check("gateway_coordinates_and_altitude",
              int((~np.isclose(gateway_actual, gateway_expected, rtol=0, atol=1e-5)).sum()), 0)
    check("transport_initial_site", sorted(set(tables["transport_uavs"]["initial_site"])), ["O01"])
    check("relay_initial_site", sorted(set(tables["relay_uavs"]["initial_site"])), ["O01"])
    for pool_name, units_name in (("battery_pool", "battery_units"),
                                  ("relay_energy_pool", "relay_energy_units")):
        pool = tables[pool_name].set_index("type_id")["stock_qty"].to_dict()
        unit_counts = tables[units_name].groupby("type_id").size().to_dict()
        check(f"resource_stock:{pool_name}", unit_counts,
              {k: int(v) for k, v in pool.items()})
        check(f"resource_initial_soc:{units_name}",
              int((~np.isclose(tables[units_name]["initial_soc"], 1.0)).sum()), 0)
    transport_counts = tables["transport_uavs"].groupby("type_id").size().to_dict()
    check("transport_uav_type_counts", transport_counts, {"A": 4, "B": 2, "C": 2})
    check("transport_battery_type_counts",
          tables["battery_pool"].set_index("type_id")["stock_qty"].astype(int).to_dict(),
          {"A": 6, "B": 4, "C": 4})
    check("relay_uav_ids", sorted(tables["relay_uavs"]["uav_id"]), ["R01", "R02"])
    check("relay_energy_total", int(tables["relay_energy_pool"]["stock_qty"].sum()), 6)
    positive_fields = {
        "boxes": ["mass_kg", "volume_m3", "expected_s", "priority"],
        "transport_types": ["empty_mass_kg", "max_payload_kg", "capacity_m3", "cruise_speed_mps",
                            "range_empty_m", "range_full_m", "energy_use_kwh", "prep_s",
                            "load_per_box_s", "handoff_base_s", "handoff_per_box_s",
                            "climb_speed_mps", "descent_speed_mps", "climb_efficiency"],
        "battery_pool": ["stock_qty", "full_charge_s"],
        "relay_type": ["empty_mass_kg", "module_mass_kg", "takeoff_mass_kg",
                       "cruise_speed_mps", "cruise_power_kw", "energy_use_kwh", "prep_s",
                       "link_setup_s", "turnaround_s", "climb_speed_mps", "descent_speed_mps",
                       "climb_efficiency", "hover_power_kw",
                       "comm_extra_power_kw", "max_hover_agl_m"],
        "relay_energy_pool": ["stock_qty", "full_charge_s"],
    }
    for name, fields in positive_fields.items():
        check(f"positive_values:{name}", int((tables[name][fields] <= 0).sum().sum()), 0)
    for name in ("transport_types", "relay_type"):
        check(f"nonnegative_descent_efficiency:{name}",
              int((tables[name]["descent_efficiency"] < 0).sum()), 0)
    for name in ("transport_types", "relay_type"):
        soc = tables[name]["return_soc_min"]
        check(f"return_soc_range:{name}", int(((soc <= 0) | (soc >= 1)).sum()), 0)

    expected_hard = pd.concat([
        boxes["expected_s"].where(boxes["medical_bool"]),
        boxes["first_deadline_s"].where(boxes["is_first_batch"]),
    ], axis=1).min(axis=1)
    check("hard_deadline_min", int((~np.isclose(boxes["hard_deadline_s"], expected_hard,
                                        equal_nan=True)).sum()), 0)
    check("nonfirst_first_deadline_null",
          int(boxes.loc[~boxes["is_first_batch"], "first_deadline_s"].notna().sum()), 0)
    check("first_deadline_present",
          int(boxes.loc[boxes["is_first_batch"], "first_deadline_s"].isna().sum()), 0)
    check("medical_hard_deadline_present",
          int(boxes.loc[boxes["medical_bool"], "hard_deadline_s"].isna().sum()), 0)
    check("positive_applicable_deadlines",
          int((boxes[["first_deadline_s", "hard_deadline_s"]] <= 0).sum().sum()), 0)
    check("first_boxes_not_above_total", int((summary["first_boxes"] > summary["total_boxes"]).sum()), 0)
    check("summary_first_deadline_applicability",
          int((summary["first_boxes"].gt(0) != summary["first_deadline_s"].notna()).sum()), 0)

    mismatches = []
    groups = boxes.groupby(["zone_id", "item_type"], dropna=False)
    summary_keys = set(zip(summary["zone_id"], summary["item_type"]))
    box_keys = set(groups.groups)
    check("summary_group_keys", len(summary_keys.symmetric_difference(box_keys)), 0)
    for row in summary.itertuples():
        key = (row.zone_id, row.item_type)
        if key not in groups.groups:
            mismatches.append(f"{key}:missing")
            continue
        group = groups.get_group(key)
        comparisons = {
            "total_boxes": (len(group), row.total_boxes),
            "first_boxes": (int(group["is_first_batch"].sum()), row.first_boxes),
            "unit_mass_kg": (group["mass_kg"].iloc[0], row.unit_mass_kg),
            "unit_volume_m3": (group["volume_m3"].iloc[0], row.unit_volume_m3),
            "priority": (group["priority"].iloc[0], row.priority),
            "expected_s": (group["expected_s"].iloc[0], row.expected_s),
        }
        first = group.loc[group["is_first_batch"], "first_deadline_s"]
        comparisons["first_deadline_s"] = (first.iloc[0] if len(first) else np.nan, row.first_deadline_s)
        for field, (observed, listed) in comparisons.items():
            if not np.isclose(observed, listed, equal_nan=True):
                mismatches.append(f"{key}:{field}")
        for field in ("mass_kg", "volume_m3", "priority", "expected_s", "first_deadline_s"):
            values = group.loc[group["is_first_batch"], field] if field == "first_deadline_s" else group[field]
            if len(values) and values.nunique(dropna=False) != 1:
                mismatches.append(f"{key}:{field}:inconsistent_boxes")
    check("summary_box_reconciliation", len(mismatches), 0, detail="; ".join(mismatches[:12]))

    dem_meta = json.loads((run_dir / "meta" / "dem_meta.json").read_text(encoding="utf-8"))
    check("dem_shape", dem_meta["shape_rows_cols"], [1309, 1486])
    check("dem_epsg", dem_meta["epsg"], 4326)
    check("dem_nodata", dem_meta["nodata"], -32767.0)
    check("dem_lon_step_1arcsec", dem_meta["lon_step_deg"], 1 / 3600,
          bool(np.isclose(dem_meta["lon_step_deg"], 1 / 3600, rtol=0, atol=1e-10)))
    check("dem_lat_step_1arcsec", dem_meta["lat_step_deg"], 1 / 3600,
          bool(np.isclose(dem_meta["lat_step_deg"], 1 / 3600, rtol=0, atol=1e-10)))
    check("dem_mat_tif_equal", dem_meta["mat_tif_pixel_equal"], True)
    lon_lo = dem_meta["lon_first_center"] - dem_meta["lon_step_deg"] / 2
    lon_hi = dem_meta["lon_last_center"] + dem_meta["lon_step_deg"] / 2
    lat_lo = dem_meta["lat_last_center"] - dem_meta["lat_step_deg"] / 2
    lat_hi = dem_meta["lat_first_center"] + dem_meta["lat_step_deg"] / 2
    outside = int((~nodes["lon_deg"].between(lon_lo, lon_hi) |
                   ~nodes["lat_deg"].between(lat_lo, lat_hi)).sum())
    check("nodes_in_dem", outside, 0)
    check("node_operation_altitude", int((~np.isclose(
        nodes["operation_alt_m"], nodes["ground_elev_m"] + np.where(nodes.kind.eq("zone"), 30, 0)
    )).sum()), 0)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    proj_x, proj_y = transformer.transform(nodes["lon_deg"].to_numpy(), nodes["lat_deg"].to_numpy())
    check("node_utm_projection", int((~np.isclose(nodes["x_m"], proj_x, rtol=0, atol=1e-4) |
                                      ~np.isclose(nodes["y_m"], proj_y, rtol=0, atol=1e-4)).sum()), 0)
    check("arcs_count", len(arcs), 240)
    check("arcs_unique", int(arcs.duplicated(["from_id", "to_id"]).sum()), 0)
    check("arcs_no_self", int(arcs["from_id"].eq(arcs["to_id"]).sum()), 0)
    expected_pairs = {(a, b) for a in nodes["node_id"] for b in nodes["node_id"] if a != b}
    actual_pairs = set(zip(arcs["from_id"], arcs["to_id"]))
    check("arcs_complete_pairs", len(expected_pairs.symmetric_difference(actual_pairs)), 0)
    check("arcs_valid_or_explained", int((~arcs["valid_flag"] & arcs["invalid_reason"].isna()).sum()), 0)
    check("arcs_all_valid", int(arcs["valid_flag"].sum()), 240)
    valid = arcs.loc[arcs["valid_flag"]]
    check("arcs_positive_distance", int((valid["distance_m"] <= 0).sum()), 0)
    check("arcs_clearance_50m", int((~np.isclose(valid["cruise_alt_m"] - valid["max_dsm_m"], 50)).sum()), 0)
    check("arcs_nonnegative_climb_descent", int(((valid["climb_m"] < 0) | (valid["descent_m"] < 0)).sum()), 0)
    reverse = valid.merge(valid, left_on=["from_id", "to_id"], right_on=["to_id", "from_id"],
                          suffixes=("_out", "_back"))
    asymmetric = (~np.isclose(reverse["distance_m_out"], reverse["distance_m_back"]) |
                  ~np.isclose(reverse["max_dsm_m_out"], reverse["max_dsm_m_back"]))
    check("reverse_arc_geometry", int(asymmetric.sum()), 0)
    # 从原始 MAT 与标准节点重新计算全部弧，检查派生 CSV 未被整体等量篡改。
    mat = scipy.io.loadmat(next((project / DEM_DIR).glob("*.mat")))
    dem = np.asarray(mat["dem"])
    lons, lats = np.asarray(mat["longitude"]).ravel(), np.asarray(mat["latitude"]).ravel()
    lon_step, lat_step = float(lons[1] - lons[0]), float(lats[0] - lats[1])
    node_by_id = nodes.set_index("node_id")
    bad_recomputed = 0
    for arc in valid.itertuples():
        origin, dest = node_by_id.loc[arc.from_id], node_by_id.loc[arc.to_id]
        x0, x1 = ((point.lon_deg - lons[0]) / lon_step + 0.5 for point in (origin, dest))
        y0, y1 = ((lats[0] - point.lat_deg) / lat_step + 0.5 for point in (origin, dest))
        cells = _line_cells(x0, y0, x1, y1, dem.shape[1], dem.shape[0])
        expected_max = max(float(dem[r, c]) for r, c in cells)
        expected_distance = math.hypot(dest.x_m - origin.x_m, dest.y_m - origin.y_m)
        expected_cruise = expected_max + 50
        metrics = ((arc.distance_m, expected_distance), (arc.max_dsm_m, expected_max),
                   (arc.cruise_alt_m, expected_cruise),
                   (arc.start_alt_m, origin.operation_alt_m), (arc.end_alt_m, dest.operation_alt_m),
                   (arc.climb_m, max(0, expected_cruise - origin.operation_alt_m)),
                   (arc.descent_m, max(0, expected_cruise - dest.operation_alt_m)))
        if len(cells) != arc.pixels_traversed or any(
                not np.isclose(actual, expected, rtol=0, atol=1e-4) for actual, expected in metrics):
            bad_recomputed += 1
    check("arcs_recomputed_from_sources", bad_recomputed, 0)

    report = pd.DataFrame(checks)
    write_csv(run_dir / "meta" / "qa_report.csv", report)
    passed = bool(report["status"].eq("PASS").all())
    summary_text = (f"数据校验：{'通过' if passed else '失败'}\n"
                    f"检查项：{len(report)}；通过：{int(report.status.eq('PASS').sum())}；失败：{int(report.status.eq('FAIL').sum())}\n"
                    f"逐箱：{len(boxes)}；总质量：{boxes.mass_kg.sum():.3f} kg；总体积：{boxes.volume_m3.sum():.3f} m³\n"
                    f"有向航段：{len(arcs)}；有效：{int(arcs.valid_flag.sum())}\n"
                    "只有存在 meta/READY.txt 时，clean 与 derived 才可供后续建模读取。\n")
    (run_dir / "检查结果.txt").write_text(summary_text, encoding="utf-8")
    ready = run_dir / "meta" / "READY.txt"
    if passed:
        paths = ([run_dir / "clean" / f"{name}.csv" for name in names] +
                 [run_dir / "derived" / "arc_geometry.csv"] +
                 [run_dir / "meta" / filename for filename in
                  ("source_manifest.json", "schema.json", "dem_meta.json", "gateway.json", "qa_report.csv")])
        artifacts = {
            str(path.relative_to(run_dir)).replace("\\", "/"): sha256(path) for path in paths
        }
        artifacts_path = run_dir / "meta" / "validated_artifacts.json"
        dump_json(artifacts_path, artifacts)
        ready.write_text(f"PASS\nmanifest_sha256={sha256(artifacts_path)}\n", encoding="utf-8")
    # Windows 默认 GBK 控制台无法输出上标 ³；UTF-8 文件仍保留标准单位。
    print(summary_text.replace("m³", "m3").strip())
    return passed


def record_failure(run_dir: Path, stage: str, exc: Exception) -> None:
    revoke_ready(run_dir)
    report = pd.DataFrame([{
        "check_id": f"exception:{stage}", "status": "FAIL", "expected": "stage completes",
        "actual": type(exc).__name__, "detail": str(exc),
    }])
    write_csv(run_dir / "meta" / "qa_report.csv", report)
    (run_dir / "检查结果.txt").write_text(
        f"数据校验：失败\n阶段：{stage}\n错误：{type(exc).__name__}: {exc}\n"
        "无 READY.txt，本次数据不得用于建模。\n", encoding="utf-8")


def validate(project: Path, run_dir: Path) -> bool:
    revoke_ready(run_dir)
    try:
        return _validate_impl(project, run_dir)
    except Exception as exc:
        record_failure(run_dir, "validate", exc)
        raise


def verify_ready(run_dir: Path) -> bool:
    """供后续建模代码调用：READY、QA 与验收时的全部文件哈希必须一致。"""
    ready = run_dir / "meta" / "READY.txt"
    artifacts_path = run_dir / "meta" / "validated_artifacts.json"
    report_path = run_dir / "meta" / "qa_report.csv"
    if not all(path.is_file() for path in (ready, artifacts_path, report_path)):
        return False
    lines = ready.read_text(encoding="utf-8").splitlines()
    if len(lines) != 2 or lines[0] != "PASS" or lines[1] != f"manifest_sha256={sha256(artifacts_path)}":
        return False
    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    if any(not (run_dir / relative).is_file() or sha256(run_dir / relative) != old_hash
           for relative, old_hash in artifacts.items()):
        return False
    qa = pd.read_csv(report_path)
    return bool(len(qa) and qa["status"].eq("PASS").all())


def main_stage(stage: str) -> None:
    parser = argparse.ArgumentParser(description=f"D 题数据管线：{stage}")
    parser.add_argument("--run-dir", required=True, type=Path, help="本次输出目录")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    run_dir = args.run_dir.resolve()
    try:
        if stage == "extract":
            extract(project, run_dir)
        elif stage == "normalize":
            normalize(run_dir)
        elif stage == "geo":
            geo(project, run_dir)
        elif stage == "validate":
            if not validate(project, run_dir):
                raise SystemExit(1)
        else:
            raise ValueError(f"未知阶段：{stage}")
    except Exception as exc:
        record_failure(run_dir, stage, exc)
        raise
