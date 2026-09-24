"""问题三基线：固定已验收 Q2 运输计划，贪心安排通信中继。

搜索是有限网格启发式；只有保守区间复核通过才写 Q3_READY.txt。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import scipy.io
from pyproj import Transformer

from _0_pipeline import _line_cells, verify_ready
from q2_0_baseline import charge_to_full_s
from q2_1_optimize import Context


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_Q2_DIR = (CODE_DIR / "2_outputs" / "5_optimize" /
                  "q2_opt5_260924_151100_table")
EXPECTED_MANIFEST = "dab98bafa9b2101b0b6265be77ec45e95baecd5646eadf0c6acb905f3e45ca5e"
EPS = 1e-7

TRANSPORT_COLUMNS = [
    "架次编号", "无人机编号", "机型编号", "电池编号", "开始时刻（s）",
    "访问服务区顺序", "返回O01时刻（s）", "架次能耗（kWh）",
]
DELIVERY_COLUMNS = ["货箱编号", "架次编号", "服务区编号", "交付完成时刻（s）"]
RELAY_COLUMNS = [
    "中继架次编号", "中继无人机编号", "能源组件编号", "开始时刻（s）",
    "悬停经度（°）", "悬停纬度（°）", "悬停海拔（m）", "建链完成时刻（s）",
    "服务结束时刻（s）", "返回O01时刻（s）", "架次能耗（kWh）",
]
COMM_COLUMNS = [
    "运输架次编号", "通信阶段", "开始时刻（s）", "结束时刻（s）",
    "保障方式", "中继架次编号",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


@dataclass(frozen=True)
class Radio:
    tx_power_dbm: float
    gain_dbi: float
    sensitivity_dbm: float
    fade_margin_db: float


def bidirectional_limit_db(a: Radio, b: Radio, system_loss_db: float) -> float:
    """传输控制与回传均可用时允许的最大传播损耗，dB。"""
    values = (a.tx_power_dbm, a.gain_dbi, a.sensitivity_dbm, a.fade_margin_db,
              b.tx_power_dbm, b.gain_dbi, b.sensitivity_dbm, b.fade_margin_db,
              system_loss_db)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("链路预算含非有限参数")
    ab = (a.tx_power_dbm + a.gain_dbi + b.gain_dbi - system_loss_db -
          (b.sensitivity_dbm + b.fade_margin_db))
    ba = (b.tx_power_dbm + b.gain_dbi + a.gain_dbi - system_loss_db -
          (a.sensitivity_dbm + a.fade_margin_db))
    return min(ab, ba)


def path_loss_db(distance_m: float, frequency_mhz: float, obstructed: bool,
                 obstruction_loss_db: float) -> float:
    values = (distance_m, frequency_mhz, obstruction_loss_db)
    if not all(math.isfinite(float(v)) for v in values) or distance_m <= 0 or frequency_mhz <= 0:
        raise ValueError("传播距离/频率必须为正且有限")
    return (32.45 + 20.0 * math.log10(frequency_mhz) +
            20.0 * math.log10(distance_m / 1000.0) +
            (obstruction_loss_db if obstructed else 0.0))


@dataclass(frozen=True)
class TrajectorySegment:
    stage: str
    start_s: float
    end_s: float
    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float

    def __post_init__(self) -> None:
        values = (self.start_s, self.end_s, self.x0, self.y0, self.z0,
                  self.x1, self.y1, self.z1)
        if not all(math.isfinite(float(v)) for v in values) or self.end_s <= self.start_s:
            raise ValueError("轨迹区间必须为正长且所有参数有限")

    def position(self, t: float) -> tuple[float, float, float]:
        if not math.isfinite(t) or t < self.start_s - EPS or t > self.end_s + EPS:
            raise ValueError("时刻不在轨迹区间内")
        alpha = max(0.0, min(1.0, (t - self.start_s) / (self.end_s - self.start_s)))
        return (self.x0 + alpha * (self.x1 - self.x0),
                self.y0 + alpha * (self.y1 - self.y0),
                self.z0 + alpha * (self.z1 - self.z0))


@dataclass(frozen=True)
class LinkResult:
    available: bool
    margin_db: float
    obstructed: bool
    distance_m: float


class DemGrid:
    """官方 DEM.mat 的经纬度中心栅格；路径仍在 UTM 49N 米坐标计算。"""

    def __init__(self, mat_path: Path):
        mat = scipy.io.loadmat(mat_path)
        self.dem = np.asarray(mat["dem"], dtype=np.float64)
        self.lats = np.asarray(mat["latitude"], dtype=np.float64).ravel()
        self.lons = np.asarray(mat["longitude"], dtype=np.float64).ravel()
        self.nodata = float(np.asarray(mat["nodata"]).item())
        if self.dem.shape != (len(self.lats), len(self.lons)):
            raise ValueError("DEM 经纬度轴与矩阵尺寸不符")
        self.lon_step = float(self.lons[1] - self.lons[0])
        self.lat_step = float(self.lats[0] - self.lats[1])
        if self.lon_step <= 0 or self.lat_step <= 0:
            raise ValueError("DEM 坐标轴方向错误")
        self.to_lonlat = Transformer.from_crs(32649, 4326, always_xy=True)
        self.to_xy = Transformer.from_crs(4326, 32649, always_xy=True)

    def grid_xy(self, x_m: float, y_m: float) -> tuple[float, float]:
        lon, lat = self.to_lonlat.transform(x_m, y_m)
        return ((lon - float(self.lons[0])) / self.lon_step + 0.5,
                (float(self.lats[0]) - lat) / self.lat_step + 0.5)

    def terrain_at(self, x_m: float, y_m: float) -> float:
        gx, gy = self.grid_xy(x_m, y_m)
        col, row = math.floor(gx), math.floor(gy)
        if not (0 <= row < self.dem.shape[0] and 0 <= col < self.dem.shape[1]):
            raise ValueError("坐标位于 DEM 范围外")
        value = float(self.dem[row, col])
        if not math.isfinite(value) or value == self.nodata:
            raise ValueError("DEM 像元无效")
        return value

    @staticmethod
    def _cell_interval(x0: float, y0: float, x1: float, y1: float,
                       col: int, row: int) -> tuple[float, float] | None:
        """线段与闭合栅格单元的参数交区间。"""
        low, high = 0.0, 1.0
        for p0, delta, left, right in (
            (x0, x1 - x0, col, col + 1), (y0, y1 - y0, row, row + 1),
        ):
            if abs(delta) < 1e-14:
                if p0 < left - 1e-12 or p0 > right + 1e-12:
                    return None
                continue
            t0, t1 = (left - p0) / delta, (right - p0) / delta
            low, high = max(low, min(t0, t1)), min(high, max(t0, t1))
            if high < low - 1e-12:
                return None
        return max(0.0, low), min(1.0, high)

    def obstructed(self, a: tuple[float, float, float],
                   b: tuple[float, float, float]) -> bool:
        x0, y0 = self.grid_xy(a[0], a[1])
        x1, y1 = self.grid_xy(b[0], b[1])
        cells = _line_cells(x0, y0, x1, y1, self.dem.shape[1], self.dem.shape[0])
        for row, col in cells:
            interval = self._cell_interval(x0, y0, x1, y1, col, row)
            if interval is None:
                continue
            ground = float(self.dem[row, col])
            if not math.isfinite(ground) or ground == self.nodata:
                raise ValueError("视线路径经过无效 DEM 像元")
            t0, t1 = interval
            z0 = a[2] + t0 * (b[2] - a[2])
            z1 = a[2] + t1 * (b[2] - a[2])
            if ground > min(z0, z1) + 1e-6:
                return True
        return False

    def arc_geometry(self, a: tuple[float, float, float],
                     b: tuple[float, float, float]) -> dict[str, float]:
        x0, y0 = self.grid_xy(a[0], a[1])
        x1, y1 = self.grid_xy(b[0], b[1])
        cells = _line_cells(x0, y0, x1, y1, self.dem.shape[1], self.dem.shape[0])
        if not cells:
            raise ValueError("候选航段没有 DEM 像元")
        heights = [float(self.dem[row, col]) for row, col in cells]
        if any(not math.isfinite(h) or h == self.nodata for h in heights):
            raise ValueError("候选航段经过无效 DEM")
        cruise = max(heights) + 50.0
        return {
            "distance_m": math.hypot(b[0] - a[0], b[1] - a[1]),
            "cruise_alt_m": cruise,
            "climb_m": max(0.0, cruise - a[2]),
            "descent_m": max(0.0, cruise - b[2]),
        }


class LinkEvaluator:
    def __init__(self, dem: DemGrid, gateway: tuple[float, float, float],
                 radios: dict[str, Radio], frequency_mhz: float,
                 system_loss_db: float, obstruction_loss_db: float):
        self.dem = dem
        self.gateway = gateway
        self.radios = radios
        self.frequency_mhz = frequency_mhz
        self.system_loss_db = system_loss_db
        self.obstruction_loss_db = obstruction_loss_db
        self._fixed_cache: dict[tuple[float, float, float, str], LinkResult] = {}

    def link(self, a: tuple[float, float, float], radio_a: str,
             b: tuple[float, float, float], radio_b: str) -> LinkResult:
        distance = math.dist(a, b)
        obstructed = self.dem.obstructed(a, b)
        loss = path_loss_db(distance, self.frequency_mhz, obstructed,
                            self.obstruction_loss_db)
        limit = bidirectional_limit_db(self.radios[radio_a], self.radios[radio_b],
                                       self.system_loss_db)
        return LinkResult(loss <= limit + EPS, limit - loss, obstructed, distance)

    def direct(self, transport: tuple[float, float, float]) -> LinkResult:
        return self.link(transport, "transport", self.gateway, "gateway")

    def backhaul(self, hover: tuple[float, float, float]) -> LinkResult:
        key = (*hover, "backhaul")
        if key not in self._fixed_cache:
            self._fixed_cache[key] = self.link(hover, "relay_backhaul", self.gateway,
                                               "gateway")
        return self._fixed_cache[key]

    def access(self, transport: tuple[float, float, float],
               hover: tuple[float, float, float]) -> LinkResult:
        return self.link(transport, "transport", hover, "relay_access")


def find_data_run(code_dir: Path, expected_manifest: str) -> Path:
    matches = []
    for path in (code_dir / "0_outputs").glob("run_*"):
        ready = path / "meta" / "READY.txt"
        if not ready.is_file() or not verify_ready(path):
            continue
        lines = ready.read_text(encoding="utf-8-sig").splitlines()
        found = next((line.split("=", 1)[1].strip() for line in lines
                      if line.startswith("manifest_sha256=")), None)
        if found == expected_manifest:
            matches.append(path)
    if not matches:
        raise FileNotFoundError("未找到与第二问源数据哈希一致的 PASS 数据运行目录")
    return max(matches, key=lambda p: p.name)


def load_comm(data_run: Path) -> tuple[dict[str, Radio], dict[str, float]]:
    rows = read_rows(data_run / "clean" / "comm_params.csv")
    param = {(row["category"], row["symbol"]): float(row["value"]) for row in rows}
    sensitivity = param[("接收参数", "Psens")]
    margin = param[("接收参数", "M")]
    def radio(category: str) -> Radio:
        return Radio(param[(category, "Pt")], param[(category, "G")],
                     sensitivity, margin)
    radios = {
        "transport": radio("运输无人机"),
        "relay_access": radio("中继接入端"),
        "relay_backhaul": radio("中继回传端"),
        "gateway": radio("固定网关 G01"),
    }
    return radios, {
        "frequency_mhz": param[("传播参数", "f")],
        "system_loss_db": param[("传播参数", "Lsys")],
        "obstruction_loss_db": param[("传播参数", "Lobs")],
        "gateway_agl_m": param[("固定网关 G01", "hG")],
    }


def load_dem(data_run: Path) -> DemGrid:
    meta = json.loads((data_run / "meta" / "dem_meta.json").read_text(encoding="utf-8"))
    mat_path = data_run.parent.parent.parent / Path(meta["source_mat"])
    if not mat_path.is_file():
        raise FileNotFoundError(f"找不到官方 DEM.mat：{mat_path}")
    return DemGrid(mat_path)


def _close(actual: float, expected: float, label: str, tol: float = 2e-5) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol):
        raise ValueError(f"{label} 重放不一致：computed={actual:.9f}, CSV={expected:.9f}")


def _node_positions(data_run: Path) -> dict[str, tuple[float, float, float]]:
    nodes = read_rows(data_run / "clean" / "nodes.csv")
    return {
        row["node_id"]: (float(row["x_m"]), float(row["y_m"]),
                          float(row["operation_alt_m"]))
        for row in nodes
    }


def _add_segment(segments: list[TrajectorySegment], stage: str,
                 start: float, end: float, a: tuple[float, float, float],
                 b: tuple[float, float, float]) -> None:
    if end - start > 1e-9:
        segments.append(TrajectorySegment(stage, start, end, *a, *b))


def build_trajectory(ctx: Context, row: dict[str, str],
                     positions: dict[str, tuple[float, float, float]],
                     legs_from_csv: list[dict[str, str]],
                     deliveries: dict[str, dict[str, str]]) -> tuple[list[TrajectorySegment],
                                                                      dict[str, Any]]:
    batch_id, type_id = row["batch_id"], row["type_id"]
    zones = row["visit_order"].split(">")
    box_ids = row["box_ids"].split(";")
    if not zones or len(set(zones)) != len(zones) or not box_ids or len(set(box_ids)) != len(box_ids):
        raise ValueError(f"{batch_id} 路线/货箱编号为空或重复")
    grouped = [(zone, [bid for bid in box_ids if str(ctx.boxes[bid]["zone_id"]) == zone])
               for zone in zones]
    if any(not ids for _, ids in grouped) or sum(len(ids) for _, ids in grouped) != len(box_ids):
        raise ValueError(f"{batch_id} 访问服务区与货箱目的地不一致")
    route = ctx.route(grouped)
    if list(route.box_ids) != box_ids:
        raise ValueError(f"{batch_id} 的逐箱交接顺序与问题二重放规则不符")
    option = ctx.evaluate(route, type_id)
    if option is None:
        raise ValueError(f"{batch_id} 路线的载荷/体积/能量不再可行")
    start = float(row["start_s"])
    launch = float(row["launch_s"])
    returned = float(row["return_s"])
    if start < -EPS:
        raise ValueError(f"{batch_id} 开始时刻早于 0")
    _close(start + float(option["launch_offset_s"]), launch, f"{batch_id} 起飞时刻")
    _close(start + float(option["duration_s"]), returned, f"{batch_id} 返航时刻")
    _close(float(option["energy_kwh"]), float(row["energy_kwh"]), f"{batch_id} 能耗")
    segments: list[TrajectorySegment] = []
    model = ctx.models[type_id]
    for index, leg in enumerate(option["legs"]):
        origin, dest = leg["from_id"], leg["to_id"]
        arc = ctx.arcs[(origin, dest)]
        begin = start + float(leg["arrival_offset_s"]) - float(leg["flight_s"])
        arrive = start + float(leg["arrival_offset_s"])
        a, b = positions[origin], positions[dest]
        cruise_alt = float(arc["cruise_alt_m"])
        climb_s = float(arc["climb_m"]) / float(model["climb_speed_mps"])
        cruise_s = float(arc["distance_m"]) / float(model["cruise_speed_mps"])
        descent_s = float(arc["descent_m"]) / float(model["descent_speed_mps"])
        _close(climb_s + cruise_s + descent_s, float(leg["flight_s"]),
               f"{batch_id} 航段 {index + 1} 时间")
        climb_end = begin + climb_s
        cruise_end = climb_end + cruise_s
        _add_segment(segments, "爬升", begin, climb_end, a,
                     (a[0], a[1], cruise_alt))
        _add_segment(segments, "巡航", climb_end, cruise_end,
                     (a[0], a[1], cruise_alt), (b[0], b[1], cruise_alt))
        _add_segment(segments, "下降", cruise_end, arrive,
                     (b[0], b[1], cruise_alt), b)
        if index < len(legs_from_csv):
            recorded_leg = legs_from_csv[index]
            if recorded_leg["起点"] != origin or recorded_leg["终点"] != dest:
                raise ValueError(f"{batch_id} 航段 {index + 1} 节点与 Q2 表不符")
            _close(arrive, float(recorded_leg["抵达时刻（s）"]),
                   f"{batch_id} 航段 {index + 1} 抵达")
            _close(float(leg["energy_kwh"]), float(recorded_leg["航段能耗（kWh）"]),
                   f"{batch_id} 航段 {index + 1} 能耗")
        if dest != "O01":
            visit = option["visits"][index]
            handoff_end = start + float(visit["handoff_end_offset_s"])
            _add_segment(segments, "投送", arrive, handoff_end, b, b)
    if len(legs_from_csv) != len(option["legs"]):
        raise ValueError(f"{batch_id} 航段 CSV 数目与路线不符")
    _close(segments[0].start_s, launch, f"{batch_id} 首段开始")
    _close(segments[-1].end_s, returned, f"{batch_id} 末段结束")
    for prev, curr in zip(segments, segments[1:]):
        _close(prev.end_s, curr.start_s, f"{batch_id} 阶段时间连续")
        if math.dist(prev.position(prev.end_s), curr.position(curr.start_s)) > 1e-5:
            raise ValueError(f"{batch_id} 阶段三维轨迹不连续")
    for bid, offset in option["completion_offsets"].items():
        if bid not in deliveries or deliveries[bid]["架次编号"] != batch_id:
            raise ValueError(f"{bid} 不在 Q2 官方交付表中或架次不符")
        _close(start + float(offset), float(deliveries[bid]["交付完成时刻（s）"]),
               f"{bid} 交付完成")
        hard = ctx.boxes[bid]["hard_deadline_s"]
        if not pd.isna(hard) and start + float(offset) > float(hard) + EPS:
            raise ValueError(f"{bid} 违反硬截止")
    return segments, option


def replay_q2(data_run: Path, q2_dir: Path) -> tuple[Context,
                                                    dict[str, list[TrajectorySegment]],
                                                    dict[str, Any]]:
    summary_path = q2_dir / "Q2_运行摘要.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = str(summary["source_manifest_sha256"])
    ready_lines = (data_run / "meta" / "READY.txt").read_text(encoding="utf-8-sig")
    if (summary.get("status") != "PASS" or expected not in ready_lines or
            not (q2_dir / "Q2_READY.txt").is_file() or not verify_ready(data_run)):
        raise ValueError("Q2 方案或本地源数据没有通过同哈希验收")
    if expected != EXPECTED_MANIFEST:
        raise ValueError("所选 Q2 方案不是设计稿审定的数据版本")
    q2_checks = read_rows(q2_dir / "Q2_验收检查.csv")
    if len(q2_checks) != 207 or any(row["status"] != "PASS" for row in q2_checks):
        raise ValueError("Q2 方案 5 的 207 项验收并非全部 PASS")
    official_sorties = {r["架次编号"]: r for r in read_rows(q2_dir / "Q2_运输架次.csv")}
    official_deliveries = {r["货箱编号"]: r for r in read_rows(q2_dir / "Q2_逐箱交付.csv")}
    schedule_rows = read_rows(q2_dir / "Q2_架次与电池周转.csv")
    leg_rows = defaultdict(list)
    for row in read_rows(q2_dir / "Q2_路线航段.csv"):
        leg_rows[row["架次编号"]].append(row)
    for values in leg_rows.values():
        values.sort(key=lambda r: int(r["航段序号"]))
    if (len(official_sorties) != 19 or len(schedule_rows) != 19 or
            len(official_deliveries) != 80):
        raise ValueError("Q2 基线输入规模不是已验收的 19 架次、80 箱")
    ctx = Context(data_run)
    positions = _node_positions(data_run)
    trajectories: dict[str, list[TrajectorySegment]] = {}
    uav_events = defaultdict(list)
    battery_events = defaultdict(list)
    seen_boxes: set[str] = set()
    for row in schedule_rows:
        batch_id = row["batch_id"]
        if batch_id in trajectories or batch_id not in official_sorties:
            raise ValueError(f"Q2 架次编号缺失或重复：{batch_id}")
        other = official_sorties[batch_id]
        for key, official_key in (("uav_id", "无人机编号"), ("type_id", "机型编号"),
                                  ("battery_id", "电池编号"),
                                  ("visit_order", "访问服务区顺序")):
            if row[key] != other[official_key]:
                raise ValueError(f"{batch_id} Q2 周转表与官方运输表的 {key} 不同")
        _close(float(row["start_s"]), float(other["开始时刻（s）"]), f"{batch_id} 开始")
        _close(float(row["return_s"]), float(other["返回O01时刻（s）"]), f"{batch_id} 返回")
        _close(float(row["energy_kwh"]), float(other["架次能耗（kWh）"]), f"{batch_id} 能耗")
        segments, option = build_trajectory(ctx, row, positions,
                                            leg_rows[batch_id], official_deliveries)
        trajectories[batch_id] = segments
        seen_boxes.update(option["completion_offsets"])
        uav_events[row["uav_id"]].append((float(row["start_s"]), float(row["return_s"])))
        battery_events[row["battery_id"]].append((float(row["start_s"]),
                                                   float(row["charge_end_s"])))
    if set(official_sorties) != set(trajectories) or seen_boxes != set(official_deliveries):
        raise ValueError("Q2 架次或 80 箱覆盖不一致")
    for resource, events in [*uav_events.items(), *battery_events.items()]:
        ordered = sorted(events)
        for (_, previous_end), (next_start, _) in zip(ordered, ordered[1:]):
            if next_start < previous_end - EPS:
                raise ValueError(f"Q2 资源 {resource} 出现重叠占用")
    hashes = {path.name: sha256(path) for path in q2_dir.iterdir()
              if path.name in {"Q2_运输架次.csv", "Q2_逐箱交付.csv",
                               "Q2_架次与电池周转.csv", "Q2_路线航段.csv"}}
    return ctx, trajectories, {
        "source_manifest_sha256": expected,
        "q2_dir": str(q2_dir.resolve()), "data_run": str(data_run.resolve()),
        "q2_file_sha256": hashes, "sorties": len(trajectories),
        "boxes": len(official_deliveries), "q2_checks": len(q2_checks),
        "transport_energy_kwh": sum(float(r["架次能耗（kWh）"])
                                    for r in official_sorties.values()),
        "transport_last_return_s": max(float(r["返回O01时刻（s）"])
                                       for r in official_sorties.values()),
    }


@dataclass(frozen=True)
class TimeSlice:
    slice_id: int
    sortie_id: str
    stage: str
    start_s: float
    end_s: float
    midpoint: tuple[float, float, float]
    direct_margin_db: float
    direct_available: bool
    segment: TrajectorySegment

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def scan_direct(trajectories: dict[str, list[TrajectorySegment]],
                links: LinkEvaluator, sample_step_s: float) -> list[TimeSlice]:
    if not math.isfinite(sample_step_s) or sample_step_s <= 0:
        raise ValueError("通信采样步长必须为正")
    slices: list[TimeSlice] = []
    for sortie_id in sorted(trajectories):
        for segment in trajectories[sortie_id]:
            pieces = max(1, math.ceil((segment.end_s - segment.start_s) / sample_step_s))
            for index in range(pieces):
                start = segment.start_s + (segment.end_s - segment.start_s) * index / pieces
                end = segment.start_s + (segment.end_s - segment.start_s) * (index + 1) / pieces
                mid = segment.position((start + end) / 2.0)
                result = links.direct(mid)
                slices.append(TimeSlice(len(slices), sortie_id, segment.stage,
                                        start, end, mid, result.margin_db,
                                        result.available, segment))
    return slices


def blind_intervals(slices: list[TimeSlice]) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    for item in sorted((s for s in slices if not s.direct_available),
                       key=lambda s: (s.sortie_id, s.start_s)):
        if (intervals and intervals[-1]["sortie_id"] == item.sortie_id and
                abs(intervals[-1]["end_s"] - item.start_s) <= 1e-6):
            intervals[-1]["end_s"] = item.end_s
            intervals[-1]["slice_ids"].append(item.slice_id)
            intervals[-1]["worst_direct_margin_db"] = min(
                intervals[-1]["worst_direct_margin_db"], item.direct_margin_db)
        else:
            intervals.append({
                "sortie_id": item.sortie_id, "start_s": item.start_s,
                "end_s": item.end_s, "slice_ids": [item.slice_id],
                "worst_direct_margin_db": item.direct_margin_db,
            })
    return intervals


@dataclass(frozen=True)
class RelayCandidate:
    candidate_id: str
    x_m: float
    y_m: float
    alt_m: float
    lon_deg: float
    lat_deg: float
    agl_m: float
    outward_s: float
    homeward_s: float
    travel_energy_kwh: float
    max_service_s: float
    backhaul_margin_db: float

    @property
    def hover(self) -> tuple[float, float, float]:
        return self.x_m, self.y_m, self.alt_m


def load_relay_resources(data_run: Path) -> dict[str, Any]:
    clean = data_run / "clean"
    types = read_rows(clean / "relay_type.csv")
    uavs = read_rows(clean / "relay_uavs.csv")
    units = read_rows(clean / "relay_energy_units.csv")
    pool = read_rows(clean / "relay_energy_pool.csv")
    if len(types) != 1 or len(pool) != 1 or len(uavs) != 2 or len(units) != 6:
        raise ValueError("中继资源数量不符合官方 2 机、6 组件")
    params = {k: float(v) for k, v in types[0].items()
              if k in {"takeoff_mass_kg", "cruise_speed_mps", "cruise_power_kw",
                       "energy_use_kwh", "return_soc_min", "prep_s", "link_setup_s",
                       "turnaround_s", "climb_speed_mps", "descent_speed_mps",
                       "climb_efficiency", "hover_power_kw", "comm_extra_power_kw",
                       "max_hover_agl_m"}}
    params["full_charge_s"] = float(pool[0]["full_charge_s"])
    if (params["return_soc_min"] != 0.2 or params["turnaround_s"] != 300.0 or
            params["max_hover_agl_m"] != 300.0):
        raise ValueError("中继返航余量/周转/悬停上限与核对的数据版本不符")
    return {"params": params, "uav_ids": sorted(r["uav_id"] for r in uavs),
            "unit_ids": sorted(r["unit_id"] for r in units)}


def _relay_leg(arc: dict[str, float], params: dict[str, float]) -> tuple[float, float]:
    cruise_time = arc["distance_m"] / params["cruise_speed_mps"]
    duration = (arc["climb_m"] / params["climb_speed_mps"] + cruise_time +
                arc["descent_m"] / params["descent_speed_mps"])
    energy = (params["cruise_power_kw"] * cruise_time / 3600.0 +
              params["takeoff_mass_kg"] * 9.80665 * arc["climb_m"] /
              (3_600_000.0 * params["climb_efficiency"]))
    return duration, energy


def _candidate_xy(slices: list[TimeSlice], gateway: tuple[float, float, float],
                  spacing_m: int = 500) -> list[tuple[float, float]]:
    points: set[tuple[int, int]] = set()
    for item in slices:
        if item.direct_available:
            continue
        for fraction in (0.5, 0.75, 1.0):
            x = gateway[0] + fraction * (item.midpoint[0] - gateway[0])
            y = gateway[1] + fraction * (item.midpoint[1] - gateway[1])
            gx, gy = round(x / spacing_m), round(y / spacing_m)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    points.add((gx + dx, gy + dy))
    return [(float(x * spacing_m), float(y * spacing_m)) for x, y in sorted(points)]


def generate_candidates(dem: DemGrid, links: LinkEvaluator,
                        slices: list[TimeSlice], relay: dict[str, Any],
                        spacing_m: int = 500,
                        heights_agl_m: tuple[int, ...] = (30, 50, 100, 150, 200, 250, 300),
                        max_points: int = 5000,
                        extra_xy_points: Iterable[tuple[float, float]] = ()) -> tuple[list[RelayCandidate], dict[str, int]]:
    params = relay["params"]
    xy_points = sorted(set(_candidate_xy(slices, links.gateway, spacing_m)) |
                       set(extra_xy_points))
    if len(xy_points) > max_points:
        raise ValueError(f"候选网格点 {len(xy_points)} 超出显式预算 {max_points}；未声称不可行")
    candidates: list[RelayCandidate] = []
    counts: dict[str, int] = defaultdict(int)
    center = (links.gateway[0], links.gateway[1],
              links.gateway[2] - 20.0)
    service_power = params["hover_power_kw"] + params["comm_extra_power_kw"]
    for x, y in xy_points:
        try:
            ground = dem.terrain_at(x, y)
        except ValueError:
            counts["outside_or_invalid_dem"] += len(heights_agl_m)
            continue
        lon, lat = dem.to_lonlat.transform(x, y)
        for agl in heights_agl_m:
            if agl > params["max_hover_agl_m"] + EPS:
                counts["height_limit"] += 1
                continue
            hover = (x, y, ground + agl)
            # 先用无遮蔽最乐观链路淘汰明显超距的点。
            distance = math.dist(hover, links.gateway)
            clear_loss = path_loss_db(distance, links.frequency_mhz, False,
                                      links.obstruction_loss_db)
            back_limit = bidirectional_limit_db(links.radios["relay_backhaul"],
                                                 links.radios["gateway"],
                                                 links.system_loss_db)
            if clear_loss > back_limit + EPS:
                counts["backhaul_range"] += 1
                continue
            backhaul = links.backhaul(hover)
            if not backhaul.available:
                counts["backhaul"] += 1
                continue
            try:
                outward = dem.arc_geometry(center, hover)
                homeward = dem.arc_geometry(hover, center)
            except ValueError:
                counts["relay_arc"] += 1
                continue
            # PDF 的基线筛选口径，不是官方全局硬约束。
            if hover[2] > min(outward["cruise_alt_m"],
                              homeward["cruise_alt_m"]) + EPS:
                counts["pdf_cruise_screen"] += 1
                continue
            outward_s, outward_energy = _relay_leg(outward, params)
            homeward_s, homeward_energy = _relay_leg(homeward, params)
            travel_energy = outward_energy + homeward_energy
            max_energy = ((1.0 - params["return_soc_min"]) *
                          params["energy_use_kwh"])
            setup_energy = service_power * params["link_setup_s"] / 3600.0
            max_service = (max_energy - travel_energy - setup_energy) * 3600.0 / service_power
            if max_service <= 0:
                counts["energy"] += 1
                continue
            candidates.append(RelayCandidate(
                f"H{x:+.0f}_{y:+.0f}_{agl:03d}", x, y, hover[2], lon, lat,
                float(agl), outward_s, homeward_s, travel_energy,
                max_service, backhaul.margin_db,
            ))
    counts["grid_points"] = len(xy_points)
    counts["candidate_count"] = len(candidates)
    return candidates, dict(counts)


def fine_grid_near_uncovered(slices: list[TimeSlice], uncovered: list[int],
                             spacing_m: int = 200,
                             representative_count: int = 32) -> list[tuple[float, float]]:
    """Bounded 200 m local refinement around temporally spread coarse misses."""
    if not uncovered:
        return []
    ordered = sorted((slices[index] for index in uncovered),
                     key=lambda item: (item.start_s, item.sortie_id))
    count = min(representative_count, len(ordered))
    indices = {round(i * (len(ordered) - 1) / max(1, count - 1))
               for i in range(count)}
    points: set[tuple[float, float]] = set()
    for index in sorted(indices):
        item = ordered[index]
        gx = round(item.midpoint[0] / spacing_m)
        gy = round(item.midpoint[1] / spacing_m)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                points.add((float((gx + dx) * spacing_m),
                            float((gy + dy) * spacing_m)))
    return sorted(points)


@dataclass(frozen=True)
class RelayMission:
    mission_id: str
    uav_id: str
    unit_id: str
    candidate: RelayCandidate
    start_s: float
    ready_s: float
    service_end_s: float
    return_s: float
    energy_kwh: float


class CoverageCache:
    """Search-only midpoint coverage cache; never acts as interval evidence."""

    def __init__(self, links: LinkEvaluator, slices: list[TimeSlice]):
        self.links = links
        self.slices = {item.slice_id: item for item in slices}
        self._available: dict[tuple[str, int], bool] = {}
        self._clear_range_m = 1000.0 * 10.0 ** ((
            bidirectional_limit_db(links.radios["transport"],
                                   links.radios["relay_access"],
                                   links.system_loss_db) - 32.45 -
            20.0 * math.log10(links.frequency_mhz)) / 20.0)

    def access(self, candidate: RelayCandidate, item: TimeSlice) -> bool:
        key = (candidate.candidate_id, item.slice_id)
        if key not in self._available:
            if math.dist(candidate.hover, item.midpoint) > self._clear_range_m + EPS:
                self._available[key] = False
            else:
                self._available[key] = self.links.access(item.midpoint,
                                                          candidate.hover).available
        return self._available[key]

    def optimistic_gain(self, candidate: RelayCandidate,
                        items: list[TimeSlice], ready_s: float) -> float:
        end_s = ready_s + candidate.max_service_s
        x, y, z = candidate.hover
        radius2 = self._clear_range_m ** 2
        return sum(item.duration_s for item in items
                   if ready_s <= item.start_s + EPS and item.end_s <= end_s + EPS
                   and (item.midpoint[0] - x) ** 2 +
                       (item.midpoint[1] - y) ** 2 +
                       (item.midpoint[2] - z) ** 2 <= radius2)


def _resource_choice(candidate: RelayCandidate, first_s: float,
                     relay: dict[str, Any], uav_ready: dict[str, float],
                     unit_ready: dict[str, float]) -> tuple[str, str, float, float] | None:
    params = relay["params"]
    lead = params["prep_s"] + candidate.outward_s + params["link_setup_s"]
    latest_start = first_s - lead
    if latest_start < -EPS:
        return None
    pairs = [(max(uav_ready[uav], unit_ready[unit]), uav, unit)
             for uav in relay["uav_ids"] for unit in relay["unit_ids"]]
    available, uav_id, unit_id = min(pairs)
    if available > latest_start + EPS:
        return None
    start = max(0.0, available, latest_start)
    return uav_id, unit_id, start, start + lead


def schedule_relays(slices: list[TimeSlice], candidates: list[RelayCandidate],
                    links: LinkEvaluator, relay: dict[str, Any],
                    shortlist_size: int = 72) -> tuple[list[RelayMission],
                                                         dict[int, str],
                                                         list[int], dict[str, Any]]:
    """Cover midpoint-blind slices in time order with a bounded, deterministic greedy list.

    A finite-grid/shortlist failure is reported as a baseline miss, not proof of
    physical infeasibility. Interval verification is a separate mandatory step.
    """
    if shortlist_size < 1:
        raise ValueError("shortlist_size 必须为正")
    params = relay["params"]
    cache = CoverageCache(links, slices)
    blind = sorted((item for item in slices if not item.direct_available),
                   key=lambda item: (item.start_s, item.sortie_id, item.slice_id))
    pending = {item.slice_id for item in blind}
    uncovered: list[int] = []
    assigned: dict[int, str] = {}
    missions: list[RelayMission] = []
    uav_ready = {uav: 0.0 for uav in relay["uav_ids"]}
    unit_ready = {unit: 0.0 for unit in relay["unit_ids"]}
    counters: dict[str, int] = defaultdict(int)
    service_power = params["hover_power_kw"] + params["comm_extra_power_kw"]
    while pending:
        first = next(item for item in blind if item.slice_id in pending)
        remaining = [item for item in blind if item.slice_id in pending]
        ranked = []
        for candidate in candidates:
            choice = _resource_choice(candidate, first.start_s, relay,
                                      uav_ready, unit_ready)
            if choice is None:
                continue
            if not cache.access(candidate, first):
                continue
            optimistic = cache.optimistic_gain(candidate, remaining, choice[3])
            if optimistic < first.duration_s - EPS:
                continue
            ranked.append((optimistic, candidate.candidate_id, candidate, choice))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        counters["feasible_candidate_checks"] += len(ranked)
        best = None
        for _, _, candidate, choice in ranked[:shortlist_size]:
            uav_id, unit_id, start_s, ready_s = choice
            horizon = ready_s + candidate.max_service_s
            covered = [item for item in remaining
                       if ready_s <= item.start_s + EPS and
                       item.end_s <= horizon + EPS and cache.access(candidate, item)]
            if not covered or first.slice_id not in {item.slice_id for item in covered}:
                continue
            covered.sort(key=lambda item: (item.end_s, item.start_s, item.slice_id))
            prefix_gain = 0.0
            prefix_ids: set[int] = set()
            for prefix_index, item in enumerate(covered):
                prefix_gain += item.duration_s
                prefix_ids.add(item.slice_id)
                if first.slice_id not in prefix_ids:
                    continue
                end_s = item.end_s
                energy = (candidate.travel_energy_kwh +
                          service_power * (params["link_setup_s"] +
                                           end_s - ready_s) / 3600.0)
                if energy > (params["energy_use_kwh"] *
                             (1.0 - params["return_soc_min"]) + EPS):
                    raise AssertionError("调度计算出的中继架次超出返航能量约束")
                score = prefix_gain / max(energy, EPS)
                signature = (score, prefix_gain, -end_s - candidate.homeward_s,
                             -energy)
                if (best is None or signature > best[0] or
                        (signature == best[0] and
                         candidate.candidate_id < best[1].candidate_id)):
                    best = (signature, candidate, choice, end_s, energy,
                            covered[:prefix_index + 1])
        counters["exact_shortlisted_checks"] += min(len(ranked), shortlist_size)
        if best is None:
            uncovered.append(first.slice_id)
            pending.remove(first.slice_id)
            counters["baseline_uncovered_slices"] += 1
            continue
        _, candidate, (uav_id, unit_id, start_s, ready_s), end_s, energy, covered = best
        return_s = end_s + candidate.homeward_s
        mission_id = f"R{len(missions) + 1:03d}"
        missions.append(RelayMission(mission_id, uav_id, unit_id, candidate,
                                     start_s, ready_s, end_s, return_s, energy))
        uav_ready[uav_id] = return_s + params["turnaround_s"]
        soc = 1.0 - energy / params["energy_use_kwh"]
        unit_ready[unit_id] = return_s + charge_to_full_s(soc,
                                                           params["full_charge_s"])
        for item in covered:
            pending.remove(item.slice_id)
            assigned[item.slice_id] = mission_id
        counters["covered_midpoint_slices"] += len(covered)
    return missions, assigned, uncovered, {
        **counters, "midpoint_blind_slices": len(blind),
        "candidate_shortlist_size": shortlist_size,
        "uav_next_ready_s": uav_ready, "unit_next_ready_s": unit_ready,
        "cache_size": len(cache._available),
    }


def _format_float(value: float) -> str:
    return f"{value:.9f}"


def _format_time(value: float) -> str:
    return repr(float(value))


def relay_rows_from_missions(missions: list[RelayMission]) -> list[dict[str, str]]:
    rows = []
    for mission in missions:
        candidate = mission.candidate
        rows.append(dict(zip(RELAY_COLUMNS, (
            mission.mission_id, mission.uav_id, mission.unit_id,
            _format_time(mission.start_s), _format_float(candidate.lon_deg),
            _format_float(candidate.lat_deg), _format_float(candidate.alt_m),
            _format_time(mission.ready_s), _format_time(mission.service_end_s),
            _format_time(mission.return_s), _format_float(mission.energy_kwh),
        ))))
    return rows


def communication_rows(slices: list[TimeSlice],
                       assigned: dict[int, str],
                       official_bounds: Mapping[str, tuple[float, float]]) -> list[dict[str, str]]:
    """Emit every air-time slice, including honest '未覆盖' diagnostic rows."""
    rows: list[dict[str, Any]] = []
    for item in slices:
        mission_id = "" if item.direct_available else assigned.get(item.slice_id, "")
        method = ("直连" if item.direct_available else
                  "中继" if mission_id else "未覆盖")
        if (rows and rows[-1]["运输架次编号"] == item.sortie_id and
                rows[-1]["通信阶段"] == item.stage and
                rows[-1]["保障方式"] == method and
                rows[-1]["中继架次编号"] == mission_id and
                abs(rows[-1]["_end"] - item.start_s) <= 1e-6):
            rows[-1]["_end"] = item.end_s
        else:
            rows.append({"运输架次编号": item.sortie_id, "通信阶段": item.stage,
                         "_start": item.start_s, "_end": item.end_s,
                         "保障方式": method, "中继架次编号": mission_id})
    last_row_by_sortie = {row["运输架次编号"]: row for row in rows}
    for sortie_id, row in last_row_by_sortie.items():
        _, official_return = official_bounds[sortie_id]
        if abs(row["_end"] - official_return) > 2e-5:
            raise ValueError(f"{sortie_id} 重建返航时间与 Q2 官方表不一致")
        row["_end"] = official_return
    for row in rows:
        row["开始时刻（s）"] = _format_time(row.pop("_start"))
        row["结束时刻（s）"] = _format_time(row.pop("_end"))
    return rows


def _scene(data_run: Path, q2_dir: Path) -> tuple[dict[str, Any], DemGrid,
                                                 LinkEvaluator,
                                                 dict[str, list[TrajectorySegment]],
                                                 dict[str, Any]]:
    _, trajectories, source_meta = replay_q2(data_run, q2_dir)
    dem_meta = json.loads((data_run / "meta" / "dem_meta.json").read_text(
        encoding="utf-8"))
    dem_source = data_run.parent.parent.parent / Path(dem_meta["source_mat"])
    source_meta["dem_source_sha256"] = sha256(dem_source)
    dem = load_dem(data_run)
    positions = _node_positions(data_run)
    radios, propagation = load_comm(data_run)
    o01 = positions["O01"]
    gateway = (o01[0], o01[1],
               o01[2] + propagation["gateway_agl_m"])
    links = LinkEvaluator(dem, gateway, radios,
                          propagation["frequency_mhz"],
                          propagation["system_loss_db"],
                          propagation["obstruction_loss_db"])
    relay = load_relay_resources(data_run)
    return relay, dem, links, trajectories, source_meta


def _copy_transport_inputs(q2_dir: Path, table_dir: Path) -> None:
    for q2_name, q3_name in (("Q2_运输架次.csv", "Q3_运输架次.csv"),
                             ("Q2_逐箱交付.csv", "Q3_逐箱交付.csv")):
        shutil.copy2(q2_dir / q2_name, table_dir / q3_name)


def _save_diagnostics(table_dir: Path, slices: list[TimeSlice],
                      uncovered: list[int], missions: list[RelayMission],
                      relay: dict[str, Any]) -> None:
    blind_rows = []
    for item in slices:
        if not item.direct_available:
            blind_rows.append({
                "slice_id": item.slice_id, "sortie_id": item.sortie_id,
                "stage": item.stage, "start_s": _format_float(item.start_s),
                "end_s": _format_float(item.end_s),
                "direct_margin_db": _format_float(item.direct_margin_db),
                "midpoint_status": "UNCOVERED" if item.slice_id in uncovered else "ASSIGNED",
            })
    write_rows(table_dir / "Q3_直连盲段采样.csv", blind_rows,
               ["slice_id", "sortie_id", "stage", "start_s", "end_s",
                "direct_margin_db", "midpoint_status"])
    resource_rows = []
    for mission in missions:
        soc = 1.0 - mission.energy_kwh / relay["params"]["energy_use_kwh"]
        resource_rows.append({
            "mission_id": mission.mission_id, "uav_id": mission.uav_id,
            "energy_unit_id": mission.unit_id,
            "start_s": _format_float(mission.start_s),
            "link_ready_s": _format_float(mission.ready_s),
            "service_end_s": _format_float(mission.service_end_s),
            "return_s": _format_float(mission.return_s),
            "uav_next_ready_s": _format_float(mission.return_s +
                         relay["params"]["turnaround_s"]),
            "unit_next_ready_s": _format_float(mission.return_s +
                         charge_to_full_s(soc, relay["params"]["full_charge_s"])),
            "return_soc": _format_float(soc),
        })
    write_rows(table_dir / "Q3_中继资源时间线.csv", resource_rows,
               ["mission_id", "uav_id", "energy_unit_id", "start_s",
                "link_ready_s", "service_end_s", "return_s", "uav_next_ready_s",
                "unit_next_ready_s", "return_soc"])


def _save_figures(figure_dir: Path, slices: list[TimeSlice],
                  assigned: dict[int, str], missions: list[RelayMission]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure_dir.mkdir(parents=True, exist_ok=False)
    sortie_ids = sorted({item.sortie_id for item in slices})
    colors = {"direct": "#2b8cbe", "relay": "#41ab5d", "uncovered": "#de2d26"}
    fig, ax = plt.subplots(figsize=(15, max(6, 0.35 * len(sortie_ids) + 2)))
    for item in slices:
        index = sortie_ids.index(item.sortie_id)
        kind = ("direct" if item.direct_available else
                "relay" if item.slice_id in assigned else "uncovered")
        ax.broken_barh([(item.start_s, item.duration_s)],
                       (index - 0.34, 0.68), facecolors=colors[kind], linewidth=0)
    ax.set_yticks(range(len(sortie_ids)), sortie_ids)
    ax.set_xlabel("Time since dispatch / s")
    ax.set_ylabel("Transport sortie")
    ax.set_title("Q3 fixed-Q2 baseline: sampled communication state")
    ax.legend(handles=[Patch(facecolor=colors[key], label=key)
                       for key in ("direct", "relay", "uncovered")], loc="upper right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q3_通信状态时序.{extension}", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    uavs = sorted({mission.uav_id for mission in missions})
    for mission in missions:
        index = uavs.index(mission.uav_id)
        ax.broken_barh([(mission.start_s, mission.ready_s - mission.start_s)],
                       (index - 0.3, 0.6), facecolors="#9ecae1")
        ax.broken_barh([(mission.ready_s,
                         mission.service_end_s - mission.ready_s)],
                       (index - 0.3, 0.6), facecolors="#41ab5d")
        ax.broken_barh([(mission.service_end_s,
                         mission.return_s - mission.service_end_s)],
                       (index - 0.3, 0.6), facecolors="#9ecae1")
        ax.text(mission.ready_s, index + 0.12, mission.mission_id, fontsize=8)
    ax.set_yticks(range(len(uavs)), uavs)
    ax.set_xlabel("Time since dispatch / s")
    ax.set_title("Q3 relay aircraft occupation and service")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q3_中继资源占用.{extension}", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    blind = [item for item in slices if not item.direct_available]
    ax.scatter([item.midpoint[0] for item in blind],
               [item.midpoint[1] for item in blind], s=6, alpha=0.25,
               color="#de2d26", label="Direct-link blind samples")
    for mission in missions:
        ax.scatter(mission.candidate.x_m, mission.candidate.y_m,
                   s=55, marker="^", color="#238b45")
        ax.annotate(mission.mission_id,
                    (mission.candidate.x_m, mission.candidate.y_m), fontsize=8)
    ax.set_xlabel("UTM 49N Easting / m")
    ax.set_ylabel("UTM 49N Northing / m")
    ax.set_title("Q3 relay hover positions and sampled blind points")
    ax.axis("equal")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"Q3_中继悬停位置.{extension}", dpi=200)
    plt.close(fig)


def validate_official_tables(table_dir: Path, data_run: Path, q2_dir: Path,
                             trajectories: dict[str, list[TrajectorySegment]],
                             links: LinkEvaluator, dem: DemGrid) -> dict[str, Any]:
    """Reopen four exported files and validate without the search/schedule cache."""
    from q3_2_validate import verify_continuous_coverage
    from q3_3_resources import validate_relay_rows

    source_map = (("Q2_运输架次.csv", "Q3_运输架次.csv"),
                  ("Q2_逐箱交付.csv", "Q3_逐箱交付.csv"))
    copy_checks = {}
    for q2_name, q3_name in source_map:
        copy_checks[q3_name] = sha256(q2_dir / q2_name) == sha256(table_dir / q3_name)
    relay_rows = read_rows(table_dir / "Q3_中继架次.csv")
    comm_rows = read_rows(table_dir / "Q3_通信保障.csv")
    comm = verify_continuous_coverage(trajectories, links, comm_rows, relay_rows)
    resource = validate_relay_rows(relay_rows, data_run, links, dem)
    diagnostic_uncovered = sum(row["保障方式"] == "未覆盖" for row in comm_rows)
    if not all(copy_checks.values()) or resource["status"] != "PASS":
        status = "ERROR"
    elif diagnostic_uncovered or comm["status"] == "FAIL":
        status = "BASELINE_UNCOVERED"
    elif comm["status"] == "UNVERIFIED":
        status = "UNVERIFIED"
    else:
        status = "PASS"
    return {
        "status": status, "source_copies_identical": copy_checks,
        "diagnostic_uncovered_rows": diagnostic_uncovered,
        "communication": comm, "relay_resources": resource,
    }


def _save_verification(table_dir: Path, validation: dict[str, Any]) -> None:
    comm = validation["communication"]
    resource = validation["relay_resources"]
    interval_columns = ["sortie_id", "stage", "start_s", "end_s", "mode",
                        "relay_id", "comm_row_index", "status", "method",
                        "margin_db", "worst_distance_m",
                        "pessimistic_obstructed", "witness_s", "reason"]
    write_rows(table_dir / "Q3_逐区间验收.csv", comm["checks"], interval_columns)
    checks = [{"check_id": "Q2_FIXED_COPY", "status":
               "PASS" if all(validation["source_copies_identical"].values()) else "FAIL",
               "detail": "Q3 两张运输补表与已验收 Q2 文件逐字节相同"},
              {"check_id": "RELAY_RESOURCE", "status": resource["status"],
               "detail": resource.get("first_failure", "") or "中继物理与资源独立复核"},
              {"check_id": "COMM_CONTINUOUS", "status": comm["status"],
               "detail": (f"PASS={comm['certified_interval_count']}, "
                          f"FAIL={comm['failed_interval_count']}, "
                          f"UNVERIFIED={comm['unverified_interval_count']}")},
              {"check_id": "BASELINE_STATUS", "status": validation["status"],
               "detail": f"未覆盖诊断行={validation['diagnostic_uncovered_rows']}"}]
    write_rows(table_dir / "Q3_验收检查.csv", checks,
               ["check_id", "status", "detail"])
    json_dump(table_dir / "Q3_通信问题.json", {
        "issues": comm["issues"],
        "failed_or_unverified_intervals": [item for item in comm["checks"]
                                           if item["status"] != "PASS"],
        "resource_issues": resource.get("issues", []),
    })


def run_baseline(data_run: Path, q2_dir: Path, output_root: Path,
                 sample_step_s: float, shortlist_size: int) -> dict[str, Any]:
    relay, dem, links, trajectories, source_meta = _scene(data_run, q2_dir)
    slices = scan_direct(trajectories, links, sample_step_s)
    candidates, candidate_stats = generate_candidates(dem, links, slices, relay)
    missions, assigned, uncovered, schedule_stats = schedule_relays(
        slices, candidates, links, relay, shortlist_size=shortlist_size)
    coarse_uncovered_count = len(uncovered)
    fine_points = fine_grid_near_uncovered(slices, uncovered)
    if fine_points:
        refined_candidates, refined_stats = generate_candidates(
            dem, links, slices, relay, extra_xy_points=fine_points)
        refined_missions, refined_assigned, refined_uncovered, refined_schedule = (
            schedule_relays(slices, refined_candidates, links, relay,
                            shortlist_size=shortlist_size))
        if len(refined_uncovered) < len(uncovered):
            missions, assigned, uncovered = (refined_missions, refined_assigned,
                                             refined_uncovered)
            candidate_stats, schedule_stats = refined_stats, refined_schedule
            selected_search = "local_200m_refinement"
        else:
            selected_search = "coarse_500m"
    else:
        refined_stats = {}
        selected_search = "coarse_500m"
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    table_dir = output_root / f"q3_base_{timestamp}_table"
    figure_dir = output_root / f"q3_base_{timestamp}_figure"
    table_dir.mkdir(parents=True, exist_ok=False)
    _copy_transport_inputs(q2_dir, table_dir)
    write_rows(table_dir / "Q3_中继架次.csv", relay_rows_from_missions(missions),
               RELAY_COLUMNS)
    official_bounds = {
        row["架次编号"]: (float(row["开始时刻（s）"]),
                         float(row["返回O01时刻（s）"]))
        for row in read_rows(table_dir / "Q3_运输架次.csv")
    }
    write_rows(table_dir / "Q3_通信保障.csv",
               communication_rows(slices, assigned, official_bounds),
               COMM_COLUMNS)
    _save_diagnostics(table_dir, slices, uncovered, missions, relay)
    json_dump(table_dir / "Q3_直连盲区汇总.json", blind_intervals(slices))
    json_dump(table_dir / "Q3_候选筛选统计.json", candidate_stats)
    validation = validate_official_tables(table_dir, data_run, q2_dir,
                                          trajectories, links, dem)
    _save_verification(table_dir, validation)
    _save_figures(figure_dir, slices, assigned, missions)
    comm_report = validation["communication"]
    resource_report = validation["relay_resources"]
    summary = {
        "status": validation["status"],
        "interpretation": ("仅固定 Q2 方案、当前有限网格及贪心基线的结果；"
                           "失败不证明第三问整体无可行解"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source_meta,
        "sample_step_s": sample_step_s,
        "direct_blind_sample_count": sum(not item.direct_available for item in slices),
        "direct_blind_sample_duration_s": sum(item.duration_s for item in slices
                                               if not item.direct_available),
        "direct_blind_interval_count": len(blind_intervals(slices)),
        "relay_mission_count": len(missions),
        "relay_energy_kwh": sum(mission.energy_kwh for mission in missions),
        "midpoint_uncovered_slice_count": len(uncovered),
        "midpoint_uncovered_duration_s": sum(slices[i].duration_s for i in uncovered),
        "candidate_stats": candidate_stats,
        "search_comparison": {
            "selected": selected_search,
            "coarse_uncovered_slices": coarse_uncovered_count,
            "fine_point_count": len(fine_points),
            "refined_candidate_stats": refined_stats,
            "refined_uncovered_slices": (len(refined_uncovered)
                                          if fine_points else None),
        },
        "schedule_stats": schedule_stats,
        "verification": {
            "source_copies_identical": validation["source_copies_identical"],
            "diagnostic_uncovered_rows": validation["diagnostic_uncovered_rows"],
            "communication_status": comm_report["status"],
            "certified_interval_count": comm_report["certified_interval_count"],
            "failed_interval_count": comm_report["failed_interval_count"],
            "unverified_interval_count": comm_report["unverified_interval_count"],
            "worst_certified_margin_db": comm_report["worst_certified_margin_db"],
            "first_comm_issue": comm_report["issues"][0] if comm_report["issues"] else None,
            "relay_resource_status": resource_report["status"],
            "first_resource_failure": resource_report.get("first_failure"),
        },
        "table_dir": str(table_dir.resolve()),
        "figure_dir": str(figure_dir.resolve()),
    }
    json_dump(table_dir / "Q3_运行摘要.json", summary)
    if summary["status"] == "PASS":
        (table_dir / "Q3_READY.txt").write_text(
            "Q3 baseline independently verified PASS\n"
            f"source_manifest_sha256={source_meta['source_manifest_sha256']}\n",
            encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-run", type=Path, default=None,
                        help="0_outputs 中同 Q2 清单哈希的 PASS 目录")
    parser.add_argument("--q2-dir", type=Path, default=DEFAULT_Q2_DIR,
                        help="已验收 Q2 方案 5 的 table 目录")
    parser.add_argument("--output-root", type=Path,
                        default=CODE_DIR / "3_outputs" / "0_baseline")
    parser.add_argument("--sample-step-s", type=float, default=20.0)
    parser.add_argument("--shortlist-size", type=int, default=72)
    parser.add_argument("--validate-only", type=Path, default=None,
                        metavar="Q3_TABLE_DIR", help="只读重验指定的四张已导出表")
    args = parser.parse_args(argv)
    data_run = (args.data_run.resolve() if args.data_run is not None else
                find_data_run(CODE_DIR, EXPECTED_MANIFEST))
    q2_dir = args.q2_dir.resolve()
    if args.validate_only is not None:
        table_dir = args.validate_only.resolve()
        _, dem, links, trajectories, _ = _scene(data_run, q2_dir)
        report = validate_official_tables(table_dir, data_run, q2_dir,
                                          trajectories, links, dem)
        print(json.dumps({
            "status": report["status"],
            "communication_status": report["communication"]["status"],
            "relay_resource_status": report["relay_resources"]["status"],
            "uncovered_rows": report["diagnostic_uncovered_rows"],
            "ready_marker_present": (table_dir / "Q3_READY.txt").is_file(),
        }, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "PASS" else 2
    summary = run_baseline(data_run, q2_dir, args.output_root.resolve(),
                           args.sample_step_s, args.shortlist_size)
    print(json.dumps({key: summary[key] for key in (
        "status", "relay_mission_count", "midpoint_uncovered_slice_count",
        "table_dir", "figure_dir")}, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
