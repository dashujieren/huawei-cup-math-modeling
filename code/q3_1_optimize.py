"""问题三方案 1：互补中继选点与硬时限优先的运输—通信联合排程。

从原始 80 箱组批，先联合安排硬时限货物和中继，再插入其余整批货物。
全程考虑 8 架运输机、14 组电池、2 架中继机及 6 组能源组件的周转。
站点筛选保留两站互补覆盖；通信分段保留站点切换边界。
有限路线/站点与时间预算不证明全局最优；独立复核通过才写 Q3_READY.txt。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import scipy.io
from scipy.spatial import cKDTree
from pyproj import Transformer

from _0_pipeline import _line_cells, verify_ready
from q2_0_baseline import charge_to_full_s
from q2_1_optimize import Context, check_plan


CODE_DIR = Path(__file__).resolve().parent
EXPECTED_MANIFEST = "dab98bafa9b2101b0b6265be77ec45e95baecd5646eadf0c6acb905f3e45ca5e"
EPS = 1e-7


class ProgressBar:
    """显示可计数任务的真实完成量；搜索器另显示已用时间预算。"""

    def __init__(self, label: str, total: float):
        self.label = label
        self.total = max(float(total), 1.0)
        self.tty = sys.stdout.isatty()
        self.last_print = 0.0
        self.last_bucket = -1
        self.current = 0.0
        self.update(0, force=True)

    def update(self, done: float, *, force: bool = False) -> None:
        self.current = max(0.0, min(float(done), self.total))
        now = time.monotonic()
        ratio = self.current / self.total
        bucket = min(10, int(ratio * 10))
        if not force and self.current < self.total:
            if self.tty and now - self.last_print < 0.5:
                return
            if not self.tty and bucket <= self.last_bucket and now - self.last_print < 10:
                return
        filled = int(ratio * 24)
        message = (f"Q3 [{self.label}] [{'#' * filled}{'.' * (24 - filled)}] "
                   f"{ratio:5.0%} ({self.current:.0f}/{self.total:.0f})")
        print(("\r" if self.tty else "") + message,
              end="" if self.tty else "\n", flush=True)
        self.last_print, self.last_bucket = now, bucket

    def close(self, *, completed: bool = True) -> None:
        if completed or self.tty:
            self.update(self.total if completed else self.current, force=True)
        if self.tty:
            print(flush=True)


class SolverBudgetBar:
    """CP-SAT 无可信的解空间百分比，因此只展示已用墙钟预算。"""

    def __init__(self, label: str, seconds: float):
        self.bar = ProgressBar(f"{label}：时间预算", seconds)
        self.started = time.monotonic()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._refresh, daemon=True)

    def _refresh(self) -> None:
        while not self.stopped.wait(0.5):
            self.bar.update(time.monotonic() - self.started)

    def __enter__(self) -> "SolverBudgetBar":
        self.thread.start()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.stopped.set()
        self.thread.join()
        self.bar.update(time.monotonic() - self.started, force=True)
        if self.bar.tty:
            print(flush=True)

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
        # 起终点高于沿线地形+50m 时，巡航面必须至少到达起终点高度。
        cruise = max(max(heights) + 50.0, a[2], b[2])
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
        raise FileNotFoundError("未找到与第三问输入哈希一致的 PASS 数据运行目录")
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
                        home: tuple[float, float, float],
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
    center = home
    service_power = params["hover_power_kw"] + params["comm_extra_power_kw"]
    progress = ProgressBar("物理站点认证", len(xy_points))
    for number, (x, y) in enumerate(xy_points, 1):
        progress.update(number - 1)
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
    progress.close()
    counts["grid_points"] = len(xy_points)
    counts["candidate_count"] = len(candidates)
    return candidates, dict(counts)


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
    """每个经证明的短区间单独成行，避免合并后失去连续链路证书。"""
    rows: list[dict[str, Any]] = []
    for item in slices:
        mission_id = "" if item.direct_available else assigned.get(item.slice_id, "")
        method = ("直连" if item.direct_available else
                  "中继" if mission_id else "未覆盖")
        rows.append({"运输架次编号": item.sortie_id, "通信阶段": item.stage,
                     "_start": item.start_s, "_end": item.end_s,
                     "保障方式": method, "中继架次编号": mission_id})
    last_row_by_sortie = {row["运输架次编号"]: row for row in rows}
    for sortie_id, row in last_row_by_sortie.items():
        _, official_return = official_bounds[sortie_id]
        if abs(row["_end"] - official_return) > 2e-5:
            raise ValueError(f"{sortie_id} 重建返航时间与 Q3 运输表不一致")
        row["_end"] = official_return
    for row in rows:
        row["开始时刻（s）"] = _format_time(row.pop("_start"))
        row["结束时刻（s）"] = _format_time(row.pop("_end"))
    return rows


def _scene(data_run: Path) -> tuple[dict[str, Any], DemGrid,
                                    LinkEvaluator, dict[str, Any]]:
    if not verify_ready(data_run):
        raise ValueError(f"数据目录未通过 0_outputs 验收：{data_run}")
    ready_lines = (data_run / "meta" / "READY.txt").read_text(
        encoding="utf-8-sig").splitlines()
    manifest = next((line.split("=", 1)[1].strip() for line in ready_lines
                     if line.startswith("manifest_sha256=")), None)
    if manifest != EXPECTED_MANIFEST:
        raise ValueError("第三问原始数据清单哈希与核对版本不符")
    source_meta = {"data_run": str(data_run.resolve()),
                   "source_manifest_sha256": manifest}
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
    return relay, dem, links, source_meta


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
                  assigned: dict[int, str], missions: list[RelayMission],
                  scheme_label: str = "Q3 scheme 1") -> None:
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
    ax.set_title(f"{scheme_label}: certified communication intervals")
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


@dataclass(frozen=True)
class RouteTask:
    batch_id: str
    route: Any


@dataclass
class JointState:
    sorties: list[dict[str, Any]]
    deliveries: list[dict[str, Any]]
    trajectories: dict[str, list[TrajectorySegment]]
    slices: list[TimeSlice]
    assigned: dict[int, str]
    missions: list[RelayMission]


OBJECTIVE_KEYS = ("soft_weighted_lateness_s", "joint_makespan_s",
                  "total_energy_kwh", "total_sorties")


def _objective_metrics(state: JointState) -> dict[str, float]:
    """按最终连续时刻和实际能耗计算题面四项指标。"""
    return {
        "soft_weighted_lateness_s": sum(
            float(d["priority"]) * float(d["soft_lateness_s"])
            for d in state.deliveries if d["hard_deadline_s"] is None),
        "joint_makespan_s": max(
            [float(s["return_s"]) for s in state.sorties] +
            [float(m.return_s) for m in state.missions], default=0.0),
        "total_energy_kwh": sum(float(s["energy_kwh"]) for s in state.sorties) +
                            sum(float(m.energy_kwh) for m in state.missions),
        "total_sorties": len(state.sorties) + len(state.missions),
    }


def equal_weight_ratios(metrics: Mapping[str, float],
                        reference: Mapping[str, float]) -> dict[str, float]:
    """逐项基线归一化；零基线项保留基线分值 1。"""
    if any(float(reference[key]) < 0 for key in OBJECTIVE_KEYS):
        raise ValueError("基线四项指标不可为负")
    return {key: (float(metrics[key]) / float(reference[key])
                  if float(reference[key]) > 0 else 1.0 + float(metrics[key]))
            for key in OBJECTIVE_KEYS}


def equal_weight_score(metrics: Mapping[str, float],
                       reference: Mapping[str, float]) -> float:
    """四项按用户选定的等权规则求平均，PASS 基线得分为 1。"""
    return sum(equal_weight_ratios(metrics, reference).values()) / len(OBJECTIVE_KEYS)


def _equal_weight_coefficients(reference: Mapping[str, float]) -> dict[str, int]:
    """CP-SAT 整数近似目标；最终接纳仍使用实际浮点指标。"""
    units = {"soft_weighted_lateness_s": 1.0,
             "joint_makespan_s": 1.0,
             "total_energy_kwh": 1000.0,  # 模型能量项单位为 Wh
             "total_sorties": 1.0}
    scale = 1_000_000_000
    return {key: max(1, round(scale / (max(float(reference[key]), 1.0) * units[key])))
            for key in OBJECTIVE_KEYS}


@dataclass
class RouteProfile:
    """路线的通信几何只计算一次，排程仅平移时间。"""
    segments: list[TrajectorySegment]
    slices: list[TimeSlice]
    ranked_sites: dict[int, tuple[int, ...]]
    support: dict[int, set[int]]
    uncovered_slices: list[int]


def _empty_joint_state() -> JointState:
    return JointState([], [], {}, [], {}, [])


def _latest_hard_start(ctx: Context, route: Any, option: dict[str, Any]) -> float:
    return min((float(ctx.boxes[bid]["hard_deadline_s"]) -
                float(option["completion_offsets"][bid])
                for bid in route.box_ids
                if not pd.isna(ctx.boxes[bid]["hard_deadline_s"])),
               default=math.inf)


def _relative_trajectory(ctx: Context, route: Any, option: dict[str, Any],
                         positions: dict[str, tuple[float, float, float]]) -> list[TrajectorySegment]:
    segments: list[TrajectorySegment] = []
    model = ctx.models[option["type_id"]]
    for index, leg in enumerate(option["legs"]):
        origin, destination = leg["from_id"], leg["to_id"]
        arc = ctx.arcs[(origin, destination)]
        arrival = float(leg["arrival_offset_s"])
        begin = arrival - float(leg["flight_s"])
        a, b = positions[origin], positions[destination]
        cruise_alt = float(arc["cruise_alt_m"])
        climb_end = begin + float(arc["climb_m"]) / float(model["climb_speed_mps"])
        cruise_end = climb_end + float(arc["distance_m"]) / float(model["cruise_speed_mps"])
        _add_segment(segments, "爬升", begin, climb_end, a,
                     (a[0], a[1], cruise_alt))
        _add_segment(segments, "巡航", climb_end, cruise_end,
                     (a[0], a[1], cruise_alt), (b[0], b[1], cruise_alt))
        _add_segment(segments, "下降", cruise_end, arrival,
                     (b[0], b[1], cruise_alt), b)
        if destination != "O01":
            handoff_end = float(option["visits"][index]["handoff_end_offset_s"])
            _add_segment(segments, "投送", arrival, handoff_end, b, b)
    if not segments:
        raise ValueError("运输路线没有可复核的飞行轨迹")
    for left, right in zip(segments, segments[1:]):
        if abs(left.end_s - right.start_s) > 2e-5:
            raise ValueError("重建的运输飞行/投送轨迹不连续")
    return segments


def _certified_slices(sortie_id: str, segments: list[TrajectorySegment],
                      links: LinkEvaluator, step_s: float) -> list[TimeSlice]:
    if not math.isfinite(step_s) or step_s <= 0:
        raise ValueError("通信分段长度必须为正")
    slices: list[TimeSlice] = []
    for segment in segments:
        parts = max(1, math.ceil((segment.end_s - segment.start_s) / step_s))
        edges = [segment.start_s + (segment.end_s - segment.start_s) * i / parts
                 for i in range(parts + 1)]
        edges[0], edges[-1] = segment.start_s, segment.end_s
        for begin, end in zip(edges, edges[1:]):
            if end <= begin:
                continue
            certificate = certify_link_interval(
                segment, begin, end, links.gateway, "transport", "gateway", links)
            midpoint = segment.position((begin + end) / 2.0)
            slices.append(TimeSlice(len(slices), sortie_id, segment.stage,
                                    begin, end, midpoint,
                                    float(certificate["margin_db"])
                                    if certificate["margin_db"] is not None else -math.inf,
                                    certificate["status"] == "PASS", segment))
    return slices


def _shift_template(segments: list[TrajectorySegment], slices: list[TimeSlice],
                    start_s: float, slice_base: int) -> tuple[list[TrajectorySegment], list[TimeSlice]]:
    shifted_segments = [replace(segment, start_s=segment.start_s + start_s,
                                end_s=segment.end_s + start_s)
                        for segment in segments]
    lookup = {id(before): after for before, after in zip(segments, shifted_segments)}
    shifted_slices = [
        replace(item, slice_id=slice_base + index,
                start_s=item.start_s + start_s, end_s=item.end_s + start_s,
                segment=lookup[id(item.segment)])
        for index, item in enumerate(slices)
    ]
    return shifted_segments, shifted_slices


def _access_certified(candidate: RelayCandidate, item: TimeSlice,
                      links: LinkEvaluator, cache: dict[tuple[Any, ...], bool]) -> bool:
    endpoints = (item.segment.position(item.start_s),
                 item.segment.position(item.end_s))
    key = (candidate.candidate_id,
           *(tuple(round(value, 6) for value in point) for point in endpoints))
    if key not in cache:
        budget = bidirectional_limit_db(links.radios["transport"],
                                        links.radios["relay_access"],
                                        links.system_loss_db)
        max_distance = max(math.dist(point, candidate.hover) for point in endpoints)
        # 足够近时，即使整段都有遮挡也满足双向链路预算。
        if max_distance <= 0:
            cache[key] = True
        elif path_loss_db(max_distance, links.frequency_mhz, False,
                          links.obstruction_loss_db) > budget:
            cache[key] = False
        elif path_loss_db(max_distance, links.frequency_mhz, True,
                        links.obstruction_loss_db) < budget - 1e-6:
            cache[key] = True
        else:
            cache[key] = (certify_link_interval(
                item.segment, item.start_s, item.end_s, candidate.hover,
                "transport", "relay_access", links)["status"] == "PASS")
    return cache[key]




def _make_sortie(ctx: Context, seed: RouteTask, kind: str, option: dict[str, Any],
                 uav_id: str, battery: Any, start: float) -> tuple[dict[str, Any],
                                                                    list[dict[str, Any]]]:
    end = start + float(option["duration_s"])
    charge_s = charge_to_full_s(float(option["return_soc"]), battery.full_charge_s)
    sortie = {
        "batch_id": seed.batch_id, "type_id": kind, "uav_id": uav_id,
        "battery_id": battery.battery_id, "route": seed.route,
        "visit_order": ">".join(seed.route.zones), "box_ids": list(seed.route.box_ids),
        "start_s": start, "launch_s": start + float(option["launch_offset_s"]),
        "return_s": end, "charge_start_s": end, "charge_end_s": end + charge_s,
        "charge_s": charge_s, "mass_kg": option["mass_kg"],
        "volume_m3": option["volume_m3"], "energy_kwh": option["energy_kwh"],
        "return_soc": option["return_soc"], "legs": option["legs"],
        "visits": option["visits"],
    }
    deliveries = []
    for box_id in seed.route.box_ids:
        box = ctx.boxes[box_id]
        finish = start + float(option["completion_offsets"][box_id])
        hard = None if pd.isna(box["hard_deadline_s"]) else float(box["hard_deadline_s"])
        expected = float(box["expected_s"])
        deliveries.append({
            "box_id": box_id, "batch_id": seed.batch_id,
            "zone_id": str(box["zone_id"]), "sequence": option["sequence"][box_id],
            "complete_s": finish, "hard_deadline_s": hard,
            "expected_s": expected, "priority": float(box["priority"]),
            "hard_slack_s": None if hard is None else hard - finish,
            "soft_lateness_s": max(0.0, finish - expected),
        })
    return sortie, deliveries


def validate_official_tables(table_dir: Path, data_run: Path,
                             links: LinkEvaluator, dem: DemGrid) -> dict[str, Any]:
    """从已写出的 Q3 表重新构建运输轨迹，再验收通信与中继资源。"""

    ctx = Context(data_run)
    positions = _node_positions(data_run)
    transport_issues: list[str] = []
    plan = {"sorties": [], "deliveries": []}
    trajectories: dict[str, list[TrajectorySegment]] = {}
    detail_rows = read_rows(table_dir / "Q3_运输明细.csv")
    official_sorties = {row["架次编号"]: row
                        for row in read_rows(table_dir / "Q3_运输架次.csv")}
    official_deliveries = {row["货箱编号"]: row
                           for row in read_rows(table_dir / "Q3_逐箱交付.csv")}
    batteries = {b.battery_id: b for b in ctx.batteries}
    if len(detail_rows) != len(official_sorties):
        transport_issues.append("运输明细与官方架次数不同")
    for raw in detail_rows:
        try:
            batch_id = raw["batch_id"]
            ids = raw["box_ids"].split(";")
            zones = raw["visit_order"].split(">")
            route = ctx.route([(zone, [bid for bid in ids
                                        if ctx.boxes[bid]["zone_id"] == zone])
                               for zone in zones])
            if tuple(ids) != route.box_ids or batch_id in trajectories:
                raise ValueError("重复架次或箱序不符")
            kind = raw["type_id"]
            option = ctx.evaluate(route, kind)
            if option is None:
                raise ValueError("载荷、体积或能量不可行")
            seed = RouteTask(batch_id, route)
            sortie, deliveries = _make_sortie(
                ctx, seed, kind, option, raw["uav_id"],
                batteries[raw["battery_id"]], float(raw["start_s"]))
            for key in ("launch_s", "return_s", "charge_start_s", "charge_end_s",
                        "charge_s", "mass_kg", "volume_m3", "energy_kwh", "return_soc"):
                _close(float(raw[key]), float(sortie[key]), f"{batch_id}:{key}")
            official = official_sorties[batch_id]
            for key, value in (
                    ("无人机编号", sortie["uav_id"]), ("机型编号", kind),
                    ("电池编号", sortie["battery_id"]),
                    ("访问服务区顺序", sortie["visit_order"])):
                if official[key] != value:
                    raise ValueError(f"官方运输表 {key} 与明细不符")
            for key, value in (
                    ("开始时刻（s）", sortie["start_s"]),
                    ("返回O01时刻（s）", sortie["return_s"]),
                    ("架次能耗（kWh）", sortie["energy_kwh"])):
                _close(float(official[key]), float(value), f"{batch_id}:{key}")
            for delivery in deliveries:
                row = official_deliveries[delivery["box_id"]]
                if (row["架次编号"] != batch_id or
                        row["服务区编号"] != delivery["zone_id"]):
                    raise ValueError(f"{delivery['box_id']} 交付关系不一致")
                _close(float(row["交付完成时刻（s）"]),
                       delivery["complete_s"], delivery["box_id"])
            relative = _relative_trajectory(ctx, route, option, positions)
            trajectories[batch_id] = [
                replace(segment, start_s=segment.start_s + sortie["start_s"],
                        end_s=segment.end_s + sortie["start_s"])
                for segment in relative]
            plan["sorties"].append(sortie)
            plan["deliveries"].extend(deliveries)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            transport_issues.append(f"{raw.get('batch_id', '?')}: {exc}")
    if (len(official_deliveries) != 80 or len(plan["deliveries"]) != 80 or
            len({d["box_id"] for d in plan["deliveries"]}) != 80):
        transport_issues.append("逐箱交付未完整覆盖 80 箱")
    if len(official_sorties) != len(trajectories):
        transport_issues.append("运输架次缺失或重复")
    if not transport_issues:
        transport_issues.extend(
            f"{item['check_id']}: {item['detail']}"
            for item in check_plan(ctx, plan) if item["status"] != "PASS")
    relay_rows = read_rows(table_dir / "Q3_中继架次.csv")
    comm_rows = read_rows(table_dir / "Q3_通信保障.csv")
    comm = verify_continuous_coverage(trajectories, links, comm_rows, relay_rows)
    resource = validate_relay_rows(relay_rows, data_run, links, dem)
    diagnostic_uncovered = sum(row["保障方式"] == "未覆盖" for row in comm_rows)
    if transport_issues or resource["status"] != "PASS":
        status = "ERROR"
    elif diagnostic_uncovered or comm["status"] == "FAIL":
        status = "COVERAGE_INCOMPLETE"
    elif comm["status"] == "UNVERIFIED":
        status = "UNVERIFIED"
    else:
        status = "PASS"
    return {
        "status": status, "transport_status": "FAIL" if transport_issues else "PASS",
        "transport_issues": transport_issues,
        "diagnostic_uncovered_rows": diagnostic_uncovered,
        "communication": comm, "relay_resources": resource,
    }


def _save_verification(table_dir: Path, validation: dict[str, Any],
                       scheme_check_id: str = "SCHEME1_STATUS") -> None:
    comm = validation["communication"]
    resource = validation["relay_resources"]
    interval_columns = ["sortie_id", "stage", "start_s", "end_s", "mode",
                        "relay_id", "comm_row_index", "status", "method",
                        "margin_db", "worst_distance_m",
                        "pessimistic_obstructed", "witness_s", "reason"]
    write_rows(table_dir / "Q3_逐区间验收.csv", comm["checks"], interval_columns)
    checks = [{"check_id": "Q3_TRANSPORT", "status": validation["transport_status"],
               "detail": "; ".join(validation["transport_issues"][:3]) or
               "80 箱、运输轨迹、硬截止及机/电池资源复核通过"},
              {"check_id": "RELAY_RESOURCE", "status": resource["status"],
               "detail": resource.get("first_failure", "") or "中继物理与资源独立复核"},
              {"check_id": "COMM_CONTINUOUS", "status": comm["status"],
               "detail": (f"PASS={comm['certified_interval_count']}, "
                          f"FAIL={comm['failed_interval_count']}, "
                          f"UNVERIFIED={comm['unverified_interval_count']}")},
               {"check_id": scheme_check_id, "status": validation["status"],
               "detail": f"未覆盖诊断行={validation['diagnostic_uncovered_rows']}"}]
    write_rows(table_dir / "Q3_验收检查.csv", checks,
               ["check_id", "status", "detail"])
    json_dump(table_dir / "Q3_通信问题.json", {
        "transport_issues": validation["transport_issues"],
        "issues": comm["issues"],
        "failed_or_unverified_intervals": [item for item in comm["checks"]
                                           if item["status"] != "PASS"],
        "resource_issues": resource.get("issues", []),
    })


def _save_joint_tables(table_dir: Path, state: JointState,
                       relay: dict[str, Any]) -> None:
    sorties = sorted(state.sorties, key=lambda row: row["batch_id"])
    official_sorties = [{
        "架次编号": s["batch_id"], "无人机编号": s["uav_id"],
        "机型编号": s["type_id"], "电池编号": s["battery_id"],
        "开始时刻（s）": _format_time(s["start_s"]),
        "访问服务区顺序": s["visit_order"],
        "返回O01时刻（s）": _format_time(s["return_s"]),
        "架次能耗（kWh）": _format_time(s["energy_kwh"]),
    } for s in sorties]
    official_deliveries = [{
        "货箱编号": d["box_id"], "架次编号": d["batch_id"],
        "服务区编号": d["zone_id"],
        "交付完成时刻（s）": _format_time(d["complete_s"]),
    } for d in sorted(state.deliveries, key=lambda row: row["box_id"])]
    write_rows(table_dir / "Q3_运输架次.csv", official_sorties, TRANSPORT_COLUMNS)
    write_rows(table_dir / "Q3_逐箱交付.csv", official_deliveries, DELIVERY_COLUMNS)
    detail_columns = [
        "batch_id", "type_id", "uav_id", "battery_id", "visit_order",
        "box_ids", "start_s", "launch_s", "return_s", "charge_start_s",
        "charge_end_s", "charge_s", "mass_kg", "volume_m3", "energy_kwh",
        "return_soc",
    ]
    details = []
    for sortie in sorties:
        row = {key: sortie[key] for key in detail_columns}
        row["box_ids"] = ";".join(row["box_ids"])
        for key in detail_columns[6:]:
            row[key] = _format_time(row[key])
        details.append(row)
    write_rows(table_dir / "Q3_运输明细.csv", details, detail_columns)
    write_rows(table_dir / "Q3_逐箱时限检查.csv",
               state.deliveries, list(state.deliveries[0]))
    write_rows(table_dir / "Q3_中继架次.csv",
               relay_rows_from_missions(state.missions), RELAY_COLUMNS)
    bounds = {s["batch_id"]: (s["start_s"], s["return_s"]) for s in sorties}
    write_rows(table_dir / "Q3_通信保障.csv",
               communication_rows(state.slices, state.assigned, bounds),
               COMM_COLUMNS)
    _save_diagnostics(table_dir, state.slices, [], state.missions, relay)
    json_dump(table_dir / "Q3_待保障区间汇总.json",
              blind_intervals(state.slices))


def _raw_route_sets(ctx: Context) -> list[tuple[str, list[RouteTask]]]:
    """只用已清洗的 80 箱生成同区组批；不同容量给联合排程留修复余地。"""
    by_zone: dict[str, list[str]] = defaultdict(list)
    for box_id, box in ctx.boxes.items():
        by_zone[str(box["zone_id"])].append(box_id)

    def box_key(box_id: str) -> tuple[float, float, float, str]:
        box = ctx.boxes[box_id]
        hard = float(box["hard_deadline_s"]) if not pd.isna(box["hard_deadline_s"]) else math.inf
        return (hard, float(box["expected_s"]), -float(box["priority"]), box_id)

    route_sets: list[tuple[str, list[RouteTask]]] = []
    signatures: set[tuple[tuple[str, ...], ...]] = set()
    for name, hard_cap, soft_cap in (
        ("balanced", 3, 6), ("compact", 8, 12),
        ("hard_split", 1, 6), ("small_batches", 2, 3),
    ):
        routes = []
        for zone in sorted(by_zone):
            ordered = sorted(by_zone[zone], key=box_key)
            hard = [bid for bid in ordered
                    if not pd.isna(ctx.boxes[bid]["hard_deadline_s"])]
            soft = [bid for bid in ordered
                    if pd.isna(ctx.boxes[bid]["hard_deadline_s"])]
            for source, cap in ((hard, hard_cap), (soft, soft_cap)):
                bins: list[list[str]] = []
                for box_id in source:
                    placed = False
                    for group in bins:
                        if len(group) >= cap:
                            continue
                        trial = ctx.route([(zone, [*group, box_id])])
                        if ctx.options(trial):
                            group.append(box_id)
                            placed = True
                            break
                    if not placed:
                        trial = ctx.route([(zone, [box_id])])
                        if not ctx.options(trial):
                            raise ValueError(f"货箱 {box_id} 对所有运输机型均不可行")
                        bins.append([box_id])
                routes.extend(ctx.route([(zone, group)]) for group in bins)
        covered = [bid for route in routes for bid in route.box_ids]
        if len(covered) != 80 or set(covered) != set(ctx.boxes):
            raise AssertionError("原始数据组批未恰好覆盖 80 箱")
        signature = tuple(sorted(tuple(route.box_ids) for route in routes))
        if signature not in signatures:
            signatures.add(signature)
            route_sets.append((name, [RouteTask(f"T{i:03d}", route)
                                      for i, route in enumerate(routes, 1)]))
    # 额外尝试把相邻服务区的软箱合为双停靠架次；硬截止箱仍独立优先。
    compact = next((tasks for name, tasks in route_sets if name == "compact"), None)
    if compact is None:
        # 紧凑组批可能与 balanced 完全相同并被去重；仍需尝试跨区组合。
        compact = next((tasks for name, tasks in route_sets if name == "balanced"), None)
    if compact is not None:
        hard_routes = [task.route for task in compact if any(
            not pd.isna(ctx.boxes[bid]["hard_deadline_s"])
            for bid in task.route.box_ids)]
        soft_routes = [task.route for task in compact if all(
            pd.isna(ctx.boxes[bid]["hard_deadline_s"])
            for bid in task.route.box_ids)]
        mixed = list(hard_routes)
        while soft_routes:
            first = soft_routes.pop(0)
            ranked = []
            for index, other in enumerate(soft_routes):
                if first.zones[0] == other.zones[0]:
                    continue
                for visits in (first.visits + other.visits,
                               other.visits + first.visits):
                    route = ctx.route([(zone, list(ids)) for zone, ids in visits])
                    options = ctx.options(route)
                    if options:
                        ranked.append((min(opt["duration_s"] for opt in options.values()),
                                       index, route))
            if ranked:
                _, index, merged = min(ranked, key=lambda row: (row[0], row[1]))
                soft_routes.pop(index)
                mixed.append(merged)
            else:
                mixed.append(first)
        signature = tuple(sorted(tuple(route.box_ids) for route in mixed))
        if signature not in signatures:
            route_sets.insert(2, ("neighbor_pair", [
                RouteTask(f"T{i:03d}", route) for i, route in enumerate(mixed, 1)]))
    singles = [ctx.route([(str(box["zone_id"]), [box_id])])
               for box_id, box in sorted(ctx.boxes.items())]
    route_sets.append(("single_box_fallback", [
        RouteTask(f"T{i:03d}", route) for i, route in enumerate(singles, 1)]))
    return route_sets


def _source_slices_from_raw(ctx: Context, positions: dict[str, tuple[float, float, float]],
                            links: LinkEvaluator, step_s: float,
                            route_sets: list[tuple[str, list[RouteTask]]]) -> list[TimeSlice]:
    """用原始箱构造各服务区、各可用机型的飞行空间走廊，寻找中继候选点。"""
    by_zone: dict[str, list[str]] = defaultdict(list)
    for box_id, box in ctx.boxes.items():
        by_zone[str(box["zone_id"])].append(box_id)
    paired = next((tasks for name, tasks in route_sets
                   if name == "neighbor_pair"), [])
    total = len(by_zone) * len(ctx.models) + sum(
        len(ctx.options(task.route)) for task in paired if len(task.route.zones) >= 2)
    progress = ProgressBar("源路线通信分段", total)
    completed = 0
    slices: list[TimeSlice] = []
    for zone, ids in sorted(by_zone.items()):
        for kind in sorted(ctx.models):
            progress.update(completed)
            feasible = next((ctx.route([(zone, [bid])]) for bid in ids
                             if kind in ctx.options(ctx.route([(zone, [bid])]))), None)
            completed += 1
            if feasible is None:
                continue
            option = ctx.options(feasible)[kind]
            segments = _relative_trajectory(ctx, feasible, option, positions)
            slices.extend(_certified_slices(f"SRC_{zone}_{kind}", segments,
                                            links, step_s))
    for task in paired:
        if len(task.route.zones) < 2:
            continue
        for kind, option in ctx.options(task.route).items():
            progress.update(completed)
            completed += 1
            segments = _relative_trajectory(ctx, task.route, option, positions)
            slices.extend(_certified_slices(
                f"SRC_{task.batch_id}_{kind}", segments, links, step_s))
    progress.close()
    if not slices:
        raise ValueError("原始货箱未形成可计算的运输走廊")
    return slices


def _route_profile(ctx: Context, task: RouteTask, kind: str,
                   option: dict[str, Any],
                   positions: dict[str, tuple[float, float, float]],
                   links: LinkEvaluator, step_s: float,
                   candidates: list[RelayCandidate], site_tree: cKDTree | None,
                   access_cache: dict[tuple[Any, ...], bool]) -> RouteProfile:
    """先用乐观距离界筛选，再对固定路线做一次严格链路证明。"""
    segments = _relative_trajectory(ctx, task.route, option, positions)
    slices = _certified_slices(task.batch_id, segments, links, step_s)
    budget = bidirectional_limit_db(links.radios["transport"],
                                    links.radios["relay_access"],
                                    links.system_loss_db)
    optimistic_range_m = (1000.0 * 10.0 ** (
        (budget - 32.45 - 20.0 * math.log10(links.frequency_mhz)) / 20.0))
    support: dict[int, set[int]] = {}
    blind_ids: list[int] = []
    uncovered: list[int] = []
    for item in slices:
        if item.direct_available:
            continue
        blind_ids.append(item.slice_id)
        nearby = ([] if site_tree is None else site_tree.query_ball_point(
            item.midpoint, optimistic_range_m + 1e-4))
        available = {int(index) for index in nearby
                     if _access_certified(candidates[int(index)], item,
                                          links, access_cache)}
        support[item.slice_id] = available
        if not available:
            uncovered.append(item.slice_id)
    ranked: dict[int, tuple[int, ...]] = {}
    following: dict[int, int] = {}
    for slice_id in reversed(blind_ids):
        lengths = {index: following.get(index, 0) + 1
                   for index in support[slice_id]}
        ranked[slice_id] = tuple(sorted(lengths, key=lambda index: (
            -lengths[index], candidates[index].travel_energy_kwh,
            -candidates[index].max_service_s, candidates[index].candidate_id)))
        following = lengths
    return RouteProfile(segments, slices, ranked, support, uncovered)


@dataclass(frozen=True)
class BlindBlock:
    begin_s: float
    end_s: float
    sites: frozenset[int]
    slice_ids: tuple[int, ...]


@dataclass
class JointChoice:
    task: RouteTask
    kind: str
    option: dict[str, Any]
    profile: RouteProfile
    blocks: list[BlindBlock]


def _blind_blocks(profile: RouteProfile,
                  sites: list[RelayCandidate],
                  split_on_site_change: bool = False) -> list[BlindBlock] | None:
    """相邻盲区共用中继；精细排班时保留站点集合变化边界。"""
    blocks: list[BlindBlock] = []
    current: list[int] = []
    common: set[int] = set()
    previous_supported: set[int] = set()
    begin = end = 0.0
    for item in profile.slices:
        if item.direct_available:
            if current:
                blocks.append(BlindBlock(begin, end, frozenset(common), tuple(current)))
                current, common, previous_supported = [], set(), set()
            continue
        supported = {j for j in profile.support[item.slice_id]
                     if sites[j].max_service_s >= item.end_s - item.start_s - EPS}
        if not supported:
            return None
        extending = common.intersection(supported)
        extending = {j for j in extending
                     if sites[j].max_service_s >= item.end_s - begin - EPS}
        if current and (item.start_s > end + 1e-5 or not extending or
                        (split_on_site_change and supported != previous_supported)):
            blocks.append(BlindBlock(begin, end, frozenset(common), tuple(current)))
            current, common = [], set()
        if not current:
            begin, common = item.start_s, set(supported)
        else:
            common = extending
        end = item.end_s
        current.append(item.slice_id)
        previous_supported = supported
    if current:
        blocks.append(BlindBlock(begin, end, frozenset(common), tuple(current)))
    return blocks


def _shortlist_sites(source_slices: list[TimeSlice], all_sites: list[RelayCandidate],
                     links: LinkEvaluator, limit: int,
                     cache: dict[tuple[Any, ...], bool]) -> list[RelayCandidate]:
    """保留逐段覆盖和双走廊互补站点；单点覆盖率不能替代两站联合覆盖。"""
    blind = [item for item in source_slices if not item.direct_available]
    if not blind:
        return []
    # 每条单区空间走廊取一个完整的机型剖面；其他走廊另取抽样用于排名。
    # 这些仅用于选候选，所有正式路线随后仍逐段认证。
    primary: dict[str, str] = {}
    for item in blind:
        if item.sortie_id.startswith("SRC_S"):
            zone = item.sortie_id.rsplit("_", 1)[0]
            old = primary.get(zone)
            if old is None or item.sortie_id.endswith("_B"):
                primary[zone] = item.sortie_id
    primary_names = set(primary.values())
    stride = max(1, math.ceil(len(blind) / 480))
    sample = [item for i, item in enumerate(blind)
              if item.sortie_id in primary_names or i % stride == 0 or i == len(blind)-1]
    tree = cKDTree(np.asarray([site.hover for site in all_sites], dtype=float))
    budget = bidirectional_limit_db(links.radios["transport"],
                                    links.radios["relay_access"], links.system_loss_db)
    radius = 1000.0 * 10.0 ** ((budget - 32.45 -
                20.0 * math.log10(links.frequency_mhz)) / 20.0)
    covers = [0 for _ in all_sites]
    groups: dict[str, int] = defaultdict(int)
    progress = ProgressBar("站点完整走廊认证", len(sample))
    for index, item in enumerate(sample):
        progress.update(index)
        bit = 1 << index
        if item.sortie_id in primary_names:
            groups[item.sortie_id] |= bit
        for j in tree.query_ball_point(item.midpoint, radius + 1e-4):
            if _access_certified(all_sites[int(j)], item, links, cache):
                covers[int(j)] |= bit
    progress.close()
    universe = (1 << len(sample)) - 1
    union = 0
    for mask in covers:
        union |= mask
    if union != universe:
        raise ValueError(f"{(universe ^ union).bit_count()} 个通信分段没有可认证中继点")
    selected: list[int] = []
    remaining = universe
    while remaining:
        j = max(range(len(all_sites)), key=lambda k: (
            (covers[k] & remaining).bit_count(), -all_sites[k].outward_s,
            -all_sites[k].travel_energy_kwh))
        selected.append(j)
        remaining &= ~covers[j]
    # 枚举一条或两条空间走廊的两站覆盖组合。按覆盖模式压缩等价候选，
    # 优先保留稀缺组合，再复用已选站点。不能把每段各有一个站点误作两站可覆盖。
    targets = sorted(set(groups.values()) | {
        left | right for i, left in enumerate(groups.values())
        for right in list(groups.values())[i+1:]})
    obligations = []
    progress = ProgressBar("互补中继站点组合", len(targets))
    for n, target in enumerate(targets):
        progress.update(n)
        patterns: dict[int, int] = {}
        for j, cover in enumerate(covers):
            mask = cover & target
            if not mask:
                continue
            old = patterns.get(mask)
            if old is None or (
                all_sites[j].outward_s + all_sites[j].homeward_s,
                all_sites[j].travel_energy_kwh) < (
                all_sites[old].outward_s + all_sites[old].homeward_s,
                all_sites[old].travel_energy_kwh):
                patterns[mask] = j
        entries = list(patterns.items())
        pairs = []
        for i, (mask, j) in enumerate(entries):
            if mask == target:
                pairs.append((j,))
            else:
                pairs.extend((j, k) for other, k in entries[i+1:]
                             if mask | other == target)
        if pairs:
            obligations.append((len(pairs), target, pairs))
    progress.close()
    for _, target, pairs in sorted(obligations, key=lambda x: (x[0], x[1])):
        best = min(pairs, key=lambda pair: (
            sum(j not in selected for j in pair),
            sum(all_sites[j].outward_s + all_sites[j].homeward_s for j in pair),
            sum(all_sites[j].travel_energy_kwh for j in pair), pair))
        selected.extend(j for j in best if j not in selected)
    if len(selected) > limit:
        raise ValueError(f"逐段与互补覆盖需保留 {len(selected)} 站点，"
                         f"超过 --relay-sites={limit}，请增大候选预算")
    print(f"Q3：保留 {len(selected)} 个站点，保障 {len(obligations)} 组"
          "单/双走廊的两站覆盖可能性", flush=True)
    return [all_sites[j] for j in selected]


def _joint_choices(ctx: Context, route_sets: list[tuple[str, list[RouteTask]]],
                   positions: dict[str, tuple[float, float, float]],
                   links: LinkEvaluator, step_s: float,
                   sites: list[RelayCandidate],
                   cache: dict[tuple[Any, ...], bool]) -> list[JointChoice]:
    unique: dict[tuple[tuple[str, tuple[str, ...]], ...], RouteTask] = {}
    for _, tasks in route_sets:
        for task in tasks:
            unique.setdefault(task.route.visits, task)
    tree = cKDTree(np.asarray([site.hover for site in sites], dtype=float)) if sites else None
    choices: list[JointChoice] = []
    progress = ProgressBar("运输路线通信认证", len(unique))
    for number, task in enumerate(unique.values(), 1):
        progress.update(number - 1)
        named = RouteTask(f"C{number:03d}", task.route)
        for kind, option in sorted(ctx.options(task.route).items()):
            if _latest_hard_start(ctx, task.route, option) < -EPS:
                continue
            profile = _route_profile(ctx, named, kind, option, positions,
                                     links, step_s, sites, tree, cache)
            if profile.uncovered_slices:
                continue
            blocks = _blind_blocks(profile, sites, split_on_site_change=True)
            if blocks is not None:
                choices.append(JointChoice(named, kind, option, profile, blocks))
    progress.close()
    supported = {box for choice in choices for box in choice.task.route.box_ids}
    missing = set(ctx.boxes) - supported
    if missing:
        raise ValueError(f"{len(missing)} 箱没有可用路线/中继组合（示例：{sorted(missing)[:5]}）；"
                         "可提高 --relay-sites 或缩小 --candidate-spacing-m")
    return choices


def _prune_dominated_sites(sites: list[RelayCandidate],
                           choices: list[JointChoice]
                           ) -> tuple[list[RelayCandidate], list[JointChoice]]:
    """只删除对所有候选盲段均不优的站点，保持当前模型的可行域。"""
    blocks = [block for choice in choices for block in choice.blocks]
    signatures = [frozenset(index for index, block in enumerate(blocks)
                            if site_index in block.sites)
                  for site_index in range(len(sites))]
    kept = []
    for i, site in enumerate(sites):
        if not signatures[i]:
            continue
        dominated = any(
            i != j and signatures[i] <= signatures[j] and
            site.outward_s >= other.outward_s and
            site.homeward_s >= other.homeward_s and
            site.max_service_s <= other.max_service_s and
            site.travel_energy_kwh >= other.travel_energy_kwh and
            (signatures[i] != signatures[j] or
             site.outward_s > other.outward_s or
             site.homeward_s > other.homeward_s or
             site.max_service_s < other.max_service_s or
             site.travel_energy_kwh > other.travel_energy_kwh)
            for j, other in enumerate(sites))
        if not dominated:
            kept.append(i)
    remap = {old: new for new, old in enumerate(kept)}
    reduced = []
    for choice in choices:
        profile = choice.profile
        support = {slice_id: {remap[j] for j in available if j in remap}
                   for slice_id, available in profile.support.items()}
        new_profile = RouteProfile(profile.segments, profile.slices, {},
                                   support, [])
        new_blocks = [BlindBlock(
            block.begin_s, block.end_s,
            frozenset(remap[j] for j in block.sites if j in remap),
            block.slice_ids) for block in choice.blocks]
        if any(not block.sites for block in new_blocks):
            raise AssertionError("站点支配筛选删除了唯一可用的中继覆盖")
        reduced.append(JointChoice(choice.task, choice.kind, choice.option,
                                   new_profile, new_blocks))
    return [sites[i] for i in kept], reduced


def _solve_joint(ctx: Context, relay: dict[str, Any], sites: list[RelayCandidate],
                 choices: list[JointChoice], horizon: int, slots: int,
                 time_limit: float, workers: int, hard_only: bool,
                 hints: dict[str, Any] | None = None,
                 fixed_transport: dict[tuple[Any, str], tuple[int, str, str]] | None = None,
                 fixed_partial_transport: dict[tuple[Any, str], tuple[int, str, str]] | None = None,
                 fixed_relay_schedule: dict[int, tuple[int, int, int]] | None = None,
                 feasibility_only: bool = False,
                 diversity_cuts: list[list[tuple[int, int, int]]] | None = None,
                 coverage_cuts: list[list[tuple[int, float, float]]] | None = None,
                 max_concurrent_blind: int | None = None,
                 require_all_routes: bool = False,
                 slot_site_indices: list[int] | None = None,
                 relax_relay_resources: bool = False,
                 relax_relay_uavs: bool = False,
                 relax_relay_units: bool = False,
                 wanted_boxes: set[str] | None = None,
                 resource_available: dict[str, dict[str, int]] | None = None,
                 time_grid_s: int | None = None,
                 objective_mode: str = "balanced",
<<<<<<< HEAD
                 objective_reference: Mapping[str, float] | None = None,
=======
                 metric_limits: Mapping[str, int] | None = None,
>>>>>>> 5f2f7185d793316650042d4cd868b2c6cf3a90c1
                 explicit_relay_uavs: bool = True,
                 search_seed: int = 20260923) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise RuntimeError("缺少 OR-Tools。请在 code 目录运行："
                           r".\.venv\Scripts\python.exe -m pip install -r requirements.txt") from exc
    wanted = (set(wanted_boxes) if wanted_boxes is not None else
              ({bid for bid, box in ctx.boxes.items()
                if not pd.isna(box["hard_deadline_s"])}
               if hard_only else set(ctx.boxes)))
    if not wanted <= set(ctx.boxes):
        raise ValueError("指定货箱包含不在输入数据中的编号")
    pool = [choice for choice in choices if set(choice.task.route.box_ids) <= wanted]
    if not pool:
        raise ValueError("当前阶段没有运输路线候选")
    if slot_site_indices is not None and (len(slot_site_indices) != slots or
                                          any(not 0 <= j < len(sites)
                                              for j in slot_site_indices)):
        raise ValueError("固定中继站点必须逐架次指定有效站点编号")
    model = cp_model.CpModel()
    resource_available = resource_available or {}
    selected, starts, ends = [], [], []
    uav_intervals: dict[str, list[Any]] = defaultdict(list)
    battery_intervals: dict[str, list[Any]] = defaultdict(list)
    transport_uav_by_type: dict[str, list[Any]] = defaultdict(list)
    transport_battery_by_type: dict[str, list[Any]] = defaultdict(list)
    aggregate_transport = fixed_transport is None and fixed_partial_transport is None
    uav_assign: dict[tuple[int, str], Any] = {}
    battery_assign: dict[tuple[int, str], Any] = {}
    for i, choice in enumerate(pool):
        sel = model.NewBoolVar(f"route_{i}")
        if require_all_routes:
            model.Add(sel == 1)
        fixed = (fixed_transport.get((choice.task.route.visits, choice.kind))
                 if fixed_transport is not None else None)
        partial_fixed = (fixed_partial_transport.get(
            (choice.task.route.visits, choice.kind))
            if fixed_partial_transport is not None else None)
        if partial_fixed is not None:
            fixed = partial_fixed
            model.Add(sel == 1)
        if fixed_transport is not None:
            if fixed is None:
                raise ValueError("固定运输排程包含与候选不一致的路线")
            model.Add(sel == 1)
        latest = min(horizon, math.floor(_latest_hard_start(
            ctx, choice.task.route, choice.option))) if math.isfinite(
                _latest_hard_start(ctx, choice.task.route, choice.option)) else horizon
        latest = max(0, latest)
        start = model.NewIntVar(0, latest, f"route_start_{i}")
        if time_grid_s is not None:
            model.AddModuloEquality(0, start, time_grid_s)
        end = model.NewIntVar(0, horizon + math.ceil(choice.option["duration_s"]),
                              f"route_end_{i}")
        duration = math.ceil(choice.option["duration_s"] - EPS)
        model.Add(end == start + duration).OnlyEnforceIf(sel)
        model.Add(end == 0).OnlyEnforceIf(sel.Not())
        model.Add(start == 0).OnlyEnforceIf(sel.Not())
        if fixed is not None:
            model.Add(start == fixed[0])
        selected.append(sel); starts.append(start); ends.append(end)
        for box_id in choice.task.route.box_ids:
            deadline = ctx.boxes[box_id]["hard_deadline_s"]
            if not pd.isna(deadline):
                model.Add(start + math.ceil(choice.option["completion_offsets"][box_id] - EPS)
                          <= math.floor(float(deadline) + EPS)).OnlyEnforceIf(sel)
        uvars = []
        if aggregate_transport:
            transport_uav_by_type[choice.kind].append(model.NewOptionalIntervalVar(
                start, duration, start + duration, sel, f"ui_{i}"))
        else:
            for uav in ctx.uavs:
                if uav.type_id != choice.kind:
                    continue
                use = model.NewBoolVar(f"u_{i}_{uav.uav_id}")
                model.AddImplication(use, sel)
                model.Add(start >= resource_available.get("transport_uav", {}).get(
                    uav.uav_id, 0)).OnlyEnforceIf(use)
                uav_intervals[uav.uav_id].append(model.NewOptionalIntervalVar(
                    start, duration, start + duration, use, f"ui_{i}_{uav.uav_id}"))
                uav_assign[(i, uav.uav_id)] = use
                uvars.append(use)
                if fixed is not None:
                    model.Add(use == int(uav.uav_id == fixed[1]))
            model.Add(sum(uvars) == sel)
        bvars = []
        if aggregate_transport:
            matching = [battery for battery in ctx.batteries
                        if battery.type_id == choice.kind]
            if not matching or any(battery.full_charge_s != matching[0].full_charge_s
                                   for battery in matching):
                raise ValueError("同型运输电池充电参数不一致，不能合并排程")
            occupied = duration + math.ceil(charge_to_full_s(
                choice.option["return_soc"], matching[0].full_charge_s) - EPS)
            transport_battery_by_type[choice.kind].append(
                model.NewOptionalIntervalVar(start, occupied,
                                             start + occupied, sel, f"bi_{i}"))
        else:
            for battery in ctx.batteries:
                if battery.type_id != choice.kind:
                    continue
                use = model.NewBoolVar(f"b_{i}_{battery.battery_id}")
                model.AddImplication(use, sel)
                model.Add(start >= resource_available.get("transport_battery", {}).get(
                    battery.battery_id, 0)).OnlyEnforceIf(use)
                occupied = duration + math.ceil(charge_to_full_s(
                    choice.option["return_soc"], battery.full_charge_s) - EPS)
                battery_intervals[battery.battery_id].append(model.NewOptionalIntervalVar(
                    start, occupied, start + occupied, use,
                    f"bi_{i}_{battery.battery_id}"))
                battery_assign[(i, battery.battery_id)] = use
                bvars.append(use)
                if fixed is not None:
                    model.Add(use == int(battery.battery_id == fixed[2]))
            model.Add(sum(bvars) == sel)
    for box_id in sorted(wanted):
        model.Add(sum(selected[i] for i, choice in enumerate(pool)
                      if box_id in choice.task.route.box_ids) == 1)
    if fixed_partial_transport is not None:
        available_keys = {(choice.task.route.visits, choice.kind) for choice in pool}
        if not set(fixed_partial_transport) <= available_keys:
            raise ValueError("固定早期运输路线不在当前候选池中")
    # 失败的中继子问题只触发排程多样化；这些排除约束不是原题无解证明。
    for cut_number, cut in enumerate(diversity_cuts or []):
        different = []
        for i, low, high in cut:
            if not 0 <= i < len(pool):
                raise ValueError("排程反馈引用了不存在的路线")
            before = model.NewBoolVar(f"cut_{cut_number}_{i}_before")
            after = model.NewBoolVar(f"cut_{cut_number}_{i}_after")
            model.Add(starts[i] < low).OnlyEnforceIf(before)
            model.Add(starts[i] > high).OnlyEnforceIf(after)
            different.extend((selected[i].Not(), before, after))
        if different:
            model.AddBoolOr(different)
    # 若一组通信分段无法由两处站点覆盖，它们不能同时处于盲区。
    # 这是对相对分段的时间析取约束，而非只排除某一轮的起飞时刻。
    for cut_number, cut in enumerate(coverage_cuts or []):
        alternatives = []
        for i, _, _ in cut:
            if not 0 <= i < len(pool):
                raise ValueError("通信反馈引用了不存在的路线")
            alternatives.append(selected[i].Not())
        for a, (i, _, end_offset) in enumerate(cut):
            for b, (j, begin_offset, _) in enumerate(cut):
                if a == b:
                    continue
                before = model.NewBoolVar(f"coverage_{cut_number}_{a}_before_{b}")
                model.Add(starts[i] - starts[j] <=
                          math.floor(begin_offset - end_offset + EPS)).OnlyEnforceIf(before)
                alternatives.append(before)
        model.AddBoolOr(alternatives)
    for intervals in [*uav_intervals.values(), *battery_intervals.values()]:
        model.AddNoOverlap(intervals)
    if aggregate_transport:
        for kind in {uav.type_id for uav in ctx.uavs}:
            group = [uav for uav in ctx.uavs if uav.type_id == kind]
            intervals = list(transport_uav_by_type[kind])
            for uav in group:
                available_s = resource_available.get("transport_uav", {}).get(
                    uav.uav_id, 0)
                if available_s > 0:
                    intervals.append(model.NewIntervalVar(
                        0, available_s, available_s, f"prior_u_{uav.uav_id}"))
            model.AddCumulative(intervals, [1] * len(intervals), len(group))
        for kind in {battery.type_id for battery in ctx.batteries}:
            group = [battery for battery in ctx.batteries
                     if battery.type_id == kind]
            intervals = list(transport_battery_by_type[kind])
            for battery in group:
                available_s = resource_available.get("transport_battery", {}).get(
                    battery.battery_id, 0)
                if available_s > 0:
                    intervals.append(model.NewIntervalVar(
                        0, available_s, available_s,
                        f"prior_b_{battery.battery_id}"))
            model.AddCumulative(intervals, [1] * len(intervals), len(group))
    if max_concurrent_blind is not None:
        if max_concurrent_blind < 1 or slots:
            raise ValueError("运输盲段并发上限仅适用于无中继架次的运输主问题")
        blind_intervals = []
        for i, choice in enumerate(pool):
            for k, block in enumerate(choice.blocks):
                begin = math.floor(block.begin_s + EPS)
                end = math.ceil(block.end_s - EPS)
                blind_intervals.append(model.NewOptionalIntervalVar(
                    starts[i] + begin, end - begin, starts[i] + end,
                    selected[i], f"blind_{i}_{k}"))
        model.AddCumulative(blind_intervals,
                            [1] * len(blind_intervals), max_concurrent_blind)
    params = relay["params"]
    prep = math.ceil(params["prep_s"] - EPS)
    setup = math.ceil(params["link_setup_s"] - EPS)
    outward = [math.ceil(site.outward_s - EPS) for site in sites]
    homeward = [math.ceil(site.homeward_s - EPS) for site in sites]
    max_service = [math.floor(site.max_service_s - EPS) for site in sites]
    relay_span_bound = (horizon + max(homeward, default=0) +
                        math.ceil(max(params["turnaround_s"],
                                      params["full_charge_s"])) + 1)
    active, site_vars, launches, readies, service_ends, returns = [], [], [], [], [], []
    relay_uav_intervals: list[Any] = []
    relay_uav_named_intervals: dict[str, list[Any]] = defaultdict(list)
    relay_uav_named_assign: dict[tuple[int, str], Any] = {}
    relay_unit_intervals: list[Any] = []
    relay_charge_vars: list[Any] = []
    for r in range(slots):
        use = model.NewBoolVar(f"relay_{r}")
        which = [model.NewBoolVar(f"site_{r}_{j}") for j in range(len(sites))]
        model.Add(sum(which) == use)
        if slot_site_indices is not None:
            for j, site_var in enumerate(which):
                model.Add(site_var == use if j == slot_site_indices[r] else site_var == 0)
        if fixed_relay_schedule is not None and r in fixed_relay_schedule:
            site_index, fixed_ready, fixed_end = fixed_relay_schedule[r]
            if not 0 <= site_index < len(sites):
                raise ValueError("固定中继架次的站点编号无效")
            model.Add(use == 1)
            model.Add(which[site_index] == 1)
        launch = model.NewIntVar(0, horizon, f"relay_launch_{r}")
        if time_grid_s is not None:
            model.AddModuloEquality(0, launch, time_grid_s)
        ready = model.NewIntVar(0, horizon, f"relay_ready_{r}")
        service_end = model.NewIntVar(0, horizon, f"relay_service_end_{r}")
        returned = model.NewIntVar(0, horizon + max(homeward, default=0),
                                    f"relay_return_{r}")
        model.Add(ready == launch + prep + setup +
                  sum(outward[j] * which[j] for j in range(len(sites)))).OnlyEnforceIf(use)
        model.Add(returned == service_end +
                  sum(homeward[j] * which[j] for j in range(len(sites)))).OnlyEnforceIf(use)
        model.Add(service_end >= ready + 1).OnlyEnforceIf(use)
        model.Add(service_end - ready <=
                  sum(max_service[j] * which[j] for j in range(len(sites)))).OnlyEnforceIf(use)
        if fixed_relay_schedule is not None and r in fixed_relay_schedule:
            model.Add(ready == fixed_ready)
            model.Add(service_end == fixed_end)
        for var in (launch, ready, service_end, returned):
            model.Add(var == 0).OnlyEnforceIf(use.Not())
        active.append(use); site_vars.append(which)
        launches.append(launch); readies.append(ready)
        service_ends.append(service_end); returns.append(returned)
        uav_span = model.NewIntVar(0, relay_span_bound, f"relay_uav_span_{r}")
        model.Add(uav_span == returned - launch +
                  math.ceil(params["turnaround_s"]))
        if explicit_relay_uavs and not relax_relay_resources and not relax_relay_uavs:
            relay_uses = []
            for uav_id in relay["uav_ids"]:
                owns = model.NewBoolVar(f"relay_uav_{r}_{uav_id}")
                model.AddImplication(owns, use)
                model.Add(launch >= resource_available.get("relay_uav", {}).get(
                    uav_id, 0)).OnlyEnforceIf(owns)
                relay_uav_named_intervals[uav_id].append(
                    model.NewOptionalIntervalVar(
                        launch, uav_span,
                        returned + math.ceil(params["turnaround_s"]), owns,
                        f"relay_uav_occupancy_{r}_{uav_id}"))
                relay_uav_named_assign[(r, uav_id)] = owns
                relay_uses.append(owns)
            model.Add(sum(relay_uses) == use)
        else:
            relay_uav_intervals.append(model.NewOptionalIntervalVar(
                launch, uav_span,
                returned + math.ceil(params["turnaround_s"]), use,
                f"relay_uav_occupancy_{r}"))
        # 组件充电使用题目两阶段公式。SOC>=90% 的短任务只需补足慢充段，
        # 不能套用 SOC<90% 的公式，否则会把这些任务多占用数百秒。
        energy_use_mwh = round(params["energy_use_kwh"] * 1_000_000)
        service_power_kw = params["hover_power_kw"] + params["comm_extra_power_kw"]
        service_mwh_per_s = math.ceil(service_power_kw * 1_000_000 / 3600 - EPS)
        setup_mwh = math.ceil(service_power_kw * params["link_setup_s"] *
                              1_000_000 / 3600 - EPS)
        energy_mwh = (sum(math.ceil(site.travel_energy_kwh * 1_000_000 - EPS) *
                          which[j] for j, site in enumerate(sites)) +
                      setup_mwh * use + service_mwh_per_s *
                      (service_end - ready))
        full_s = math.ceil(params["full_charge_s"] - EPS)
        charge_s = model.NewIntVar(0, full_s, f"relay_charge_{r}")
        relay_charge_vars.append(charge_s)
        short_discharge = model.NewBoolVar(f"relay_soc_above_90_{r}")
        model.Add(10 * energy_mwh <= energy_use_mwh).OnlyEnforceIf(short_discharge)
        model.Add(10 * energy_mwh >= energy_use_mwh + 1).OnlyEnforceIf(
            short_discharge.Not())
        fast_numerator = 7 * full_s * energy_mwh
        fast_denominator = 2 * energy_use_mwh
        model.Add(charge_s * fast_denominator >= fast_numerator).OnlyEnforceIf(
            short_discharge)
        model.Add(charge_s * fast_denominator <=
                  fast_numerator + fast_denominator - 1).OnlyEnforceIf(
            short_discharge)
        slow_numerator = full_s * (5 * energy_use_mwh + 13 * energy_mwh)
        slow_denominator = 18 * energy_use_mwh
        model.Add(charge_s * slow_denominator >= slow_numerator).OnlyEnforceIf(
            short_discharge.Not())
        model.Add(charge_s * slow_denominator <=
                  slow_numerator + slow_denominator - 1).OnlyEnforceIf(
            short_discharge.Not())
        unit_span = model.NewIntVar(0, relay_span_bound, f"relay_unit_span_{r}")
        model.Add(unit_span == returned - launch + charge_s)
        occupied_end = model.NewIntVar(0, relay_span_bound,
                                        f"relay_unit_end_{r}")
        model.Add(occupied_end == returned + charge_s)
        relay_unit_intervals.append(model.NewOptionalIntervalVar(
            launch, unit_span, occupied_end, use,
            f"relay_unit_occupancy_{r}"))
        if r and slot_site_indices is None and fixed_relay_schedule is None:
            model.Add(active[r - 1] >= use)
            model.Add(launches[r - 1] <= launch).OnlyEnforceIf(use)
    if not relax_relay_resources:
        if explicit_relay_uavs and not relax_relay_uavs:
            for uav_id in relay["uav_ids"]:
                model.AddNoOverlap(relay_uav_named_intervals[uav_id])
        else:
            for uav_id in relay["uav_ids"]:
                available_s = resource_available.get("relay_uav", {}).get(uav_id, 0)
                if available_s > 0:
                    relay_uav_intervals.append(model.NewIntervalVar(
                        0, available_s, available_s, f"prior_ru_{uav_id}"))
        for unit_id in relay["unit_ids"]:
            available_s = resource_available.get("relay_unit", {}).get(unit_id, 0)
            if available_s > 0:
                relay_unit_intervals.append(model.NewIntervalVar(
                    0, available_s, available_s, f"prior_re_{unit_id}"))
        if not relax_relay_uavs and not explicit_relay_uavs:
            model.AddCumulative(relay_uav_intervals,
                                [1] * len(relay_uav_intervals),
                                len(relay["uav_ids"]))
        if not relax_relay_units:
            model.AddCumulative(relay_unit_intervals,
                                [1] * len(relay_unit_intervals),
                                len(relay["unit_ids"]))
    block_assign: dict[tuple[int, int, int], Any] = {}
    for i, choice in enumerate(pool):
        for k, block in enumerate(choice.blocks):
            if max_concurrent_blind is not None:
                continue
            zvars = []
            for r in range(slots):
                z = model.NewBoolVar(f"cover_{i}_{k}_{r}")
                model.AddImplication(z, active[r])
                model.Add(readies[r] <= starts[i] + math.floor(block.begin_s + EPS)).OnlyEnforceIf(z)
                model.Add(service_ends[r] >= starts[i] + math.ceil(block.end_s - EPS)).OnlyEnforceIf(z)
                model.Add(z <= sum(site_vars[r][j] for j in block.sites))
                block_assign[(i, k, r)] = z
                zvars.append(z)
            model.Add(sum(zvars) == selected[i])
    late_vars = []
    for i, choice in enumerate(pool):
        for box_id in choice.task.route.box_ids:
            box = ctx.boxes[box_id]
            if not pd.isna(box["hard_deadline_s"]):
                continue
            late = model.NewIntVar(0, horizon * 2, f"late_{i}_{box_id}")
            model.Add(late >= starts[i] +
                      math.ceil(choice.option["completion_offsets"][box_id] - EPS) -
                      math.floor(float(box["expected_s"]) + EPS)).OnlyEnforceIf(selected[i])
            model.Add(late == 0).OnlyEnforceIf(selected[i].Not())
            late_vars.append((late, max(1, round(float(box["priority"])))) )
    makespan = model.NewIntVar(0, max(
        horizon + max(homeward, default=0),
        horizon + max(math.ceil(c.option["duration_s"]) for c in pool)),
        "joint_makespan")
    model.AddMaxEquality(makespan, ends + returns)
    transport_wh = sum(round(float(choice.option["energy_kwh"]) * 1000) * selected[i]
                       for i, choice in enumerate(pool))
    relay_power_kw = params["hover_power_kw"] + params["comm_extra_power_kw"]
    service_wh_per_second = math.ceil(relay_power_kw * 1000.0 / 3600.0 - EPS)
    relay_wh = (sum(round(site.travel_energy_kwh * 1000) * site_vars[r][j]
                    for r in range(slots) for j, site in enumerate(sites)) +
                sum(service_wh_per_second * (service_ends[r] - readies[r]) +
                    round(relay_power_kw * params["link_setup_s"] * 1000 / 3600) * active[r]
                    for r in range(slots)))
    # 方案二使用焦耳统一计量目标；上面的 Wh 近似仅保留给旧方案目标。
    energy_j = (
        sum(round(float(choice.option["energy_kwh"]) * 3_600_000) * selected[i]
            for i, choice in enumerate(pool))
        + sum(round(site.travel_energy_kwh * 3_600_000) * site_vars[r][j]
              for r in range(slots) for j, site in enumerate(sites))
        + sum(round(relay_power_kw * 1000) * (service_ends[r] - readies[r])
              + round(relay_power_kw * params["link_setup_s"] * 1000) * active[r]
              for r in range(slots)))
    weighted_late = sum(weight * var for var, weight in late_vars)
    for metric, upper in (metric_limits or {}).items():
        if metric == "weighted_lateness":
            model.Add(weighted_late <= upper)
        elif metric == "makespan_s":
            model.Add(makespan <= upper)
        elif metric == "energy_j":
            model.Add(energy_j <= upper)
        elif metric == "transport_sorties":
            model.Add(sum(selected) <= upper)
        elif metric == "relay_sorties":
            model.Add(sum(active) <= upper)
        else:
            raise ValueError(f"不支持的指标界限：{metric}")
    # 优先处理期望时刻的加权迟到，再兼顾返航、能耗和架次数。
    if not feasibility_only:
        if objective_mode == "sorties":
            model.Minimize(100000 * sum(selected) + 1000 * sum(active) +
                           makespan + transport_wh + relay_wh)
        elif objective_mode == "relay_sorties":
            model.Minimize(100000 * sum(active) + 1000 * sum(selected) +
                           makespan + transport_wh + relay_wh)
        elif objective_mode == "balanced":
            model.Minimize(10000 * sum(weight * var for var, weight in late_vars) +
                           10 * makespan + transport_wh + relay_wh +
                           50 * sum(selected) + 100 * sum(active))
<<<<<<< HEAD
        elif objective_mode == "equal_normalized":
            if objective_reference is None:
                raise ValueError("等权目标必须提供已验收基线的四项指标")
            coef = _equal_weight_coefficients(objective_reference)
            model.Minimize(
                coef["soft_weighted_lateness_s"] *
                sum(weight * var for var, weight in late_vars) +
                coef["joint_makespan_s"] * makespan +
                coef["total_energy_kwh"] * (transport_wh + relay_wh) +
                coef["total_sorties"] * (sum(selected) + sum(active)))
=======
        elif objective_mode == "scheme2_lateness":
            model.Minimize(weighted_late)
        elif objective_mode == "scheme2_makespan":
            model.Minimize(makespan)
        elif objective_mode == "scheme2_energy":
            model.Minimize(energy_j)
        elif objective_mode == "scheme2_sorties":
            model.Minimize(sum(selected))
        elif objective_mode == "scheme2_relay_sorties":
            model.Minimize(sum(active))
>>>>>>> 5f2f7185d793316650042d4cd868b2c6cf3a90c1
        else:
            raise ValueError("不支持的目标函数模式")
    if hints:
        for i, choice in enumerate(pool):
            key = (choice.task.route.visits, choice.kind)
            if key in hints.get("routes", {}):
                model.AddHint(selected[i], 1)
                model.AddHint(starts[i], hints["routes"][key])
        for slot, site_index, uav_id, unit_id, ready, service_end in hints.get("relays", []):
            if slot >= slots or site_index >= len(sites):
                continue
            model.AddHint(active[slot], 1)
            model.AddHint(site_vars[slot][site_index], 1)
            model.AddHint(readies[slot], ready)
            model.AddHint(service_ends[slot], service_end)
            model.AddHint(launches[slot], ready - prep - setup - outward[site_index])
            model.AddHint(returns[slot], service_end + homeward[site_index])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = search_seed
    label = "运输排程" if slots == 0 else "中继排班"
    with SolverBudgetBar(label, time_limit):
        status = solver.Solve(model)
    report = {"stage": "hard" if hard_only else "full",
              "solver_status": solver.StatusName(status),
              "route_options": len(pool), "relay_sites": len(sites),
              "relay_slots": slots, "boxes": len(wanted),
              "wall_time_s": solver.WallTime(),
              "branches": solver.NumBranches(), "conflicts": solver.NumConflicts(),
              "time_limit_s": time_limit}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, report
    route_solution = {}
    for i, choice in enumerate(pool):
        if solver.BooleanValue(selected[i]):
            route_solution[(choice.task.route.visits, choice.kind)] = solver.Value(starts[i])
    if not feasibility_only:
        report["objective_value"] = solver.ObjectiveValue()
        report["best_bound"] = solver.BestObjectiveBound()
    report["selected_transport_sorties"] = len(route_solution)
    report["selected_relay_sorties"] = sum(solver.BooleanValue(v) for v in active)
    relay_ids = [r for r in range(slots) if solver.BooleanValue(active[r])]
    def assign_identical(ids: list[str], occupied_until: list[int]) -> dict[int, str]:
        resource_kind = ("relay_uav" if ids is relay["uav_ids"] else "relay_unit")
        available = {resource_id: resource_available.get(resource_kind, {}).get(
            resource_id, 0) for resource_id in ids}
        assignment: dict[int, str] = {}
        for r in sorted(relay_ids, key=lambda k: (solver.Value(launches[k]), k)):
            start = solver.Value(launches[r])
            free = [resource_id for resource_id in ids
                    if available[resource_id] <= start]
            if not free:
                if relax_relay_resources or relax_relay_uavs or relax_relay_units:
                    assignment[r] = ids[0]
                    continue
                raise AssertionError("中继累计资源约束与具体编号分配不一致")
            resource_id = min(free, key=lambda x: (available[x], x))
            assignment[r] = resource_id
            available[resource_id] = occupied_until[r]
        return assignment
    uav_until = [solver.Value(returns[r]) + math.ceil(params["turnaround_s"])
                 for r in range(slots)]
    unit_until = [solver.Value(returns[r]) + solver.Value(relay_charge_vars[r])
                  for r in range(slots)]
    if explicit_relay_uavs and not relax_relay_resources and not relax_relay_uavs:
        relay_uav_ids = {
            r: next(uav_id for (slot, uav_id), var in relay_uav_named_assign.items()
                    if slot == r and solver.BooleanValue(var))
            for r in relay_ids
        }
    else:
        relay_uav_ids = assign_identical(relay["uav_ids"], uav_until)
    relay_unit_ids = assign_identical(relay["unit_ids"], unit_until)
    final_available = {kind: dict(resource_available.get(kind, {})) for kind in
                       ("transport_uav", "transport_battery", "relay_uav", "relay_unit")}
    transport_ids: dict[int, tuple[str, str]] = {}
    for i in sorted((i for i in range(len(pool)) if solver.BooleanValue(selected[i])),
                    key=lambda index: (solver.Value(starts[index]), index)):
        choice = pool[i]
        start = solver.Value(starts[i])
        end = solver.Value(ends[i])
        if aggregate_transport:
            available_uavs = [uav.uav_id for uav in ctx.uavs
                              if uav.type_id == choice.kind and
                              final_available["transport_uav"].get(uav.uav_id, 0) <= start]
            available_batteries = [battery for battery in ctx.batteries
                                   if battery.type_id == choice.kind and
                                   final_available["transport_battery"].get(
                                       battery.battery_id, 0) <= start]
            if not available_uavs or not available_batteries:
                raise AssertionError("同型累计运输资源无法分配具体编号")
            uav_id = min(available_uavs, key=lambda uid: (
                final_available["transport_uav"].get(uid, 0), uid))
            battery = min(available_batteries, key=lambda item: (
                final_available["transport_battery"].get(item.battery_id, 0),
                item.battery_id))
            battery_id = battery.battery_id
        else:
            uav_id = next(uid for (j, uid), var in uav_assign.items()
                          if j == i and solver.BooleanValue(var))
            battery_id = next(bid for (j, bid), var in battery_assign.items()
                              if j == i and solver.BooleanValue(var))
            battery = next(battery for battery in ctx.batteries
                           if battery.battery_id == battery_id)
        transport_ids[i] = uav_id, battery_id
        final_available["transport_uav"][uav_id] = end
        occupied = math.ceil(choice.option["duration_s"] - EPS) + math.ceil(
            charge_to_full_s(choice.option["return_soc"],
                             battery.full_charge_s) - EPS)
        final_available["transport_battery"][battery_id] = start + occupied
    for r in relay_ids:
        final_available["relay_uav"][relay_uav_ids[r]] = max(
            final_available["relay_uav"].get(relay_uav_ids[r], 0), uav_until[r])
        final_available["relay_unit"][relay_unit_ids[r]] = max(
            final_available["relay_unit"].get(relay_unit_ids[r], 0), unit_until[r])
    return {
        "pool": pool,
        "routes": [(i, solver.Value(starts[i]), *transport_ids[i])
                   for i in range(len(pool)) if solver.BooleanValue(selected[i])],
        "relays": [(r, next(j for j, var in enumerate(site_vars[r])
                           if solver.BooleanValue(var)),
                    relay_uav_ids[r], relay_unit_ids[r],
                    solver.Value(readies[r]), solver.Value(service_ends[r]))
                   for r in relay_ids],
        "block_relay": {(i, k): r for (i, k, r), var in block_assign.items()
                        if solver.BooleanValue(var)},
        "route_hints": route_solution,
        "resource_available": final_available,
    }, report


def _materialize_solution(ctx: Context, relay: dict[str, Any],
                          sites: list[RelayCandidate], solution: dict[str, Any]) -> JointState:
    state = _empty_joint_state()
    params = relay["params"]
    batteries = {battery.battery_id: battery for battery in ctx.batteries}
    relay_id_by_slot: dict[int, str] = {}
    used_slots = set(solution["block_relay"].values())
    # 可行性求解可能启用没有承担覆盖的空中继槽；省去这些任务只会释放资源。
    for number, (slot, site_index, uav_id, unit_id, ready, end) in enumerate(
            sorted((row for row in solution["relays"] if row[0] in used_slots),
                   key=lambda row: row[4]), 1):
        site = sites[site_index]
        relay_id = f"R{number:03d}"
        relay_id_by_slot[slot] = relay_id
        start = ready - params["prep_s"] - site.outward_s - params["link_setup_s"]
        energy = site.travel_energy_kwh + (
            params["hover_power_kw"] + params["comm_extra_power_kw"]) * (
            params["link_setup_s"] + end - ready) / 3600.0
        state.missions.append(RelayMission(
            relay_id, uav_id, unit_id, site, float(start), float(ready),
            float(end), float(end + site.homeward_s), float(energy)))
    for number, (i, start, uav_id, battery_id) in enumerate(
            sorted(solution["routes"], key=lambda row: (row[1], row[0])), 1):
        choice = solution["pool"][i]
        task = RouteTask(f"T{number:03d}", choice.task.route)
        sortie, deliveries = _make_sortie(ctx, task, choice.kind, choice.option,
                                          uav_id, batteries[battery_id], float(start))
        segments, slices = _shift_template(choice.profile.segments,
                                            choice.profile.slices,
                                            float(start), len(state.slices))
        slices = [replace(item, sortie_id=task.batch_id) for item in slices]
        state.sorties.append(sortie)
        state.deliveries.extend(deliveries)
        state.trajectories[task.batch_id] = segments
        for k, block in enumerate(choice.blocks):
            mission_id = relay_id_by_slot[solution["block_relay"][(i, k)]]
            for old_id in block.slice_ids:
                state.assigned[len(state.slices) + old_id] = mission_id
        state.slices.extend(slices)
    return state


def _complete_soft_batches(
        ctx: Context, relay: dict[str, Any], sites: list[RelayCandidate],
        choices: list[JointChoice], route_sets: list[tuple[str, list[RouteTask]]],
        prefix: dict[str, Any], horizon: int, seconds: float, workers: int
        ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """完整保留已排通任务，再安排软箱；搜索预算不足时仍有可检查的构造路径。"""
    result = {key: (list(value) if isinstance(value, list) else dict(value)
                    if isinstance(value, dict) else value)
              for key, value in prefix.items()}
    done = {bid for i, *_ in result["routes"] for bid in result["pool"][i].task.route.box_ids}
    tasks = next(tasks for name, tasks in route_sets if name == "balanced")
    pending = [task for task in tasks if not set(task.route.box_ids) & done]
    reports = []
    for n, task in enumerate(pending, 1):
        remaining = set(task.route.box_ids)
        pool = [c for c in choices if c.task.route.visits == task.route.visits]
        available = result["resource_available"]
        latest = max((t for rows in available.values() for t in rows.values()), default=0)
        local_horizon = max(horizon, latest + 12000)
        print(f"Q3 逐批插入 {n}/{len(pending)}：已排 {len(done)}/80 箱，"
              f"本批 {len(remaining)} 箱", flush=True)
        addition, report = _solve_joint(
            ctx, relay, sites, pool, local_horizon, 4, seconds, workers, False,
            wanted_boxes=remaining, resource_available=available,
            objective_mode="balanced")
        reports.append({"stage": "soft_batch", "result": report})
        if addition is None:
            return None, reports
        offset = len(result["pool"])
        slot_offset = max((r[0] for r in result["relays"]), default=-1) + 1
        result["pool"].extend(addition["pool"])
        result["routes"].extend((i+offset, start, u, b)
                                for i, start, u, b in addition["routes"])
        result["relays"].extend((slot+slot_offset, j, u, b, ready, end)
                                for slot, j, u, b, ready, end in addition["relays"])
        result["block_relay"].update({(i+offset, k): slot+slot_offset
                                      for (i,k),slot in addition["block_relay"].items()})
        result["resource_available"] = addition["resource_available"]
        done.update(remaining)
    if done != set(ctx.boxes):
        raise AssertionError("逐批插入结束后并未恰好安排全部货箱")
    return result, reports


def _geometry_cache_key(source: dict[str, Any], step: float, spacing: int, limit: int) -> str:
    """缓存仅减少重复选点；输入或关联代码改变就失效，不复用任何验收结论。"""
    dependencies = ("q3_1_optimize.py", "q2_1_optimize.py",
                    "q2_0_baseline.py", "q1_0_baseline.py", "_0_pipeline.py")
    key = {"source": source, "sample_step_s": float(step), "spacing_m": spacing,
           "site_limit": limit, "code": {name: sha256(CODE_DIR / name)
                                       for name in dependencies}}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()


def _solution_hints(solution: dict[str, Any]) -> dict[str, Any]:
    """把分批构造的中继槽重新压紧，仅作为搜索提示而不固定排程。"""
    relays = sorted(solution["relays"], key=lambda row: (row[4], row[0]))
    return {
        "routes": {(solution["pool"][i].task.route.visits,
                    solution["pool"][i].kind): start
                   for i, start, *_ in solution["routes"]},
        "relays": [(slot, site, uav, unit, ready, end)
                   for slot, (_, site, uav, unit, ready, end) in enumerate(relays)],
    }


def _search_equal_improvement(
        ctx: Context, relay: dict[str, Any], sites: list[RelayCandidate],
        choices: list[JointChoice], baseline_solution: dict[str, Any],
        baseline_state: JointState, data_run: Path, links: LinkEvaluator,
        dem: Any, table_dir: Path, horizon: int, slots: int,
        seconds: float, workers: int
        ) -> tuple[JointState, dict[str, Any]]:
    """先重排已选航次，再开放合并路线；每个候选均独立复核。"""
    reference = _objective_metrics(baseline_state)
    best_state, best_score = baseline_state, equal_weight_score(reference, reference)
    best_solution = baseline_solution
    selected = [baseline_solution["pool"][i] for i, *_ in baseline_solution["routes"]]
    stages = (("critical_shift", selected, True),
              ("route_merge", choices, False))
    report: dict[str, Any] = {
        "status": "NO_IMPROVEMENT", "baseline": reference,
        "baseline_score": best_score, "stages": [],
        "global_optimality_proven": False,
    }
    for stage, pool, all_routes in stages:
        hints = _solution_hints(best_solution)
        stage_slots = max(slots, len(hints["relays"]))
        print(f"Q3 等权优化 {stage}：{len(pool)} 个路线/机型候选、"
              f"{stage_slots} 个中继槽、{seconds / len(stages):.1f}s 预算", flush=True)
        try:
            candidate, solver_report = _solve_joint(
                ctx, relay, sites, pool, horizon, stage_slots,
                seconds / len(stages), workers, False, hints=hints,
                require_all_routes=all_routes,
                objective_mode="equal_normalized", objective_reference=reference)
            row: dict[str, Any] = {"stage": stage, "solver": solver_report}
            if candidate is not None:
                state = _materialize_solution(ctx, relay, sites, candidate)
                candidate_dir = table_dir / f"_candidate_{stage}"
                _save_joint_tables(candidate_dir, state, relay)
                validation = validate_official_tables(candidate_dir, data_run, links, dem)
                _save_verification(candidate_dir, validation)
                metrics = _objective_metrics(state)
                score = equal_weight_score(metrics, reference)
                row.update({"validation_status": validation["status"],
                            "metrics": metrics, "score": score})
                if validation["status"] == "PASS" and score < best_score - 1e-8:
                    best_state, best_score, best_solution = state, score, candidate
                    row["accepted"] = True
                else:
                    row["accepted"] = False
            report["stages"].append(row)
        except (ValueError, KeyError, AssertionError, RuntimeError, IndexError) as exc:
            report["stages"].append({"stage": stage, "error":
                                     f"{type(exc).__name__}: {exc}"})
    final_metrics = _objective_metrics(best_state)
    report.update({"status": "IMPROVED" if best_score < report["baseline_score"] - 1e-8
                   else "NO_IMPROVEMENT",
                   "final": final_metrics,
                   "final_normalized": equal_weight_ratios(final_metrics, reference),
                   "final_score": best_score})
    return best_state, report


def run_scheme1(data_run: Path, output_root: Path, sample_step_s: float,
                 spacing_m: int, site_limit: int, relay_slots: int,
                 horizon: int, transport_seconds: float, relay_seconds: float,
                 workers: int, attempts: int, shift_s: int,
                 optimize_seconds: float = 0.0) -> dict[str, Any]:
    if (sample_step_s <= 0 or min(spacing_m, site_limit, relay_slots,
                                 horizon, workers, attempts, shift_s) < 1 or
             transport_seconds <= 0 or relay_seconds <= 0 or optimize_seconds < 0):
        raise ValueError("分段、候选数、资源时域、尝试次数和时间预算必须为正")
    relay, dem, links, source_meta = _scene(data_run)
    ctx = Context(data_run)
    positions = _node_positions(data_run)
    route_sets = _raw_route_sets(ctx)
    source_slices = _source_slices_from_raw(ctx, positions, links,
                                            sample_step_s, route_sets)
    print(f"Q3：已生成 {len(source_slices)} 个源路线通信分段，正在认证中继站点……",
          flush=True)
    cache_dir = Path(tempfile.gettempdir()) / "huawei_q3_geometry"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _geometry_cache_key(source_meta, sample_step_s, spacing_m, site_limit)
    cache_path = cache_dir / f"{cache_key}.json"
    access_cache: dict[tuple[Any, ...], bool] = {}
    cache_record = None
    if cache_path.exists():
        try:
            cache_record = json.loads(cache_path.read_text(encoding="utf-8"))
            sites = [RelayCandidate(**row) for row in cache_record["sites"]]
            if not sites or len(sites) > site_limit:
                raise ValueError("缓存站点数量无效")
            candidate_stats = cache_record["stats"]
            print(f"Q3：复用相同数据、参数与代码的 {len(sites)} 个站点；"
                  "正式运输路线和最终结果仍重新认证", flush=True)
        except (ValueError, TypeError, KeyError, OSError):
            cache_record = None
    if cache_record is None:
        all_sites, candidate_stats = generate_candidates(
            dem, links, source_slices, relay, positions["O01"],
            spacing_m=spacing_m, max_points=12000)
        if not all_sites:
            raise ValueError("DEM 和返航能量约束下没有可用中继悬停点")
        print(f"Q3：{len(all_sites)} 个物理站点候选；正在筛选互补覆盖……", flush=True)
        sites = _shortlist_sites(source_slices, all_sites, links, site_limit, access_cache)
        json_dump(cache_path, {"sites": [asdict(site) for site in sites],
                              "stats": candidate_stats})
    print(f"Q3：{len(sites)} 个中继站点；开始生成运输与通信组合", flush=True)
    choices = _joint_choices(ctx, route_sets, positions, links,
                             sample_step_s, sites, access_cache)
    print(f"Q3：{len(choices)} 个可认证的路线/机型组合；"
          "开始硬时限货物与中继联合排程", flush=True)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    run_label = "q3_equal" if optimize_seconds > 0 else "q3_opt1"
    table_dir = output_root / f"{run_label}_{stamp}_table"
    figure_dir = output_root / f"{run_label}_{stamp}_figure"
    table_dir.mkdir(parents=True, exist_ok=False)
    # 先联合解决有硬时限的货箱。中继站点、启停、充电和运输时间同时决策，
    # 避免运输先占满早期时窗后，中继只能反复宣告无法覆盖。
    hard_boxes = {bid for bid, box in ctx.boxes.items()
                  if not pd.isna(box["hard_deadline_s"])}
    grouped = {task.route.visits for name, tasks in route_sets
               if name in ("balanced", "compact") for task in tasks}
    hard_grouped = [c for c in choices if c.task.route.visits in grouped
                    and set(c.task.route.box_ids) <= hard_boxes]
    attempt_log: list[dict[str, Any]] = []
    hard_solution = None
    hard_variants = [(hard_grouped, min(10, relay_slots)),
                     (hard_grouped, relay_slots),
                     ([c for c in choices if set(c.task.route.box_ids) <= hard_boxes],
                      relay_slots)]
    for stage, (pool, count) in enumerate(hard_variants[:attempts], 1):
        print(f"Q3 硬时限联合排程 {stage}：{len(hard_boxes)} 箱、"
              f"{len(pool)} 个路线/机型候选、{count} 个中继架次槽", flush=True)
        hard_horizon = min(horizon, math.ceil(max(
            _latest_hard_start(ctx, c.task.route, c.option) + c.option["duration_s"]
            for c in pool)))
        hard_solution, report = _solve_joint(
            ctx, relay, sites, pool, hard_horizon, count, transport_seconds,
            workers, True, feasibility_only=True)
        attempt_log.append({"stage": "hard_joint", "result": report})
        print(f"Q3 硬时限联合排程：{report['solver_status']}", flush=True)
        if hard_solution is not None:
            break
    winning = None
    if hard_solution is not None:
        used = set(hard_solution["block_relay"].values())
        hard_solution["relays"] = [r for r in hard_solution["relays"] if r[0] in used]
        fixed = {(hard_solution["pool"][i].task.route.visits,
                  hard_solution["pool"][i].kind): (start, uav, battery)
                 for i, start, uav, battery in hard_solution["routes"]}
        fixed_relays = {slot: (j, ready, end)
                        for slot, j, _, _, ready, end in hard_solution["relays"]}
        selected = [hard_solution["pool"][i] for i, *_ in hard_solution["routes"]]
        soft_grouped = [c for c in choices if c.task.route.visits in grouped
                        and not set(c.task.route.box_ids) & hard_boxes]
        print(f"Q3：硬时限 {len(hard_boxes)}/{len(hard_boxes)} 箱已排通，"
              f"{len(selected)} 运输架次；正在安排其余 {80-len(hard_boxes)} 箱", flush=True)
        full_solution, report = _solve_joint(
            ctx, relay, sites, selected + soft_grouped, horizon, relay_slots,
            relay_seconds, workers, False, fixed_partial_transport=fixed,
            fixed_relay_schedule=fixed_relays, feasibility_only=True)
        attempt_log.append({"stage": "all_boxes_joint", "result": report})
        if full_solution is None:
            # 已有硬时限方案始终保留。软时限只计迟到，不应因搜索超时丢弃已交付货箱。
            print("Q3：全部软箱联合插入尚未得到解，改为按整批逐次插入并检查资源", flush=True)
            full_solution, reports = _complete_soft_batches(
                ctx, relay, sites, choices, route_sets, hard_solution,
                horizon, min(relay_seconds, 30.0), workers)
            attempt_log.extend(reports)
            report = {"solver_status": "FEASIBLE_CONSTRUCTION"}
        if full_solution is not None:
            winning = full_solution, sites, report
    summary: dict[str, Any] = {
        "status": "SEARCH_INCOMPLETE",
        "method": "complementary relay sites; hard-deadline joint CP-SAT; "
                  "grouped soft-delivery insertion; independent continuous verification",
        "interpretation": "有限候选和时间预算内未找到完整可验收方案；UNKNOWN 不证明不可行",
        "source": source_meta, "candidate_stats": candidate_stats,
        "shortlisted_relay_sites": len(sites),
        "attempts": attempt_log, "attempt_count": len(attempt_log),
        "global_optimality_proven": False,
        "table_dir": str(table_dir.resolve()), "figure_dir": None,
    }
    json_dump(table_dir / "Q3_候选筛选统计.json", candidate_stats)
    if winning is None:
        json_dump(table_dir / "Q3_运行摘要.json", summary)
        return summary
    full_solution, winning_sites, relay_report = winning
    state = _materialize_solution(ctx, relay, winning_sites, full_solution)
    baseline_dir = table_dir / "_baseline" if optimize_seconds > 0 else table_dir
    _save_joint_tables(baseline_dir, state, relay)
    validation = validate_official_tables(baseline_dir, data_run, links, dem)
    if optimize_seconds > 0:
        _save_verification(baseline_dir, validation)
        if validation["status"] == "PASS":
            state, optimization = _search_equal_improvement(
                ctx, relay, winning_sites, choices, full_solution, state,
                data_run, links, dem, table_dir, horizon, relay_slots,
                optimize_seconds, workers)
            summary["optimization"] = optimization
        else:
            summary["optimization"] = {
                "status": "SKIPPED_BASELINE_NOT_PASS",
                "baseline_validation_status": validation["status"]}
        _save_joint_tables(table_dir, state, relay)
        validation = validate_official_tables(table_dir, data_run, links, dem)
        if (validation["status"] != "PASS" and
                summary["optimization"]["status"] == "IMPROVED"):
            raise AssertionError("候选预验收 PASS 但最终导出未通过，停止交付")
    _save_verification(table_dir, validation)
    summary.update({
        "status": validation["status"],
        "interpretation": "分层候选方案已导出并独立复核；未证明全局最优",
        "candidate_pool_optimality_proven": False,
        "relay_solver_status": relay_report["solver_status"],
        "transport_sorties": len(state.sorties),
        "relay_sorties": len(state.missions),
        "delivered_boxes": len(state.deliveries),
        "transport_last_return_s": max(s["return_s"] for s in state.sorties),
        "relay_last_return_s": max((m.return_s for m in state.missions), default=0.0),
        "joint_makespan_s": max([s["return_s"] for s in state.sorties] +
                                [m.return_s for m in state.missions]),
        "transport_energy_kwh": sum(s["energy_kwh"] for s in state.sorties),
        "relay_energy_kwh": sum(m.energy_kwh for m in state.missions),
        "soft_weighted_lateness_s": sum(d["priority"] * d["soft_lateness_s"]
                                        for d in state.deliveries
                                        if d["hard_deadline_s"] is None),
        "communication_status": validation["communication"]["status"],
        "transport_status": validation["transport_status"],
        "relay_resource_status": validation["relay_resources"]["status"],
    })
    summary["total_energy_kwh"] = (summary["transport_energy_kwh"] +
                                   summary["relay_energy_kwh"])
    if optimize_seconds > 0 and summary["optimization"]["status"] in (
            "IMPROVED", "NO_IMPROVEMENT"):
        summary["method"] += "; normalized equal-weight incumbent-guided joint search"
        summary["interpretation"] = (
            "候选方案通过独立复核且等权得分低于本次 PASS 基线；未证明全局最优"
            if summary["optimization"]["status"] == "IMPROVED" else
            "限时搜索未找到更优且通过独立复核的方案；保留本次 PASS 基线，未证明最优")
        summary["equal_weight_score"] = equal_weight_score(
            _objective_metrics(state), summary["optimization"]["baseline"])
    if validation["status"] == "PASS":
        _save_figures(figure_dir, state.slices, state.assigned, state.missions)
        (table_dir / "Q3_READY.txt").write_text(
            ("Q3 equal-weight search independently verified PASS\n"
             if optimize_seconds > 0 else
             "Q3 decomposed scheme 1 independently verified PASS\n") +
            f"source_manifest_sha256={source_meta['source_manifest_sha256']}\n",
            encoding="utf-8")
        summary["figure_dir"] = str(figure_dir.resolve())
    json_dump(table_dir / "Q3_运行摘要.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-run", type=Path, default=None,
                        help="0_outputs 中经过验收的原始数据目录")
    parser.add_argument("--output-root", type=Path,
                        default=None, help="结果根目录；优化模式默认写入 3_outputs/2_equal_optimize")
    parser.add_argument("--optimize-seconds", type=float, default=0.0,
                        help="等权优化总搜索秒数；0 表示只运行原基线")
    parser.add_argument("--sample-step-s", type=float, default=20.0,
                        help="通信候选原子的最长秒数；可设 15，最终仍做连续区间复核")
    parser.add_argument("--candidate-spacing-m", type=int, default=500)
    parser.add_argument("--relay-sites", type=int, default=72)
    parser.add_argument("--relay-slots", type=int, default=18)
    parser.add_argument("--horizon-s", type=int, default=21600,
                        help="运输开始/中继服务结束的搜索时域；返航及充电可在该时域之后结束")
    parser.add_argument("--transport-time-limit-s", type=float, default=120.0,
                        help="每个硬时限运输与中继联合模型的时间上限")
    parser.add_argument("--relay-time-limit-s", type=float, default=120.0,
                        help="其余货箱插入联合模型的时间上限")
    parser.add_argument("--attempts", type=int, default=3,
                        help="硬时限候选扩展次数，最多采用三种候选规模")
    parser.add_argument("--feedback-shift-s", type=int, default=120,
                        help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validate-only", type=Path, default=None,
                        metavar="Q3_TABLE_DIR", help="只重验已导出的第三问表格")
    args = parser.parse_args(argv)
    data_run = (args.data_run.resolve() if args.data_run is not None else
                find_data_run(CODE_DIR, EXPECTED_MANIFEST))
    if args.validate_only is not None:
        _, dem, links, _ = _scene(data_run)
        report = validate_official_tables(args.validate_only.resolve(),
                                          data_run, links, dem)
        print(json.dumps({"status": report["status"],
                          "transport_status": report["transport_status"],
                          "communication_status": report["communication"]["status"],
                          "relay_resource_status": report["relay_resources"]["status"]},
                         ensure_ascii=False, indent=2))
        return 0 if report["status"] == "PASS" else 2
    output_root = (args.output_root if args.output_root is not None else
                   CODE_DIR / "3_outputs" /
                   ("2_equal_optimize" if args.optimize_seconds > 0 else "1_optimize"))
    summary = run_scheme1(data_run, output_root.resolve(),
                           args.sample_step_s, args.candidate_spacing_m,
                           args.relay_sites, args.relay_slots, args.horizon_s,
                           args.transport_time_limit_s, args.relay_time_limit_s,
                           args.workers, args.attempts, args.feedback_shift_s,
                           args.optimize_seconds)
    print(json.dumps({key: summary.get(key) for key in (
        "status", "transport_sorties", "relay_sorties", "delivered_boxes",
        "joint_makespan_s", "total_energy_kwh", "equal_weight_score",
        "table_dir", "figure_dir")},
        ensure_ascii=False, indent=2))
    if args.optimize_seconds > 0:
        print("等权优化状态：" + summary.get("optimization", {}).get(
            "status", "BASELINE_INCOMPLETE"), flush=True)
    return 0 if summary["status"] == "PASS" else 2

# 通信与中继资源复核内置于同一脚本，保持单文件交付。
DECISION_GUARD_DB = 1e-6
TIME_TOL_S = 1e-7


@dataclass(frozen=True)
class _RelayWindow:
    relay_id: str
    hover: tuple[float, float, float]
    ready_s: float
    service_end_s: float
    backhaul_status: str
    backhaul_margin_db: float | None


@dataclass(frozen=True)
class _Allocation:
    row_index: int
    sortie_id: str
    stage: str
    start_s: float
    end_s: float
    mode: str
    relay_id: str


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} 不是数字：{value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} 不是有限数字：{value!r}")
    return number


def _bidirectional_limit(a: Any, b: Any, system_loss_db: float) -> float:
    """独立重算上下行预算，取更严格的一向。"""
    forward = (a.tx_power_dbm + a.gain_dbi + b.gain_dbi - system_loss_db -
               b.sensitivity_dbm - b.fade_margin_db)
    backward = (b.tx_power_dbm + b.gain_dbi + a.gain_dbi - system_loss_db -
                a.sensitivity_dbm - a.fade_margin_db)
    return min(forward, backward)


def _fspl_db(distance_m: float, frequency_mhz: float) -> float:
    if distance_m <= 0 or frequency_mhz <= 0:
        raise ValueError("传播距离或频率不合法")
    return 32.45 + 20.0 * math.log10(frequency_mhz) + 20.0 * math.log10(
        distance_m / 1000.0)


def _point_certificate(links: Any, moving: tuple[float, float, float],
                       moving_radio: str, fixed: tuple[float, float, float],
                       fixed_radio: str) -> dict[str, Any]:
    try:
        distance = math.dist(moving, fixed)
        obstructed = bool(links.dem.obstructed(moving, fixed))
        budget = _bidirectional_limit(links.radios[moving_radio],
                                      links.radios[fixed_radio],
                                      _finite_number(links.system_loss_db, "系统损耗"))
        loss = (_fspl_db(distance, _finite_number(links.frequency_mhz, "频率")) +
                (_finite_number(links.obstruction_loss_db, "遮挡损耗")
                 if obstructed else 0.0))
        margin = _finite_number(budget - loss, "点链路余量")
    except (ValueError, KeyError, ArithmeticError, TypeError, AttributeError) as exc:
        return {"status": "UNVERIFIED", "method": "exact_point",
                "margin_db": None, "reason": f"无法计算精确点链路：{exc}"}
    if margin > DECISION_GUARD_DB:
        status = "PASS"
    elif margin < -DECISION_GUARD_DB:
        status = "FAIL"
    else:
        status = "UNVERIFIED"
    return {"status": status, "method": "exact_point", "margin_db": margin,
            "obstructed": obstructed,
            "reason": "独立计算 DEM 遮挡、自由空间损耗与双向预算"}


def certify_link_interval(segment: Any, start_s: float, end_s: float,
                          fixed: tuple[float, float, float],
                          moving_radio: str, fixed_radio: str,
                          links: Any) -> dict[str, Any]:
    """证明一段线性轨迹到固定端点的连续链路。

    返回 JSON 可序列化字典。PASS 是全区间证明，FAIL 含确切失联时刻；
    UNVERIFIED 表示保守最坏界不通过、已检查点又未提供反例。
    """
    start = _finite_number(start_s, "区间起点")
    end = _finite_number(end_s, "区间终点")
    if (start < segment.start_s - TIME_TOL_S or
            end > segment.end_s + TIME_TOL_S):
        raise ValueError("待验收区间不在轨迹分段内")
    start = max(start, segment.start_s)
    end = min(end, segment.end_s)
    if not start < end:
        raise ValueError("裁剪后待验收区间非正长")
    if not all(math.isfinite(float(v)) for v in fixed):
        raise ValueError("固定通信端点坐标不合法")
    p0, p1 = segment.position(start), segment.position(end)
    stationary = p0 == p1
    if stationary:
        point = _point_certificate(links, p0, moving_radio, fixed, fixed_radio)
        point.update({"witness_s": start if point["status"] == "FAIL" else None,
                      "worst_distance_m": math.dist(p0, fixed),
                      "pessimistic_obstructed": None})
        return point

    # 对线性轨迹到固定点的距离平方是凸二次函数，闭区间最大值在端点。
    distance = max(math.dist(p0, fixed), math.dist(p1, fixed))
    try:
        budget = _bidirectional_limit(links.radios[moving_radio],
                                      links.radios[fixed_radio],
                                      _finite_number(links.system_loss_db, "系统损耗"))
        frequency = _finite_number(links.frequency_mhz, "频率")
        obstruction = _finite_number(links.obstruction_loss_db, "遮挡损耗")
        if not math.isfinite(budget):
            raise ValueError("双向预算不是有限值")
        # 遮挡状态可以任意变化，但每时刻附加损耗不超过以下上界。
        worst_loss = _fspl_db(distance, frequency) + max(0.0, obstruction)
        worst_margin = budget - worst_loss
    except (ValueError, KeyError, ArithmeticError) as exc:
        return {"status": "UNVERIFIED", "method": "interval_bound",
                "margin_db": None, "worst_distance_m": distance,
                "pessimistic_obstructed": True, "witness_s": None,
                "reason": f"无法计算保守区间界：{exc}"}
    if worst_margin > DECISION_GUARD_DB:
        return {"status": "PASS", "method": "worst_obstruction_endpoint_distance",
                "margin_db": worst_margin, "worst_distance_m": distance,
                "pessimistic_obstructed": True, "witness_s": None,
                "reason": "端点最大距离且全时段按遮挡计损仍满足双向预算"}

    # 点验收只用于查找 FAIL 反例，绝不用来宣称整个连续区间 PASS。
    checks = []
    for instant in (start, (start + end) / 2.0, end):
        point = _point_certificate(links, segment.position(instant), moving_radio,
                                   fixed, fixed_radio)
        checks.append((instant, point))
        if point["status"] == "FAIL":
            return {"status": "FAIL", "method": "exact_point_counterexample",
                    "margin_db": point["margin_db"],
                    "worst_distance_m": distance,
                    "pessimistic_obstructed": True, "witness_s": instant,
                    "reason": "区间内存在精确计算出的失联时刻"}
    checked_margins = [point["margin_db"] for _, point in checks
                       if point["margin_db"] is not None]
    return {"status": "UNVERIFIED", "method": "insufficient_interval_bound",
            "margin_db": worst_margin, "worst_distance_m": distance,
            "pessimistic_obstructed": True, "witness_s": None,
            "checked_min_margin_db": min(checked_margins) if checked_margins else None,
            "reason": "最坏遮挡界不能证明整段；有限点通过不等于连续通过"}


def _issue(issues: list[dict[str, str]], kind: str, detail: str,
           status: str = "FAIL") -> None:
    issues.append({"status": status, "kind": kind, "detail": detail})


def _relay_windows(relay_rows: Iterable[Mapping[str, Any]], links: Any,
                   issues: list[dict[str, str]]) -> dict[str, _RelayWindow]:
    relays: dict[str, _RelayWindow] = {}
    for row_index, row in enumerate(relay_rows, 1):
        relay_id = str(row.get("中继架次编号", "")).strip()
        if not relay_id or relay_id in relays:
            _issue(issues, "relay_id", f"中继表第 {row_index} 行编号为空或重复")
            continue
        try:
            start = _finite_number(row["开始时刻（s）"], "中继开始")
            ready = _finite_number(row["建链完成时刻（s）"], "建链完成")
            service_end = _finite_number(row["服务结束时刻（s）"], "服务结束")
            returned = _finite_number(row["返回O01时刻（s）"], "中继返回")
            lon = _finite_number(row["悬停经度（°）"], "中继悬停经度")
            lat = _finite_number(row["悬停纬度（°）"], "中继悬停纬度")
            altitude = _finite_number(row["悬停海拔（m）"], "中继悬停海拔")
            if not 0 <= start < ready <= service_end <= returned:
                raise ValueError("中继准备/建链/服务/返航时序无效")
            x, y = links.dem.to_xy.transform(lon, lat)
            hover = (_finite_number(x, "悬停 x"), _finite_number(y, "悬停 y"), altitude)
            backhaul = _point_certificate(links, hover, "relay_backhaul",
                                           links.gateway, "gateway")
        except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
            _issue(issues, "relay_row", f"中继表第 {row_index} 行无效：{exc}")
            continue
        relays[relay_id] = _RelayWindow(relay_id, hover, ready, service_end,
                                        backhaul["status"], backhaul["margin_db"])
        if backhaul["status"] != "PASS":
            _issue(issues, "relay_backhaul",
                   f"{relay_id} 固定回传链路 {backhaul['status']}：{backhaul['reason']}",
                   backhaul["status"])
    return relays


def _allocations(comm_rows: Iterable[Mapping[str, Any]],
                 issues: list[dict[str, str]]) -> list[_Allocation]:
    allocations = []
    for row_index, row in enumerate(comm_rows, 1):
        try:
            sortie_id = str(row["运输架次编号"]).strip()
            stage = str(row["通信阶段"]).strip()
            mode = str(row["保障方式"]).strip()
            relay_id = str(row.get("中继架次编号", "")).strip()
            start = _finite_number(row["开始时刻（s）"], "通信开始")
            end = _finite_number(row["结束时刻（s）"], "通信结束")
            if not sortie_id or not stage or not start < end:
                raise ValueError("架次/阶段为空或通信区间非正长")
            if mode == "直连" and relay_id:
                raise ValueError("直连行不得填写中继架次")
            if mode == "中继" and not relay_id:
                raise ValueError("中继行缺少中继架次编号")
            if mode not in {"直连", "中继"}:
                raise ValueError(f"未知保障方式 {mode!r}")
        except (KeyError, ValueError, TypeError) as exc:
            _issue(issues, "comm_row", f"通信表第 {row_index} 行无效：{exc}")
            continue
        allocations.append(_Allocation(row_index, sortie_id, stage,
                                       start, end, mode, relay_id))
    return allocations


def verify_continuous_coverage(
    trajectories: Mapping[str, list[Any]], links: Any,
    comm_rows: Iterable[Mapping[str, Any]],
    relay_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """核实 Q3 官方通信表声明是否覆盖各运输架次的完整飞行与投送轨迹。

    数据行使用官方中文表头；时间建议以可往返的浮点文本（如
    ``repr(float)``）输出。小于等于 ``TIME_TOL_S`` 的时序舍入差会被
    吸附到上一行/轨迹端点，并对吸附后的完整区间重新证明链路；超过
    容差的空档或重叠一律 FAIL。返回值可用
    ``json.dump(..., allow_nan=False)`` 保存。
    """
    issues: list[dict[str, str]] = []
    if not trajectories:
        _issue(issues, "empty_trajectories", "未提供任何运输轨迹")
    relays = _relay_windows(relay_rows, links, issues)
    allocations = _allocations(comm_rows, issues)
    by_sortie: dict[str, list[_Allocation]] = defaultdict(list)
    used_relays: set[str] = set()
    for row in allocations:
        by_sortie[row.sortie_id].append(row)
        if row.mode == "中继":
            used_relays.add(row.relay_id)
    for relay_id in sorted(set(relays) - used_relays):
        _issue(issues, "unused_relay", f"中继架次 {relay_id} 未保障任何运输区间")
    for sortie_id in sorted(set(by_sortie) - set(trajectories)):
        _issue(issues, "unknown_sortie", f"通信表引用未知运输架次 {sortie_id}")

    checks: list[dict[str, Any]] = []
    for sortie_id, segments in sorted(trajectories.items()):
        ordered_segments = sorted(segments, key=lambda seg: seg.start_s)
        if not ordered_segments:
            _issue(issues, "empty_trajectory", f"{sortie_id} 没有飞行轨迹")
            continue
        for before, after in zip(ordered_segments, ordered_segments[1:]):
            if abs(before.end_s - after.start_s) > TIME_TOL_S:
                _issue(issues, "trajectory_discontinuity",
                       f"{sortie_id} 轨迹在 {before.end_s} 与 {after.start_s} 之间不连续")
        rows = sorted(by_sortie[sortie_id], key=lambda row: (row.start_s, row.end_s))
        cursor = ordered_segments[0].start_s
        if not rows:
            _issue(issues, "missing_coverage", f"{sortie_id} 没有通信保障行")
            continue
        for row_index, raw_row in enumerate(rows):
            row = raw_row
            if abs(row.start_s - cursor) <= TIME_TOL_S:
                row = replace(row, start_s=cursor)
            else:
                relation = "空档" if row.start_s > cursor else "重叠"
                _issue(issues, "coverage_partition",
                       f"{sortie_id} 通信保障在 {cursor} 与 {row.start_s} 之间{relation}")
            if (row_index == len(rows) - 1 and
                    abs(row.end_s - ordered_segments[-1].end_s) <= TIME_TOL_S):
                row = replace(row, end_s=ordered_segments[-1].end_s)
            if row.start_s >= row.end_s:
                _issue(issues, "coverage_partition",
                       f"{sortie_id} 第 {row.row_index} 行吸附后区间非正长")
                continue
            cursor = row.end_s
            if (row.start_s < ordered_segments[0].start_s - TIME_TOL_S or
                    row.end_s > ordered_segments[-1].end_s + TIME_TOL_S):
                _issue(issues, "coverage_scope", f"{sortie_id} 第 {row.row_index} 行超出飞行轨迹")
            relay = relays.get(row.relay_id) if row.mode == "中继" else None
            if row.mode == "中继":
                if relay is None:
                    _issue(issues, "relay_reference",
                           f"{sortie_id} 第 {row.row_index} 行引用未知中继 {row.relay_id}")
                    continue
                if row.start_s < relay.ready_s or row.end_s > relay.service_end_s:
                    _issue(issues, "relay_window",
                           f"{sortie_id} 第 {row.row_index} 行不在 {row.relay_id} 的已建链服务时段")
            intersections = 0
            for segment in ordered_segments:
                start = max(row.start_s, segment.start_s)
                end = min(row.end_s, segment.end_s)
                if abs(start - segment.start_s) <= TIME_TOL_S:
                    start = segment.start_s
                if abs(end - segment.end_s) <= TIME_TOL_S:
                    end = segment.end_s
                if not start < end or (row.stage != segment.stage and
                                       end - start <= TIME_TOL_S):
                    continue
                intersections += 1
                if row.stage != segment.stage:
                    _issue(issues, "stage_mismatch",
                           f"{sortie_id} 第 {row.row_index} 行写 {row.stage}，轨迹为 {segment.stage}")
                fixed = links.gateway if relay is None else relay.hover
                target_radio = "gateway" if relay is None else "relay_access"
                try:
                    certificate = certify_link_interval(
                        segment, start, end, fixed, "transport", target_radio, links)
                except (ValueError, KeyError, ArithmeticError) as exc:
                    certificate = {"status": "UNVERIFIED", "method": "invalid_interval",
                                   "margin_db": None, "worst_distance_m": None,
                                   "pessimistic_obstructed": None, "witness_s": None,
                                   "reason": f"无法复核该通信分段：{exc}"}
                checks.append({"sortie_id": sortie_id, "stage": segment.stage,
                               "start_s": start, "end_s": end,
                               "mode": row.mode, "relay_id": row.relay_id,
                               "comm_row_index": row.row_index, **certificate})
            if not intersections:
                _issue(issues, "coverage_scope",
                       f"{sortie_id} 第 {row.row_index} 行未覆盖任何飞行轨迹")
        if abs(cursor - ordered_segments[-1].end_s) > TIME_TOL_S:
            _issue(issues, "coverage_partition",
                   f"{sortie_id} 通信保障终点 {cursor} 不等于返航 {ordered_segments[-1].end_s}")

    statuses = [item["status"] for item in issues] + [item["status"] for item in checks]
    status = ("FAIL" if "FAIL" in statuses else
              "UNVERIFIED" if "UNVERIFIED" in statuses else "PASS")
    certified_margins = [item["margin_db"] for item in checks
                         if item["status"] == "PASS" and item["margin_db"] is not None]
    return {
        "status": status,
        "time_tolerance_s": TIME_TOL_S,
        "certified_interval_count": sum(x["status"] == "PASS" for x in checks),
        "failed_interval_count": sum(x["status"] == "FAIL" for x in checks),
        "unverified_interval_count": sum(x["status"] == "UNVERIFIED" for x in checks),
        "worst_certified_margin_db": min(certified_margins) if certified_margins else None,
        "issues": issues,
        "checks": checks,
    }

RESOURCE_RELAY_COLUMNS = (
    "中继架次编号", "中继无人机编号", "能源组件编号", "开始时刻（s）",
    "悬停经度（°）", "悬停纬度（°）", "悬停海拔（m）", "建链完成时刻（s）",
    "服务结束时刻（s）", "返回O01时刻（s）", "架次能耗（kWh）",
)
RESOURCE_TIME_TOL_S = 2e-5
ENERGY_TOL_KWH = 2e-6
HEIGHT_TOL_M = 2e-3
RESOURCE_TOL_S = 1e-8
HARD_ENERGY_TOL_KWH = 1e-9


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _number(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 不是数值") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} 不是有限数值")
    return value


def _recharge_duration_s(return_soc: float, full_charge_s: float) -> float:
    """Official 0–90% / 90–100% split, recomputed without scheduler helpers."""
    if not (0.0 <= return_soc <= 1.0 and full_charge_s > 0):
        raise ValueError("SOC 或充满时长无效")
    if return_soc < 0.9:
        return full_charge_s * (0.65 * (0.9 - return_soc) / 0.9 + 0.35)
    return full_charge_s * 0.35 * (1.0 - return_soc) / 0.1


def _flight(arc: dict[str, float], params: dict[str, float]) -> tuple[float, float]:
    """Reconstruct one leg from length and ascent, not the scheduler's cache."""
    climb = float(arc["climb_m"])
    descent = float(arc["descent_m"])
    distance = float(arc["distance_m"])
    cruise_s = distance / params["cruise_speed_mps"]
    flight_s = (climb / params["climb_speed_mps"] + cruise_s +
                descent / params["descent_speed_mps"])
    energy_kwh = (params["cruise_power_kw"] * cruise_s / 3600.0 +
                  params["takeoff_mass_kg"] * 9.80665 * climb /
                  (3_600_000.0 * params["climb_efficiency"]))
    return flight_s, energy_kwh


def validate_relay_rows(relay_rows: list[dict[str, str]], data_run: Path,
                        links: LinkEvaluator, dem: DemGrid) -> dict[str, Any]:
    """Audit official Q3 relay rows; return a JSON-serializable PASS/FAIL report.

    A missing/invalid source table or a malformed sortie is a FAIL, not a silent
    skip. The resource clocks are rebuilt chronologically from each audited row.
    """
    checks: list[dict[str, str]] = []
    issues: list[dict[str, str]] = []

    def add(name: str, passed: bool, detail: str = "",
            mission_id: str = "") -> None:
        record = {"check": name, "status": "PASS" if passed else "FAIL",
                  "mission_id": mission_id, "detail": detail}
        checks.append(record)
        if not passed:
            issues.append(record)

    def result(total_energy: float = 0.0, audited_count: int = 0,
               uav_count: int = 0, component_count: int = 0) -> dict[str, Any]:
        return {
            "status": "FAIL" if issues else "PASS",
            "checks": checks,
            "issues": issues,
            "first_failure": issues[0] if issues else None,
            "summary": {
                "relay_sorties": len(relay_rows) if isinstance(relay_rows, list) else 0,
                "audited_sorties": audited_count,
                "uav_count": uav_count,
                "component_count": component_count,
                "total_reported_energy_kwh": round(total_energy, 9),
            },
        }

    if not isinstance(relay_rows, list) or any(not isinstance(row, dict)
                                                for row in relay_rows):
        add("table_type", False, "中继架次必须是字典行列表")
        return result()
    for index, row in enumerate(relay_rows, 1):
        missing = [field for field in RESOURCE_RELAY_COLUMNS if field not in row]
        if missing:
            add("official_columns", False,
                f"第 {index} 行缺少字段：{', '.join(missing)}")
    if issues:
        return result()

    try:
        clean = data_run / "clean"
        type_rows = _read_csv(clean / "relay_type.csv")
        uav_rows = _read_csv(clean / "relay_uavs.csv")
        unit_rows = _read_csv(clean / "relay_energy_units.csv")
        pool_rows = _read_csv(clean / "relay_energy_pool.csv")
        center_rows = [row for row in _read_csv(clean / "nodes.csv")
                       if row.get("node_id") == "O01"]
        if not (len(type_rows) == 1 and len(pool_rows) == 1 and
                len(center_rows) == 1):
            raise ValueError("R 型号、能源池或 O01 节点缺失/重复")
        if len(uav_rows) != 2 or len(unit_rows) != 6:
            raise ValueError("中继资源不是 2 架无人机和 6 个能源组件")
        required = (
            "takeoff_mass_kg", "cruise_speed_mps", "cruise_power_kw",
            "energy_use_kwh", "return_soc_min", "prep_s", "link_setup_s",
            "turnaround_s", "climb_speed_mps", "descent_speed_mps",
            "climb_efficiency", "hover_power_kw", "comm_extra_power_kw",
            "max_hover_agl_m",
        )
        params = {key: _number(type_rows[0][key], key) for key in required}
        params["full_charge_s"] = _number(pool_rows[0]["full_charge_s"],
                                            "full_charge_s")
        if any(params[key] <= 0 for key in (
                "takeoff_mass_kg", "cruise_speed_mps", "cruise_power_kw",
                "energy_use_kwh", "climb_speed_mps", "descent_speed_mps",
                "climb_efficiency", "full_charge_s", "max_hover_agl_m")):
            raise ValueError("飞行/充电参数必须为正")
        if not (0 < params["return_soc_min"] < 1):
            raise ValueError("返航 SOC 下限无效")
        if any(params[key] < 0 for key in ("prep_s", "link_setup_s",
                                            "turnaround_s", "hover_power_kw",
                                            "comm_extra_power_kw")):
            raise ValueError("时间或悬停功率参数为负")
        uav_ids = [row["uav_id"] for row in uav_rows]
        unit_ids = [row["unit_id"] for row in unit_rows]
        if len(set(uav_ids)) != 2 or len(set(unit_ids)) != 6:
            raise ValueError("中继无人机或能源组件编号重复")
        center = center_rows[0]
        home = (_number(center["x_m"], "O01 x_m"),
                _number(center["y_m"], "O01 y_m"),
                _number(center["operation_alt_m"], "O01 operation_alt_m"))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        add("official_resources", False, str(exc))
        return result()
    add("official_resources", True, "2 架中继机、6 个组件和 O01 坐标已加载")
    add("gateway_location", math.hypot(home[0] - links.gateway[0],
                                        home[1] - links.gateway[1]) <= 2.0,
        "G01 的平面坐标应与 O01 相同")

    parsed: list[dict[str, Any]] = []
    seen_missions: set[str] = set()
    total_energy = 0.0
    numeric_fields = {
        "start": "开始时刻（s）", "lon": "悬停经度（°）",
        "lat": "悬停纬度（°）", "alt": "悬停海拔（m）",
        "ready": "建链完成时刻（s）", "end": "服务结束时刻（s）",
        "return": "返回O01时刻（s）", "energy": "架次能耗（kWh）",
    }
    for index, row in enumerate(relay_rows, 1):
        mission_id = row["中继架次编号"].strip()
        if not mission_id or mission_id in seen_missions:
            add("mission_id_unique", False,
                f"第 {index} 行中继架次编号为空或重复", mission_id)
        else:
            add("mission_id_unique", True, mission_id=mission_id)
            seen_missions.add(mission_id)
        uav_id, unit_id = row["中继无人机编号"].strip(), row["能源组件编号"].strip()
        add("known_uav", uav_id in uav_ids, uav_id, mission_id)
        add("known_component", unit_id in unit_ids, unit_id, mission_id)
        try:
            values = {name: _number(row[column], column)
                      for name, column in numeric_fields.items()}
        except ValueError as exc:
            add("finite_numeric_fields", False, str(exc), mission_id)
            continue
        add("finite_numeric_fields", True, mission_id=mission_id)
        add("geographic_coordinates",
            -180.0 <= values["lon"] <= 180.0 and
            -90.0 <= values["lat"] <= 90.0,
            "悬停点经纬度应落在合法范围", mission_id)
        if not (-180.0 <= values["lon"] <= 180.0 and
                -90.0 <= values["lat"] <= 90.0):
            continue
        parsed.append({"mission_id": mission_id, "uav_id": uav_id,
                       "unit_id": unit_id, **values})
        total_energy += values["energy"]

    uav_ready = {uav_id: 0.0 for uav_id in uav_ids}
    unit_ready = {unit_id: 0.0 for unit_id in unit_ids}
    audited = 0
    for item in sorted(parsed, key=lambda entry: (entry["start"],
                                                   entry["mission_id"])):
        mission_id = item["mission_id"]
        start, ready = item["start"], item["ready"]
        end, returned = item["end"], item["return"]
        energy = item["energy"]
        add("time_order", start >= -RESOURCE_TOL_S and
            ready >= start - RESOURCE_TOL_S and
            end >= ready - RESOURCE_TOL_S and
            returned >= end - RESOURCE_TOL_S,
            f"start={start}, ready={ready}, end={end}, return={returned}",
            mission_id)
        add("nonnegative_energy", energy >= -HARD_ENERGY_TOL_KWH,
            f"reported={energy:.9f} kWh", mission_id)
        try:
            x, y = dem.to_xy.transform(item["lon"], item["lat"])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("经纬度投影得到非有限坐标")
            hover = (x, y, item["alt"])
            ground = dem.terrain_at(x, y)
            agl = hover[2] - ground
            add("hover_agl", agl > -HEIGHT_TOL_M and
                agl <= params["max_hover_agl_m"] + HEIGHT_TOL_M,
                f"AGL={agl:.6f} m, limit={params['max_hover_agl_m']:.6f} m",
                mission_id)
            backhaul = links.backhaul(hover)
            add("bidirectional_backhaul", backhaul.available,
                f"margin={backhaul.margin_db:.6f} dB", mission_id)
            outward = dem.arc_geometry(home, hover)
            inward = dem.arc_geometry(hover, home)
            add("relay_arc_altitude",
                hover[2] <= min(outward["cruise_alt_m"],
                                inward["cruise_alt_m"]) + HEIGHT_TOL_M,
                           "中继航段巡航高度不能低于悬停点", mission_id)
            outward_s, outward_energy = _flight(outward, params)
            inward_s, inward_energy = _flight(inward, params)
            setup = params["link_setup_s"]
            expected_ready = start + params["prep_s"] + outward_s + setup
            expected_return = end + inward_s
            expected_energy = (outward_energy + inward_energy +
                               (params["hover_power_kw"] +
                                params["comm_extra_power_kw"]) *
                               (setup + max(0.0, end - ready)) / 3600.0)
            add("link_ready_replay",
                abs(ready - expected_ready) <= RESOURCE_TIME_TOL_S,
                f"reported={ready:.9f}, replay={expected_ready:.9f}",
                mission_id)
            add("return_time_replay",
                abs(returned - expected_return) <= RESOURCE_TIME_TOL_S,
                f"reported={returned:.9f}, replay={expected_return:.9f}",
                mission_id)
            add("energy_replay",
                abs(energy - expected_energy) <= ENERGY_TOL_KWH,
                f"reported={energy:.9f}, replay={expected_energy:.9f} kWh",
                mission_id)
        except (ValueError, KeyError, OverflowError, ZeroDivisionError) as exc:
            add("physical_replay", False, str(exc), mission_id)

        soc = 1.0 - energy / params["energy_use_kwh"]
        add("return_soc",
            soc >= params["return_soc_min"] -
            HARD_ENERGY_TOL_KWH / params["energy_use_kwh"] and
            soc <= 1.0 + HARD_ENERGY_TOL_KWH / params["energy_use_kwh"],
            f"SOC={soc:.9f}, minimum={params['return_soc_min']:.9f}",
            mission_id)
        uav_id, unit_id = item["uav_id"], item["unit_id"]
        if uav_id in uav_ready:
            add("uav_turnaround", start >= uav_ready[uav_id] - RESOURCE_TOL_S,
                f"start={start:.9f}, earliest={uav_ready[uav_id]:.9f}",
                mission_id)
            uav_ready[uav_id] = max(uav_ready[uav_id],
                                    returned + params["turnaround_s"])
        if unit_id in unit_ready:
            add("component_recharge", start >= unit_ready[unit_id] - RESOURCE_TOL_S,
                f"start={start:.9f}, earliest={unit_ready[unit_id]:.9f}",
                mission_id)
            if 0.0 <= soc <= 1.0:
                charging = _recharge_duration_s(soc, params["full_charge_s"])
                unit_ready[unit_id] = max(unit_ready[unit_id],
                                          returned + charging)
            else:
                # Invalid SOC must not leave the component apparently reusable.
                unit_ready[unit_id] = math.inf
        audited += 1
    return result(total_energy, audited, len(uav_ids), len(unit_ids))

if __name__ == "__main__":
    raise SystemExit(main())
