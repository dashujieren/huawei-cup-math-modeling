"""问题三方案 1：从 80 箱原始数据联合安排运输与通信中继。

候选组批可按通信及资源冲突动态拆分；运输与中继使用区间日历并行排程。
有限候选和搜索预算不证明最优；只有独立复核全部通过才写 Q3_READY.txt。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
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
    ax.set_title("Q3 scheme 1: certified communication intervals")
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


def _save_verification(table_dir: Path, validation: dict[str, Any]) -> None:
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
               {"check_id": "SCHEME1_STATUS", "status": validation["status"],
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


def _task_order(ctx: Context, tasks: list[RouteTask], mode: str) -> list[RouteTask]:
    def key(task: RouteTask) -> tuple[float, float, float, str]:
        options = ctx.options(task.route)
        latest = max(_latest_hard_start(ctx, task.route, option)
                     for option in options.values())
        expected = min(float(ctx.boxes[bid]["expected_s"]) -
                       float(option["completion_offsets"][bid])
                       for option in options.values() for bid in task.route.box_ids)
        urgency = (latest if mode == "hard_slack" else
                   min((float(ctx.boxes[bid]["hard_deadline_s"])
                        for bid in task.route.box_ids
                        if not pd.isna(ctx.boxes[bid]["hard_deadline_s"])),
                       default=math.inf))
        return urgency, expected, -len(task.route.box_ids), task.batch_id
    return sorted(tasks, key=key)


def _source_slices_from_raw(ctx: Context, positions: dict[str, tuple[float, float, float]],
                            links: LinkEvaluator, step_s: float,
                            route_sets: list[tuple[str, list[RouteTask]]]) -> list[TimeSlice]:
    """用原始箱构造各服务区、各可用机型的飞行空间走廊，寻找中继候选点。"""
    by_zone: dict[str, list[str]] = defaultdict(list)
    for box_id, box in ctx.boxes.items():
        by_zone[str(box["zone_id"])].append(box_id)
    slices: list[TimeSlice] = []
    for zone, ids in sorted(by_zone.items()):
        for kind in sorted(ctx.models):
            feasible = next((ctx.route([(zone, [bid])]) for bid in ids
                             if kind in ctx.options(ctx.route([(zone, [bid])]))), None)
            if feasible is None:
                continue
            option = ctx.options(feasible)[kind]
            segments = _relative_trajectory(ctx, feasible, option, positions)
            slices.extend(_certified_slices(f"SRC_{zone}_{kind}", segments,
                                            links, step_s))
    paired = next((tasks for name, tasks in route_sets
                   if name == "neighbor_pair"), [])
    for task in paired:
        if len(task.route.zones) < 2:
            continue
        for kind, option in ctx.options(task.route).items():
            segments = _relative_trajectory(ctx, task.route, option, positions)
            slices.extend(_certified_slices(
                f"SRC_{task.batch_id}_{kind}", segments, links, step_s))
    if not slices:
        raise ValueError("原始货箱未形成可计算的运输走廊")
    return slices


def _state_score(state: JointState) -> tuple[float, float, int, float]:
    delay = sum(d["priority"] * d["soft_lateness_s"] for d in state.deliveries
                if d["hard_deadline_s"] is None)
    last = max([row["return_s"] for row in state.sorties] +
               [mission.return_s for mission in state.missions], default=0.0)
    energy = sum(row["energy_kwh"] for row in state.sorties) + sum(
        mission.energy_kwh for mission in state.missions)
    return delay, last, len(state.missions), energy










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


def _intersects(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 - EPS and b0 < a1 - EPS


def _transport_slot(state: JointState, uav_id: str, battery_id: str,
                    earliest: float, duration: float, charge: float) -> float:
    """按资源占用区间找空档；无需沿用其他架次的全局起飞顺序。"""
    start = max(0.0, earliest)
    for _ in range(2 * len(state.sorties) + 4):
        blocking = []
        for old in state.sorties:
            if old["uav_id"] == uav_id and _intersects(
                    start, start + duration, old["start_s"], old["return_s"]):
                blocking.append(old["return_s"])
            if old["battery_id"] == battery_id and _intersects(
                    start, start + duration + charge,
                    old["start_s"], old["charge_end_s"]):
                blocking.append(old["charge_end_s"])
        if not blocking:
            return start
        start = max(blocking)
    raise RuntimeError("运输资源空档查找未收敛")


def _relay_slot_shift(missions: list[RelayMission], uav_id: str, unit_id: str,
                      start: float, returned: float, energy: float,
                      relay: dict[str, Any]) -> float:
    params = relay["params"]
    soc = 1.0 - energy / params["energy_use_kwh"]
    uav_end = returned + params["turnaround_s"]
    unit_end = returned + charge_to_full_s(soc, params["full_charge_s"])
    shifts = []
    for old in missions:
        if old.uav_id == uav_id and _intersects(
                start, uav_end, old.start_s,
                old.return_s + params["turnaround_s"]):
            shifts.append(old.return_s + params["turnaround_s"] - start)
        if old.unit_id == unit_id:
            old_soc = 1.0 - old.energy_kwh / params["energy_use_kwh"]
            old_end = old.return_s + charge_to_full_s(old_soc, params["full_charge_s"])
            if _intersects(start, unit_end, old.start_s, old_end):
                shifts.append(old_end - start)
    return max(shifts, default=0.0)


def _relay_options_fast(state: JointState, profile: RouteProfile,
                        shifted_slices: list[TimeSlice],
                        candidates: list[RelayCandidate], links: LinkEvaluator,
                        relay: dict[str, Any], candidate_limit: int,
                        access_cache: dict[tuple[Any, ...], bool],
                        limit: int = 4
                        ) -> tuple[list[tuple[list[RelayMission], dict[int, str]]],
                                   float | None, dict[str, int]]:
    """只遍历预先证明可覆盖的点；时间平移不会改变空间链路。"""
    blind = [(item, local_id) for local_id, item in enumerate(shifted_slices)
             if not item.direct_available]
    variants = [(list(state.missions), dict(state.assigned))]
    params = relay["params"]
    power = params["hover_power_kw"] + params["comm_extra_power_kw"]
    reasons: dict[str, int] = defaultdict(int)
    suggested: float | None = None
    for place, (first, local_id) in enumerate(blind):
        expanded = []
        for missions, assigned in variants:
            if first.slice_id in assigned:
                expanded.append((missions, assigned))
                continue
            shared = next((mission for mission in missions
                           if mission.ready_s <= first.start_s + EPS and
                           first.end_s <= mission.service_end_s + EPS and
                           _access_certified(mission.candidate, first,
                                             links, access_cache)), None)
            if shared is not None:
                copied = dict(assigned)
                copied[first.slice_id] = shared.mission_id
                expanded.append((missions, copied))
                continue
            # 后续运输任务可以沿用已经起飞的中继，并把原服务窗口延长。
            # 延长后重新核算悬停能耗、返航时刻以及机体/组件日历。
            for previous in missions:
                if not (previous.ready_s <= first.start_s + EPS and
                        previous.service_end_s < first.end_s - EPS):
                    continue
                if not _access_certified(previous.candidate, first,
                                         links, access_cache):
                    continue
                new_end = first.end_s
                if new_end - previous.ready_s > previous.candidate.max_service_s + EPS:
                    reasons["relay_service_limit"] += 1
                    continue
                new_energy = (previous.candidate.travel_energy_kwh + power *
                              (params["link_setup_s"] +
                               new_end - previous.ready_s) / 3600.0)
                if new_energy > ((1.0 - params["return_soc_min"]) *
                                 params["energy_use_kwh"] + EPS):
                    reasons["relay_energy_limit"] += 1
                    continue
                new_return = new_end + previous.candidate.homeward_s
                others = [item for item in missions
                          if item.mission_id != previous.mission_id]
                if _relay_slot_shift(others, previous.uav_id,
                                     previous.unit_id, previous.start_s,
                                     new_return, new_energy, relay) > EPS:
                    reasons["relay_resource_calendar"] += 1
                    continue
                extended = replace(previous, service_end_s=new_end,
                                   return_s=new_return, energy_kwh=new_energy)
                copied = dict(assigned)
                copied[first.slice_id] = previous.mission_id
                expanded.append(([
                    extended if item.mission_id == previous.mission_id else item
                    for item in missions], copied))
            ranked = profile.ranked_sites.get(local_id, ())
            if not ranked:
                reasons["no_spatial_relay_site"] += 1
                continue
            choices = []
            for site_index in ranked[:candidate_limit]:
                candidate = candidates[site_index]
                upper = first.start_s + candidate.max_service_s
                if first.end_s > upper + EPS:
                    reasons["relay_service_limit"] += 1
                    continue
                included = [first]
                for later, later_local in blind[place + 1:]:
                    if later.slice_id in assigned:
                        continue
                    if (later.end_s > upper + EPS or
                            site_index not in profile.support.get(later_local, set())):
                        break
                    included.append(later)
                end = included[-1].end_s
                energy = candidate.travel_energy_kwh + power * (
                    params["link_setup_s"] + end - first.start_s) / 3600.0
                if energy > (1.0 - params["return_soc_min"]) * params["energy_use_kwh"] + EPS:
                    reasons["relay_energy_limit"] += 1
                    continue
                launch = first.start_s - (params["prep_s"] + candidate.outward_s +
                                          params["link_setup_s"])
                if launch < -EPS:
                    wait = -launch
                    suggested = wait if suggested is None else min(suggested, wait)
                    reasons["relay_needs_earlier_launch"] += 1
                    continue
                returned = end + candidate.homeward_s
                for uav_id in relay["uav_ids"]:
                    for unit_id in relay["unit_ids"]:
                        wait = _relay_slot_shift(missions, uav_id, unit_id,
                                                 launch, returned, energy, relay)
                        if wait > EPS:
                            suggested = wait if suggested is None else min(suggested, wait)
                            reasons["relay_resource_calendar"] += 1
                            continue
                        use_uav = sum(m.uav_id == uav_id for m in missions)
                        use_unit = sum(m.unit_id == unit_id for m in missions)
                        choices.append(((-len(included), energy, use_uav,
                                         use_unit, returned, candidate.candidate_id),
                                        candidate, uav_id, unit_id, included,
                                        launch, end, energy))
            choices.sort(key=lambda row: row[0])
            site_uses: dict[str, int] = defaultdict(int)
            for _, candidate, uav_id, unit_id, included, launch, end, energy in choices:
                if site_uses[candidate.candidate_id] >= 2:
                    continue
                site_uses[candidate.candidate_id] += 1
                mission_id = f"R{len(missions) + 1:03d}"
                mission = RelayMission(mission_id, uav_id, unit_id, candidate,
                                       launch, first.start_s, end,
                                       end + candidate.homeward_s, energy)
                copied = dict(assigned)
                copied.update({item.slice_id: mission_id for item in included})
                expanded.append(([*missions, mission], copied))
                if len(expanded) >= limit:
                    break
        if not expanded:
            return [], suggested, dict(reasons)
        expanded.sort(key=lambda pair: (
            len(pair[0]) - len(state.missions),
            sum(m.energy_kwh for m in pair[0][len(state.missions):]),
            max((m.return_s for m in pair[0]), default=0.0)))
        variants = expanded[:limit]
    return variants, 0.0, dict(reasons)


def _insert_task(ctx: Context, state: JointState, task: RouteTask,
                 candidates: list[RelayCandidate], links: LinkEvaluator,
                 relay: dict[str, Any], positions: dict[str, tuple[float, float, float]],
                 step_s: float, site_tree: cKDTree | None,
                 profiles: dict[tuple[str, Any, str], RouteProfile],
                 access_cache: dict[tuple[Any, ...], bool], limit: int,
                 candidate_limit: int, start_probes: int, soft_horizon_s: float,
                 earliest_start_s: float = 0.0
                 ) -> tuple[list[JointState], dict[str, int]]:
    if not math.isfinite(earliest_start_s) or earliest_start_s < 0:
        raise ValueError("运输候选最早出发时刻必须非负且有限")
    options = ctx.options(task.route)
    reasons: dict[str, int] = defaultdict(int)
    if not options:
        return [], {"no_feasible_transport_type": 1}
    choices: list[tuple[tuple[Any, ...], JointState]] = []
    for kind in sorted(options, key=lambda k: (options[k]["duration_s"], k)):
        option = options[kind]
        latest = _latest_hard_start(ctx, task.route, option)
        if latest < -EPS:
            reasons["deadline_before_time_zero"] += 1
            continue
        key = task.batch_id, task.route, kind
        if key not in profiles:
            profiles[key] = _route_profile(
                ctx, task, kind, option, positions, links, step_s,
                candidates, site_tree, access_cache)
        profile = profiles[key]
        if profile.uncovered_slices:
            reasons["route_has_uncoverable_communication_slices"] += 1
            continue
        pairs = []
        for uav in ctx.uavs:
            if uav.type_id != kind:
                continue
            for battery in ctx.batteries:
                if battery.type_id != kind:
                    continue
                charge = charge_to_full_s(float(option["return_soc"]),
                                          battery.full_charge_s)
                first = _transport_slot(state, uav.uav_id, battery.battery_id,
                                        0.0, float(option["duration_s"]), charge)
                pairs.append((first, uav.uav_id, battery.battery_id,
                              battery, charge))
        pairs.sort(key=lambda row: (row[0], row[1], row[2]))
        used: dict[str, int] = defaultdict(int)
        considered = 0
        for earliest, uav_id, _, battery, charge in pairs:
            if used[uav_id] >= 2 or considered >= 8:
                continue
            used[uav_id] += 1
            considered += 1
            start = max(earliest, earliest_start_s)
            horizon = latest if math.isfinite(latest) else soft_horizon_s
            successes = 0
            for _ in range(start_probes):
                start = _transport_slot(state, uav_id, battery.battery_id,
                                        start, float(option["duration_s"]), charge)
                if start > horizon + EPS:
                    reasons["deadline_or_horizon"] += 1
                    break
                shifted, shifted_slices = _shift_template(
                    profile.segments, profile.slices, start, len(state.slices))
                variants, wait, failed = _relay_options_fast(
                    state, profile, shifted_slices, candidates, links, relay,
                    candidate_limit, access_cache)
                for name, count in failed.items():
                    reasons[name] += count
                if variants:
                    sortie, deliveries = _make_sortie(
                        ctx, task, kind, option, uav_id, battery, start)
                    for missions, assigned in variants:
                        trial = JointState(
                            [*state.sorties, sortie], [*state.deliveries, *deliveries],
                            {**state.trajectories, task.batch_id: shifted},
                            [*state.slices, *shifted_slices], assigned, missions)
                        choices.append(((*_state_score(trial), start, kind,
                                         uav_id, battery.battery_id), trial))
                    successes += 1
                    if successes >= 2:
                        break
                    start += 120.0
                elif failed.get("no_spatial_relay_site") or failed.get(
                        "route_has_uncoverable_communication_slices"):
                    break
                else:
                    start += max(15.0, float(wait) if wait is not None else 90.0)
    choices.sort(key=lambda row: row[0])
    distinct: list[JointState] = []
    seen: set[tuple[Any, ...]] = set()
    for _, trial in choices:
        current = trial.sorties[-1]
        signature = (current["type_id"], current["uav_id"],
                     current["battery_id"], round(current["start_s"] / 60),
                     tuple(sorted((m.candidate.candidate_id, m.uav_id,
                                   m.unit_id) for m in trial.missions)))
        if signature in seen:
            continue
        seen.add(signature)
        distinct.append(trial)
        if len(distinct) >= limit:
            break
    return distinct, dict(reasons)


def _attempt_schedule(ctx: Context, tasks: list[RouteTask],
                      candidates: list[RelayCandidate], relay: dict[str, Any],
                      links: LinkEvaluator, dem: DemGrid,
                      positions: dict[str, tuple[float, float, float]],
                      step_s: float, site_tree: cKDTree | None,
                      beam_width: int, candidate_limit: int,
                      start_probes: int, max_repairs: int,
                      soft_horizon_s: float,
                      access_cache: dict[tuple[Any, ...], bool],
                      grouping: str
                      ) -> tuple[JointState, dict[str, Any]]:
    beam = [_empty_joint_state()]
    profiles: dict[tuple[str, Any, str], RouteProfile] = {}
    pending = _task_order(ctx, tasks, "hard_slack")
    repairs = 0
    index = 0
    started = perf_counter()
    while index < len(pending):
        task = pending[index]
        successors: list[JointState] = []
        failures: dict[str, int] = defaultdict(int)
        # 先尝试少量已证明可达的中继点。只有本批排不进时才扩大范围。
        limits = [min(candidate_limit, len(candidates)),
                  min(max(4 * candidate_limit, 64), len(candidates)),
                  len(candidates)]
        for widen, site_limit in enumerate(dict.fromkeys(limits)):
            for state in beam:
                states, reasons = _insert_task(
                    ctx, state, task, candidates, links, relay, positions,
                    step_s, site_tree, profiles, access_cache,
                    max(3, beam_width // 2), site_limit,
                    start_probes + 4 * widen, soft_horizon_s)
                successors.extend(states)
                for name, count in reasons.items():
                    failures[name] += count
            if successors:
                break
        if not successors:
            pieces = (_split_current_task(ctx, task, pending)
                      if repairs < max_repairs else None)
            if pieces is not None:
                pending[index:index + 1] = _task_order(ctx, pieces, "hard_slack")
                repairs += 1
                print(f"Q3 {grouping}: 拆分 {task.batch_id} -> "
                      f"{','.join(part.batch_id for part in pieces)}; "
                      f"当前已交付={len(beam[0].deliveries)}/80 箱",
                      flush=True)
                continue
            best = min(beam, key=_state_score)
            return best, {
                "status": "INCOMPLETE", "failed_batch_id": task.batch_id,
                "failed_box_ids": list(task.route.box_ids),
                "failed_zone_ids": list(task.route.zones),
                "failure_counts": dict(failures),
                "adaptive_splits": repairs,
                "scheduled_sorties": len(best.sorties),
                "scheduled_boxes": len(best.deliveries),
                "elapsed_s": round(perf_counter() - started, 2),
            }
        successors.sort(key=_state_score)
        # 同分数附近保留不同的中继悬停位置和机型，避免单一局部排程淹没搜索束。
        diverse = []
        signatures = set()
        for state in successors:
            recent = state.sorties[-1]
            signature = (recent["type_id"], recent["uav_id"],
                         tuple(sorted((m.candidate.candidate_id,
                                       m.uav_id, m.unit_id)
                                      for m in state.missions)))
            if signature in signatures:
                continue
            signatures.add(signature)
            diverse.append(state)
            if len(diverse) >= beam_width:
                break
        chosen = {id(state) for state in diverse}
        beam = diverse + [state for state in successors
                          if id(state) not in chosen][:max(0, beam_width - len(diverse))]
        index += 1
        if index == 1 or index % 5 == 0 or index == len(pending):
            best = beam[0]
            print(f"Q3 {grouping}: {index}/{len(pending)} 批, "
                  f"已交付={len(best.deliveries)}/80 箱, "
                  f"运输={len(best.sorties)} 架次, "
                  f"耗时={perf_counter() - started:.1f}s", flush=True)
    first_failure: dict[str, Any] | None = None
    for state in beam:
        transport_bad = [row for row in check_plan(
            ctx, {"sorties": state.sorties, "deliveries": state.deliveries})
            if row["status"] != "PASS"]
        bounds = {s["batch_id"]: (s["start_s"], s["return_s"])
                  for s in state.sorties}
        comm = verify_continuous_coverage(
            state.trajectories, links,
            communication_rows(state.slices, state.assigned, bounds),
            relay_rows_from_missions(state.missions))
        resources = validate_relay_rows(
            relay_rows_from_missions(state.missions), ctx.data_run, links, dem)
        if not transport_bad and comm["status"] == "PASS" and resources["status"] == "PASS":
            return state, {"status": "PASS", "scheduled_sorties": len(state.sorties),
                           "scheduled_boxes": len(state.deliveries),
                           "adaptive_splits": repairs,
                           "elapsed_s": round(perf_counter() - started, 2)}
        if first_failure is None:
            first_failure = {"transport_failure": transport_bad[0] if transport_bad else None,
                             "communication_status": comm["status"],
                             "communication_failure": comm["issues"][0] if comm["issues"] else None,
                             "resource_status": resources["status"],
                             "resource_failure": resources.get("first_failure")}
    return beam[0], {"status": "FAILED_VALIDATION",
                     "scheduled_sorties": len(beam[0].sorties),
                     "scheduled_boxes": len(beam[0].deliveries),
                     "adaptive_splits": repairs,
                     "elapsed_s": round(perf_counter() - started, 2),
                     **(first_failure or {})}


def _split_current_task(ctx: Context, task: RouteTask,
                        pending: list[RouteTask]) -> list[RouteTask] | None:
    """只拆尚未安排的当前批次，不清空已通过的排程。"""
    if len(task.route.box_ids) <= 1:
        return None
    visits = task.route.visits
    if len(visits) > 1:
        pieces = [[visit] for visit in visits]
    else:
        zone, ids = visits[0]
        middle = len(ids) // 2
        pieces = [[(zone, ids[:middle])], [(zone, ids[middle:])]]
    routes = [ctx.route([(zone, list(ids)) for zone, ids in piece])
              for piece in pieces]
    if any(not ctx.options(route) for route in routes):
        return None
    used = {item.batch_id for item in pending}
    pieces: list[RouteTask] = []
    number = 1
    for route in routes:
        while f"T{number:03d}" in used:
            number += 1
        batch_id = f"T{number:03d}"
        used.add(batch_id)
        pieces.append(RouteTask(batch_id, route))
    if sorted(bid for part in pieces for bid in part.route.box_ids) != sorted(task.route.box_ids):
        raise AssertionError("拆批导致货箱缺失或重复")
    return pieces


def run_scheme1(data_run: Path, output_root: Path, sample_step_s: float,
                max_groupings: int, beam_width: int, max_repairs: int,
                soft_horizon_s: float, spacing_m: int,
                candidate_limit: int, start_probes: int) -> dict[str, Any]:
    if min(max_groupings, beam_width, spacing_m, candidate_limit,
           start_probes) < 1 or max_repairs < 0:
        raise ValueError("组批数、束宽、候选网格、候选数和时间探测数必须为正；拆批数不能为负")
    relay, dem, links, source_meta = _scene(data_run)
    ctx = Context(data_run)
    positions = _node_positions(data_run)
    route_sets = _raw_route_sets(ctx)
    source_slices = _source_slices_from_raw(
        ctx, positions, links, sample_step_s, route_sets)
    candidates, candidate_stats = generate_candidates(
        dem, links, source_slices, relay, positions["O01"], spacing_m=spacing_m,
        max_points=12000)
    site_tree = (cKDTree(np.asarray([item.hover for item in candidates], dtype=float))
                 if candidates else None)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    table_dir = output_root / f"q3_opt1_{stamp}_table"
    figure_dir = output_root / f"q3_opt1_{stamp}_figure"
    # 组批方案各尝试一次；每次内部保留已排任务并对当前批逐级放宽候选、局部拆分。
    preferred = ("balanced", "compact", "neighbor_pair",
                 "small_batches", "hard_split", "single_box_fallback")
    route_sets.sort(key=lambda pair: preferred.index(pair[0])
                    if pair[0] in preferred else len(preferred))
    attempted: list[dict[str, Any]] = []
    selected: JointState | None = None
    selected_config: dict[str, Any] | None = None
    best_partial: tuple[int, JointState, dict[str, Any]] | None = None
    access_cache: dict[tuple[Any, ...], bool] = {}
    for name, tasks in route_sets[:max_groupings]:
        print(f"Q3 方案1: 开始组批 {name}, 初始 {len(tasks)} 架次候选", flush=True)
        state, outcome = _attempt_schedule(
            ctx, tasks, candidates, relay, links, dem, positions,
            sample_step_s, site_tree, beam_width, candidate_limit,
            start_probes, max_repairs, soft_horizon_s, access_cache, name)
        config = {"grouping": name, "task_order": "hard_slack",
                  "initial_transport_batches": len(tasks),
                  "candidate_limit": candidate_limit,
                  "start_probes": start_probes,
                  "beam_width": beam_width}
        attempted.append({**config, **outcome})
        print(f"Q3 方案1 {len(attempted)}/{min(max_groupings, len(route_sets))}: "
              f"{name}, 拆批={outcome.get('adaptive_splits', 0)}, "
              f"{outcome['status']}, "
              f"已交付={len(state.deliveries)}/80 箱, "
              f"运输={len(state.sorties)} 架次", flush=True)
        if best_partial is None or len(state.deliveries) > best_partial[0]:
            best_partial = len(state.deliveries), state, config
        if outcome["status"] == "PASS":
            selected, selected_config = state, {**config, **outcome}
            # 首次完整通过后停止；此处求可行方案，不宣称全局最优。
            break
    table_dir.mkdir(parents=True, exist_ok=False)
    if selected is None:
        summary = {
            "status": "SEARCH_INCOMPLETE",
            "interpretation": "当前搜索预算内未找到完整的 80 箱联合方案；未设固定运输架次数上限，也不证明题目无解",
            "source": source_meta, "candidate_stats": candidate_stats,
            "attempts": attempted, "global_optimality_proven": False,
            "best_partial_delivered_boxes": best_partial[0] if best_partial else 0,
            "best_partial_transport_sorties": len(best_partial[1].sorties) if best_partial else 0,
            "best_partial_grouping": best_partial[2] if best_partial else None,
            "table_dir": str(table_dir.resolve()), "figure_dir": None,
        }
        json_dump(table_dir / "Q3_运行摘要.json", summary)
        return summary
    _save_joint_tables(table_dir, selected, relay)
    json_dump(table_dir / "Q3_候选筛选统计.json", candidate_stats)
    validation = validate_official_tables(table_dir, data_run, links, dem)
    _save_verification(table_dir, validation)
    transport_energy = sum(s["energy_kwh"] for s in selected.sorties)
    relay_energy = sum(m.energy_kwh for m in selected.missions)
    summary = {
        "status": validation["status"],
        "method": "deadline-first raw-box batching with cached communication corridors and local repair",
        "interpretation": "从 80 箱独立形成架次；保留已排任务，仅对当前批扩大候选或拆批；有限候选不证明全局最优",
        "assumption": "同一中继架次可同时保障多架运输机，每架运输机每个时刻只采用一种通信方式",
        "global_optimality_proven": False, "source": source_meta,
        "selected_configuration": selected_config, "attempts": attempted,
        "candidate_stats": candidate_stats,
        "transport_sorties": len(selected.sorties),
        "relay_sorties": len(selected.missions),
        "delivered_boxes": len(selected.deliveries),
        "transport_last_return_s": max(s["return_s"] for s in selected.sorties),
        "relay_last_return_s": max((m.return_s for m in selected.missions), default=0.0),
        "joint_makespan_s": max([s["return_s"] for s in selected.sorties] +
                                [m.return_s for m in selected.missions]),
        "transport_energy_kwh": transport_energy,
        "relay_energy_kwh": relay_energy,
        "total_energy_kwh": transport_energy + relay_energy,
        "soft_weighted_lateness_s": sum(d["priority"] * d["soft_lateness_s"]
                                        for d in selected.deliveries
                                        if d["hard_deadline_s"] is None),
        "communication_status": validation["communication"]["status"],
        "transport_status": validation["transport_status"],
        "relay_resource_status": validation["relay_resources"]["status"],
        "table_dir": str(table_dir.resolve()),
        "figure_dir": str(figure_dir.resolve()) if validation["status"] == "PASS" else None,
    }
    json_dump(table_dir / "Q3_运行摘要.json", summary)
    if summary["status"] == "PASS":
        _save_figures(figure_dir, selected.slices, selected.assigned, selected.missions)
        (table_dir / "Q3_READY.txt").write_text(
            "Q3 direct scheme 1 independently verified PASS\n"
            f"source_manifest_sha256={source_meta['source_manifest_sha256']}\n",
            encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-run", type=Path, default=None,
                        help="0_outputs 中经过验收的原始数据目录")
    parser.add_argument("--output-root", type=Path,
                        default=CODE_DIR / "3_outputs" / "1_optimize")
    parser.add_argument("--sample-step-s", type=float, default=20.0)
    parser.add_argument("--candidate-spacing-m", type=int, default=500)
    parser.add_argument("--max-groupings", type=int, default=5,
                        help="最多尝试几种初始组批，不在同一组批上重复从头排程")
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--max-repairs", type=int, default=16,
                        help="每种组批允许对当前未排入架次拆批的次数")
    parser.add_argument("--candidate-limit", type=int, default=16,
                        help="先搜索每个盲区排名靠前的中继点；失败时自动扩大")
    parser.add_argument("--start-probes", type=int, default=8,
                        help="每个运输机/电池组合探测的起飞时刻数")
    parser.add_argument("--soft-horizon-s", type=float, default=86400.0)
    parser.add_argument("--validate-only", type=Path, default=None,
                        metavar="Q3_TABLE_DIR", help="重验已导出的第三问表格")
    args = parser.parse_args(argv)
    data_run = (args.data_run.resolve() if args.data_run is not None else
                find_data_run(CODE_DIR, EXPECTED_MANIFEST))
    if args.validate_only is not None:
        table_dir = args.validate_only.resolve()
        _, dem, links, _ = _scene(data_run)
        report = validate_official_tables(table_dir, data_run, links, dem)
        print(json.dumps({"status": report["status"],
                          "communication_status": report["communication"]["status"],
                          "relay_resource_status": report["relay_resources"]["status"],
                          "transport_status": report["transport_status"]},
                         ensure_ascii=False, indent=2))
        return 0 if report["status"] == "PASS" else 2
    summary = run_scheme1(
        data_run, args.output_root.resolve(), args.sample_step_s,
        args.max_groupings, args.beam_width, args.max_repairs,
        args.soft_horizon_s, args.candidate_spacing_m,
        args.candidate_limit, args.start_probes)
    print(json.dumps({key: summary.get(key) for key in (
        "status", "transport_sorties", "relay_sorties", "delivered_boxes",
        "joint_makespan_s", "total_energy_kwh", "table_dir", "figure_dir")},
        ensure_ascii=False, indent=2))
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
