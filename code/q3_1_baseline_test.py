"""Small, deterministic checks for the question-three physical baseline."""

import math
from types import SimpleNamespace

import pytest

from q3_0_baseline import (
    LinkResult,
    Radio,
    RelayCandidate,
    TimeSlice,
    TrajectorySegment,
    _resource_choice,
    bidirectional_limit_db,
    communication_rows,
    generate_candidates,
    path_loss_db,
    schedule_relays,
)


def test_communication_final_time_uses_official_q2_return_precision() -> None:
    segment = TrajectorySegment("下降", 10.0, 20.000000002, 0, 0, 50, 0, 0, 0)
    item = TimeSlice(0, "Q001", "下降", 10.0, 20.000000002,
                     segment.position(15.0), 5.0, True, segment)
    rows = communication_rows([item], {}, {"Q001": (0.0, 20.0)})
    assert float(rows[-1]["结束时刻（s）"]) == 20.0


def test_bidirectional_budget_uses_weaker_direction() -> None:
    ground = Radio(
        tx_power_dbm=20.0,
        gain_dbi=3.0,
        sensitivity_dbm=-90.0,
        fade_margin_db=6.0,
    )
    aircraft = Radio(
        tx_power_dbm=10.0,
        gain_dbi=5.0,
        sensitivity_dbm=-80.0,
        fade_margin_db=4.0,
    )

    # Ground -> aircraft: 102 dB; aircraft -> ground: 100 dB.
    assert bidirectional_limit_db(ground, aircraft, system_loss_db=2.0) == pytest.approx(100.0)
    assert bidirectional_limit_db(aircraft, ground, system_loss_db=2.0) == pytest.approx(100.0)


def test_obstruction_adds_loss_instead_of_forcing_disconnection() -> None:
    clear = path_loss_db(
        distance_m=1000.0,
        frequency_mhz=1000.0,
        obstructed=False,
        obstruction_loss_db=10.0,
    )
    blocked = path_loss_db(
        distance_m=1000.0,
        frequency_mhz=1000.0,
        obstructed=True,
        obstruction_loss_db=10.0,
    )

    assert clear == pytest.approx(92.45)
    assert blocked == pytest.approx(clear + 10.0)
    assert math.isfinite(blocked)


@pytest.mark.parametrize(
    ("distance_m", "frequency_mhz"),
    [
        (0.0, 1000.0),
        (-1.0, 1000.0),
        (math.nan, 1000.0),
        (math.inf, 1000.0),
        (1000.0, 0.0),
        (1000.0, math.nan),
        (1000.0, math.inf),
    ],
)
def test_path_loss_rejects_nonpositive_or_nonfinite_inputs(
    distance_m: float, frequency_mhz: float
) -> None:
    with pytest.raises(ValueError):
        path_loss_db(
            distance_m=distance_m,
            frequency_mhz=frequency_mhz,
            obstructed=False,
            obstruction_loss_db=10.0,
        )


def test_trajectory_position_interpolates_both_endpoints_and_midpoint() -> None:
    segment = TrajectorySegment(
        stage="巡航",
        start_s=5.0,
        end_s=15.0,
        x0=0.0,
        y0=2.0,
        z0=3.0,
        x1=10.0,
        y1=12.0,
        z1=13.0,
    )

    assert segment.position(5.0) == pytest.approx((0.0, 2.0, 3.0))
    assert segment.position(10.0) == pytest.approx((5.0, 7.0, 8.0))
    assert segment.position(15.0) == pytest.approx((10.0, 12.0, 13.0))


@pytest.mark.parametrize("end_s", [5.0, 4.0, math.nan, math.inf])
def test_trajectory_rejects_zero_or_invalid_duration(end_s: float) -> None:
    with pytest.raises(ValueError):
        TrajectorySegment(
            stage="爬升",
            start_s=5.0,
            end_s=end_s,
            x0=0.0,
            y0=0.0,
            z0=0.0,
            x1=0.0,
            y1=0.0,
            z1=20.0,
        )


def test_trajectory_rejects_nonfinite_coordinate_or_query_time() -> None:
    with pytest.raises(ValueError):
        TrajectorySegment(
            stage="下降",
            start_s=0.0,
            end_s=10.0,
            x0=0.0,
            y0=0.0,
            z0=math.nan,
            x1=0.0,
            y1=0.0,
            z1=0.0,
        )

    segment = TrajectorySegment(
        stage="投送",
        start_s=0.0,
        end_s=10.0,
        x0=1.0,
        y0=2.0,
        z0=3.0,
        x1=1.0,
        y1=2.0,
        z1=3.0,
    )
    with pytest.raises(ValueError):
        segment.position(math.nan)


def _blind_slice(slice_id: int, start_s: float, end_s: float,
                 x_m: float = 0.0) -> TimeSlice:
    segment = TrajectorySegment(
        stage="巡航", start_s=start_s, end_s=end_s,
        x0=x_m, y0=0.0, z0=50.0, x1=x_m, y1=0.0, z1=50.0,
    )
    return TimeSlice(
        slice_id=slice_id, sortie_id=f"T{slice_id:02d}", stage="巡航",
        start_s=start_s, end_s=end_s, midpoint=(x_m, 0.0, 50.0),
        direct_margin_db=-1.0, direct_available=False, segment=segment,
    )


def _candidate(candidate_id: str, x_m: float = 0.0,
               max_service_s: float = 30.0) -> RelayCandidate:
    return RelayCandidate(
        candidate_id=candidate_id, x_m=x_m, y_m=0.0, alt_m=50.0,
        lon_deg=105.0, lat_deg=30.0, agl_m=50.0,
        outward_s=20.0, homeward_s=10.0, travel_energy_kwh=0.1,
        max_service_s=max_service_s, backhaul_margin_db=10.0,
    )


def _relay_resources(uav_ids: tuple[str, ...] = ("R01", "R02"),
                     unit_ids: tuple[str, ...] = ("E01", "E02")) -> dict:
    return {
        "uav_ids": list(uav_ids), "unit_ids": list(unit_ids),
        "params": {
            "prep_s": 10.0, "link_setup_s": 5.0,
            "hover_power_kw": 1.0, "comm_extra_power_kw": 0.2,
            "energy_use_kwh": 1.0, "return_soc_min": 0.2,
            "turnaround_s": 30.0, "full_charge_s": 100.0,
            "max_hover_agl_m": 300.0, "cruise_power_kw": 0.5,
            "cruise_speed_mps": 10.0, "climb_speed_mps": 5.0,
            "descent_speed_mps": 5.0, "takeoff_mass_kg": 10.0,
            "climb_efficiency": 0.8,
        },
    }


class _LinksStub:
    def __init__(self, coverage: dict[float, set[float]]):
        self.coverage = coverage
        generous = Radio(100.0, 0.0, -100.0, 0.0)
        self.radios = {
            "transport": generous, "relay_access": generous,
            "relay_backhaul": generous, "gateway": generous,
        }
        self.frequency_mhz = 2400.0
        self.system_loss_db = 0.0
        self.obstruction_loss_db = 10.0
        self.gateway = (0.0, 0.0, 20.0)

    def access(self, transport: tuple[float, float, float],
               hover: tuple[float, float, float]) -> LinkResult:
        available = transport[0] in self.coverage.get(hover[0], set())
        return LinkResult(available, 10.0 if available else -10.0,
                          False, math.dist(transport, hover))

    def backhaul(self, hover: tuple[float, float, float]) -> LinkResult:
        return LinkResult(True, 10.0, False, math.dist(hover, self.gateway))


@pytest.mark.parametrize("first_s,covered", [(34.0, False), (35.0, True)])
def test_scheduler_requires_prep_flight_and_link_before_blind_start(
    first_s: float, covered: bool
) -> None:
    # 10 s preparation + 20 s outward flight + 5 s link setup = 35 s lead.
    blind = _blind_slice(0, first_s, first_s + 10.0)
    missions, assigned, uncovered, _ = schedule_relays(
        [blind], [_candidate("H0")], _LinksStub({0.0: {0.0}}),
        _relay_resources(),
    )
    assert bool(missions) is covered
    assert bool(assigned) is covered
    assert uncovered == ([] if covered else [blind.slice_id])
    if covered:
        assert missions[0].start_s == pytest.approx(0.0)
        assert missions[0].ready_s == pytest.approx(first_s)


def test_resource_choice_needs_both_uav_and_energy_unit_ready() -> None:
    candidate = _candidate("H0")
    relay = _relay_resources()
    choice = _resource_choice(
        candidate, first_s=100.0, relay=relay,
        uav_ready={"R01": 60.0, "R02": 90.0},
        unit_ready={"E01": 80.0, "E02": 62.0},
    )
    assert choice == ("R01", "E02", 65.0, 100.0)
    assert _resource_choice(
        candidate, first_s=100.0, relay=relay,
        uav_ready={"R01": 66.0, "R02": 90.0},
        unit_ready={"E01": 80.0, "E02": 66.0},
    ) is None


def test_scheduler_uses_two_uavs_and_two_units_for_overlapping_blind_windows() -> None:
    slices = [_blind_slice(0, 40.0, 50.0, 0.0),
              _blind_slice(1, 45.0, 55.0, 10.0)]
    candidates = [_candidate("H0", 0.0, max_service_s=10.0),
                  _candidate("H10", 10.0, max_service_s=10.0)]
    links = _LinksStub({0.0: {0.0}, 10.0: {10.0}})
    missions, assigned, uncovered, _ = schedule_relays(
        slices, candidates, links, _relay_resources(),
    )
    assert uncovered == []
    assert assigned == {0: "R001", 1: "R002"}
    assert [(m.uav_id, m.unit_id) for m in missions] == [
        ("R01", "E01"), ("R02", "E02"),
    ]

    # Missing either resource class must prevent the second simultaneous mission.
    for relay in (_relay_resources(uav_ids=("R01",)),
                  _relay_resources(unit_ids=("E01",))):
        fewer, fewer_assigned, fewer_uncovered, _ = schedule_relays(
            slices, candidates, links, relay,
        )
        assert len(fewer) == 1
        assert fewer_assigned == {0: "R001"}
        assert fewer_uncovered == [1]


def test_greedy_shares_one_relay_and_counts_link_setup_energy() -> None:
    slices = [_blind_slice(0, 40.0, 50.0, 0.0),
              _blind_slice(1, 55.0, 65.0, 10.0)]
    candidates = [_candidate("A_shared", 0.0),
                  _candidate("B_first_only", 100.0)]
    links = _LinksStub({0.0: {0.0, 10.0}, 100.0: {0.0}})
    missions, assigned, uncovered, _ = schedule_relays(
        slices, candidates, links, _relay_resources(),
    )
    assert uncovered == []
    assert assigned == {0: "R001", 1: "R001"}
    assert len(missions) == 1
    mission = missions[0]
    assert mission.candidate.candidate_id == "A_shared"
    assert mission.service_end_s == pytest.approx(65.0)
    assert mission.return_s == pytest.approx(75.0)
    # Travel + (5 s setup + 25 s service) at 1.2 kW; ground prep is not flight energy.
    assert mission.energy_kwh == pytest.approx(0.1 + 1.2 * 30.0 / 3600.0)


def test_candidate_screen_enforces_agl_and_pdf_cruise_height(monkeypatch) -> None:
    import q3_0_baseline as q3

    monkeypatch.setattr(q3, "_candidate_xy", lambda *_args, **_kwargs: [(0.0, 0.0)])

    class _DemStub:
        to_lonlat = SimpleNamespace(transform=lambda _x, _y: (105.0, 30.0))

        @staticmethod
        def terrain_at(_x: float, _y: float) -> float:
            return 0.0

        @staticmethod
        def arc_geometry(_a, _b) -> dict[str, float]:
            return {"distance_m": 100.0, "cruise_alt_m": 150.0,
                    "climb_m": 20.0, "descent_m": 0.0}

    candidates, counts = generate_candidates(
        _DemStub(), _LinksStub({}), [_blind_slice(0, 40.0, 50.0)],
        _relay_resources(), heights_agl_m=(50, 200, 350),
    )
    assert [candidate.agl_m for candidate in candidates] == [50.0]
    assert counts["pdf_cruise_screen"] == 1
    assert counts["height_limit"] == 1
    assert counts["candidate_count"] == 1
