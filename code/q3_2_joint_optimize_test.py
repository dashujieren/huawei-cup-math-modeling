"""第三问回退排程与中继窗口延长的有限单元测试。"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import q3_1_optimize as q3
import q3_2_joint_optimize as joint


def test_promoted_order_preserves_every_task():
    tasks = [q3.RouteTask(name, SimpleNamespace(box_ids=(name,)))
             for name in ("A", "B", "C", "D")]
    moved = joint.promoted_order(tasks, 3, 1)
    assert [item.batch_id for item in moved] == ["A", "D", "B", "C"]
    assert [item.batch_id for item in tasks] == ["A", "B", "C", "D"]
    with pytest.raises(ValueError):
        joint.promoted_order(tasks, 1, 1)


def test_existing_relay_window_can_be_extended_and_repriced():
    site = q3.RelayCandidate("H1", 0, 0, 100, 109, 23, 100,
                             10, 20, 0.2, 500, 10)
    old = q3.RelayMission("R001", "R01", "E01", site,
                          0, 100, 200, 220,
                          0.2 + 1.1 * (30 + 100) / 3600)
    segment = q3.TrajectorySegment("cruise", 190, 250,
                                   1, 1, 100, 2, 2, 100)
    blind = q3.TimeSlice(17, "T002", "cruise", 190, 250,
                         (1.5, 1.5, 100), -1, False, segment)
    state = q3.JointState([], [], {}, [], {}, [old])
    profile = q3.RouteProfile([], [blind], {}, {}, [])
    relay = {"params": {"hover_power_kw": 1.05,
                         "comm_extra_power_kw": 0.05,
                         "link_setup_s": 30,
                         "return_soc_min": 0.2,
                         "energy_use_kwh": 3.2,
                         "turnaround_s": 300,
                         "full_charge_s": 1800},
             "uav_ids": ["R01", "R02"], "unit_ids": ["E01"]}
    with patch.object(q3, "_access_certified", return_value=True):
        options, _, _ = q3._relay_options_fast(
            state, profile, [blind], [site], None, relay, 1, {})
    assert len(options) == 1
    missions, assignments = options[0]
    assert assignments == {17: "R001"}
    assert len(missions) == 1
    assert missions[0].service_end_s == 250
    assert missions[0].return_s == 270
    assert missions[0].energy_kwh == pytest.approx(
        0.2 + 1.1 * (30 + 150) / 3600)
    # 旧班次结束后可以衔接的机体任务，延长窗口后会发生周转冲突。
    following = q3.RelayMission("R002", "R01", "E02", site,
                                540, 760, 900, 920, 0.5)
    busy_state = q3.JointState([], [], {}, [], {}, [old, following])
    with patch.object(q3, "_access_certified", return_value=True):
        rejected, _, reasons = q3._relay_options_fast(
            busy_state, profile, [blind], [site], None, relay, 1, {})
    assert rejected == []
    assert reasons["relay_resource_calendar"] >= 1


def test_rewind_moves_blocked_hard_task_before_conflicting_task():
    tasks = [q3.RouteTask(name, SimpleNamespace(box_ids=(name,), zones=(name,)))
             for name in ("A", "B")]
    ctx = SimpleNamespace(data_run=Path("unused"))

    def insert(_ctx, state, task, *_args):
        if task.batch_id == "B" and any(
                sortie["batch_id"] == "A" for sortie in state.sorties):
            return [], {"relay_resource_calendar": 1}
        sortie = {"batch_id": task.batch_id, "type_id": "A",
                  "uav_id": "U01", "battery_id": "B01",
                  "return_s": 1, "energy_kwh": 1}
        delivery = {"priority": 1, "soft_lateness_s": 0,
                    "hard_deadline_s": None}
        return [q3.JointState([*state.sorties, sortie],
                              [*state.deliveries, delivery],
                              {}, [], {}, [])], {}

    with (patch.object(q3, "_task_order", side_effect=lambda _ctx, items, _mode: items),
          patch.object(q3, "_insert_task", side_effect=insert),
          patch.object(joint, "validate_state", return_value={"status": "PASS"})):
        state, result = joint.schedule_with_rewind(
            ctx, tasks, [], {}, None, None, {}, 20, None, {},
            grouping="unit", beam_width=1, candidate_limit=1,
            start_probes=1, max_rewinds=1, max_splits=0,
            soft_horizon_s=1000, max_seconds=0)
    assert result["status"] == "PASS"
    assert result["rewinds"] == 1
    assert [sortie["batch_id"] for sortie in state.sorties] == ["B", "A"]
