"""第三问全局候选列主问题的小型可复现测试。"""

from types import SimpleNamespace

import numpy as np

from q3_3_global_joint import (
    BlindBlock, ProfileChoice, RelayColumn, Rows, TransportColumn,
    add_capacity_rows, assign_entities, blind_blocks, choose_sites,
    preselect_sites,
    solve_master,
)


def transport(name, box, start, end, site=1):
    task = SimpleNamespace(route=SimpleNamespace(box_ids=(box,)))
    return TransportColumn(name, task, "A", {"energy_kwh": 1.0}, None,
                           start, end, end + 20,
                           (BlindBlock(start + 10, start + 20,
                                       frozenset((site,))),))


def relay(name, site, launch, ready, service_end, returned, charged):
    return RelayColumn(name, site, SimpleNamespace(), launch, ready,
                       service_end, returned, returned + 30, charged, 1.0)


def test_two_transports_share_one_relay_mission():
    cols = [transport("A", "box-a", 100, 200),
            transport("B", "box-b", 100, 200)]
    missions = [relay("R", 1, 0, 50, 180, 220, 300)]
    answer = solve_master({"box-a", "box-b"}, cols, missions,
                          {"A": 2}, {"A": 2}, 1, 1, 5.0)
    assert answer.status == "CANDIDATE_FEASIBLE"
    assert answer.transport_indices == [0, 1]
    assert answer.relay_indices == [0]


def test_transport_resource_capacity_requires_later_option():
    cols = [transport("A", "box-a", 100, 200),
            transport("B1", "box-b", 100, 200),
            transport("B2", "box-b", 220, 320)]
    missions = [relay("R", 1, 0, 50, 300, 350, 450)]
    answer = solve_master({"box-a", "box-b"}, cols, missions,
                          {"A": 1}, {"A": 1}, 1, 1, 5.0)
    assert answer.status == "CANDIDATE_FEASIBLE"
    assert answer.transport_indices == [0, 2]


def test_missing_relay_window_is_candidate_failure_not_infeasibility():
    answer = solve_master(
        {"box-a"}, [transport("A", "box-a", 100, 200)],
        [relay("R", 2, 0, 50, 180, 220, 300)],
        {"A": 1}, {"A": 1}, 1, 1, 5.0)
    assert answer.status == "CANDIDATE_INCOMPLETE"
    assert answer.diagnostics["boxes_without_covered_columns"] == ["box-a"]


def test_all_direct_route_does_not_require_relay_column():
    base = transport("A", "box-a", 0, 100)
    direct = TransportColumn(base.column_id, base.task, base.kind,
                             base.option, base.profile, base.start_s,
                             base.return_s, base.charge_end_s, ())
    answer = solve_master({"box-a"}, [direct], [],
                          {"A": 1}, {"A": 1}, 1, 1, 5.0)
    assert answer.status == "CANDIDATE_FEASIBLE"
    assert answer.relay_indices == []


def test_half_open_resource_intervals_and_charging_overlap():
    rows = Rows(3)
    count = add_capacity_rows(rows, [(0, 0, 100), (1, 100, 200),
                                     (2, 50, 150)], 1)
    assert count >= 2
    constraint = rows.constraint()
    assert np.all(constraint.A @ np.array([1, 1, 0]) <= constraint.ub)
    assert np.any(constraint.A @ np.array([1, 0, 1]) > constraint.ub)


def test_resource_interval_coloring_matches_capacity():
    entities = assign_entities(
        [(0, "R", 0, 100), (1, "R", 50, 150), (2, "R", 100, 200)],
        {"R": ["R01", "R02"]})
    assert entities[0] != entities[1]
    assert entities[0] == entities[2]


def test_blind_blocks_split_long_continuous_gap():
    slices = [SimpleNamespace(slice_id=i, direct_available=False,
                              start_s=i * 100.0, end_s=(i + 1) * 100.0)
              for i in range(5)]
    profile = SimpleNamespace(slices=slices,
                              support={i: {1, 2} for i in range(5)})
    blocks = blind_blocks(profile, {1, 2}, max_block_s=300)
    assert [(block.start_s, block.end_s) for block in blocks] == [
        (0.0, 300.0), (300.0, 500.0)]


def test_spatial_preselection_keeps_shared_coverage(monkeypatch):
    import q3_3_global_joint as module

    candidates = [SimpleNamespace(candidate_id=name, hover=(i, 0, 0),
                                  travel_energy_kwh=1.0,
                                  max_service_s=1000.0)
                  for i, name in enumerate(("left", "shared", "right"))]
    slices = [SimpleNamespace(direct_available=False, midpoint=(i, 0, 0),
                              accepted={"shared", side})
              for i, side in ((0, "left"), (2, "right"))]
    monkeypatch.setattr(module.q3, "_access_certified",
                        lambda site, item, _links, _cache:
                        site.candidate_id in item.accepted)
    sites, stats = preselect_sites(slices, candidates, None, {}, 1, 3)
    assert [site.candidate_id for site in sites] == ["shared"]
    assert stats["source_slices_uncovered_after_budget"] == 0


def test_site_selection_keeps_alternatives_after_minimum_cover():
    sites = [SimpleNamespace(x_m=i * 500.0, y_m=0.0,
                             travel_energy_kwh=1.0,
                             max_service_s=1000.0,
                             candidate_id=str(i)) for i in range(3)]
    profile = SimpleNamespace(support={0: {0, 1}, 1: {0, 2}})
    choices = [ProfileChoice(None, "A", {}, profile)]
    selected, stats = choose_sites(choices, sites, 3)
    assert selected == {0, 1, 2}
    assert stats["uncovered_profile_slices_after_pruning"] == 0
