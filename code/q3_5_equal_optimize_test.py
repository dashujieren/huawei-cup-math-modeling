"""第三问等权优化的轻量回归测试；不会启动正式求解。"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import q3_1_optimize as q3


class EqualWeightObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = dict(zip(q3.OBJECTIVE_KEYS, (100.0, 200.0, 10.0, 40.0)))

    def test_baseline_scores_one_and_each_component_has_equal_weight(self) -> None:
        self.assertAlmostEqual(q3.equal_weight_score(self.base, self.base), 1.0)
        for key in q3.OBJECTIVE_KEYS:
            candidate = dict(self.base)
            candidate[key] *= 0.5
            self.assertAlmostEqual(q3.equal_weight_score(candidate, self.base), 0.875)

    def test_zero_baseline_component_is_finite_and_baseline_stays_one(self) -> None:
        base = dict(self.base, soft_weighted_lateness_s=0.0)
        self.assertAlmostEqual(q3.equal_weight_score(base, base), 1.0)
        candidate = dict(base, soft_weighted_lateness_s=1.0)
        self.assertGreater(q3.equal_weight_score(candidate, base), 1.0)

    def test_integer_coefficients_use_watt_hours_for_energy(self) -> None:
        coef = q3._equal_weight_coefficients(self.base)
        self.assertEqual(coef["soft_weighted_lateness_s"], 10_000_000)
        self.assertEqual(coef["total_energy_kwh"], 100_000)
        self.assertEqual(coef["total_sorties"], 25_000_000)

    def test_all_batch_routes_are_in_hints_and_relay_slots_are_compact(self) -> None:
        choice_a = SimpleNamespace(task=SimpleNamespace(route=SimpleNamespace(
            visits=(("S001", ("A",)),))), kind="A")
        choice_b = SimpleNamespace(task=SimpleNamespace(route=SimpleNamespace(
            visits=(("S002", ("B",)),))), kind="B")
        solution = {"pool": [choice_a, choice_b],
                    "routes": [(0, 10, "U01", "B01"),
                               (1, 20, "U02", "B02")],
                    "relays": [(9, 1, "R1", "E1", 30, 40),
                               (2, 0, "R2", "E2", 10, 20)]}
        hints = q3._solution_hints(solution)
        self.assertEqual(len(hints["routes"]), 2)
        self.assertEqual([row[0] for row in hints["relays"]], [0, 1])
        self.assertEqual([row[4] for row in hints["relays"]], [10, 30])

    def test_no_feasible_candidate_preserves_baseline(self) -> None:
        state = q3.JointState([], [], {}, [], {}, [])
        solution = {"pool": [], "routes": [], "relays": []}
        with patch.object(q3, "_solve_joint", return_value=(None, {"solver_status": "UNKNOWN"})):
            result, report = q3._search_equal_improvement(
                None, {}, [], [], solution, state, Path("."), None, None,
                Path("."), 100, 2, 2.0, 1)
        self.assertIs(result, state)
        self.assertEqual(report["status"], "NO_IMPROVEMENT")
        self.assertEqual(len(report["stages"]), 2)

    def test_candidate_must_pass_independent_validation_before_acceptance(self) -> None:
        baseline = q3.JointState(
            [{"return_s": 100.0, "energy_kwh": 5.0},
             {"return_s": 90.0, "energy_kwh": 5.0}], [], {}, [], {}, [])
        candidate_state = q3.JointState(
            [{"return_s": 80.0, "energy_kwh": 4.0}], [], {}, [], {}, [])
        solution = {"pool": [], "routes": [], "relays": []}
        for status, expected in (("PASS", "IMPROVED"), ("FAIL", "NO_IMPROVEMENT")):
            with self.subTest(validation=status), \
                 patch.object(q3, "_solve_joint", side_effect=[
                     (solution, {"solver_status": "FEASIBLE"}),
                     (None, {"solver_status": "UNKNOWN"})]), \
                 patch.object(q3, "_materialize_solution", return_value=candidate_state), \
                 patch.object(q3, "_save_joint_tables"), \
                 patch.object(q3, "validate_official_tables",
                              return_value={"status": status}), \
                 patch.object(q3, "_save_verification"):
                result, report = q3._search_equal_improvement(
                    None, {}, [], [], solution, baseline, Path("."), None, None,
                    Path("."), 100, 2, 2.0, 1)
            self.assertEqual(report["status"], expected)
            self.assertIs(result, candidate_state if status == "PASS" else baseline)

    def test_official_input_contains_31_hard_deadline_boxes(self) -> None:
        path = q3.CODE_DIR / "0_outputs" / "run_20260924_154122" / "clean" / "boxes.csv"
        if not path.exists():
            self.skipTest("本地尚无正式清洗数据")
        boxes = pd.read_csv(path)
        self.assertEqual(len(boxes), 80)
        self.assertEqual(int(boxes["hard_deadline_s"].notna().sum()), 31)
        medical = boxes.loc[boxes["box_id"] == "S001-MED-02"]
        self.assertEqual(float(medical.iloc[0]["hard_deadline_s"]), 3600.0)


if __name__ == "__main__":
    unittest.main()
