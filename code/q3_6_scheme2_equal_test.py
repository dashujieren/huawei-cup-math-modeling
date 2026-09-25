from __future__ import annotations

import math
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import q3_1_optimize as base
import q3_6_scheme2_equal_optimize as target
from q2_1_optimize import Context


class Scheme2EqualTests(unittest.TestCase):
    def test_more_than_31_transport_sorties_can_win(self) -> None:
        old = {"weighted_lateness": 632182.8956649068,
               "makespan_s": 15159.195158195356,
               "energy_kwh": 91.96517872557403,
               "transport_sorties": 31, "relay_sorties": 12}
        reference = target._reference(old)
        self.assertAlmostEqual(target._score(old, reference), 1.0)
        better = dict(old, weighted_lateness=400000.0, transport_sorties=32)
        self.assertTrue(target._is_improvement(better, 1.0, reference))
        worse = dict(old, transport_sorties=32)
        self.assertFalse(target._is_improvement(worse, 1.0, reference))

    def test_split_options_keep_both_children(self) -> None:
        code = Path(__file__).resolve().parent
        ctx = Context(target._default_data_run())
        table = code / "3_outputs" / "2_optimize" / "q3_opt2_260925_143402_492744_table"
        rows = base.read_rows(table / "Q3_运输明细.csv")
        for row in rows:
            ids = row["box_ids"].split(";")
            if len(ids) < 2:
                continue
            zones = row["visit_order"].split(">")
            route = ctx.route([(zone, [bid for bid in ids
                                       if str(ctx.boxes[bid]["zone_id"]) == zone])
                               for zone in zones])
            option = ctx.evaluate(route, row["type_id"])
            choice = SimpleNamespace(task=SimpleNamespace(route=route),
                                     kind=row["type_id"], option=option)
            parts = target._partition_routes(ctx, choice)
            if not parts:
                continue
            for first, second in parts:
                self.assertFalse(set(first.box_ids) & set(second.box_ids))
                self.assertEqual(set(route.box_ids),
                                 set(first.box_ids) | set(second.box_ids))
            starts = {(route.visits, row["type_id"]): round(float(row["start_s"]))}
            reference = target._reference({
                "weighted_lateness": 632182.8956649068,
                "makespan_s": 15159.195158195356,
                "energy_kwh": 91.96517872557403,
                "transport_sorties": 31, "relay_sorties": 12})
            options = target._split_options(ctx, [choice], starts, reference, 10,
                                            time.monotonic() + 10)
            if options:
                self.assertEqual(2, len(options))
                self.assertEqual((0, 0), options[0][0])
                self.assertEqual(set(route.box_ids),
                                 set(options[0][1].box_ids) |
                                 set(options[1][1].box_ids))
                return
        self.fail("方案二原表中未找到可拆分的可飞路线")


if __name__ == "__main__":
    unittest.main()
