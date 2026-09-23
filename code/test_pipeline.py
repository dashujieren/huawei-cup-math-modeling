"""在临时目录验收数据管线，不触碰原始附件与正式运行结果。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import _line_cells, extract, geo, normalize, require_integer, validate, verify_ready


PROJECT = Path(__file__).resolve().parent.parent


def test_supercover_is_reversible_and_covers_corner_neighbours() -> None:
    forward = _line_cells(0.5, 0.5, 2.5, 2.5, 4, 4)
    backward = _line_cells(2.5, 2.5, 0.5, 0.5, 4, 4)
    assert forward == backward
    assert {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)} <= forward


def test_fractional_stock_is_rejected() -> None:
    with pytest.raises(ValueError, match="整数"):
        require_integer(pd.Series([6.5]), "stock_qty", 1)


def test_full_pipeline_and_failed_qa_revokes_ready(tmp_path: Path) -> None:
    run_dir = tmp_path / "test_run"
    extract(PROJECT, run_dir)
    normalize(run_dir)
    geo(PROJECT, run_dir)

    boxes_path = run_dir / "clean" / "boxes.csv"
    boxes = pd.read_csv(boxes_path)
    assert len(boxes) == 80
    assert boxes.loc[~boxes["is_first_batch"], "first_deadline_s"].isna().all()
    medical_nonfirst = boxes.loc[boxes["medical_bool"] & ~boxes["is_first_batch"]]
    assert len(medical_nonfirst) == 1
    assert np.isclose(medical_nonfirst.iloc[0]["hard_deadline_s"],
                      medical_nonfirst.iloc[0]["expected_s"])
    both = boxes.loc[boxes["medical_bool"] & boxes["is_first_batch"]]
    assert np.allclose(both["hard_deadline_s"],
                       both[["first_deadline_s", "expected_s"]].min(axis=1))

    assert validate(PROJECT, run_dir)
    ready = run_dir / "meta" / "READY.txt"
    assert ready.is_file()
    assert verify_ready(run_dir)

    arc_path = run_dir / "derived" / "arc_geometry.csv"
    arcs = pd.read_csv(arc_path)
    original_arcs = arcs.copy()
    arcs["max_dsm_m"] -= 100
    arcs["cruise_alt_m"] -= 100
    arcs.to_csv(arc_path, index=False)
    assert not verify_ready(run_dir)
    assert not validate(PROJECT, run_dir)
    assert not ready.exists()
    arcs_qa = pd.read_csv(run_dir / "meta" / "qa_report.csv")
    assert arcs_qa.loc[arcs_qa["check_id"].eq("arcs_recomputed_from_sources"), "status"].iloc[0] == "FAIL"

    original_arcs.to_csv(arc_path, index=False)
    assert validate(PROJECT, run_dir)
    assert ready.is_file()
    assert verify_ready(run_dir)

    boxes.loc[0, "mass_kg"] += 1.0
    boxes.to_csv(boxes_path, index=False)
    assert not validate(PROJECT, run_dir)
    assert not ready.exists()
    qa = pd.read_csv(run_dir / "meta" / "qa_report.csv")
    assert (qa["status"] == "FAIL").any()

    boxes.drop(columns=["mass_kg"]).to_csv(boxes_path, index=False)
    with pytest.raises(KeyError):
        validate(PROJECT, run_dir)
    assert not ready.exists()
    exception_qa = pd.read_csv(run_dir / "meta" / "qa_report.csv")
    assert exception_qa.iloc[0]["check_id"] == "exception:validate"
