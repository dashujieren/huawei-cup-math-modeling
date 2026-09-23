from datetime import datetime
from pathlib import Path

from _0_pipeline import extract, geo, normalize, record_failure, validate, verify_ready


def main() -> None:
    project = Path(__file__).resolve().parent.parent
    run_dir = Path(__file__).resolve().parent / "0_outputs" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if run_dir.exists():
        raise FileExistsError(run_dir)
    print(f"运行目录：{run_dir}")
    try:
        extract(project, run_dir)
        normalize(run_dir)
        geo(project, run_dir)
        ok = validate(project, run_dir)
        if ok and not verify_ready(run_dir):
            raise RuntimeError("验收文件哈希不一致，禁止发布")
    except Exception as exc:
        record_failure(run_dir, "run_all", exc)
        raise
    print(f"数据验收：{'通过' if ok else '失败'}")
    print(f"检查报告：{run_dir / 'meta' / 'qa_report.csv'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
