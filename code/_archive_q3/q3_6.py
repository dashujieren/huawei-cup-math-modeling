"""问题三方案六：从方案三可行解继续做时间优先联合优化。

本文件是 PyCharm 的统一运行入口；求解与独立验收实现位于
q3_8_time_optimize.py，历史结果仍可由原脚本读取和复核。
"""

from __future__ import annotations

import sys
from pathlib import Path

from q3_8_time_optimize import main as optimize_main


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(value == "--output-root" or value.startswith("--output-root=")
               for value in arguments):
        arguments.extend([
            "--output-root",
            str(Path(__file__).resolve().parent / "3_outputs" / "6_time"),
        ])
    return optimize_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
