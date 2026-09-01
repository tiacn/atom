# -*- coding: utf-8 -*-
"""运行 Step 5.5A 的正式失败诊断。"""

import subprocess
import sys
from pathlib import Path


def main():
    this_dir = Path(__file__).resolve().parent
    scripts = (
        "01_literal_steady_state.py",
        "02_velocity_convergence.py",
        "03_linear_thermal_vs_elecsus.py",
        "04_residual_diagnosis.py",
    )
    for script in scripts:
        print()
        print("=" * 78)
        print(f"运行 {script}")
        print("=" * 78)
        subprocess.run(
            [sys.executable, str(this_dir / script)],
            cwd=str(this_dir),
            check=True,
        )

    print()
    print("所有诊断脚本执行成功")
    print("STEP 5.5A OVERALL: FAIL（full steady-state 暗态与 ElecSus 热平衡假设不同）")


if __name__ == "__main__":
    main()
