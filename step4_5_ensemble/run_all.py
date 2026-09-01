# -*- coding: utf-8 -*-
"""逐项运行 Step 4.5；默认包含完整 MC convergence。"""

import subprocess
import sys
from pathlib import Path


def main():
    this_dir = Path(__file__).resolve().parent
    scripts = [
        "01_ensemble_binning.py",
        "02_zero_light.py",
        "03_uniform_field.py",
        "04_stress.py",
        "05_history_memory.py",
        "06_slice_convergence.py",
        "07_mc_convergence.py",
    ]

    for script in scripts:
        print()
        print("=" * 72)
        print(f"运行 {script}")
        print("=" * 72)
        subprocess.run(
            [sys.executable, str(this_dir / script)],
            cwd=str(this_dir),
            check=True,
        )

    print()
    print("全部 PASS：Step 4.5 N 原子系综集成验证完成")


if __name__ == "__main__":
    main()

