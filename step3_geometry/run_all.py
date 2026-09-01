# -*- coding: utf-8 -*-
"""按顺序运行第三步的全部几何验证。"""

import subprocess
import sys
from pathlib import Path


def main():
    this_dir = Path(__file__).resolve().parent
    scripts = [
        "01_positions.py",
        "02_velocities.py",
        "03_entry_points.py",
        "04_slice_history.py",
        "05_full_ensemble.py",
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
    print("全部 PASS：第三步几何与运动历史验证完成")


if __name__ == "__main__":
    main()

