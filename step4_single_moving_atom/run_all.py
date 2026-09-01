# -*- coding: utf-8 -*-
"""按顺序运行第四步的全部单原子验证。"""

import subprocess
import sys
from pathlib import Path


def main():
    this_dir = Path(__file__).resolve().parent
    scripts = [
        "01_hand_trajectory.py",
        "02_zero_light.py",
        "03_uniform_field.py",
        "04_doppler.py",
        "05_random_atom.py",
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
    print("全部 PASS：第四步单个运动原子验证完成")


if __name__ == "__main__":
    main()

