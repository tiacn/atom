# -*- coding: utf-8 -*-
"""逐项运行 Step 5.0 convention validation。"""

import subprocess
import sys
from pathlib import Path


def main():
    this_dir = Path(__file__).resolve().parent
    scripts = (
        "01_phasor_and_spherical.py",
        "02_lowering_contraction.py",
        "03_absolute_units.py",
        "04_two_level_response.py",
        "05_k39_basis_mapping.py",
        "06_elecsus_crosscheck.py",
    )

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
    print("全部 PASS：Step 5.0 convention 和绝对单位链验证完成")


if __name__ == "__main__":
    main()
