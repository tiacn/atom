# -*- coding: utf-8 -*-
"""逐项运行 Step 5 正式 polarization 验证。"""

import subprocess
import sys
from pathlib import Path


def main():
    this_dir = Path(__file__).resolve().parent
    scripts = [
        "01_population_zero.py",
        "02_single_coherence.py",
        "03_random_contraction.py",
        "04_basis_roundtrip.py",
        "05_two_level_regression.py",
        "06_real_ensemble.py",
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
    print("全部 PASS：Step 5 rho_bar -> dipole -> P 正式验证完成")


if __name__ == "__main__":
    main()

