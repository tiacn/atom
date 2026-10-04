# -*- coding: utf-8 -*-
"""逐项运行 Step 5B。"""

import subprocess
import sys
from pathlib import Path


def main():
    this_dir = Path(__file__).resolve().parent
    scripts = (
        "01_reduced_propagation.py",
        "02_single_detuning.py",
        "03_saturation_scan.py",
        "04_mc_convergence.py",
        "05_detuning_spectrum.py",
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
    print("全部 PASS：Step 5B finite transit-time + optical pumping 验证完成")


if __name__ == "__main__":
    main()
