# -*- coding: utf-8 -*-
"""按顺序执行第二步的全部独立验证。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main():
    this_dir = Path(__file__).resolve().parent
    scripts = [
        "01_atomic_model.py",
        "02_vectorization.py",
        "03_compare_L.py",
        "04_compare_evolution.py",
        "05_detuning.py",
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
    print("全部 PASS：第二步 Liouvillian 独立验证完成")


if __name__ == "__main__":
    main()
