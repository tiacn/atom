# -*- coding: utf-8 -*-
"""逐项运行 Step 6A。"""

import subprocess
import sys
import time
from pathlib import Path


def main():
    this_dir = Path(__file__).resolve().parent
    started = time.perf_counter()
    for script in ("01_controls.py", "02_memory_scan.py"):
        print()
        print("=" * 78)
        print(f"运行 {script}")
        print("=" * 78)
        subprocess.run(
            [sys.executable, str(this_dir / script)],
            cwd=str(this_dir),
            check=True,
        )
    elapsed = time.perf_counter() - started
    print()
    print(f"全部 PASS：Step 6A magnetic-history benchmark（{elapsed:.3f} s）")


if __name__ == "__main__":
    main()
