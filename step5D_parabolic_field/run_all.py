# -*- coding: utf-8 -*-
"""Run all Step 5D checks in order."""

from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    scripts = (
        "01_high_field_benchmark.py",
        "02_propagator_controls.py",
        "03_spatial_convergence.py",
        "04_trajectory_benchmark.py",
    )
    for script in scripts:
        print(f"\n=== {script} ===", flush=True)
        subprocess.run([sys.executable, "-u", str(root / script)], check=True)
    print("\nALL STEP 5D TESTS PASS", flush=True)


if __name__ == "__main__":
    main()
