# -*- coding: utf-8 -*-
"""让 Step 5.0 始终使用工作区源码和冻结的 Step 2 接口。"""

from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[3]
SOURCE_DIRS = (
    WORKSPACE / "MC" / "step_by_step" / "step2_liouvillian",
    WORKSPACE / "pylcp",
    WORKSPACE / "ElecSus" / "elecsus" / "libs",
)

for source_dir in reversed(SOURCE_DIRS):
    source = str(source_dir)
    if source not in sys.path:
        sys.path.insert(0, source)
