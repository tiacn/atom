# -*- coding: utf-8 -*-
"""加载 Step 5.5A 所需的冻结模块。"""

from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[3]
SOURCE_DIRS = (
    WORKSPACE / "MC" / "step_by_step" / "step5_polarization",
    WORKSPACE / "MC" / "step_by_step" / "step5_0_convention_validation",
    WORKSPACE / "MC" / "step_by_step" / "step4_5_ensemble",
    WORKSPACE / "MC" / "step_by_step" / "step4_single_moving_atom",
    WORKSPACE / "MC" / "step_by_step" / "step3_geometry",
    WORKSPACE / "MC" / "step_by_step" / "step2_liouvillian",
    WORKSPACE / "pylcp",
    WORKSPACE / "ElecSus" / "elecsus" / "libs",
)

for source_dir in reversed(SOURCE_DIRS):
    source = str(source_dir)
    if source not in sys.path:
        sys.path.insert(0, source)

