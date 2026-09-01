# -*- coding: utf-8 -*-
"""加载已经验证的 step2、step3、step4。"""

import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
DIRECTORIES = [
    WORKSPACE / "MC" / "step_by_step" / "step3_geometry",
    WORKSPACE / "MC" / "step_by_step" / "step2_liouvillian",
    WORKSPACE / "MC" / "step_by_step" / "step4_single_moving_atom",
]

for directory in DIRECTORIES:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

