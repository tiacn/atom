# -*- coding: utf-8 -*-
"""把已验证的第二步和第三步目录加入模块搜索路径。"""

import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
STEP2 = WORKSPACE / "MC" / "step_by_step" / "step2_liouvillian"
STEP3 = WORKSPACE / "MC" / "step_by_step" / "step3_geometry"

for directory in (STEP3, STEP2):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

