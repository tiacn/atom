# -*- coding: utf-8 -*-
"""第 1 小步：打印一条手工单原子轨迹。"""

import numpy as np

from trajectory_factory import hand_trajectory


def main():
    print("第 1 小步：手工单原子轨迹")
    trajectory, _ = hand_trajectory()

    print(f"current position (mm) = {trajectory.current_position_m*1e3}")
    print(f"velocity (m/s) = {trajectory.velocity_m_s}")
    print(f"current slice = {trajectory.current_slice}")
    print(f"entry surface = {trajectory.entry.surface}")
    print(f"entry point (mm) = {trajectory.entry.point_m*1e3}")
    print(f"time since entry (us) = {trajectory.entry.time_s*1e6:.9f}")
    print(f"slice indices = {trajectory.slice_indices}")
    print(f"dt per slice (ns) = {trajectory.dt_slices*1e9}")
    print(f"sum(dt) (us) = {np.sum(trajectory.dt_slices)*1e6:.9f}")

    assert trajectory.entry.surface == "side"
    assert len(trajectory.segments) >= 2
    assert abs(np.sum(trajectory.dt_slices) - trajectory.entry.time_s) < 1e-15
    print("PASS: 手工原子的入口、切片顺序和停留时间可完整重建")


if __name__ == "__main__":
    main()

