# -*- coding: utf-8 -*-
"""第 5 小步：使用目标参数生成 5000 条完整几何轨迹。"""

from collections import Counter

import numpy as np

from config import CONFIG
from geometry import generate_ensemble
from validation import validate_entry, validate_slice_history


def main():
    print("第 5 小步：完整几何系综")
    trajectories, edges, sigma = generate_ensemble(CONFIG)

    surfaces = Counter()
    current_counts = Counter()
    times = []
    segment_counts = []

    for trajectory in trajectories:
        validate_entry(
            trajectory.entry,
            trajectory.current_position_m,
            trajectory.velocity_m_s,
            CONFIG.beam_radius_m,
            CONFIG.cell_length_m,
        )
        validate_slice_history(trajectory, edges)
        surfaces[trajectory.entry.surface] += 1
        current_counts[trajectory.current_slice] += 1
        times.append(trajectory.entry.time_s)
        segment_counts.append(len(trajectory.segments))

    times = np.asarray(times)
    segment_counts = np.asarray(segment_counts)
    percent = {
        key: 100.0 * value / len(trajectories)
        for key, value in surfaces.items()
    }

    print(f"atoms = {len(trajectories)}")
    print(f"slices = {CONFIG.n_slices}, dz = {CONFIG.dz_m*1e3:.3f} mm")
    print(f"atoms per current slice = {min(current_counts.values())}")
    print(f"1D velocity sigma = {sigma:.3f} m/s")
    print(f"entry counts = {dict(surfaces)}")
    print(f"entry percentages = {percent}")
    print(
        "time-since-entry quantiles [1,50,99]% (us) = "
        f"{np.quantile(times,[0.01,0.5,0.99])*1e6}"
    )
    print(
        "number-of-segments quantiles [1,50,99]% = "
        f"{np.quantile(segment_counts,[0.01,0.5,0.99])}"
    )

    assert len(trajectories) == CONFIG.n_atoms
    assert all(current_counts[k] == CONFIG.atoms_per_slice for k in range(CONFIG.n_slices))
    assert np.all(times > 0.0)
    assert np.all(segment_counts >= 1)

    example = trajectories[len(trajectories) // 2]
    print("example trajectory fields:")
    print(f"  current_slice = {example.current_slice}")
    print(f"  entry_surface = {example.entry.surface}")
    print(f"  slice_indices = {example.slice_indices}")
    print(f"  dt_slices (ns) = {example.dt_slices*1e9}")
    print("PASS: 5000 条轨迹满足全部有限圆柱和历史切片不变量")


if __name__ == "__main__":
    main()

