# -*- coding: utf-8 -*-
"""第 3 小步：验证有限圆柱的反向入口。"""

from collections import Counter

import numpy as np

from config import CONFIG
from geometry import (
    find_previous_entry,
    sample_maxwell_boltzmann_velocities,
    sample_positions_by_slice,
)
from validation import validate_entry


def deterministic_cases():
    R = CONFIG.beam_radius_m
    L = CONFIG.cell_length_m
    center = np.array([0.0, 0.0, 0.5 * L])

    cases = [
        ("z0", center, np.array([0.0, 0.0, 100.0])),
        ("zL", center, np.array([0.0, 0.0, -100.0])),
        ("side", center, np.array([100.0, 0.0, 0.0])),
        # 侧壁交点会落到 z<0，因此合法入口必须是 z0。
        ("z0", np.array([0.0, 0.0, 1e-4]), np.array([1.0, 0.0, 100.0])),
    ]

    for expected_surface, position, velocity in cases:
        entry = find_previous_entry(position, velocity, R, L)
        validate_entry(entry, position, velocity, R, L)
        assert entry.surface == expected_surface
        print(
            f"case expected={expected_surface}: "
            f"surface={entry.surface}, time={entry.time_s*1e6:.6f} us"
        )


def main():
    print("第 3 小步：反向寻找合法入口")
    deterministic_cases()

    rng = np.random.default_rng(CONFIG.seed + 2)
    positions, _ = sample_positions_by_slice(
        rng,
        CONFIG.n_slices,
        200,
        CONFIG.beam_radius_m,
        CONFIG.cell_length_m,
    )
    velocities, _ = sample_maxwell_boltzmann_velocities(
        rng,
        len(positions),
        CONFIG.temperature_C,
    )

    surfaces = Counter()
    times = []
    for position, velocity in zip(positions, velocities):
        entry = find_previous_entry(
            position,
            velocity,
            CONFIG.beam_radius_m,
            CONFIG.cell_length_m,
        )
        validate_entry(
            entry,
            position,
            velocity,
            CONFIG.beam_radius_m,
            CONFIG.cell_length_m,
        )
        surfaces[entry.surface] += 1
        times.append(entry.time_s)

    print(f"random trajectories = {len(times)}")
    print(f"entry surfaces = {dict(surfaces)}")
    print(f"median time since entry = {np.median(times)*1e6:.6f} us")
    assert sum(surfaces.values()) == len(positions)
    assert set(surfaces) == {"side", "z0", "zL"}

    print("PASS: 所有入口均为反向射线首先遇到的合法有限圆柱边界")


if __name__ == "__main__":
    main()

