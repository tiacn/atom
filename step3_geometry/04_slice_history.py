# -*- coding: utf-8 -*-
"""第 4 小步：验证从入口到当前位置的逐切片历史。"""

from collections import Counter

import numpy as np

from config import CONFIG
from geometry import (
    Trajectory,
    find_previous_entry,
    sample_maxwell_boltzmann_velocities,
    sample_positions_by_slice,
    slice_edges,
    split_history_into_slices,
)
from validation import validate_slice_history


def main():
    print("第 4 小步：入口到当前位置的切片累积")
    rng = np.random.default_rng(CONFIG.seed + 3)
    positions, current_slices = sample_positions_by_slice(
        rng,
        CONFIG.n_slices,
        100,
        CONFIG.beam_radius_m,
        CONFIG.cell_length_m,
    )
    velocities, _ = sample_maxwell_boltzmann_velocities(
        rng,
        len(positions),
        CONFIG.temperature_C,
    )
    edges = slice_edges(CONFIG.cell_length_m, CONFIG.n_slices)

    number_of_segments = Counter()
    max_time_error = 0.0
    max_endpoint_error = 0.0

    for position, velocity, current_slice in zip(
        positions,
        velocities,
        current_slices,
    ):
        entry = find_previous_entry(
            position,
            velocity,
            CONFIG.beam_radius_m,
            CONFIG.cell_length_m,
        )
        segments = split_history_into_slices(
            entry,
            position,
            velocity,
            edges,
        )
        trajectory = Trajectory(
            current_position_m=position,
            velocity_m_s=velocity,
            current_slice=int(current_slice),
            entry=entry,
            segments=segments,
        )
        validate_slice_history(trajectory, edges)

        number_of_segments[len(segments)] += 1
        max_time_error = max(
            max_time_error,
            abs(np.sum(trajectory.dt_slices) - entry.time_s),
        )
        max_endpoint_error = max(
            max_endpoint_error,
            np.max(np.abs(segments[-1].end_m - position)),
        )

    print(f"trajectories = {len(positions)}")
    print(f"segment-count range = {min(number_of_segments)}/{max(number_of_segments)}")
    print(f"max sum(dt)-entry_time error = {max_time_error:.3e} s")
    print(f"max reconstructed endpoint error = {max_endpoint_error:.3e} m")
    print("PASS: 切片顺序、停留时间和轨迹端点全部一致")


if __name__ == "__main__":
    main()

