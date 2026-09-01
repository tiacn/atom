# -*- coding: utf-8 -*-
"""第四步测试使用的手工与随机单原子轨迹。"""

import numpy as np

import path_setup  # noqa: F401

from config import CONFIG
from geometry import (
    Trajectory,
    find_previous_entry,
    sample_maxwell_boltzmann_velocities,
    sample_positions_by_slice,
    slice_edges,
    split_history_into_slices,
)
from validation import validate_entry, validate_slice_history


def make_trajectory(position_m, velocity_m_s):
    position = np.asarray(position_m, dtype=float)
    velocity = np.asarray(velocity_m_s, dtype=float)
    edges = slice_edges(CONFIG.cell_length_m, CONFIG.n_slices)
    current_slice = int(
        np.clip(
            np.searchsorted(edges, position[2], side="right") - 1,
            0,
            CONFIG.n_slices - 1,
        )
    )
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
        current_slice=current_slice,
        entry=entry,
        segments=segments,
    )
    validate_entry(
        entry,
        position,
        velocity,
        CONFIG.beam_radius_m,
        CONFIG.cell_length_m,
    )
    validate_slice_history(trajectory, edges)
    return trajectory, edges


def hand_trajectory():
    """一条会从侧壁进入并向 +z 穿过多个切片的可读轨迹。"""
    return make_trajectory(
        position_m=np.array([0.0, 0.0, 12.65e-3]),
        velocity_m_s=np.array([100.0, 20.0, 100.0]),
    )


def random_trajectory(seed=20260808):
    rng = np.random.default_rng(seed)
    positions, slices = sample_positions_by_slice(
        rng,
        CONFIG.n_slices,
        1,
        CONFIG.beam_radius_m,
        CONFIG.cell_length_m,
    )
    selected = int(rng.integers(0, CONFIG.n_slices))
    velocity, _ = sample_maxwell_boltzmann_velocities(
        rng,
        1,
        CONFIG.temperature_C,
    )
    trajectory, edges = make_trajectory(positions[selected], velocity[0])
    assert trajectory.current_slice == slices[selected]
    return trajectory, edges

