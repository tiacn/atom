# -*- coding: utf-8 -*-
"""测试 6：同一连续 B(z) 在 25/50/100/200/400 slices 下的收敛。"""

import numpy as np

import path_setup  # noqa: F401

from config import CONFIG
from geometry import Trajectory, slice_edges, split_history_into_slices
from moving_atom import SingleAtomEngine, SliceFields, evolve_trajectory
from trajectory_factory import hand_trajectory


SLICE_COUNTS = (25, 50, 100, 200, 400)


def continuous_B(z_m):
    x = np.asarray(z_m) / CONFIG.cell_length_m
    return 100.0 + 1400.0 * x**2 + 250.0 * np.sin(2.0 * np.pi * x)


def trajectory_for_edges(base, edges):
    segments = split_history_into_slices(
        base.entry,
        base.current_position_m,
        base.velocity_m_s,
        edges,
    )
    current_slice = int(
        np.clip(
            np.searchsorted(edges, base.current_position_m[2], side="right") - 1,
            0,
            len(edges) - 2,
        )
    )
    return Trajectory(
        current_position_m=base.current_position_m,
        velocity_m_s=base.velocity_m_s,
        current_slice=current_slice,
        entry=base.entry,
        segments=segments,
    )


def fields_for_edges(edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_slices = len(centers)
    return SliceFields(
        B_z_G=continuous_B(centers),
        saturation=np.full(n_slices, 0.05),
        polarization_cart=np.tile(
            np.array([1.0, 0.0, 0.0], dtype=complex),
            (n_slices, 1),
        ),
        laser_detuning_MHz=np.full(n_slices, 40.0),
    )


def main():
    print("测试 6：slice convergence")
    base, _ = hand_trajectory()
    engine = SingleAtomEngine()
    rho_by_count = {}

    for n_slices in SLICE_COUNTS:
        edges = slice_edges(CONFIG.cell_length_m, n_slices)
        trajectory = trajectory_for_edges(base, edges)
        result = evolve_trajectory(
            trajectory,
            fields_for_edges(edges),
            engine,
        )
        rho_by_count[n_slices] = result.rho_final
        print(
            f"n_slices={n_slices:3d}, "
            f"history segments={len(trajectory.segments):2d}"
        )

    reference = rho_by_count[400]
    errors = {}
    for n_slices in SLICE_COUNTS[:-1]:
        errors[n_slices] = np.linalg.norm(rho_by_count[n_slices] - reference)
        print(f"  ||rho[{n_slices}]-rho[400]|| = {errors[n_slices]:.6e}")

    assert errors[100] < errors[25]
    assert errors[200] < errors[50]
    assert errors[200] < 5e-3
    print("PASS: 连续 B(z) 的最终 rho 随 z 切片加密而收敛")


if __name__ == "__main__":
    main()

