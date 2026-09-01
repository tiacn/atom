# -*- coding: utf-8 -*-
"""测试 3：全系综中每个原子的均匀 L 分段/总时间等价。"""

import numpy as np
from scipy.linalg import expm

import path_setup  # noqa: F401

from liouvillian import rho_to_vec, vec_to_rho
from moving_atom import SingleAtomEngine, SliceFields, evolve_trajectory
from trajectory_factory import hand_trajectory, random_trajectory


def main():
    print("测试 3：全系综均匀场")
    hand, _ = hand_trajectory()
    random_atom, _ = random_trajectory(seed=45301)
    trajectories = (hand, random_atom)
    fields = SliceFields.uniform(
        100,
        B_z_G=300.0,
        saturation=0.03,
        polarization_cart=(1.0, 0.0, 0.0),
        laser_detuning_MHz=40.0,
    )
    engine = SingleAtomEngine()
    errors = []

    for atom_index, trajectory in enumerate(trajectories):
        result = evolve_trajectory(trajectory, fields, engine)
        L = engine.build_L(
            B_z_G=300.0,
            saturation=0.03,
            polarization_cart=(1.0, 0.0, 0.0),
            laser_detuning_MHz=40.0,
            v_z_m_s=trajectory.velocity_m_s[2],
        )
        total_time = trajectory.entry.time_s
        rho_direct = vec_to_rho(
            expm(L * total_time) @ rho_to_vec(result.rho_initial),
            engine.N,
        )
        error = np.max(np.abs(result.rho_final - rho_direct))
        errors.append(error)
        print(
            f"atom {atom_index}: segments={len(trajectory.segments)}, "
            f"flight={total_time*1e6:.6f} us, error={error:.3e}"
        )

    assert len(trajectories[0].segments) > 1
    assert max(errors) < 1e-9
    print("PASS: 系综中每个原子的均匀场分段演化等于总时间演化")


if __name__ == "__main__":
    main()
