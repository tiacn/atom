# -*- coding: utf-8 -*-
"""第 3 小步：逐切片累积必须等于同一 L 的一次总时间演化。"""

import numpy as np
from scipy.linalg import expm

import path_setup  # noqa: F401

from config import CONFIG
from liouvillian import rho_to_vec, vec_to_rho
from moving_atom import SingleAtomEngine, SliceFields, evolve_trajectory
from trajectory_factory import hand_trajectory


def main():
    print("第 3 小步：均匀场的切片累积")
    trajectory, _ = hand_trajectory()
    engine = SingleAtomEngine()
    fields = SliceFields.uniform(
        CONFIG.n_slices,
        B_z_G=300.0,
        saturation=0.05,
        polarization_cart=(1.0, 0.0, 0.0),
        laser_detuning_MHz=80.0,
    )

    result = evolve_trajectory(trajectory, fields, engine)
    total_time = np.sum(trajectory.dt_slices)
    L = engine.build_L(
        B_z_G=300.0,
        saturation=0.05,
        polarization_cart=(1.0, 0.0, 0.0),
        laser_detuning_MHz=80.0,
        v_z_m_s=trajectory.velocity_m_s[2],
    )
    rho_one_step = vec_to_rho(
        expm(L * total_time) @ rho_to_vec(result.rho_initial),
        engine.N,
    )
    error = np.max(np.abs(result.rho_final - rho_one_step))

    print(f"slice count = {len(result.steps)}")
    print(f"total time = {total_time*1e6:.9f} us")
    print(f"max |rho_sliced-rho_one_step| = {error:.3e}")
    assert error < 1e-10
    print("PASS: 相同 L 的逐切片累积等于一次总时间演化")


if __name__ == "__main__":
    main()

