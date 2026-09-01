# -*- coding: utf-8 -*-
"""测试 5：相同当前条件和总时间，不同历史应产生不同 rho。"""

import numpy as np

import path_setup  # noqa: F401

from config import CONFIG
from moving_atom import SingleAtomEngine, SliceFields, evolve_trajectory
from trajectory_factory import hand_trajectory


def make_fields(history_type):
    B = np.full(CONFIG.n_slices, 1200.0)
    if history_type == "B1_to_B2":
        # 手工轨迹经过 [48,49,50]；前两段使用 B1，当前切片使用 B2。
        B[48:50] = 50.0
    elif history_type != "B2_to_B2":
        raise ValueError(history_type)

    return SliceFields(
        B_z_G=B,
        saturation=np.full(CONFIG.n_slices, 0.08),
        polarization_cart=np.tile(
            np.array([1.0, 0.0, 0.0], dtype=complex),
            (CONFIG.n_slices, 1),
        ),
        laser_detuning_MHz=np.full(CONFIG.n_slices, 40.0),
    )


def main():
    print("测试 5：历史记忆")
    trajectory, _ = hand_trajectory()
    engine = SingleAtomEngine()

    result_A = evolve_trajectory(
        trajectory,
        make_fields("B1_to_B2"),
        engine,
    )
    result_B = evolve_trajectory(
        trajectory,
        make_fields("B2_to_B2"),
        engine,
    )

    full_difference = np.linalg.norm(result_A.rho_final - result_B.rho_final)
    excited_A = np.trace(result_A.rho_final[engine.N_G:, engine.N_G:]).real
    excited_B = np.trace(result_B.rho_final[engine.N_G:, engine.N_G:]).real
    optical_difference = np.linalg.norm(
        result_A.rho_final[:engine.N_G, engine.N_G:]
        - result_B.rho_final[:engine.N_G, engine.N_G:]
    )

    print(f"same current slice = {trajectory.current_slice}")
    print(f"same total time = {trajectory.entry.time_s*1e6:.6f} us")
    print(f"full rho Frobenius difference = {full_difference:.6e}")
    print(f"optical coherence difference = {optical_difference:.6e}")
    print(f"excited population A/B = {excited_A:.6e}/{excited_B:.6e}")

    assert full_difference > 1e-7
    assert optical_difference > 1e-9
    print("PASS: 当前条件和总时间相同并不足以确定 rho，过去场历史被保留")


if __name__ == "__main__":
    main()

