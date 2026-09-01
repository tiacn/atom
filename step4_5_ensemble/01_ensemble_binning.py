# -*- coding: utf-8 -*-
"""测试 1：N 原子独立演化、当前切片计数和 rho_bar。"""

import numpy as np

import path_setup  # noqa: F401

from config import CONFIG
from ensemble import (
    bin_current_density_matrices,
    evolve_ensemble,
    validate_density_collection,
)
from geometry import generate_ensemble
from moving_atom import SingleAtomEngine
from test_support import nonuniform_fields
from trajectory_factory import hand_trajectory, make_trajectory


def main():
    print("测试 1：N 原子当前切片 binning")
    # 人工放入一条明确经过 [48, 49, 50] 的历史轨迹；另外两条轨迹只在
    # 当前 slice 内运动。这样 historical visits 严格大于原子数，测试不会
    # 因为“所有原子都只有一个 segment”而虚假通过。
    before, _ = make_trajectory(
        position_m=np.array([0.0, 0.0, 2.60e-3]),
        velocity_m_s=np.array([300.0, 0.0, 0.0]),
    )
    multi_slice, _ = hand_trajectory()
    after, _ = make_trajectory(
        position_m=np.array([0.0, 0.0, 20.10e-3]),
        velocity_m_s=np.array([300.0, 0.0, 0.0]),
    )
    trajectories = (before, multi_slice, after)

    assert multi_slice.slice_indices.tolist() == [48, 49, 50]
    assert multi_slice.current_slice == 50

    fields = nonuniform_fields(CONFIG)
    engine = SingleAtomEngine()
    result = evolve_ensemble(trajectories, fields, engine)

    history_visits = sum(len(trajectory.segments) for trajectory in trajectories)
    individual_worst = validate_density_collection(
        result.rho_final_all,
        engine.N_G,
    )
    occupied = result.counts > 0
    average_worst = validate_density_collection(
        result.rho_bar[occupied],
        engine.N_G,
    )

    print(f"N atoms = {len(trajectories)}")
    print(f"current slices = {result.current_slices.tolist()}")
    print(f"multi-slice history = {multi_slice.slice_indices.tolist()}")
    print(f"sum current counts = {result.counts.sum()}")
    print(f"historical slice visits = {history_visits}")
    print(f"individual worst = {individual_worst}")
    print(f"rho_bar worst = {average_worst}")

    assert result.counts.sum() == len(trajectories)
    assert history_visits > len(trajectories)

    expected_current_counts = np.zeros(CONFIG.n_slices, dtype=int)
    np.add.at(expected_current_counts, result.current_slices, 1)
    assert np.array_equal(result.counts, expected_current_counts)

    # 48、49 只属于 multi_slice 原子的历史，不是任何原子的当前位置。
    # 若错误地按历史 segments 做 binning，这两片会出现计数和 rho 贡献。
    assert result.counts[48] == 0
    assert result.counts[49] == 0
    assert np.max(np.abs(result.rho_bar[48])) == 0.0
    assert np.max(np.abs(result.rho_bar[49])) == 0.0
    assert result.counts[50] == 1

    # 每片只有一个当前原子时，rho_bar 必须等于该原子的 rho_final。
    for atom, k in enumerate(result.current_slices):
        assert np.max(np.abs(result.rho_bar[k] - result.rho_final_all[atom])) < 1e-14

    # 目标几何配置（100 slices x 50 atoms）也必须严格满足当前切片计数。
    full_trajectories, _, _ = generate_ensemble(CONFIG)
    full_current_slices = np.array(
        [trajectory.current_slice for trajectory in full_trajectories],
        dtype=int,
    )
    # 计数验证不需要重复做 5000 次昂贵 OBE；使用已验证的入口热态检查 binning。
    thermal_batch = np.repeat(
        engine.initial_thermal_state()[None, :, :],
        len(full_trajectories),
        axis=0,
    )
    full_counts, full_rho_bar = bin_current_density_matrices(
        thermal_batch,
        full_current_slices,
        CONFIG.n_slices,
    )
    print(
        f"target geometry counts min/max = "
        f"{full_counts.min()}/{full_counts.max()}"
    )
    assert full_counts.sum() == CONFIG.n_atoms
    assert np.all(full_counts == CONFIG.atoms_per_slice)
    assert np.max(
        np.abs(full_rho_bar - engine.initial_thermal_state()[None, :, :])
    ) < 1e-14

    print("PASS: 每个原子只按 current_slice 计数一次，rho_bar 定义正确")


if __name__ == "__main__":
    main()
