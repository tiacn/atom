# -*- coding: utf-8 -*-
"""第 5 小步：随机一个原子并输出完整密度矩阵历史。"""

import numpy as np

import path_setup  # noqa: F401

from config import CONFIG
from moving_atom import (
    SingleAtomEngine,
    SliceFields,
    density_matrix_diagnostics,
    evolve_trajectory,
)
from trajectory_factory import random_trajectory


def main():
    print("第 5 小步：随机单原子完整演化")
    trajectory, _ = random_trajectory()
    engine = SingleAtomEngine()

    # 先用已知、外部给定的切片场；还没有光场自洽反馈。
    fields = SliceFields(
        B_z_G=np.linspace(100.0, 500.0, CONFIG.n_slices),
        saturation=np.full(CONFIG.n_slices, 0.05),
        polarization_cart=np.tile(
            np.array([1.0, 0.0, 0.0], dtype=complex),
            (CONFIG.n_slices, 1),
        ),
        laser_detuning_MHz=np.full(CONFIG.n_slices, 40.0),
    )
    result = evolve_trajectory(trajectory, fields, engine)
    diagnostics = density_matrix_diagnostics(result.rho_final, engine.N_G)

    print(f"current position (mm) = {trajectory.current_position_m*1e3}")
    print(f"velocity (m/s) = {trajectory.velocity_m_s}")
    print(f"entry surface = {trajectory.entry.surface}")
    print(f"entry point (mm) = {trajectory.entry.point_m*1e3}")
    print(f"time since entry (us) = {trajectory.entry.time_s*1e6:.6f}")
    print(f"slice indices = {trajectory.slice_indices}")

    for index, step in enumerate(result.steps):
        excited = np.trace(step.rho_after[engine.N_G:, engine.N_G:]).real
        print(
            f"  step {index}: slice={step.slice_index}, "
            f"dt={step.dt_s*1e9:.3f} ns, "
            f"B={step.B_z_G:.3f} G, s={step.saturation:.3f}, "
            f"det_eff={step.effective_detuning_MHz:.3f} MHz, "
            f"rho_ee={excited:.6e}"
        )

    print(f"final trace = {diagnostics['trace']}")
    print(f"final Hermiticity error = {diagnostics['hermiticity_error']:.3e}")
    print(f"final minimum eigenvalue = {diagnostics['minimum_eigenvalue']:.3e}")
    print(f"final excited population = {diagnostics['excited_population']:.6e}")

    assert abs(diagnostics["trace"] - 1.0) < 1e-9
    assert diagnostics["hermiticity_error"] < 1e-9
    assert diagnostics["minimum_eigenvalue"] > -1e-9
    assert diagnostics["excited_population"] >= -1e-12
    print("PASS: 随机单原子的轨迹、Doppler 和逐切片密度矩阵完整闭环")


if __name__ == "__main__":
    main()

