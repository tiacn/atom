# -*- coding: utf-8 -*-
"""第 2 小步：零光场下运动原子必须保持热初态。"""

import numpy as np

import path_setup  # noqa: F401

from config import CONFIG
from moving_atom import (
    SingleAtomEngine,
    SliceFields,
    density_matrix_diagnostics,
    evolve_trajectory,
)
from trajectory_factory import hand_trajectory


def main():
    print("第 2 小步：零光场运动")
    trajectory, _ = hand_trajectory()
    engine = SingleAtomEngine()

    # 故意使用非均匀 B，确认没有光时不会凭空产生激发或布居变化。
    fields = SliceFields(
        B_z_G=np.linspace(0.0, 3000.0, CONFIG.n_slices),
        saturation=np.zeros(CONFIG.n_slices),
        polarization_cart=np.tile(
            np.array([1.0, 0.0, 0.0], dtype=complex),
            (CONFIG.n_slices, 1),
        ),
        laser_detuning_MHz=np.full(CONFIG.n_slices, 120.0),
    )
    result = evolve_trajectory(trajectory, fields, engine)
    difference = np.max(np.abs(result.rho_final - result.rho_initial))
    diagnostics = density_matrix_diagnostics(result.rho_final, engine.N_G)

    print(f"segments evolved = {len(result.steps)}")
    print(f"max |rho_final-rho_initial| = {difference:.3e}")
    print(f"trace = {diagnostics['trace']}")
    print(f"excited population = {diagnostics['excited_population']:.3e}")

    assert difference < 1e-11
    assert abs(diagnostics["trace"] - 1.0) < 1e-11
    assert diagnostics["excited_population"] < 1e-12
    print("PASS: 零光场下任意轨迹和非均匀 B 不改变热初态")


if __name__ == "__main__":
    main()

