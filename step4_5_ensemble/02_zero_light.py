# -*- coding: utf-8 -*-
"""测试 2：全系综零光、非均匀 B。"""

import numpy as np

from ensemble import evolve_ensemble, validate_density_collection
from moving_atom import SingleAtomEngine, SliceFields
from test_support import make_ensemble


def main():
    print("测试 2：全系综零光")
    config, trajectories, _, _ = make_ensemble(
        n_slices=5,
        atoms_per_slice=1,
        seed=45201,
    )
    fields = SliceFields(
        B_z_G=np.linspace(0.0, 3000.0, config.n_slices),
        saturation=np.zeros(config.n_slices),
        polarization_cart=np.tile(
            np.array([1.0, 0.0, 0.0], dtype=complex),
            (config.n_slices, 1),
        ),
        laser_detuning_MHz=np.linspace(-200.0, 200.0, config.n_slices),
    )
    engine = SingleAtomEngine()
    initial = engine.initial_thermal_state()
    result = evolve_ensemble(trajectories, fields, engine)

    individual_error = np.max(np.abs(result.rho_final_all - initial[None, :, :]))
    occupied = result.counts > 0
    average_error = np.max(np.abs(result.rho_bar[occupied] - initial[None, :, :]))
    individual_worst = validate_density_collection(result.rho_final_all, engine.N_G)
    average_worst = validate_density_collection(result.rho_bar[occupied], engine.N_G)

    print(f"individual max thermal error = {individual_error:.3e}")
    print(f"rho_bar max thermal error = {average_error:.3e}")
    print(f"individual worst = {individual_worst}")
    print(f"rho_bar worst = {average_worst}")
    assert individual_error < 1e-10
    assert average_error < 1e-10
    print("PASS: 零光下所有原子和所有 rho_bar 都保持入口热初态")


if __name__ == "__main__":
    main()

