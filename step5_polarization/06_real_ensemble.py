# -*- coding: utf-8 -*-
"""测试 6：真实 trajectory -> rho_bar -> P，并覆盖全系综零光。"""

from pathlib import Path

import numpy as np

import path_setup  # noqa: F401

from ensemble import evolve_ensemble
from moving_atom import SingleAtomEngine, SliceFields
from numberDensityEqs import numDenK
from polarization import (
    NATURAL_K39_FRACTION,
    P_UNIT,
    density_matrices_to_polarization,
    k39_number_density,
    save_polarization_profile,
)
from test_support import make_ensemble, nonuniform_fields


def validate_profile(profile, n_slices, dimension):
    assert profile.rho_bar.shape == (n_slices, dimension, dimension)
    assert profile.P_q.shape == (n_slices, 3)
    assert profile.P_cart.shape == (n_slices, 3)
    assert np.all(np.isfinite(profile.rho_bar))
    assert np.all(np.isfinite(profile.P_q))
    assert np.all(np.isfinite(profile.P_cart))
    assert profile.number_density_m3 > 0.0
    assert profile.d0_C_m > 0.0


def main():
    print("测试 6：小型真实 ensemble 的 rho_bar -> P")
    config, trajectories, _, _ = make_ensemble(
        n_slices=3,
        atoms_per_slice=1,
        seed=50601,
    )
    engine = SingleAtomEngine()
    C_all = engine.ham.d_q_bare["g->e"]
    density = k39_number_density(config.temperature_C)

    # 与 ElecSus 经验公式直接交叉检查；同位素比例只乘在总 K 数密度之后。
    elecsus_total = float(numDenK(density.temperature_K))
    relative_density_error = abs(
        density.total_potassium_m3 / elecsus_total - 1.0
    )
    assert relative_density_error < 1e-14
    assert density.k39_m3 == NATURAL_K39_FRACTION * density.total_potassium_m3

    fields = nonuniform_fields(config, saturation=0.02, detuning_MHz=35.0)
    ensemble_result = evolve_ensemble(trajectories, fields, engine)
    profile = density_matrices_to_polarization(
        ensemble_result.rho_bar,
        C_all,
        density.k39_m3,
    )
    validate_profile(profile, config.n_slices, engine.N)

    occupied = ensemble_result.counts > 0
    assert np.all(occupied)
    assert ensemble_result.counts.sum() == config.n_atoms
    assert np.linalg.norm(profile.P_q[occupied]) > 0.0
    assert np.linalg.norm(profile.P_cart[occupied]) > 0.0

    # P 对 rho 是线性的：逐原子算 P 后按 current_slice 平均，必须等于 P(rho_bar)。
    for k in np.flatnonzero(occupied):
        atom_indices = np.flatnonzero(ensemble_result.current_slices == k)
        individual_P = []
        for atom_index in atom_indices:
            atom_profile = density_matrices_to_polarization(
                ensemble_result.rho_final_all[atom_index : atom_index + 1],
                C_all,
                density.k39_m3,
            )
            individual_P.append(atom_profile.P_q[0])
        assert np.max(np.abs(profile.P_q[k] - np.mean(individual_P, axis=0))) < 1e-25

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_path = save_polarization_profile(
        profile,
        output_dir / "real_ensemble_polarization.npz",
    )
    with np.load(output_path) as saved:
        assert saved["rho_bar"].shape == (config.n_slices, engine.N, engine.N)
        assert saved["P_q"].shape == (config.n_slices, 3)
        assert saved["P_cart"].shape == (config.n_slices, 3)
        assert str(saved["P_unit"]) == P_UNIT
        assert str(saved["phasor_convention"]) == profile.phasor_convention
        assert str(saved["polarization_phasor_convention"]) == (
            profile.polarization_phasor_convention
        )

    print(f"T = {density.temperature_K:.2f} K")
    print(f"total K density = {density.total_potassium_m3:.9e} m^-3")
    print(f"K39 fraction = {density.k39_fraction:.6f}")
    print(f"N_K39 = {density.k39_m3:.9e} m^-3")
    print(f"max |P_q| = {np.max(np.abs(profile.P_q)):.9e} {P_UNIT}")
    print(f"saved: {output_path}")

    print("测试 6b：同一完整 ensemble 的零光 P(z)=0")
    zero_fields = SliceFields(
        B_z_G=np.linspace(0.0, 3000.0, config.n_slices),
        saturation=np.zeros(config.n_slices),
        polarization_cart=np.tile(
            np.array([1.0, 0.0, 0.0], dtype=complex),
            (config.n_slices, 1),
        ),
        laser_detuning_MHz=np.linspace(-200.0, 200.0, config.n_slices),
    )
    zero_ensemble = evolve_ensemble(trajectories, zero_fields, engine)
    zero_profile = density_matrices_to_polarization(
        zero_ensemble.rho_bar,
        C_all,
        density.k39_m3,
    )
    validate_profile(zero_profile, config.n_slices, engine.N)
    max_zero_P = float(np.max(np.abs(zero_profile.P_q)))
    print(f"zero-light max |P_q| = {max_zero_P:.3e} {P_UNIT}")
    assert max_zero_P < 1e-30
    zero_output_path = save_polarization_profile(
        zero_profile,
        output_dir / "zero_light_ensemble_polarization.npz",
    )
    print(f"saved: {zero_output_path}")
    print("PASS: trajectory -> rho_final -> rho_bar -> SI P 的批量接口和零光极限均正确")


if __name__ == "__main__":
    main()
