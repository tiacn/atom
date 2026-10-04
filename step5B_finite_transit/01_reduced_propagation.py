# -*- coding: utf-8 -*-
"""测试 1：严格 51 维传播、冻结逐 segment 演化和 detuning cache 收敛。"""

import numpy as np

import path_setup  # noqa: F401

from config import GeometryConfig
from ensemble import validate_density_collection
from geometry import generate_ensemble
from moving_atom import SliceFields, evolve_trajectory
from trajectory_factory import hand_trajectory
from finite_transit import (
    FiniteTransitConfig,
    UniformFiniteTransitModel,
    evolve_finite_transit_ensemble,
)


def main():
    print("测试 1：reduced finite-time propagator")
    geometry = GeometryConfig(
        n_slices=8,
        atoms_per_slice=2,
        temperature_C=20.0,
        seed=55101,
    )
    trajectories, _, _ = generate_ensemble(geometry)
    config = FiniteTransitConfig(
        saturation=1e-3,
        external_detuning_MHz=-68.0,
        delta_m=+1,
        detuning_cache_resolution_MHz=None,
    )
    model = UniformFiniteTransitModel(config)
    print(f"full/reduced dimension = {model.N**2}/{len(model.active_indices)}")
    print(f"invariant leakage = {model.invariant_leakage:.3e}")
    assert len(model.active_indices) == 51
    assert model.invariant_leakage == 0.0

    trajectory, hand_edges = hand_trajectory()
    internal_detuning = (
        config.external_detuning_MHz + 15.864
    )
    fields = SliceFields.uniform(
        len(hand_edges) - 1,
        B_z_G=config.B_z_G,
        saturation=config.saturation,
        polarization_cart=model.pylcp_jones,
        laser_detuning_MHz=internal_detuning,
    )
    frozen = evolve_trajectory(trajectory, fields, model.engine)
    reduced = model.propagate(
        trajectory.velocity_m_s[2],
        trajectory.entry.time_s,
        return_rho=True,
    )
    rho_error = np.max(np.abs(frozen.rho_final - reduced.rho_final))
    print(f"segments = {len(trajectory.segments)}")
    print(f"reduced vs frozen rho error = {rho_error:.3e}")
    assert rho_error < 2e-10
    validate_density_collection(reduced.rho_final[None, :, :], model.N_G)

    exact = evolve_finite_transit_ensemble(
        trajectories,
        model,
        geometry.n_slices,
        return_rho=True,
    )
    assert exact.formal_P_profile_error < 1e-25

    print("detuning cache convergence")
    errors = {}
    for resolution in (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
        cached_model = UniformFiniteTransitModel(
            FiniteTransitConfig(
                saturation=config.saturation,
                external_detuning_MHz=config.external_detuning_MHz,
                delta_m=config.delta_m,
                detuning_cache_resolution_MHz=resolution,
            )
        )
        cached = evolve_finite_transit_ensemble(
            trajectories,
            cached_model,
            geometry.n_slices,
            return_rho=True,
        )
        errors[resolution] = {
            "rho": float(np.max(np.abs(cached.rho_final_all - exact.rho_final_all))),
            "P": float(
                np.linalg.norm(cached.P_q_all - exact.P_q_all)
                / np.linalg.norm(exact.P_q_all)
            ),
            "dark": float(
                np.max(
                    np.abs(
                        cached.dark_population_all - exact.dark_population_all
                    )
                )
            ),
        }
        print(f"  {resolution:.2f} MHz: {errors[resolution]}")

    assert errors[0.25]["P"] < 4e-3
    assert errors[0.03125]["P"] < 2e-3
    assert errors[0.03125]["P"] < errors[0.25]["P"]
    assert errors[0.03125]["rho"] < errors[0.25]["rho"]
    print("PASS：严格降维、uniform segment 合并、Step 5 P 和 cache 精度均通过")


if __name__ == "__main__":
    main()
