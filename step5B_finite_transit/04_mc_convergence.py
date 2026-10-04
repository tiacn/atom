# -*- coding: utf-8 -*-
"""测试 4：finite-transit P、dark population 和 response difference 的 MC 收敛。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import path_setup  # noqa: F401

from config import GeometryConfig
from geometry import generate_ensemble
from polarization import spherical_to_cartesian
from finite_transit import (
    FiniteTransitConfig,
    UniformFiniteTransitModel,
    evolve_finite_transit_ensemble,
    linear_and_elecsus_reference,
)


def slice_profile(values, slices, selected, n_slices):
    values = np.asarray(values)
    output = np.zeros((n_slices,) + values.shape[1:], dtype=values.dtype)
    counts = np.bincount(slices[selected], minlength=n_slices)
    np.add.at(output, slices[selected], values[selected])
    divisor = counts.reshape((n_slices,) + (1,) * (values.ndim - 1))
    output /= divisor
    return output


def complex_dispersion(values, axis=0):
    mean = np.mean(values, axis=axis, keepdims=True)
    return np.sqrt(np.mean(np.abs(values - mean) ** 2, axis=axis))


def main():
    print("测试 4：MC convergence at finite-transit crossover")
    atoms_per_slice_values = np.array([20, 50, 100, 200])
    seeds = np.array([55401, 55402, 55403, 55404, 55405])
    n_slices = 100
    config = FiniteTransitConfig(
        temperature_C=20.0,
        B_z_G=0.0,
        external_detuning_MHz=-68.0,
        saturation=1.0,
        delta_m=+1,
        detuning_cache_resolution_MHz=0.25,
    )
    model = UniformFiniteTransitModel(config)
    reference = linear_and_elecsus_reference(config)
    weak_config = FiniteTransitConfig(
        temperature_C=20.0,
        B_z_G=0.0,
        external_detuning_MHz=-68.0,
        saturation=1.0e-6,
        delta_m=+1,
        detuning_cache_resolution_MHz=0.25,
        track_optical_coherence=False,
    )
    weak_model = UniformFiniteTransitModel(weak_config)
    weak_reference = linear_and_elecsus_reference(weak_config)
    weak_chi_200 = np.zeros(len(seeds), dtype=complex)

    shape = (len(atoms_per_slice_values), len(seeds))
    chi = np.zeros(shape, dtype=complex)
    dark_mean = np.zeros(shape)
    excited_mean = np.zeros(shape)
    optical_mean = np.zeros(shape)
    P_q_profiles = np.zeros(shape + (n_slices, 3), dtype=complex)
    P_cart_profiles = np.zeros_like(P_q_profiles)
    dark_profiles = np.zeros(shape + (n_slices,))

    for seed_index, seed in enumerate(seeds):
        geometry = GeometryConfig(
            n_slices=n_slices,
            atoms_per_slice=int(atoms_per_slice_values[-1]),
            temperature_C=config.temperature_C,
            seed=int(seed),
        )
        trajectories, _, _ = generate_ensemble(geometry)
        full = evolve_finite_transit_ensemble(
            trajectories,
            model,
            n_slices,
            return_rho=False,
            progress_every=10000,
        )
        weak_full = evolve_finite_transit_ensemble(
            trajectories,
            weak_model,
            n_slices,
            return_rho=False,
        )
        weak_chi_200[seed_index] = weak_model.effective_chi_from_P_q(
            weak_full.global_P_q
        )
        for size_index, atoms_per_slice in enumerate(atoms_per_slice_values):
            selected_parts = []
            for k in range(n_slices):
                in_slice = np.flatnonzero(full.current_slices == k)
                selected_parts.append(in_slice[:atoms_per_slice])
            selected = np.concatenate(selected_parts)
            P_q = slice_profile(
                full.P_q_all,
                full.current_slices,
                selected,
                n_slices,
            )
            dark = slice_profile(
                full.dark_population_all,
                full.current_slices,
                selected,
                n_slices,
            )
            P_q_profiles[size_index, seed_index] = P_q
            P_cart_profiles[size_index, seed_index] = spherical_to_cartesian(P_q)
            dark_profiles[size_index, seed_index] = dark
            global_P = np.mean(full.P_q_all[selected], axis=0)
            chi[size_index, seed_index] = model.effective_chi_from_P_q(global_P)
            dark_mean[size_index, seed_index] = np.mean(
                full.dark_population_all[selected]
            )
            excited_mean[size_index, seed_index] = np.mean(
                full.excited_population_all[selected]
            )
            optical_mean[size_index, seed_index] = np.mean(
                full.optical_coherence_norm_all[selected]
            )
        print(f"completed seed {seed}")

    chi_dispersion = complex_dispersion(chi, axis=1)
    chi_relative_dispersion = chi_dispersion / np.abs(np.mean(chi, axis=1))
    Pq_profile_dispersion = np.array(
        [
            np.sqrt(
                np.mean(
                    np.abs(
                        profiles - np.mean(profiles, axis=0, keepdims=True)
                    )
                    ** 2
                )
            )
            / np.sqrt(np.mean(np.abs(np.mean(profiles, axis=0)) ** 2))
            for profiles in P_q_profiles
        ]
    )
    Pcart_profile_dispersion = np.array(
        [
            np.sqrt(
                np.mean(
                    np.abs(
                        profiles - np.mean(profiles, axis=0, keepdims=True)
                    )
                    ** 2
                )
            )
            / np.sqrt(np.mean(np.abs(np.mean(profiles, axis=0)) ** 2))
            for profiles in P_cart_profiles
        ]
    )
    dark_profile_dispersion = np.array(
        [
            np.sqrt(
                np.mean(
                    (
                        profiles - np.mean(profiles, axis=0, keepdims=True)
                    )
                    ** 2
                )
            )
            for profiles in dark_profiles
        ]
    )
    response_difference = chi - reference["chi_linear"]
    difference_dispersion = complex_dispersion(response_difference, axis=1)
    weak_mean = np.mean(weak_chi_200)
    weak_dispersion = float(complex_dispersion(weak_chi_200, axis=0))
    weak_mean_linear_error = float(
        abs(weak_mean - weak_reference["chi_linear"])
        / abs(weak_reference["chi_linear"])
    )
    weak_mean_elecsus_error = float(
        abs(weak_mean - weak_reference["chi_elecsus"])
        / abs(weak_reference["chi_elecsus"])
    )

    for index, atoms_per_slice in enumerate(atoms_per_slice_values):
        print(
            f"N/slice={atoms_per_slice:3d}: "
            f"chi rel dispersion={chi_relative_dispersion[index]:.3e}, "
            f"Pq profile={Pq_profile_dispersion[index]:.3e}, "
            f"Pcart profile={Pcart_profile_dispersion[index]:.3e}, "
            f"dark profile abs={dark_profile_dispersion[index]:.3e}, "
            f"chi_disp*sqrt(N)={chi_dispersion[index]*np.sqrt(atoms_per_slice):.3e}"
        )

    assert chi_relative_dispersion[-1] < chi_relative_dispersion[0]
    assert Pq_profile_dispersion[-1] < Pq_profile_dispersion[0]
    assert Pcart_profile_dispersion[-1] < Pcart_profile_dispersion[0]
    assert dark_profile_dispersion[-1] < dark_profile_dispersion[0]
    assert difference_dispersion[-1] < difference_dispersion[0]
    print(f"weak 200/slice seed chi = {weak_chi_200}")
    print(f"weak seed-mean vs linear error = {weak_mean_linear_error:.3e}")
    print(f"weak seed-mean vs ElecSus error = {weak_mean_elecsus_error:.3e}")
    print(f"weak seed dispersion = {weak_dispersion:.3e}")
    assert weak_mean_linear_error < 0.05
    assert weak_mean_elecsus_error < 0.05

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mc_convergence.npz"
    np.savez_compressed(
        output_path,
        atoms_per_slice=atoms_per_slice_values,
        seeds=seeds,
        chi_effective=chi,
        chi_linear=np.array(reference["chi_linear"]),
        chi_elecsus=np.array(reference["chi_elecsus"]),
        response_difference=response_difference,
        dark_population_mean=dark_mean,
        excited_population_mean=excited_mean,
        optical_coherence_norm_mean=optical_mean,
        P_q_profiles=P_q_profiles,
        P_cart_profiles=P_cart_profiles,
        dark_population_profiles=dark_profiles,
        chi_relative_dispersion=chi_relative_dispersion,
        Pq_profile_relative_dispersion=Pq_profile_dispersion,
        Pcart_profile_relative_dispersion=Pcart_profile_dispersion,
        dark_profile_absolute_dispersion=dark_profile_dispersion,
        response_difference_dispersion=difference_dispersion,
        chi_dispersion_sqrt_atoms_per_slice=(
            chi_dispersion * np.sqrt(atoms_per_slice_values)
        ),
        weak_chi_200_atoms_per_slice=weak_chi_200,
        weak_chi_linear=np.array(weak_reference["chi_linear"]),
        weak_chi_elecsus=np.array(weak_reference["chi_elecsus"]),
        weak_seed_mean_linear_relative_error=np.array(weak_mean_linear_error),
        weak_seed_mean_elecsus_relative_error=np.array(weak_mean_elecsus_error),
        weak_seed_dispersion=np.array(weak_dispersion),
    )

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.loglog(
        atoms_per_slice_values,
        chi_relative_dispersion,
        "o-",
        label="global effective response",
    )
    axis.loglog(
        atoms_per_slice_values,
        Pq_profile_dispersion,
        "s-",
        label="Pq(z) profile",
    )
    guide = chi_relative_dispersion[0] * np.sqrt(
        atoms_per_slice_values[0] / atoms_per_slice_values
    )
    axis.loglog(atoms_per_slice_values, guide, "--", label="1/sqrt(N) guide")
    axis.set_xlabel("atoms per slice")
    axis.set_ylabel("relative seed dispersion")
    axis.grid(alpha=0.3, which="both")
    axis.legend()
    figure.tight_layout()
    figure_path = output_dir / "mc_convergence.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    print(f"saved: {output_path}")
    print(f"saved: {figure_path}")
    print("PASS：P、coherence、dark population 和 trajectory-linear difference 均呈 MC 收敛")


if __name__ == "__main__":
    main()
