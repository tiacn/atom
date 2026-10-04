# -*- coding: utf-8 -*-
"""测试 2：5000 原子单 detuning 时间分箱、空间 profile 和三基准比较。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import path_setup  # noqa: F401

from config import GeometryConfig
from ensemble import validate_density_collection
from geometry import generate_ensemble
from finite_transit import (
    FiniteTransitConfig,
    UniformFiniteTransitModel,
    evolve_finite_transit_ensemble,
    linear_and_elecsus_reference,
    save_ensemble_result,
    time_binned_statistics,
)


def main():
    print("测试 2：5000 atoms, single detuning finite-transit diagnostics")
    geometry = GeometryConfig(
        n_slices=100,
        atoms_per_slice=50,
        temperature_C=20.0,
        seed=55201,
    )
    trajectories, edges, sigma_v = generate_ensemble(geometry)
    config = FiniteTransitConfig(
        temperature_C=geometry.temperature_C,
        B_z_G=0.0,
        external_detuning_MHz=-68.0,
        saturation=1.0e-2,
        delta_m=+1,
        detuning_cache_resolution_MHz=0.25,
    )
    model = UniformFiniteTransitModel(config)
    result = evolve_finite_transit_ensemble(
        trajectories,
        model,
        geometry.n_slices,
        return_rho=True,
        progress_every=1000,
    )
    reference = linear_and_elecsus_reference(config)
    chi_trajectory = model.effective_chi_from_P_q(result.global_P_q)
    deviation_linear = abs(
        chi_trajectory - reference["chi_linear"]
    ) / abs(reference["chi_linear"])
    deviation_elecsus = abs(
        chi_trajectory - reference["chi_elecsus"]
    ) / abs(reference["chi_elecsus"])

    assert result.counts.sum() == geometry.n_atoms
    assert np.all(result.counts == geometry.atoms_per_slice)
    assert result.formal_P_profile_error < 1e-24
    individual_worst = validate_density_collection(
        result.rho_final_all,
        model.N_G,
        tolerance=2e-8,
    )
    average_worst = validate_density_collection(
        result.rho_bar,
        model.N_G,
        tolerance=2e-8,
    )

    bins, bin_edges = time_binned_statistics(result, n_bins=10)
    shortest = np.argsort(result.time_since_entry_s)[:50]
    longest = np.argsort(result.time_since_entry_s)[-50:]
    short_rho_mean = np.mean(result.rho_final_all[shortest], axis=0)
    entry_error_short = float(np.max(np.abs(short_rho_mean - model.rho_entry)))
    dark_spearman = float(
        spearmanr(
            result.time_since_entry_s,
            result.dark_population_all,
        ).statistic
    )
    slice_time_dark_correlation = float(
        np.corrcoef(
            result.mean_time_by_slice_s,
            result.dark_population_by_slice,
        )[0, 1]
    )

    print(f"sigma_v = {sigma_v:.6f} m/s")
    print(
        "transit time us percentiles =",
        np.percentile(
            result.time_since_entry_s * 1e6,
            [0, 1, 10, 50, 90, 99, 100],
        ),
    )
    print(f"shortest-1% mean rho entry error = {entry_error_short:.3e}")
    print(
        f"dark population shortest/longest 1% = "
        f"{np.mean(result.dark_population_all[shortest]):.6f}/"
        f"{np.mean(result.dark_population_all[longest]):.6f}"
    )
    print(f"time-dark Spearman correlation = {dark_spearman:.6f}")
    print(f"slice mean-time/dark correlation = {slice_time_dark_correlation:.6f}")
    print(f"chi trajectory = {chi_trajectory}")
    print(f"chi linear = {reference['chi_linear']}")
    print(f"chi ElecSus = {reference['chi_elecsus']}")
    print(f"trajectory-linear relative deviation = {deviation_linear:.6e}")
    print(f"trajectory-ElecSus relative deviation = {deviation_elecsus:.6e}")
    print(f"individual density worst = {individual_worst}")
    print(f"rho_bar density worst = {average_worst}")
    print("time quantile bins:")
    for row in bins:
        print(
            f"  N={row['count']:4d}, <t>={row['mean_time_s']*1e6:8.4f} us, "
            f"dark={row['mean_dark_population']:.6f}, "
            f"exc={row['mean_excited_population']:.3e}, "
            f"|P|={row['mean_Pq_norm_C_m2']:.3e} C/m^2"
        )

    assert entry_error_short < 2e-2
    assert bins[0]["mean_dark_population"] < bins[-1]["mean_dark_population"]
    assert np.mean(result.dark_population_all[longest]) > np.mean(
        result.dark_population_all[shortest]
    )
    assert dark_spearman > 0.15

    # B=0 下用同一批前 500 条轨迹验证 Delta-m=-1 镜像分量。
    mirror_count = 500
    mirror_model = UniformFiniteTransitModel(
        FiniteTransitConfig(
            temperature_C=config.temperature_C,
            B_z_G=config.B_z_G,
            external_detuning_MHz=config.external_detuning_MHz,
            saturation=config.saturation,
            delta_m=-1,
            detuning_cache_resolution_MHz=0.25,
        )
    )
    mirror = evolve_finite_transit_ensemble(
        trajectories[:mirror_count],
        mirror_model,
        geometry.n_slices,
        return_rho=False,
    )
    dark_symmetry_error = float(
        np.max(
            np.abs(
                mirror.dark_population_all
                - result.dark_population_all[:mirror_count]
            )
        )
    )
    excited_symmetry_error = float(
        np.max(
            np.abs(
                mirror.excited_population_all
                - result.excited_population_all[:mirror_count]
            )
        )
    )
    print(f"Delta-m mirror dark/excited errors = {dark_symmetry_error:.3e}/{excited_symmetry_error:.3e}")
    assert dark_symmetry_error < 1e-9
    assert excited_symmetry_error < 1e-9

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_path = save_ensemble_result(
        result,
        output_dir / "single_detuning_5000_atoms.npz",
        extra={
            "z_edges_m": edges,
            "time_bin_statistics": bins,
            "time_bin_edges_s": bin_edges,
            "chi_trajectory_effective": np.array(chi_trajectory),
            "chi_linear": np.array(reference["chi_linear"]),
            "chi_elecsus": np.array(reference["chi_elecsus"]),
            "trajectory_linear_relative_deviation": np.array(deviation_linear),
            "trajectory_elecsus_relative_deviation": np.array(deviation_elecsus),
            "external_detuning_MHz": np.array(config.external_detuning_MHz),
            "B_z_G": np.array(config.B_z_G),
            "saturation": np.array(config.saturation),
            "delta_m": np.array(config.delta_m),
            "cache_resolution_MHz": np.array(
                config.detuning_cache_resolution_MHz
            ),
            "mirror_dark_population": mirror.dark_population_all,
            "slice_time_dark_correlation": np.array(
                slice_time_dark_correlation
            ),
        },
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    time_us = bins["mean_time_s"] * 1e6
    axes[0].plot(time_us, bins["mean_dark_population"], "o-", label="dark population")
    axes[0].plot(time_us, bins["mean_excited_population"], "s-", label="excited population")
    axes[0].set_xlabel("mean time since entry (us)")
    axes[0].set_ylabel("population")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    z_mm = 0.5 * (edges[:-1] + edges[1:]) * 1e3
    axes[1].plot(z_mm, result.dark_population_by_slice, label="dark population")
    axes[1].set_xlabel("z (mm)")
    axes[1].set_ylabel("slice mean dark population")
    second = axes[1].twinx()
    second.plot(
        z_mm,
        result.mean_time_by_slice_s * 1e6,
        color="tab:orange",
        alpha=0.75,
        label="mean transit time",
    )
    second.set_ylabel("mean time since entry (us)")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure_path = output_dir / "single_detuning_time_and_space.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    print(f"saved: {output_path}")
    print(f"saved: {figure_path}")
    print("PASS：5000 原子 trajectory -> rho_bar -> P 全链和时间 pumping 趋势成立")


if __name__ == "__main__":
    main()
