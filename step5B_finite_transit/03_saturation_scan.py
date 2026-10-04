# -*- coding: utf-8 -*-
"""测试 3：从弱光有限 transit 到明显 optical pumping 的 saturation 扫描。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import path_setup  # noqa: F401

from benchmark import LinearThermalResponse
from config import GeometryConfig
from geometry import generate_ensemble
from finite_transit import (
    FiniteTransitConfig,
    UniformFiniteTransitModel,
    evolve_finite_transit_ensemble,
    linear_and_elecsus_reference,
    pumping_time_1_minus_inv_e,
    pumping_time_dark_fraction,
)


def main():
    print("测试 3：saturation scan and pumping-time crossover")
    geometry = GeometryConfig(
        n_slices=100,
        atoms_per_slice=20,
        temperature_C=20.0,
        seed=55301,
    )
    trajectories, _, _ = generate_ensemble(geometry)
    transit_times = np.array([item.entry.time_s for item in trajectories])
    saturations = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    resonant_velocity_m_s = (
        -68.0 - (-168.177645075)
    ) * 0.770108385049

    dtype = [
        ("saturation", float),
        ("mean_transit_s", float),
        ("median_transit_s", float),
        ("mean_dark_population", float),
        ("median_dark_population", float),
        ("mean_excited_population", float),
        ("mean_optical_coherence_norm", float),
        ("chi_effective", complex),
        ("chi_linear", complex),
        ("chi_elecsus", complex),
        ("deviation_from_linear", float),
        ("deviation_from_elecsus", float),
        ("deviation_from_weak_common_MC", float),
        ("tau_dark_10pct_s", float),
        ("tau_full_1_minus_inv_e_s", float),
        ("median_transit_over_tau10", float),
    ]
    rows = np.zeros(len(saturations), dtype=dtype)
    P_profiles = np.zeros((len(saturations), geometry.n_slices, 3), dtype=complex)
    dark_profiles = np.zeros((len(saturations), geometry.n_slices))
    pumping_curve_times = np.logspace(-9, 0, 500)
    pumping_curves = np.zeros((len(saturations), len(pumping_curve_times)))
    weak_chi = None

    for index, saturation in enumerate(saturations):
        config = FiniteTransitConfig(
            temperature_C=geometry.temperature_C,
            B_z_G=0.0,
            external_detuning_MHz=-68.0,
            saturation=float(saturation),
            delta_m=+1,
            detuning_cache_resolution_MHz=0.25,
        )
        model = UniformFiniteTransitModel(config)
        result = evolve_finite_transit_ensemble(
            trajectories,
            model,
            geometry.n_slices,
            return_rho=False,
        )
        reference = linear_and_elecsus_reference(config)
        chi = model.effective_chi_from_P_q(result.global_P_q)
        if weak_chi is None:
            weak_chi = chi
        tau10, _, _ = pumping_time_dark_fraction(
            model,
            fraction_of_full_change=0.1,
            v_z_m_s=resonant_velocity_m_s,
        )
        tau_full, _, _ = pumping_time_1_minus_inv_e(
            model,
            v_z_m_s=resonant_velocity_m_s,
        )
        pumping_curve = model.pumping_curve(
            pumping_curve_times,
            v_z_m_s=resonant_velocity_m_s,
        )
        pumping_curves[index] = pumping_curve["dark_population"]

        rows[index] = (
            saturation,
            float(np.mean(transit_times)),
            float(np.median(transit_times)),
            float(np.mean(result.dark_population_all)),
            float(np.median(result.dark_population_all)),
            float(np.mean(result.excited_population_all)),
            float(np.mean(result.optical_coherence_norm_all)),
            chi,
            reference["chi_linear"],
            reference["chi_elecsus"],
            abs(chi - reference["chi_linear"]) / abs(reference["chi_linear"]),
            abs(chi - reference["chi_elecsus"]) / abs(reference["chi_elecsus"]),
            abs(chi - weak_chi) / abs(weak_chi),
            tau10,
            tau_full,
            float(np.median(transit_times) / tau10),
        )
        P_profiles[index] = result.P_q_by_slice
        dark_profiles[index] = result.dark_population_by_slice
        print(
            f"s={saturation:.0e}: dark={rows[index]['mean_dark_population']:.6f}, "
            f"exc={rows[index]['mean_excited_population']:.3e}, "
            f"dev(linear)={rows[index]['deviation_from_linear']:.3e}, "
            f"dev(common weak)={rows[index]['deviation_from_weak_common_MC']:.3e}, "
            f"tau10={tau10*1e6:.3e} us, "
            f"median/tau10={rows[index]['median_transit_over_tau10']:.3e}"
        )

    # 相同 MC 样本下，最低三个 s 应处在近似强度无关区；高 s 必须出现明显变化。
    low_spread = np.max(
        np.abs(rows["chi_effective"][:3] / rows["chi_effective"][0] - 1.0)
    )
    high_dark_change = rows["mean_dark_population"][-1] - rows["mean_dark_population"][0]
    high_response_change = rows["deviation_from_weak_common_MC"][-1]
    print(f"low-s common-MC chi spread = {low_spread:.3e}")
    print(f"high-s mean dark increase = {high_dark_change:.3e}")
    print(f"high-s response change from weak = {high_response_change:.3e}")
    assert low_spread < 5e-3
    assert high_dark_change > 5e-3
    assert high_response_change > 0.1

    # 精确 t=0 是 rho_entry、P=0；长时间则连接 full steady dark state。
    strong_model = UniformFiniteTransitModel(
        FiniteTransitConfig(
            saturation=1.0,
            external_detuning_MHz=-68.0,
            detuning_cache_resolution_MHz=None,
        )
    )
    limits = strong_model.pumping_curve(
        np.array([0.0, 1e-8, 1e-6, 1.0]),
        v_z_m_s=resonant_velocity_m_s,
    )
    assert abs(limits["dark_population"][0] - 0.125) < 1e-12
    assert np.linalg.norm(limits["P_q"][0]) < 1e-25
    assert limits["dark_population"][-1] > 0.999
    assert np.linalg.norm(limits["P_q"][-1]) < 1e-20

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "saturation_scan.npz"
    np.savez_compressed(
        output_path,
        saturation_statistics=rows,
        P_q_by_slice=P_profiles,
        dark_population_by_slice=dark_profiles,
        pumping_curve_time_s=pumping_curve_times,
        pumping_curve_dark_population=pumping_curves,
        resonant_velocity_m_s=np.array(resonant_velocity_m_s),
        exact_limit_times_s=np.array([0.0, 1e-8, 1e-6, 1.0]),
        exact_limit_dark_population=limits["dark_population"],
        exact_limit_P_q=limits["P_q"],
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogx(
        saturations,
        rows["mean_dark_population"],
        "o-",
        label="ensemble mean dark",
    )
    axes[0].set_xlabel("saturation s")
    axes[0].set_ylabel("dark population")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].loglog(
        saturations,
        np.maximum(rows["deviation_from_weak_common_MC"], 1e-8),
        "o-",
        label="response deviation from weak MC",
    )
    axes[1].loglog(
        saturations,
        rows["median_transit_over_tau10"],
        "s-",
        label="median transit / tau10",
    )
    axes[1].set_xlabel("saturation s")
    axes[1].set_ylabel("dimensionless")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()
    figure.tight_layout()
    figure_path = output_dir / "saturation_crossover.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    print(f"saved: {output_path}")
    print(f"saved: {figure_path}")
    print("PASS：弱光公共 MC 极限、finite-transit crossover 和长时间暗态已连续连接")


if __name__ == "__main__":
    main()

