# -*- coding: utf-8 -*-
"""测试 5：5000 条相同真实轨迹的完整 nonlinear effective-response spectrum。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0

import path_setup  # noqa: F401

from benchmark import BenchmarkConfig, LinearThermalResponse, spectrum_metrics
from config import GeometryConfig
from geometry import generate_ensemble
from finite_transit import FiniteTransitConfig, UniformFiniteTransitModel


def trajectory_spectrum(model, trajectories, detuning_axis_MHz, progress_every=0):
    P_q = np.zeros((len(detuning_axis_MHz), 3), dtype=complex)
    dark = np.zeros(len(detuning_axis_MHz))
    excited = np.zeros(len(detuning_axis_MHz))
    for detuning_index, detuning in enumerate(detuning_axis_MHz):
        P_sum = np.zeros(3, dtype=complex)
        dark_sum = 0.0
        excited_sum = 0.0
        for trajectory in trajectories:
            atom = model.propagate(
                trajectory.velocity_m_s[2],
                trajectory.entry.time_s,
                return_rho=False,
                external_detuning_MHz=float(detuning),
            )
            P_sum += atom.P_q
            dark_sum += atom.dark_population
            excited_sum += atom.excited_population
        P_q[detuning_index] = P_sum / len(trajectories)
        dark[detuning_index] = dark_sum / len(trajectories)
        excited[detuning_index] = excited_sum / len(trajectories)
        if progress_every and (detuning_index + 1) % progress_every == 0:
            print(
                f"  spectrum {detuning_index+1}/{len(detuning_axis_MHz)}, "
                f"cache={len(model._cache)}",
                flush=True,
            )
    q_index = model.target_q_index
    chi_effective = P_q[:, q_index] / (
        epsilon_0 * model.target_Ecal_q_V_m
    )
    return P_q, chi_effective, dark, excited


def main():
    print("测试 5：full detuning spectrum")
    geometry = GeometryConfig(
        n_slices=100,
        atoms_per_slice=50,
        temperature_C=20.0,
        seed=55501,
    )
    trajectories, _, _ = generate_ensemble(geometry)
    saturation = 1.0

    # 先在 200 条共同轨迹、9 个 detuning 上验证 cache resolution。
    convergence_axis = np.linspace(-900.0, 700.0, 9)
    convergence_trajectories = trajectories[:200]
    exact_model = UniformFiniteTransitModel(
        FiniteTransitConfig(
            temperature_C=geometry.temperature_C,
            saturation=saturation,
            detuning_cache_resolution_MHz=None,
            track_optical_coherence=False,
        )
    )
    _, exact_chi, _, _ = trajectory_spectrum(
        exact_model,
        convergence_trajectories,
        convergence_axis,
    )
    cache_errors = {}
    for resolution in (1.0, 0.5, 0.25):
        cached_model = UniformFiniteTransitModel(
            FiniteTransitConfig(
                temperature_C=geometry.temperature_C,
                saturation=saturation,
                detuning_cache_resolution_MHz=resolution,
                track_optical_coherence=False,
            )
        )
        _, cached_chi, _, _ = trajectory_spectrum(
            cached_model,
            convergence_trajectories,
            convergence_axis,
        )
        cache_errors[resolution] = float(
            np.linalg.norm(cached_chi - exact_chi) / np.linalg.norm(exact_chi)
        )
        print(
            f"cache resolution {resolution:.2f} MHz spectrum error = "
            f"{cache_errors[resolution]:.3e}"
        )
    assert cache_errors[0.25] < 5e-3

    detuning_axis = np.arange(-1200.0, 1200.1, 10.0)
    model = UniformFiniteTransitModel(
        FiniteTransitConfig(
            temperature_C=geometry.temperature_C,
            B_z_G=0.0,
            external_detuning_MHz=0.0,
            saturation=saturation,
            delta_m=+1,
            detuning_cache_resolution_MHz=0.25,
            track_optical_coherence=False,
        )
    )
    P_trajectory_q, chi_trajectory, dark, excited = trajectory_spectrum(
        model,
        trajectories,
        detuning_axis,
        progress_every=20,
    )

    response = LinearThermalResponse(
        BenchmarkConfig(
            temperature_C=geometry.temperature_C,
            saturation=saturation,
        )
    )
    chi_linear = response.doppler_average(
        detuning_axis,
        B_z_G=0.0,
        delta_m=+1,
        integration_step_MHz=0.25,
        velocity_range_sigma=7.0,
    )
    chi_elecsus = response.elecsus_chi(detuning_axis, 0.0, +1)
    metrics_linear_elecsus = spectrum_metrics(
        detuning_axis,
        chi_linear,
        chi_elecsus,
    )
    metrics_trajectory_linear = spectrum_metrics(
        detuning_axis,
        chi_trajectory,
        chi_linear,
    )
    integrated_difference = float(
        np.linalg.norm(chi_trajectory - chi_linear) / np.linalg.norm(chi_linear)
    )

    print(f"linear-ElecSus complex L2 = {metrics_linear_elecsus.relative_complex_l2:.3e}")
    print(f"trajectory-linear complex L2 = {integrated_difference:.3e}")
    print(
        f"trajectory/linear peak positions = "
        f"{metrics_trajectory_linear.peak_position_model_MHz:+.1f}/"
        f"{metrics_trajectory_linear.peak_position_reference_MHz:+.1f} MHz"
    )
    print(
        f"trajectory-linear FWHM difference = "
        f"{metrics_trajectory_linear.fwhm_error_MHz:+.2f} MHz"
    )
    assert np.all(np.isfinite(chi_trajectory))
    assert metrics_linear_elecsus.relative_complex_l2 < 5e-3
    assert integrated_difference > 0.1
    assert np.max(chi_trajectory.imag) > 0.0

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "detuning_spectrum_5000_atoms.npz"
    np.savez_compressed(
        output_path,
        detuning_MHz=detuning_axis,
        P_trajectory_q=P_trajectory_q,
        chi_effective_trajectory=chi_trajectory,
        chi_linear=chi_linear,
        chi_elecsus=chi_elecsus,
        dark_population=dark,
        excited_population=excited,
        saturation=np.array(saturation),
        atoms_per_slice=np.array(geometry.atoms_per_slice),
        seed=np.array(geometry.seed),
        cache_resolution_MHz=np.array(0.25),
        cache_convergence_resolution_MHz=np.asarray(tuple(cache_errors)),
        cache_convergence_relative_error=np.asarray(tuple(cache_errors.values())),
        trajectory_linear_complex_L2=np.array(integrated_difference),
        trajectory_linear_peak_position_error_MHz=np.array(
            metrics_trajectory_linear.peak_position_error_MHz
        ),
        trajectory_linear_peak_amplitude_error=np.array(
            metrics_trajectory_linear.peak_imag_relative_error
        ),
        trajectory_linear_FWHM_error_MHz=np.array(
            metrics_trajectory_linear.fwhm_error_MHz
        ),
        linear_elecsus_complex_L2=np.array(
            metrics_linear_elecsus.relative_complex_l2
        ),
    )

    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(detuning_axis, chi_elecsus.imag, label="ElecSus")
    axes[0].plot(detuning_axis, chi_linear.imag, "--", label="linear thermal OBE")
    axes[0].plot(detuning_axis, chi_trajectory.imag, label="finite trajectory effective")
    axes[0].set_ylabel("Im effective response")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(detuning_axis, chi_elecsus.real, label="ElecSus")
    axes[1].plot(detuning_axis, chi_linear.real, "--", label="linear thermal OBE")
    axes[1].plot(detuning_axis, chi_trajectory.real, label="finite trajectory effective")
    axes[1].set_xlabel("detuning (MHz, ElecSus external axis)")
    axes[1].set_ylabel("Re effective response")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure_path = output_dir / "detuning_spectrum.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    print(f"saved: {output_path}")
    print(f"saved: {figure_path}")
    print("PASS：完整谱展示 finite-transit nonlinear response 对 linear/ElecSus 的偏离")


if __name__ == "__main__":
    main()

