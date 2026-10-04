# -*- coding: utf-8 -*-
"""弱光下比较 ElecSus、线性热 OBE 与有限轨迹 OBE 的完整失谐谱。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0

import path_setup  # noqa: F401

from benchmark import BenchmarkConfig, LinearThermalResponse, spectrum_metrics
from config import GeometryConfig
from finite_transit import FiniteTransitConfig, UniformFiniteTransitModel
from geometry import generate_ensemble


def trajectory_spectrum(model, trajectories, detuning_axis_MHz, progress_every=0):
    """对同一批真实轨迹逐失谐计算有限时间的系综平均响应。"""
    P_q = np.zeros((len(detuning_axis_MHz), 3), dtype=complex)
    for detuning_index, detuning in enumerate(detuning_axis_MHz):
        P_sum = np.zeros(3, dtype=complex)
        for trajectory in trajectories:
            atom = model.propagate(
                trajectory.velocity_m_s[2],
                trajectory.entry.time_s,
                return_rho=False,
                external_detuning_MHz=float(detuning),
            )
            P_sum += atom.P_q
        P_q[detuning_index] = P_sum / len(trajectories)
        if progress_every and (detuning_index + 1) % progress_every == 0:
            print(
                f"  weak spectrum {detuning_index + 1}/{len(detuning_axis_MHz)}, "
                f"cache={len(model._cache)}",
                flush=True,
            )

    q_index = model.target_q_index
    chi_effective = P_q[:, q_index] / (
        epsilon_0 * model.target_Ecal_q_V_m
    )
    return P_q, chi_effective


def relative_l2(model, reference):
    return float(np.linalg.norm(model - reference) / np.linalg.norm(reference))


def main():
    print("附加验证：s=1e-6 弱光完整谱三曲线比较")
    saturation = 1.0e-6
    geometry = GeometryConfig(
        n_slices=100,
        atoms_per_slice=50,
        temperature_C=20.0,
        seed=55501,
    )
    trajectories, _, _ = generate_ensemble(geometry)
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
    P_trajectory_q, chi_trajectory = trajectory_spectrum(
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

    trajectory_linear_l2 = relative_l2(chi_trajectory, chi_linear)
    trajectory_elecsus_l2 = relative_l2(chi_trajectory, chi_elecsus)
    linear_elecsus_metrics = spectrum_metrics(
        detuning_axis,
        chi_linear,
        chi_elecsus,
    )
    trajectory_linear_correlation = float(
        abs(np.vdot(chi_linear, chi_trajectory))
        / (np.linalg.norm(chi_linear) * np.linalg.norm(chi_trajectory))
    )

    print(f"trajectory-linear complex L2 = {trajectory_linear_l2:.6e}")
    print(f"trajectory-ElecSus complex L2 = {trajectory_elecsus_l2:.6e}")
    print(
        "linear-ElecSus complex L2 = "
        f"{linear_elecsus_metrics.relative_complex_l2:.6e}"
    )
    print(
        "trajectory-linear complex correlation = "
        f"{trajectory_linear_correlation:.9f}"
    )

    assert np.all(np.isfinite(chi_trajectory))
    assert linear_elecsus_metrics.relative_complex_l2 < 5.0e-3
    assert trajectory_linear_l2 < 1.0e-1
    assert trajectory_linear_correlation > 0.995

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "weak_light_detuning_spectrum_5000_atoms.npz"
    np.savez_compressed(
        data_path,
        detuning_MHz=detuning_axis,
        P_trajectory_q=P_trajectory_q,
        chi_effective_trajectory=chi_trajectory,
        chi_linear=chi_linear,
        chi_elecsus=chi_elecsus,
        saturation=np.array(saturation),
        atoms_per_slice=np.array(geometry.atoms_per_slice),
        total_atoms=np.array(len(trajectories)),
        seed=np.array(geometry.seed),
        trajectory_linear_complex_L2=np.array(trajectory_linear_l2),
        trajectory_elecsus_complex_L2=np.array(trajectory_elecsus_l2),
        linear_elecsus_complex_L2=np.array(
            linear_elecsus_metrics.relative_complex_l2
        ),
        trajectory_linear_complex_correlation=np.array(
            trajectory_linear_correlation
        ),
    )

    scale = float(np.max(np.abs(chi_elecsus)))
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(8, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.65]},
    )
    figure.suptitle("Weak light: s=1e-6, 5000 finite trajectories")
    for values, style, label in (
        (chi_trajectory, {"color": "tab:green", "linewidth": 2.5}, "finite trajectory"),
        (chi_elecsus, {"color": "tab:blue", "linewidth": 1.8}, "ElecSus"),
        (chi_linear, {"color": "tab:orange", "linestyle": "--", "linewidth": 1.6}, "linear thermal OBE"),
    ):
        axes[0].plot(detuning_axis, values.imag, label=label, **style)
        axes[1].plot(detuning_axis, values.real, label=label, **style)

    axes[0].set_ylabel("Im response (absorption)")
    axes[1].set_ylabel("Re response (dispersion)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        detuning_axis,
        100.0 * np.abs(chi_trajectory - chi_elecsus) / scale,
        color="tab:green",
        label="|trajectory - ElecSus| / max|ElecSus|",
    )
    axes[2].plot(
        detuning_axis,
        100.0 * np.abs(chi_linear - chi_elecsus) / scale,
        "--",
        color="tab:orange",
        label="|linear OBE - ElecSus| / max|ElecSus|",
    )
    axes[2].set_xlabel("detuning (MHz, ElecSus external axis)")
    axes[2].set_ylabel("difference (%)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    figure.tight_layout()

    figure_path = output_dir / "weak_light_detuning_spectrum.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print(f"saved: {data_path}")
    print(f"saved: {figure_path}")
    print("PASS：弱光有限轨迹谱回到线性 OBE / ElecSus 基准")


if __name__ == "__main__":
    main()
