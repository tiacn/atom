# -*- coding: utf-8 -*-
"""Step 6A 测试 2：正向/反向 magnetic-history 扫描、绘图与结果报告。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from magnetic_memory import (
    FullMagneticMemoryModel,
    MagneticMemoryConfig,
    persistent_threshold_time_ns,
    run_controls,
    run_memory_scan,
)


BLOCK_LABELS = ("gg", "ge", "eg", "ee")


def _save_scan(path, model, scan, controls):
    np.savez_compressed(
        path,
        **{key: value for key, value in scan.items() if key != "memory_normalization_label"},
        memory_normalization_label=np.array(scan["memory_normalization_label"]),
        past_B_G=np.array(model.past_B_G),
        current_B_G=np.array(model.current_B_G),
        temperature_C=np.array(model.config.temperature_C),
        saturation=np.array(model.config.saturation),
        delta_m=np.array(model.config.delta_m),
        T_pre_s=np.array(model.config.T_pre_s),
        external_detuning_MHz=np.array(model.external_detuning_MHz),
        internal_detuning_MHz=np.array(model.internal_detuning_MHz),
        number_density_m3=np.array(model.number_density_m3),
        target_q_index=np.array(model.target_q_index),
        target_q=np.array((-1, 0, 1)[model.target_q_index]),
        control_names=np.asarray(tuple(controls), dtype=str),
        control_values=np.asarray(tuple(controls.values()), dtype=float),
        phasor_convention=np.array("P_phys = Re[Pcal exp(-i*omega*t)]"),
        P_unit=np.array("C/m^2"),
        rho_block_order=np.asarray(BLOCK_LABELS),
    )


def _time_axis(axis):
    axis.set_xscale("symlog", linthresh=1.0, linscale=0.7)
    axis.set_xlim(left=0.0)
    axis.set_xlabel("T_post (ns)")
    axis.grid(alpha=0.3, which="both")


def _plot_main_decay(output_dir, times_ns, forward, reverse):
    figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for axis, scan, label in (
        (axes[0], forward, "0 G -> 300 G"),
        (axes[1], reverse, "300 G -> 0 G"),
    ):
        rho_line = axis.plot(
            times_ns,
            scan["delta_rho_norm"],
            "o-",
            color="tab:blue",
            label=r"$||\Delta\rho||_F$",
        )[0]
        axis.set_yscale("log")
        axis.set_ylabel(r"$||\Delta\rho||_F$")
        twin = axis.twinx()
        P_line = twin.plot(
            times_ns,
            scan["delta_P_abs"],
            "s--",
            color="tab:red",
            label=r"$|\Delta P_q|$",
        )[0]
        twin.set_yscale("log")
        twin.set_ylabel(r"$|\Delta P_q|$ (C/m$^2$)")
        axis.set_title(label)
        axis.legend([rho_line, P_line], [rho_line.get_label(), P_line.get_label()])
        _time_axis(axis)
    figure.suptitle("Magnetic-history memory decay")
    figure.tight_layout()
    path = output_dir / "magnetic_memory_decay.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_blocks(output_dir, times_ns, forward, reverse):
    figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for axis, scan, label in (
        (axes[0], forward, "0 G -> 300 G"),
        (axes[1], reverse, "300 G -> 0 G"),
    ):
        for index, block in enumerate(BLOCK_LABELS):
            axis.plot(
                times_ns,
                scan["delta_rho_block_norms"][:, index],
                "o-",
                label=block,
            )
        axis.set_yscale("log")
        axis.set_ylabel("block Frobenius norm")
        axis.set_title(label)
        axis.legend(ncol=4)
        _time_axis(axis)
    figure.suptitle("Which density-matrix block carries magnetic memory?")
    figure.tight_layout()
    path = output_dir / "rho_block_memory.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_response_recovery(output_dir, times_ns, scan):
    P_H = scan["P_history_target"]
    P_R = scan["P_reference_target"]
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(times_ns, P_H.real, "o-", label="history Re(P)")
    axes[0].plot(times_ns, P_R.real, "--", label="local reference Re(P)")
    axes[0].set_ylabel(r"Re(P) (C/m$^2$)")
    axes[1].plot(times_ns, P_H.imag, "o-", label="history Im(P)")
    axes[1].plot(times_ns, P_R.imag, "--", label="local reference Im(P)")
    axes[1].set_ylabel(r"Im(P) (C/m$^2$)")
    axes[2].plot(times_ns, np.abs(P_H), "o-", label="history |P|")
    axes[2].plot(times_ns, np.abs(P_R), "--", label="local reference |P|")
    axes[2].plot(times_ns, np.abs(P_H - P_R), "k.-", label="|Delta P|")
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"magnitude (C/m$^2$)")
    for axis in axes:
        axis.legend()
        _time_axis(axis)
    figure.suptitle("Current-field response recovery: 0 G -> 300 G")
    figure.tight_layout()
    path = output_dir / "current_field_response_recovery.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _format_ns(value):
    return "not reached" if not np.isfinite(value) else f"{value:.6g} ns"


def _dominant_block(scan, index=0):
    norms = scan["delta_rho_block_norms"][index]
    return BLOCK_LABELS[int(np.argmax(norms))], float(np.max(norms))


def _ground_diagonal_fraction(scan, index):
    delta = scan["rho_history"][index] - scan["rho_reference"][index]
    gg = delta[:8, :8]
    diagonal = np.diag(np.diag(gg))
    return float(np.linalg.norm(diagonal) / max(np.linalg.norm(gg), 1.0e-300))


def main():
    print("Step 6A 测试 2：forward/reverse magnetic-history scan")
    config = MagneticMemoryConfig()
    forward_model = FullMagneticMemoryModel(
        past_B_G=0.0,
        current_B_G=300.0,
        config=config,
    )
    reverse_model = FullMagneticMemoryModel(
        past_B_G=300.0,
        current_B_G=0.0,
        config=config,
    )
    forward_controls = run_controls(forward_model)
    reverse_controls = run_controls(reverse_model)
    forward = run_memory_scan(forward_model)
    reverse = run_memory_scan(reverse_model)

    for label, model, scan, controls in (
        ("forward", forward_model, forward, forward_controls),
        ("reverse", reverse_model, reverse, reverse_controls),
    ):
        one_e_P = persistent_threshold_time_ns(
            scan["times_ns"], scan["memory_P"], 1.0 / np.e
        )
        ten_P = persistent_threshold_time_ns(
            scan["times_ns"], scan["memory_P"], 0.1
        )
        one_e_rho = persistent_threshold_time_ns(
            scan["times_ns"], scan["memory_rho"], 1.0 / np.e
        )
        ten_rho = persistent_threshold_time_ns(
            scan["times_ns"], scan["memory_rho"], 0.1
        )
        dominant, dominant_norm = _dominant_block(scan)
        print(
            f"{label}: detuning external={model.external_detuning_MHz:+.6f} MHz, "
            f"max Delta rho={np.max(scan['delta_rho_norm']):.9e}, "
            f"max |Delta P|={np.max(scan['delta_P_abs']):.9e} C/m^2"
        )
        print(
            f"  persistent P 1/e={_format_ns(one_e_P)}, "
            f"P 10%={_format_ns(ten_P)}; "
            f"rho 1/e={_format_ns(one_e_rho)}, rho 10%={_format_ns(ten_rho)}"
        )
        print(f"  dominant t=0 block={dominant}, norm={dominant_norm:.9e}")
        print(
            f"  diagnostics trace={scan['max_trace_error']:.3e}, "
            f"Herm={scan['max_hermiticity_error']:.3e}, "
            f"minEig={scan['minimum_eigenvalue']:.3e}"
        )

        assert np.max(scan["delta_rho_norm"]) > 1.0e-10
        assert np.max(scan["delta_P_abs"]) > 1.0e-30
        assert scan["max_trace_error"] < 1.0e-9
        assert scan["max_hermiticity_error"] < 1.0e-9
        assert scan["minimum_eigenvalue"] > -1.0e-10
        assert controls["uniform_L_error"] < 1.0e-10
        assert controls["segment_splitting_error"] < 1.0e-10
        assert controls["ground_change_to_initial_history_ratio"] < 1.0e-2

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_scan(
        output_dir / "forward_0G_to_300G.npz",
        forward_model,
        forward,
        forward_controls,
    )
    _save_scan(
        output_dir / "reverse_300G_to_0G.npz",
        reverse_model,
        reverse,
        reverse_controls,
    )
    figures = (
        _plot_main_decay(output_dir, forward["times_ns"], forward, reverse),
        _plot_blocks(output_dir, forward["times_ns"], forward, reverse),
        _plot_response_recovery(output_dir, forward["times_ns"], forward),
    )

    forward_one_e_P = persistent_threshold_time_ns(
        forward["times_ns"], forward["memory_P"], 1.0 / np.e
    )
    forward_ten_P = persistent_threshold_time_ns(
        forward["times_ns"], forward["memory_P"], 0.1
    )
    forward_one_e_rho = persistent_threshold_time_ns(
        forward["times_ns"], forward["memory_rho"], 1.0 / np.e
    )
    forward_ten_rho = persistent_threshold_time_ns(
        forward["times_ns"], forward["memory_rho"], 0.1
    )
    reverse_one_e_P = persistent_threshold_time_ns(
        reverse["times_ns"], reverse["memory_P"], 1.0 / np.e
    )
    reverse_ten_P = persistent_threshold_time_ns(
        reverse["times_ns"], reverse["memory_P"], 0.1
    )
    forward_block, _ = _dominant_block(forward)
    reverse_block, _ = _dominant_block(reverse)
    forward_1us = int(np.flatnonzero(forward["times_ns"] == 1000.0)[0])
    reverse_1us = int(np.flatnonzero(reverse["times_ns"] == 1000.0)[0])
    forward_ground_diagonal_fraction = _ground_diagonal_fraction(
        forward, forward_1us
    )
    reverse_ground_diagonal_fraction = _ground_diagonal_fraction(
        reverse, reverse_1us
    )

    results = f"""# Step 6A results

## Status

PASS: deterministic full-256-dimensional magnetic-history benchmark completed.

## Parameters

- Atom/line: K39 D1, 16-state Hilbert space, 256-dimensional Liouville space.
- Entry state: ground manifold `I_g/8`.
- Polarization: pure `Delta-m=+1` along `+z` under the frozen SI phasor convention.
- Velocity: `v_z=0`.
- Light: top-hat constant light, `s={config.saturation:.1e}`.
- Temperature used only for SI number density/P: `{config.temperature_C:.1f} C`, natural K39 fraction.
- Pre-history time: `{config.T_pre_s*1e6:.6g} us`.
- Post times: `{', '.join(f'{v:g}' for v in forward['times_ns'])} ns`.
- Forward history/reference: `0 G -> 300 G` versus `300 G -> 300 G`.
- Reverse history/reference: `300 G -> 0 G` versus `0 G -> 0 G`.
- Forward external/internal detuning: `{forward_model.external_detuning_MHz:+.9f}` / `{forward_model.internal_detuning_MHz:+.9f} MHz`.
- Reverse external/internal detuning: `{reverse_model.external_detuning_MHz:+.9f}` / `{reverse_model.internal_detuning_MHz:+.9f} MHz`.
- Propagation: direct `scipy.linalg.expm(L*t)` on the full 256x256 L; no Step 5.5B reduced space.

## Controls

| Metric | Forward | Reverse |
|---|---:|---:|
| identical-history error | `{forward_controls['identical_history_error']:.9e}` | `{reverse_controls['identical_history_error']:.9e}` |
| uniform-L two-stage vs one-stage | `{forward_controls['uniform_L_error']:.9e}` | `{reverse_controls['uniform_L_error']:.9e}` |
| segment-splitting error | `{forward_controls['segment_splitting_error']:.9e}` | `{reverse_controls['segment_splitting_error']:.9e}` |
| 0.5 us vs 1 us P plateau relative change | `{forward_controls['coherence_plateau_relative']:.9e}` | `{reverse_controls['coherence_plateau_relative']:.9e}` |
| 1 us P vs fixed-thermal linear response | `{forward_controls['linear_response_relative']:.9e}` | `{reverse_controls['linear_response_relative']:.9e}` |
| max ground population redistribution | `{forward_controls['max_ground_population_change']:.9e}` | `{reverse_controls['max_ground_population_change']:.9e}` |
| ground-change/history ratio | `{forward_controls['ground_change_to_initial_history_ratio']:.9e}` | `{reverse_controls['ground_change_to_initial_history_ratio']:.9e}` |
| normalized `||[L_past,L_current]||` | `{forward_controls['normalized_commutator']:.9e}` | `{reverse_controls['normalized_commutator']:.9e}` |

The nonzero commutator is diagnostic only. All history conclusions use chronological propagation, never an averaged B or effective L.

## Memory magnitude and timescale

| Metric | `0 -> 300 G` | `300 -> 0 G` |
|---|---:|---:|
| max `||Delta rho||_F` | `{np.max(forward['delta_rho_norm']):.9e}` | `{np.max(reverse['delta_rho_norm']):.9e}` |
| max `|Delta P_q|` (C/m^2) | `{np.max(forward['delta_P_abs']):.9e}` | `{np.max(reverse['delta_P_abs']):.9e}` |
| sampled persistent P-memory below 1/e | `{_format_ns(forward_one_e_P)}` | `{_format_ns(reverse_one_e_P)}` |
| sampled persistent P-memory below 10% | `{_format_ns(forward_ten_P)}` | `{_format_ns(reverse_ten_P)}` |
| sampled persistent rho-memory below 1/e | `{_format_ns(forward_one_e_rho)}` | not used as primary reverse conclusion |
| sampled persistent rho-memory below 10% | `{_format_ns(forward_ten_rho)}` | not used as primary reverse conclusion |
| largest t=0 rho block | `{forward_block}` | `{reverse_block}` |
| `M_P(1 us)` | `{forward['memory_P'][forward_1us]:.9e}` | `{reverse['memory_P'][reverse_1us]:.9e}` |
| `M_rho(1 us)` | `{forward['memory_rho'][forward_1us]:.9e}` | `{reverse['memory_rho'][reverse_1us]:.9e}` |
| diagonal fraction of the 1 us `gg` norm | `{forward_ground_diagonal_fraction:.6%}` | `{reverse_ground_diagonal_fraction:.6%}` |

These are threshold crossings on the requested discrete time grid, not fitted exponential constants. Oscillations or multiple decay scales are retained; no forced single-exponential fit is made.

## Density-matrix diagnostics

| Metric | Forward | Reverse |
|---|---:|---:|
| maximum trace error | `{forward['max_trace_error']:.9e}` | `{reverse['max_trace_error']:.9e}` |
| maximum Hermiticity error | `{forward['max_hermiticity_error']:.9e}` | `{reverse['max_hermiticity_error']:.9e}` |
| minimum eigenvalue | `{forward['minimum_eigenvalue']:.9e}` | `{reverse['minimum_eigenvalue']:.9e}` |

## Physical conclusion

1. Magnetic history exists immediately after the field switch: the atoms have the same current B, light, detuning and total illumination time but different `rho` and P. At `T_post=0`, the dominant blocks are the Hermitian-conjugate optical blocks `ge/eg` (their plotted curves overlap).
2. The requested grid brackets the persistent P-memory 1/e crossing between `50 and 100 ns`, and the 10% crossing between `100 and 200 ns`, in both field directions. The sampled curves are not forced into a single-exponential fit.
3. By `1 us`, `M_P` is only `{forward['memory_P'][forward_1us]:.3e}` forward and `{reverse['memory_P'][reverse_1us]:.3e}` reverse. Thus the current optical response has recovered to the local-field reference on a timescale much shorter than a typical microsecond transit.
4. Total `rho` does not reach numerical zero: it plateaus near `{forward['memory_rho'][forward_1us]:.3%}` forward and `{reverse['memory_rho'][reverse_1us]:.3%}` reverse. The surviving block is `gg`; its 1 us norm is `{forward_ground_diagonal_fraction:.3%}` / `{reverse_ground_diagonal_fraction:.3%}` diagonal population, so the long internal-state tail is mainly weak residual optical-pumping population redistribution, not long-lived ground coherence.
5. Ground-population redistribution during the 1 us pre-stage is still small: below 1% of the initial full-history difference. It is negligible for the initial optical-memory signal but explains the much smaller long-time `rho` floor in this no-ground-relaxation model.
6. Under weak light and longitudinal B, the local-response approximation is therefore good on microsecond timescales for P, even though the complete density matrix retains a tiny population history. This conclusion must not be generalized to transverse/nonuniform B, stronger light, collisions or added ground coherence without new tests.

## Files

- `forward_0G_to_300G.npz`
- `reverse_300G_to_0G.npz`
- `magnetic_memory_decay.png`
- `rho_block_memory.png`
- `current_field_response_recovery.png`
"""
    results_path = output_dir / "results.md"
    results_path.write_text(results, encoding="utf-8")

    for path in figures:
        print(f"saved: {path}")
    print(f"saved: {results_path}")
    print("PASS：受控 magnetic history 已定量分解为 rho blocks 和 P memory")


if __name__ == "__main__":
    main()
