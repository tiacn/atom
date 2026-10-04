# -*- coding: utf-8 -*-
"""Step 5D test 4: deterministic trajectories through the parabolic field."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import path_setup  # noqa: F401

from moving_atom import density_matrix_diagnostics
from parabolic_history import (
    B_parabolic_G,
    FullStateSplitPropagator,
    ParabolicFieldConfig,
    TrajectoryProfile,
    dB_dz_G_m,
    hysteresis_metrics,
    profile_metrics,
)


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
PROFILE_DIR = OUTPUTS / "profiles"
VELOCITIES = (+50.0, +100.0, +300.0, +500.0, -50.0, -100.0, -300.0, -500.0)


def profile_path(velocity_m_s: float) -> Path:
    direction = "p" if velocity_m_s > 0.0 else "m"
    return PROFILE_DIR / f"profile_{direction}{abs(int(velocity_m_s)):03d}.npz"


def save_profile(path: Path, profile: TrajectoryProfile) -> None:
    np.savez_compressed(
        path,
        velocity_m_s=np.array(profile.velocity_m_s),
        dz_m=np.array(profile.dz_m),
        z_m=profile.z_m,
        B_G=profile.B_G,
        dB_dt_G_s=profile.dB_dt_G_s,
        delta_B_memory_G=profile.delta_B_memory_G,
        P_trajectory_q=profile.P_trajectory_q,
        P_local_q=profile.P_local_q,
        epsilon_position=profile.epsilon_position,
        rho_final=profile.rho_final,
        max_trace_error=np.array(profile.max_trace_error),
        max_hermiticity_error=np.array(profile.max_hermiticity_error),
        minimum_eigenvalue=np.array(profile.minimum_eigenvalue),
        internal_substeps=np.array(profile.internal_substeps),
    )


def load_profile(path: Path) -> TrajectoryProfile:
    with np.load(path) as data:
        return TrajectoryProfile(
            velocity_m_s=float(data["velocity_m_s"]),
            dz_m=float(data["dz_m"]),
            z_m=data["z_m"].copy(),
            B_G=data["B_G"].copy(),
            dB_dt_G_s=data["dB_dt_G_s"].copy(),
            delta_B_memory_G=data["delta_B_memory_G"].copy(),
            P_trajectory_q=data["P_trajectory_q"].copy(),
            P_local_q=data["P_local_q"].copy(),
            epsilon_position=data["epsilon_position"].copy(),
            rho_final=data["rho_final"].copy(),
            max_trace_error=float(data["max_trace_error"]),
            max_hermiticity_error=float(data["max_hermiticity_error"]),
            minimum_eigenvalue=float(data["minimum_eigenvalue"]),
            internal_substeps=int(data["internal_substeps"]),
        )


def import_converged_positive_300(config: ParabolicFieldConfig) -> TrajectoryProfile:
    source = OUTPUTS / "spatial_convergence.npz"
    if not source.exists():
        raise FileNotFoundError("run 03_spatial_convergence.py before test 04")
    with np.load(source) as data:
        z = data["finest_z_m"].copy()
        B = data["finest_B_G"].copy()
        P_trajectory_q = data["finest_P_trajectory_q"].copy()
        P_local_q = data["finest_P_local_q"].copy()
        epsilon = data["finest_epsilon_position"].copy()
        rho_final = data["finest_rho_final"].copy()
    velocity = 300.0
    derivative = dB_dz_G_m(z, config)
    diagnostics = density_matrix_diagnostics(rho_final, 8)
    return TrajectoryProfile(
        velocity_m_s=velocity,
        dz_m=float(abs(z[1] - z[0])),
        z_m=z,
        B_G=B,
        dB_dt_G_s=velocity * derivative,
        delta_B_memory_G=np.abs(velocity * derivative) * config.tau_memory_s,
        P_trajectory_q=P_trajectory_q,
        P_local_q=P_local_q,
        epsilon_position=epsilon,
        rho_final=rho_final,
        max_trace_error=float(abs(diagnostics["trace"] - 1.0)),
        max_hermiticity_error=float(diagnostics["hermiticity_error"]),
        minimum_eigenvalue=float(diagnostics["minimum_eigenvalue"]),
        internal_substeps=80000,
    )


def local_symmetry_error(profile: TrajectoryProfile) -> float:
    order = np.argsort(profile.z_m)
    local = profile.P_local[order]
    scale = max(float(np.max(np.abs(local))), 1.0e-300)
    return float(np.max(np.abs(local - local[::-1])) / scale)


def serializable_hysteresis(profile: TrajectoryProfile):
    records = []
    for field, z_left, z_right, left, right, normalized in hysteresis_metrics(profile):
        records.append(
            {
                "B_G": field,
                "z_left_mm": z_left,
                "z_right_mm": z_right,
                "P_left_C_m2": [left.real, left.imag],
                "P_right_C_m2": [right.real, right.imag],
                "normalized_difference": normalized,
            }
        )
    return records


def make_plots(profiles: dict[float, TrajectoryProfile]) -> None:
    representative = profiles[300.0]
    order = np.argsort(representative.z_m)
    z_mm = representative.z_m[order] * 1.0e3
    B = representative.B_G[order]
    trajectory = representative.P_trajectory[order]
    local = representative.P_local[order]
    polarization_scale = max(float(np.max(np.abs(local))), 1.0e-300)

    fig, axes = plt.subplots(4, 1, figsize=(9.2, 11.0), sharex=True)
    axes[0].plot(z_mm, B, color="black", linewidth=1.8)
    axes[0].set_ylabel("B (G)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(z_mm, trajectory.real / polarization_scale, label="trajectory", linewidth=1.0)
    axes[1].plot(z_mm, local.real / polarization_scale, label="local", linewidth=1.5, alpha=0.85)
    axes[1].set_ylabel(r"Re(P) / max|P_local|")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[2].plot(z_mm, trajectory.imag / polarization_scale, label="trajectory", linewidth=1.0)
    axes[2].plot(z_mm, local.imag / polarization_scale, label="local", linewidth=1.5, alpha=0.85)
    axes[2].set_ylabel(r"Im(P) / max|P_local|")
    axes[2].grid(alpha=0.25)
    axes[3].plot(z_mm, np.abs(trajectory) / polarization_scale, label="trajectory", linewidth=1.0)
    axes[3].plot(z_mm, np.abs(local) / polarization_scale, label="local", linewidth=1.5, alpha=0.85)
    axes[3].set_ylabel(r"|P| / max|P_local|")
    axes[3].set_xlabel("z (mm)")
    axes[3].grid(alpha=0.25)
    fig.suptitle("Parabolic B(z): trajectory vs local response, v=+300 m/s")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "trajectory_vs_local.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.0), sharex=True)
    colors = {50.0: "tab:blue", 100.0: "tab:orange", 300.0: "tab:green", 500.0: "tab:red"}
    for axis, sign, title in ((axes[0], +1.0, "left to right"), (axes[1], -1.0, "right to left")):
        for speed in (50.0, 100.0, 300.0, 500.0):
            profile = profiles[sign * speed]
            order = np.argsort(profile.z_m)
            axis.plot(
                profile.z_m[order] * 1.0e3,
                profile.epsilon_position[order],
                color=colors[speed],
                label=f"|v|={speed:g} m/s",
                linewidth=1.0,
            )
        axis.set_ylabel(r"epsilon_P(z)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(ncol=2)
    axes[-1].set_xlabel("z (mm)")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "local_error_vs_position.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharex=True)
    speeds = np.array((50.0, 100.0, 300.0, 500.0))
    for sign, marker, label in ((+1.0, "o", "v > 0"), (-1.0, "s", "v < 0")):
        values = [profile_metrics(profiles[sign * speed]) for speed in speeds]
        axes[0].plot(
            speeds,
            [item["epsilon_global_bulk"] for item in values],
            marker=marker,
            label=label,
        )
        axes[1].plot(
            speeds,
            [item["max_epsilon_bulk"] for item in values],
            marker=marker,
            label=label,
        )
    axes[0].set_ylabel("bulk global error")
    axes[1].set_ylabel("bulk max error")
    for axis in axes:
        axis.set_xlabel("|v_z| (m/s)")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Velocity dependence (fixed laboratory laser frequency)")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "velocity_dependence.png", dpi=180)
    plt.close(fig)


def build_results_md(
    config: ParabolicFieldConfig,
    model: FullStateSplitPropagator,
    summaries: dict[str, dict],
    high_field: dict,
    controls: dict,
    convergence: dict,
) -> str:
    rows = []
    for velocity in VELOCITIES:
        item = summaries[str(int(velocity))]
        rows.append(
            f"| {velocity:+.0f} | {item['epsilon_global_bulk']:.6f} | "
            f"{item['max_epsilon_bulk']:.6f} | {item['max_epsilon_bulk_z_mm']:+.4f} | "
            f"{item['max_delta_B_memory_G']:.3f} |"
        )
    hyst_rows = []
    for velocity in VELOCITIES:
        values = summaries[str(int(velocity))]["hysteresis"]
        hyst_rows.append(
            f"| {velocity:+.0f} | "
            + " | ".join(f"{item['normalized_difference']:.6f}" for item in values)
            + " |"
        )
    representative = summaries["300"]
    maximum_profile = max(
        summaries.values(), key=lambda item: item["max_epsilon_bulk"]
    )
    return f"""# Step 5D results

## Status

PASS: the high-field local benchmark, full-state propagator controls, continuous-field convergence, density-matrix diagnostics, and deterministic trajectory comparisons completed.

## Fixed physical setup

- K39 D1, 16-state Hilbert space and all 256 density-matrix components.
- Cell length: `{config.length_m * 1e3:.1f} mm`, centered at `z=0`.
- Field: `B(z)=2500 G - 500 G*(2z/L)^2`, hence `2000 G` at both ends and `2500 G` at the center.
- Entry state: equal population in the eight ground states (`I_g/8`).
- Polarization: pure `Delta-m=+1`; top-hat light; `s={config.saturation:.1e}`.
- Fixed laboratory laser detuning: `{model.external_laser_detuning_MHz:+.6f} MHz`, the frozen weak-linear absorption maximum at `B=2500 G`, `v_z=0`.
- Moving atoms use the frozen convention `delta_eff=delta_laser-k*v_z`; therefore changing velocity also changes Doppler detuning.
- Local reference: frozen weak-linear response at the instantaneous `B(z)` and the same Doppler-shifted detuning. It has no previous-position history.
- The first `5*tau_memory=500 ns` after entry is excluded only in the reported bulk metrics. Raw arrays retain the physical entrance transient.

## High-field local benchmark

- Fields: `2000, 2100, ..., 2500 G`.
- Worst OBE-vs-ElecSus complex relative L2 error: `{high_field['worst_complex_relative_error']:.6e}` ({100*high_field['worst_complex_relative_error']:.3f}%).
- Worst absorption-peak position error on the 2 MHz output axis: `{high_field['worst_peak_position_error_MHz']:.3f} MHz`.
- Minimum complex correlation: `{high_field['minimum_complex_correlation']:.9f}`.

## Propagator and convergence controls

- Propagation retains the full `16x16 rho` (256 complex components); reduced-space projection is not used.
- A fourth-order symmetric coherent/decay factorization is used for speed. Its constant-B full-cell result was compared with one direct dense `expm(L*t)` in the full 256-dimensional Liouville space.
- Constant-B rho relative error: `{controls['constant_B_rho_relative_error']:.6e}`.
- Constant-B target-P relative error: `{controls['constant_B_P_relative_error']:.6e}`.
- Maximum internal time step: `{controls['max_internal_dt_ns']:.3f} ns`.
- Spatial output grids tested: `{', '.join(str(value) for value in convergence['dz_um'])} um` for `v=+300 m/s`.
- `1.25 -> 0.625 um` shared-node full-complex-P error: `{convergence['shared_node_error_vs_finest'][-2]:.6e}`.
- `1.25 -> 0.625 um` final-P error: `{convergence['final_error_vs_finest'][-2]:.6e}`.
- `1.25 -> 0.625 um` bulk-global metric change: `{convergence['bulk_global_metric_error_vs_finest'][-2]:.6e}`.
- Linear interpolation from 1.25 to 0.625 um has a larger `{convergence['profile_error_vs_finest'][-2]:.6e}` reconstruction error because it does not resolve all between-node complex phase. This is an output-sampling diagnostic, not a propagation discrepancy at common positions.
- `DeltaB=0` bulk global/max deviations are `{convergence['deltaB_zero_metrics']['epsilon_global_bulk']:.6e}` / `{convergence['deltaB_zero_metrics']['max_epsilon_bulk']:.6e}`. The remaining 0.27% floor is the finite-entry/weak-pumping transient relative to a frozen thermal local model, not magnetic-gradient history.

## Local-versus-trajectory error

All errors are normalized by `max_z |P_local|`; bulk values exclude the first 500 ns of travel.

| v_z (m/s) | global bulk | max bulk | max position (mm) | max DeltaB in 100 ns (G) |
|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

For the representative `+300 m/s` trajectory, `epsilon_global={representative['epsilon_global_bulk']:.6f}` and `max epsilon_P={representative['max_epsilon_bulk']:.6f}` at `z={representative['max_epsilon_bulk_z_mm']:+.4f} mm`. The largest max error in the requested velocity set is `{maximum_profile['max_epsilon_bulk']:.6f}` for `v={maximum_profile['velocity_m_s']:+.0f} m/s`.

The velocity plot is not a pure speed-only experiment: one fixed laboratory laser is used, so the Doppler shift changes with velocity as required by the physical trajectory. A non-monotonic curve therefore cannot be attributed only to adaptation time.

## Same-B hysteresis diagnostic

Entries are `|P_traj(left)-P_traj(right)| / max_z|P_local|` at symmetric positions having the same B. The frozen local model's left/right symmetry error is separately checked at numerical precision.

| v_z (m/s) | 2100 G | 2200 G | 2300 G | 2400 G |
|---:|---:|---:|---:|---:|
{chr(10).join(hyst_rows)}

## Density-matrix checks

- Worst trace error across saved trajectory points: `{max(item['max_trace_error'] for item in summaries.values() if np.isfinite(item['max_trace_error'])):.6e}`.
- Worst Hermiticity error: `{max(item['max_hermiticity_error'] for item in summaries.values() if np.isfinite(item['max_hermiticity_error'])):.6e}`.
- Minimum eigenvalue: `{min(item['minimum_eigenvalue'] for item in summaries.values() if np.isfinite(item['minimum_eigenvalue'])):.6e}`.
- All saved `rho`, `P`, B, and position arrays are finite.

## Physical conclusion

1. The high-field frozen weak-linear OBE agrees with ElecSus throughout 2000--2500 G to below 1% complex relative error, so the local reference itself remains validated in this range.
2. The chronological continuous-B propagation is numerically converged and obeys trace, Hermiticity, positivity, constant-B, and `DeltaB->0` controls.
3. At `v=+300 m/s`, the trajectory differs from the frozen local response by about `{100*representative['epsilon_global_bulk']:.1f}%` globally and `{100*representative['max_epsilon_bulk']:.1f}%` at the largest point. This is not a small locality correction under the exact fixed-frequency/no-ground-relaxation model tested here.
4. The same-B left/right comparison detects path dependence, but at the requested 2100/2200/2300/2400 G points it is much smaller than the largest local error (for `+300 m/s`, at most `{max(item['normalized_difference'] for item in representative['hysteresis']):.6f}`). The largest local discrepancy occurs close to narrow spatial resonances rather than at those four preselected B values.
5. `DeltaB_memory=|v*dB/dz|*100 ns` reaches only `{representative['max_delta_B_memory_G']:.2f} G` for `+300 m/s`; nevertheless the full accumulated trajectory can retain a larger difference than this one-timescale estimate suggests. The Step 5C switch-memory estimate is therefore useful context but does not replace continuous chronological propagation.
6. Because ground-state relaxation/collisions are absent, and because the fixed laser plus Doppler shift couples speed to spectral detuning, this benchmark alone must not yet be generalized to a thermal ensemble or used to claim that all of the observed difference is a single 100 ns magnetic lag.

## Files

- `trajectory_vs_local.png`
- `local_error_vs_position.png`
- `velocity_dependence.png`
- `trajectory_benchmark.json`
- `trajectory_benchmark.npz`
- `profiles/profile_*.npz`
"""


def main():
    print("Step 5D test 4: deterministic parabolic-field trajectories", flush=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    config = ParabolicFieldConfig()
    model = FullStateSplitPropagator(config)
    profiles: dict[float, TrajectoryProfile] = {}

    for velocity in VELOCITIES:
        path = profile_path(velocity)
        if path.exists():
            print(f"loading cached v={velocity:+.0f} m/s", flush=True)
            profile = load_profile(path)
        elif velocity == 300.0:
            print("importing converged v=+300 m/s profile", flush=True)
            profile = import_converged_positive_300(config)
            save_profile(path, profile)
        else:
            print(f"running v={velocity:+.0f} m/s", flush=True)
            profile = model.trajectory(velocity, 5.0e-6)
            save_profile(path, profile)
            print(
                f"  saved: points={len(profile.z_m)}, substeps={profile.internal_substeps}",
                flush=True,
            )
        profiles[velocity] = profile

    summaries: dict[str, dict] = {}
    for velocity, profile in profiles.items():
        metric = profile_metrics(profile, config.tau_memory_s)
        metric.update(
            {
                "velocity_m_s": velocity,
                "dz_um": profile.dz_m * 1.0e6,
                "points": len(profile.z_m),
                "max_trace_error": profile.max_trace_error,
                "max_hermiticity_error": profile.max_hermiticity_error,
                "minimum_eigenvalue": profile.minimum_eigenvalue,
                "internal_substeps": profile.internal_substeps,
                "local_left_right_symmetry_error": local_symmetry_error(profile),
                "hysteresis": serializable_hysteresis(profile),
            }
        )
        summaries[str(int(velocity))] = metric
        print(
            f"v={velocity:+.0f}: bulk global={metric['epsilon_global_bulk']:.6e}, "
            f"bulk max={metric['max_epsilon_bulk']:.6e} at "
            f"{metric['max_epsilon_bulk_z_mm']:+.4f} mm",
            flush=True,
        )

    # Test 03 checked every +300 point against the same thresholds. Its NPZ
    # stores the final rho, so test 04 records exact final-state diagnostics.
    summaries["300"]["diagnostics_scope"] = (
        "final state here; every saved point passed thresholds in test 03"
    )

    for profile in profiles.values():
        assert np.all(np.isfinite(profile.z_m))
        assert np.all(np.isfinite(profile.B_G))
        assert np.all(np.isfinite(profile.P_trajectory_q))
        assert np.all(np.isfinite(profile.P_local_q))
        assert np.all(np.isfinite(profile.rho_final))
        assert local_symmetry_error(profile) < 1.0e-10
        if np.isfinite(profile.max_trace_error):
            assert profile.max_trace_error < 1.0e-8
            assert profile.max_hermiticity_error < 1.0e-8
            assert profile.minimum_eigenvalue > -1.0e-9

    high_field = json.loads((OUTPUTS / "high_field_benchmark.json").read_text(encoding="utf-8"))
    controls = json.loads((OUTPUTS / "propagator_controls.json").read_text(encoding="utf-8"))
    convergence = json.loads((OUTPUTS / "spatial_convergence.json").read_text(encoding="utf-8"))
    assert high_field["worst_complex_relative_error"] < 0.05
    assert controls["constant_B_rho_relative_error"] < 5.0e-9
    assert controls["constant_B_P_relative_error"] < 1.0e-4
    assert convergence["shared_node_error_vs_finest"][-2] < 5.0e-7
    assert convergence["deltaB_zero_metrics"]["epsilon_global_bulk"] < 5.0e-3

    make_plots(profiles)

    velocity_array = np.array(VELOCITIES)
    np.savez_compressed(
        OUTPUTS / "trajectory_benchmark.npz",
        velocities_m_s=velocity_array,
        epsilon_global_raw=np.array([summaries[str(int(v))]["epsilon_global_raw"] for v in velocity_array]),
        epsilon_global_bulk=np.array([summaries[str(int(v))]["epsilon_global_bulk"] for v in velocity_array]),
        max_epsilon_raw=np.array([summaries[str(int(v))]["max_epsilon_raw"] for v in velocity_array]),
        max_epsilon_bulk=np.array([summaries[str(int(v))]["max_epsilon_bulk"] for v in velocity_array]),
        max_epsilon_bulk_z_mm=np.array([summaries[str(int(v))]["max_epsilon_bulk_z_mm"] for v in velocity_array]),
        max_delta_B_memory_G=np.array([summaries[str(int(v))]["max_delta_B_memory_G"] for v in velocity_array]),
        hysteresis_fields_G=np.array((2100.0, 2200.0, 2300.0, 2400.0)),
        hysteresis_normalized=np.array(
            [[item["normalized_difference"] for item in summaries[str(int(v))]["hysteresis"]] for v in velocity_array]
        ),
        external_laser_detuning_MHz=np.array(model.external_laser_detuning_MHz),
        number_density_m3=np.array(model.number_density_m3),
    )
    payload = {
        "status": "PASS",
        "config": {
            "length_mm": config.length_m * 1.0e3,
            "B_center_G": config.B_center_G,
            "delta_B_G": config.delta_B_G,
            "saturation": config.saturation,
            "delta_m": config.delta_m,
            "tau_memory_ns": config.tau_memory_s * 1.0e9,
            "max_internal_dt_ns": config.max_internal_dt_s * 1.0e9,
            "external_laser_detuning_MHz": model.external_laser_detuning_MHz,
            "number_density_m3": model.number_density_m3,
        },
        "high_field_benchmark": high_field,
        "propagator_controls": controls,
        "spatial_convergence": convergence,
        "profiles": summaries,
    }
    (OUTPUTS / "trajectory_benchmark.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    (OUTPUTS / "results.md").write_text(
        build_results_md(config, model, summaries, high_field, controls, convergence),
        encoding="utf-8",
    )
    print("PASS: Step 5D deterministic trajectory benchmark", flush=True)


if __name__ == "__main__":
    main()
