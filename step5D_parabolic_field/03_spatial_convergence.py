# -*- coding: utf-8 -*-
"""Step 5D 测试 3：抛物线 B(z) 的空间离散与 DeltaB->0 control。"""

import json
from pathlib import Path

import numpy as np

from parabolic_history import (
    FullStateSplitPropagator,
    ParabolicFieldConfig,
    profile_metrics,
)


def interpolate_complex(x_new, x, values):
    return np.interp(x_new, x, values.real) + 1.0j * np.interp(
        x_new, x, values.imag
    )


def main():
    print("Step 5D 测试 3：spatial convergence + DeltaB->0")
    config = ParabolicFieldConfig()
    model = FullStateSplitPropagator(config)
    dz_values_um = np.array([10.0, 5.0, 2.5, 1.25, 0.625])
    profiles = []
    metrics = []
    for dz_um in dz_values_um:
        profile = model.trajectory(+300.0, dz_um * 1.0e-6)
        profile_metric = profile_metrics(profile, config.tau_memory_s)
        profiles.append(profile)
        metrics.append(profile_metric)
        print(
            f"dz={dz_um:5.1f} um, points={len(profile.z_m)}, "
            f"bulk global={profile_metric['epsilon_global_bulk']:.6e}, "
            f"bulk max={profile_metric['max_epsilon_bulk']:.6e}, "
            f"substeps={profile.internal_substeps}"
        )

    reference = profiles[-1]
    profile_errors = []
    shared_node_errors = []
    final_errors = []
    metric_errors = []
    reference_scale = np.linalg.norm(reference.P_local)
    for profile, metric in zip(profiles, metrics):
        coarse_order = np.argsort(profile.z_m)
        reference_order = np.argsort(reference.z_m)
        interpolated = interpolate_complex(
            reference.z_m[reference_order],
            profile.z_m[coarse_order],
            profile.P_trajectory[coarse_order],
        )
        error = float(
            np.linalg.norm(interpolated - reference.P_trajectory[reference_order])
            / reference_scale
        )
        reference_step = abs(reference.z_m[1] - reference.z_m[0])
        shared_indices = np.rint(
            (profile.z_m - reference.z_m[0]) / reference_step
        ).astype(int)
        if not np.allclose(
            reference.z_m[shared_indices], profile.z_m, atol=1.0e-13, rtol=0.0
        ):
            raise AssertionError("convergence grids are not nested")
        shared_scale = np.linalg.norm(reference.P_local[shared_indices])
        shared_error = float(
            np.linalg.norm(
                profile.P_trajectory - reference.P_trajectory[shared_indices]
            )
            / shared_scale
        )
        final_error = float(
            abs(profile.P_trajectory[-1] - reference.P_trajectory[-1])
            / np.max(np.abs(reference.P_local))
        )
        metric_error = float(
            abs(
                metric["epsilon_global_bulk"]
                - metrics[-1]["epsilon_global_bulk"]
            )
        )
        profile_errors.append(error)
        shared_node_errors.append(shared_error)
        final_errors.append(final_error)
        metric_errors.append(metric_error)
        print(
            f"  vs {dz_values_um[-1]:g} um: "
            f"interpolated={error:.6e}, shared={shared_error:.6e}, "
            f"final={final_error:.6e}, "
            f"bulk-global metric={metric_error:.6e}"
        )

    constant = model.trajectory(+300.0, 5.0e-6, delta_B_G=0.0)
    constant_metrics = profile_metrics(constant, config.tau_memory_s)
    print(
        "DeltaB=0: raw max="
        f"{constant_metrics['max_epsilon_raw']:.6e} at "
        f"{constant_metrics['max_epsilon_raw_z_mm']:+.6f} mm, "
        f"bulk global={constant_metrics['epsilon_global_bulk']:.6e}, "
        f"bulk max={constant_metrics['max_epsilon_bulk']:.6e}"
    )

    # The shared-node error tests propagation convergence.  The interpolated
    # curve error is retained as a separate output-resolution diagnostic: the
    # complex optical response can oscillate between otherwise converged nodes.
    assert shared_node_errors[-2] < 5.0e-7
    assert final_errors[-2] < 5.0e-5
    assert metric_errors[-2] < 5.0e-5
    # raw max 保留 rho_entry 无 optical coherence 的物理入口瞬态；
    # flat-field locality 只在排除 5*tau_memory 后判断。
    assert constant_metrics["max_epsilon_raw"] > 0.5
    assert constant_metrics["epsilon_global_bulk"] < 5.0e-3
    assert constant_metrics["max_epsilon_bulk"] < 5.0e-3
    assert reference.max_trace_error < 1.0e-8
    assert reference.max_hermiticity_error < 1.0e-8
    assert reference.minimum_eigenvalue > -1.0e-9

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "spatial_convergence.npz",
        dz_um=dz_values_um,
        profile_error_vs_finest=np.asarray(profile_errors),
        shared_node_error_vs_finest=np.asarray(shared_node_errors),
        final_error_vs_finest=np.asarray(final_errors),
        bulk_global_metric_error_vs_finest=np.asarray(metric_errors),
        epsilon_global_bulk=np.asarray(
            [metric["epsilon_global_bulk"] for metric in metrics]
        ),
        max_epsilon_bulk=np.asarray(
            [metric["max_epsilon_bulk"] for metric in metrics]
        ),
        finest_z_m=reference.z_m,
        finest_B_G=reference.B_G,
        finest_P_trajectory_q=reference.P_trajectory_q,
        finest_P_local_q=reference.P_local_q,
        finest_epsilon_position=reference.epsilon_position,
        finest_rho_final=reference.rho_final,
        constant_z_m=constant.z_m,
        constant_P_trajectory_q=constant.P_trajectory_q,
        constant_P_local_q=constant.P_local_q,
        constant_epsilon_position=constant.epsilon_position,
    )
    payload = {
        "velocity_m_s": 300.0,
        "dz_um": dz_values_um.tolist(),
        "profile_error_vs_finest": profile_errors,
        "shared_node_error_vs_finest": shared_node_errors,
        "final_error_vs_finest": final_errors,
        "bulk_global_metric_error_vs_finest": metric_errors,
        "metrics": metrics,
        "deltaB_zero_metrics": constant_metrics,
        "max_internal_dt_ns": config.max_internal_dt_s * 1.0e9,
    }
    (output_dir / "spatial_convergence.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("PASS：0.625 um 输出网格收敛，DeltaB->0 bulk 回到 local")


if __name__ == "__main__":
    main()
