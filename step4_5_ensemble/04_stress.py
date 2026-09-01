# -*- coding: utf-8 -*-
"""测试 4：完整几何系综的长轨迹、最多 segments 和极慢原子。"""

import numpy as np

import path_setup  # noqa: F401

from config import CONFIG
from geometry import generate_ensemble
from moving_atom import (
    SingleAtomEngine,
    SliceFields,
    density_matrix_diagnostics,
    evolve_trajectory,
)


def strongly_nonuniform_fields():
    """构造明显非均匀的 B(z)，强制不同 segment 使用不同 L。"""
    z_fraction = (np.arange(CONFIG.n_slices) + 0.5) / CONFIG.n_slices
    B_z_G = (
        50.0
        + 2200.0 * z_fraction**2
        + 350.0 * np.sin(4.0 * np.pi * z_fraction)
    )
    return SliceFields(
        B_z_G=B_z_G,
        saturation=np.full(CONFIG.n_slices, 0.03),
        polarization_cart=np.tile(
            np.array([1.0, 0.0, 0.0], dtype=complex),
            (CONFIG.n_slices, 1),
        ),
        laser_detuning_MHz=np.full(CONFIG.n_slices, 40.0),
    )


def main():
    print("测试 4：长轨迹 stress test")
    trajectories, _, _ = generate_ensemble(CONFIG)
    flight_times = np.array([item.entry.time_s for item in trajectories])
    segment_counts = np.array([len(item.segments) for item in trajectories])
    speeds = np.array([np.linalg.norm(item.velocity_m_s) for item in trajectories])

    extreme_indices = (
        ("most_segments", int(np.argmax(segment_counts))),
        ("longest_flight", int(np.argmax(flight_times))),
        ("slowest_atom", int(np.argmin(speeds))),
    )

    print(f"N geometry atoms = {len(trajectories)}")
    print(f"maximum flight time = {flight_times.max()*1e6:.6f} us")
    print(f"maximum segments = {segment_counts.max()}")
    print(f"minimum speed = {speeds.min():.6f} m/s")

    engine = SingleAtomEngine()
    fields = strongly_nonuniform_fields()
    for label, atom_index in extreme_indices:
        trajectory = trajectories[atom_index]
        result = evolve_trajectory(trajectory, fields, engine)
        rho = result.rho_final
        diagnostics = density_matrix_diagnostics(rho, engine.N_G)
        expected_slices = trajectory.slice_indices.tolist()
        evolved_slices = [step.slice_index for step in result.steps]
        encountered_B = np.array([step.B_z_G for step in result.steps])
        print(
            f"{label}: atom={atom_index}, "
            f"flight={trajectory.entry.time_s*1e6:.6f} us, "
            f"segments={len(trajectory.segments)}, "
            f"distinct_B={len(np.unique(encountered_B))}, "
            f"speed={np.linalg.norm(trajectory.velocity_m_s):.6f} m/s, "
            f"trace_error={abs(diagnostics['trace']-1):.3e}, "
            f"herm={diagnostics['hermiticity_error']:.3e}, "
            f"min_eig={diagnostics['minimum_eigenvalue']:.3e}"
        )
        # 直接证明调用走完了 trajectory 中的每个 segment，而非一次总时间传播。
        assert len(result.steps) == len(trajectory.segments)
        assert evolved_slices == expected_slices
        assert np.all(np.isfinite(rho))
        assert abs(diagnostics["trace"] - 1.0) < 1e-8
        assert diagnostics["hermiticity_error"] < 1e-8
        assert diagnostics["minimum_eigenvalue"] > -1e-8

        if label == "most_segments":
            assert len(result.steps) == segment_counts.max()
            assert len(np.unique(encountered_B)) > 1
            assert np.ptp(encountered_B) > 100.0

    print(
        "PASS: 最多 segments、最长飞行和极慢原子均在非均匀 B(z) 下"
        "逐 segment 演化，密度矩阵不变量保持"
    )


if __name__ == "__main__":
    main()
