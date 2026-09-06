# -*- coding: utf-8 -*-
"""Step 6B 测试 2：full-state split propagator 与 dense full-256 expm。"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

import path_setup  # noqa: F401

from liouvillian import rho_to_vec, vec_to_rho
from parabolic_history import FullStateSplitPropagator, ParabolicFieldConfig


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-300))


def main():
    print("Step 6B 测试 2：full-state propagator controls")
    model = FullStateSplitPropagator(ParabolicFieldConfig())
    assert model.N**2 == 256
    assert model.decay_completeness_error < 1.0e-12

    rng = np.random.default_rng(6602)
    A = rng.normal(size=(model.N, model.N)) + 1.0j * rng.normal(
        size=(model.N, model.N)
    )
    rho_random = A @ A.conj().T
    rho_random /= np.trace(rho_random)
    decay_dt = 37.0e-9
    decay_direct = vec_to_rho(
        expm(model.engine.L_decay * decay_dt) @ rho_to_vec(rho_random),
        model.N,
    )
    decay_analytic = model._decay_step(rho_random, decay_dt)
    decay_error = relative(decay_analytic, decay_direct)

    # 先建立带 optical coherence 的代表态，避免只在 rho_entry 上做过弱测试。
    rho_probe = model.dense_step(
        model.rho_entry,
        B_z_G=2250.0,
        velocity_m_s=300.0,
        duration_s=1.0e-6,
    )
    step_errors = {}
    for dt_ns in (10.0, 5.0, 2.5, 1.25):
        dense = model.dense_step(rho_probe, 2250.0, 300.0, dt_ns * 1.0e-9)
        split = model.fourth_order_step(
            rho_probe,
            model.hamiltonian(2250.0, 300.0),
            dt_ns * 1.0e-9,
        )
        error = relative(split, dense)
        step_errors[dt_ns] = error
        print(f"single step {dt_ns:4.1f} ns relative rho error = {error:.6e}")

    # constant-B path: split many full-state steps vs one dense full-256 exponential.
    total_time = 25.0e-3 / 300.0
    rho_split, substeps = model.field_segment(
        model.rho_entry,
        B_z_G=2250.0,
        velocity_m_s=300.0,
        duration_s=total_time,
        max_dt_s=1.25e-9,
    )
    rho_dense = model.dense_step(
        model.rho_entry,
        B_z_G=2250.0,
        velocity_m_s=300.0,
        duration_s=total_time,
    )
    uniform_rho_error = relative(rho_split, rho_dense)
    P_split = model.polarization_q(rho_split)[model.target_q_index]
    P_dense = model.polarization_q(rho_dense)[model.target_q_index]
    uniform_P_error = float(abs(P_split - P_dense) / max(abs(P_dense), 1.0e-300))

    print(f"decay analytic/full-256 error = {decay_error:.6e}")
    print(f"constant-B substeps = {substeps}")
    print(f"constant-B rho relative error = {uniform_rho_error:.6e}")
    print(f"constant-B P relative error = {uniform_P_error:.6e}")
    assert decay_error < 1.0e-12
    assert step_errors[1.25] < 2.0e-9
    assert step_errors[1.25] < step_errors[2.5]
    assert uniform_rho_error < 5.0e-9
    assert uniform_P_error < 1.0e-4

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "full_liouville_dimension": 256,
        "reduced_space_used": False,
        "decay_completeness_error": model.decay_completeness_error,
        "decay_analytic_vs_full256_error": decay_error,
        "single_step_rho_relative_error": {
            str(key): value for key, value in step_errors.items()
        },
        "constant_B_total_time_us": total_time * 1.0e6,
        "constant_B_internal_substeps": substeps,
        "constant_B_rho_relative_error": uniform_rho_error,
        "constant_B_P_relative_error": uniform_P_error,
        "max_internal_dt_ns": 1.25,
    }
    (output_dir / "propagator_controls.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("PASS：full-state split propagation 回归 dense full-256 expm")


if __name__ == "__main__":
    main()
