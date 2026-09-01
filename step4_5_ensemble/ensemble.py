# -*- coding: utf-8 -*-
"""N 原子独立演化和仅按当前位置进行的系综 binning。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import path_setup  # noqa: F401

from moving_atom import density_matrix_diagnostics, evolve_trajectory


@dataclass(frozen=True)
class EnsembleResult:
    rho_final_all: np.ndarray
    current_slices: np.ndarray
    counts: np.ndarray
    rho_bar: np.ndarray


def bin_current_density_matrices(rho_final_all, current_slices, n_slices):
    """每个原子只按 current_slice 贡献一次。"""
    rho_final_all = np.asarray(rho_final_all, dtype=complex)
    current_slices = np.asarray(current_slices, dtype=int)
    n_slices = int(n_slices)

    if rho_final_all.ndim != 3:
        raise ValueError("rho_final_all 必须为 (N_atoms,N,N)")
    if current_slices.shape != (len(rho_final_all),):
        raise ValueError("current_slices 长度必须等于原子数")
    if np.any((current_slices < 0) | (current_slices >= n_slices)):
        raise ValueError("存在越界 current_slice")

    N_states = rho_final_all.shape[1]
    counts = np.bincount(current_slices, minlength=n_slices)
    rho_sum = np.zeros((n_slices, N_states, N_states), dtype=complex)
    np.add.at(rho_sum, current_slices, rho_final_all)

    rho_bar = np.zeros_like(rho_sum)
    occupied = counts > 0
    rho_bar[occupied] = rho_sum[occupied] / counts[occupied, None, None]
    return counts, rho_bar


def evolve_ensemble(trajectories, fields, engine, progress_every=0):
    """逐原子调用已有 evolve_trajectory，只保存当前 rho_final。"""
    n_atoms = len(trajectories)
    rho_final_all = np.empty((n_atoms, engine.N, engine.N), dtype=complex)
    current_slices = np.empty(n_atoms, dtype=int)

    for atom_index, trajectory in enumerate(trajectories):
        single_result = evolve_trajectory(trajectory, fields, engine)
        rho_final_all[atom_index] = single_result.rho_final
        current_slices[atom_index] = trajectory.current_slice
        if progress_every and (atom_index + 1) % progress_every == 0:
            print(f"  evolved {atom_index+1}/{n_atoms} atoms", flush=True)

    counts, rho_bar = bin_current_density_matrices(
        rho_final_all,
        current_slices,
        len(fields.B_z_G),
    )
    return EnsembleResult(
        rho_final_all=rho_final_all,
        current_slices=current_slices,
        counts=counts,
        rho_bar=rho_bar,
    )


def validate_density_collection(rho_collection, N_G, tolerance=1e-9):
    """验证一批 density matrices 的 trace、Hermiticity、PSD 和有限性。"""
    matrices = np.asarray(rho_collection)
    if not np.all(np.isfinite(matrices)):
        raise AssertionError("density matrices 中存在 NaN 或 Inf")

    worst = {
        "trace_error": 0.0,
        "hermiticity_error": 0.0,
        "negative_eigenvalue": 0.0,
        "excited_population_min": np.inf,
        "excited_population_max": -np.inf,
    }
    for rho in matrices:
        diagnostics = density_matrix_diagnostics(rho, N_G)
        worst["trace_error"] = max(
            worst["trace_error"],
            abs(diagnostics["trace"] - 1.0),
        )
        worst["hermiticity_error"] = max(
            worst["hermiticity_error"],
            diagnostics["hermiticity_error"],
        )
        worst["negative_eigenvalue"] = min(
            worst["negative_eigenvalue"],
            diagnostics["minimum_eigenvalue"],
        )
        worst["excited_population_min"] = min(
            worst["excited_population_min"],
            diagnostics["excited_population"],
        )
        worst["excited_population_max"] = max(
            worst["excited_population_max"],
            diagnostics["excited_population"],
        )

    assert worst["trace_error"] < tolerance
    assert worst["hermiticity_error"] < tolerance
    assert worst["negative_eigenvalue"] > -tolerance
    return worst


def ensemble_observables(rho_bar, N_G):
    """MC 收敛需要观察完整矩阵、激发态布居和光学相干。"""
    rho_bar = np.asarray(rho_bar)
    excited = np.trace(rho_bar[:, N_G:, N_G:], axis1=1, axis2=2).real
    optical_block = rho_bar[:, :N_G, N_G:]
    optical_norm = np.linalg.norm(optical_block.reshape(len(rho_bar), -1), axis=1)
    return {
        "excited_population": excited,
        "optical_coherence_norm": optical_norm,
    }


def full_matrix_rms_difference(first, second):
    delta = np.asarray(first) - np.asarray(second)
    return float(np.sqrt(np.mean(np.abs(delta) ** 2)))

