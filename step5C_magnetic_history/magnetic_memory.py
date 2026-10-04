# -*- coding: utf-8 -*-
"""Step 5C 的完整 256 维受控 magnetic-history 实验。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import epsilon_0
from scipy.linalg import expm

import path_setup  # noqa: F401

from benchmark import (
    BenchmarkConfig,
    K39_ISOTOPE_SHIFT_MHZ,
    LinearThermalResponse,
    q_index_for_delta_m,
    si_jones_for_delta_m,
)
from conventions import cart2spherical, ecal_amplitude_from_saturation
from liouvillian import rho_to_vec, vec_to_rho
from moving_atom import SingleAtomEngine, density_matrix_diagnostics
from polarization import density_matrix_to_polarization, k39_number_density


POST_TIMES_NS = np.array(
    [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    dtype=float,
)


@dataclass(frozen=True)
class MagneticMemoryConfig:
    temperature_C: float = 20.0
    saturation: float = 1.0e-6
    delta_m: int = +1
    T_pre_s: float = 1.0e-6
    post_times_ns: tuple[float, ...] = tuple(POST_TIMES_NS)


def determine_local_absorption_peak_MHz(B_z_G, delta_m=+1, step_MHz=0.01):
    """从冻结 fixed-thermal linear benchmark 数值确定 v_z=0 主峰。"""
    response = LinearThermalResponse(BenchmarkConfig(saturation=1.0e-6))
    frequencies, _ = response.transition_lines(float(B_z_G), int(delta_m))
    axis = np.arange(
        float(np.min(frequencies)) - 50.0,
        float(np.max(frequencies)) + 50.0 + step_MHz,
        float(step_MHz),
    )
    chi = response.local_chi(axis, float(B_z_G), int(delta_m))
    peak_index = int(np.argmax(chi.imag))
    return float(axis[peak_index]), complex(chi[peak_index])


class FullMagneticMemoryModel:
    """只使用冻结 SingleAtomEngine 的完整 256 维 Liouvillian。"""

    def __init__(self, current_B_G, past_B_G, config=MagneticMemoryConfig()):
        self.config = config
        self.current_B_G = float(current_B_G)
        self.past_B_G = float(past_B_G)
        self.engine = SingleAtomEngine()
        self.N = self.engine.N
        self.N_G = self.engine.N_G
        self.rho_entry = self.engine.initial_thermal_state()

        self.external_detuning_MHz, self.peak_linear_chi = (
            determine_local_absorption_peak_MHz(
                self.current_B_G,
                config.delta_m,
            )
        )
        self.internal_detuning_MHz = (
            self.external_detuning_MHz + K39_ISOTOPE_SHIFT_MHZ
        )

        # SingleAtomEngine 的 polarization_cart 是 PyLCP 负频率侧的 Cartesian
        # 表示。冻结 Step 5.0 已验证：由 SI 正频率 Jones vector 先取共轭。
        self.si_jones = si_jones_for_delta_m(config.delta_m)
        self.pylcp_polarization_cart = np.conjugate(self.si_jones)
        common = dict(
            saturation=config.saturation,
            polarization_cart=self.pylcp_polarization_cart,
            laser_detuning_MHz=self.internal_detuning_MHz,
            v_z_m_s=0.0,
        )
        self.L_past = self.engine.build_L(B_z_G=self.past_B_G, **common)
        self.L_current = self.engine.build_L(B_z_G=self.current_B_G, **common)

        density = k39_number_density(config.temperature_C)
        self.number_density_m3 = density.k39_m3
        self.C_all = self.engine.ham.d_q_bare["g->e"]
        self.target_q_index = q_index_for_delta_m(config.delta_m)
        Ecal_cart = (
            ecal_amplitude_from_saturation(config.saturation) * self.si_jones
        )
        self.target_Ecal_q_V_m = cart2spherical(Ecal_cart)[self.target_q_index]

    def propagate(self, L, duration_s, rho):
        """直接使用完整 256x256 expm；不投影到 reduced space。"""
        duration_s = float(duration_s)
        if duration_s == 0.0:
            return np.asarray(rho, dtype=complex).copy()
        return vec_to_rho(
            expm(np.asarray(L) * duration_s) @ rho_to_vec(rho),
            self.N,
        )

    def two_stage(self, L_first, L_second, T_first_s, T_second_s):
        rho_after_first = self.propagate(L_first, T_first_s, self.rho_entry)
        return self.propagate(L_second, T_second_s, rho_after_first)

    def polarization(self, rho):
        P_q, P_cart = density_matrix_to_polarization(
            rho,
            self.C_all,
            self.number_density_m3,
        )
        return P_q, P_cart

    def target_polarization(self, rho):
        return complex(self.polarization(rho)[0][self.target_q_index])

    def linear_target_polarization(self, B_z_G):
        response = LinearThermalResponse(
            BenchmarkConfig(
                temperature_C=self.config.temperature_C,
                saturation=self.config.saturation,
            )
        )
        chi = response.local_chi(
            np.array([self.external_detuning_MHz]),
            float(B_z_G),
            self.config.delta_m,
        )[0]
        return complex(epsilon_0 * chi * self.target_Ecal_q_V_m)

    def commutator_metrics(self):
        commutator = self.L_past @ self.L_current - self.L_current @ self.L_past
        norm_past = np.linalg.norm(self.L_past)
        norm_current = np.linalg.norm(self.L_current)
        return {
            "commutator_frobenius_s-2": float(np.linalg.norm(commutator)),
            "normalized_commutator": float(
                np.linalg.norm(commutator) / (norm_past * norm_current)
            ),
        }


def rho_block_norms(delta_rho, N_G):
    return np.array(
        [
            np.linalg.norm(delta_rho[:N_G, :N_G]),
            np.linalg.norm(delta_rho[:N_G, N_G:]),
            np.linalg.norm(delta_rho[N_G:, :N_G]),
            np.linalg.norm(delta_rho[N_G:, N_G:]),
        ],
        dtype=float,
    )


def ground_population_change(rho, rho_entry, N_G):
    return np.asarray(
        np.diag(rho[:N_G, :N_G]).real
        - np.diag(rho_entry[:N_G, :N_G]).real,
        dtype=float,
    )


def run_controls(model, representative_post_s=500.0e-9, split_count=7):
    """运行 identical、uniform-L、segment-splitting 和 weak-pumping controls。"""
    T_pre = model.config.T_pre_s

    identical_a = model.two_stage(
        model.L_current,
        model.L_current,
        T_pre,
        representative_post_s,
    )
    identical_b = model.two_stage(
        model.L_current,
        model.L_current,
        T_pre,
        representative_post_s,
    )
    identical_error = float(np.linalg.norm(identical_a - identical_b))

    uniform_two = identical_a
    uniform_once = model.propagate(
        model.L_current,
        T_pre + representative_post_s,
        model.rho_entry,
    )
    uniform_error = float(np.linalg.norm(uniform_two - uniform_once))

    segment_total_s = T_pre + representative_post_s
    rho_split = model.rho_entry.copy()
    for _ in range(int(split_count)):
        rho_split = model.propagate(
            model.L_current,
            segment_total_s / split_count,
            rho_split,
        )
    split_error = float(np.linalg.norm(rho_split - uniform_once))

    rho_half = model.propagate(
        model.L_current,
        0.5 * T_pre,
        model.rho_entry,
    )
    rho_pre_current = model.propagate(
        model.L_current,
        T_pre,
        model.rho_entry,
    )
    rho_pre_past = model.propagate(
        model.L_past,
        T_pre,
        model.rho_entry,
    )
    P_half = model.target_polarization(rho_half)
    P_pre_current = model.target_polarization(rho_pre_current)
    P_linear_current = model.linear_target_polarization(model.current_B_G)
    coherence_plateau_relative = float(
        abs(P_pre_current - P_half) / max(abs(P_pre_current), 1.0e-300)
    )
    linear_response_relative = float(
        abs(P_pre_current - P_linear_current)
        / max(abs(P_linear_current), 1.0e-300)
    )

    ground_current = ground_population_change(
        rho_pre_current,
        model.rho_entry,
        model.N_G,
    )
    ground_past = ground_population_change(
        rho_pre_past,
        model.rho_entry,
        model.N_G,
    )
    initial_history_rho = float(np.linalg.norm(rho_pre_past - rho_pre_current))
    max_ground_population_change = float(
        max(np.max(np.abs(ground_current)), np.max(np.abs(ground_past)))
    )
    ground_change_norm = float(
        max(np.linalg.norm(ground_current), np.linalg.norm(ground_past))
    )

    diagnostics = density_matrix_diagnostics(uniform_once, model.N_G)
    result = {
        "identical_history_error": identical_error,
        "uniform_L_error": uniform_error,
        "segment_splitting_error": split_error,
        "coherence_plateau_relative": coherence_plateau_relative,
        "linear_response_relative": linear_response_relative,
        "max_ground_population_change": max_ground_population_change,
        "ground_population_change_norm": ground_change_norm,
        "ground_change_to_initial_history_ratio": float(
            ground_change_norm / max(initial_history_rho, 1.0e-300)
        ),
        "initial_history_rho_norm": initial_history_rho,
        "trace_error": float(abs(diagnostics["trace"] - 1.0)),
        "hermiticity_error": diagnostics["hermiticity_error"],
        "minimum_eigenvalue": diagnostics["minimum_eigenvalue"],
    }
    result.update(model.commutator_metrics())
    return result


def run_memory_scan(model):
    """比较 past->current 与 current->current，保持其他条件完全相同。"""
    times_ns = np.asarray(model.config.post_times_ns, dtype=float)
    times_s = times_ns * 1.0e-9
    rho_pre_history = model.propagate(
        model.L_past,
        model.config.T_pre_s,
        model.rho_entry,
    )
    rho_pre_reference = model.propagate(
        model.L_current,
        model.config.T_pre_s,
        model.rho_entry,
    )

    rho_history = np.empty((len(times_s), model.N, model.N), dtype=complex)
    rho_reference = np.empty_like(rho_history)
    P_history_q = np.empty((len(times_s), 3), dtype=complex)
    P_reference_q = np.empty_like(P_history_q)
    P_history_cart = np.empty_like(P_history_q)
    P_reference_cart = np.empty_like(P_history_q)
    rho_norm = np.empty(len(times_s), dtype=float)
    block_norms = np.empty((len(times_s), 4), dtype=float)

    max_trace_error = 0.0
    max_hermiticity_error = 0.0
    minimum_eigenvalue = np.inf
    for index, T_post_s in enumerate(times_s):
        rho_H = model.propagate(model.L_current, T_post_s, rho_pre_history)
        rho_R = model.propagate(model.L_current, T_post_s, rho_pre_reference)
        rho_history[index] = rho_H
        rho_reference[index] = rho_R
        P_history_q[index], P_history_cart[index] = model.polarization(rho_H)
        P_reference_q[index], P_reference_cart[index] = model.polarization(rho_R)
        delta_rho = rho_H - rho_R
        rho_norm[index] = np.linalg.norm(delta_rho)
        block_norms[index] = rho_block_norms(delta_rho, model.N_G)

        for rho in (rho_H, rho_R):
            if not np.all(np.isfinite(rho)):
                raise AssertionError("正式时间点出现 NaN/Inf")
            diagnostics = density_matrix_diagnostics(rho, model.N_G)
            max_trace_error = max(
                max_trace_error,
                float(abs(diagnostics["trace"] - 1.0)),
            )
            max_hermiticity_error = max(
                max_hermiticity_error,
                diagnostics["hermiticity_error"],
            )
            minimum_eigenvalue = min(
                minimum_eigenvalue,
                diagnostics["minimum_eigenvalue"],
            )

    target = model.target_q_index
    P_history_target = P_history_q[:, target]
    P_reference_target = P_reference_q[:, target]
    delta_P = P_history_target - P_reference_target
    delta_P_abs = np.abs(delta_P)
    if delta_P_abs[0] <= 1.0e-300:
        normalization = float(np.max(np.abs(P_reference_target)))
        normalization_label = "max |P_reference|"
    else:
        normalization = float(delta_P_abs[0])
        normalization_label = "|Delta P(0)|"
    memory_P = delta_P_abs / normalization
    memory_rho = rho_norm / max(rho_norm[0], 1.0e-300)

    ground_population_history = np.array(
        [
            np.diag(rho[: model.N_G, : model.N_G]).real
            for rho in rho_history
        ]
    )
    ground_population_reference = np.array(
        [
            np.diag(rho[: model.N_G, : model.N_G]).real
            for rho in rho_reference
        ]
    )

    return {
        "times_ns": times_ns,
        "rho_history": rho_history,
        "rho_reference": rho_reference,
        "delta_rho_norm": rho_norm,
        "delta_rho_block_norms": block_norms,
        "P_history_q": P_history_q,
        "P_reference_q": P_reference_q,
        "P_history_cart": P_history_cart,
        "P_reference_cart": P_reference_cart,
        "P_history_target": P_history_target,
        "P_reference_target": P_reference_target,
        "delta_P_target": delta_P,
        "delta_P_abs": delta_P_abs,
        "memory_P": memory_P,
        "memory_rho": memory_rho,
        "memory_normalization": normalization,
        "memory_normalization_label": normalization_label,
        "ground_population_history": ground_population_history,
        "ground_population_reference": ground_population_reference,
        "max_trace_error": max_trace_error,
        "max_hermiticity_error": max_hermiticity_error,
        "minimum_eigenvalue": minimum_eigenvalue,
    }


def persistent_threshold_time_ns(times_ns, normalized_memory, threshold):
    """返回离散扫描中此后始终不超过 threshold 的首个时间；否则 NaN。"""
    times_ns = np.asarray(times_ns, dtype=float)
    memory = np.asarray(normalized_memory, dtype=float)
    suffix_max = np.maximum.accumulate(memory[::-1])[::-1]
    matches = np.flatnonzero(suffix_max <= float(threshold))
    return float(times_ns[matches[0]]) if len(matches) else float("nan")
