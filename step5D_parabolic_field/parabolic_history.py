# -*- coding: utf-8 -*-
"""真实纵向抛物线 B(z) 的 full-state trajectory/local benchmark。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import epsilon_0
from scipy.linalg import eigh, expm

import path_setup  # noqa: F401

from benchmark import (
    BenchmarkConfig,
    K39_ISOTOPE_SHIFT_MHZ,
    LinearThermalResponse,
    q_index_for_delta_m,
    si_jones_for_delta_m,
)
from conventions import cart2spherical, ecal_amplitude_from_saturation
from k39_model import GAMMA_D1, LAMBDA_D1
from liouvillian import physical_hamiltonian_from_fields, rho_to_vec, vec_to_rho
from moving_atom import (
    SingleAtomEngine,
    density_matrix_diagnostics,
    effective_detuning_rad_s,
    normalized_dimensionless_E_q,
)
from polarization import density_matrix_to_polarization, k39_number_density


@dataclass(frozen=True)
class ParabolicFieldConfig:
    length_m: float = 25.0e-3
    B_center_G: float = 2500.0
    delta_B_G: float = 500.0
    temperature_C: float = 20.0
    saturation: float = 1.0e-6
    delta_m: int = +1
    tau_memory_s: float = 100.0e-9
    max_internal_dt_s: float = 1.25e-9


@dataclass(frozen=True)
class TrajectoryProfile:
    velocity_m_s: float
    dz_m: float
    z_m: np.ndarray
    B_G: np.ndarray
    dB_dt_G_s: np.ndarray
    delta_B_memory_G: np.ndarray
    P_trajectory_q: np.ndarray
    P_local_q: np.ndarray
    epsilon_position: np.ndarray
    rho_final: np.ndarray
    max_trace_error: float
    max_hermiticity_error: float
    minimum_eigenvalue: float
    internal_substeps: int

    @property
    def target_index(self):
        return 0

    @property
    def P_trajectory(self):
        return self.P_trajectory_q[:, self.target_index]

    @property
    def P_local(self):
        return self.P_local_q[:, self.target_index]

    @property
    def epsilon_global(self):
        return float(
            np.linalg.norm(self.P_trajectory - self.P_local)
            / np.linalg.norm(self.P_local)
        )

    @property
    def max_epsilon(self):
        return float(np.max(self.epsilon_position))


def B_parabolic_G(z_m, config=ParabolicFieldConfig()):
    z = np.asarray(z_m, dtype=float)
    return config.B_center_G - config.delta_B_G * (
        2.0 * z / config.length_m
    ) ** 2


def dB_dz_G_m(z_m, config=ParabolicFieldConfig()):
    z = np.asarray(z_m, dtype=float)
    return -8.0 * config.delta_B_G * z / config.length_m**2


def determine_peak_fast_MHz(B_z_G, delta_m=+1, window_MHz=30.0, step_MHz=0.01):
    """只在各跃迁中心附近扫描，数值确定局域 weak-linear 主峰。"""
    response = LinearThermalResponse(BenchmarkConfig(saturation=1.0e-6))
    frequencies, _ = response.transition_lines(float(B_z_G), int(delta_m))
    axes = [
        np.arange(freq - window_MHz, freq + window_MHz + step_MHz, step_MHz)
        for freq in frequencies
    ]
    axis = np.unique(np.concatenate(axes))
    chi = response.local_chi(axis, float(B_z_G), int(delta_m))
    peak = int(np.argmax(chi.imag))
    return float(axis[peak]), complex(chi[peak])


class FullStateSplitPropagator:
    """完整 16x16 rho（256 components）的对称 Liouvillian factorization。

    不做 reachable-subspace 投影。coherent 部分用 16x16 Hermitian eigensystem
    精确传播，冻结 spontaneous-decay Lindbladian 使用其解析 block map。正式
    controls 会逐点与 dense expm(full 256x256 L) 比较。
    """

    def __init__(self, config=ParabolicFieldConfig()):
        self.config = config
        self.engine = SingleAtomEngine()
        self.N = self.engine.N
        self.N_G = self.engine.N_G
        self.rho_entry = self.engine.initial_thermal_state()
        self.C_all = self.engine.ham.d_q_bare["g->e"]
        self.si_jones = si_jones_for_delta_m(config.delta_m)
        self.pylcp_polarization_cart = np.conjugate(self.si_jones)
        self.E_q_pylcp = normalized_dimensionless_E_q(
            config.saturation,
            self.pylcp_polarization_cart,
        )
        self.external_laser_detuning_MHz, self.center_peak_chi = (
            determine_peak_fast_MHz(config.B_center_G, config.delta_m)
        )
        self.internal_laser_detuning_MHz = (
            self.external_laser_detuning_MHz + K39_ISOTOPE_SHIFT_MHZ
        )
        density = k39_number_density(config.temperature_C)
        self.number_density_m3 = density.k39_m3
        self.target_q_index = q_index_for_delta_m(config.delta_m)
        Ecal_cart = (
            ecal_amplitude_from_saturation(config.saturation) * self.si_jones
        )
        self.target_Ecal_q_V_m = cart2spherical(Ecal_cart)[self.target_q_index]
        self.linear_response = LinearThermalResponse(
            BenchmarkConfig(
                temperature_C=config.temperature_C,
                saturation=config.saturation,
            )
        )

        completeness = np.sum(
            [C.conj().T @ C for C in self.C_all],
            axis=0,
        )
        projector_e = np.zeros((self.N, self.N), dtype=complex)
        projector_e[self.N_G :, self.N_G :] = np.eye(self.N - self.N_G)
        self.decay_completeness_error = float(
            np.max(np.abs(completeness - projector_e))
        )

    def hamiltonian(self, B_z_G, velocity_m_s):
        delta_eff = effective_detuning_rad_s(
            self.internal_laser_detuning_MHz,
            velocity_m_s,
        )
        return physical_hamiltonian_from_fields(
            self.engine.ham,
            self.E_q_pylcp,
            np.array([0.0, 0.0, float(B_z_G)]),
            detuning_rad_s=delta_eff,
        )

    @staticmethod
    def _coherent_propagator(energies, vectors, duration_s):
        phases = np.exp(-1.0j * energies * float(duration_s))
        return (vectors * phases) @ vectors.conj().T

    @classmethod
    def _coherent_step(cls, rho, H, duration_s):
        energies, vectors = eigh(H, check_finite=False)
        U = cls._coherent_propagator(energies, vectors, duration_s)
        return U @ rho @ U.conj().T

    def _decay_step(self, rho, duration_s):
        factor = float(np.exp(-GAMMA_D1 * float(duration_s)))
        coherence_factor = np.sqrt(factor)
        result = np.asarray(rho, dtype=complex).copy()
        excited = np.zeros_like(result)
        excited[self.N_G :, self.N_G :] = rho[self.N_G :, self.N_G :]
        feeding = np.zeros_like(result)
        for C in self.C_all:
            feeding += C @ excited @ C.conj().T
        result[: self.N_G, : self.N_G] += (
            (1.0 - factor) * feeding[: self.N_G, : self.N_G]
        )
        result[: self.N_G, self.N_G :] *= coherence_factor
        result[self.N_G :, : self.N_G] *= coherence_factor
        result[self.N_G :, self.N_G :] *= factor
        return result

    def strang_step(self, rho, H, duration_s):
        half = 0.5 * float(duration_s)
        result = self._coherent_step(rho, H, half)
        result = self._decay_step(result, duration_s)
        return self._coherent_step(result, H, half)

    def _strang_with_propagator(self, rho, U_half, duration_s):
        result = U_half @ rho @ U_half.conj().T
        result = self._decay_step(result, duration_s)
        return U_half @ result @ U_half.conj().T

    def fourth_order_step(self, rho, H, duration_s):
        """Suzuki five-factor fourth-order composition of symmetric S2."""
        energies, vectors = eigh(H, check_finite=False)
        coefficient = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        middle = 1.0 - 4.0 * coefficient
        U_outer = self._coherent_propagator(
            energies, vectors, 0.5 * coefficient * duration_s
        )
        U_middle = self._coherent_propagator(
            energies, vectors, 0.5 * middle * duration_s
        )
        result = np.asarray(rho, dtype=complex).copy()
        for _ in range(2):
            result = self._strang_with_propagator(
                result, U_outer, coefficient * duration_s
            )
        result = self._strang_with_propagator(
            result, U_middle, middle * duration_s
        )
        for _ in range(2):
            result = self._strang_with_propagator(
                result, U_outer, coefficient * duration_s
            )
        return result

    def field_segment(self, rho, B_z_G, velocity_m_s, duration_s, max_dt_s=None):
        if max_dt_s is None:
            max_dt_s = self.config.max_internal_dt_s
        count = max(1, int(np.ceil(float(duration_s) / float(max_dt_s))))
        sub_dt = float(duration_s) / count
        H = self.hamiltonian(B_z_G, velocity_m_s)
        energies, vectors = eigh(H, check_finite=False)
        coefficient = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        middle = 1.0 - 4.0 * coefficient
        U_outer = self._coherent_propagator(
            energies, vectors, 0.5 * coefficient * sub_dt
        )
        U_middle = self._coherent_propagator(
            energies, vectors, 0.5 * middle * sub_dt
        )
        result = np.asarray(rho, dtype=complex).copy()
        for _ in range(count):
            for _ in range(2):
                result = self._strang_with_propagator(
                    result, U_outer, coefficient * sub_dt
                )
            result = self._strang_with_propagator(
                result, U_middle, middle * sub_dt
            )
            for _ in range(2):
                result = self._strang_with_propagator(
                    result, U_outer, coefficient * sub_dt
                )
        return result, count

    def dense_step(self, rho, B_z_G, velocity_m_s, duration_s):
        L = self.engine.build_L(
            B_z_G=float(B_z_G),
            saturation=self.config.saturation,
            polarization_cart=self.pylcp_polarization_cart,
            laser_detuning_MHz=self.internal_laser_detuning_MHz,
            v_z_m_s=float(velocity_m_s),
        )
        return vec_to_rho(
            expm(L * float(duration_s)) @ rho_to_vec(rho),
            self.N,
        )

    def polarization_q(self, rho):
        return density_matrix_to_polarization(
            rho,
            self.C_all,
            self.number_density_m3,
        )[0]

    def local_P_q(self, B_values_G, velocity_m_s):
        B_values = np.asarray(B_values_G, dtype=float)
        effective_external = (
            self.external_laser_detuning_MHz
            - float(velocity_m_s) / LAMBDA_D1 / 1.0e6
        )
        values = np.zeros((len(B_values), 3), dtype=complex)
        # 对称抛物线会重复 B；只计算唯一值并映射回来。
        unique_B, inverse = np.unique(B_values, return_inverse=True)
        unique_P = np.empty(len(unique_B), dtype=complex)
        for index, B_G in enumerate(unique_B):
            chi = self.linear_response.local_chi(
                np.array([effective_external]),
                float(B_G),
                self.config.delta_m,
            )[0]
            unique_P[index] = (
                epsilon_0 * chi * self.target_Ecal_q_V_m
            )
        values[:, self.target_q_index] = unique_P[inverse]
        return values

    def trajectory(self, velocity_m_s, dz_m, delta_B_G=None, max_dt_s=None):
        velocity = float(velocity_m_s)
        if velocity == 0.0:
            raise ValueError("trajectory velocity 不能为 0")
        if delta_B_G is None:
            delta_B_G = self.config.delta_B_G
        ratio = self.config.length_m / float(dz_m)
        n_steps = max(1, int(np.rint(ratio)))
        if n_steps * float(dz_m) < self.config.length_m * (1.0 - 1.0e-12):
            n_steps += 1
        dz_exact = self.config.length_m / n_steps
        if velocity > 0.0:
            z = np.linspace(-self.config.length_m / 2.0, self.config.length_m / 2.0, n_steps + 1)
        else:
            z = np.linspace(self.config.length_m / 2.0, -self.config.length_m / 2.0, n_steps + 1)

        local_config = ParabolicFieldConfig(
            length_m=self.config.length_m,
            B_center_G=self.config.B_center_G,
            delta_B_G=float(delta_B_G),
            temperature_C=self.config.temperature_C,
            saturation=self.config.saturation,
            delta_m=self.config.delta_m,
            tau_memory_s=self.config.tau_memory_s,
            max_internal_dt_s=self.config.max_internal_dt_s,
        )
        B_values = B_parabolic_G(z, local_config)
        derivative = dB_dz_G_m(z, local_config)
        dB_dt = velocity * derivative
        delta_B_memory = np.abs(dB_dt) * self.config.tau_memory_s
        P_local_q = self.local_P_q(B_values, velocity)
        P_trajectory_q = np.empty_like(P_local_q)

        rho = self.rho_entry.copy()
        P_trajectory_q[0] = self.polarization_q(rho)
        max_trace_error = 0.0
        max_hermiticity_error = 0.0
        minimum_eigenvalue = np.inf
        internal_substeps = 0
        segment_time = dz_exact / abs(velocity)
        for index in range(n_steps):
            if max_dt_s is None:
                internal_limit = self.config.max_internal_dt_s
            else:
                internal_limit = float(max_dt_s)
            used = max(1, int(np.ceil(segment_time / internal_limit)))
            sub_dt = segment_time / used
            for sub_index in range(used):
                fraction = (sub_index + 0.5) / used
                midpoint = z[index] + fraction * (z[index + 1] - z[index])
                B_midpoint = float(B_parabolic_G(midpoint, local_config))
                H = self.hamiltonian(B_midpoint, velocity)
                rho = self.fourth_order_step(rho, H, sub_dt)
            internal_substeps += used
            P_trajectory_q[index + 1] = self.polarization_q(rho)
            diagnostics = density_matrix_diagnostics(rho, self.N_G)
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

        target = self.target_q_index
        scale = float(np.max(np.abs(P_local_q[:, target])))
        epsilon = np.abs(
            P_trajectory_q[:, target] - P_local_q[:, target]
        ) / max(scale, 1.0e-300)
        return TrajectoryProfile(
            velocity_m_s=velocity,
            dz_m=dz_exact,
            z_m=z,
            B_G=B_values,
            dB_dt_G_s=dB_dt,
            delta_B_memory_G=delta_B_memory,
            P_trajectory_q=P_trajectory_q,
            P_local_q=P_local_q,
            epsilon_position=epsilon,
            rho_final=rho,
            max_trace_error=max_trace_error,
            max_hermiticity_error=max_hermiticity_error,
            minimum_eigenvalue=minimum_eigenvalue,
            internal_substeps=internal_substeps,
        )


def bulk_mask(profile, tau_multiple=5.0, tau_memory_s=100.0e-9):
    """排除入口后的 coherence-build-up 边界层；方向由轨迹数组自动处理。"""
    distance = abs(profile.velocity_m_s) * float(tau_multiple) * float(tau_memory_s)
    travelled = np.abs(profile.z_m - profile.z_m[0])
    return travelled >= distance


def profile_metrics(profile, tau_memory_s=100.0e-9):
    target = profile.target_index
    delta = profile.P_trajectory_q[:, target] - profile.P_local_q[:, target]
    local = profile.P_local_q[:, target]
    mask = bulk_mask(profile, tau_memory_s=tau_memory_s)
    bulk_delta = delta[mask]
    bulk_local = local[mask]
    bulk_epsilon = profile.epsilon_position[mask]
    bulk_indices = np.flatnonzero(mask)
    raw_index = int(np.argmax(profile.epsilon_position))
    bulk_relative_index = int(np.argmax(bulk_epsilon))
    bulk_index = int(bulk_indices[bulk_relative_index])
    return {
        "epsilon_global_raw": float(np.linalg.norm(delta) / np.linalg.norm(local)),
        "max_epsilon_raw": float(profile.epsilon_position[raw_index]),
        "max_epsilon_raw_z_mm": float(profile.z_m[raw_index] * 1.0e3),
        "epsilon_global_bulk": float(
            np.linalg.norm(bulk_delta) / np.linalg.norm(bulk_local)
        ),
        "max_epsilon_bulk": float(bulk_epsilon[bulk_relative_index]),
        "max_epsilon_bulk_z_mm": float(profile.z_m[bulk_index] * 1.0e3),
        "excluded_entry_distance_mm": float(
            abs(profile.velocity_m_s) * 5.0 * tau_memory_s * 1.0e3
        ),
        "max_delta_B_memory_G": float(np.max(profile.delta_B_memory_G)),
    }


def hysteresis_metrics(profile, target_fields_G=(2100.0, 2200.0, 2300.0, 2400.0)):
    """同一轨迹升场/降场经过相同 B 时的 P 差异。"""
    target = profile.target_index
    scale = float(np.max(np.abs(profile.P_local_q[:, target])))
    order = np.argsort(profile.z_m)
    z_sorted = profile.z_m[order]
    P_sorted = profile.P_trajectory_q[order, target]
    output = []
    for field in target_fields_G:
        fraction = np.sqrt(
            (2500.0 - float(field)) / 500.0
        )
        z_abs = 0.5 * 25.0e-3 * fraction
        left = np.interp(-z_abs, z_sorted, P_sorted.real) + 1.0j * np.interp(
            -z_abs, z_sorted, P_sorted.imag
        )
        right = np.interp(z_abs, z_sorted, P_sorted.real) + 1.0j * np.interp(
            z_abs, z_sorted, P_sorted.imag
        )
        output.append(
            (
                float(field),
                float(-z_abs * 1.0e3),
                float(z_abs * 1.0e3),
                complex(left),
                complex(right),
                float(abs(left - right) / max(scale, 1.0e-300)),
            )
        )
    return output
