# -*- coding: utf-8 -*-
"""单个运动原子沿历史切片的密度矩阵演化。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

import path_setup  # noqa: F401

from k39_model import (
    K_D1,
    build_k39_d1_hamiltonian,
    detuning_MHz_to_rad_s,
    thermal_ground_state,
)
from liouvillian import (
    coherent_liouvillian,
    decay_liouvillian,
    physical_hamiltonian_from_fields,
    rho_to_vec,
    vec_to_rho,
)
from pylcp.common import cart2spherical


@dataclass(frozen=True)
class SliceFields:
    B_z_G: np.ndarray
    saturation: np.ndarray
    polarization_cart: np.ndarray
    laser_detuning_MHz: np.ndarray

    @classmethod
    def uniform(
        cls,
        n_slices,
        B_z_G=300.0,
        saturation=0.05,
        polarization_cart=(1.0, 0.0, 0.0),
        laser_detuning_MHz=0.0,
    ):
        n_slices = int(n_slices)
        polarization = np.asarray(polarization_cart, dtype=complex)
        return cls(
            B_z_G=np.full(n_slices, float(B_z_G)),
            saturation=np.full(n_slices, float(saturation)),
            polarization_cart=np.tile(polarization, (n_slices, 1)),
            laser_detuning_MHz=np.full(n_slices, float(laser_detuning_MHz)),
        )

    def validate(self, n_slices):
        n_slices = int(n_slices)
        if np.asarray(self.B_z_G).shape != (n_slices,):
            raise ValueError("B_z_G shape 必须为 (n_slices,)")
        if np.asarray(self.saturation).shape != (n_slices,):
            raise ValueError("saturation shape 必须为 (n_slices,)")
        if np.asarray(self.polarization_cart).shape != (n_slices, 3):
            raise ValueError("polarization_cart shape 必须为 (n_slices,3)")
        if np.asarray(self.laser_detuning_MHz).shape != (n_slices,):
            raise ValueError("laser_detuning_MHz shape 必须为 (n_slices,)")
        if np.any(np.asarray(self.saturation) < 0.0):
            raise ValueError("饱和参数不能为负")


@dataclass(frozen=True)
class EvolutionStep:
    slice_index: int
    dt_s: float
    B_z_G: float
    saturation: float
    polarization_cart: np.ndarray
    laser_detuning_MHz: float
    doppler_shift_MHz: float
    effective_detuning_MHz: float
    rho_before: np.ndarray
    rho_after: np.ndarray


@dataclass(frozen=True)
class EvolutionResult:
    rho_initial: np.ndarray
    rho_final: np.ndarray
    steps: tuple[EvolutionStep, ...]


def normalized_dimensionless_E_q(saturation, polarization_cart):
    """把饱和参数和笛卡尔偏振转成 pylcp 的无量纲球基底电场。"""
    saturation = float(saturation)
    polarization = np.asarray(polarization_cart, dtype=complex)
    if saturation < 0.0:
        raise ValueError("saturation 不能为负")
    if abs(polarization[2]) > 1e-12:
        raise ValueError("光沿 z 传播时偏振必须横向，E_z 应为 0")

    norm = np.sqrt(np.vdot(polarization, polarization).real)
    if norm <= 0.0:
        if saturation == 0.0:
            return np.zeros(3, dtype=complex)
        raise ValueError("非零光强需要非零偏振向量")

    polarization = polarization / norm
    return np.sqrt(2.0 * saturation) * cart2spherical(polarization)


def doppler_shift_MHz(v_z_m_s):
    """返回 k*v_z/(2*pi)，单位 MHz；正速度表示沿 +z。"""
    return K_D1 * float(v_z_m_s) / (2.0 * np.pi * 1e6)


def effective_detuning_rad_s(laser_detuning_MHz, v_z_m_s):
    """delta_eff = delta_laser - k*v_z，返回 rad/s。"""
    return detuning_MHz_to_rad_s(laser_detuning_MHz) - K_D1 * float(v_z_m_s)


class SingleAtomEngine:
    """预计算场无关部分，并为单原子轨迹构造/缓存 L。"""

    def __init__(self):
        self.ham, self.basis_g, self.basis_e = build_k39_d1_hamiltonian()
        self.N_G = len(self.basis_g)
        self.N_E = len(self.basis_e)
        self.N = self.ham.n
        self.L_decay = decay_liouvillian(self.ham.d_q_bare["g->e"])
        self._L_cache = {}

    def initial_thermal_state(self):
        return thermal_ground_state(self.N_G, self.N_E)

    def build_L(
        self,
        B_z_G,
        saturation,
        polarization_cart,
        laser_detuning_MHz,
        v_z_m_s,
    ):
        polarization = np.asarray(polarization_cart, dtype=complex)
        key = (
            float(B_z_G),
            float(saturation),
            tuple(complex(value) for value in polarization),
            float(laser_detuning_MHz),
            float(v_z_m_s),
        )
        if key in self._L_cache:
            return self._L_cache[key]

        E_q = normalized_dimensionless_E_q(saturation, polarization)
        delta_eff = effective_detuning_rad_s(laser_detuning_MHz, v_z_m_s)
        H = physical_hamiltonian_from_fields(
            self.ham,
            E_q,
            np.array([0.0, 0.0, float(B_z_G)]),
            detuning_rad_s=delta_eff,
        )
        L = coherent_liouvillian(H) + self.L_decay
        self._L_cache[key] = L
        return L


def evolve_trajectory(trajectory, fields, engine, rho_initial=None):
    """沿 trajectory.segments 依次累积完整密度矩阵。"""
    n_slices = len(fields.B_z_G)
    fields.validate(n_slices)

    if rho_initial is None:
        rho_initial = engine.initial_thermal_state()
    rho = np.asarray(rho_initial, dtype=complex).copy()
    if rho.shape != (engine.N, engine.N):
        raise ValueError(f"rho_initial shape 必须为 {(engine.N, engine.N)}")

    steps = []
    v_z = float(trajectory.velocity_m_s[2])
    doppler_MHz = doppler_shift_MHz(v_z)

    for segment in trajectory.segments:
        k = int(segment.slice_index)
        L = engine.build_L(
            B_z_G=fields.B_z_G[k],
            saturation=fields.saturation[k],
            polarization_cart=fields.polarization_cart[k],
            laser_detuning_MHz=fields.laser_detuning_MHz[k],
            v_z_m_s=v_z,
        )

        rho_before = rho.copy()
        rho = vec_to_rho(
            expm(L * segment.dt_s) @ rho_to_vec(rho),
            engine.N,
        )

        steps.append(
            EvolutionStep(
                slice_index=k,
                dt_s=segment.dt_s,
                B_z_G=float(fields.B_z_G[k]),
                saturation=float(fields.saturation[k]),
                polarization_cart=np.asarray(fields.polarization_cart[k]).copy(),
                laser_detuning_MHz=float(fields.laser_detuning_MHz[k]),
                doppler_shift_MHz=doppler_MHz,
                effective_detuning_MHz=(
                    float(fields.laser_detuning_MHz[k]) - doppler_MHz
                ),
                rho_before=rho_before,
                rho_after=rho.copy(),
            )
        )

    return EvolutionResult(
        rho_initial=np.asarray(rho_initial).copy(),
        rho_final=rho,
        steps=tuple(steps),
    )


def density_matrix_diagnostics(rho, N_G):
    rho = np.asarray(rho)
    return {
        "trace": np.trace(rho),
        "hermiticity_error": float(np.max(np.abs(rho - rho.conj().T))),
        "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(rho)).real),
        "excited_population": float(np.trace(rho[N_G:, N_G:]).real),
    }

