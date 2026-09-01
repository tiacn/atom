# -*- coding: utf-8 -*-
"""把旋转框架 density matrix 转成 SI 宏观复极化 phasor。

本模块只负责 ``rho -> P``。它不负责轨迹演化、ensemble binning、
susceptibility 或 Maxwell 传播。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann

import path_setup  # noqa: F401

# 这些量和变换直接来自已经通过的 Step 5.0，避免另建一套 convention。
from conventions import (
    PHASOR_CONVENTION,
    Q_VALUES,
    cart2spherical,
    dipole_scale_from_spontaneous_emission,
    polarization_phasor_spherical,
    spherical2cart,
)


P_UNIT = "C/m^2"
NUMBER_DENSITY_UNIT = "1/m^3"
DIPOLE_UNIT = "C*m"
POLARIZATION_PHASOR_CONVENTION = "P_phys = Re[Pcal exp(-i*omega*t)]"

# ElecSus spectra.py 默认天然钾比例：K40=0.01%，K41=6.73%，
# K39=1-K40-K41=93.26%。若实验使用富集 K39，调用者应显式传入实际比例。
NATURAL_K40_FRACTION = 0.0001
NATURAL_K41_FRACTION = 0.0673
NATURAL_K39_FRACTION = 1.0 - NATURAL_K40_FRACTION - NATURAL_K41_FRACTION


@dataclass(frozen=True)
class IsotopeNumberDensity:
    """钾蒸气总数密度和其中 K39 的真实数密度。"""

    temperature_K: float
    total_potassium_m3: float
    k39_fraction: float
    k39_m3: float


@dataclass(frozen=True)
class PolarizationProfile:
    """逐 slice 的 density matrix 和 SI 极化结果。"""

    rho_bar: np.ndarray
    P_q: np.ndarray
    P_cart: np.ndarray
    number_density_m3: float
    d0_C_m: float
    phasor_convention: str = PHASOR_CONVENTION
    polarization_phasor_convention: str = POLARIZATION_PHASOR_CONVENTION


def potassium_number_density_m3(temperature_K):
    """返回总钾蒸气数密度，单位 m^-3。

    经验式逐字对应 ElecSus ``numberDensityEqs.py::numDenK``，其来源为
    Alcock, Itkin and Horrigan, Can. Metall. Q. 23 (1984) 309-313。
    中间量 ``p_atm`` 是以标准大气压为单位的蒸气压。
    """
    temperature_K = float(temperature_K)
    if not np.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError("temperature_K 必须是有限正数")

    if temperature_K < 336.8:
        p_atm = 10.0 ** (4.961 - 4646.0 / temperature_K)
    else:
        p_atm = 10.0 ** (
            8.233 - 4693.0 / temperature_K - 1.2403 * np.log10(temperature_K)
        )
    return float(101325.0 * p_atm / (Boltzmann * temperature_K))


def k39_number_density(temperature_C, k39_fraction=NATURAL_K39_FRACTION):
    """返回指定温度和同位素比例下的 K39 数密度信息。

    默认值代表天然钾。纯/富集 K39 样品可传 ``k39_fraction=1.0`` 或实验值。
    这里的同位素比例是物理样品参数，与 MC 原子数完全无关。
    """
    temperature_C = float(temperature_C)
    k39_fraction = float(k39_fraction)
    if not np.isfinite(temperature_C):
        raise ValueError("temperature_C 必须有限")
    if not np.isfinite(k39_fraction) or not 0.0 <= k39_fraction <= 1.0:
        raise ValueError("k39_fraction 必须位于 [0,1]")

    temperature_K = temperature_C + 273.15
    total = potassium_number_density_m3(temperature_K)
    return IsotopeNumberDensity(
        temperature_K=temperature_K,
        total_potassium_m3=total,
        k39_fraction=k39_fraction,
        k39_m3=k39_fraction * total,
    )


def _validate_rho_and_operators(rho, d_q_bare):
    rho = np.asarray(rho, dtype=complex)
    d_q_bare = np.asarray(d_q_bare, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho 必须是方阵")
    if d_q_bare.shape != (3, rho.shape[0], rho.shape[1]):
        raise ValueError("d_q_bare shape 必须为 (3,N,N)，顺序 [q=-1,0,+1]")
    if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(d_q_bare)):
        raise ValueError("rho 和 d_q_bare 必须全部有限")
    return rho, d_q_bare


def _validate_number_density(number_density_m3):
    number_density_m3 = float(number_density_m3)
    if not np.isfinite(number_density_m3) or number_density_m3 < 0.0:
        raise ValueError("number_density_m3 必须是有限非负数")
    return number_density_m3


def spherical_to_cartesian(P_q):
    """把最后一维为 ``[-1,0,+1]`` 的球分量转为 ``[x,y,z]``。"""
    P_q = np.asarray(P_q, dtype=complex)
    if P_q.ndim < 1 or P_q.shape[-1] != 3:
        raise ValueError("P_q 最后一维必须为 3，顺序 [q=-1,0,+1]")
    flat = P_q.reshape(-1, 3)
    if len(flat) == 0:
        return np.empty_like(P_q)
    converted = np.stack([spherical2cart(vector) for vector in flat], axis=0)
    return converted.reshape(P_q.shape)


def cartesian_to_spherical(P_cart):
    """把最后一维为 ``[x,y,z]`` 的 Cartesian 分量转为球分量。"""
    P_cart = np.asarray(P_cart, dtype=complex)
    if P_cart.ndim < 1 or P_cart.shape[-1] != 3:
        raise ValueError("P_cart 最后一维必须为 3，顺序 [x,y,z]")
    flat = P_cart.reshape(-1, 3)
    if len(flat) == 0:
        return np.empty_like(P_cart)
    converted = np.stack([cart2spherical(vector) for vector in flat], axis=0)
    return converted.reshape(P_cart.shape)


def density_matrix_to_polarization(rho, d_q_bare, number_density_m3):
    """单个 ``rho[N,N] -> (P_q[3], P_cart[3])``，输出单位 C/m^2。"""
    rho, d_q_bare = _validate_rho_and_operators(rho, d_q_bare)
    number_density_m3 = _validate_number_density(number_density_m3)

    # Step 5.0 已验证：Pcal_q=2*N_iso*d0*Tr(C_q rho)，C_q=|g><e|。
    P_q = polarization_phasor_spherical(
        d_q_bare,
        rho,
        number_density_m3,
    )
    P_cart = spherical_to_cartesian(P_q)
    return P_q, P_cart


def density_matrices_to_polarization(rho_bar, d_q_bare, number_density_m3):
    """逐 slice 批量转换并返回带元数据的正式结果对象。"""
    rho_bar = np.asarray(rho_bar, dtype=complex)
    if rho_bar.ndim != 3 or rho_bar.shape[1] != rho_bar.shape[2]:
        raise ValueError("rho_bar shape 必须为 (n_slices,N,N)")
    if not np.all(np.isfinite(rho_bar)):
        raise ValueError("rho_bar 必须全部有限")
    number_density_m3 = _validate_number_density(number_density_m3)

    d_q_bare = np.asarray(d_q_bare, dtype=complex)
    expected_operator_shape = (3, rho_bar.shape[1], rho_bar.shape[2])
    if d_q_bare.shape != expected_operator_shape:
        raise ValueError("d_q_bare shape 必须为 (3,N,N)，顺序 [q=-1,0,+1]")
    if not np.all(np.isfinite(d_q_bare)):
        raise ValueError("d_q_bare 必须全部有限")

    P_q = np.empty((len(rho_bar), 3), dtype=complex)
    for slice_index, rho in enumerate(rho_bar):
        P_q[slice_index], _ = density_matrix_to_polarization(
            rho,
            d_q_bare,
            number_density_m3,
        )
    P_cart = spherical_to_cartesian(P_q)

    return PolarizationProfile(
        rho_bar=rho_bar.copy(),
        P_q=P_q,
        P_cart=P_cart,
        number_density_m3=number_density_m3,
        d0_C_m=float(dipole_scale_from_spontaneous_emission()),
    )
def save_polarization_profile(profile, output_path):
    """把正式 Step 5 输出及 convention/单位元数据保存为压缩 NPZ。"""
    if not isinstance(profile, PolarizationProfile):
        raise TypeError("profile 必须是 PolarizationProfile")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        rho_bar=profile.rho_bar,
        P_q=profile.P_q,
        P_cart=profile.P_cart,
        number_density_m3=np.array(profile.number_density_m3),
        d0_C_m=np.array(profile.d0_C_m),
        phasor_convention=np.array(profile.phasor_convention),
        polarization_phasor_convention=np.array(
            profile.polarization_phasor_convention
        ),
        q_values=np.asarray(Q_VALUES, dtype=int),
        P_unit=np.array(P_UNIT),
        number_density_unit=np.array(NUMBER_DENSITY_UNIT),
        dipole_unit=np.array(DIPOLE_UNIT),
    )
    return output_path
