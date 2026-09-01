# -*- coding: utf-8 -*-
"""Step 5.0 唯一使用的 SI phasor、场强和极化约定。"""

from __future__ import annotations

import numpy as np
from scipy.constants import c, epsilon_0, hbar

import path_setup  # noqa: F401

from k39_model import GAMMA_D1, LAMBDA_D1
from pylcp.common import cart2spherical, spherical2cart


Q_VALUES = (-1, 0, 1)

# 全部 SI 正频率量使用：
#   E_phys(r,t) = Re[Ecal(r) exp(-i omega t)]
#   Ecal(r) = Ecal_0 exp(+i k z)
# 因此 E_phys = (Ecal exp(-i omega t) + c.c.)/2。
PHASOR_CONVENTION = "E_phys = Re[Ecal_0 exp(i*k*z - i*omega*t)]"


def transition_omega(lambda_m=LAMBDA_D1):
    return 2.0 * np.pi * c / float(lambda_m)


def dipole_scale_from_spontaneous_emission(
    gamma_rad_s=GAMMA_D1,
    lambda_m=LAMBDA_D1,
):
    """PyLCP normalized C_q 对应的绝对偶极矩尺度 d0，单位 C m。"""
    omega = transition_omega(lambda_m)
    return np.sqrt(
        3.0
        * np.pi
        * epsilon_0
        * hbar
        * c**3
        * float(gamma_rad_s)
        / omega**3
    )


def saturation_intensity_W_m2(
    gamma_rad_s=GAMMA_D1,
    lambda_m=LAMBDA_D1,
):
    """PyLCP transition.Isat 的 SI 形式，单位 W/m^2。"""
    omega = transition_omega(lambda_m)
    return hbar * omega**3 * float(gamma_rad_s) / (12.0 * np.pi * c**2)


def ecal_amplitude_from_saturation(saturation):
    """由 s=I/Isat 返回 SI 正频率 phasor 峰值 |Ecal|，单位 V/m。"""
    saturation = float(saturation)
    if saturation < 0.0:
        raise ValueError("saturation 不能为负")
    intensity = saturation * saturation_intensity_W_m2()
    return np.sqrt(2.0 * intensity / (c * epsilon_0))


def saturation_from_ecal(Ecal_cart_V_m):
    """由 SI 正频率 Cartesian phasor 反算 s。"""
    Ecal = np.asarray(Ecal_cart_V_m, dtype=complex)
    intensity = 0.5 * c * epsilon_0 * np.vdot(Ecal, Ecal).real
    return float(intensity / saturation_intensity_W_m2())


def normalized_pylcp_Eq_from_si_phasor(Ecal_cart_V_m):
    """SI 正频率 phasor -> PyLCP normalized E_q，顺序 [-1,0,+1]。

    PyLCP fields.py 返回的驱动具有 exp(-i k.r + i delta*t)，对应本文件
    正频率 phasor 的负频率（复共轭）包络。因此这里先共轭 Cartesian
    Ecal，再转球基底。这个共轭对于圆偏振不能省略。
    """
    Ecal = np.asarray(Ecal_cart_V_m, dtype=complex)
    d0 = dipole_scale_from_spontaneous_emission()
    return (
        2.0
        * d0
        / (hbar * GAMMA_D1)
        * cart2spherical(np.conjugate(Ecal))
    )


def si_phasor_from_normalized_pylcp_Eq(Eq_pylcp):
    """normalized PyLCP E_q -> SI 正频率 Cartesian phasor [V/m]。"""
    Eq_pylcp = np.asarray(Eq_pylcp, dtype=complex)
    d0 = dipole_scale_from_spontaneous_emission()
    negative_frequency_cart = spherical2cart(Eq_pylcp)
    return np.conjugate(
        hbar * GAMMA_D1 / (2.0 * d0) * negative_frequency_cart
    )


def lowering_contractions(d_q_bare, rho):
    """返回 mu_q=Tr(C_q rho)，顺序 [-1,0,+1]。"""
    d_q_bare = np.asarray(d_q_bare, dtype=complex)
    rho = np.asarray(rho, dtype=complex)
    return np.array([np.trace(C_q @ rho) for C_q in d_q_bare])


def polarization_phasor_spherical(d_q_bare, rho, number_density_m3):
    """候选 SI 正频率极化 phasor Pcal_q，单位 C/m^2。

    rho 使用与冻结 OBE 相同的旋转框架。C_q=|g><e|，所以
    Tr(C_q rho)=rho_eg 是随 exp(-i omega t) 的 lowering coherence。
    factor 2 来自 E_phys/P_phys 都写成 Re[phasor exp(-i omega t)]。
    """
    d0 = dipole_scale_from_spontaneous_emission()
    return (
        2.0
        * float(number_density_m3)
        * d0
        * lowering_contractions(d_q_bare, rho)
    )
