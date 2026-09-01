# -*- coding: utf-8 -*-
"""测试 3：Gamma -> d0 -> Isat -> Ecal -> Omega -> normalized E_q。"""

import numpy as np
from scipy.constants import c, elementary_charge, epsilon_0, hbar, physical_constants

import path_setup  # noqa: F401

import pylcp
from AtomConstants import KD1Transition
from k39_model import GAMMA_D1, LAMBDA_D1

from conventions import (
    dipole_scale_from_spontaneous_emission,
    ecal_amplitude_from_saturation,
    normalized_pylcp_Eq_from_si_phasor,
    saturation_from_ecal,
    saturation_intensity_W_m2,
    transition_omega,
)


def main():
    print("测试 3：绝对单位链")
    omega = transition_omega()
    d0 = dipole_scale_from_spontaneous_emission()
    atomic_dipole_unit = elementary_charge * physical_constants["Bohr radius"][0]

    # 自发辐射反算：Gamma = omega^3*d0^2/(3*pi*eps0*hbar*c^3)。
    gamma_back = omega**3 * d0**2 / (
        3.0 * np.pi * epsilon_0 * hbar * c**3
    )
    gamma_error = abs(gamma_back / GAMMA_D1 - 1.0)

    atom = pylcp.atom("39K")
    pylcp_transition = atom.transition[0]
    Isat_formula = saturation_intensity_W_m2()
    Isat_pylcp = pylcp_transition.Isat * 10.0  # mW/cm^2 -> W/m^2
    Isat_error = abs(Isat_formula / Isat_pylcp - 1.0)

    # ElecSus dipoleStrength 是电子 reduced dipole；其线强中另有 1/3。
    # PyLCP normalized C_q 的 d0 应对应 dipoleStrength/sqrt(3)。
    d0_elecsus = KD1Transition.dipoleStrength / np.sqrt(3.0)
    elecsus_error = abs(d0_elecsus / d0 - 1.0)

    saturation = 0.037
    E0 = ecal_amplitude_from_saturation(saturation)
    Ecal = E0 * np.array([1.0, 0.0, 0.0], dtype=complex)
    Eq_py = normalized_pylcp_Eq_from_si_phasor(Ecal)
    s_back = saturation_from_ecal(Ecal)

    omega_rabi_si = d0 * E0 / hbar
    omega_rabi_s = GAMMA_D1 * np.sqrt(saturation / 2.0)
    omega_rabi_py = GAMMA_D1 * np.linalg.norm(Eq_py) / 2.0

    # 对单位 Clebsch 的两能级，PyLCP H_ge=-Gamma*E_q/4=-Omega*/2。
    H_ge_pylcp = -GAMMA_D1 * Eq_py[0] / 4.0
    H_ge_expected = -omega_rabi_si / (2.0 * np.sqrt(2.0))

    # 数值平均 E_phys^2，直接验证 Re[...] convention 的 1/2。
    phases = np.linspace(0.0, 2.0 * np.pi, 20000, endpoint=False)
    E_phys = np.real(E0 * np.exp(-1j * phases))
    mean_square = np.mean(E_phys**2)

    print(f"Gamma = {GAMMA_D1:.9e} s^-1")
    print(f"lambda = {LAMBDA_D1*1e9:.12f} nm")
    print(
        f"d0 = {d0:.9e} C m = {d0/3.33564e-30:.6f} D "
        f"= {d0/atomic_dipole_unit:.6f} e*a0"
    )
    print(f"ElecSus dipoleStrength/sqrt(3) = {d0_elecsus:.9e} C m")
    print(f"Isat = {Isat_formula:.9f} W/m^2 = {Isat_formula/10.0:.9f} mW/cm^2")
    print(f"s={saturation}: |Ecal|={E0:.9f} V/m, |E_q^Py|={np.linalg.norm(Eq_py):.9f}")
    print(f"Omega SI/Py/s formula = {omega_rabi_si:.9e}/{omega_rabi_py:.9e}/{omega_rabi_s:.9e} rad/s")
    print(f"<E_phys^2>/(|Ecal|^2/2) = {mean_square/(E0**2/2.0):.12f}")

    assert gamma_error < 1e-14
    assert Isat_error < 1e-13
    assert elecsus_error < 5e-5  # ElecSus 的 Gamma 和 lambda 被舍入。
    assert abs(s_back / saturation - 1.0) < 1e-14
    assert abs(np.linalg.norm(Eq_py) / np.sqrt(2.0 * saturation) - 1.0) < 1e-14
    assert abs(omega_rabi_si / omega_rabi_s - 1.0) < 1e-14
    assert abs(omega_rabi_py / omega_rabi_si - 1.0) < 1e-14
    assert abs(H_ge_pylcp - H_ge_expected) < 1e-8
    assert abs(mean_square / (E0**2 / 2.0) - 1.0) < 1e-14
    print("PASS: d0、Isat、Ecal、s、Omega、sqrt(2s) 和全部 factor 2 一致")


if __name__ == "__main__":
    main()
