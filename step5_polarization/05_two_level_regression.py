# -*- coding: utf-8 -*-
"""测试 5：正式 polarization 接口回归两能级弱光解析响应。"""

import numpy as np
from scipy.constants import epsilon_0, hbar

import path_setup  # noqa: F401

from conventions import (
    dipole_scale_from_spontaneous_emission,
    ecal_amplitude_from_saturation,
)
from k39_model import GAMMA_D1
from liouvillian import (
    coherent_liouvillian,
    decay_liouvillian,
    rho_to_vec,
    vec_to_rho,
)
from polarization import density_matrix_to_polarization


def steady_state(L, dimension):
    matrix = np.asarray(L, dtype=complex).copy()
    rhs = np.zeros(dimension**2, dtype=complex)
    matrix[-1, :] = rho_to_vec(np.eye(dimension))
    rhs[-1] = 1.0
    return vec_to_rho(np.linalg.solve(matrix, rhs), dimension)


def main():
    print("测试 5：正式 rho -> P 接口的两能级弱光回归")
    gamma = GAMMA_D1
    d0 = dipole_scale_from_spontaneous_emission()
    number_density = 1.0e15
    saturation = 1.0e-8
    Ecal = ecal_amplitude_from_saturation(saturation)
    Omega = d0 * Ecal / hbar

    C = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    C_all = np.zeros((3, 2, 2), dtype=complex)
    C_all[1] = C
    L_decay = decay_liouvillian(C_all, gamma)

    detunings = gamma * np.linspace(-4.0, 4.0, 17)
    chi_from_formal_interface = []
    for delta in detunings:
        H = np.array(
            [[0.0, -Omega / 2.0], [-Omega / 2.0, -delta]],
            dtype=complex,
        )
        rho = steady_state(coherent_liouvillian(H) + L_decay, 2)
        P_q, _ = density_matrix_to_polarization(
            rho,
            C_all,
            number_density,
        )
        assert P_q[0] == 0.0 and P_q[2] == 0.0
        # χ 只在这个严格弱光回归测试中用于和解析解比较；正式模块不输出 χ。
        chi_from_formal_interface.append(P_q[1] / (epsilon_0 * Ecal))

    chi = np.asarray(chi_from_formal_interface)
    prefactor = number_density * d0**2 / (epsilon_0 * hbar)
    chi_analytic = prefactor * 1j / (gamma / 2.0 - 1j * detunings)
    relative_error = np.linalg.norm(chi - chi_analytic) / np.linalg.norm(
        chi_analytic
    )
    center = len(detunings) // 2

    print(f"relative chi error = {relative_error:.3e}")
    print(f"chi(0) = {chi[center]}")
    assert relative_error < 2e-8
    assert int(np.argmax(chi.imag)) == center
    assert chi[center].imag > 0.0
    assert chi[center + 2].real < 0.0
    assert chi[center - 2].real > 0.0
    print("PASS: 正式接口保持 Step 5.0 的 factor 2、符号、共轭和绝对幅值精度")


if __name__ == "__main__":
    main()

