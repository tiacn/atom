# -*- coding: utf-8 -*-
"""测试 4：两能级弱光响应锁定 P^(+) 的共轭、符号和 factor 2。"""

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


def steady_state(L, dimension):
    matrix = np.asarray(L, dtype=complex).copy()
    rhs = np.zeros(dimension**2, dtype=complex)
    matrix[-1, :] = rho_to_vec(np.eye(dimension))
    rhs[-1] = 1.0
    return vec_to_rho(np.linalg.solve(matrix, rhs), dimension)


def main():
    print("测试 4：两能级弱光 detuning response")
    gamma = GAMMA_D1
    d0 = dipole_scale_from_spontaneous_emission()
    number_density = 1.0e15
    saturation = 1.0e-8
    Ecal = ecal_amplitude_from_saturation(saturation)
    Omega = d0 * Ecal / hbar

    C = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    C_all = np.zeros((3, 2, 2), dtype=complex)
    C_all[1] = C  # q=0，避免圆偏振名称参与这个绝对符号测试。
    L_decay = decay_liouvillian(C_all, gamma)

    detunings = gamma * np.linspace(-4.0, 4.0, 17)
    mu_lowering = []
    mu_raising = []
    excited = []
    for delta in detunings:
        H = np.array(
            [[0.0, -Omega / 2.0], [-Omega / 2.0, -delta]],
            dtype=complex,
        )
        rho = steady_state(coherent_liouvillian(H) + L_decay, 2)
        mu_lowering.append(np.trace(C @ rho))
        mu_raising.append(np.trace(C.conj().T @ rho))
        excited.append(rho[1, 1].real)

    mu_lowering = np.asarray(mu_lowering)
    mu_raising = np.asarray(mu_raising)
    excited = np.asarray(excited)

    prefactor = number_density * d0**2 / (epsilon_0 * hbar)
    chi_analytic = prefactor * 1j / (gamma / 2.0 - 1j * detunings)
    candidates = {
        "2*N*d0*Tr(C*rho)": 2.0 * number_density * d0 * mu_lowering / (epsilon_0 * Ecal),
        "N*d0*Tr(C*rho)": number_density * d0 * mu_lowering / (epsilon_0 * Ecal),
        "2*N*d0*Tr(Cdag*rho)": 2.0 * number_density * d0 * mu_raising / (epsilon_0 * Ecal),
        "-2*N*d0*Tr(C*rho)": -2.0 * number_density * d0 * mu_lowering / (epsilon_0 * Ecal),
    }
    scores = {
        name: float(np.linalg.norm(value - chi_analytic) / np.linalg.norm(chi_analytic))
        for name, value in candidates.items()
    }
    chi = candidates["2*N*d0*Tr(C*rho)"]

    for name, score in scores.items():
        print(f"candidate {name}: relative error = {score:.3e}")

    resonance_index = int(np.argmax(chi.imag))
    center = len(detunings) // 2
    real_odd_error = np.max(np.abs(chi.real + chi.real[::-1]))
    imag_even_error = np.max(np.abs(chi.imag - chi.imag[::-1]))
    resonance_expected = 2.0 * number_density * d0**2 / (
        epsilon_0 * hbar * gamma
    )

    print(f"resonance detuning/Gamma = {detunings[resonance_index]/gamma:+.3f}")
    print(f"chi(0) = {chi[center]}")
    print(f"analytic Im[chi(0)] = {resonance_expected:.9e}")
    print(f"Re odd error = {real_odd_error:.3e}")
    print(f"Im even error = {imag_even_error:.3e}")
    print(f"max excited population = {np.max(excited):.3e}")

    assert scores["2*N*d0*Tr(C*rho)"] < 2e-8
    assert 0.49 < scores["N*d0*Tr(C*rho)"] < 0.51
    assert scores["2*N*d0*Tr(Cdag*rho)"] > 1.0
    assert scores["-2*N*d0*Tr(C*rho)"] > 1.9
    assert resonance_index == center
    assert abs(chi[center].real) < 1e-12 * chi[center].imag
    assert chi[center].imag > 0.0  # exp(+ikz) 下 Im(chi)>0 表示衰减。
    assert chi[center + 2].real < 0.0  # delta=omega_L-omega_0 > 0。
    assert chi[center - 2].real > 0.0
    assert real_odd_error < 1e-16
    assert imag_even_error < 1e-16
    assert abs(chi[center].imag / resonance_expected - 1.0) < 2e-8
    assert np.max(excited) < 1e-8
    print("PASS: rho_eg、正号和 factor 2 唯一匹配标准弱光解析响应")


if __name__ == "__main__":
    main()
