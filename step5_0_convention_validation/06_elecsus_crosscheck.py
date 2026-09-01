# -*- coding: utf-8 -*-
"""测试 6：极简冷原子、弱光 K39 D1 与 ElecSus susceptibility 交叉检查。"""

import numpy as np
from scipy.constants import epsilon_0, hbar

import path_setup  # noqa: F401

import spectra
from conventions import dipole_scale_from_spontaneous_emission
from k39_model import GAMMA_D1, build_k39_d1_hamiltonian
from numberDensityEqs import numDenK


def main():
    print("测试 6：K39 D1 与 ElecSus 极简弱光交叉检查")
    ham, basis_g, basis_e = build_k39_d1_hamiltonian()
    N_G = len(basis_g)
    H_g_MHz = np.diag(ham.H_0[:N_G, :N_G]).real / (2.0 * np.pi * 1e6)
    H_e_MHz = np.diag(ham.H_0[N_G:, N_G:]).real / (2.0 * np.pi * 1e6)
    C_minus = ham.d_q_bare["g->e"][0, :N_G, N_G:]

    detuning_MHz = np.arange(-350.0, 351.0, 2.0)
    number_density = numDenK(293.15)
    d0 = dipole_scale_from_spontaneous_emission()
    chi_direct = np.zeros(detuning_MHz.shape, dtype=complex)

    # ElecSus K39_D1.IsotopeShift=+15.864 MHz 被加到 ground manifold；
    # 因此它的外部 detuning X 对应当前 Hamiltonian 的 X+15.864 MHz。
    internal_axis_MHz = detuning_MHz + 15.864
    for g in range(N_G):
        for e in range(len(basis_e)):
            line_strength = abs(C_minus[g, e]) ** 2 / N_G
            transition_MHz = H_e_MHz[e] - H_g_MHz[g]
            delta = 2.0 * np.pi * 1e6 * (
                internal_axis_MHz - transition_MHz
            )
            chi_direct += (
                number_density
                * d0**2
                / (epsilon_0 * hbar)
                * line_strength
                * 1j
                / (GAMMA_D1 / 2.0 - 1j * delta)
            )

    params = {
        "Elem": "K",
        "Dline": "D1",
        "Bfield": 0.0,
        "T": 20.0,
        "GammaBuf": 0.0,
        "shift": 0.0,
        "DoppTemp": -273.149,  # 约 1 mK，仅用于关闭本交叉检查的 Doppler 展宽。
        "Constrain": False,
        "K40frac": 0.0,
        "K41frac": 0.0,
        "BoltzmannFactor": False,
    }
    chi_plus, chi_minus, _ = spectra.calc_chi(detuning_MHz, params)

    # B~0 时两个圆分量应相同；这里只比较单个分量，不使用 left/right
    # 名称决定 q 映射。ElecSus 会把跃迁中心取整到 MHz，且原子常数有舍入，
    # 所以这是数量级、吸收符号和整体线形检查，不是逐点 benchmark。
    circular_difference = np.linalg.norm(chi_plus - chi_minus) / np.linalg.norm(chi_plus)
    norm_ratio = np.linalg.norm(chi_plus) / np.linalg.norm(chi_direct)
    correlation = abs(np.vdot(chi_direct, chi_plus)) / (
        np.linalg.norm(chi_direct) * np.linalg.norm(chi_plus)
    )
    peak_direct = detuning_MHz[int(np.argmax(chi_direct.imag))]
    peak_elecsus = detuning_MHz[int(np.argmax(chi_plus.imag))]

    print(f"number density = {number_density:.9e} m^-3")
    print(f"direct/ElecSus absorption peak = {peak_direct:+.1f}/{peak_elecsus:+.1f} MHz")
    print(f"||chi_ElecSus||/||chi_direct|| = {norm_ratio:.6f}")
    print(f"complex line-shape correlation = {correlation:.6f}")
    print(f"ElecSus two circular components relative difference = {circular_difference:.3e}")
    print(f"peak Im chi direct/ElecSus = {np.max(chi_direct.imag):.6e}/{np.max(chi_plus.imag):.6e}")

    assert abs(peak_direct - peak_elecsus) <= 2.0
    assert 0.85 < norm_ratio < 1.15
    assert correlation > 0.98
    assert circular_difference < 1e-5
    assert np.max(chi_direct.imag) > 0.0
    assert np.max(chi_plus.imag) > 0.0
    print("PASS: ElecSus 单偏振的共振、吸收符号、复线形和绝对量级一致")


if __name__ == "__main__":
    main()
