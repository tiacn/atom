# -*- coding: utf-8 -*-
"""测试 2：人工 rho_eg 与逐项手算完全一致。"""

import numpy as np

import path_setup  # noqa: F401

from conventions import Q_VALUES, dipole_scale_from_spontaneous_emission
from k39_model import build_k39_d1_hamiltonian
from polarization import density_matrix_to_polarization


def main():
    print("测试 2：单个 rho_eg 的 q 分量、索引和 factor 2")
    ham, _, _ = build_k39_d1_hamiltonian()
    C_all = ham.d_q_bare["g->e"]
    q_index, g, e = (int(value) for value in np.argwhere(np.abs(C_all) > 1e-12)[0])
    assert np.count_nonzero(np.abs(C_all[:, g, e]) > 1e-12) == 1

    rho = np.zeros((ham.n, ham.n), dtype=complex)
    rho_eg = 0.137 - 0.219j
    rho[e, g] = rho_eg
    number_density = 1.2345e16
    d0 = dipole_scale_from_spontaneous_emission()

    P_q, _ = density_matrix_to_polarization(rho, C_all, number_density)
    manual = np.array(
        [2.0 * number_density * d0 * C_q[g, e] * rho_eg for C_q in C_all]
    )

    print(f"selected q = {Q_VALUES[q_index]:+d}")
    print(f"C_q[g,e] = {C_all[q_index, g, e]}")
    print(f"rho[e,g] = {rho_eg}")
    print(f"P_q = {P_q} C/m^2")
    assert np.max(np.abs(P_q - manual)) < 1e-26
    assert abs(P_q[q_index]) > 1e-14
    assert np.count_nonzero(np.abs(P_q) > 1e-20) == 1
    print("PASS: 只有正确 q 非零，rho_eg、绝对尺度和 factor 2 均匹配手算")


if __name__ == "__main__":
    main()

