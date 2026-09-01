# -*- coding: utf-8 -*-
"""测试 2：验证 d_q_bare lowering block 与 rho_eg contraction。"""

import numpy as np

import path_setup  # noqa: F401

from conventions import lowering_contractions
from k39_model import build_k39_d1_hamiltonian


def main():
    print("测试 2：lowering operator 与人工 rho contraction")
    ham, basis_g, basis_e = build_k39_d1_hamiltonian()
    N_G = len(basis_g)
    C_all = ham.d_q_bare["g->e"]
    C_dag_all = ham.d_q_star["g->e"]

    assert np.count_nonzero(np.abs(C_all[:, N_G:, :]) > 1e-14) == 0
    assert np.count_nonzero(np.abs(C_all[:, :, :N_G]) > 1e-14) == 0
    assert np.allclose(C_dag_all, np.conjugate(np.swapaxes(C_all, 1, 2)))

    nonzero = np.argwhere(np.abs(C_all) > 1e-12)[0]
    q_index, g, e_global = (int(value) for value in nonzero)
    coefficient = C_all[q_index, g, e_global]

    # 人工矩阵只设置 rho_eg（row=e, column=g）；不要求它本身是物理态，
    # 目的是让索引、转置和共轭错误无法被 Hermiticity 掩盖。
    rho = np.zeros((ham.n, ham.n), dtype=complex)
    rho_eg = 0.137 - 0.219j
    rho[e_global, g] = rho_eg

    contractions = lowering_contractions(C_all, rho)
    direct = np.trace(C_all[q_index] @ rho)
    manual = coefficient * rho_eg
    wrong_rho_ge = coefficient * rho[g, e_global]

    print(f"chosen q = {(-1, 0, 1)[q_index]:+d}")
    print(f"C_q[g,e] = {coefficient}")
    print(f"rho_eg = rho[{e_global},{g}] = {rho_eg}")
    print(f"Tr(C_q rho) = {direct}")
    print(f"manual C_q[g,e]*rho[e,g] = {manual}")
    print(f"wrong rho_ge contraction = {wrong_rho_ge}")

    assert abs(direct - manual) < 1e-15
    assert abs(contractions[q_index] - manual) < 1e-15
    assert abs(manual) > 1e-3
    assert wrong_rho_ge == 0.0
    print("PASS: d_q_bare=|g><e|，Tr(C_q rho) 精确选取 rho_eg")


if __name__ == "__main__":
    main()
