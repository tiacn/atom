# -*- coding: utf-8 -*-
"""第 1 小步：只检查 K39 D1 原子模型。"""

import numpy as np

from k39_model import (
    A_HFS_EXCITED_HZ,
    A_HFS_GROUND_HZ,
    build_k39_d1_hamiltonian,
)


def manifold_splitting_MHz(H):
    energies = np.linalg.eigvalsh(H) / (2.0 * np.pi * 1e6)
    return float(energies.max() - energies.min())


def main():
    ham, basis_g, basis_e = build_k39_d1_hamiltonian()

    print("第 1 小步：K39 D1 原子模型")
    print(f"ground states = {len(basis_g)}")
    print(f"excited states = {len(basis_e)}")
    print(f"total states = {ham.n}")

    H_g = ham.H_0[: len(basis_g), : len(basis_g)]
    H_e = ham.H_0[len(basis_g) :, len(basis_g) :]
    ground_split = manifold_splitting_MHz(H_g)
    excited_split = manifold_splitting_MHz(H_e)

    print(f"ground hyperfine splitting = {ground_split:.6f} MHz")
    print(f"excited hyperfine splitting = {excited_split:.6f} MHz")

    d_q = ham.d_q_bare["g->e"][:, : len(basis_g), len(basis_g) :]

    # 每个 excited state 的所有自发辐射通道概率之和应为 1。
    column_strength = np.sum(np.abs(d_q) ** 2, axis=(0, 1))
    normalization_error = np.max(np.abs(column_strength - 1.0))

    # pylcp 这个 lowering operator 的非零元满足 m_g = m_e + q。
    selection_rule_error = 0
    nonzero_count = 0
    for q_index, q in enumerate((-1, 0, 1)):
        for g in range(len(basis_g)):
            for e in range(len(basis_e)):
                if abs(d_q[q_index, g, e]) > 1e-12:
                    nonzero_count += 1
                    m_g = basis_g[g, 1]
                    m_e = basis_e[e, 1]
                    if abs(m_g - (m_e + q)) > 1e-12:
                        selection_rule_error += 1

    assert ham.n == 16
    # 对 I=3/2、J=1/2，F=1 与 F=2 的间隔为 2A。断言直接从模型常数
    # 推导，避免再次保留与 k39_model.py 不一致的旧硬编码。
    expected_ground_split = 2.0 * A_HFS_GROUND_HZ / 1e6
    expected_excited_split = 2.0 * A_HFS_EXCITED_HZ / 1e6
    assert abs(ground_split - expected_ground_split) < 1e-9
    assert abs(excited_split - expected_excited_split) < 1e-9
    assert normalization_error < 1e-12
    assert selection_rule_error == 0

    print(f"nonzero dipole elements = {nonzero_count}")
    print(f"dipole normalization max error = {normalization_error:.3e}")
    print(f"selection-rule violations = {selection_rule_error}")
    print("PASS: 原子维数、超精细分裂、偶极矩方向和选择规则正确")


if __name__ == "__main__":
    main()
