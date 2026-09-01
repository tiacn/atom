# -*- coding: utf-8 -*-
"""第 2 小步：验证密度矩阵向量化和 coherent Liouvillian。"""

import numpy as np
from scipy.linalg import expm

from liouvillian import coherent_liouvillian, rho_to_vec, vec_to_rho


def demo_mixed_order_liouvillian(H):
    """复现 demo 的错误组合：列主序 index + 默认 C-order flatten。"""
    N = H.shape[0]
    L = np.zeros((N * N, N * N), dtype=complex)

    def column_index(i, j):
        return i + j * N

    for i in range(N):
        for j in range(N):
            for k in range(N):
                L[column_index(i, j), column_index(k, j)] += -1j * H[i, k]
                L[column_index(i, j), column_index(i, k)] += +1j * H[k, j]
    return L


def main():
    print("第 2 小步：密度矩阵向量化")

    H = np.array(
        [
            [0.0, 0.3 - 0.2j, 0.0],
            [0.3 + 0.2j, 1.0, 0.1],
            [0.0, 0.1, 2.0],
        ],
        dtype=complex,
    )
    rho = np.array(
        [
            [0.5, 0.1 + 0.04j, 0.0],
            [0.1 - 0.04j, 0.3, 0.02j],
            [0.0, -0.02j, 0.2],
        ],
        dtype=complex,
    )

    expected = -1j * (H @ rho - rho @ H)

    # 正确做法：列优先 vec 和列优先 L 配套。
    L = coherent_liouvillian(H)
    actual = vec_to_rho(L @ rho_to_vec(rho), len(H))
    correct_error = np.max(np.abs(actual - expected))

    # demo 的混用方式：L 按列优先构造，却输入 C-order flatten。
    L_mixed = demo_mixed_order_liouvillian(H)
    mixed_result = (L_mixed @ rho.flatten()).reshape(rho.shape)
    mixed_error = np.max(np.abs(mixed_result - expected))

    print(f"正确 F-order 误差 = {correct_error:.3e}")
    print(f"demo 混合顺序误差 = {mixed_error:.3e}")

    assert correct_error < 1e-13
    assert mixed_error > 1e-3

    # 再检查一个有限时间的纯幺正演化。
    dt = 0.2
    rho_t = vec_to_rho(expm(L * dt) @ rho_to_vec(rho), len(H))
    trace_error = abs(np.trace(rho_t) - np.trace(rho))
    hermiticity_error = np.max(np.abs(rho_t - rho_t.conj().T))

    assert trace_error < 1e-12
    assert hermiticity_error < 1e-12

    print(f"有限时间 trace error = {trace_error:.3e}")
    print(f"有限时间 Hermiticity error = {hermiticity_error:.3e}")
    print("PASS: 向量化规则已统一，确认并修正了 demo 的顺序 bug")


if __name__ == "__main__":
    main()

