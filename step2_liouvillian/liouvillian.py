# -*- coding: utf-8 -*-
"""与 pylcp 约定一致、但独立实现的 Liouvillian 工具。"""

from __future__ import annotations

import numpy as np

from k39_model import GAMMA_D1
from pylcp.common import cart2spherical


def rho_to_vec(rho):
    """密度矩阵 -> 列优先向量，index(i,j)=i+j*N。"""
    return np.asarray(rho, dtype=complex).reshape(-1, order="F")


def vec_to_rho(rho_vec, N):
    """列优先向量 -> 密度矩阵。"""
    return np.asarray(rho_vec, dtype=complex).reshape((N, N), order="F")


def coherent_liouvillian(H):
    """返回满足 L vec(rho)=vec(-i[H,rho]) 的矩阵。

    这里使用列优先 vec 公式：
        vec(H rho) = (I kron H) vec(rho)
        vec(rho H) = (H^T kron I) vec(rho)
    """
    H = np.asarray(H, dtype=complex)
    N = H.shape[0]
    identity = np.eye(N, dtype=complex)
    return -1j * (
        np.kron(identity, H)
        - np.kron(H.T, identity)
    )


def operator_to_superoperator(operator_fn, N):
    """通过作用在矩阵基 E_ij 上，构造任意线性超算符。

    这个写法速度不是最快，但定义直接，适合验证阶段，能避免手写四重索引。
    """
    result = np.zeros((N * N, N * N), dtype=complex)
    for j in range(N):
        for i in range(N):
            basis_matrix = np.zeros((N, N), dtype=complex)
            basis_matrix[i, j] = 1.0
            column = i + j * N
            result[:, column] = rho_to_vec(operator_fn(basis_matrix))
    return result


def decay_liouvillian(d_q_lowering, gamma=GAMMA_D1):
    """由三个 lowering operators 构造 Lindblad 自发辐射项。

    D[rho] = gamma * sum_q(
        C_q rho C_q^dagger
        - 1/2 {C_q^dagger C_q, rho}
    )
    """
    d_q_lowering = np.asarray(d_q_lowering, dtype=complex)
    N = d_q_lowering.shape[1]

    def dissipator(rho):
        drho = np.zeros_like(rho)
        for C in d_q_lowering:
            C_dag_C = C.conj().T @ C
            drho += C @ rho @ C.conj().T
            drho -= 0.5 * (C_dag_C @ rho + rho @ C_dag_C)
        return gamma * drho

    return operator_to_superoperator(dissipator, N)


def excited_state_projector(ham):
    """返回激发态流形投影算符 P_e。"""
    N_G = int(ham.ns[0])
    projector = np.zeros((ham.n, ham.n), dtype=complex)
    projector[N_G:, N_G:] = np.eye(ham.n - N_G)
    return projector


def detuning_liouvillian(ham, detuning_rad_s):
    """旋转框架失谐项 L_delta = -i[-delta*P_e, rho]。"""
    H_detuning = -float(detuning_rad_s) * excited_state_projector(ham)
    return coherent_liouvillian(H_detuning)


def physical_hamiltonian_from_fields(
    ham,
    E_q,
    B_cart,
    detuning_rad_s=0.0,
):
    """按 pylcp OBE 的缩放约定组装恒定场 Hamiltonian。

    E_q 是 pylcp 的无量纲球基底电场，顺序为 [-1,0,+1]。
    对饱和参数 s，pylcp 使用 |E_q| 总幅度 sqrt(2*s)。

    pylcp OBE 中的光耦合矩阵使用 gamma*d_q/4，因此这里也使用 gamma/4。
    """
    E_q = np.asarray(E_q, dtype=complex)
    B_q = cart2spherical(np.asarray(B_cart, dtype=float))

    H = np.asarray(ham.H_0, dtype=complex).copy()

    # 旋转框架失谐。它通过对易子作用于 g-e 光学相干，
    # 不直接给 excited-state populations 添加衰减或复相位。
    H -= float(detuning_rad_s) * excited_state_projector(ham)

    # Zeeman: H_B = -sum_q mu_q * B_q^*
    for q_index in range(3):
        H -= ham.mu_q[q_index] * np.conjugate(B_q[q_index])

    d_bare = ham.d_q_bare["g->e"]
    d_star = ham.d_q_star["g->e"]

    # 球张量标量积：sum_q (-1)^q d_q E_{-q}
    for q_index, q in enumerate((-1, 0, 1)):
        phase = (-1.0) ** q
        E_minus_q = E_q[2 - q_index]
        H -= (
            GAMMA_D1
            / 4.0
            * phase
            * (
                d_bare[q_index] * E_minus_q
                + d_star[q_index] * np.conjugate(E_minus_q)
            )
        )

    if not np.allclose(H, H.conj().T, atol=1e-8):
        raise RuntimeError("组装出的 Hamiltonian 不是 Hermitian")
    return H


def build_direct_L(ham, E_q, B_cart, detuning_rad_s=0.0):
    """完全从旋转框架 H 和 Lindblad 公式直接构造 L。"""
    H = physical_hamiltonian_from_fields(
        ham,
        E_q,
        B_cart,
        detuning_rad_s=detuning_rad_s,
    )
    L_coherent = coherent_liouvillian(H)
    L_decay = decay_liouvillian(ham.d_q_bare["g->e"], GAMMA_D1)
    return L_coherent + L_decay


def build_pylcp_component_L(obe, position=None, time=0.0):
    """按照 pylcp.__drhodt 使用的预计算分量组装参考 L。

    这里读取 pylcp 已经公开存放在 obe.ev_mat 中的矩阵，但不调用 demo。
    """
    if position is None:
        position = np.zeros(3)

    L = (
        np.asarray(obe.ev_mat["H0"], dtype=complex)
        + np.asarray(obe.ev_mat["decay"], dtype=complex)
    )

    for key in obe.laserBeams:
        E_q = obe.laserBeams[key].total_electric_field(position, time)
        for q_index, q in enumerate((-1, 0, 1)):
            E_minus_q = E_q[2 - q_index]
            if abs(E_minus_q) > 1e-14:
                phase = (-1.0) ** q
                L -= (
                    phase
                    * E_minus_q
                    * np.asarray(obe.ev_mat["d_q"][key][q_index])
                )
                L -= (
                    phase
                    * np.conjugate(E_minus_q)
                    * np.asarray(obe.ev_mat["d_q*"][key][q_index])
                )

    B_q = cart2spherical(obe.magField.Field(position, time))
    for q_index in range(3):
        if abs(B_q[q_index]) > 1e-14:
            L -= (
                np.asarray(obe.ev_mat["B"][q_index])
                * np.conjugate(B_q[q_index])
            )

    return L
