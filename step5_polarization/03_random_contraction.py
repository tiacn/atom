# -*- coding: utf-8 -*-
"""测试 3：随机 Hermitian rho 的显式求和与 trace contraction 一致。"""

import numpy as np

import path_setup  # noqa: F401

from conventions import dipole_scale_from_spontaneous_emission
from k39_model import build_k39_d1_hamiltonian
from polarization import density_matrix_to_polarization


def explicit_trace_contraction(C_q, rho):
    total = 0.0j
    for row in range(len(rho)):
        for column in range(len(rho)):
            total += C_q[row, column] * rho[column, row]
    return total


def main():
    print("测试 3：随机 Hermitian density matrix contraction")
    rng = np.random.default_rng(50301)
    ham, _, _ = build_k39_d1_hamiltonian()
    C_all = ham.d_q_bare["g->e"]
    raw = rng.normal(size=(ham.n, ham.n)) + 1j * rng.normal(
        size=(ham.n, ham.n)
    )
    rho = raw @ raw.conj().T
    rho /= np.trace(rho)
    number_density = 3.21e15
    d0 = dipole_scale_from_spontaneous_emission()

    P_q, _ = density_matrix_to_polarization(rho, C_all, number_density)
    explicit_mu = np.array(
        [explicit_trace_contraction(C_q, rho) for C_q in C_all]
    )
    trace_mu = np.array([np.trace(C_q @ rho) for C_q in C_all])
    explicit_P = 2.0 * number_density * d0 * explicit_mu

    mu_error = np.max(np.abs(explicit_mu - trace_mu))
    P_error = np.max(np.abs(P_q - explicit_P))
    print(f"max explicit-vs-trace mu error = {mu_error:.3e}")
    print(f"max explicit-vs-module P error = {P_error:.3e} C/m^2")
    assert np.max(np.abs(rho - rho.conj().T)) < 1e-15
    assert mu_error < 1e-16
    assert P_error < 1e-28
    print("PASS: 未出现索引、转置或共轭偏差")


if __name__ == "__main__":
    main()

