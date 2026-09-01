# -*- coding: utf-8 -*-
"""测试 1：只有 population、没有 optical coherence 时 P 严格为零。"""

import numpy as np

import path_setup  # noqa: F401

from k39_model import build_k39_d1_hamiltonian
from polarization import density_matrix_to_polarization, k39_number_density


def main():
    print("测试 1：population-only rho -> P=0")
    ham, _, _ = build_k39_d1_hamiltonian()
    populations = np.arange(1, ham.n + 1, dtype=float)
    populations /= populations.sum()
    rho = np.diag(populations).astype(complex)
    density = k39_number_density(70.0)

    P_q, P_cart = density_matrix_to_polarization(
        rho,
        ham.d_q_bare["g->e"],
        density.k39_m3,
    )
    print(f"N_K39 = {density.k39_m3:.9e} m^-3")
    print(f"P_q = {P_q} C/m^2")
    assert np.array_equal(P_q, np.zeros(3, dtype=complex))
    assert np.array_equal(P_cart, np.zeros(3, dtype=complex))
    print("PASS: population 不会被误当作 optical coherence")


if __name__ == "__main__":
    main()

