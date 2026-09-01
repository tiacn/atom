# -*- coding: utf-8 -*-
"""测试 5：K39 D1 的 Cartesian/spherical、Jones vector 与 Delta-m 映射。"""

import numpy as np

import path_setup  # noqa: F401

from conventions import (
    ecal_amplitude_from_saturation,
    normalized_pylcp_Eq_from_si_phasor,
    polarization_phasor_spherical,
)
from k39_model import build_k39_d1_hamiltonian
from pylcp.common import cart2spherical, spherical2cart


def driven_lowering_q(Eq_py):
    result = []
    for q_index, q in enumerate((-1, 0, 1)):
        if abs(Eq_py[2 - q_index]) > 1e-12:
            result.append(q)
    return result


def allowed_delta_m(C_q, basis_g, basis_e):
    values = set()
    for g, e in np.argwhere(np.abs(C_q) > 1e-12):
        values.add(float(basis_e[e, 1] - basis_g[g, 1]))
    return values


def main():
    print("测试 5：明确 Jones vector 的 Delta-m 映射")
    ham, basis_g, basis_e = build_k39_d1_hamiltonian()
    N_G = len(basis_g)
    C = ham.d_q_bare["g->e"][:, :N_G, N_G:]
    saturation = 0.02
    E0 = ecal_amplitude_from_saturation(saturation)

    cases = {
        "x": np.array([1.0, 0.0, 0.0], dtype=complex),
        "(x+i*y)/sqrt(2)": np.array([1.0, 1.0j, 0.0]) / np.sqrt(2.0),
        "(x-i*y)/sqrt(2)": np.array([1.0, -1.0j, 0.0]) / np.sqrt(2.0),
    }

    mappings = {}
    for label, jones in cases.items():
        Ecal_q = cart2spherical(jones)
        Eq_py = normalized_pylcp_Eq_from_si_phasor(E0 * jones)
        Eq_py_unit = Eq_py / np.sqrt(2.0 * saturation)
        lowering_q = driven_lowering_q(Eq_py_unit)
        delta_m = sorted(
            set().union(
                *(allowed_delta_m(C[q + 1], basis_g, basis_e) for q in lowering_q)
            )
        )
        mappings[label] = (Ecal_q, Eq_py_unit, lowering_q, delta_m)
        print(f"{label}:")
        print(f"  SI positive-frequency Ecal_q = {Ecal_q}")
        print(f"  PyLCP negative-frequency E_q = {Eq_py_unit}")
        print(f"  driven lowering C_q = {lowering_q}")
        print(f"  absorption Delta-m=m_e-m_g = {delta_m}")

    assert np.allclose(mappings["(x+i*y)/sqrt(2)"][0], [1.0, 0.0, 0.0])
    assert np.allclose(mappings["(x+i*y)/sqrt(2)"][1], [0.0, 0.0, -1.0])
    assert mappings["(x+i*y)/sqrt(2)"][2] == [-1]
    assert mappings["(x+i*y)/sqrt(2)"][3] == [1.0]

    assert np.allclose(mappings["(x-i*y)/sqrt(2)"][0], [0.0, 0.0, -1.0])
    assert np.allclose(mappings["(x-i*y)/sqrt(2)"][1], [1.0, 0.0, 0.0])
    assert mappings["(x-i*y)/sqrt(2)"][2] == [1]
    assert mappings["(x-i*y)/sqrt(2)"][3] == [-1.0]
    assert mappings["x"][2] == [-1, 1]
    assert mappings["x"][3] == [-1.0, 1.0]

    # 用一个 q=-1 lowering coherence 检查输出 P_q 与 SI 输入 Jones 的
    # 同一球分量相对应，不需要 left/right 名称。
    rho = np.zeros((ham.n, ham.n), dtype=complex)
    q_index = 0
    g, e_local = np.argwhere(np.abs(C[q_index]) > 1e-12)[0]
    e_global = N_G + int(e_local)
    rho[e_global, int(g)] = 0.2 + 0.1j
    P_q = polarization_phasor_spherical(
        ham.d_q_bare["g->e"], rho, number_density_m3=1.0
    )
    P_cart = spherical2cart(P_q)
    assert abs(P_q[0]) > 0.0
    assert abs(P_q[1]) == 0.0
    assert abs(P_q[2]) == 0.0
    assert np.allclose(cart2spherical(P_cart), P_q)
    print(f"artificial q=-1 coherence -> Pcal_q = {P_q}")
    print("PASS: Cartesian/spherical 相位、纯 q 分量与 Delta-m 映射明确")


if __name__ == "__main__":
    main()
