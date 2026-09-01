# -*- coding: utf-8 -*-
"""第 4 小步：比较 expm(L*t) 和 pylcp 参考 L 的 ODE 积分。"""

import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

from k39_model import build_reference_obe, thermal_ground_state
from liouvillian import (
    build_direct_L,
    build_pylcp_component_L,
    rho_to_vec,
    vec_to_rho,
)


def main():
    print("第 4 小步：比较时间演化")

    ctx = build_reference_obe(
        saturation=0.05,
        polarization_cart=(1.0, 0.0, 0.0),
        B_z_G=300.0,
    )
    ham = ctx["ham"]
    obe = ctx["obe"]
    N = ctx["N"]

    rho0 = thermal_ground_state(ctx["N_G"], ctx["N_E"])
    position = np.zeros(3)
    E_q = obe.laserBeams["g->e"].total_electric_field(position, 0.0)
    B_cart = obe.magField.Field(position, 0.0)

    L_direct = build_direct_L(ham, E_q, B_cart)
    L_pylcp = build_pylcp_component_L(obe, position, 0.0)
    t_final = 50e-9

    rho_expm = vec_to_rho(
        expm(L_direct * t_final) @ rho_to_vec(rho0),
        N,
    )

    # 用 pylcp 预计算出的参考 L 做独立的 solve_ivp 积分。
    # 03_compare_L.py 已逐元素确认这个 L 就是 pylcp.__drhodt 的常系数形式。
    sol = solve_ivp(
        lambda _t, rho_vec: L_pylcp @ rho_vec,
        (0.0, t_final),
        rho_to_vec(rho0),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step=1e-9,
    )
    rho_pylcp = vec_to_rho(sol.y[:, -1], N)

    max_error = np.max(np.abs(rho_expm - rho_pylcp))
    trace_error = abs(np.trace(rho_expm) - 1.0)
    hermiticity_error = np.max(np.abs(rho_expm - rho_expm.conj().T))
    min_eigenvalue = np.min(np.linalg.eigvalsh(rho_expm)).real

    excited_population = np.trace(
        rho_expm[ctx["N_G"] :, ctx["N_G"] :]
    ).real

    print(f"max |rho_expm-rho_pylcp| = {max_error:.3e}")
    print(f"trace error = {trace_error:.3e}")
    print(f"Hermiticity error = {hermiticity_error:.3e}")
    print(f"minimum eigenvalue = {min_eigenvalue:.3e}")
    print(f"excited population = {excited_population:.6e}")

    assert max_error < 1e-8
    assert trace_error < 1e-10
    assert hermiticity_error < 1e-10
    assert min_eigenvalue > -1e-10

    print("PASS: expm(L*t) 与 pylcp OBE 数值积分逐元素一致")


if __name__ == "__main__":
    main()
