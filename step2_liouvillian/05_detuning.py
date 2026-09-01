# -*- coding: utf-8 -*-
"""第 5 小步：验证激光失谐的旋转框架写法和符号。"""

import numpy as np

from k39_model import build_reference_obe, detuning_MHz_to_rad_s
from liouvillian import (
    build_direct_L,
    build_pylcp_component_L,
    detuning_liouvillian,
    excited_state_projector,
    rho_to_vec,
)


def rotating_frame_transform(ham, detuning_rad_s, time):
    """返回 rho_rot = U^dagger rho_lab U 对应的向量空间矩阵 S。

    U = diag(I_g, exp(-i*delta*t) I_e)
    对列优先 vec，有 vec(A rho B)=(B^T kron A)vec(rho)。
    """
    N_G = int(ham.ns[0])
    U = np.eye(ham.n, dtype=complex)
    U[N_G:, N_G:] *= np.exp(-1j * detuning_rad_s * time)
    return np.kron(U.T, U.conj().T)


def main():
    print("第 5 小步：激光失谐")

    detuning_MHz = 80.0
    delta = detuning_MHz_to_rad_s(detuning_MHz)
    ctx = build_reference_obe(
        saturation=0.05,
        polarization_cart=(1.0, 0.0, 0.0),
        B_z_G=300.0,
        detuning_MHz=detuning_MHz,
    )
    ham = ctx["ham"]
    obe = ctx["obe"]
    position = np.zeros(3)
    B_cart = obe.magField.Field(position, 0.0)

    print(f"detuning = {detuning_MHz:.1f} MHz")
    print(f"angular detuning = {delta:.6e} rad/s")
    assert abs(delta / (2.0 * np.pi * 1e6) - detuning_MHz) < 1e-12

    # t=0 时激光相位为零，可直接用 E_q(0) 构造静态旋转框架 L。
    E_q_0 = obe.laserBeams["g->e"].total_electric_field(position, 0.0)
    L_rotating = build_direct_L(
        ham,
        E_q_0,
        B_cart,
        detuning_rad_s=delta,
    )
    L_lab_0 = build_pylcp_component_L(obe, position, 0.0)
    L_delta = detuning_liouvillian(ham, delta)

    zero_time_error = np.max(np.abs(L_rotating - (L_lab_0 + L_delta)))
    zero_time_scale = np.max(np.abs(L_rotating))
    print(
        "t=0 rotating/lab relation relative error = "
        f"{zero_time_error / zero_time_scale:.3e}"
    )
    assert zero_time_error / zero_time_scale < 1e-12

    # 在任意时刻，先把 pylcp 的时变实验室框架 L 做相似变换，
    # 再加坐标变换导数项 L_delta，应得到同一个静态 L_rotating。
    for time_ns in (0.7, 2.3, 5.1):
        time = time_ns * 1e-9
        L_lab_t = build_pylcp_component_L(obe, position, time)
        S = rotating_frame_transform(ham, delta, time)
        S_inv = S.conj().T
        L_transformed = S @ L_lab_t @ S_inv + L_delta

        error = np.max(np.abs(L_rotating - L_transformed))
        relative_error = error / zero_time_scale
        print(
            f"t={time_ns:.1f} ns frame-equivalence relative error = "
            f"{relative_error:.3e}"
        )
        assert relative_error < 1e-12

    # 失谐项不应直接改变任意对角布居。
    populations = np.linspace(1.0, 2.0, ham.n)
    populations /= populations.sum()
    rho_diagonal = np.diag(populations).astype(complex)
    direct_population_change = L_delta @ rho_to_vec(rho_diagonal)
    population_error = np.max(np.abs(direct_population_change))
    print(f"detuning acting on diagonal rho = {population_error:.3e}")
    assert population_error < 1e-12

    # 检查符号：对 rho_ge，H_delta=-delta*P_e 给出
    # d(rho_ge)/dt = -i*delta*rho_ge。
    N_G = int(ham.ns[0])
    rho_ge = np.zeros((ham.n, ham.n), dtype=complex)
    rho_ge[0, N_G] = 1.0
    drho_ge = (L_delta @ rho_to_vec(rho_ge)).reshape(
        (ham.n, ham.n),
        order="F",
    )
    sign_error = abs(drho_ge[0, N_G] - (-1j * delta))
    print(f"positive-detuning sign error = {sign_error:.3e}")
    assert sign_error < 1e-6

    # 正负失谐应严格反号。
    sign_symmetry_error = np.max(
        np.abs(
            detuning_liouvillian(ham, delta)
            + detuning_liouvillian(ham, -delta)
        )
    )
    print(f"plus/minus detuning antisymmetry error = {sign_symmetry_error:.3e}")
    assert sign_symmetry_error < 1e-12

    # 投影算符本身也做一个结构检查，防止误加到 ground manifold。
    P_e = excited_state_projector(ham)
    assert np.allclose(P_e[:N_G, :N_G], 0.0)
    assert np.allclose(P_e[N_G:, N_G:], np.eye(ham.n - N_G))

    print("PASS: 激光失谐的单位、旋转框架结构、时间相位和正负号正确")


if __name__ == "__main__":
    main()
