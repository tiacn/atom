# -*- coding: utf-8 -*-
"""测试 4：P 的 Cartesian/spherical 批量往返保持复相位。"""

import numpy as np

from polarization import cartesian_to_spherical, spherical_to_cartesian


def main():
    print("测试 4：P_q <-> P_cart complex roundtrip")
    rng = np.random.default_rng(50401)
    P_q = 1e-9 * (
        rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))
    )
    P_cart = spherical_to_cartesian(P_q)
    P_q_roundtrip = cartesian_to_spherical(P_cart)
    P_cart_roundtrip = spherical_to_cartesian(
        cartesian_to_spherical(P_cart)
    )

    q_error = np.max(np.abs(P_q_roundtrip - P_q))
    cart_error = np.max(np.abs(P_cart_roundtrip - P_cart))
    print(f"q roundtrip error = {q_error:.3e} C/m^2")
    print(f"Cartesian roundtrip error = {cart_error:.3e} C/m^2")
    assert q_error < 1e-24
    assert cart_error < 1e-24
    print("PASS: [q=-1,0,+1] 与 [x,y,z] 映射不改变复振幅和相位")


if __name__ == "__main__":
    main()

