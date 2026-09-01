# -*- coding: utf-8 -*-
"""第 3 小步：逐矩阵比较自写 L 和 pylcp OBE 的 L。"""

import numpy as np

from k39_model import build_reference_obe
from liouvillian import (
    build_direct_L,
    build_pylcp_component_L,
    decay_liouvillian,
)


def compare_case(name, saturation, polarization, B_z_G, tolerance=1e-6):
    ctx = build_reference_obe(
        saturation=saturation,
        polarization_cart=polarization,
        B_z_G=B_z_G,
    )
    ham = ctx["ham"]
    obe = ctx["obe"]

    position = np.zeros(3)
    E_q = obe.laserBeams["g->e"].total_electric_field(position, 0.0)
    B_cart = obe.magField.Field(position, 0.0)

    L_direct = build_direct_L(ham, E_q, B_cart)
    L_pylcp = build_pylcp_component_L(obe, position, 0.0)

    absolute_error = np.max(np.abs(L_direct - L_pylcp))
    scale = max(np.max(np.abs(L_pylcp)), 1.0)
    relative_error = absolute_error / scale

    print(
        f"{name}: abs error={absolute_error:.3e}, "
        f"relative error={relative_error:.3e}"
    )
    assert relative_error < tolerance


def main():
    print("第 3 小步：比较完整 256x256 Liouvillian")

    # 先单独检查最容易写错的自发辐射项。
    ctx = build_reference_obe(saturation=0.0, B_z_G=0.0)
    ham = ctx["ham"]
    obe = ctx["obe"]
    L_decay_direct = decay_liouvillian(ham.d_q_bare["g->e"])
    L_decay_pylcp = np.asarray(obe.ev_mat["decay"])
    decay_error = np.max(np.abs(L_decay_direct - L_decay_pylcp))
    decay_scale = np.max(np.abs(L_decay_pylcp))
    decay_relative_error = decay_error / decay_scale
    print(
        f"decay only: abs error={decay_error:.3e}, "
        f"relative error={decay_relative_error:.3e}"
    )
    assert decay_relative_error < 1e-12

    compare_case(
        name="no light, B=300 G",
        saturation=0.0,
        polarization=(1.0, 0.0, 0.0),
        B_z_G=300.0,
    )
    compare_case(
        name="x polarization, s=0.05, B=300 G",
        saturation=0.05,
        polarization=(1.0, 0.0, 0.0),
        B_z_G=300.0,
    )
    compare_case(
        name="elliptical polarization, s=0.05, B=300 G",
        saturation=0.05,
        polarization=(1.0, 0.4, 0.0),
        B_z_G=300.0,
    )

    print("PASS: 自写 Hamiltonian、decay 和 pylcp OBE 的完整 L 一致")


if __name__ == "__main__":
    main()

