# -*- coding: utf-8 -*-
"""第 4 小步：验证 delta_eff=delta-k*v_z 的大小和符号。"""

import numpy as np

import path_setup  # noqa: F401

from k39_model import LAMBDA_D1
from moving_atom import (
    SingleAtomEngine,
    doppler_shift_MHz,
    effective_detuning_rad_s,
)


def main():
    print("第 4 小步：Doppler 失谐")
    engine = SingleAtomEngine()
    laser_detuning_MHz = 40.0
    speed = 100.0
    shift = doppler_shift_MHz(speed)

    print(f"v_z = +{speed:.1f} m/s -> Doppler shift = {shift:.6f} MHz")
    print(
        "effective detuning = "
        f"{laser_detuning_MHz-shift:.6f} MHz"
    )

    common = dict(
        B_z_G=300.0,
        saturation=0.05,
        polarization_cart=(1.0, 0.0, 0.0),
    )
    L_moving_plus = engine.build_L(
        **common,
        laser_detuning_MHz=laser_detuning_MHz,
        v_z_m_s=+speed,
    )
    L_static_shifted_plus = engine.build_L(
        **common,
        laser_detuning_MHz=laser_detuning_MHz - shift,
        v_z_m_s=0.0,
    )
    plus_error = np.max(np.abs(L_moving_plus - L_static_shifted_plus))

    L_moving_minus = engine.build_L(
        **common,
        laser_detuning_MHz=laser_detuning_MHz,
        v_z_m_s=-speed,
    )
    L_static_shifted_minus = engine.build_L(
        **common,
        laser_detuning_MHz=laser_detuning_MHz + shift,
        v_z_m_s=0.0,
    )
    minus_error = np.max(np.abs(L_moving_minus - L_static_shifted_minus))

    delta_plus = effective_detuning_rad_s(laser_detuning_MHz, +speed)
    delta_minus = effective_detuning_rad_s(laser_detuning_MHz, -speed)

    print(f"+v equivalence max error = {plus_error:.3e}")
    print(f"-v equivalence max error = {minus_error:.3e}")
    assert delta_plus < delta_minus
    assert abs(shift - speed / LAMBDA_D1 / 1e6) < 1e-12
    assert plus_error < 1e-6
    assert minus_error < 1e-6
    print("PASS: 正负速度的 Doppler 大小、符号和静止原子等价性正确")


if __name__ == "__main__":
    main()
