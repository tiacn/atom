# -*- coding: utf-8 -*-
"""测试 1：固定 SI phasor，并验证 PyLCP 球基底和场相位。"""

import numpy as np

import path_setup  # noqa: F401

import pylcp
from k39_model import K_D1
from pylcp.common import cart2spherical, spherical2cart

from conventions import (
    PHASOR_CONVENTION,
    ecal_amplitude_from_saturation,
    normalized_pylcp_Eq_from_si_phasor,
    si_phasor_from_normalized_pylcp_Eq,
)


def main():
    print("测试 1：SI phasor 与 PyLCP 球基底")
    print(f"adopted convention: {PHASOR_CONVENTION}")
    print("PyLCP array order = [q=-1, q=0, q=+1]")

    vector = np.array([0.31 + 0.27j, -0.42 + 0.19j, 0.11 - 0.07j])
    roundtrip = spherical2cart(cart2spherical(vector))
    inverse_error = np.max(np.abs(roundtrip - vector))
    print(f"Cartesian -> spherical -> Cartesian error = {inverse_error:.3e}")
    assert inverse_error < 1e-14

    x = np.array([1.0, 0.0, 0.0], dtype=complex)
    jones_plus_i = np.array([1.0, 1.0j, 0.0]) / np.sqrt(2.0)
    jones_minus_i = np.array([1.0, -1.0j, 0.0]) / np.sqrt(2.0)
    print(f"x spherical = {cart2spherical(x)}")
    print(f"(x+i*y)/sqrt(2) spherical = {cart2spherical(jones_plus_i)}")
    print(f"(x-i*y)/sqrt(2) spherical = {cart2spherical(jones_minus_i)}")

    assert np.allclose(
        cart2spherical(x),
        [1.0 / np.sqrt(2.0), 0.0, -1.0 / np.sqrt(2.0)],
    )
    assert np.allclose(cart2spherical(jones_plus_i), [1.0, 0.0, 0.0])
    assert np.allclose(cart2spherical(jones_minus_i), [0.0, 0.0, -1.0])

    saturation = 0.07
    Ecal = ecal_amplitude_from_saturation(saturation) * jones_plus_i
    Eq_py = normalized_pylcp_Eq_from_si_phasor(Ecal)
    Ecal_roundtrip = si_phasor_from_normalized_pylcp_Eq(Eq_py)
    field_roundtrip_error = np.max(np.abs(Ecal_roundtrip - Ecal))
    print(f"SI phasor <-> normalized PyLCP E_q error = {field_roundtrip_error:.3e} V/m")
    assert field_roundtrip_error < 1e-12

    # 直接验证 fields.py 的实现相位：exp(-ikz+i*delta*t) 是本文件
    # SI 正频率 exp(+ikz-i*omega*t) 包络的复共轭方向。
    detuning = 2.0 * np.pi * 1.3e6
    beam = pylcp.infinitePlaneWaveBeam(
        kvec=np.array([0.0, 0.0, K_D1]),
        pol=np.conjugate(jones_plus_i),
        pol_coord="cartesian",
        s=saturation,
        delta=detuning,
    )
    position = np.array([0.0, 0.0, 0.37e-6])
    time = 23e-9
    source_value = beam.electric_field(position, time)
    expected = (
        np.sqrt(2.0 * saturation)
        * cart2spherical(np.conjugate(jones_plus_i))
        * np.exp(-1j * K_D1 * position[2] + 1j * detuning * time)
    )
    source_phase_error = np.max(np.abs(source_value - expected))
    print(f"PyLCP fields.py phase/source error = {source_phase_error:.3e}")
    assert source_phase_error < 1e-13
    print("PASS: phasor、球基底顺序、相位和 SI/PyLCP 共轭接口已固定")


if __name__ == "__main__":
    main()
