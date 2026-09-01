# -*- coding: utf-8 -*-
"""K39 D1 原子模型。

只负责构造 pylcp Hamiltonian/OBE，并把所有物理单位和基底约定集中在这里。
本文件不依赖 v5b_demo 或 hybrid_engine。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import constants as cts


# 让脚本无论从哪个当前目录启动，都优先使用工作区内的 pylcp 源码。
WORKSPACE = Path(__file__).resolve().parents[3]
PYLCP_SOURCE = WORKSPACE / "pylcp"
if str(PYLCP_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYLCP_SOURCE))

import pylcp  # noqa: E402
import pylcp.hamiltonians as hh  # noqa: E402


# ---------------------------------------------------------------------------
# 兼容较新 sympy：pylcp 老版本会把 np.float64 直接传给 wigner 函数。
# ---------------------------------------------------------------------------
def _as_wigner_number(value):
    from sympy import Rational

    value = float(value)
    if abs(value - round(value)) < 1e-12:
        return int(round(value))
    if abs(2.0 * value - round(2.0 * value)) < 1e-12:
        return Rational(int(round(2.0 * value)), 2)
    raise ValueError(f"量子数必须是整数或半整数，收到 {value}")


def _wig3j(j1, j2, j3, m1, m2, m3):
    from sympy.physics.wigner import wigner_3j

    args = [_as_wigner_number(x) for x in (j1, j2, j3, m1, m2, m3)]
    return float(wigner_3j(*args))


def _wig6j(j1, j2, j3, j4, j5, j6):
    from sympy.physics.wigner import wigner_6j

    args = [_as_wigner_number(x) for x in (j1, j2, j3, j4, j5, j6)]
    return float(wigner_6j(*args))


hh.wig3j = _wig3j
hh.wig6j = _wig6j


# ---------------------------------------------------------------------------
# K39 D1 常数
# ---------------------------------------------------------------------------
# 本文件 OBE 的 Hamiltonian/Liouvillian 统一使用角频率 rad/s。
# 对照源码：
#   PyLCP: pylcp/pylcp/atom.py，K39 分支（I、gI、gJ、Ahfs、lambda、tau）。
#   ElecSus: ElecSus/elecsus/libs/AtomConstants.py，K39、KD1Transition、
#             K39_D1；电子自旋 g 因子另见 FundamentalConstants.py。
I_NUC = 1.5
J_GROUND = 0.5
J_EXCITED = 0.5
G_J_GROUND = 2.00229421  # PyLCP K39 4S1/2；ElecSus 用 gL=1 和电子 gs 构造 Zeeman 项。
G_J_EXCITED = 2.0 / 3.0  # PyLCP K39 4P1/2；也是 LS 耦合的 Landé gJ。
G_I = -0.00014193489  # PyLCP atom.py 与 ElecSus AtomConstants.py 相同。

# PyLCP 以 Hz 保存 Ahfs；ElecSus 以 MHz 保存 As/Ap。这里先保存 Hz，
# 再乘 2*pi 变成当前 Hamiltonian 使用的 rad/s。
A_HFS_GROUND_HZ = 230.8598601e6  # PyLCP: 230.8598601e6 Hz；ElecSus K39.As: 230.8598601 MHz。
A_HFS_EXCITED_HZ = 27.775e6  # PyLCP: 27.775e6 Hz；ElecSus K39_D1.Ap: 27.775 MHz。
A_HFS_GROUND = 2.0 * np.pi * A_HFS_GROUND_HZ
A_HFS_EXCITED = 2.0 * np.pi * A_HFS_EXCITED_HZ

# PyLCP 用激发态寿命 tau=26.72 ns；ElecSus KD1Transition.NatGamma=5.956 MHz，
# 其 NatGamma 对应 Gamma/(2*pi)。6.035 MHz 是 ElecSus 的 K D2 值，不属于 D1。
D1_LIFETIME_S = 26.72e-9
GAMMA_D1_HZ = 1.0 / (2.0 * np.pi * D1_LIFETIME_S)
GAMMA_D1 = 2.0 * np.pi * GAMMA_D1_HZ

# 采用 PyLCP 的 K39 同位素 D1 波长；ElecSus 的天然钾加权线中心为
# 770.108353667 nm，两者差约 0.000031 nm，对当前 Doppler 计算可忽略。
LAMBDA_D1 = 770.108385049e-9
K_D1 = 2.0 * np.pi / LAMBDA_D1

# scipy 给出 Hz/T；乘 1e-4 得 Hz/G，再乘 2*pi 得 rad/s/G。
MU_B = (
    2.0
    * np.pi
    * cts.physical_constants["Bohr magneton in Hz/T"][0]
    * 1e-4
)


def build_k39_d1_hamiltonian():
    """构造 8+8 维 K39 D1 Hamiltonian。

    返回
    ----
    ham
        pylcp.hamiltonian。
    basis_g, basis_e
        每一行是 [F, m_F]。
    """
    H_g, mu_g, basis_g_columns = hh.hyperfine_coupled(
        J=J_GROUND,
        I=I_NUC,
        gJ=G_J_GROUND,
        gI=G_I,
        Ahfs=A_HFS_GROUND,
        muB=MU_B,
        return_basis=True,
    )
    H_e, mu_e, basis_e_columns = hh.hyperfine_coupled(
        J=J_EXCITED,
        I=I_NUC,
        gJ=G_J_EXCITED,
        gI=G_I,
        Ahfs=A_HFS_EXCITED,
        muB=MU_B,
        return_basis=True,
    )

    # hyperfine_coupled 返回 shape=(2,N)，这里统一成每行一个 [F,m_F]。
    basis_g = basis_g_columns.T
    basis_e = basis_e_columns.T

    # d_q 的行是 ground，列是 excited，不转置。
    d_q_ge, dipole_basis_g, dipole_basis_e = (
        hh.dqij_two_hyperfine_manifolds(
            J=J_GROUND,
            Jp=J_EXCITED,
            I=I_NUC,
            normalize=True,
            return_basis=True,
        )
    )

    if not np.array_equal(basis_g, dipole_basis_g):
        raise RuntimeError("ground basis 顺序与偶极矩 basis 不一致")
    if not np.array_equal(basis_e, dipole_basis_e):
        raise RuntimeError("excited basis 顺序与偶极矩 basis 不一致")

    ham = pylcp.hamiltonian()
    ham.add_H_0_block("g", H_g)
    ham.add_H_0_block("e", H_e)
    ham.add_mu_q_block("g", mu_g, muB=1.0)
    ham.add_mu_q_block("e", mu_e, muB=1.0)
    ham.add_d_q_block(
        "g",
        "e",
        d_q_ge,
        k=K_D1,
        gamma=GAMMA_D1,
    )
    ham.make_full_matrices()

    return ham, basis_g, basis_e


def build_reference_obe(
    saturation=0.05,
    polarization_cart=(1.0, 0.0, 0.0),
    B_z_G=300.0,
    detuning_MHz=0.0,
):
    """构造恒定 x/y 偏振、恒定 Bz 的 pylcp OBE。

    pylcp 用角频率 delta (rad/s) 积分激光相位；对外接口使用 MHz。
    当 detuning_MHz != 0 时，pylcp 实验室框架中的光场会显含时间。
    """
    ham, basis_g, basis_e = build_k39_d1_hamiltonian()

    beam = pylcp.infinitePlaneWaveBeam(
        kvec=np.array([0.0, 0.0, K_D1]),
        pol=np.asarray(polarization_cart, dtype=float),
        pol_coord="cartesian",
        s=float(saturation),
        delta=detuning_MHz_to_rad_s(detuning_MHz),
    )
    beams = pylcp.laserBeams([beam])
    magnetic_field = pylcp.constantMagneticField(
        np.array([0.0, 0.0, float(B_z_G)])
    )

    obe = pylcp.obe(
        beams,
        magnetic_field,
        hamitlonian=ham,
        transform_into_re_im=False,
        use_sparse_matrices=False,
        r0=np.zeros(3),
        v0=np.zeros(3),
    )

    return {
        "ham": ham,
        "obe": obe,
        "basis_g": basis_g,
        "basis_e": basis_e,
        "N_G": len(basis_g),
        "N_E": len(basis_e),
        "N": ham.n,
        "detuning_MHz": float(detuning_MHz),
        "detuning_rad_s": detuning_MHz_to_rad_s(detuning_MHz),
    }


def thermal_ground_state(N_G=8, N_E=8):
    """本阶段采用 8 个基态等布居、激发态为零的初态。"""
    rho = np.zeros((N_G + N_E, N_G + N_E), dtype=complex)
    rho[np.arange(N_G), np.arange(N_G)] = 1.0 / N_G
    return rho


def detuning_MHz_to_rad_s(detuning_MHz):
    """普通频率 MHz -> 角频率 rad/s。"""
    return 2.0 * np.pi * float(detuning_MHz) * 1e6
