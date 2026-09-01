# -*- coding: utf-8 -*-
"""Step 5.5A 的字面稳态测试和固定热布居线性诊断。

``literal`` 分支严格求完整 Liouvillian 的无限时间稳态。
``linear_thermal`` 分支保持 ElecSus 相同的等基态布居，只用于诊断
原子结构、线强、Doppler 和 convention，不替代 literal 判据。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import Boltzmann, epsilon_0, hbar
from scipy.signal import fftconvolve
from scipy.special import roots_hermite

import path_setup  # noqa: F401

import AtomConstants as AC
import spectra
from config import M_K39
from conventions import (
    cart2spherical,
    ecal_amplitude_from_saturation,
    normalized_pylcp_Eq_from_si_phasor,
)
from k39_model import (
    GAMMA_D1,
    LAMBDA_D1,
    build_k39_d1_hamiltonian,
    detuning_MHz_to_rad_s,
)
from liouvillian import (
    coherent_liouvillian,
    decay_liouvillian,
    physical_hamiltonian_from_fields,
    rho_to_vec,
    vec_to_rho,
)
from polarization import (
    NATURAL_K39_FRACTION,
    density_matrix_to_polarization,
    k39_number_density,
)


# ElecSus K39_D1.IsotopeShift；它把 K39 ground manifold 上移 15.864 MHz。
# 因此当前 OBE 的内部 detuning = ElecSus 外部 axis + 15.864 MHz。
K39_ISOTOPE_SHIFT_MHZ = 15.864

DELTA_M_VALUES = (+1, -1)


@dataclass(frozen=True)
class BenchmarkConfig:
    temperature_C: float = 20.0
    k39_fraction: float = NATURAL_K39_FRACTION
    saturation: float = 1.0e-8

    @property
    def temperature_K(self):
        return self.temperature_C + 273.15

    @property
    def sigma_v_m_s(self):
        return float(np.sqrt(Boltzmann * self.temperature_K / M_K39))

    @property
    def sigma_doppler_MHz(self):
        return self.sigma_v_m_s / LAMBDA_D1 / 1.0e6


@dataclass(frozen=True)
class SteadyStatePoint:
    rho: np.ndarray
    P_q: np.ndarray
    P_cart: np.ndarray
    chi: complex
    liouvillian_residual: float
    trace_error: float
    hermiticity_error: float
    minimum_eigenvalue: float


@dataclass(frozen=True)
class SpectrumMetrics:
    relative_complex_l2: float
    relative_real_l2: float
    relative_imag_l2: float
    max_point_relative: float
    peak_position_model_MHz: float
    peak_position_reference_MHz: float
    peak_position_error_MHz: float
    peak_imag_relative_error: float
    fwhm_model_MHz: float
    fwhm_reference_MHz: float
    fwhm_error_MHz: float
    complex_correlation: float


def si_jones_for_delta_m(delta_m):
    """返回沿 +z 传播、在本项目 phasor 下驱动指定 Delta-m 的 Jones 向量。"""
    if delta_m == +1:
        return np.array([1.0, 1.0j, 0.0], dtype=complex) / np.sqrt(2.0)
    if delta_m == -1:
        return np.array([1.0, -1.0j, 0.0], dtype=complex) / np.sqrt(2.0)
    raise ValueError("delta_m 只允许 +1 或 -1")


def q_index_for_delta_m(delta_m):
    # C_q 是 lowering operator；吸收 Delta-m=+1 对应 q=-1。
    if delta_m == +1:
        return 0
    if delta_m == -1:
        return 2
    raise ValueError("delta_m 只允许 +1 或 -1")


def external_to_internal_detuning_MHz(detuning_MHz, v_z_m_s=0.0):
    doppler_MHz = float(v_z_m_s) / LAMBDA_D1 / 1.0e6
    return float(detuning_MHz) + K39_ISOTOPE_SHIFT_MHZ - doppler_MHz


class LiteralSteadyStateSolver:
    """完整 16 能级 Liouvillian 的无限时间稳态求解器。"""

    def __init__(self, config=BenchmarkConfig()):
        self.config = config
        self.ham, self.basis_g, self.basis_e = build_k39_d1_hamiltonian()
        self.N_G = len(self.basis_g)
        self.N = self.ham.n
        self.C_all = self.ham.d_q_bare["g->e"]
        self.L_decay = decay_liouvillian(self.C_all, GAMMA_D1)
        self.trace_row = rho_to_vec(np.eye(self.N))
        self.density_info = k39_number_density(
            config.temperature_C,
            config.k39_fraction,
        )
        self.number_density_m3 = self.density_info.k39_m3

    def solve(
        self,
        B_z_G,
        detuning_MHz,
        v_z_m_s,
        delta_m,
        saturation=None,
    ):
        if saturation is None:
            saturation = self.config.saturation
        jones = si_jones_for_delta_m(delta_m)
        Ecal_cart = ecal_amplitude_from_saturation(saturation) * jones
        E_q_pylcp = normalized_pylcp_Eq_from_si_phasor(Ecal_cart)
        internal_detuning = external_to_internal_detuning_MHz(
            detuning_MHz,
            v_z_m_s,
        )
        H = physical_hamiltonian_from_fields(
            self.ham,
            E_q_pylcp,
            np.array([0.0, 0.0, float(B_z_G)]),
            detuning_rad_s=detuning_MHz_to_rad_s(internal_detuning),
        )
        L = coherent_liouvillian(H) + self.L_decay

        constrained = L.copy()
        rhs = np.zeros(self.N**2, dtype=complex)
        constrained[-1, :] = self.trace_row
        rhs[-1] = 1.0
        # NumPy 直接调用本环境的优化 LAPACK；残差和 density diagnostics 在下方
        # 独立检查，因此不依赖 scipy.linalg.solve 的 condition-estimate warning。
        rho_vec = np.linalg.solve(constrained, rhs)
        rho = vec_to_rho(rho_vec, self.N)
        P_q, P_cart = density_matrix_to_polarization(
            rho,
            self.C_all,
            self.number_density_m3,
        )

        Ecal_q = cart2spherical(Ecal_cart)
        q_index = q_index_for_delta_m(delta_m)
        chi = P_q[q_index] / (epsilon_0 * Ecal_q[q_index])
        hermitian = 0.5 * (rho + rho.conj().T)
        return SteadyStatePoint(
            rho=rho,
            P_q=P_q,
            P_cart=P_cart,
            chi=complex(chi),
            liouvillian_residual=float(np.linalg.norm(L @ rho_vec)),
            trace_error=float(abs(np.trace(rho) - 1.0)),
            hermiticity_error=float(np.max(np.abs(rho - rho.conj().T))),
            minimum_eigenvalue=float(np.min(np.linalg.eigvalsh(hermitian))),
        )

    def dark_state_index(self, delta_m):
        target_m = 2.0 if delta_m == +1 else -2.0
        matches = np.flatnonzero(
            np.isclose(self.basis_g[:, 0], 2.0)
            & np.isclose(self.basis_g[:, 1], target_m)
        )
        if len(matches) != 1:
            raise RuntimeError("未唯一找到 stretched dark state")
        return int(matches[0])

    def gauss_hermite_average_at_detuning(
        self,
        B_z_G,
        detuning_MHz,
        delta_m,
        n_nodes,
        saturation=None,
    ):
        nodes, weights = roots_hermite(int(n_nodes))
        velocities = np.sqrt(2.0) * self.config.sigma_v_m_s * nodes
        chi_values = np.array(
            [
                self.solve(
                    B_z_G,
                    detuning_MHz,
                    velocity,
                    delta_m,
                    saturation=saturation,
                ).chi
                for velocity in velocities
            ]
        )
        return complex(np.dot(weights, chi_values) / np.sqrt(np.pi))


class LinearThermalResponse:
    """固定等基态布居的 OBE 一阶响应；用于与 ElecSus 做根因诊断。"""

    def __init__(self, config=BenchmarkConfig()):
        self.config = config
        self.ham, self.basis_g, self.basis_e = build_k39_d1_hamiltonian()
        self.N_G = len(self.basis_g)
        self.C_all = self.ham.d_q_bare["g->e"]
        self.density_info = k39_number_density(
            config.temperature_C,
            config.k39_fraction,
        )
        self.number_density_m3 = self.density_info.k39_m3

    def transition_lines(self, B_z_G, delta_m):
        zero_field = np.zeros(3, dtype=complex)
        H = physical_hamiltonian_from_fields(
            self.ham,
            zero_field,
            np.array([0.0, 0.0, float(B_z_G)]),
            detuning_rad_s=0.0,
        )
        H_g = H[: self.N_G, : self.N_G]
        H_e = H[self.N_G :, self.N_G :]
        energy_g, U_g = np.linalg.eigh(H_g)
        energy_e, U_e = np.linalg.eigh(H_e)

        q_index = q_index_for_delta_m(delta_m)
        C_ge = self.C_all[q_index, : self.N_G, self.N_G :]
        C_eigen = U_g.conj().T @ C_ge @ U_e
        strengths = np.abs(C_eigen) ** 2 / self.N_G
        frequencies = (
            (energy_e[None, :] - energy_g[:, None]) / (2.0 * np.pi * 1e6)
            - K39_ISOTOPE_SHIFT_MHZ
        )
        mask = strengths > 1e-16
        return frequencies[mask], strengths[mask]

    def local_chi(
        self,
        detuning_MHz,
        B_z_G,
        delta_m,
        emulate_elecsus_numerics=False,
    ):
        axis = np.asarray(detuning_MHz, dtype=float)
        frequencies, strengths = self.transition_lines(B_z_G, delta_m)
        gamma_rad_s = GAMMA_D1
        effective_dipole_sq = self._d0_C_m() ** 2
        if emulate_elecsus_numerics:
            # spectra.py::FreqStren 用 int() 把每条跃迁中心截断到整数 MHz，
            # 并丢弃 cleb^2<=0.0005 的弱线。
            frequencies = np.trunc(frequencies)
            keep = strengths * self.N_G > 0.0005
            frequencies = frequencies[keep]
            strengths = strengths[keep]

            # spectra.py 使用舍入 NatGamma、ElecSus 波长和总 K 密度自展宽。
            gamma0 = 2.0 * np.pi * AC.KD1Transition.NatGamma * 1.0e6
            gamma_self = (
                2.0
                * np.pi
                * gamma0
                * self.density_info.total_potassium_m3
                * (AC.KD1Transition.wavelength / (2.0 * np.pi)) ** 3
            )
            gamma_rad_s = gamma0 + gamma_self
            # ElecSus dipoleStrength=sqrt(3)*d0，FreqStren 另含 1/3。
            effective_dipole_sq = AC.KD1Transition.dipoleStrength**2 / 3.0

        delta_rad_s = 2.0 * np.pi * 1.0e6 * (
            axis[:, None] - frequencies[None, :]
        )
        prefactor = self.number_density_m3 * (
            effective_dipole_sq / (epsilon_0 * hbar)
        )
        return prefactor * np.sum(
            strengths[None, :]
            * 1.0j
            / (gamma_rad_s / 2.0 - 1.0j * delta_rad_s),
            axis=1,
        )

    def doppler_average(
        self,
        output_axis_MHz,
        B_z_G,
        delta_m,
        integration_step_MHz=0.25,
        velocity_range_sigma=7.0,
        emulate_elecsus_numerics=False,
    ):
        """用均匀 Doppler-shift 网格和 FFT 卷积做确定性积分。

        这是变量替换 ``u=v_z/lambda`` 后的复合梯形积分。局域响应的每个
        网格点都来自同一冻结 Hamiltonian/dipole 的线性 OBE。
        """
        output_axis = np.asarray(output_axis_MHz, dtype=float)
        step = float(integration_step_MHz)
        if emulate_elecsus_numerics:
            sigma = (
                self.config.sigma_v_m_s
                / AC.KD1Transition.wavelength
                / 1.0e6
            )
        else:
            sigma = self.config.sigma_doppler_MHz
        cutoff = float(velocity_range_sigma) * sigma
        limit = float(np.max(np.abs(output_axis)) + cutoff + 2.0 * step)
        half_points = int(np.ceil(limit / step))
        grid = np.arange(-half_points, half_points + 1, dtype=float) * step

        local = self.local_chi(
            grid,
            B_z_G,
            delta_m,
            emulate_elecsus_numerics=emulate_elecsus_numerics,
        )
        gaussian = np.exp(-0.5 * (grid / sigma) ** 2) / (
            np.sqrt(2.0 * np.pi) * sigma
        )
        gaussian[np.abs(grid) > cutoff] = 0.0
        convolved = fftconvolve(local, gaussian, mode="same") * step
        real = np.interp(output_axis, grid, convolved.real)
        imag = np.interp(output_axis, grid, convolved.imag)
        return real + 1.0j * imag

    def elecsus_chi(self, detuning_MHz, B_z_G, delta_m):
        params = {
            "Elem": "K",
            "Dline": "D1",
            "Bfield": float(B_z_G),
            "T": self.config.temperature_C,
            "GammaBuf": 0.0,
            "shift": 0.0,
            "Constrain": True,
            "DoppTemp": self.config.temperature_C,
            # 先让 ElecSus 只算纯 K39，再乘本项目采用的实际 K39 fraction。
            "K40frac": 0.0,
            "K41frac": 0.0,
            "BoltzmannFactor": False,
        }
        chi_plus, chi_minus, _ = spectra.calc_chi(detuning_MHz, params)
        # 不按 Plus/Minus 或 Left/Right 名称猜映射。直接比较 FreqStren 给出的
        # 跃迁能量和线强可知：本项目吸收 Delta-m=+1 与 ElecSus 的 Right
        # transition list 相同，而 Delta-m=-1 与 Left list 相同。calc_chi 最终
        # 返回 totalChiPlus=Left、totalChiMinus=Right，因此这里必须交叉选择。
        selected = chi_minus if delta_m == +1 else chi_plus
        return self.config.k39_fraction * np.asarray(selected)

    @staticmethod
    def _d0_C_m():
        from conventions import dipole_scale_from_spontaneous_emission

        return float(dipole_scale_from_spontaneous_emission())


def _relative_l2(model, reference):
    denominator = np.linalg.norm(reference)
    if denominator == 0.0:
        return float(np.linalg.norm(model - reference))
    return float(np.linalg.norm(model - reference) / denominator)


def _fwhm(axis, absorption):
    axis = np.asarray(axis, dtype=float)
    y = np.asarray(absorption, dtype=float)
    peak = int(np.argmax(y))
    half = 0.5 * y[peak]
    left_candidates = np.flatnonzero(y[:peak] <= half)
    right_candidates = np.flatnonzero(y[peak + 1 :] <= half)
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return float("nan")
    left_low = int(left_candidates[-1])
    right_low = peak + 1 + int(right_candidates[0])
    left = np.interp(half, y[left_low : left_low + 2], axis[left_low : left_low + 2])
    # 右侧 y 递减，反转后再插值。
    right = np.interp(
        half,
        y[right_low - 1 : right_low + 1][::-1],
        axis[right_low - 1 : right_low + 1][::-1],
    )
    return float(right - left)


def spectrum_metrics(axis_MHz, model, reference):
    axis = np.asarray(axis_MHz, dtype=float)
    model = np.asarray(model, dtype=complex)
    reference = np.asarray(reference, dtype=complex)
    peak_model_index = int(np.argmax(model.imag))
    peak_reference_index = int(np.argmax(reference.imag))
    peak_model = float(axis[peak_model_index])
    peak_reference = float(axis[peak_reference_index])
    peak_ref_amplitude = float(np.max(reference.imag))
    maximum_scale = float(np.max(np.abs(reference)))
    correlation = abs(np.vdot(reference, model)) / (
        np.linalg.norm(reference) * np.linalg.norm(model)
    )
    return SpectrumMetrics(
        relative_complex_l2=_relative_l2(model, reference),
        relative_real_l2=_relative_l2(model.real, reference.real),
        relative_imag_l2=_relative_l2(model.imag, reference.imag),
        max_point_relative=float(np.max(np.abs(model - reference)) / maximum_scale),
        peak_position_model_MHz=peak_model,
        peak_position_reference_MHz=peak_reference,
        peak_position_error_MHz=peak_model - peak_reference,
        peak_imag_relative_error=float(
            (np.max(model.imag) - peak_ref_amplitude) / peak_ref_amplitude
        ),
        fwhm_model_MHz=_fwhm(axis, model.imag),
        fwhm_reference_MHz=_fwhm(axis, reference.imag),
        fwhm_error_MHz=_fwhm(axis, model.imag) - _fwhm(axis, reference.imag),
        complex_correlation=float(correlation),
    )
