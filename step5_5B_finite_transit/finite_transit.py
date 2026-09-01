# -*- coding: utf-8 -*-
"""Uniform-field finite-transit trajectory OBE and ensemble observables.

物理逻辑仍来自冻结的 Step 2--5。这里仅利用本阶段的 uniform B/top-hat 条件，
把逐 segment 的同一个 L 合并成总驻留时间，并在严格不变的可达子空间传播。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.constants import epsilon_0
from scipy.linalg import eig

import path_setup  # noqa: F401

from benchmark import (
    BenchmarkConfig,
    K39_ISOTOPE_SHIFT_MHZ,
    LinearThermalResponse,
    q_index_for_delta_m,
    si_jones_for_delta_m,
)
from conventions import (
    PHASOR_CONVENTION,
    Q_VALUES,
    cart2spherical,
    dipole_scale_from_spontaneous_emission,
    ecal_amplitude_from_saturation,
)
from ensemble import bin_current_density_matrices
from k39_model import K_D1, detuning_MHz_to_rad_s
from liouvillian import (
    detuning_liouvillian,
    excited_state_projector,
    rho_to_vec,
    vec_to_rho,
)
from moving_atom import SingleAtomEngine
from polarization import (
    NATURAL_K39_FRACTION,
    density_matrices_to_polarization,
    k39_number_density,
    POLARIZATION_PHASOR_CONVENTION,
    spherical_to_cartesian,
)


ENTRY_SURFACE_CODES = {"side": 0, "z0": 1, "zL": 2}
ENTRY_SURFACE_NAMES = np.array(["side", "z0", "zL"])


@dataclass(frozen=True)
class FiniteTransitConfig:
    temperature_C: float = 20.0
    B_z_G: float = 0.0
    external_detuning_MHz: float = -68.0
    saturation: float = 1.0e-2
    delta_m: int = +1
    k39_fraction: float = NATURAL_K39_FRACTION
    detuning_cache_resolution_MHz: float | None = 0.25
    track_optical_coherence: bool = True


@dataclass
class _ModeCache:
    eigenvalues: np.ndarray
    observable_weights: np.ndarray
    optical_modes: np.ndarray
    full_modes: np.ndarray | None
    eigenvector_condition: float


@dataclass(frozen=True)
class AtomEvolution:
    rho_final: np.ndarray | None
    P_q: np.ndarray
    P_cart: np.ndarray
    dark_population: float
    excited_population: float
    optical_coherence_norm: float
    effective_detuning_MHz: float
    cache_detuning_MHz: float
    eigenvector_condition: float


@dataclass(frozen=True)
class FiniteTransitEnsembleResult:
    positions_m: np.ndarray
    velocities_m_s: np.ndarray
    time_since_entry_s: np.ndarray
    entry_surface_code: np.ndarray
    segment_counts: np.ndarray
    current_slices: np.ndarray
    counts: np.ndarray
    rho_final_all: np.ndarray | None
    rho_bar: np.ndarray | None
    P_q_all: np.ndarray
    P_cart_all: np.ndarray
    P_q_by_slice: np.ndarray
    P_cart_by_slice: np.ndarray
    dark_population_all: np.ndarray
    dark_population_by_slice: np.ndarray
    excited_population_all: np.ndarray
    excited_population_by_slice: np.ndarray
    optical_coherence_norm_all: np.ndarray
    optical_coherence_norm_by_slice: np.ndarray
    mean_time_by_slice_s: np.ndarray
    effective_detuning_MHz: np.ndarray
    global_P_q: np.ndarray
    global_P_cart: np.ndarray
    formal_P_profile_error: float
    max_eigenvector_condition: float
    number_density_m3: float
    d0_C_m: float


def _expectation_row(operator):
    """返回满足 row @ vec(rho) = Tr(operator rho) 的非共轭行向量。"""
    return rho_to_vec(np.asarray(operator).T)


def _reachable_indices(L, initial_vec, tolerance=1e-14):
    active = set(np.flatnonzero(np.abs(initial_vec) > tolerance).tolist())
    while True:
        previous = len(active)
        columns = np.asarray(sorted(active), dtype=int)
        rows = np.flatnonzero(
            np.any(np.abs(L[:, columns]) > tolerance, axis=1)
        )
        active.update(rows.tolist())
        if len(active) == previous:
            return np.asarray(sorted(active), dtype=int)


class UniformFiniteTransitModel:
    """纯圆偏振、uniform B/top-hat 下的严格有限时间传播器。"""

    def __init__(self, config=FiniteTransitConfig()):
        if config.delta_m not in (+1, -1):
            raise ValueError("delta_m 只允许 +1 或 -1")
        if config.saturation <= 0.0:
            raise ValueError("本阶段传播器要求 saturation>0")
        if (
            config.detuning_cache_resolution_MHz is not None
            and config.detuning_cache_resolution_MHz <= 0.0
        ):
            raise ValueError("detuning cache resolution 必须为正或 None")

        self.config = config
        self.engine = SingleAtomEngine()
        self.N = self.engine.N
        self.N_G = self.engine.N_G
        self.C_all = self.engine.ham.d_q_bare["g->e"]
        self.rho_entry = self.engine.initial_thermal_state()
        self.entry_vec = rho_to_vec(self.rho_entry)

        self.si_jones = si_jones_for_delta_m(config.delta_m)
        # 冻结 moving_atom 接口接收 PyLCP 的负频率 Jones 输入。
        self.pylcp_jones = np.conjugate(self.si_jones)
        self.Ecal_cart_V_m = (
            ecal_amplitude_from_saturation(config.saturation) * self.si_jones
        )
        self.Ecal_q_V_m = cart2spherical(self.Ecal_cart_V_m)
        self.target_q_index = q_index_for_delta_m(config.delta_m)

        density = k39_number_density(
            config.temperature_C,
            config.k39_fraction,
        )
        self.number_density_m3 = density.k39_m3
        self.d0_C_m = float(dipole_scale_from_spontaneous_emission())
        self.P_scale = 2.0 * self.number_density_m3 * self.d0_C_m

        # L(delta)=L(delta=0)+delta_rad_s*L_unit。B 和光场均固定。
        L_zero = self.engine.build_L(
            B_z_G=config.B_z_G,
            saturation=config.saturation,
            polarization_cart=self.pylcp_jones,
            laser_detuning_MHz=0.0,
            v_z_m_s=0.0,
        )
        L_unit = detuning_liouvillian(self.engine.ham, 1.0)
        representative_delta = self._effective_detuning_MHz(0.0)
        representative_L = L_zero + detuning_MHz_to_rad_s(
            representative_delta
        ) * L_unit
        self.active_indices = _reachable_indices(
            representative_L,
            self.entry_vec,
        )
        self.inactive_indices = np.setdiff1d(
            np.arange(self.N**2),
            self.active_indices,
        )
        self.L_zero_reduced = L_zero[
            np.ix_(self.active_indices, self.active_indices)
        ]
        self.L_unit_reduced = L_unit[
            np.ix_(self.active_indices, self.active_indices)
        ]
        self.entry_vec_reduced = self.entry_vec[self.active_indices]

        leak_zero = L_zero[
            np.ix_(self.inactive_indices, self.active_indices)
        ]
        leak_unit = L_unit[
            np.ix_(self.inactive_indices, self.active_indices)
        ]
        self.invariant_leakage = float(
            max(
                np.max(np.abs(leak_zero), initial=0.0),
                np.max(np.abs(leak_unit), initial=0.0),
            )
        )
        if self.invariant_leakage != 0.0:
            raise RuntimeError("选出的 reduced Liouvillian 不是严格不变子空间")

        dark_index = self._dark_state_index()
        dark_projector = np.zeros((self.N, self.N), dtype=complex)
        dark_projector[dark_index, dark_index] = 1.0
        observable_operators = list(self.C_all)
        observable_operators.append(dark_projector)
        observable_operators.append(excited_state_projector(self.engine.ham))
        full_rows = np.stack(
            [_expectation_row(operator) for operator in observable_operators]
        )
        self.observable_rows_reduced = full_rows[:, self.active_indices]

        optical_full_indices = np.array(
            [
                ground + excited * self.N
                for excited in range(self.N_G, self.N)
                for ground in range(self.N_G)
            ],
            dtype=int,
        )
        active_lookup = {
            int(full_index): reduced_index
            for reduced_index, full_index in enumerate(self.active_indices)
        }
        self.optical_reduced_indices = np.array(
            [
                active_lookup[index]
                for index in optical_full_indices
                if index in active_lookup
            ],
            dtype=int,
        )
        self._cache: dict[int | float, _ModeCache] = {}

    def _dark_state_index(self):
        target_m = 2.0 if self.config.delta_m == +1 else -2.0
        matches = np.flatnonzero(
            np.isclose(self.engine.basis_g[:, 0], 2.0)
            & np.isclose(self.engine.basis_g[:, 1], target_m)
        )
        if len(matches) != 1:
            raise RuntimeError("未唯一找到 stretched dark state")
        return int(matches[0])

    def _effective_detuning_MHz(
        self,
        v_z_m_s,
        external_detuning_MHz=None,
    ):
        if external_detuning_MHz is None:
            external_detuning_MHz = self.config.external_detuning_MHz
        internal_laser = (
            float(external_detuning_MHz) + K39_ISOTOPE_SHIFT_MHZ
        )
        doppler = K_D1 * float(v_z_m_s) / (2.0 * np.pi * 1.0e6)
        return internal_laser - doppler

    def _cache_key_and_detuning(self, effective_detuning_MHz):
        resolution = self.config.detuning_cache_resolution_MHz
        if resolution is None:
            value = float(effective_detuning_MHz)
            return value, value
        key = int(np.rint(float(effective_detuning_MHz) / resolution))
        return key, key * resolution

    def _mode(self, effective_detuning_MHz, require_full=False):
        key, cache_detuning = self._cache_key_and_detuning(
            effective_detuning_MHz
        )
        cached = self._cache.get(key)
        if cached is not None and (not require_full or cached.full_modes is not None):
            return cached, cache_detuning

        L_reduced = self.L_zero_reduced + detuning_MHz_to_rad_s(
            cache_detuning
        ) * self.L_unit_reduced
        eigenvalues, eigenvectors = eig(L_reduced, check_finite=False)
        coefficients = np.linalg.solve(eigenvectors, self.entry_vec_reduced)
        observable_weights = (
            self.observable_rows_reduced @ eigenvectors
        ) * coefficients[None, :]
        if self.config.track_optical_coherence:
            optical_modes = (
                eigenvectors[self.optical_reduced_indices, :]
                * coefficients[None, :]
            )
        else:
            optical_modes = np.empty((0, len(eigenvalues)), dtype=complex)
        full_modes = (
            eigenvectors * coefficients[None, :] if require_full else None
        )
        mode = _ModeCache(
            eigenvalues=eigenvalues,
            observable_weights=observable_weights,
            optical_modes=optical_modes,
            full_modes=full_modes,
            eigenvector_condition=float(np.linalg.cond(eigenvectors)),
        )
        self._cache[key] = mode
        return mode, cache_detuning

    def propagate(
        self,
        v_z_m_s,
        time_since_entry_s,
        return_rho=False,
        external_detuning_MHz=None,
    ):
        effective_detuning = self._effective_detuning_MHz(
            v_z_m_s,
            external_detuning_MHz=external_detuning_MHz,
        )
        mode, cache_detuning = self._mode(
            effective_detuning,
            require_full=return_rho,
        )
        exponential = np.exp(mode.eigenvalues * float(time_since_entry_s))
        observables = mode.observable_weights @ exponential
        mu_q = observables[:3]
        dark_population = float(observables[3].real)
        excited_population = float(observables[4].real)
        if self.config.track_optical_coherence:
            optical_values = mode.optical_modes @ exponential
            optical_norm = float(np.linalg.norm(optical_values))
        else:
            optical_norm = float("nan")
        P_q = self.P_scale * mu_q
        P_cart = spherical_to_cartesian(P_q)

        rho = None
        if return_rho:
            reduced_vec = mode.full_modes @ exponential
            full_vec = np.zeros(self.N**2, dtype=complex)
            full_vec[self.active_indices] = reduced_vec
            rho = vec_to_rho(full_vec, self.N)

        return AtomEvolution(
            rho_final=rho,
            P_q=P_q,
            P_cart=P_cart,
            dark_population=dark_population,
            excited_population=excited_population,
            optical_coherence_norm=optical_norm,
            effective_detuning_MHz=effective_detuning,
            cache_detuning_MHz=cache_detuning,
            eigenvector_condition=mode.eigenvector_condition,
        )

    def pumping_curve(self, times_s, v_z_m_s=0.0):
        times = np.asarray(times_s, dtype=float)
        effective_detuning = self._effective_detuning_MHz(v_z_m_s)
        mode, _ = self._mode(effective_detuning, require_full=False)
        exponential = np.exp(mode.eigenvalues[:, None] * times[None, :])
        observables = mode.observable_weights @ exponential
        mu_q = observables[:3].T
        return {
            "time_s": times,
            "P_q": self.P_scale * mu_q,
            "dark_population": observables[3].real,
            "excited_population": observables[4].real,
        }

    @property
    def target_Ecal_q_V_m(self):
        return complex(self.Ecal_q_V_m[self.target_q_index])

    def effective_chi_from_P_q(self, P_q):
        return complex(
            np.asarray(P_q)[self.target_q_index]
            / (epsilon_0 * self.target_Ecal_q_V_m)
        )


def _slice_average(values, current_slices, counts, n_slices):
    values = np.asarray(values)
    output = np.zeros((n_slices,) + values.shape[1:], dtype=values.dtype)
    np.add.at(output, current_slices, values)
    occupied = counts > 0
    reshape = (len(counts),) + (1,) * (values.ndim - 1)
    output[occupied] /= counts.reshape(reshape)[occupied]
    return output


def evolve_finite_transit_ensemble(
    trajectories,
    model,
    n_slices,
    return_rho=False,
    progress_every=0,
):
    n_atoms = len(trajectories)
    positions = np.empty((n_atoms, 3))
    velocities = np.empty((n_atoms, 3))
    times = np.empty(n_atoms)
    entry_codes = np.empty(n_atoms, dtype=np.int8)
    segment_counts = np.empty(n_atoms, dtype=np.int16)
    current_slices = np.empty(n_atoms, dtype=int)
    P_q_all = np.empty((n_atoms, 3), dtype=complex)
    P_cart_all = np.empty((n_atoms, 3), dtype=complex)
    dark_all = np.empty(n_atoms)
    excited_all = np.empty(n_atoms)
    optical_all = np.empty(n_atoms)
    effective_detuning = np.empty(n_atoms)
    conditions = np.empty(n_atoms)
    rho_all = (
        np.empty((n_atoms, model.N, model.N), dtype=complex)
        if return_rho
        else None
    )

    for atom_index, trajectory in enumerate(trajectories):
        # geometry 已验证 sum(segment.dt)=entry.time；uniform fields 允许严格合并。
        duration = float(trajectory.entry.time_s)
        segment_duration = sum(segment.dt_s for segment in trajectory.segments)
        if abs(segment_duration - duration) > 1e-12 * max(duration, 1e-12):
            raise AssertionError("trajectory segments 与 time_since_entry 不一致")
        atom = model.propagate(
            trajectory.velocity_m_s[2],
            duration,
            return_rho=return_rho,
        )
        positions[atom_index] = trajectory.current_position_m
        velocities[atom_index] = trajectory.velocity_m_s
        times[atom_index] = duration
        entry_codes[atom_index] = ENTRY_SURFACE_CODES[trajectory.entry.surface]
        segment_counts[atom_index] = len(trajectory.segments)
        current_slices[atom_index] = trajectory.current_slice
        P_q_all[atom_index] = atom.P_q
        P_cart_all[atom_index] = atom.P_cart
        dark_all[atom_index] = atom.dark_population
        excited_all[atom_index] = atom.excited_population
        optical_all[atom_index] = atom.optical_coherence_norm
        effective_detuning[atom_index] = atom.effective_detuning_MHz
        conditions[atom_index] = atom.eigenvector_condition
        if return_rho:
            rho_all[atom_index] = atom.rho_final
        if progress_every and (atom_index + 1) % progress_every == 0:
            print(f"  evolved {atom_index+1}/{n_atoms} atoms", flush=True)

    counts = np.bincount(current_slices, minlength=n_slices)
    P_q_by_slice = _slice_average(P_q_all, current_slices, counts, n_slices)
    P_cart_by_slice = spherical_to_cartesian(P_q_by_slice)
    dark_by_slice = _slice_average(dark_all, current_slices, counts, n_slices)
    excited_by_slice = _slice_average(
        excited_all, current_slices, counts, n_slices
    )
    optical_by_slice = _slice_average(
        optical_all, current_slices, counts, n_slices
    )
    mean_time_by_slice = _slice_average(times, current_slices, counts, n_slices)

    rho_bar = None
    formal_error = float("nan")
    if return_rho:
        formal_counts, rho_bar = bin_current_density_matrices(
            rho_all,
            current_slices,
            n_slices,
        )
        if not np.array_equal(formal_counts, counts):
            raise AssertionError("formal rho binning counts 不一致")
        profile = density_matrices_to_polarization(
            rho_bar,
            model.C_all,
            model.number_density_m3,
        )
        occupied = counts > 0
        formal_error = float(
            np.max(np.abs(profile.P_q[occupied] - P_q_by_slice[occupied]))
        )

    global_P_q = np.mean(P_q_all, axis=0)
    return FiniteTransitEnsembleResult(
        positions_m=positions,
        velocities_m_s=velocities,
        time_since_entry_s=times,
        entry_surface_code=entry_codes,
        segment_counts=segment_counts,
        current_slices=current_slices,
        counts=counts,
        rho_final_all=rho_all,
        rho_bar=rho_bar,
        P_q_all=P_q_all,
        P_cart_all=P_cart_all,
        P_q_by_slice=P_q_by_slice,
        P_cart_by_slice=P_cart_by_slice,
        dark_population_all=dark_all,
        dark_population_by_slice=dark_by_slice,
        excited_population_all=excited_all,
        excited_population_by_slice=excited_by_slice,
        optical_coherence_norm_all=optical_all,
        optical_coherence_norm_by_slice=optical_by_slice,
        mean_time_by_slice_s=mean_time_by_slice,
        effective_detuning_MHz=effective_detuning,
        global_P_q=global_P_q,
        global_P_cart=spherical_to_cartesian(global_P_q),
        formal_P_profile_error=formal_error,
        max_eigenvector_condition=float(np.max(conditions)),
        number_density_m3=model.number_density_m3,
        d0_C_m=model.d0_C_m,
    )


def time_binned_statistics(result, n_bins=8):
    times = np.asarray(result.time_since_entry_s)
    edges = np.quantile(times, np.linspace(0.0, 1.0, int(n_bins) + 1))
    # 保证最大值落入最后一箱。
    bin_indices = np.searchsorted(edges[1:-1], times, side="right")
    rows = []
    for index in range(n_bins):
        mask = bin_indices == index
        rows.append(
            (
                int(np.sum(mask)),
                float(np.mean(times[mask])),
                float(np.median(times[mask])),
                float(np.mean(result.dark_population_all[mask])),
                float(np.mean(result.excited_population_all[mask])),
                float(np.mean(result.optical_coherence_norm_all[mask])),
                float(np.mean(np.linalg.norm(result.P_q_all[mask], axis=1))),
            )
        )
    dtype = [
        ("count", int),
        ("mean_time_s", float),
        ("median_time_s", float),
        ("mean_dark_population", float),
        ("mean_excited_population", float),
        ("mean_optical_coherence_norm", float),
        ("mean_Pq_norm_C_m2", float),
    ]
    return np.array(rows, dtype=dtype), edges


def linear_and_elecsus_reference(config):
    benchmark_config = BenchmarkConfig(
        temperature_C=config.temperature_C,
        k39_fraction=config.k39_fraction,
        saturation=config.saturation,
    )
    response = LinearThermalResponse(benchmark_config)
    axis = np.array([config.external_detuning_MHz])
    chi_linear = response.doppler_average(
        axis,
        B_z_G=config.B_z_G,
        delta_m=config.delta_m,
        integration_step_MHz=0.25,
        velocity_range_sigma=7.0,
    )[0]
    chi_elecsus = response.elecsus_chi(
        axis,
        config.B_z_G,
        config.delta_m,
    )[0]
    jones = si_jones_for_delta_m(config.delta_m)
    Ecal = ecal_amplitude_from_saturation(config.saturation) * jones
    Ecal_q = cart2spherical(Ecal)
    q_index = q_index_for_delta_m(config.delta_m)
    P_linear_q = np.zeros(3, dtype=complex)
    P_elecsus_q = np.zeros(3, dtype=complex)
    P_linear_q[q_index] = epsilon_0 * Ecal_q[q_index] * chi_linear
    P_elecsus_q[q_index] = epsilon_0 * Ecal_q[q_index] * chi_elecsus
    return {
        "chi_linear": complex(chi_linear),
        "chi_elecsus": complex(chi_elecsus),
        "P_linear_q": P_linear_q,
        "P_elecsus_q": P_elecsus_q,
    }


def pumping_time_1_minus_inv_e(model, v_z_m_s=0.0):
    times = np.concatenate(
        ([0.0], np.logspace(-10, 0, 1200))
    )
    curve = model.pumping_curve(times, v_z_m_s=v_z_m_s)
    initial = curve["dark_population"][0]
    target = initial + (1.0 - 1.0 / np.e) * (1.0 - initial)
    reached = np.flatnonzero(curve["dark_population"] >= target)
    if len(reached) == 0:
        return float("nan"), curve, target
    index = int(reached[0])
    if index == 0:
        return 0.0, curve, target
    t0, t1 = times[index - 1], times[index]
    p0 = curve["dark_population"][index - 1]
    p1 = curve["dark_population"][index]
    fraction = (target - p0) / (p1 - p0)
    tau = t0 + fraction * (t1 - t0)
    return float(tau), curve, float(target)


def pumping_time_dark_fraction(
    model,
    fraction_of_full_change=0.1,
    v_z_m_s=0.0,
):
    """dark population 完成 ``fraction*(1-p_initial)`` 时的早期时间尺度。"""
    fraction_of_full_change = float(fraction_of_full_change)
    if not 0.0 < fraction_of_full_change < 1.0:
        raise ValueError("fraction_of_full_change 必须位于 (0,1)")
    times = np.concatenate(([0.0], np.logspace(-10, 0, 1600)))
    curve = model.pumping_curve(times, v_z_m_s=v_z_m_s)
    initial = float(curve["dark_population"][0])
    target = initial + fraction_of_full_change * (1.0 - initial)
    reached = np.flatnonzero(curve["dark_population"] >= target)
    if len(reached) == 0:
        return float("nan"), curve, target
    index = int(reached[0])
    if index == 0:
        return 0.0, curve, target
    t0, t1 = times[index - 1], times[index]
    p0 = curve["dark_population"][index - 1]
    p1 = curve["dark_population"][index]
    tau = t0 + (target - p0) / (p1 - p0) * (t1 - t0)
    return float(tau), curve, float(target)


def save_ensemble_result(result, output_path, extra=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: value
        for key, value in result.__dict__.items()
        if value is not None
    }
    payload.update(
        {
            "q_values": np.asarray(Q_VALUES, dtype=int),
            "P_unit": np.array("C/m^2"),
            "number_density_unit": np.array("1/m^3"),
            "dipole_unit": np.array("C*m"),
            "electric_phasor_convention": np.array(PHASOR_CONVENTION),
            "polarization_phasor_convention": np.array(
                POLARIZATION_PHASOR_CONVENTION
            ),
        }
    )
    if extra:
        payload.update(extra)
    np.savez_compressed(output_path, **payload)
    return output_path
