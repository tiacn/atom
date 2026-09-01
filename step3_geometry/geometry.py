# -*- coding: utf-8 -*-
"""有限圆柱中的位置、速度、入口和历史切片算法。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import K_B, M_K39


@dataclass(frozen=True)
class EntryResult:
    time_s: float
    point_m: np.ndarray
    surface: str


@dataclass(frozen=True)
class SliceSegment:
    slice_index: int
    dt_s: float
    start_m: np.ndarray
    end_m: np.ndarray


@dataclass(frozen=True)
class Trajectory:
    current_position_m: np.ndarray
    velocity_m_s: np.ndarray
    current_slice: int
    entry: EntryResult
    segments: tuple[SliceSegment, ...]

    @property
    def slice_indices(self):
        return np.array([item.slice_index for item in self.segments], dtype=int)

    @property
    def dt_slices(self):
        return np.array([item.dt_s for item in self.segments], dtype=float)


def slice_edges(cell_length_m, n_slices):
    return np.linspace(0.0, float(cell_length_m), int(n_slices) + 1)


def sample_positions_by_slice(
    rng,
    n_slices,
    atoms_per_slice,
    beam_radius_m,
    cell_length_m,
):
    """在每个切片的 top-hat 圆盘体积中均匀采样当前位置。"""
    n_slices = int(n_slices)
    atoms_per_slice = int(atoms_per_slice)
    n_atoms = n_slices * atoms_per_slice

    indices = np.repeat(np.arange(n_slices), atoms_per_slice)

    # 面积均匀圆盘：r^2 在 [0,R^2] 上均匀，而不是 r 本身均匀。
    radius = float(beam_radius_m) * np.sqrt(rng.random(n_atoms))
    azimuth = 2.0 * np.pi * rng.random(n_atoms)
    x = radius * np.cos(azimuth)
    y = radius * np.sin(azimuth)

    local_z_fraction = rng.random(n_atoms)
    dz = float(cell_length_m) / n_slices
    z = (indices + local_z_fraction) * dz

    return np.column_stack((x, y, z)), indices


def sample_maxwell_boltzmann_velocities(
    rng,
    n_atoms,
    temperature_C,
    atom_mass_kg=M_K39,
):
    """采样三维 Maxwell-Boltzmann 速度；每个分量独立为高斯分布。"""
    temperature_K = float(temperature_C) + 273.15
    sigma = np.sqrt(K_B * temperature_K / float(atom_mass_kg))
    velocities = rng.normal(0.0, sigma, size=(int(n_atoms), 3))
    return velocities, sigma


def is_inside_finite_cylinder(point, beam_radius_m, cell_length_m, tolerance=0.0):
    point = np.asarray(point, dtype=float)
    radial_sq = point[0] ** 2 + point[1] ** 2
    return bool(
        radial_sq <= (beam_radius_m + tolerance) ** 2
        and -tolerance <= point[2] <= cell_length_m + tolerance
    )


def find_previous_entry(
    current_position_m,
    velocity_m_s,
    beam_radius_m,
    cell_length_m,
    velocity_tolerance=1e-14,
):
    """沿 r-v*t 反向追踪，返回首先遇到的合法有限圆柱边界。"""
    r = np.asarray(current_position_m, dtype=float)
    v = np.asarray(velocity_m_s, dtype=float)
    radius = float(beam_radius_m)
    length = float(cell_length_m)

    if not is_inside_finite_cylinder(r, radius, length, tolerance=1e-12):
        raise ValueError("current_position_m 不在有限光束柱内")
    if np.linalg.norm(v) <= velocity_tolerance:
        raise ValueError("速度过小，无法定义反向入口")

    x, y, z = r
    vx, vy, vz = v
    candidates = []
    spatial_tolerance = max(radius, length) * 1e-10

    # 侧壁：(x-vx*t)^2+(y-vy*t)^2=R^2。
    transverse_speed_sq = vx * vx + vy * vy
    if transverse_speed_sq > velocity_tolerance**2:
        dot_rv = x * vx + y * vy
        discriminant_part = (
            dot_rv * dot_rv
            + transverse_speed_sq * (radius * radius - x * x - y * y)
        )
        if discriminant_part >= -spatial_tolerance**2 * transverse_speed_sq:
            discriminant_part = max(discriminant_part, 0.0)
            t_side = (dot_rv + np.sqrt(discriminant_part)) / transverse_speed_sq
            if t_side > 0.0:
                point_side = r - v * t_side
                if -spatial_tolerance <= point_side[2] <= length + spatial_tolerance:
                    candidates.append((t_side, "side", point_side))

    # 反向运动沿 -v。若 vz>0，过去方向朝 z=0；若 vz<0，朝 z=L。
    if vz > velocity_tolerance:
        t_z0 = z / vz
        if t_z0 > 0.0:
            point_z0 = r - v * t_z0
            if point_z0[0] ** 2 + point_z0[1] ** 2 <= (
                radius + spatial_tolerance
            ) ** 2:
                candidates.append((t_z0, "z0", point_z0))
    elif vz < -velocity_tolerance:
        t_zL = (z - length) / vz
        if t_zL > 0.0:
            point_zL = r - v * t_zL
            if point_zL[0] ** 2 + point_zL[1] ** 2 <= (
                radius + spatial_tolerance
            ) ** 2:
                candidates.append((t_zL, "zL", point_zL))

    if not candidates:
        raise RuntimeError("没有找到合法入口；有限圆柱求交算法异常")

    # 当前点在凸区域内部，反向射线首先遇到的边界就是入口。
    time_s, surface, point_m = min(candidates, key=lambda item: item[0])
    return EntryResult(float(time_s), np.asarray(point_m), surface)


def split_history_into_slices(
    entry,
    current_position_m,
    velocity_m_s,
    z_edges_m,
):
    """把 entry -> current 的历史线段按 z 切片分段。"""
    current = np.asarray(current_position_m, dtype=float)
    velocity = np.asarray(velocity_m_s, dtype=float)
    edges = np.asarray(z_edges_m, dtype=float)
    duration = float(entry.time_s)

    if duration <= 0.0:
        raise ValueError("入口飞行时间必须为正")

    start = np.asarray(entry.point_m, dtype=float)
    endpoint_error = np.max(np.abs(start + velocity * duration - current))
    if endpoint_error > 1e-9:
        raise ValueError(f"入口、速度和当前位置不一致，误差 {endpoint_error}")

    cut_times = [0.0, duration]
    vz = velocity[2]
    if abs(vz) > 1e-14:
        for boundary_z in edges[1:-1]:
            crossing_time = (boundary_z - start[2]) / vz
            if 1e-15 < crossing_time < duration - 1e-15:
                cut_times.append(float(crossing_time))

    cut_times = np.array(sorted(cut_times))
    segments = []
    n_slices = len(edges) - 1

    for t_start, t_end in zip(cut_times[:-1], cut_times[1:]):
        dt = t_end - t_start
        if dt <= 0.0:
            continue

        point_start = start + velocity * t_start
        point_end = start + velocity * t_end
        point_mid = start + velocity * (0.5 * (t_start + t_end))
        index = int(np.searchsorted(edges, point_mid[2], side="right") - 1)
        index = int(np.clip(index, 0, n_slices - 1))

        segments.append(
            SliceSegment(
                slice_index=index,
                dt_s=float(dt),
                start_m=point_start,
                end_m=point_end,
            )
        )

    if not segments:
        raise RuntimeError("入口到当前位置没有生成任何切片段")
    return tuple(segments)


def generate_ensemble(config):
    """按目标配置生成完整几何轨迹系综。"""
    rng = np.random.default_rng(config.seed)
    positions, current_slices = sample_positions_by_slice(
        rng,
        config.n_slices,
        config.atoms_per_slice,
        config.beam_radius_m,
        config.cell_length_m,
    )
    velocities, sigma = sample_maxwell_boltzmann_velocities(
        rng,
        len(positions),
        config.temperature_C,
    )
    edges = slice_edges(config.cell_length_m, config.n_slices)

    trajectories = []
    for position, velocity, current_slice in zip(
        positions,
        velocities,
        current_slices,
    ):
        entry = find_previous_entry(
            position,
            velocity,
            config.beam_radius_m,
            config.cell_length_m,
        )
        segments = split_history_into_slices(
            entry,
            position,
            velocity,
            edges,
        )
        trajectories.append(
            Trajectory(
                current_position_m=position,
                velocity_m_s=velocity,
                current_slice=int(current_slice),
                entry=entry,
                segments=segments,
            )
        )

    return tuple(trajectories), edges, sigma

