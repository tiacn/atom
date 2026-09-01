# -*- coding: utf-8 -*-
"""多个第三步脚本共用的几何不变量检查。"""

import numpy as np

from geometry import is_inside_finite_cylinder


def validate_entry(entry, current, velocity, radius, length):
    current = np.asarray(current)
    velocity = np.asarray(velocity)
    point = entry.point_m
    tolerance = max(radius, length) * 1e-8

    reconstruction_error = np.max(
        np.abs(point + velocity * entry.time_s - current)
    )
    assert reconstruction_error < tolerance

    radial = np.hypot(point[0], point[1])
    if entry.surface == "side":
        assert abs(radial - radius) < tolerance
        assert -tolerance <= point[2] <= length + tolerance
    elif entry.surface == "z0":
        assert abs(point[2]) < tolerance
        assert radial <= radius + tolerance
    elif entry.surface == "zL":
        assert abs(point[2] - length) < tolerance
        assert radial <= radius + tolerance
    else:
        raise AssertionError(f"未知入口类型 {entry.surface}")

    # 凸圆柱中，入口与当前点之间的整条线段都应在内部。
    for fraction in (0.1, 0.25, 0.5, 0.75, 0.9):
        interior_point = point + fraction * (current - point)
        assert is_inside_finite_cylinder(
            interior_point,
            radius,
            length,
            tolerance=tolerance,
        )

    # 再向过去移动一个极小时间，应已经在圆柱外。
    probe_dt = max(entry.time_s * 1e-6, 1e-15)
    before_entry = current - velocity * (entry.time_s + probe_dt)
    assert not is_inside_finite_cylinder(
        before_entry,
        radius,
        length,
        tolerance=0.0,
    )


def validate_slice_history(trajectory, z_edges):
    segments = trajectory.segments
    tolerance = max(z_edges[-1], 1.0) * 1e-10

    assert len(segments) >= 1
    assert all(segment.dt_s > 0.0 for segment in segments)
    assert abs(sum(segment.dt_s for segment in segments) - trajectory.entry.time_s) < 1e-12
    assert np.max(np.abs(segments[0].start_m - trajectory.entry.point_m)) < tolerance
    assert np.max(
        np.abs(segments[-1].end_m - trajectory.current_position_m)
    ) < tolerance
    assert segments[-1].slice_index == trajectory.current_slice

    for first, second in zip(segments[:-1], segments[1:]):
        assert np.max(np.abs(first.end_m - second.start_m)) < tolerance
        assert abs(second.slice_index - first.slice_index) == 1

    indices = trajectory.slice_indices
    vz = trajectory.velocity_m_s[2]
    if vz > 0.0:
        assert np.all(np.diff(indices) >= 0)
    elif vz < 0.0:
        assert np.all(np.diff(indices) <= 0)

    for segment in segments:
        midpoint_z = 0.5 * (segment.start_m[2] + segment.end_m[2])
        lo = z_edges[segment.slice_index]
        hi = z_edges[segment.slice_index + 1]
        assert lo - tolerance <= midpoint_z <= hi + tolerance

