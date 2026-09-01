# -*- coding: utf-8 -*-
"""第 1 小步：验证有限光束柱内的位置采样。"""

import numpy as np

from config import CONFIG
from geometry import sample_positions_by_slice


def main():
    print("第 1 小步：位置采样")
    rng = np.random.default_rng(CONFIG.seed)
    positions, indices = sample_positions_by_slice(
        rng,
        CONFIG.n_slices,
        200,
        CONFIG.beam_radius_m,
        CONFIG.cell_length_m,
    )

    radial_sq = positions[:, 0] ** 2 + positions[:, 1] ** 2
    counts = np.bincount(indices, minlength=CONFIG.n_slices)
    dz = CONFIG.cell_length_m / CONFIG.n_slices
    local_z = positions[:, 2] / dz - indices

    radial_second_moment = np.mean(radial_sq)
    expected_radial_second_moment = CONFIG.beam_radius_m**2 / 2.0
    radial_relative_error = abs(
        radial_second_moment / expected_radial_second_moment - 1.0
    )

    print(f"sample count = {len(positions)}")
    print(f"atoms per slice min/max = {counts.min()}/{counts.max()}")
    print(f"max radius = {np.sqrt(radial_sq.max())*1e3:.6f} mm")
    print(f"mean x = {np.mean(positions[:,0])*1e6:.3f} um")
    print(f"mean y = {np.mean(positions[:,1])*1e6:.3f} um")
    print(f"E[r^2] relative error = {radial_relative_error:.3e}")
    print(f"mean local-z fraction = {np.mean(local_z):.6f}")
    print(f"var local-z fraction = {np.var(local_z):.6f}")

    assert np.all(counts == 200)
    assert np.all(radial_sq <= CONFIG.beam_radius_m**2)
    assert np.all((positions[:, 2] >= 0.0) & (positions[:, 2] <= CONFIG.cell_length_m))
    assert abs(np.mean(positions[:, 0])) < 0.01 * CONFIG.beam_radius_m
    assert abs(np.mean(positions[:, 1])) < 0.01 * CONFIG.beam_radius_m
    assert radial_relative_error < 0.02
    assert abs(np.mean(local_z) - 0.5) < 0.01
    assert abs(np.var(local_z) - 1.0 / 12.0) < 0.003

    print("PASS: 每切片位置和横向圆盘均为均匀采样")


if __name__ == "__main__":
    main()

