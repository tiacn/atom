# -*- coding: utf-8 -*-
"""第 2 小步：验证 K39 Maxwell-Boltzmann 速度采样。"""

import numpy as np

from config import CONFIG
from geometry import sample_maxwell_boltzmann_velocities


def main():
    print("第 2 小步：Maxwell-Boltzmann 速度")
    rng = np.random.default_rng(CONFIG.seed + 1)
    velocities, sigma = sample_maxwell_boltzmann_velocities(
        rng,
        200_000,
        CONFIG.temperature_C,
    )
    speeds = np.linalg.norm(velocities, axis=1)

    means = np.mean(velocities, axis=0)
    standard_deviations = np.std(velocities, axis=0)
    correlation = np.corrcoef(velocities, rowvar=False)
    off_diagonal = correlation - np.eye(3)

    expected_mean_speed = 2.0 * sigma * np.sqrt(2.0 / np.pi)
    expected_mean_speed_sq = 3.0 * sigma**2
    mean_speed_relative_error = abs(np.mean(speeds) / expected_mean_speed - 1.0)
    speed_sq_relative_error = abs(
        np.mean(speeds**2) / expected_mean_speed_sq - 1.0
    )

    direction_second_moment = np.mean(
        (velocities / speeds[:, None]) ** 2,
        axis=0,
    )

    print(f"temperature = {CONFIG.temperature_C:.1f} C")
    print(f"1D sigma = {sigma:.3f} m/s")
    print(f"component means = {means}")
    print(f"component std = {standard_deviations}")
    print(f"max component correlation = {np.max(np.abs(off_diagonal)):.3e}")
    print(f"mean speed = {np.mean(speeds):.3f} m/s")
    print(f"mean-speed relative error = {mean_speed_relative_error:.3e}")
    print(f"mean-speed-squared relative error = {speed_sq_relative_error:.3e}")
    print(f"direction E[n_i^2] = {direction_second_moment}")

    assert np.max(np.abs(means)) < 0.01 * sigma
    assert np.max(np.abs(standard_deviations / sigma - 1.0)) < 0.01
    assert np.max(np.abs(off_diagonal)) < 0.01
    assert mean_speed_relative_error < 0.005
    assert speed_sq_relative_error < 0.005
    assert np.max(np.abs(direction_second_moment - 1.0 / 3.0)) < 0.005

    print("PASS: 速度大小、分量独立性和方向各向同性符合 Maxwell 理论")


if __name__ == "__main__":
    main()

