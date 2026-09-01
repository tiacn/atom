# -*- coding: utf-8 -*-
"""测试 7：代表性单切片中，多 seed、10/20/50/100/200 原子的 MC 收敛。"""

import os

import numpy as np

from ensemble import (
    ensemble_observables,
    evolve_ensemble,
    validate_density_collection,
)
from moving_atom import SingleAtomEngine, SliceFields
from test_support import make_ensemble


FULL_SAMPLE_SIZES = (10, 20, 50, 100, 200)
FULL_SEEDS = tuple(range(45701, 45711))
QUICK_SEEDS = FULL_SEEDS[:3]


def dispersion(matrices):
    matrices = np.asarray(matrices)
    if len(matrices) < 2:
        raise ValueError("dispersion 至少需要两个 seed")
    center = np.mean(matrices, axis=0)
    squared = np.abs(matrices - center) ** 2
    degrees_of_freedom = (len(matrices) - 1) * int(np.prod(matrices.shape[1:]))
    return float(np.sqrt(np.sum(squared) / degrees_of_freedom))


def overall_decrease(sample_sizes, values, label):
    """检查总体趋势，不要求有限 seed 下每一对相邻 N 都单调。"""
    sizes = np.asarray(sample_sizes, dtype=float)
    values = np.array([values[size] for size in sample_sizes], dtype=float)
    slope = float(np.polyfit(np.log(sizes), np.log(values), 1)[0])
    small_N_level = float(np.exp(np.mean(np.log(values[:2]))))
    large_N_level = float(np.exp(np.mean(np.log(values[-3:]))))
    print(
        f"  {label}: log-log slope={slope:+.3f}, "
        f"large-N/small-N={large_N_level/small_N_level:.3f}"
    )
    assert slope < 0.0
    assert large_N_level < small_N_level
    assert values[-1] < values[0]


def main():
    quick = os.environ.get("STEP45_MC_QUICK", "0") == "1"
    sample_sizes = FULL_SAMPLE_SIZES
    seeds = QUICK_SEEDS if quick else FULL_SEEDS
    maximum = max(sample_sizes)

    print("测试 7：MC convergence")
    print(f"mode = {'quick' if quick else 'full'}")
    print(f"sample sizes = {sample_sizes}")
    print(f"seeds = {seeds}")
    print("spatial bins = 1 representative slice")

    fields = SliceFields.uniform(
        1,
        B_z_G=300.0,
        saturation=0.03,
        polarization_cart=(1.0, 0.0, 0.0),
        laser_detuning_MHz=40.0,
    )

    estimates = {size: [] for size in sample_sizes}
    excited = {size: [] for size in sample_sizes}
    optical = {size: [] for size in sample_sizes}

    for seed in seeds:
        # 每个 seed 是独立重复实验。使用独立 engine，避免把前一 seed 中
        # 几乎不会复用的 velocity-dependent L 缓存保留到后续 seed。
        # 这只限定测试进程的资源生命周期，不改变任何单原子演化逻辑。
        engine = SingleAtomEngine()
        print(f"seed = {seed}", flush=True)
        _, trajectories, _, _ = make_ensemble(
            n_slices=1,
            atoms_per_slice=maximum,
            seed=seed,
        )
        result = evolve_ensemble(
            trajectories,
            fields,
            engine,
            progress_every=25 if not quick else 0,
        )
        validate_density_collection(result.rho_final_all, engine.N_G)

        # 使用同一个最大样本的前缀，避免不同 N 之间额外引入不相关抽样噪声。
        for size in sample_sizes:
            rho_bar = np.mean(result.rho_final_all[:size], axis=0, keepdims=True)
            obs = ensemble_observables(rho_bar, engine.N_G)
            estimates[size].append(rho_bar[0])
            excited[size].append(obs["excited_population"][0])
            optical[size].append(obs["optical_coherence_norm"][0])

        del result, engine

    matrix_dispersion = {}
    excited_dispersion = {}
    optical_dispersion = {}
    for size in sample_sizes:
        matrix_dispersion[size] = dispersion(estimates[size])
        excited_dispersion[size] = float(np.std(excited[size], ddof=1))
        optical_dispersion[size] = float(np.std(optical[size], ddof=1))
        print(
            f"N={size:3d}: full-rho dispersion={matrix_dispersion[size]:.3e}, "
            f"rho_ee std={excited_dispersion[size]:.3e}, "
            f"optical std={optical_dispersion[size]:.3e}; "
            f"sqrt(N)*dispersion: "
            f"rho={matrix_dispersion[size]*np.sqrt(size):.3e}, "
            f"rho_ee={excited_dispersion[size]*np.sqrt(size):.3e}, "
            f"optical={optical_dispersion[size]*np.sqrt(size):.3e}"
        )

    print("overall trend checks (adjacent N need not be monotonic):")
    overall_decrease(sample_sizes, matrix_dispersion, "full rho")
    overall_decrease(sample_sizes, excited_dispersion, "rho_ee")
    overall_decrease(sample_sizes, optical_dispersion, "optical coherence")
    print(
        f"PASS: {len(seeds)} 个 seed 下三类统计误差总体随 N 下降；"
        "sqrt(N) 缩放量已输出，仅用于观察 MC 标度"
    )


if __name__ == "__main__":
    main()
