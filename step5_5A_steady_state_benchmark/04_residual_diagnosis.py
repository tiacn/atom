# -*- coding: utf-8 -*-
"""测试 4：区分 ElecSus 数值舍入与真实 Hamiltonian 模型残差。"""

import numpy as np

from benchmark import BenchmarkConfig, DELTA_M_VALUES, LinearThermalResponse, spectrum_metrics


def main():
    print("测试 4：ElecSus numerical conventions residual diagnosis")
    response = LinearThermalResponse(BenchmarkConfig())
    axis = np.arange(-1400.0, 1400.1, 2.0)

    for B_z_G in (0.0, 300.0):
        for delta_m in DELTA_M_VALUES:
            reference = response.elecsus_chi(axis, B_z_G, delta_m)
            frozen = response.doppler_average(
                axis,
                B_z_G,
                delta_m,
                integration_step_MHz=0.25,
                velocity_range_sigma=7.0,
            )
            emulated = response.doppler_average(
                axis,
                B_z_G,
                delta_m,
                integration_step_MHz=0.25,
                velocity_range_sigma=7.0,
                emulate_elecsus_numerics=True,
            )
            frozen_metrics = spectrum_metrics(axis, frozen, reference)
            emulated_metrics = spectrum_metrics(axis, emulated, reference)
            explained_fraction = 1.0 - (
                emulated_metrics.relative_complex_l2
                / frozen_metrics.relative_complex_l2
            )
            print(
                f"B={B_z_G:5.0f} G, Delta-m={delta_m:+d}: "
                f"frozen={frozen_metrics.relative_complex_l2:.3e}, "
                f"ElecSus-numerics={emulated_metrics.relative_complex_l2:.3e}, "
                f"explained={explained_fraction:+.1%}"
            )

            assert np.max(emulated.imag) > 0.0
            assert emulated_metrics.complex_correlation > 0.999

    print("PASS：已分别量化舍入/弱线截断与剩余 Hamiltonian 模型残差")


if __name__ == "__main__":
    main()

