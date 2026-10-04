# -*- coding: utf-8 -*-
"""测试 3：固定热布居的一阶 OBE 与 ElecSus 做 B=0/非零 B 诊断。"""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmark import (
    BenchmarkConfig,
    DELTA_M_VALUES,
    LinearThermalResponse,
    spectrum_metrics,
)


def main():
    print("测试 3：linear thermal OBE vs ElecSus（根因诊断，不是 literal PASS）")
    config = BenchmarkConfig()
    response = LinearThermalResponse(config)
    axis = np.arange(-1400.0, 1400.1, 2.0)
    fields_G = (0.0, 100.0, 300.0)

    output = {
        "detuning_MHz": axis,
        "temperature_C": np.array(config.temperature_C),
        "k39_fraction": np.array(config.k39_fraction),
        "number_density_m3": np.array(response.number_density_m3),
        "sigma_v_m_s": np.array(config.sigma_v_m_s),
        "sigma_doppler_MHz": np.array(config.sigma_doppler_MHz),
    }
    all_metrics = {}

    for B_z_G in fields_G:
        for delta_m in DELTA_M_VALUES:
            model = response.doppler_average(
                axis,
                B_z_G=B_z_G,
                delta_m=delta_m,
                integration_step_MHz=0.25,
                velocity_range_sigma=7.0,
            )
            reference = response.elecsus_chi(axis, B_z_G, delta_m)
            metrics = spectrum_metrics(axis, model, reference)
            key = f"B{int(B_z_G):04d}_dm{'p' if delta_m > 0 else 'm'}1"
            output[f"chi_linear_{key}"] = model
            output[f"chi_elecsus_{key}"] = reference
            for metric_name, metric_value in asdict(metrics).items():
                output[f"metric_{key}_{metric_name}"] = np.array(metric_value)
            all_metrics[(B_z_G, delta_m)] = metrics

            print(
                f"B={B_z_G:6.1f} G, Delta-m={delta_m:+d}: "
                f"complex={metrics.relative_complex_l2:.3e}, "
                f"Re={metrics.relative_real_l2:.3e}, "
                f"Im={metrics.relative_imag_l2:.3e}, "
                f"peak dnu={metrics.peak_position_error_MHz:+.2f} MHz, "
                f"peak amp={metrics.peak_imag_relative_error:+.3e}, "
                f"FWHM diff={metrics.fwhm_error_MHz:+.2f} MHz, "
                f"corr={metrics.complex_correlation:.8f}"
            )
            assert np.max(model.imag) > 0.0
            assert np.max(reference.imag) > 0.0

    # B=0 两个 Delta-m 分量必须互相简并。
    plus_zero = output["chi_linear_B0000_dmp1"]
    minus_zero = output["chi_linear_B0000_dmm1"]
    assert np.linalg.norm(plus_zero - minus_zero) / np.linalg.norm(plus_zero) < 1e-12

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "linear_thermal_vs_elecsus.npz"
    np.savez_compressed(output_path, **output)
    print(f"saved: {output_path}")

    worst_complex = max(metric.relative_complex_l2 for metric in all_metrics.values())
    worst_peak = max(abs(metric.peak_position_error_MHz) for metric in all_metrics.values())
    print(f"worst complex L2 error = {worst_complex:.3e}")
    print(f"worst sampled peak position error = {worst_peak:.2f} MHz")

    # 这是诊断阈值：若不满足，优先检查 q mapping/常数/线强，而不是调整比例。
    assert worst_complex < 0.05
    assert worst_peak <= 4.0
    print("PASS（诊断）：固定热布居的一阶 OBE 在 B=0/100/300 G 与 ElecSus 一致")
    print("注意：此结果不能消除 literal full steady-state 的暗态 FAIL")


if __name__ == "__main__":
    main()

