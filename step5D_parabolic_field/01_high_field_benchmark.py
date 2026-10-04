# -*- coding: utf-8 -*-
"""Step 5D 测试 1：2000--2500 G fixed-thermal OBE vs ElecSus。"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import path_setup  # noqa: F401

from benchmark import BenchmarkConfig, LinearThermalResponse, spectrum_metrics


def main():
    print("Step 5D 测试 1：high-field local benchmark")
    config = BenchmarkConfig(saturation=1.0e-6)
    response = LinearThermalResponse(config)
    fields = np.arange(2000.0, 2500.1, 100.0)
    transition_ranges = [response.transition_lines(B, +1)[0] for B in fields]
    lower = min(float(np.min(values)) for values in transition_ranges)
    upper = max(float(np.max(values)) for values in transition_ranges)
    margin = 7.0 * config.sigma_doppler_MHz + 50.0
    axis = np.arange(np.floor(lower - margin), np.ceil(upper + margin) + 2.0, 2.0)

    chi_obe = np.empty((len(fields), len(axis)), dtype=complex)
    chi_elecsus = np.empty_like(chi_obe)
    metrics = []
    for index, B_G in enumerate(fields):
        chi_obe[index] = response.doppler_average(
            axis,
            B_z_G=B_G,
            delta_m=+1,
            integration_step_MHz=0.25,
            velocity_range_sigma=7.0,
        )
        chi_elecsus[index] = response.elecsus_chi(axis, B_G, +1)
        metric = spectrum_metrics(axis, chi_obe[index], chi_elecsus[index])
        metrics.append(asdict(metric))
        print(
            f"B={B_G:.0f} G: complex L2={metric.relative_complex_l2:.6e}, "
            f"peak error={metric.peak_position_error_MHz:+.3f} MHz, "
            f"corr={metric.complex_correlation:.9f}"
        )

    worst_complex = max(item["relative_complex_l2"] for item in metrics)
    worst_peak = max(abs(item["peak_position_error_MHz"]) for item in metrics)
    minimum_correlation = min(item["complex_correlation"] for item in metrics)
    assert worst_complex < 5.0e-2
    assert worst_peak <= 4.0
    assert minimum_correlation > 0.999

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "high_field_local_benchmark.npz",
        fields_G=fields,
        detuning_MHz=axis,
        chi_obe=chi_obe,
        chi_elecsus=chi_elecsus,
        metric_names=np.asarray(tuple(metrics[0])),
        metric_values=np.asarray(
            [[item[name] for name in metrics[0]] for item in metrics],
            dtype=float,
        ),
        worst_complex_relative_error=np.array(worst_complex),
        worst_peak_position_error_MHz=np.array(worst_peak),
        minimum_complex_correlation=np.array(minimum_correlation),
    )
    summary = {
        "fields_G": fields.tolist(),
        "detuning_axis_MHz": [float(axis[0]), float(axis[-1]), 2.0],
        "worst_complex_relative_error": worst_complex,
        "worst_peak_position_error_MHz": worst_peak,
        "minimum_complex_correlation": minimum_correlation,
        "per_field": metrics,
    }
    (output_dir / "high_field_benchmark.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"worst complex relative error = {worst_complex:.6e}")
    print("PASS：2000--2500 G fixed-thermal linear OBE/ElecSus benchmark")


if __name__ == "__main__":
    main()
