# -*- coding: utf-8 -*-
"""测试 2：固定热布居线性诊断的确定性 Doppler 积分收敛。"""

from pathlib import Path

import numpy as np

from benchmark import BenchmarkConfig, LinearThermalResponse


def relative_error(value, reference):
    return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))


def main():
    print("测试 2：deterministic Doppler integration convergence")
    config = BenchmarkConfig()
    response = LinearThermalResponse(config)
    axis = np.arange(-1200.0, 1200.1, 4.0)
    reference = response.doppler_average(
        axis,
        B_z_G=0.0,
        delta_m=+1,
        integration_step_MHz=0.125,
        velocity_range_sigma=8.0,
    )

    step_errors = {}
    for step in (4.0, 2.0, 1.0, 0.5, 0.25):
        value = response.doppler_average(
            axis,
            B_z_G=0.0,
            delta_m=+1,
            integration_step_MHz=step,
            velocity_range_sigma=8.0,
        )
        step_errors[step] = relative_error(value, reference)
        print(f"integration step={step:5.3f} MHz: relative error={step_errors[step]:.3e}")

    range_errors = {}
    for range_sigma in (3.0, 4.0, 5.0, 6.0, 7.0):
        value = response.doppler_average(
            axis,
            B_z_G=0.0,
            delta_m=+1,
            integration_step_MHz=0.25,
            velocity_range_sigma=range_sigma,
        )
        range_errors[range_sigma] = relative_error(value, reference)
        print(
            f"velocity range=+/-{range_sigma:.0f} sigma: "
            f"relative error={range_errors[range_sigma]:.3e}"
        )

    assert step_errors[0.25] < 2e-6
    assert step_errors[0.5] < step_errors[1.0] < step_errors[2.0]
    assert range_errors[7.0] < 2e-6
    assert range_errors[6.0] < range_errors[5.0] < range_errors[4.0]
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "velocity_integration_convergence.npz"
    np.savez_compressed(
        output_path,
        detuning_MHz=axis,
        reference_chi=reference,
        integration_steps_MHz=np.asarray(tuple(step_errors)),
        step_relative_errors=np.asarray(tuple(step_errors.values())),
        velocity_ranges_sigma=np.asarray(tuple(range_errors)),
        range_relative_errors=np.asarray(tuple(range_errors.values())),
        reference_step_MHz=np.array(0.125),
        reference_range_sigma=np.array(8.0),
    )
    print(f"saved: {output_path}")
    print("PASS：0.25 MHz、+/-7 sigma 的确定性积分已数值收敛")


if __name__ == "__main__":
    main()
