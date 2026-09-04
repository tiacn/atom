# -*- coding: utf-8 -*-
"""Step 6A 测试 1：频率选择、完整 256 维传播 controls 和弱抽运诊断。"""

import json
from pathlib import Path

import numpy as np

from magnetic_memory import (
    FullMagneticMemoryModel,
    MagneticMemoryConfig,
    run_controls,
)


def main():
    print("Step 6A 测试 1：peak + controls")
    model = FullMagneticMemoryModel(
        past_B_G=0.0,
        current_B_G=300.0,
        config=MagneticMemoryConfig(),
    )
    controls = run_controls(model)
    print(
        "external/internal detuning = "
        f"{model.external_detuning_MHz:+.6f}/"
        f"{model.internal_detuning_MHz:+.6f} MHz"
    )
    for key, value in controls.items():
        print(f"{key} = {value:.9e}")

    assert model.N**2 == 256
    assert controls["identical_history_error"] < 1.0e-14
    assert controls["uniform_L_error"] < 1.0e-10
    assert controls["segment_splitting_error"] < 1.0e-10
    assert controls["coherence_plateau_relative"] < 1.0e-3
    assert controls["linear_response_relative"] < 2.0e-3
    # 当前值约 0.61%：ground population redistribution 比完整历史差异
    # 小 160 倍以上，满足“远小于”，但不人为要求达到千分之一。
    assert controls["ground_change_to_initial_history_ratio"] < 1.0e-2
    assert controls["trace_error"] < 1.0e-9
    assert controls["hermiticity_error"] < 1.0e-9
    assert controls["minimum_eigenvalue"] > -1.0e-10
    assert np.isfinite(controls["normalized_commutator"])

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "controls.json"
    payload = {
        "hilbert_dimension": model.N,
        "liouville_dimension": model.N**2,
        "past_B_G": model.past_B_G,
        "current_B_G": model.current_B_G,
        "temperature_C": model.config.temperature_C,
        "saturation": model.config.saturation,
        "delta_m": model.config.delta_m,
        "T_pre_us": model.config.T_pre_s * 1.0e6,
        "external_detuning_MHz": model.external_detuning_MHz,
        "internal_detuning_MHz": model.internal_detuning_MHz,
        **controls,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved: {output_path}")
    print("PASS：完整 256 维 propagation controls 和 weak-pumping diagnostic")


if __name__ == "__main__":
    main()
