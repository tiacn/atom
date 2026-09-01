# -*- coding: utf-8 -*-
"""测试 1：按原要求求 full-L 稳态，识别无限时间暗态抽运。"""

from pathlib import Path

import numpy as np

from benchmark import (
    BenchmarkConfig,
    DELTA_M_VALUES,
    LinearThermalResponse,
    LiteralSteadyStateSolver,
)


def main():
    print("测试 1：literal full-L steady state")
    config = BenchmarkConfig()
    solver = LiteralSteadyStateSolver(config)
    linear = LinearThermalResponse(config)
    axis = np.arange(-1200.0, 1200.1, 2.0)
    saturations = (1e-4, 1e-6, 1e-8)
    gh_nodes = (8, 16, 32)
    fields_nonzero_G = (100.0, 300.0)
    literal_chi = np.zeros((2, len(saturations)), dtype=complex)
    dark_populations = np.zeros((2, len(saturations)))
    gh_chi = np.zeros((2, len(gh_nodes)), dtype=complex)
    nonzero_B_chi = np.zeros((2, len(fields_nonzero_G)), dtype=complex)
    nonzero_B_dark = np.zeros((2, len(fields_nonzero_G)))
    peak_detunings = np.zeros((2, 1 + len(fields_nonzero_G)))

    for dm_index, delta_m in enumerate(DELTA_M_VALUES):
        elecsus = linear.elecsus_chi(axis, 0.0, delta_m)
        peak_index = int(np.argmax(elecsus.imag))
        peak_detuning = float(axis[peak_index])
        peak_detunings[dm_index, 0] = peak_detuning
        dark_index = solver.dark_state_index(delta_m)
        print(f"Delta-m={delta_m:+d}, ElecSus peak={peak_detuning:+.1f} MHz")

        for saturation_index, saturation in enumerate(saturations):
            point = solver.solve(
                B_z_G=0.0,
                detuning_MHz=peak_detuning,
                v_z_m_s=0.0,
                delta_m=delta_m,
                saturation=saturation,
            )
            dark_population = float(point.rho[dark_index, dark_index].real)
            literal_chi[dm_index, saturation_index] = point.chi
            dark_populations[dm_index, saturation_index] = dark_population
            print(
                f"  s={saturation:.0e}: chi={point.chi}, "
                f"dark population={dark_population:.12f}, "
                f"||Lrho||={point.liouvillian_residual:.3e}"
            )
            assert abs(point.chi) < 1e-20
            assert abs(dark_population - 1.0) < 1e-10
            assert point.trace_error < 1e-10
            assert point.hermiticity_error < 1e-10
            assert point.minimum_eigenvalue > -1e-10

        gh_results = []
        for node_index, n_nodes in enumerate(gh_nodes):
            averaged = solver.gauss_hermite_average_at_detuning(
                B_z_G=0.0,
                detuning_MHz=peak_detuning,
                delta_m=delta_m,
                n_nodes=n_nodes,
                saturation=1e-6,
            )
            gh_results.append(averaged)
            gh_chi[dm_index, node_index] = averaged
            print(f"  GH nodes={n_nodes:2d}: <chi>={averaged}")
            assert abs(averaged) < 1e-20

        elecsus_peak = elecsus[peak_index]
        relative_error = abs(gh_results[-1] - elecsus_peak) / abs(elecsus_peak)
        print(f"  literal-vs-ElecSus peak relative error = {relative_error:.6f}")
        assert abs(relative_error - 1.0) < 1e-12

        # B=0 已经使正式 PASS 判据失败；继续抽查两个非零 uniform B，确认
        # stretched dark state 在 Faraday geometry 下仍然存在，而不是零场简并伪象。
        for field_index, B_z_G in enumerate(fields_nonzero_G):
            elecsus_B = linear.elecsus_chi(axis, B_z_G, delta_m)
            peak_B = float(axis[int(np.argmax(elecsus_B.imag))])
            peak_detunings[dm_index, field_index + 1] = peak_B
            point_B = solver.solve(
                B_z_G=B_z_G,
                detuning_MHz=peak_B,
                v_z_m_s=0.0,
                delta_m=delta_m,
                saturation=1e-6,
            )
            dark_population_B = float(point_B.rho[dark_index, dark_index].real)
            nonzero_B_chi[dm_index, field_index] = point_B.chi
            nonzero_B_dark[dm_index, field_index] = dark_population_B
            print(
                f"  B={B_z_G:.0f} G: peak={peak_B:+.1f} MHz, "
                f"chi={point_B.chi}, dark population={dark_population_B:.12f}"
            )
            assert abs(point_B.chi) < 1e-20
            assert abs(dark_population_B - 1.0) < 1e-10

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "literal_steady_state_dark_test.npz"
    np.savez_compressed(
        output_path,
        delta_m=np.asarray(DELTA_M_VALUES),
        saturations=np.asarray(saturations),
        literal_chi=literal_chi,
        dark_populations=dark_populations,
        gauss_hermite_nodes=np.asarray(gh_nodes),
        gauss_hermite_chi=gh_chi,
        nonzero_fields_G=np.asarray(fields_nonzero_G),
        nonzero_B_chi=nonzero_B_chi,
        nonzero_B_dark_populations=nonzero_B_dark,
        peak_detunings_MHz=peak_detunings,
        temperature_C=np.array(config.temperature_C),
        k39_fraction=np.array(config.k39_fraction),
    )
    print(f"saved: {output_path}")

    print("PASS（诊断）：数值稳态、弱光和 Doppler 节点均确认暗态抽运")
    print("STEP 5.5A PASS CRITERION: FAIL（literal steady state 不能复现 ElecSus）")


if __name__ == "__main__":
    main()
