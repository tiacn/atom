# -*- coding: utf-8 -*-
"""Step 5A 的物理结果可视化；不参与原有 run_all.py 验收。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.linalg import expm, solve
from scipy.optimize import brentq

from benchmark import (
    BenchmarkConfig,
    DELTA_M_VALUES,
    LiteralSteadyStateSolver,
    external_to_internal_detuning_MHz,
    si_jones_for_delta_m,
)
from conventions import (
    ecal_amplitude_from_saturation,
    normalized_pylcp_Eq_from_si_phasor,
)
from k39_model import detuning_MHz_to_rad_s
from liouvillian import (
    coherent_liouvillian,
    physical_hamiltonian_from_fields,
    vec_to_rho,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "visualization"
SATURATIONS = np.array([1e-4, 1e-6, 1e-8])
COLORS = ("#1768ac", "#d85828", "#5d9632")


def setup_chinese_font():
    candidates = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun")
    available = {font.name for font in font_manager.fontManager.ttflist}
    found = next((font for font in candidates if font in available), None)
    if found is None:
        raise RuntimeError("缺少中文 Matplotlib 字体；请安装微软雅黑、黑体或 Noto Sans CJK SC")
    plt.rcParams.update({"font.sans-serif": [found], "axes.unicode_minus": False,
                         "figure.dpi": 120, "savefig.dpi": 300})
    print(f"绘图字体：{found}")


def full_liouvillian(solver, B_G, detuning_MHz, delta_m, saturation):
    ecal = ecal_amplitude_from_saturation(saturation) * si_jones_for_delta_m(delta_m)
    eq = normalized_pylcp_Eq_from_si_phasor(ecal)
    internal = external_to_internal_detuning_MHz(detuning_MHz)
    hamiltonian = physical_hamiltonian_from_fields(
        solver.ham, eq, np.array([0.0, 0.0, B_G]),
        detuning_rad_s=detuning_MHz_to_rad_s(internal),
    )
    return coherent_liouvillian(hamiltonian) + solver.L_decay


def density_diagnostics(rho):
    hermitian = (rho + rho.conj().T) / 2
    return (float(abs(np.trace(rho) - 1)),
            float(np.max(abs(rho - rho.conj().T))),
            float(np.linalg.eigvalsh(hermitian)[0]))


def evolve_one(solver, B_G, detuning_MHz, delta_m, saturation):
    """从完整 L 精确消去快变量，传播慢变量；检查绝热近似残差。"""
    L = full_liouvillian(solver, B_G, detuning_MHz, delta_m, saturation)
    rho0 = np.zeros((solver.N, solver.N), complex)
    rho0[np.arange(solver.N_G), np.arange(solver.N_G)] = 1 / solver.N_G
    slow = np.arange(solver.N_G) * (solver.N + 1)
    fast = np.setdiff1d(np.arange(solver.N**2), slow)
    A = L[np.ix_(slow, slow)]
    B = L[np.ix_(slow, fast)]
    C = L[np.ix_(fast, slow)]
    D = L[np.ix_(fast, fast)]
    R = -solve(D, C, assume_a="gen", check_finite=False)
    K_complex = A + B @ R
    imaginary_generator = float(np.max(abs(K_complex.imag)))
    if imaginary_generator > 1e-7:
        raise RuntimeError(f"消去快变量后速率矩阵有过大虚部：{imaginary_generator}")
    K = K_complex.real
    # 丢弃浮点消去留下的列和误差，维持布居总和；修正量另行记录。
    conservation_correction = float(np.max(abs(np.sum(K, axis=0))))
    K[np.diag_indices_from(K)] -= np.sum(K, axis=0)
    if np.min(K - np.diag(np.diag(K))) < -1e-8:
        raise RuntimeError("有效布居速率矩阵存在负跃迁率")
    x0 = np.full(solver.N_G, 1 / solver.N_G)
    dark = solver.dark_state_index(delta_m)

    def population(t):
        return float((expm(K * t) @ x0)[dark])

    def density(t):
        if t == 0:
            return rho0
        x = expm(K * t) @ x0
        vector = np.zeros(solver.N**2, complex)
        vector[slow] = x
        vector[fast] = R @ x
        rho = vec_to_rho(vector, solver.N)
        return rho / np.trace(rho)

    # 由真实传播结果逐次扩展时间范围，避免预设弱光的抽运时标。
    upper = 1e-8
    while population(upper) < 0.99999:
        upper *= 2
        if upper > 1e12:
            raise RuntimeError("未能在合理时间内找到 99.9% 暗态布居")
    t90 = brentq(lambda t: population(t) - 0.9, 0, upper, xtol=1e-12)
    times = np.r_[0.0, np.geomspace(max(upper * 1e-8, 1e-10), upper, 240)]
    dark_population = np.array([population(t) for t in times])
    sample_indices = np.unique(np.r_[0, np.linspace(0, len(times)-1, 22, dtype=int)])
    checks = np.array([density_diagnostics(density(times[i])) for i in sample_indices])
    steady = solver.solve(B_G, detuning_MHz, 0.0, delta_m, saturation)
    final_rho = density(upper)
    final_distance = float(np.linalg.norm(final_rho - steady.rho))
    # 绝热消去只忽略快变量的 R K x 时间导数；用完整 L 计算缺项上界。
    residual_samples = []
    for index in sample_indices[1:]:
        x = expm(K * times[index]) @ x0
        approx = np.zeros(solver.N**2, complex)
        derivative = np.zeros_like(approx)
        approx[slow], approx[fast] = x, R @ x
        derivative[slow], derivative[fast] = K @ x, R @ K @ x
        residual_samples.append(np.linalg.norm(L @ approx - derivative))
    absolute_dynamic_residual = float(max(residual_samples))
    # 只在较强两档用直接全矩阵指数交叉检查；最弱一档的 L*t 条件数
    # 已使直接 expm 的 trace 漂移到百分量级，不能把它当可靠参考。
    full_expm_population_error = np.nan
    full_expm_trace_error = np.nan
    if saturation >= 1e-6:
        exact_initial = np.zeros(solver.N**2, complex)
        exact_initial[slow] = x0
        direct_rho = vec_to_rho(expm(L * t90) @ exact_initial, solver.N)
        full_expm_trace_error = float(abs(np.trace(direct_rho) - 1))
        full_expm_population_error = float(abs(
            direct_rho[dark, dark].real / np.trace(direct_rho).real - 0.9))
        if full_expm_population_error > 1e-3:
            raise RuntimeError("慢变量传播与直接完整 Liouvillian 指数不一致")
    if checks[:, 0].max() > 1e-8 or checks[:, 1].max() > 1e-6 or checks[:, 2].min() < -1e-6:
        raise RuntimeError(f"有限时间密度矩阵检查失败：trace={checks[:,0].max():.3e}, "
                           f"Hermiticity={checks[:,1].max():.3e}, λmin={checks[:,2].min():.3e}; "
                           f"s={saturation:g}, Δm={delta_m:+d}, t_end={upper:.3e} s")
    if final_distance > 2e-3 or abs(dark_population[-1] - steady.rho[dark, dark].real) > 2e-3:
        raise RuntimeError("长时间传播与稳态求解不一致")
    return dict(times_s=times, dark_population=dark_population, t90_s=t90,
                final_rho=final_rho, steady_rho=steady.rho,
                final_distance=final_distance, checks=checks,
                dynamic_residual=absolute_dynamic_residual,
                conservation_correction=conservation_correction,
                imaginary_generator=imaginary_generator,
                full_expm_population_error=full_expm_population_error,
                full_expm_trace_error=full_expm_trace_error)


def save_figure(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {name}")


def main():
    setup_chinese_font()
    OUT.mkdir(parents=True, exist_ok=True)
    config = BenchmarkConfig()
    solver = LiteralSteadyStateSolver(config)
    previous = np.load(HERE / "outputs" / "literal_steady_state_dark_test.npz")
    spectra = np.load(HERE / "outputs" / "linear_thermal_vs_elecsus.npz")
    axis = spectra["detuning_MHz"]
    saved = {"saturations": SATURATIONS, "basis_g_F_mF": solver.basis_g,
             "spectrum_detuning_MHz": axis, "temperature_C": np.array(config.temperature_C),
             "k39_fraction": np.array(config.k39_fraction)}
    t90_all = np.empty((2, len(SATURATIONS)))
    final_pop_all = np.empty_like(t90_all)
    response_all = np.empty_like(t90_all)
    peak_all = previous["peak_detunings_MHz"][:, 0]
    saved["peak_detunings_MHz"] = peak_all

    # 图 1 与 4：沿用已有 B=0 的 ElecSus 峰值失谐；每个 s 独立传播完整 L。
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for pi, dm in enumerate(DELTA_M_VALUES):
        for si, sat in enumerate(SATURATIONS):
            result = evolve_one(solver, 0.0, float(peak_all[pi]), dm, float(sat))
            key = f"dm{'p' if dm > 0 else 'm'}1_s{si}"
            for field in ("times_s", "dark_population", "final_rho", "steady_rho",
                          "checks", "t90_s", "final_distance", "dynamic_residual",
                          "conservation_correction", "imaginary_generator",
                          "full_expm_population_error", "full_expm_trace_error"):
                saved[f"time_{key}_{field}"] = np.asarray(result[field])
            t90_all[pi, si] = result["t90_s"]
            dark_idx = solver.dark_state_index(dm)
            final_pop_all[pi, si] = result["steady_rho"][dark_idx, dark_idx].real
            point = solver.solve(0.0, float(peak_all[pi]), 0.0, dm, float(sat))
            response_all[pi, si] = abs(point.chi)
            axes1[pi].semilogx(result["times_s"][1:], result["dark_population"][1:],
                               color=COLORS[si], label=f"s={sat:.0e}")
            print(f"Δm={dm:+d}, s={sat:.0e}: t90={result['t90_s']:.6g} s; "
                  f"终态距离={result['final_distance']:.2e}; "
                  f"trace={result['checks'][:,0].max():.2e}, "
                  f"Hermiticity={result['checks'][:,1].max():.2e}, "
                  f"最小本征值={result['checks'][:,2].min():.2e}")
        axes1[pi].axhline(1, color="black", ls=":", lw=1)
        axes1[pi].set(title=f"σ{'+' if dm > 0 else '-'}：|F=2,mF={'+2' if dm > 0 else '-2'}> 暗态",
                      xlabel="时间 t（秒）", ylim=(-0.02, 1.04))
        axes1[pi].grid(alpha=.25)
        axes1[pi].legend()
    axes1[0].set_ylabel("暗态布居 p_dark(t)")
    fig1.suptitle("图 1：由完整 Liouvillian 导出的光抽运时间演化（B=0 G）")
    fig1.tight_layout()
    save_figure(fig1, "01_dark_population_time.png")

    # 图 2：真实 |F,mF> 标签及两种手性各自的无限时间稳态。
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    labels = [f"|{int(f)},{int(m):+d}>" for f, m in solver.basis_g]
    x = np.arange(solver.N_G)
    for pi, dm in enumerate(DELTA_M_VALUES):
        steady = solver.solve(0.0, float(peak_all[pi]), 0.0, dm, 1e-6)
        values = steady.rho.diagonal()[:solver.N_G].real
        dark = solver.dark_state_index(dm)
        axes2[pi].bar(x-.18, np.full(solver.N_G, 1/solver.N_G), width=.35,
                      label="初始等布居", color="#9aa8b2")
        colors = ["#d85828" if i == dark else "#1768ac" for i in x]
        axes2[pi].bar(x+.18, values, width=.35, label="无限时间稳态", color=colors)
        axes2[pi].set(xticks=x, xticklabels=labels,
                      title=f"σ{'+' if dm > 0 else '-'}；橙色为暗态", xlabel="基态 |F,mF>")
        axes2[pi].tick_params(axis="x", rotation=45)
        axes2[pi].legend()
        saved[f"levels_dm{'p' if dm > 0 else 'm'}1_initial"] = np.full(solver.N_G, 1/solver.N_G)
        saved[f"levels_dm{'p' if dm > 0 else 'm'}1_steady"] = values
    axes2[0].set_ylabel("布居")
    fig2.suptitle("图 2：初始热布居与完整 Liouvillian 无限时间稳态（B=0 G，s=1e-6）")
    fig2.tight_layout()
    save_figure(fig2, "02_ground_state_populations.png")

    # 图 3：完整稳态在明确标出的稀疏失谐点逐点求解，绝不填充零数组。
    for pi, dm in enumerate(DELTA_M_VALUES):
        fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for bi, B_G in enumerate((0.0, 300.0)):
            tag = f"B{int(B_G):04d}_dm{'p' if dm > 0 else 'm'}1"
            chi_linear = spectra[f"chi_linear_{tag}"]
            chi_ref = spectra[f"chi_elecsus_{tag}"]
            # 均匀覆盖原曲线主要范围，同时纳入该条件下的吸收峰。
            peak = axis[np.argmax(chi_ref.imag)]
            sample_axis = np.unique(np.r_[np.linspace(-1200, 1200, 9), peak])
            points = [solver.solve(B_G, float(d), 0.0, dm, 1e-6) for d in sample_axis]
            chi_full = np.array([p.chi for p in points])
            if max(p.trace_error for p in points) > 1e-7 or max(p.hermiticity_error for p in points) > 1e-7 or min(p.minimum_eigenvalue for p in points) < -1e-7:
                raise RuntimeError("完整稳态频谱的密度矩阵检查失败")
            saved[f"spectrum_{tag}_full_detuning_MHz"] = sample_axis
            saved[f"spectrum_{tag}_full_chi"] = chi_full
            saved[f"spectrum_{tag}_linear_chi"] = chi_linear
            saved[f"spectrum_{tag}_elecsus_chi"] = chi_ref
            saved[f"spectrum_{tag}_full_diagnostics"] = np.array([
                (p.trace_error, p.hermiticity_error, p.minimum_eigenvalue,
                 p.liouvillian_residual) for p in points])
            for col, component in enumerate(("real", "imag")):
                ax = axes3[bi, col]
                ax.plot(axis, getattr(chi_ref, component), label="ElecSus 线性响应", lw=2)
                ax.plot(axis, getattr(chi_linear, component), label="固定热布居一阶 OBE", ls="--")
                ax.plot(sample_axis, getattr(chi_full, component), "o", color="#d85828",
                        ms=5, label="完整 L 无限时间稳态（逐点求解）")
                ax.set(title=f"B={B_G:g} G；{'Re(χ)' if col == 0 else 'Im(χ)'}",
                       ylabel="无量纲 susceptibility χ")
                ax.grid(alpha=.25)
                if bi == 1:
                    ax.set_xlabel("激光失谐（MHz，ElecSus 外部轴）")
                if bi == 0 and col == 1:
                    ax.legend(fontsize=9)
            print(f"频谱 {tag}: {len(sample_axis)} 个完整稳态点，max|chi|={max(abs(chi_full)):.2e}")
        fig3.suptitle(f"图 3：σ{'+' if dm > 0 else '-'} 响应的三种模型比较（T={config.temperature_C:g} °C）")
        fig3.tight_layout()
        save_figure(fig3, f"03_susceptibility_sigma_{'plus' if dm > 0 else 'minus'}.png")

    # 图 4：稳态与抽运时间分轴，避免把无限时间极限和动力学混为一谈。
    fig4, axes4 = plt.subplots(1, 3, figsize=(14, 4.5))
    for pi, dm in enumerate(DELTA_M_VALUES):
        label = f"σ{'+' if dm > 0 else '-'}"
        axes4[0].semilogx(SATURATIONS, final_pop_all[pi], "o-", label=label)
        axes4[1].semilogx(SATURATIONS, response_all[pi], "o-", label=label)
        axes4[2].loglog(SATURATIONS, t90_all[pi], "o-", label=label)
    for ax in axes4:
        ax.set_xlabel("饱和参数 s")
        ax.grid(alpha=.25)
        ax.legend()
    axes4[0].set(title="最终暗态布居", ylabel="p_dark(∞)", ylim=(.98, 1.005))
    axes4[1].set(title="有效响应（逐点稳态求解为零）", ylabel="|Pq/(ε0 Eq)|（无量纲）",
                 ylim=(-1e-20, 1e-20))
    axes4[1].set_yticks([0.0], labels=["0"])
    axes4[2].set(title="抽运达到 90% 的时间", ylabel="t90（秒）")
    fig4.suptitle("图 4：光强影响最终稳态及达到稳态所需时间（B=0 G）")
    fig4.tight_layout()
    save_figure(fig4, "04_intensity_steady_state_and_pumping_time.png")
    saved["summary_t90_s"] = t90_all
    saved["summary_final_dark_population"] = final_pop_all
    saved["summary_effective_response_abs"] = response_all
    np.savez_compressed(OUT / "visualization_data.npz", **saved)
    print(f"已保存 {OUT / 'visualization_data.npz'}")


if __name__ == "__main__":
    main()
