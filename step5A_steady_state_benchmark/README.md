# Step 5A：steady-state benchmark 及暗态诊断

本目录不修改 Step 2–5，不使用 trajectory、有限驻留时间或传播。

原计划要求对完整 Liouvillian 求无限时间稳态，再在严格弱光下与 ElecSus 的线性 susceptibility 比较。实际测试发现，这两个模型的边界条件并不相同：

- 当前冻结 OBE 没有基态弛豫或热库；
- 纯 `Delta-m=+1/-1` 连续光把 K39 D1 原子分别抽运到 `|F=2,mF=+2/-2>` stretched dark state；
- 该最终状态与光强趋近于零无关，只是到达暗态的时间随光强变长；
- ElecSus 固定使用热平衡/等基态布居，不包含无限时间 optical pumping。

因此

```text
lim(s -> 0) [lim(t -> infinity) rho(t,s)]
```

不等于 ElecSus 使用的固定热平衡线性响应。按原判据，Step 5A 必须判定为 FAIL，不能用经验比例修正。

目录包含两条明确分开的数据流：

```text
literal:
full L(B, delta-kv, s) -> solve L rho_ss=0, Tr(rho)=1
-> frozen Step 5 rho->P -> chi

linear_thermal diagnostic:
frozen H(B), C_q + fixed rho_g=I/8
-> first-order optical response
-> deterministic Maxwell convolution
-> ElecSus comparison
```

后者只诊断 atomic constants、Zeeman、line strength、Doppler normalization、density 和 q mapping，不能被称为原定义的 full steady-state PASS。

确定性 Doppler 积分把 `u=v_z/lambda` 作为 MHz 频移，在均匀网格上计算局域线性 OBE 响应，再用复合梯形/FFT 卷积 Maxwell Gaussian。`02_velocity_convergence.py` 对步长和 `+/-n sigma` 范围做独立收敛测试。

`04_residual_diagnosis.py` 额外模拟 ElecSus 的整数 MHz 跃迁中心、弱线截断、舍入 linewidth/波长、偶极常数和 self broadening，用来区分代码数值约定与剩余 Hamiltonian 模型差异。

可复查数据保存在 `outputs/`：字面暗态测试、速度积分收敛，以及六组固定热布居 OBE–ElecSus 复 susceptibility 和误差指标。

运行：

```powershell
python run_all.py
```

只有在后续明确加入基态重置/弛豫，或把 benchmark 定义改成“固定热平衡基态的一阶线性 OBE”后，才可能建立与 ElecSus 相同的物理边界条件。

## 可视化补充

独立入口（不调用或修改 `run_all.py`）：

```powershell
python visualize.py
```

程序读取 `outputs/literal_steady_state_dark_test.npz` 和
`outputs/linear_thermal_vs_elecsus.npz`，在
`outputs/visualization/` 保存 300 dpi PNG 和 `visualization_data.npz`。
使用当前默认参数：K39 D1、20 °C、天然 K39 占比、B=0/300 G；时间图和
光强图使用 B=0 G、各偏振原有 ElecSus 吸收峰失谐。初态为 8 个基态各
1/8，激发态为零；时间单位是秒。

| 图片 | 内容 |
| --- | --- |
| `01_dark_population_time.png` | σ+/σ- 的暗态布居随真实时间变化；s=1e-4、1e-6、1e-8。 |
| `02_ground_state_populations.png` | 8 个真实 \|F,mF> 基态的初始布居与 s=1e-6 完整稳态；橙色柱为 stretched 暗态。 |
| `03_susceptibility_sigma_plus.png`、`03_susceptibility_sigma_minus.png` | B=0/300 G 的 Re(χ)、Im(χ)：已有 ElecSus/固定热布居 OBE 全频谱，以及每组 10 个新算完整 Liouvillian 稳态失谐点。橙色零值是逐点求解结果。 |
| `04_intensity_steady_state_and_pumping_time.png` | 各 s 的最终暗态布居、有效响应 \|Pq/(ε0 Eq)\| 和达到 90% 暗态布居的 t90。 |

极弱光的直接 `expm(L t)` 涉及很大的快慢时间尺度比：在 s=1e-8、
t≈10^6 s 时，直接全矩阵指数的 trace 可漂移到约 0.4%，不能据此提取
t90。时间曲线因此先从**每个光强分别构造的完整 256×256 Liouvillian**
中，用 Schur 补消去快变量（光学相干、激发态等），再传播 8 个基态布居。
这保留原有 Hamiltonian、衰减、失谐和偏振接口，是快变量绝热消去的
数值近似，不是修改物理模型。输出 NPZ 保存该近似对完整 Liouvillian
的动态残差，以及密度矩阵检查。s=1e-4 和 1e-6 在 t90 还与直接完整
矩阵指数交叉比较：归一化暗态布居误差分别约 3e-7 和 4e-5；
s=1e-8 的直接指数因 trace 漂移未用作误差基准。三档近似重构密度矩阵
的最大 trace/Hermiticity 误差分别不超过 3e-16/2e-21，最小本征值
不低于 -2e-16；长时间结果与原稳态密度矩阵的距离不超过 9e-7。
最弱光 t90 因而应理解为有残差与稳态检查支持的绝热消去估计，
而不是独立完整时间积分的精确值。

本组图中 t90 约为 164 s、1.64e4 s、1.64e6 s，随 s 降低约按 1/s
变长；三档最终均集中到对应的 \|F=2,mF=±2> 暗态。固定热布居一阶 OBE
与 ElecSus 频谱接近，而无限时间完整稳态响应为零。以上图形与数值检查
只补充物理现象展示，Step 5A 原有正式判据仍为 **FAIL**。
