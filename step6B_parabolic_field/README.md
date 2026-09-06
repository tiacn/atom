# Step 6B：真实抛物线 B(z) 下的 local-vs-trajectory benchmark

本目录在弱光 `s=1e-6` 下研究纵向连续磁场历史：

```text
B(z) = 2500 G - 500 G * (2z/25 mm)^2,
z in [-12.5, +12.5] mm.
```

原子由一端进入、穿过完整 25 mm cell。正式速度为 `v_z=+/-50,+/-100,+/-300,+/-500 m/s`，入口态为 `I_g/8`，偏振为纯 `Delta-m=+1`。固定 laser frequency 数值选在 `B=2500 G、v_z=0` 的 frozen weak-linear local absorption 主峰；运动原子只按冻结 convention 使用 `delta-k*v_z`。

本阶段不加入 MC、强光、Gaussian beam、Maxwell propagation、碰撞或 ground relaxation。

## 完整状态空间传播

状态始终是完整 `(16,16)` density matrix，即 256 个 Liouville components；不使用 Step 5.5B 的 51 维 reduced space。由于高场 dense `expm(256x256)` 每个空间点约需 1 秒，正式细网格用 fourth-order symmetric full-state factorization 计算同一个 `L=L_H+L_decay` 的指数作用：

- Hamiltonian 部分用完整 16x16 Hermitian eigensystem 精确酉传播；
- spontaneous-decay Lindbladian 用完整 `gg/ge/eg/ee` 解析 block map；
- 每个不超过 1.25 ns 的内部小步都在其真实空间中点重新计算 `B(z)`；对正式速度这对应约 `0.0625--0.625 um` 的实际磁场采样；
- 独立 control 与 dense `expm(full 256x256 L)` 比较，并做时间步和 `dz` convergence。

这不是 reachable-space 降维，也不使用平均 B 或有效 L。

## 入口瞬态

正式入口态 `I_g/8` 没有 optical coherence，所以入口处严格有 `P_trajectory=0`，而 local steady response 非零。原始全路径 max error 会包含这一真实 coherence-build-up boundary layer，不能全部归因于 magnetic history。

因此同时报告：

- raw 指标：包含入口瞬态；
- bulk 指标：排除入口后 `5*tau_memory` 的飞行距离，再判断连续 B history。

`DeltaB->0` control 也必须在 bulk 中趋近 local；raw 入口点不会因磁场变平而消失。

## 当前测试顺序

1. `01_high_field_benchmark.py`：2000–2500 G fixed-thermal weak-linear OBE vs ElecSus。
2. `02_propagator_controls.py`：解析 decay、四阶 full-state propagation 与 dense full-256 `expm`。
3. `03_spatial_convergence.py`：`dz=10/5/2.5/1.25/0.625 um` 及 `DeltaB->0` bulk control。
4. `04_trajectory_benchmark.py`：八个正负速度、local/trajectory、同 B 滞回、图和机器可读结果。

全部运行：`python run_all.py`。正式扫描会按速度单独缓存到 `outputs/profiles/`；中断后重跑会复用已经完成的速度。

其余正式速度、hysteresis、绘图和报告脚本在上述门槛通过后运行。正式结论以 `outputs/results.md` 为准。
