# Step 5B：finite transit-time + optical pumping

本目录连接真实 geometry trajectory、有限时间 full OBE、current-slice ensemble binning 和冻结 Step 5 polarization。它不修改 Step 2–5A，不加入传播、非均匀 B、碰撞、基态弛豫或经验参数。

在本阶段限定的 uniform `B_z`、top-hat uniform light 下，同一原子的所有 trajectory segments 使用同一个 Liouvillian，因此逐 segment 演化严格等价于按 `time_since_entry` 传播一次。对于纯圆偏振和入口态 `I_g/8`，完整 256 维 Liouville 空间中只有 51 维可达，而且该子空间对 `L` 严格封闭。正式实现先用测试锁定与冻结逐 segment dense `expm` 的一致性，再用于大规模 MC。

ElecSus 外部 detuning 与当前 K39 OBE 内部 detuning 的关系仍采用 Step 5.0/5A 已验证的：

```text
delta_internal = delta_external + 15.864 MHz - v_z/lambda
```

SI 正频率 Jones vector 先共轭后送入冻结 PyLCP field 接口；输出继续使用冻结 Step 5：

```text
Pcal_q = 2*N_K39*d0*Tr(C_q rho)
```

后续脚本会依次增加单 detuning 时间分箱、saturation 扫描、MC convergence 和完整 detuning spectrum。

## 数据流

```text
thermal K39 atoms
-> finite-cylinder geometry and real entry trajectories
-> time_since_entry = sum(segment dt)
-> full OBE in the exact 51-dimensional reachable subspace
-> rho_final[a] and single-atom observables
-> current_slice-only binning
-> rho_bar[k]
-> frozen Step 5 P_q[k], P_cart[k]
```

在需要完整 density matrix 的正式 5000 原子单点测试中，逐原子 `rho_final`、`rho_bar` 和 Step 5 极化都保存。saturation、MC 和 spectrum 阶段只传播所需的严格线性 observables，避免保存重复的大矩阵。

## 测试顺序

1. `01_reduced_propagation.py`：51 维严格不变子空间、逐 segment 冻结传播、Step 5 contraction 和 detuning cache 收敛。
2. `02_single_detuning.py`：5000 原子的完整 `trajectory -> rho_bar -> P`，time quantile bins、空间 profile 和 `Delta-m` 镜像检查。
3. `03_saturation_scan.py`：`s=1e-6...10`，早期 pumping time、三个极限和 crossover。
4. `04_mc_convergence.py`：20/50/100/200 atoms per slice、5 seeds，并用弱光 200/slice seed mean 回归 Step 5A。
5. `05_detuning_spectrum.py`：同一批 5000 轨迹的 241 点完整谱，与 linear OBE/ElecSus 比较。
6. `06_weak_light_spectrum.py`：`s=1e-6` 下的三曲线回归图；由于会重新计算完整 5000-trajectory 频谱，默认不放入 `run_all.py`，需要时单独运行。

运行：

```powershell
python run_all.py
```

所有二进制结果和科学绘图保存到 `outputs/`。强 pumping 时输出的 `P/E` 始终标记为 effective response，不宣称为一般材料 susceptibility。
