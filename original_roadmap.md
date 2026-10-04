# 原子系综仿真：原始方案与长期路线总纲

本文档只定义项目最初希望实现的目标、主数据流、阶段顺序和每一步的验收边界。

本文档**不记录当前完成状态**，也不因为后续调试中出现的新问题而改变主路线。后续开发每开始一个新阶段，都应先对照本文档，确认该工作是在完成主路线中的某个明确接口，而不是无边界地增加新的物理问题。

---

## 1. 最终目标

建立一个针对 K39 D1 原子蒸气的微观—宏观自洽仿真：

```text
有限光束中的热原子 Monte Carlo 系综
→ 每个原子的真实入口、速度和空间轨迹
→ 沿轨迹经历的 B、光强、偏振、频率和 Doppler history
→ Liouvillian / optical Bloch equation 演化完整密度矩阵
→ 当前时刻各空间切片的系综平均密度矩阵 rho_bar(z)
→ 宏观复极化 P(z)
→ Maxwell 方程更新光场 E(z)
→ 使用新光场重新计算原子响应
→ 迭代到 E(z)、rho_bar(z)、P(z) 自洽收敛
```

最终的固定点关系为：

```text
E^(n)(z)
  → L[E^(n)(z), B(z), polarization(z), delta-k*v_z]
  → rho_a^(n) for every atom trajectory
  → rho_bar^(n)(z)
  → P^(n)(z)
  → Maxwell propagation
  → E_raw^(n+1)(z)
  → mixing
  → E^(n+1)(z)
```

直到：

```text
E^(n+1)(z) ≈ E^(n)(z)
```

这个自洽循环是项目不可改变的主方向。所有中间测试和扩展都应服务于这条链。

---

## 2. 三套代码各自的角色

### 2.1 PyLCP

PyLCP 用于微观原子物理：

- 构造 K39 D1 Hamiltonian；
- 提供超精细、Zeeman 和偶极跃迁算符；
- 提供 spontaneous-emission / OBE 参考实现；
- 作为自己构造 Liouvillian 的逐项对照。

大规模 Monte Carlo 中不应反复调用通用 ODE solver。正确路线是先用 PyLCP 验证微观模型，再使用经过验证的 L 矩阵传播。

### 2.2 ElecSus

ElecSus 用于：

- 热平衡、弱光、局域线性 susceptibility 基准；
- 原子常数、跃迁强度、Doppler、number density 的交叉检查；
- 均匀线性介质传播的参考；
- Maxwell/Jones 传播 convention 的比较。

ElecSus 不负责单原子轨迹、有限驻留时间、optical pumping 或完整 density-matrix history。后续非线性自洽模型不应把 ElecSus 当成原子历史的核心计算器。

### 2.3 本项目代码

本项目负责：

- 原子轨迹和当前空间快照；
- 逐轨迹 Liouvillian 演化；
- current-slice 系综平均；
- `rho_bar -> P` 的 SI 接口；
- `P -> E` 的 Maxwell 传播；
- `E -> rho -> P -> E` 的自洽迭代。

老师提供的 demo 和早期整体框架只作为思路参考。最终可信实现必须在删除 demo 后仍可运行。

---

## 3. 总体开发原则

1. 每一步只回答一个主要问题。
2. 每一步使用独立目录和独立小测试。
3. 前一步未 PASS，不进入下一步。
4. 后续阶段不随意修改已经冻结的物理内核。
5. 如果发现旧模块 bug，先写失败测试，再修复并完整回归。
6. 先保证物理和单位正确，再做缓存、降维、并行和速度优化。
7. 任何优化都必须与未优化参考结果逐元素或按明确误差指标比较。
8. 不同时加入两个新的物理效应。
9. 明确区分：代码验证、数值收敛、物理结论、模型假设。
10. 一个中途发现的问题如果不阻塞当前主接口，应放入待办，而不是自动扩展成新的主阶段。

### 3.1 阶段编号规则

为避免“验证工作”“主线推进”“主阶段内的子阶段”和“子阶段内脚本顺序”混在一起，后续统一使用以下编号：

- `Step N`：主线里程碑，也是该组子阶段的总称。完成后，最终数据流新增一个正式接口；
- `stepN_0_<name>/`、`stepN_1_<name>/`……：同一个主阶段内部依次建立的独立子阶段目录；
- `Step NA`、`Step NB`、`Step NC`……：在 Step N 正式接口完成后开展的验证分支，用于检查物理边界和适用范围，但不把主链推进到下一个接口。
- 从 Step 6 开始，一个主阶段如果需要拆成多个独立子阶段，目录按 `stepN_0_<name>/`、`stepN_1_<name>/`、`stepN_2_<name>/`……命名。例如 Step 6 可依次使用 `step6_0_maxwell_convention/`、`step6_1_maxwell_benchmark/`。下划线后的数字表示主阶段内的子阶段，不是脚本序号，也不是新的主阶段。
- 每个子阶段目录内部仍统一使用 `01_*`、`02_*`、`03_*`……给测试脚本排序，并保留独立的 `README.md` 与 `run_all.py`。禁止把脚本改成 `6_1_*`、`7_1_*` 这类名称。
- Step 5A–5D 仍使用字母表示验证路线，不能改写为 `Step 5_1`–`Step 5_4`；否则会把验证分支误解成主线内部实现步骤。
- 已存在的 `step5_0_convention_validation/` 和 Step 2–5 中的 `01_*`、`02_*` 等历史名称保持不动，避免为了形式统一而制造无物理价值的大规模重命名；新规则从 Step 6 开始执行。

例如：

```text
Step 5.0  rho -> P 前的 convention/单位检查点
Step 5    正式完成 rho_bar -> P

Step 5A   linear-thermal / ElecSus 验证分支
Step 5B   finite-transit / optical-pumping 验证分支
Step 5C   controlled magnetic-history 验证分支
Step 5D   continuous B(z) local-vs-trajectory 验证分支

Step 6    回到主线：P -> E 的 Maxwell propagation
          子阶段目录使用 step6_0_*、step6_1_*、step6_2_*...
          每个子阶段内部脚本仍使用 01_*、02_*、03_*...
```

字母分支的作用是说明“Step 5 输出的 P 在什么条件下可信、具有怎样的历史依赖”，而不是用新的编号取代主线。验证分支完成后应回到下一个整数主阶段，除非发现阻塞主接口的明确问题。

---

# 4. 主路线分步安排

## Step 1：源码、demo 和物理约定审计

### 目标

在写新代码前，弄清楚现有库和 demo 的职责、单位与限制，形成一份最小而明确的模型定义。

### 应完成的工作

1. 阅读 PyLCP 中：
   - Hamiltonian 构造；
   - Liouvillian / OBE；
   - `d_q`、`d_q_bare`；
   - 球基底；
   - B、磁矩、detuning 和 decay 单位。
2. 阅读 ElecSus 中：
   - susceptibility；
   - transition strength；
   - Doppler profile；
   - number density；
   - dielectric tensor 和传播接口。
3. 阅读老师的 Monte Carlo/L-matrix demo，明确它展示的核心思想和不能直接继承的部分。
4. 阅读早期整体循环框架，提取最终希望保留的模块接口，不把未经验证的实现当作正确答案。
5. 写清楚：
   - 原子种类和谱线；
   - Hilbert 空间维数；
   - Hamiltonian/Liouvillian 单位；
   - 时间和磁场单位；
   - detuning 和 Doppler 符号；
   - 球基底顺序；
   - density-matrix 向量化顺序。

### 验收标准

- 能清楚说明 PyLCP、ElecSus、demo 和本项目分别负责什么；
- 所有核心单位和 convention 有源码依据；
- 新代码不依赖 demo 的运行结果；
- 形成后续 Step 2 的明确输入规范。

### 本步禁止加入

- 大规模 Monte Carlo；
- 光场传播；
- 自洽迭代；
- 碰撞或基态弛豫；
- 为了让结果看起来一致而加入经验比例。

---

## Step 2：固定单原子的 Liouvillian

### 核心问题

一个 K39 D1 原子固定在给定磁场、光强、偏振和激光频率中时，能否正确构造 L，并正确传播完整密度矩阵？

### 应完成的工作

1. 构造 K39 D1 的 ground/excited basis。
2. 构造：
   - hyperfine Hamiltonian；
   - Zeeman Hamiltonian；
   - light-interaction Hamiltonian；
   - detuning Hamiltonian；
   - spontaneous-decay Liouvillian。
3. 锁定 `rho <-> vec(rho)` 的唯一顺序。
4. 自行构造完整 Liouvillian：

   ```text
   d vec(rho)/dt = L vec(rho)
   ```

5. 实现固定 L 的传播：

   ```text
   rho(t) = exp(L*t) rho(0)
   ```

6. 与 PyLCP 对照：
   - 完整 L；
   - spontaneous decay；
   - coherent 部分；
   - 完整 density-matrix evolution。
7. 验证激光 detuning 符号，为后续 Doppler 做准备。

### 必须测试

- trace conservation；
- Hermiticity；
- minimum eigenvalue；
- 零光；
- 零磁场；
- 正负 detuning；
- `expm(Lt)` 与参考 OBE 数值积分；
- 完整矩阵误差，而不只看 population。

### 验收输出

- 独立 K39 D1 原子模型；
- `build_L(...)`；
- `rho_to_vec(...)` / `vec_to_rho(...)`；
- 固定场传播接口；
- 可重复运行的测试脚本。

### 本步禁止加入

- 原子运动；
- 几何；
- 系综平均；
- susceptibility；
- Maxwell propagation。

---

## Step 3：有限光束中的原子几何和运动历史

### 核心问题

当前时刻位于有限光束中的一个热原子，从哪个边界进入、已经飞行多久、依次经过哪些 z slices、每片停留多久？

### 应完成的工作

1. 定义有限蒸气池/光束几何。
2. 在每个空间切片中采样当前原子位置。
3. 从 Maxwell-Boltzmann 分布采样三维速度。
4. 从当前位置沿 `-v` 反向追踪最近入口。
5. 支持所有合法入口：
   - 侧壁；
   - 前端面；
   - 后端面。
6. 把 `entry -> current position` 拆成逐 slice segments。
7. 输出每个原子的：
   - current position；
   - velocity；
   - entry point；
   - time since entry；
   - ordered slice history；
   - dwell time per segment；
   - current slice。

### 采样原则

采用稳态“当前空间快照”采样：在光束体积中抽取当前原子，再反向追踪入口。不要在此基础上额外加入人为 flight-time weighting。

### 必须测试

- 所有当前位置在几何体内；
- 位置分布正确；
- 三维速度统计满足 Maxwell-Boltzmann；
- 入口点位于合法边界；
- 入口前一小步位于几何体外；
- 入口到当前位置的整段位于几何体内；
- 所有 segment 时间为正；
- segment 顺序正确；
- `sum(dt_segments) == time_since_entry`；
- 最后一段是 current slice；
- 大样本 stress test 无 NaN/Inf。

### 验收输出

- 独立 geometry/trajectory 模块；
- 单原子可人工检查的轨迹；
- N 原子的几何系综；
- 不依赖 PyLCP、ElecSus 或 demo。

### 本步禁止加入

- 密度矩阵；
- L 矩阵；
- 极化；
- 光场传播。

---

## Step 4：单个运动原子的密度矩阵传播

### 核心问题

如何把 Step 2 的固定场 L 和 Step 3 的一条几何轨迹连接起来？

### 数据流

```text
rho_entry
→ exp(L_0*dt_0)
→ exp(L_1*dt_1)
→ ...
→ rho_current
```

其中每一段：

```text
L_j = L(B_j, E_j, polarization_j, delta_laser-k*v_z)
```

### 应完成的工作

1. 定义入口初态 `rho_entry`。
2. 在每个 trajectory segment 中读取当前：
   - B；
   - 光场幅值；
   - 偏振；
   - 激光 detuning；
   - 原子的 `v_z`。
3. 使用：

   ```text
   delta_eff = delta_laser - k*v_z
   ```

4. 严格按历史时间顺序传播完整 rho。
5. 原子进入下一段时继承完整 density matrix，而不是只继承 population。
6. 最终只输出当前时刻的 `rho_current`，中间状态可按测试需要选择性保存。

### 必须测试

- 手工轨迹；
- 零光非均匀 B；
- uniform-L 分段传播与总时间一次传播等价；
- Doppler 正负号；
- 时间顺序交换会产生预期差异；
- trace、Hermiticity、minimum eigenvalue、finite；
- 随机单原子端到端测试。

### 验收输出

- `evolve_trajectory(trajectory, fields, rho_entry)`；
- 单个运动原子的 `rho_current`；
- 每段演化和最终诊断。

### 本步禁止加入

- N 原子平均；
- `rho -> P`；
- Maxwell propagation；
- 自洽迭代。

---

## Step 4.5：N 原子系综与 current-slice 空间平均

### 核心问题

如何把单原子 `trajectory -> rho_current` 推广到当前时刻的 N 原子系综？

### 唯一正确的空间平均规则

每个原子独立传播到当前时刻，只按当前所在切片贡献一次：

```text
rho_bar[k] = mean(rho_current[a] for current_slice[a] == k)
```

历史经过的 slices 只用于形成该原子的 `rho_current`，不能重复参与当前空间平均。

### 应完成的工作

1. 生成 N 条几何轨迹。
2. 每个原子独立调用 Step 4。
3. 保存：
   - `rho_current_all[a]`；
   - `current_slice[a]`。
4. 按 current slice 计算：
   - counts；
   - `rho_bar[k]`。
5. 定义空 slice 的明确处理规则，不能静默伪造数据。

### 必须测试

- `sum(counts) == N`；
- 每个原子只贡献一次；
- 人工多历史切片轨迹不会重复 binning；
- 所有单原子与 `rho_bar[k]` 的 density diagnostics；
- 全系综零光；
- 全系综 uniform-L；
- 长轨迹和极慢原子 stress test；
- slice convergence；
- atoms-per-slice、多 seed 的 MC convergence；
- 观察完整 rho、excited population 和 optical coherence，而不只看 trace。

### 验收输出

- `rho_current_all`；
- `current_slice`；
- `counts`；
- `rho_bar(z)`；
- MC 和空间离散误差报告。

### 本步禁止加入

- 用 `N_MC/V_slice` 代替真实原子数密度；
- susceptibility；
- Maxwell propagation；
- 为了减小噪声而未经说明地平滑 rho。

---

## Step 5：锁定绝对单位并实现 `rho_bar -> P`

### 核心问题

如何把无量纲的系综平均密度矩阵转换为具有真实 SI 单位的宏观复极化？

### 先完成 convention validation

在写正式极化模块前，必须锁定：

- 正频率 phasor convention；
- `rho_ge` / `rho_eg`；
- lowering/raising operator；
- PyLCP 球基底顺序；
- Jones vector 与 `q`、`Delta-m` 的映射；
- `d_q_bare` 的归一化；
- 真实 reduced dipole scale；
- `E [V/m]`、saturation、Rabi frequency、Gamma 的换算；
- factor 2、复共轭和整体符号。

### 正式接口

定义清楚采用的 phasor 后，实现：

```text
rho_bar[k]
→ P_q[k, q=-1,0,+1]
→ P_cart[k, x,y,z]
```

宏观极化必须使用真实数密度：

```text
N_K39(T) = isotope_fraction * potassium_vapor_number_density(T)
```

MC 粒子数只决定统计精度，绝不能进入真实 number density。

### 必须测试

- population-only rho 得到 `P=0`；
- 人工单 optical coherence 手算；
- 显式求和与 trace contraction 一致；
- spherical/cartesian 往返；
- 两能级弱光解析响应；
- 弱光 K39 与 ElecSus 的符号、量级和线形比较；
- 小型真实 ensemble：`trajectory -> rho_bar -> P`；
- 零光完整系综得到 `P(z)=0`。

### 验收输出

- 独立 polarization 模块；
- `P_q(z)`；
- `P_cart(z)`；
- number density、dipole scale、phasor convention 元数据；
- 单位全部为 SI。

### 本步禁止加入

- 一般情况下直接把 `P/E` 称为 susceptibility；
- 强光下使用固定线性 chi；
- Maxwell propagation；
- 自洽迭代。

---

## Step 5 的验证分支总览

以下 5A–5D 都以 Step 5 已经定义好的真实 SI 极化接口为共同起点：

```text
trajectory / ensemble
→ rho_bar(z)
→ frozen polarization interface
→ P(z)
→ validation against a controlled reference
```

它们的共同任务是验证 `E -> rho -> P` 这一半的适用边界，不负责 Maxwell propagation，也不改变最终主路线。完成这些验证后，主线应继续进入 Step 6。

## Step 5A：弱光、热平衡、局域稳态基准

### 核心问题

在弱光、uniform B 和 Maxwell-Boltzmann Doppler 分布下，当前微观模型的 fixed-thermal linear response 能否定量复现 ElecSus？完整无弛豫 OBE 的无限时间稳态与该线性基准是否属于同一个物理极限？

### 应完成的工作

1. 分开定义两条不能混淆的数据流：
   - 固定 thermal ground population 的一阶线性响应；
   - 完整无 ground relaxation OBE 的无限时间稳态。
2. 对确定性速度分布做高精度 Doppler 平均。
3. 比较 OBE 与 ElecSus 的：
   - Re/Im chi；
   - resonance position；
   - Doppler width；
   - relative transition strength；
   - absolute magnitude；
   - absorption/dispersion sign。
4. 检查 B=0 和代表性 nonzero uniform B。
5. 不允许用经验比例强行拟合。

### 验收意义

- 如果 fixed-thermal linear branch 与 ElecSus 一致，则说明 atomic constants、detuning、Doppler、density 和 q mapping 正确；
- 如果 literal infinite-time OBE 因 optical pumping 得到不同稳态，应明确这是模型边界，而不是改比例消除差异。

### 本分支不推进的接口

它不实现 trajectory、不实现 Maxwell，也不把主线推进到 `P -> E`。

---

## Step 5B：finite transit-time 与 optical pumping 基准

### 核心问题

真实原子只有有限驻留时间时，响应如何从 coherence build-up、近线性热平衡响应过渡到 nonlinear optical pumping 和 dark-state limit？

### 应完成的工作

1. 使用真实 geometry trajectory 和 `time_since_entry`。
2. 在受控 uniform B / uniform light 条件下传播完整 finite-time OBE。
3. 使用冻结 Step 5 接口得到 P。
4. 扫描 saturation、驻留时间和 detuning。
5. 比较：
   - finite trajectory；
   - fixed-thermal linear OBE；
   - ElecSus。
6. 做 atoms-per-slice、多 seed MC convergence。
7. 若采用降维或缓存，必须先证明与完整 rho 传播等价。

### 验收意义

回答原子在离开光束前是否来得及建立 coherence、发生 optical pumping 或进入 dark state，并明确强光下 `P/E` 只能是条件相关的 effective response。

### 本分支不推进的接口

它仍不实现 Maxwell propagation，也不建立自洽循环。

---

## Step 5C：受控 magnetic-history 基准

### 核心问题

在当前 B、光场、总照光时间和入口态相同的条件下，仅改变过去经历的 B，当前 rho 和 P 是否不同；这种差异位于哪些 density-matrix blocks，并持续多久？

### 应完成的工作

1. 构造 deterministic 两段场历史：

   ```text
   B_past -> B_current
   vs
   B_current -> B_current
   ```

2. 保持两组所有其他参数相同。
3. 严格按时间顺序传播，禁止平均 B 或有效 L。
4. 比较：
   - full `Delta rho`；
   - `gg/ge/eg/ee` blocks；
   - `Delta P`；
   - memory decay curve。
5. 做 identical-history、uniform-L 和 segment-splitting controls。
6. 区分 optical-coherence memory、ground coherence 和 population redistribution。

### 验收意义

定量给出 magnetic memory 是否存在、主要来自哪个 rho block、以及对 P 的持续时间，为是否可以使用 local-response approximation 提供受控依据。

### 本分支不推进的接口

它不做大规模 MC、不做连续 B(z)、不做 Maxwell propagation。

---

## Step 5D：连续 B(z) 下的 local-vs-trajectory 基准

### 核心问题

在接近实验的连续纵向磁场中，运动原子的 chronological trajectory response 是否明显偏离当前位置的 instantaneous local response？

### 应完成的工作

1. 先验证目标高场范围内的 local weak-linear OBE 与 ElecSus。
2. 定义连续 `B(z)` 和代表性正负速度。
3. 严格传播：

   ```text
   rho_(k+1) = exp[L(B(z_k))*dt] rho_k
   ```

4. 同时构造无历史的 `P_local(z)`。
5. 比较：
   - `P_trajectory(z)`；
   - `P_local(z)`；
   - global error；
   - max local error；
   - same-B hysteresis；
   - velocity dependence。
6. 做 constant-B、`DeltaB -> 0`、时间步、空间网格和 density-matrix controls。
7. 明确区分 velocity、Doppler detuning、field gradient 和 history lag，不能把它们未经控制地归为同一个效应。

### 验收意义

判断最终自洽循环是否可以用局域响应替代 trajectory history，或者必须保留原子沿路径的完整 chronological propagation。

### 本分支不推进的接口

它不实现 Maxwell propagation，不是新的主线 Step 6。验证结论形成后，应回到整数主阶段 Step 6。

---

## Step 6：Maxwell-only 光场传播基准

建议从目录 `step6_0_maxwell_propagation/` 开始；如果本主阶段后续还需拆出新的独立子阶段，再建立 `step6_1_<name>/`、`step6_2_<name>/`。每个目录内部的脚本仍命名为 `01_*`、`02_*`、`03_*`……。

### 核心问题

给定一个已知的线性 chi 或已知 P(z)，是否能以正确的符号、单位、相位和偏振传播光场？

### 为什么必须先独立做

传播模块必须在没有 MC 噪声、没有 trajectory history 和没有 nonlinear OBE 的情况下先通过。否则自洽循环失败时无法判断是原子模块、传播模块还是统计噪声的问题。

### 应完成的工作

1. 从 Maxwell 波动方程明确推导采用的 envelope equation。
2. 明确：
   - 真空载波是否已经提出；
   - `exp(+ikz-iomega t)` 下的传播符号；
   - P 与 E 的空间位置；
   - transverse/longitudinal 分量处理；
   - spherical/cartesian basis；
   - slice source integration。
3. 实现：

   ```text
   known P(z) or chi(z)
   → propagate E from z=0 to z=L
   ```

4. 支持复电场和偏振演化。

### 必须测试

- vacuum：`P=0` 时 envelope 不变；
- 均匀标量 chi 的解析解；
- Beer-Lambert absorption；
- 纯实 chi 的相位累积；
- 纯 `q=-1`、纯 `q=+1` 和 x 线偏振；
- 弱各向异性介质；
- 与 ElecSus uniform linear propagation 比较；
- slice/dz convergence；
- 反向检查吸收符号，禁止出现无增益介质中的非物理放大。

### 验收输出

- 独立 Maxwell propagator；
- 解析 benchmark；
- ElecSus propagation benchmark；
- 传播后的完整复 `E(z)`；
- 强度、相位和偏振诊断。

### 本步禁止加入

- Monte Carlo；
- trajectory OBE；
- optical pumping；
- self-consistent iteration；
- Gaussian beam；
- 碰撞和弛豫。

---

## Step 7：一次单向原子—光场耦合

建议从目录 `step7_0_one_pass_coupling/` 开始；后续子阶段使用 `step7_1_<name>/`、`step7_2_<name>/`。每个目录内部的脚本仍命名为 `01_*`、`02_*`、`03_*`……。

### 核心问题

给定一个固定的输入光场 profile，能否完成一次：

```text
E^(0)(z)
→ trajectories/rho
→ rho_bar(z)
→ P(z)
→ Maxwell
→ E^(1)(z)
```

但暂时不把 `E^(1)` 再送回原子模块。

### 应完成的工作

1. 使用冻结的 Step 2–5 模块计算 P。
2. 使用 Step 6 的传播器更新 E。
3. 明确每片采用入口场、中点场还是出口场。
4. 保持同一批原子轨迹和随机 seed。
5. 保存一次更新前后的：
   - `E^(0)(z)`；
   - `rho_bar(z)`；
   - `P(z)`；
   - `E^(1)(z)`。

### 必须测试

- number density 为零时 `E^(1)=E^(0)`；
- P 为零时场不变；
- weak-linear uniform medium 回归 ElecSus/解析传播；
- 改变 dz 后结果收敛；
- q/cartesian 映射一致；
- 没有通过不稳定的逐分量 `P/E` 除法构造一般 chi；
- 输出场无 NaN/Inf。

### 验收输出

- 单次 `E -> P -> E_new` 接口；
- 一次更新的完整数据；
- 线性极限回归报告。

### 本步禁止加入

- Picard 多次迭代；
- mixing 参数调优；
- 强光生产计算；
- 新的碰撞/弛豫物理。

---

## Step 8：正式自洽固定点迭代

建议从目录 `step8_0_self_consistent_loop/` 开始；后续子阶段使用 `step8_1_<name>/`、`step8_2_<name>/`。每个目录内部的脚本仍命名为 `01_*`、`02_*`、`03_*`……。

### 核心问题

能否求得满足原子响应和 Maxwell 传播同时自洽的稳态空间光场？

### 正式循环

```text
Initialize E^(0)(z)

for n = 0,1,2,...:
    keep the same MC positions, velocities and trajectories
    evolve every atom using E^(n)(z)
    bin rho_current into rho_bar^(n)(z)
    compute P^(n)(z)
    propagate Maxwell equation → E_raw^(n+1)(z)
    mix:
        E^(n+1) = alpha*E_raw^(n+1) + (1-alpha)*E^(n)
    evaluate convergence
```

### 关键原则

1. 每次迭代必须使用同一批原子和同一随机 seed，即 common random numbers。
2. 不允许每次迭代重新抽样轨迹，否则 MC 噪声会伪装成场变化。
3. 收敛判据不能只用接近零分量上的逐点相对误差。
4. 同时观察：
   - field profile L2 change；
   - output field change；
   - transmission；
   - P profile change；
   - 连续多次迭代稳定性。
5. mixing/under-relaxation 是数值稳定手段，不能用来掩盖错误的传播符号或未收敛的 MC 噪声。

### 逐级验收顺序

1. 弱光、uniform B、单偏振、解析线性介质；
2. 弱光、uniform B、trajectory ensemble；
3. finite transit；
4. optical pumping；
5. nonuniform longitudinal B；
6. 更强输入功率。

每一级通过后才能打开下一级。

### 必须测试

- alpha 改变不应改变最终固定点，只影响收敛速度；
- 初始场猜测改变后收敛到同一稳定解，或明确记录多稳态；
- slice convergence；
- atoms-per-slice 和 seed convergence；
- zero-density 和 weak-linear regression；
- 能量/透过率合理；
- 无 NaN/Inf；
- 所有 rho 保持物理合法；
- 迭代停止后重新做一次无 mixing residual 检查。

### 验收输出

- 收敛后的 `E(z)`；
- `rho_bar(z)`；
- `P(z)`；
- transmission、phase、polarization；
- iteration history；
- convergence 与 MC uncertainty 报告。

---

## Step 9：实验条件下的正式计算

建议从目录 `step9_0_experimental_run/` 开始；后续子阶段使用 `step9_1_<name>/`、`step9_2_<name>/`。每个目录内部的脚本仍命名为 `01_*`、`02_*`、`03_*`……。

只有 Step 8 在简化条件下通过后，才进入实验参数。

### 推荐增加物理的顺序

1. 真实 longitudinal `B(z)`；
2. 实际输入功率和 detuning scan；
3. 足够大的 thermal MC ensemble；
4. 多 seed uncertainty；
5. 更真实的 transverse intensity profile；
6. 实验可观测量：transmission、rotation、ellipticity、spectrum。

### 必须保存

- 全部输入参数和单位；
- Git commit；
- 随机 seed；
- 数值分辨率；
- 运行时间；
- convergence history；
- MC uncertainty；
- `E(z)`、`P(z)`、`rho_bar(z)`；
- 机器可读 NPZ/JSON；
- 可直接查看的 PNG 和 `results.md`。

---

## Step 10：核心闭环完成后的可选物理扩展

建议从目录 `step10_0_optional_physics/` 开始；后续子阶段使用 `step10_1_<name>/`、`step10_2_<name>/`。每个目录内部的脚本仍命名为 `01_*`、`02_*`、`03_*`……。

以下内容不能在 Step 8 之前同时加入。每一项都应单独建阶段、单独验证：

- Gaussian transverse beam；
- transverse magnetic field；
- buffer-gas collisions；
- ground-state relaxation；
- spin exchange；
- wall collisions；
- dark evolution；
- beam re-entry；
- recoil 和速度改变；
- K40/K41 多同位素；
- cell windows、reflection、etalon；
- backward field 或 cavity；
- time-dependent pulse propagation。

每增加一个物理项，都必须先回答：

1. 它改变哪个方程或哪个 L 项？
2. 参数和单位来自哪里？
3. 有什么独立极限或解析结果可验证？
4. 它是否破坏已有的降维、缓存或传播优化？
5. 它是否需要重新做 MC、slice 和迭代收敛测试？

---

# 5. 主路线中的固定接口

为防止模块互相侵入，长期保持以下接口边界：

```text
atomic_model
    input: constants, B, E, polarization, detuning
    output: H, L, dipole operators

geometry
    input: cell/beam geometry, temperature, seed
    output: current positions, velocities, entry points, ordered segments

single_atom
    input: one trajectory, fields, rho_entry
    output: rho_current

ensemble
    input: rho_current_all, current_slice
    output: counts, rho_bar(z)

polarization
    input: rho_bar(z), physical number density, dipole scale
    output: P_q(z), P_cart(z) in C/m^2

maxwell
    input: E_in, P(z), dz, carrier convention
    output: E_new(z)

self_consistent_loop
    input: fixed MC ensemble and physical parameters
    output: converged E(z), rho_bar(z), P(z), diagnostics
```

任何阶段都不应绕过这些接口重新发明另一套单位或 convention。

---

# 6. 防止方向偏离的检查清单

每次准备开始新工作前，回答以下问题：

1. 这项工作对应主数据流中的哪一个箭头？
2. 如果不做它，下一主接口是否真的无法实现或验证？
3. 它是 bug 修复、数值验证，还是新的物理模型？
4. 是否同时改变了两个以上的物理假设？
5. 是否可以先在更简单的极限下验证？
6. 是否修改了冻结模块？如果是，是否已有失败测试和用户授权？
7. 是否把 ElecSus 的线性假设误用到了 nonlinear history 问题？
8. 是否把 MC 粒子数误当成真实 number density？
9. 是否把历史访问 slice 重复加入当前空间平均？
10. 是否为了速度使用了未经验证的降维、平均 L 或缓存？
11. 是否保存了可复查的参数、结果和误差？
12. 这项工作完成后，项目是否更接近 `E -> rho -> P -> E` 闭环？

如果第 12 项答案是否定的，而且它又不阻塞当前步骤，则应先放入待办，不应改变主路线。

---

# 7. 中途发现新问题时的处理规则

把新问题分成三类：

### A. 阻塞当前主接口的明确 bug

处理方式：

```text
写失败测试
→ 定位原因
→ 最小修复
→ 回归冻结测试
→ 继续当前步骤
```

### B. 不阻塞当前接口的诊断问题

处理方式：

- 记录到 backlog；
- 不自动升级为新的主阶段；
- 只有在老师、实验数据或后续接口明确需要时再处理。

### C. 新物理模型选择

例如碰撞率、ground relaxation、Gaussian beam、re-entry。

处理方式：

- 说明它会怎样改变模型；
- 给出需要的物理参数；
- 由用户/老师决定是否加入；
- 不由代码实现者自行选择经验参数。

---

# 8. 最终成功标准

项目最终完成不能只用“代码能运行”判断。必须同时满足：

1. 微观 L 与参考模型一致；
2. 几何和 trajectory 统计正确；
3. 单原子 rho 演化物理合法；
4. 系综 binning 不重复计数；
5. `rho_bar -> P` 的符号、共轭、factor 2 和 SI 单位正确；
6. Maxwell propagation 回归解析解和 ElecSus 线性极限；
7. 自洽循环在简化极限稳定收敛；
8. 结果对 slice、时间步、atoms-per-slice 和 seed 收敛；
9. 强光结果明确区分 nonlinear effective response 与线性 susceptibility；
10. 实验级结果包含统计误差和模型假设；
11. 所有结果可以通过固定 Git commit、脚本、参数和输出复现。

只有完成以下完整闭环，才算最初方案真正实现：

```text
E(z)
→ moving-atom OBE histories
→ rho_bar(z)
→ P(z)
→ Maxwell propagation
→ E_new(z)
→ converged self-consistent E(z)
```
