# 原子系综仿真项目：AI 项目说明与交接规范

本文档是本仓库的 AI 入口文件。新的本地 Agent、网页版 ChatGPT 或协作者开始工作前，应先完整阅读本文，再阅读当前阶段的 `README.md`、测试脚本和输出文件。

本文档记录的是项目背景、已经验证的物理链、各阶段边界、当前结果、尚未解决的问题，以及本地开发与 GitHub/网页版 ChatGPT 之间的交接方式。不要仅凭某次对话摘要修改冻结模块。

## 1. 仓库和工作区

- 完整本地工作区：`D:\Code\python\atom`
- 本 Git 仓库本地根目录：`D:\Code\python\atom\MC\step_by_step`
- GitHub：<https://github.com/tiacn/atom>
- 默认分支：`main`
- 本仓库主要包含逐步重写和验证的代码、测试与输出。
- 完整本地工作区中的 `ElecSus`、`pylcp`、旧 `MC/v5b_demo` 和 `MC/hybrid_engine` 不一定包含在本 Git 仓库内。网页版 AI 可以审查本仓库代码和结果，但不能因为这里只上传了 `step_by_step` 就假定第三方库源码也在仓库中。

本地目录大致关系：

```text
D:\Code\python\atom
├─ ElecSus/                 # 第三方库，本地源码审计依据
├─ pylcp/                   # 第三方库，本地源码审计依据
└─ MC/
   ├─ v5b_demo/             # 老师提供的 MC/L-matrix demo，仅作历史参考
   ├─ hybrid_engine/        # 早期 AI 生成的整体循环框架，不作为已验证核心
   └─ step_by_step/         # 当前可信、逐步验证的实现；也是本 Git 仓库根目录
```

## 2. 项目背景

目标是模拟 K39 D1 原子蒸气中运动原子的内部态与光学响应。最终希望处理：

```text
有限光束中的热原子运动
-> 每个原子的磁场、光强、偏振、频率和 Doppler 历史
-> optical Bloch equation / Liouvillian 演化
-> 当前时刻各空间切片的系综平均密度矩阵 rho_bar(z)
-> 宏观复极化 P(z)
-> 后续与 Maxwell 光场传播自洽耦合
```

项目最初希望主要使用 ElecSus。ElecSus 很适合热平衡、局域稳态、线性 susceptibility 和光场传播，但不能直接表示以下历史依赖场景：

- 单个原子从哪里进入光束；
- 原子在不同切片停留多久；
- `delta-k*v_z` 的逐原子 Doppler shift；
- 原子经过不同磁场或光场后的完整密度矩阵记忆；
- finite transit-time；
- 强光 optical pumping 和暗态形成；
- 非线性情况下不能简单用固定 `chi` 描述的极化响应。

老师提供的 `v5b_demo` 展示了 Monte Carlo 轨迹和 Liouvillian/L 矩阵演化思路。早期 `hybrid_engine` 试图一次搭建完整循环，但它由 AI 大范围生成，用户难以逐行判断物理和实现是否正确。因此项目改为以下原则：

1. 不依赖旧 demo，删除 demo 后逐步实现仍应运行。
2. 每一步新建独立目录。
3. 每一步只回答少数明确问题。
4. 先写可人工理解的小测试，再连接下一层。
5. 每一步 PASS 后冻结；后续优先新增适配层或测试，不随意重写冻结物理逻辑。
6. 物理正确性优先于速度；确认正确后再做严格等价的优化。
7. 明确区分“代码验证结论”“数值收敛结论”和“尚未加入的模型假设”。

阶段命名约定：Step 5 正式极化接口完成后的四个验证分支固定命名为 Step 5A、5B、5C、5D；它们不占用新的主线整数阶段。从后续 Step 6 开始，每个阶段内的新脚本按 `6_1_*`、`6_2_*`……，Step 7 按 `7_1_*`、`7_2_*`……排序。现有 Step 2–5 的历史脚本名不追溯修改。长期方向与阶段边界以仓库根目录 `original_roadmap.md` 为准。

## 3. 总体数据流

当前已经建立并验证到宏观极化：

```text
K39 D1 atomic model
-> L(B, E, polarization, detuning-k*v_z)
-> trajectory segments and dwell times
-> rho_final[a]
-> current_slice-only ensemble binning
-> rho_bar[k]
-> P_q[k, q=-1,0,+1]
-> P_cart[k, x,y,z]
```

尚未实现正式的 Maxwell 自洽传播：

```text
P(z) -> update E(z) -> recompute atomic histories -> iterate
```

强光和历史依赖情况下，`P/E` 只能称为某个指定条件下的 effective response，不能直接宣称为与场强无关的一般 susceptibility。

## 4. 核心物理和代码约定

后续代码不得在没有独立 convention test 的情况下改变这些约定。

### 4.1 原子模型和单位

- 原子：K39 D1。
- Hilbert 空间：8 个 ground states + 8 个 excited states，共 16 维。
- 密度矩阵：`(16,16)`；Liouville 向量：256 维。
- Hamiltonian 和 Liouvillian：`rad/s`。
- 时间：`s`。
- 对外激光 detuning 通常以 `MHz` 表示，进入 L 前转换为 `rad/s`。
- 磁场接口目前使用 `Gauss`，构造 Hamiltonian 时按已经验证的 PyLCP 约定处理。
- 密度矩阵向量化使用列优先：`rho.reshape(-1, order="F")`。

### 4.2 Doppler 和 detuning

一般内部关系：

```text
delta_eff = delta_laser - k*v_z
```

在 Step 5A/5B 使用 ElecSus 外部频率轴时：

```text
delta_internal_MHz = delta_external_MHz + 15.864 MHz - v_z/lambda/1e6
```

其中 `15.864 MHz` 是当前 K39/ElecSus 频率零点对齐所需的 isotope-shift offset。不要在新模块中重复发明另一套 detuning zero。

### 4.3 正频率 phasor

统一采用：

```text
E_phys(r,t) = Re[Ecal(r) exp(-i*omega*t)]
Ecal(r) = Ecal_0 exp(+i*k*z)
I = (c*epsilon_0/2) |Ecal|^2
```

PyLCP normalized field 使用这套 SI 包络的共轭表示：

```text
E_q^Py = (2*d0/(hbar*Gamma)) cart2spherical(conj(Ecal_cart))
|E_q^Py| = sqrt(2*s)
s = I/Isat
Omega = d0*|Ecal|/hbar = Gamma*sqrt(s/2)
```

已经验证的 K39 D1 绝对量：

```text
d0 = 2.462565313e-29 C*m
Isat = 17.045641111 W/m^2
     = 1.704564111 mW/cm^2
```

### 4.4 球基底、偏振和 coherence

- PyLCP 球数组顺序：`[q=-1, q=0, q=+1]`。
- `C_q=d_q_bare=|g><e|` 是 lowering operator。
- `Tr(C_q rho)` 选取 `rho_eg`，不是 `rho_ge`。
- 沿 `+z` 传播，在本项目 SI phasor 下：

```text
(x+i*y)/sqrt(2) -> Ecal_q=[1,0,0]  -> drives Delta-m=+1
(x-i*y)/sqrt(2) -> Ecal_q=[0,0,-1] -> drives Delta-m=-1
x polarization  -> drives both Delta-m=+1 and -1
```

不要依赖 Left/Right、Plus/Minus 名称判断偏振；ElecSus 与 PyLCP 的名称容易产生交叉映射，应按明确 Jones vector、`q` 和 `Delta-m` 检查。

### 4.5 宏观极化

项目采用 `Re[phasor]` 定义：

```text
Pcal_q = 2*N_K39*d0*Tr(C_q*rho_bar)
```

- `Pcal_q` 和 `Pcal_cart` 单位：`C/m^2`。
- `P_q` shape：`(n_slices,3)`，顺序 `[-1,0,+1]`。
- `P_cart` shape：`(n_slices,3)`，顺序 `[x,y,z]`。
- factor 2 属于 `Re[Pcal exp(-i*omega*t)]` 的 phasor 定义。
- 若使用标准解析信号 `P_phys=P^(+)+P^(-)`，则 `P^(+)` 的振幅是 `Pcal/2`。
- number density 来自真实钾蒸气压公式，不得使用 `N_MC/V_slice`。
- 当前天然 K39 fraction 默认 `0.9326`；富集样品必须显式修改。

## 5. 分阶段安排和已经完成的工作

### 前置阶段：库、demo 和旧框架审计

目的：理解 ElecSus、PyLCP、老师 demo 和早期 `hybrid_engine` 各自负责什么，避免把库的适用范围误当成完整物理模型。

完成内容：

- 阅读 ElecSus `spectra.py` 等源码，确认它主要构造线性 susceptibility、介电张量和传播接口。
- 阅读 PyLCP Hamiltonian、Liouvillian、球基底、磁矩、detuning 和场归一化源码。
- 理解 demo 的主要价值是 Monte Carlo 轨迹与 L 矩阵演化思路，而不是可直接继承的完整可靠框架。
- 决定从 demo 思路独立重写，不让新实现依赖 `v5b_demo` 或 `hybrid_engine`。
- 修正并交叉检查 K39 D1 原子常数。

此阶段主要是审计和设计，没有单独的 `step1` 目录。

### Step 2：固定单原子 Liouvillian

目录：`step2_liouvillian/`

问题：单个 K39 原子固定在给定磁场、光强、偏振和频率中时，能否正确构造 L 并演化密度矩阵？

完成内容：

- 构造 8+8 态 K39 D1 Hamiltonian 和无量纲角动量偶极算符。
- 锁定 Fortran-order 密度矩阵向量化。
- 自行构造 coherent 和 spontaneous-decay Liouvillian。
- 与 PyLCP 的完整 256x256 L 比较。
- 比较 `expm(L*t)` 与 PyLCP OBE 数值积分。
- 加入激光 detuning，验证 `H_delta=-delta*P_e` 的符号。
- 验证 trace、Hermiticity 和 minimum eigenvalue。

状态：PASS，冻结。

### Step 3：几何和轨迹历史

目录：`step3_geometry/`

问题：当前时刻位于有限圆柱光束中的一个热原子，从哪个边界进入、飞行多久、经过哪些 z slices？

完成内容：

- 在有限圆柱体内按当前空间快照采样位置。
- 采样三维 Maxwell-Boltzmann 速度。
- 反向追踪到最近合法入口：侧壁、`z=0` 或 `z=L`。
- 将入口到当前位置拆成逐 slice segments。
- 验证 `sum(dt_segments)=time_since_entry`。
- 验证每段次序、边界和 current slice。
- 生成完整 5000 原子几何样本测试。

重要含义：这是稳态“当前快照”采样，不是在入口均匀发射原子，因此不再额外添加人为 flight-time weight。

状态：PASS，冻结。

### Step 4：单个运动原子

目录：`step4_single_moving_atom/`

问题：如何把 Step 2 的 L 和 Step 3 的一条轨迹连接起来？

数据流：

```text
rho_0
-> exp(L_k0*dt_0)
-> exp(L_k1*dt_1)
-> ...
-> rho_final
```

完成内容：

- 每个 segment 使用对应 `B_k`、`s_k`、偏振和 `delta-k*v_z`。
- 原子进入下一片时继承完整密度矩阵，而非只继承 population。
- 零光非均匀 B 测试。
- uniform L 的逐片演化与一次总时间传播等价测试。
- Doppler 正负号测试。
- 随机单原子完整轨迹与密度矩阵诊断。

状态：PASS，冻结。

### Step 4.5：N 原子系综

目录：`step4_5_ensemble/`

问题：如何把单原子 `trajectory -> rho_final` 推广到当前时刻的 N 原子空间系综？

关键规则：

```text
rho_bar[k] = mean(rho_final_all[current_slice == k])
```

历史 slices 只用于形成该原子的 `rho_final`。每个原子只按当前 `current_slice` 向空间平均贡献一次，禁止把历史访问过的切片重复计数。

完成内容：

- binning 与历史多切片访问加固测试。
- `sum(N_k)=N` 和每片计数测试。
- 所有单原子及 `rho_bar[k]` 的 density diagnostics。
- 全系综零光和 uniform-field 等价测试。
- 非均匀场长轨迹 stress test。
- 历史顺序记忆：`B1->B2` 与 `B2->B2` 可产生不同终态。
- 25/50/100/200/400 slices 收敛测试；100 slices 可保留。
- 10/20/50/100/200 atoms per slice、多 seed 的 MC 收敛测试。

状态：PASS，冻结。

### Step 5.0：`rho -> P` convention validation

目录：`step5_0_convention_validation/`

问题：在写正式极化模块前，能否锁定绝对单位、球基底、coherence、共轭、detuning 和 factor 2？

完成内容：

- 锁定 SI 正频率 phasor。
- 验证 Cartesian/spherical 互逆和 `Delta-m=+/-1` Jones vector。
- 验证 `d_q_bare` 是 lowering operator，以及 contraction 使用 `rho_eg`。
- 人工单 coherence 逐项手算。
- 从 `Gamma` 和 `lambda` 推导 `d0`，与 spontaneous-emission 和 PyLCP `Isat` 关系交叉检查。
- 验证 `Ecal <-> s <-> E_q^Py <-> Omega`。
- 两能级弱光解析测试排除错误共轭、整体负号和 factor 2。
- 极简 K39 D1 与 ElecSus 做符号和数量级检查。

状态：PASS，冻结。

### Step 5：正式 `rho_bar -> P`

目录：`step5_polarization/`

问题：如何把每片无量纲 `rho_bar` 转成真实 SI 宏观复极化？

完成内容：

- 独立 `polarization.py`，不把 contraction 散落到 trajectory/ensemble。
- 单矩阵和批量接口。
- `rho -> P_q[-1,0,+1] -> P_cart[x,y,z]`。
- 使用 ElecSus 钾蒸气压经验式计算真实 `N_K39(T)`。
- population-only 必须给 `P=0`。
- 人工单 coherence、随机 Hermitian 显式求和、球基底往返测试。
- 两能级弱光正式接口回归。
- 小型真实 ensemble 和零光完整系综测试。

状态：PASS，冻结。此阶段只输出 `P`，不把一般非线性响应称为 `chi`。

### Step 5A：稳态 OBE / Doppler / ElecSus benchmark

目录：`step5A_steady_state_benchmark/`

原计划问题：弱光、uniform B、热速度分布下，完整无限时间稳态 OBE 是否等于 ElecSus？

发现了一个重要物理边界问题：

- 当前 OBE 没有 ground-state relaxation 或热库重置。
- 纯圆偏振连续照射最终把原子抽运到 stretched dark state。
- 即使 `s` 很小，`t->infinity` 后仍会到暗态，只是时间更长。
- ElecSus 固定使用热平衡/等基态布居的线性响应。

因此：

```text
lim(s->0) lim(t->infinity) rho(t,s)
```

不等于 ElecSus 的固定热平衡线性响应。原始“literal full steady-state 必须复现 ElecSus”的判据应判定为 FAIL，这是模型边界不同，不是通过经验比例可修复的代码 bug。

为诊断代码约定，目录同时建立了明确标注的 `linear_thermal` 一阶响应分支：固定 `rho_g=I/8`，做确定性 Maxwell/Doppler 卷积，再和 ElecSus 比较。该分支成功验证 atomic constants、线强、Doppler、density、detuning 和 q mapping，但不能冒充无弛豫 full OBE 的无限时间稳态。

状态：

- literal steady-state vs ElecSus：按原定义 FAIL，原因已解释。
- fixed-thermal linear diagnostic：PASS，可作为后续弱光基准。

### Step 5B：finite transit-time + optical pumping

目录：`step5B_finite_transit/`

问题：真实原子只有有限驻留时间时，响应如何从 thermal linear limit 过渡到 optical-pumped dark state？

限定条件：uniform `B_z`、top-hat uniform light、K39、无传播、无碰撞、无 ground relaxation。

完成内容：

- 真实 `trajectory -> rho_final -> current-slice rho_bar -> P`。
- uniform L 下逐 segment 与总驻留时间传播的严格等价测试。
- 纯圆偏振和 `I_g/8` 初态下，256 维 Liouville 空间只有 51 维可达；验证该子空间对 L 严格封闭。
- 51 维传播与冻结 full segmented propagation 的最大密度矩阵误差约 `2.84e-13`。
- 5000 原子单失谐完整密度矩阵、时间分箱和空间 profile。
- `s=1e-6...10` 饱和度扫描和 optical-pumping 时间尺度。
- 20/50/100/200 atoms per slice、多 seed MC 收敛。
- 5000 原子、241 detuning points 的有限轨迹完整谱。
- 新增 `06_weak_light_spectrum.py`，直接画 `s=1e-6` 时 finite trajectory、linear thermal OBE 和 ElecSus 三条曲线。

主要结果：

- 典型 median transit time 约 `1.35 us`，样本范围约 `0.003...62.5 us`。
- `s=1e-2` 时 pumping time 约 `71 us`，远长于典型驻留时间，抽运弱。
- `s=1` 时 pumping time 约 `0.81 us`，与驻留时间同量级，有限时间抽运明显。
- `s=10` 时 pumping time 约 `0.19 us`，大量原子来得及进入暗态。
- `s=1` 的 finite-trajectory complex spectrum 相对 linear thermal response 改变约 `46.4%`；这不是 ElecSus 错误，而是 nonlinear optical pumping 改变了跃迁权重。
- linear thermal OBE 与 ElecSus 完整谱 complex L2 约 `9.45e-4`。
- `s=1e-6`、5000 trajectories 时，finite trajectory 与 linear OBE 的 complex correlation 为 `0.99836`；整体范数比约 `1.00242`，但单 seed 全谱 L2 仍约 `5.81%`，主要表现为有限速度采样的锯齿，另含很短驻留原子的 coherence-build-up transient。

严格时间极限必须这样解释：

```text
t=0: P=0
-> optical coherence 建立
-> population 尚未抽运时接近 thermal linear response
-> finite-time optical pumping
-> 长时间趋向 dark state，响应下降
```

不能写成“严格 `t->0` 就等于线性稳态”。减小 `s` 消除 optical pumping，但不会自动消除有限 coherence build-up time。

状态：主要测试 PASS，可冻结为当前 uniform-field finite-transit 基线。

### Step 5C：受控 magnetic-history benchmark

目录：`step5C_magnetic_history/`

问题：在相同当前磁场、光场、总照光时间和入口态下，仅改变过去经历的纵向磁场，当前 `rho` 和 `P` 是否不同；差异能保持多久？

受控比较：

```text
0 G for 1 us -> 300 G for T_post
vs
300 G for 1 us -> 300 G for T_post
```

并做反向 `300 G -> 0 G` 对 `0 G -> 0 G`。统一使用 K39 D1、`I_g/8`、纯 `Delta-m=+1`、`v_z=0`、`s=1e-6`。laser detuning 从冻结 Step 5A 的当前场局域 weak-linear absorption 主峰数值确定。

完成内容：

- 所有正式点直接传播完整 256x256 Liouvillian，没有使用 Step 5B 的 51 维 reduced space。
- identical-history、uniform-L 和 segment-splitting controls 均达到约 `1e-13` 或更低。
- 正向/反向扫描规定的 `T_post=0...10 us`。
- 分解 `Delta rho` 的 `gg/ge/eg/ee` blocks，并用冻结 Step 5 计算 `Delta P_q`。
- 检查 trace、Hermiticity、minimum eigenvalue 和 finite。
- 验证 1 us pre-stage 时 optical coherence 已建立，而 ground-population redistribution 小于初始完整历史差异的 1%。
- 输出完整 NPZ、三个 PNG、JSON controls 和可直接由网页版 AI 阅读的 `outputs/results.md`。

主要结果：

- 磁场切换瞬间存在明确 history effect：正向/反向最大 `||Delta rho||_F` 约 `1.02e-4/1.14e-4`，最大 `|Delta P_q|` 约 `8.74e-19/1.10e-18 C/m^2`。
- 初始差异主要在互为 Hermitian conjugate 的 optical `ge/eg` blocks。
- 在给定离散扫描上，P-memory 的 1/e crossing 位于 `50...100 ns`，10% crossing 位于 `100...200 ns`；没有强行做单指数拟合。
- 到 `1 us`，归一化 P-memory 仅剩约 `4.17e-6/2.52e-6`，所以当前弱光纵向 B 条件下，微秒 transit 上的 local-response approximation 对 P 很好。
- 完整 `rho` 在 `1 us` 仍保留约 `0.677%/0.794%` 的小平台，主要位于 `gg`；其 norm 的 `92.5%/99.999%` 来自对角 population，而不是长寿命 ground coherence。这是无 ground relaxation 模型中的微弱预抽运布居历史。

状态：PASS。结论只适用于弱光、纵向 B、瞬时场切换和当前无碰撞/无 ground relaxation 模型，不能直接外推到横向场、连续梯度或强光。

### Step 5D：真实抛物线 B(z) 下的 local-vs-trajectory benchmark

目录：`step5D_parabolic_field/`

问题：在 25 mm cell、`B(z)=2500 G-500 G*(2z/L)^2` 的连续纵向高场中，按真实时间顺序传播的弱光 trajectory response 是否可以由当前位置的 frozen local response 近似？

限定条件：K39 D1、`I_g/8`、纯 `Delta-m=+1`、top-hat、`s=1e-6`、无 MC、无传播、无碰撞和 ground relaxation；速度为 `v_z=+/-50,+/-100,+/-300,+/-500 m/s`。实验采用固定 laboratory laser frequency，因此改变速度会同时改变 Doppler detuning，速度曲线不能解释成单纯的“适应时间”扫描。

完成内容：

- 在 `B=2000...2500 G` 补做 fixed-thermal weak-linear OBE vs ElecSus 高场基准。
- trajectory 始终保留完整 `(16,16)` 密度矩阵；未使用 Step 5B 的 51 维 reduced space。
- 以真实空间中点的 `B(z)` 做 chronological propagation，禁止平均 B 或单个有效 L。
- full-256 dense `expm` control、constant-B control、`DeltaB->0` control、空间网格收敛和 density-matrix diagnostics 均通过。
- 比较八个正负速度的 `P_traj(z)` 与 `P_local(z)`，并在相同 B 的左右对称位置做 history/hysteresis diagnostic。

主要结果：

- 2000–2500 G 范围 OBE–ElecSus 最差 complex relative error 为 `0.740%`，高场 local reference 本身仍通过验证。
- full-state factorized propagation 对 dense full-256 control 的 rho 相对误差约 `6.94e-10`；target-P 相对误差约 `4.10e-5`。
- `dz=1.25 -> 0.625 um` 的 shared-node full-complex-P 误差约 `6.13e-11`，正式结果已空间收敛。
- 对代表性 `v_z=+300 m/s`，bulk `epsilon_global=30.999%`，最大归一化误差 `42.116%`；`+500 m/s` 最大误差约 `52.845%`。因此在本模型和固定频率条件下，连续高场中的局域近似并不总是小修正。
- 负速度结果明显不同；这是反向路径与 Doppler detuning 同时改变后的真实结果，不能只按 `|v_z|` 解释。
- 预选 2100/2200/2300/2400 G 的左右同-B history 差异小于最大 local discrepancy；最大误差更接近窄空间共振。Step 5C 的约 100 ns switch-memory 只能提供尺度直觉，不能替代连续 chronological propagation。

状态：PASS。这里验证的是 deterministic 单原子弱光连续场 benchmark；尚不能直接外推成热速度 MC 系综结论，也不能把全部差异归因于单一 magnetic lag。

## 6. 当前数值性能和优化边界

最初改用 L 矩阵的主要原因是直接调用 PyLCP OBE 对大量原子、切片和失谐点过慢。

当前策略：

- 小测试使用完整 256 维矩阵，锁定正确性。
- uniform B/top-hat 时，相同 L 的逐 segment 传播可按半群性质合并为总时间传播。
- Step 5B 的纯圆偏振条件使用经过严格不变性验证的 51 维 reachable subspace。
- detuning mode 使用 cache；正式谱当前分辨率通常为 `0.25 MHz`。

注意：51 维子空间是当前初态、偏振和 uniform-field 条件下的结果。加入任意偏振、横向磁场、非均匀光场或新的弛豫项后，必须重新验证 reachable subspace，不能直接假定仍为 51 维。

不要为了速度在未验证时采用以下做法：

- 只传播 population，丢弃 coherence；
- 把历史 slices 重复计入空间平均；
- 用平均速度替代 Doppler 分布；
- 用 `N_MC/V` 替代真实蒸气数密度；
- 用经验比例强行把 nonlinear trajectory 谱拟合到 ElecSus；
- 在非均匀 L 下把所有 segment 错误合并成一个平均 L。

## 7. 当前可信结论

已经由独立测试支持：

1. 固定场 K39 D1 Liouvillian 与 PyLCP 一致。
2. detuning、Doppler、球基底、共轭和 factor 2 已锁定。
3. 有限圆柱当前快照几何和入口历史正确。
4. 单原子可以逐 slice 累积完整密度矩阵历史。
5. 系综平均只按 current slice 计数一次。
6. `rho_bar -> P` 的绝对 SI 单位链成立。
7. fixed-thermal linear OBE 可独立复现 ElecSus susceptibility。
8. 无 ground relaxation 的无限时间圆偏振 OBE 会进入暗态，不等于 ElecSus 热平衡线性模型。
9. finite transit-time 把 coherence 建立、近线性响应和 optical pumping 暗态连接起来。
10. 当前强光下的响应差异是模型预期的历史依赖/非线性效应，不应靠经验缩放消除。
11. 弱光纵向 B 的 optical magnetic memory 在约百纳秒内衰减；微秒时当前 P 已近似局域响应，但完整 rho 可保留很小的 ground-population 历史。
12. 2000–2500 G 的 fixed-thermal weak-linear local response 仍能在 1% complex error 内复现 ElecSus。
13. 在固定实验频率、弱光、无 ground relaxation 的连续抛物线场中，单原子 trajectory response 可在窄空间共振附近明显偏离 frozen local response；该差异已通过完整状态传播和空间收敛检查，但尚不是热系综结论。

## 8. 当前模型假设和未完成内容

以下内容尚未成为已验证物理核心：

- Maxwell 光场传播和 `E(z)` 自洽更新；
- 光场被原子吸收后对后续 slices 的反馈；
- 非均匀 B 下的热速度、大规模 MC finite-transit ensemble benchmark（deterministic 单原子连续场基准已在 Step 5D 完成）；
- Gaussian transverse intensity；
- ground-state relaxation、buffer-gas collision、spin exchange、wall collision；
- 原子离开后 dark evolution 和 re-entry；
- recoil 和速度改变；
- K40/K41 多同位素；
- 实验 cell window、反射、etalon 等效应。

当前入口态仍是 ground manifold 等布居 `I_g/8`。如果未来加入热库重置或 ground relaxation，必须明确速率、Lindblad 形式和物理来源，并新建独立验证阶段。

## 9. MC 噪声的当前判断

整体平均响应和逐 slice profile 的收敛速度不同。Step 5B 在强 pumping 条件下得到的大致 seed dispersion：

| atoms/slice | global response | per-slice `P_q(z)` profile |
|---:|---:|---:|
| 20 | 8.35% | 119% |
| 50 | 8.06% | 86.8% |
| 100 | 5.07% | 63.5% |
| 200 | 2.83% | 44.1% |

因此：

- 可以使用当前样本判断全局谱和 optical-pumping 趋势。
- 不应把当前 `P(z)` 的细小逐片锯齿解释为真实空间结构。
- 在接入 Maxwell propagation 前，需要明确如何降低逐 slice 极化噪声，例如增加样本、多个 seed、common random numbers、严格验证的 variance reduction，或调整空间表示。
- 任何平滑都必须标记为后处理，不能用平滑掩盖未收敛的 MC 估计量。

## 10. 建议的下一阶段

Step 5A–5D 验证路线完成后，不要直接把噪声较大的 `P(z)` 塞进复杂传播循环。下一条主线是 Step 6，并应拆分为：

1. Step 6：先定义独立 Maxwell-only 传播基准，在已知弱光 `chi`、uniform medium 条件下验证 Maxwell/Jones/Beer-Lambert 的符号、单位和步长收敛。
2. Step 7：把冻结 Step 5 的 `P(z)` 接入单向传播，保持原子响应不反馈，只完成一次 `E -> P -> E_new`。
3. Step 8：建立 `E(z) -> trajectories/rho -> P(z) -> E(z)` 自洽迭代，并单独测试收敛、因果顺序和 MC 噪声传播。
4. Step 9 以后再逐项接入实验磁场、强光、热 MC、Gaussian beam、碰撞或其他扩展；不能在一个阶段同时加入多个未经验证的新物理。

从 Step 6 开始，每个阶段的脚本使用阶段前缀编号，例如 `6_1_*`、`6_2_*`，下一阶段再使用 `7_1_*`、`7_2_*`。四个已有验证目录继续保持 `step5A_*`、`step5B_*`、`step5C_*`、`step5D_*`。

具体下一步必须由用户确认，不应由 AI 一次扩展多个新物理模块。

## 11. 运行和修改规则

### 11.1 阅读顺序

新的 AI 应按以下顺序读取：

1. 本 `agent.md`。
2. 当前任务涉及阶段的 `README.md`。
3. 该阶段的核心模块。
4. 编号测试脚本。
5. `run_all.py`。
6. 对应 `outputs/`。
7. 只有 convention 或库行为不明确时，才回到本地 ElecSus/PyLCP 源码。

### 11.2 修改原则

- 用户要求“分析/诊断”时，不自动修改物理代码。
- 用户要求“实现下一步”时，新建独立目录或独立测试脚本。
- 不修改冻结阶段，除非测试明确发现 bug 且用户同意修复。
- 修复 bug 时必须记录：原行为、失败测试、原因、修复和回归范围。
- 不删除旧输出来伪装新测试通过。
- 每次昂贵运行使用固定 seed，并把 seed、参数、单位和版本写入输出。
- 结论必须区分 PASS、FAIL、数值趋势和模型假设。

### 11.3 推荐测试输出

每个新阶段至少保存：

```text
README.md             # 阶段目的、边界、运行命令和结论
run_all.py            # 正式逐项入口
outputs/*.npz         # 完整机器可读数组
outputs/*.png         # 网页版可直接查看的科学绘图
outputs/results.md    # 关键参数、误差、PASS/FAIL 和运行时间
```

未来不要只保存 `.npz`。GitHub 网页和网页版 ChatGPT 不一定能直接解析二进制 NPZ；必须同时提供可直接读取的 `results.md`，必要时再提供 JSON/CSV。

## 12. 本地 Agent、GitHub 与网页版 ChatGPT 的交接规范

用户的工作模式是：本地 Agent 修改和运行代码，`step_by_step` 上传到 GitHub，网页版 ChatGPT 通过 GitHub 链接直接读取代码、图像和结果，不再由用户手工上传文件。

### 12.1 最重要的可见性规则

网页版只能看到已经 commit 并 push 到 GitHub 的内容。它看不到：

- 本地未提交修改；
- 只 commit 未 push 的提交；
- 本地新生成但被 `.gitignore` 排除的输出；
- 本地 ElecSus/PyLCP 源码，除非它们也有可访问链接。

因此本地 Agent 在交接前必须检查并明确报告：

```powershell
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
```

不要在用户没有要求时擅自 commit 或 push。但如果尚未 push，必须明确告诉用户“以下结果目前只有本地可见，网页版 ChatGPT 还无法读取”。

### 12.2 GitHub 链接必须绑定 commit

正式交接优先使用 commit SHA，而不是只给会变化的 `main`：

```text
代码/Markdown：
https://github.com/tiacn/atom/blob/<commit-sha>/<path>

Raw 文本或下载：
https://raw.githubusercontent.com/tiacn/atom/<commit-sha>/<path>
```

`main` 链接可作为方便入口，但物理结论必须注明对应 commit SHA，以免后续代码变化后网页 AI 读取到不同版本。

### 12.3 每次交接给网页版 AI 的最小信息

本地 Agent 的最终交接内容至少包括：

1. GitHub repository、branch、commit SHA。
2. 本次任务目标和明确不在范围内的物理。
3. 修改/新增文件的 GitHub 链接。
4. 正式运行命令和总运行时间。
5. PASS/FAIL 列表。
6. 关键参数：温度、B、s、detuning axis、偏振、slices、atoms/slice、seeds。
7. 关键误差指标和 density-matrix diagnostics。
8. PNG 图像链接和 `results.md` 链接。
9. 已发现 bug、数值近似和残余风险。
10. 哪些模块仍冻结、下一步建议是什么。

推荐交接模板：

```text
仓库：<repo URL>
分支：<branch>
提交：<full commit SHA>

本次范围：...
明确未加入：...

代码：<blob links>
结果摘要：<results.md link>
图像：<png links>
原始数据：<npz/csv links>

运行命令：...
运行时间：...
结果：PASS/FAIL
主要指标：...
残余问题：...
下一步：...
```

### 12.4 网页版 ChatGPT 的推荐提示语

用户可以把以下文字和 commit 链接发给网页版 ChatGPT：

```text
请先完整阅读该提交中的 agent.md，再阅读我指定阶段的 README、核心脚本、
outputs/results.md 和 PNG。所有结论必须绑定这个 commit SHA。
请区分：代码已验证结论、数值收敛程度、模型假设和尚未实现内容。
如果需要检查 NPZ 中的精确数组，先说明 GitHub 页面是否能直接读取；
不要仅凭图片猜测精确数值，也不要把本地未 push 的结果当作仓库内容。
```

### 12.5 本地 Agent 接收网页版反馈时

网页版 AI 的建议不是自动授权修改。收到反馈后，本地 Agent应：

1. 确认反馈引用的 commit 和文件版本。
2. 在本地源码中复查对应行，而不是只依赖网页摘要。
3. 把建议分类为：明确 bug、需新增测试、物理模型选择或表达改进。
4. 对明确 bug 先写失败测试，再修复。
5. 对模型选择向用户解释影响，不能替用户决定碰撞率、弛豫模型或传播近似。
6. 完成后生成新的 commit/结果链接，形成下一轮可追踪交接。

## 13. 当前重要输出索引

GitHub 中路径均相对于仓库根目录。

### Step 5A

- `step5A_steady_state_benchmark/outputs/linear_thermal_vs_elecsus.npz`
- `step5A_steady_state_benchmark/outputs/literal_steady_state_dark_test.npz`
- `step5A_steady_state_benchmark/outputs/velocity_integration_convergence.npz`

### Step 5B

- `step5B_finite_transit/outputs/single_detuning_time_and_space.png`
- `step5B_finite_transit/outputs/single_detuning_5000_atoms.npz`
- `step5B_finite_transit/outputs/saturation_crossover.png`
- `step5B_finite_transit/outputs/saturation_scan.npz`
- `step5B_finite_transit/outputs/mc_convergence.png`
- `step5B_finite_transit/outputs/mc_convergence.npz`
- `step5B_finite_transit/outputs/detuning_spectrum.png`
- `step5B_finite_transit/outputs/detuning_spectrum_5000_atoms.npz`
- `step5B_finite_transit/outputs/weak_light_detuning_spectrum.png`
- `step5B_finite_transit/outputs/weak_light_detuning_spectrum_5000_atoms.npz`

### Step 5C

- `step5C_magnetic_history/outputs/results.md`
- `step5C_magnetic_history/outputs/controls.json`
- `step5C_magnetic_history/outputs/magnetic_memory_decay.png`
- `step5C_magnetic_history/outputs/rho_block_memory.png`
- `step5C_magnetic_history/outputs/current_field_response_recovery.png`
- `step5C_magnetic_history/outputs/forward_0G_to_300G.npz`
- `step5C_magnetic_history/outputs/reverse_300G_to_0G.npz`

### Step 5D

- `step5D_parabolic_field/outputs/results.md`
- `step5D_parabolic_field/outputs/high_field_benchmark.json`
- `step5D_parabolic_field/outputs/propagator_controls.json`
- `step5D_parabolic_field/outputs/spatial_convergence.json`
- `step5D_parabolic_field/outputs/trajectory_benchmark.json`
- `step5D_parabolic_field/outputs/trajectory_vs_local.png`
- `step5D_parabolic_field/outputs/local_error_vs_position.png`
- `step5D_parabolic_field/outputs/velocity_dependence.png`

读取图像时要结合生成脚本和参数，不要只看文件名。特别是：

- `detuning_spectrum.png` 使用 `s=1`，绿线是 nonlinear finite-trajectory effective response。
- `weak_light_detuning_spectrum.png` 使用 `s=1e-6`，用于检查三条曲线在弱 pumping 极限下回归。

## 14. 给后续 AI 的最终提醒

本项目最需要避免的不是语法错误，而是把不同物理边界条件混为一谈：

- ElecSus fixed thermal linear response 不等于无弛豫 OBE 的无限时间暗态。
- `s->0` 消除 pumping，不代表严格 `t=0` 已建立稳态 coherence。
- 历史经过某 slice 不代表原子当前应在该 slice 被再次计数。
- MC 粒子数不是真实 number density。
- 强光下 `P/E` 不是一般线性 susceptibility。
- uniform-L 优化不能未经测试推广到 nonuniform field。
- 一张平滑或重合的图不能代替单位、符号、全矩阵和收敛测试。

继续工作时，坚持“一步一个物理问题、一个独立目录、一组可复查测试、一次 GitHub commit 交接”。
