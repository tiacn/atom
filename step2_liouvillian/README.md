# 第二步：独立验证 K39 D1 Liouvillian

这个目录只依赖工作区中的 `pylcp` 库，不导入：

- `MC/v5b_demo`
- `MC/hybrid_engine`
- ElecSus

目的不是马上计算完整蒸气池，而是逐层回答：

1. K39 D1 的 8 个基态和 8 个激发态是否构造正确？
2. 密度矩阵怎样变成 256 维向量？
3. 自己构造的 Liouvillian 是否和 pylcp 的 OBE 一致？
4. `exp(L*t)` 是否和 pylcp 的数值积分给出相同的密度矩阵？

## 明确采用的约定

### 1. 单位

所有 Hamiltonian 和 Liouvillian 都使用角频率单位：

- Hamiltonian：rad/s
- 衰减率 gamma：rad/s
- 时间：s
- 磁场：Gauss

因此文献中以 Hz 给出的超精细常数和 Bohr 磁子需要乘 `2*pi`。

### 2. 密度矩阵向量化

pylcp 的密度矩阵索引是：

```text
index(i, j) = i + j*N
```

这是列优先（Fortran order）。本目录统一使用：

```python
rho_vec = rho.reshape(-1, order="F")
rho = rho_vec.reshape(N, N, order="F")
```

不能和默认的 `rho.flatten()`（C order）混用。

### 3. 球基底

使用 pylcp 的顺序：

```text
[q=-1, q=0, q=+1]
```

笛卡尔到球基底的转换直接使用 `pylcp.common.cart2spherical`。

### 4. 偶极矩方向

`dqij_two_hyperfine_manifolds()` 返回：

```text
d_q.shape = (3, N_ground, N_excited)
```

行是 ground，列是 excited；它已经是激发态到基态的 lowering operator，
传给 `ham.add_d_q_block("g", "e", d_q)` 时不再转置。

## 运行顺序

在工作区根目录 `D:\Code\python\atom` 下依次运行：

```powershell
python MC\step_by_step\step2_liouvillian\01_atomic_model.py
python MC\step_by_step\step2_liouvillian\02_vectorization.py
python MC\step_by_step\step2_liouvillian\03_compare_L.py
python MC\step_by_step\step2_liouvillian\04_compare_evolution.py
python MC\step_by_step\step2_liouvillian\05_detuning.py
```

也可以一次运行：

```powershell
python MC\step_by_step\step2_liouvillian\run_all.py
```

每个脚本成功时会打印 `PASS`。如果某一步失败，应先停在该步，不继续解释后续结果。

## 每个脚本回答什么

### `01_atomic_model.py`

- 基态、激发态是否都是 8 维；
- 超精细分裂数量级是否正确；
- 偶极矩列归一化是否正确；
- 跃迁是否满足代码采用的 `m_g = m_e + q` 选择规则。

### `02_vectorization.py`

- 用一个小矩阵直接验证 `L_H vec(rho) = vec(-i[H,rho])`；
- 展示 demo 中 C-order/列主序混用为什么会失败；
- 验证正确 Liouvillian 保持迹和 Hermiticity。

### `03_compare_L.py`

分别比较自己构造和 pylcp 构造的：

- 自发辐射项；
- 无光、有磁场的完整 L；
- 有光、有磁场的完整 L。

这里比较的是整个 256×256 矩阵，不只是迹或激发态布居。

### `04_compare_evolution.py`

在恒定光场、恒定磁场、零 detuning 下比较：

- 自己的 `expm(L*t)`；
- 由 pylcp OBE 预计算矩阵组装出的参考 L，再用 `solve_ivp` 积分。

通过标准是两个完整密度矩阵逐元素一致，同时检查迹、Hermiticity 和最小本征值。

### `05_detuning.py`

- 对外使用 MHz，内部转换为 rad/s；
- 在旋转框架使用 `H_delta = -delta*P_e`；
- 验证失谐项只直接作用于 g-e 光学相干，不直接改变布居；
- 验证静态旋转框架 L 与 pylcp 中带时间相位的失谐激光严格等价；
- 检查正、负失谐的符号。

后续加入轴向速度时，应使用：

```text
delta_eff = 2*pi*detuning_MHz*1e6 - k*v_z
```

但 Doppler 项不属于当前第二步，本目录暂时只验证静止原子的激光失谐。
