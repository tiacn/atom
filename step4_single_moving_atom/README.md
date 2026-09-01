# 第四步：单个运动原子的密度矩阵历史

本目录把已经验证的两个模块连接起来：

- `step2_liouvillian`：固定场中的 K39 D1 Liouvillian；
- `step3_geometry`：有限光束柱中的入口、轨迹和逐切片停留时间。

它不导入 `v5b_demo`、`hybrid_engine` 或 ElecSus。删除旧 demo 后仍可运行。

## 本阶段计算什么

对一条已经生成的原子轨迹：

```text
entry -> slice k0 -> slice k1 -> ... -> current position
```

在每个切片构造：

```text
L_k = L(B_k, s_k, polarization_k, delta_eff)
```

其中：

```text
delta_eff = 2*pi*detuning_MHz*1e6 - k*v_z
```

再依次演化：

```text
rho_{j+1} = exp(L_k*dt_k) rho_j
```

原子进入下一切片时继承上一切片的完整密度矩阵。

## 当前物理约定

- K39 D1，16 维 Hilbert 空间；
- 初态为 8 个 ground states 等布居；
- 光沿 `+z`；
- 磁场暂时为每片给定的 `B_z`；
- 光强暂时使用 pylcp 饱和参数 `s`；
- 每片内 `B`、`s`、偏振和 L 都视为常数；
- 不考虑碰撞、壁碰撞、基态弛豫和 recoil；
- 不做系综平均、极化率或光场传播。

实际电场 `V/m` 与饱和参数 `s` 的标定留到微观模型和宏观传播连接时处理。

## 运行顺序

```powershell
python MC\step_by_step\step4_single_moving_atom\01_hand_trajectory.py
python MC\step_by_step\step4_single_moving_atom\02_zero_light.py
python MC\step_by_step\step4_single_moving_atom\03_uniform_field.py
python MC\step_by_step\step4_single_moving_atom\04_doppler.py
python MC\step_by_step\step4_single_moving_atom\05_random_atom.py
```

或一次运行：

```powershell
python MC\step_by_step\step4_single_moving_atom\run_all.py
```

## 每个脚本验证什么

### `01_hand_trajectory.py`

使用手工指定的位置和速度，只检查入口、切片顺序和停留时间，让几何结果可以人工阅读。

### `02_zero_light.py`

在 `s=0`、非均匀磁场下演化。ground manifold 的单位矩阵应保持不变，因此最终态必须仍是热初态。

### `03_uniform_field.py`

所有切片使用同一个 L，验证：

```text
exp(L*dt_n)...exp(L*dt_1) rho0
= exp(L*sum(dt)) rho0
```

### `04_doppler.py`

验证运动原子的 `delta-k*v_z` 与“静止原子直接修改激光失谐”严格等价，并检查正负速度的符号。

### `05_random_atom.py`

随机生成一个原子，输出完整轨迹、每片参数、每片激发态布居以及最终密度矩阵诊断。

