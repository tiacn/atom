# Step 4.5：N 原子系综集成验证

本目录只连接已经通过的：

- `step2_liouvillian`
- `step3_geometry`
- `step4_single_moving_atom`

不修改 OBE 内核，不导入旧 demo、`hybrid_engine`、ElecSus，也不计算极化率或传播光场。

## 系综定义

每个原子独立执行已有的：

```python
trajectory -> evolve_trajectory() -> rho_final
```

系综模块只保留：

```text
rho_final_all[a]
current_slice[a]
```

原子的历史切片只用于计算该原子的内部态，不参与空间 binning。每个原子只按照当前时刻的 `current_slice` 被计数一次：

```text
rho_bar[k] = mean(rho_final_all[current_slice == k])
```

## 脚本

```powershell
python MC\step_by_step\step4_5_ensemble\01_ensemble_binning.py
python MC\step_by_step\step4_5_ensemble\02_zero_light.py
python MC\step_by_step\step4_5_ensemble\03_uniform_field.py
python MC\step_by_step\step4_5_ensemble\04_stress.py
python MC\step_by_step\step4_5_ensemble\05_history_memory.py
python MC\step_by_step\step4_5_ensemble\06_slice_convergence.py
python MC\step_by_step\step4_5_ensemble\07_mc_convergence.py
```

也可以运行：

```powershell
python MC\step_by_step\step4_5_ensemble\run_all.py
```

`07_mc_convergence.py` 在一个代表性空间切片中使用用户指定的
`10/20/50/100/200 atoms per slice` 和 10 个 seed。这样直接检验每片
Monte Carlo 估计量本身的收敛，而不把 100 个空间切片的计算量混入统计测试。
它仍然逐个执行完整的 16 态单原子演化，是本目录最慢的脚本。

设置环境变量 `STEP45_MC_QUICK=1` 时仍保留上述五种样本量，但只使用
3 个 seed，适合快速回归；默认 full 模式才是正式的 10-seed 加固验证。

## 当前仍保持的模型假设

- 原子进入光束时，ground manifold 8 个态等布居；
- 暗区不演化；
- 不处理 re-entry；
- 不处理碰撞、基态弛豫、壁碰撞和 recoil；
- 光场仍由外部指定为饱和参数 `s`；
- 当前不计算 `rho -> chi`。
