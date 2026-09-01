# Step 5：`rho_bar -> dipole -> P`

本目录在不修改 Step 2–4.5 和 Step 5.0 的前提下，把逐 slice 的系综平均密度矩阵转换为 SI 宏观复极化。本阶段不计算一般 susceptibility，也不做 Maxwell 传播。

统一 convention：

```text
E_phys = Re[Ecal exp(i*k*z-i*omega*t)]
P_phys = Re[Pcal exp(-i*omega*t)]
PyLCP spherical order = [q=-1,0,+1]
C_q = d_q_bare = |g><e|
Pcal_q = 2*N_K39*d0*Tr(C_q*rho_bar)
```

其中 `Tr(C_q rho_bar)` 选择 `rho_eg`，`Pcal_q` 和 `Pcal_cart` 的单位都是 `C/m^2`。

## 数密度边界

`potassium_number_density_m3(T_K)` 使用与 ElecSus `numberDensityEqs.py::numDenK` 相同的钾蒸气压经验式，先得到总钾数密度。当前 OBE 只模拟 K39，因此再乘 K39 同位素比例：

```text
N_K39 = f_K39 * N_total_K
```

默认采用 ElecSus 的天然钾比例 `f_K39=1-0.0001-0.0673=0.9326`。富集样品必须显式传入实验比例。`atoms_per_slice`、`counts[k]` 和任何 MC 样本体积都不进入该公式。

## 正式接口

```python
density_matrix_to_polarization(rho, d_q_bare, number_density_m3)
density_matrices_to_polarization(rho_bar, d_q_bare, number_density_m3)
save_polarization_profile(profile, output_path)
```

批量结果 shape：

```text
rho_bar : (n_slices,N,N), dimensionless
P_q     : (n_slices,3), C/m^2, order [-1,0,+1]
P_cart  : (n_slices,3), C/m^2, order [x,y,z]
```

运行全部小测试：

```powershell
python run_all.py
```

测试覆盖 population-only、人工单 coherence、随机 Hermitian contraction、球/直角基底往返、两能级弱光正式接口回归，以及小型真实 ensemble 和全系综零光极限。最后一个测试把正式输出和单位/convention 元数据保存到 `outputs/*.npz`。
