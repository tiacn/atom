# Step 5C：受控 magnetic-history benchmark

本目录在不修改 Step 2–5B 冻结代码的前提下，用 deterministic 单原子实验回答：过去经历的纵向磁场是否影响当前 `rho` 和 SI 宏观极化 `P`，以及这种记忆在切换到当前磁场后持续多久。

## 正式比较

```text
History:         0 G for T_pre -> 300 G for T_post
Local reference: 300 G for T_pre -> 300 G for T_post
```

反向检查：

```text
History:         300 G for T_pre -> 0 G for T_post
Local reference:   0 G for T_pre -> 0 G for T_post
```

共同参数：K39 D1、`rho_entry=I_g/8`、纯 `Delta-m=+1`、`v_z=0`、top-hat constant light、`s=1e-6`、`T_pre=1 us`。每组 laser detuning 从冻结 Step 5A fixed-thermal local weak-linear absorption 数值确定，并明确保存 ElecSus external axis 与 Step 4 internal detuning。

## 传播和边界

所有正式时间点直接使用完整 256x256 Liouvillian：

```text
rho_H = exp[L(B_current) T_post] exp[L(B_past) T_pre] rho_entry
rho_R = exp[L(B_current) T_post] exp[L(B_current) T_pre] rho_entry
```

不使用 Step 5B 的 51 维 reduced space，不使用平均 B，不交换时间顺序。本阶段不加入 MC、Maxwell propagation、Gaussian beam、碰撞、ground relaxation 或连续梯度。

## 测试

1. `01_controls.py`
   - 数值确定 300 G 当前场的主吸收峰；
   - identical-history control；
   - uniform-L 两段与总时间一次传播；
   - segment splitting；
   - `T_pre=1 us` 的 coherence plateau、fixed-thermal linear response 和 weak-pumping diagnostics；
   - Liouvillian commutator。
2. `02_memory_scan.py`
   - 正向与反向历史的规定 `T_post` 扫描；
   - `Delta rho` 的 `gg/ge/eg/ee` block 分解；
   - 冻结 Step 5 的 `P_q/P_cart`；
   - trace、Hermiticity、minimum eigenvalue、finite；
   - 三张正式图、完整 NPZ 和 `outputs/results.md`。

运行：

```powershell
python run_all.py
```

正式结论以 `outputs/results.md` 为准。阈值时间来自用户指定的离散扫描点；若曲线振荡或存在多时间尺度，不做强制单指数拟合。
