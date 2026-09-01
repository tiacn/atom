# Step 5.5A：steady-state benchmark 及暗态诊断

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

不等于 ElecSus 使用的固定热平衡线性响应。按原判据，Step 5.5A 必须判定为 FAIL，不能用经验比例修正。

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
