# Step 5.0：rho -> P 前的 convention validation

本目录只验证 convention 和绝对单位链，不修改冻结的 Step 2–4.5，
不实现完整 `rho_bar -> P`，也不做传播。

统一使用

```text
E_phys(r,t) = Re[Ecal_0 exp(i*k*z - i*omega*t)]
```

其中 `Ecal` 是 SI 正频率 phasor 峰值，强度为
`I=(c*epsilon_0/2)|Ecal|^2`。

逐项运行：

```powershell
python run_all.py
```

测试内容：

1. SI phasor、PyLCP `[q=-1,0,+1]` 顺序和 Cartesian/spherical 互逆。
2. `d_q_bare=|g><e|` 以及 `Tr(C_q rho)=rho_eg`。
3. `Gamma -> d0 -> Isat -> Ecal -> Omega -> sqrt(2s)` 绝对单位链。
4. 两能级弱光解析响应锁定 `P^(+)` 的符号、共轭和 factor 2。
5. 明确 Jones vector 对应的球分量和 `Delta m`，不依赖 left/right 名称。
6. 极简冷原子 K39 D1 与 ElecSus susceptibility 的量级和符号交叉检查。

## 已验证结论

SI 正频率 phasor 与强度：

```text
E_phys(r,t) = Re[Ecal(r) exp(-i*omega*t)]
Ecal(r) = Ecal_0 exp(+i*k*z)
I = (c*epsilon_0/2) |Ecal|^2
```

PyLCP 的 normalized field 是上述 SI 正频率包络的负频率/复共轭表示：

```text
E_q^Py = (2*d0/(hbar*Gamma)) cart2spherical(conj(Ecal_cart))
|E^Py| = sqrt(2*s)
s = I/Isat
Omega = d0*|Ecal|/hbar = Gamma*sqrt(s/2)
```

绝对常数链：

```text
d0^2 = 3*pi*epsilon_0*hbar*c^3*Gamma/omega^3
Isat = hbar*omega^3*Gamma/(12*pi*c^2)
d0 = 2.462565313e-29 C*m
Isat = 17.045641111 W/m^2 = 1.704564111 mW/cm^2
```

`d_q_bare[q]` 是 lowering operator `C_q=|g><e|`，因此
`Tr(C_q rho)` 选取 `rho_eg`。采用本目录的 phasor convention 时，
定义极化 phasor `Pcal` 使

```text
P_phys = Re[Pcal exp(-i*omega*t)]
Pcal_q = 2*N*d0*Tr(C_q*rho)
```

若另用标准解析信号记号 `P_phys=P^(+)+P^(-)`，则

```text
P_q^(+)(t) = N*d0*Tr(C_q*rho)*exp(-i*omega*t) = Pcal_q*exp(-i*omega*t)/2
```

所以 factor 2 只属于 `Re[phasor]` 中的 `Pcal`，不属于标准分解的
`P^(+)` 本身。两能级弱光测试已分别排除 phasor 漏 factor 2、使用
`rho_ge` 和整体负号的候选。

PyLCP 球数组顺序是 `[q=-1,0,+1]`。不要依赖 left/right 名称；对沿
`+z` 传播的本目录 SI 正频率 Jones vector：

```text
(x+i*y)/sqrt(2): Ecal_q=[1,0,0], drives Delta-m=+1
(x-i*y)/sqrt(2): Ecal_q=[0,0,-1], drives Delta-m=-1
x:                 drives both Delta-m=+1 and -1
```

ElecSus 的 `dipoleStrength` 是电子 reduced dipole，数值约为
`sqrt(3)*d0`；其 transition strength 中另含 `1/3`，所以最终绝对线强
与 PyLCP normalized `d_q` 一致。测试 6 的比较关闭了 Doppler、限定
K39、B 约为零，只验证共振位置、吸收符号、复杂线形和绝对数量级；
它不是完整 ElecSus benchmark。
