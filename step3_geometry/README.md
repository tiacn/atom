# 第三步：独立验证原子几何与运动历史

这个目录只依赖 NumPy，不导入：

- `MC/v5b_demo`
- `MC/hybrid_engine`
- pylcp
- ElecSus

第三步不计算密度矩阵。它只回答：一个在光束柱内被观察到的原子，从哪里进入光束、以什么速度运动、到当前位置之前依次经过了哪些 z 切片、每片停留了多久。

## 当前几何

- 光束沿 `+z`；
- 光束柱半径 `R = 0.5 mm`；
- 蒸气池/光束柱长度 `L = 25 mm`；
- `z` 范围 `[0,L]`；
- 100 个等厚切片；
- 每片采样 50 个当前时刻的原子，共 5000 个；
- 温度 70 摄氏度；
- K39 速度服从三维 Maxwell-Boltzmann 分布。

横向采用 top-hat 光束柱。高斯光强分布不属于几何采样，会在以后给每条轨迹分配局域光场时处理。

## “当前快照采样”的含义

我们在某一时刻，从每个切片的光束体积内均匀抽取原子的当前位置，同时从热平衡分布抽取速度。然后沿 `-v` 反向追踪，找到原子最近一次进入有限光束柱的位置。

这种方法直接构造稳态系综的空间快照。它不是先在边界均匀发射原子，因此不需要另外手工加入飞行时间权重。

## 关键几何规则

### 1. 入口是反向射线遇到的第一个合法边界

从当前点 `r` 沿：

```text
r_past(t) = r - v*t,  t > 0
```

反向追踪。入口可以是：

- 圆柱侧壁；
- `z=0` 端面；
- `z=L` 端面。

侧壁交点必须同时满足 `0 <= z <= L`；端面交点必须仍在圆盘半径内。所有合法正时间中取最小值，因为它是反向射线离开当前凸区域时首先遇到的边界。

### 2. 只积累入口到当前位置

轨迹历史区间是：

```text
entry -> current position
```

当前位置之后直到未来出口的轨迹不属于当前密度矩阵的历史，不能加入停留时间。

### 3. 切片时间

把入口到当前位置的线段与所有 z 切片边界求交，得到：

```text
slice_indices = [k0, k1, ...]
dt_slices     = [dt0, dt1, ...]
```

必须满足：

```text
sum(dt_slices) = time_since_entry
```

## 运行顺序

在工作区根目录运行：

```powershell
python MC\step_by_step\step3_geometry\01_positions.py
python MC\step_by_step\step3_geometry\02_velocities.py
python MC\step_by_step\step3_geometry\03_entry_points.py
python MC\step_by_step\step3_geometry\04_slice_history.py
python MC\step_by_step\step3_geometry\05_full_ensemble.py
```

或一次运行：

```powershell
python MC\step_by_step\step3_geometry\run_all.py
```

## 每个脚本验证什么

### `01_positions.py`

- 每片原子数严格相同；
- 所有位置都在有限圆柱内；
- 圆盘采样满足 `E[r^2]=R^2/2`；
- 每片内部的 z 坐标均匀。

### `02_velocities.py`

- `v_x,v_y,v_z` 都服从均值 0、标准差 `sqrt(kT/m)` 的正态分布；
- 三个方向无相关；
- 平均速率和均方速率符合 Maxwell 理论；
- 方向各向同性。

### `03_entry_points.py`

- 侧壁、前端面、后端面三个手算案例；
- 所有随机入口都位于合法有限圆柱边界；
- 入口到当前位置整段在柱内；
- 入口之前立即位于柱外。

### `04_slice_history.py`

- 切片按运动方向依次排列；
- 所有停留时间为正；
- 时间之和等于入口飞行时间；
- 分段首尾准确重建入口和当前位置；
- 最后一个切片就是原子当前所在切片。

### `05_full_ensemble.py`

使用目标参数生成 5000 条完整轨迹，运行所有端到端几何不变量，并报告入口类型、飞行时间和穿越切片数统计。

