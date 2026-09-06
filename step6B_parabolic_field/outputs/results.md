# Step 6B results

## Status

PASS: the high-field local benchmark, full-state propagator controls, continuous-field convergence, density-matrix diagnostics, and deterministic trajectory comparisons completed.

## Fixed physical setup

- K39 D1, 16-state Hilbert space and all 256 density-matrix components.
- Cell length: `25.0 mm`, centered at `z=0`.
- Field: `B(z)=2500 G - 500 G*(2z/L)^2`, hence `2000 G` at both ends and `2500 G` at the center.
- Entry state: equal population in the eight ground states (`I_g/8`).
- Polarization: pure `Delta-m=+1`; top-hat light; `s=1.0e-06`.
- Fixed laboratory laser detuning: `+4460.216578 MHz`, the frozen weak-linear absorption maximum at `B=2500 G`, `v_z=0`.
- Moving atoms use the frozen convention `delta_eff=delta_laser-k*v_z`; therefore changing velocity also changes Doppler detuning.
- Local reference: frozen weak-linear response at the instantaneous `B(z)` and the same Doppler-shifted detuning. It has no previous-position history.
- The first `5*tau_memory=500 ns` after entry is excluded only in the reported bulk metrics. Raw arrays retain the physical entrance transient.

## High-field local benchmark

- Fields: `2000, 2100, ..., 2500 G`.
- Worst OBE-vs-ElecSus complex relative L2 error: `7.403921e-03` (0.740%).
- Worst absorption-peak position error on the 2 MHz output axis: `4.000 MHz`.
- Minimum complex correlation: `0.999990524`.

## Propagator and convergence controls

- Propagation retains the full `16x16 rho` (256 complex components); reduced-space projection is not used.
- A fourth-order symmetric coherent/decay factorization is used for speed. Its constant-B full-cell result was compared with one direct dense `expm(L*t)` in the full 256-dimensional Liouville space.
- Constant-B rho relative error: `6.941098e-10`.
- Constant-B target-P relative error: `4.100874e-05`.
- Maximum internal time step: `1.250 ns`.
- Spatial output grids tested: `10.0, 5.0, 2.5, 1.25, 0.625 um` for `v=+300 m/s`.
- `1.25 -> 0.625 um` shared-node full-complex-P error: `6.131367e-11`.
- `1.25 -> 0.625 um` final-P error: `2.179155e-11`.
- `1.25 -> 0.625 um` bulk-global metric change: `3.184990e-07`.
- Linear interpolation from 1.25 to 0.625 um has a larger `2.686227e-03` reconstruction error because it does not resolve all between-node complex phase. This is an output-sampling diagnostic, not a propagation discrepancy at common positions.
- `DeltaB=0` bulk global/max deviations are `2.723006e-03` / `2.749102e-03`. The remaining 0.27% floor is the finite-entry/weak-pumping transient relative to a frozen thermal local model, not magnetic-gradient history.

## Local-versus-trajectory error

All errors are normalized by `max_z |P_local|`; bulk values exclude the first 500 ns of travel.

| v_z (m/s) | global bulk | max bulk | max position (mm) | max DeltaB in 100 ns (G) |
|---:|---:|---:|---:|---:|
| +50 | 0.038522 | 0.090540 | +8.7650 | 0.400 |
| +100 | 0.088971 | 0.178017 | +9.3700 | 0.800 |
| +300 | 0.309987 | 0.421158 | +11.4638 | 2.400 |
| +500 | 0.475623 | 0.528454 | +12.4050 | 4.000 |
| -50 | 0.034989 | 0.077416 | -7.4200 | 0.400 |
| -100 | 0.040472 | 0.132919 | -6.6500 | 0.800 |
| -300 | 0.032728 | 0.051759 | -0.8500 | 2.400 |
| -500 | 0.010242 | 0.033599 | +4.9800 | 4.000 |

For the representative `+300 m/s` trajectory, `epsilon_global=0.309987` and `max epsilon_P=0.421158` at `z=+11.4638 mm`. The largest max error in the requested velocity set is `0.528454` for `v=+500 m/s`.

The velocity plot is not a pure speed-only experiment: one fixed laboratory laser is used, so the Doppler shift changes with velocity as required by the physical trajectory. A non-monotonic curve therefore cannot be attributed only to adaptation time.

## Same-B hysteresis diagnostic

Entries are `|P_traj(left)-P_traj(right)| / max_z|P_local|` at symmetric positions having the same B. The frozen local model's left/right symmetry error is separately checked at numerical precision.

| v_z (m/s) | 2100 G | 2200 G | 2300 G | 2400 G |
|---:|---:|---:|---:|---:|
| +50 | 0.000001 | 0.000005 | 0.000056 | 0.001036 |
| +100 | 0.000001 | 0.000221 | 0.000654 | 0.000021 |
| +300 | 0.000709 | 0.000776 | 0.006234 | 0.000002 |
| +500 | 0.000981 | 0.000078 | 0.000000 | 0.000001 |
| -50 | 0.000003 | 0.000001 | 0.000049 | 0.000831 |
| -100 | 0.000001 | 0.000001 | 0.000007 | 0.000039 |
| -300 | 0.000002 | 0.000003 | 0.000001 | 0.000003 |
| -500 | 0.000005 | 0.000124 | 0.000056 | 0.000026 |

## Density-matrix checks

- Worst trace error across saved trajectory points: `2.252019e-10`.
- Worst Hermiticity error: `1.720856e-13`.
- Minimum eigenvalue: `-2.035823e-16`.
- All saved `rho`, `P`, B, and position arrays are finite.

## Physical conclusion

1. The high-field frozen weak-linear OBE agrees with ElecSus throughout 2000--2500 G to below 1% complex relative error, so the local reference itself remains validated in this range.
2. The chronological continuous-B propagation is numerically converged and obeys trace, Hermiticity, positivity, constant-B, and `DeltaB->0` controls.
3. At `v=+300 m/s`, the trajectory differs from the frozen local response by about `31.0%` globally and `42.1%` at the largest point. This is not a small locality correction under the exact fixed-frequency/no-ground-relaxation model tested here.
4. The same-B left/right comparison detects path dependence, but at the requested 2100/2200/2300/2400 G points it is much smaller than the largest local error (for `+300 m/s`, at most `0.006234`). The largest local discrepancy occurs close to narrow spatial resonances rather than at those four preselected B values.
5. `DeltaB_memory=|v*dB/dz|*100 ns` reaches only `2.40 G` for `+300 m/s`; nevertheless the full accumulated trajectory can retain a larger difference than this one-timescale estimate suggests. The Step 6A switch-memory estimate is therefore useful context but does not replace continuous chronological propagation.
6. Because ground-state relaxation/collisions are absent, and because the fixed laser plus Doppler shift couples speed to spectral detuning, this benchmark alone must not yet be generalized to a thermal ensemble or used to claim that all of the observed difference is a single 100 ns magnetic lag.

## Files

- `trajectory_vs_local.png`
- `local_error_vs_position.png`
- `velocity_dependence.png`
- `trajectory_benchmark.json`
- `trajectory_benchmark.npz`
- `profiles/profile_*.npz`
