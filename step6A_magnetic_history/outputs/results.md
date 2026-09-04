# Step 6A results

## Status

PASS: deterministic full-256-dimensional magnetic-history benchmark completed.

## Parameters

- Atom/line: K39 D1, 16-state Hilbert space, 256-dimensional Liouville space.
- Entry state: ground manifold `I_g/8`.
- Polarization: pure `Delta-m=+1` along `+z` under the frozen SI phasor convention.
- Velocity: `v_z=0`.
- Light: top-hat constant light, `s=1.0e-06`.
- Temperature used only for SI number density/P: `20.0 C`, natural K39 fraction.
- Pre-history time: `1 us`.
- Post times: `0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000 ns`.
- Forward history/reference: `0 G -> 300 G` versus `300 G -> 300 G`.
- Reverse history/reference: `300 G -> 0 G` versus `0 G -> 0 G`.
- Forward external/internal detuning: `+352.811197635` / `+368.675197635 MHz`.
- Reverse external/internal detuning: `-168.177645075` / `-152.313645075 MHz`.
- Propagation: direct `scipy.linalg.expm(L*t)` on the full 256x256 L; no Step 5.5B reduced space.

## Controls

| Metric | Forward | Reverse |
|---|---:|---:|
| identical-history error | `0.000000000e+00` | `0.000000000e+00` |
| uniform-L two-stage vs one-stage | `2.960672299e-13` | `3.904892710e-14` |
| segment-splitting error | `3.110701096e-13` | `2.693107478e-14` |
| 0.5 us vs 1 us P plateau relative change | `8.668687137e-05` | `8.976359012e-05` |
| 1 us P vs fixed-thermal linear response | `4.293776478e-06` | `2.412869192e-06` |
| max ground population redistribution | `5.235902136e-07` | `4.034669580e-07` |
| ground-change/history ratio | `6.099667099e-03` | `7.469037841e-03` |
| normalized `||[L_past,L_current]||` | `4.964636918e-02` | `7.561313588e-02` |

The nonzero commutator is diagnostic only. All history conclusions use chronological propagation, never an averaged B or effective L.

## Memory magnitude and timescale

| Metric | `0 -> 300 G` | `300 -> 0 G` |
|---|---:|---:|
| max `||Delta rho||_F` | `1.019322323e-04` | `1.140953199e-04` |
| max `|Delta P_q|` (C/m^2) | `8.741853213e-19` | `1.099860888e-18` |
| sampled persistent P-memory below 1/e | `100 ns` | `100 ns` |
| sampled persistent P-memory below 10% | `200 ns` | `200 ns` |
| sampled persistent rho-memory below 1/e | `100 ns` | not used as primary reverse conclusion |
| sampled persistent rho-memory below 10% | `200 ns` | not used as primary reverse conclusion |
| largest t=0 rho block | `eg` | `eg` |
| `M_P(1 us)` | `4.166586052e-06` | `2.521222472e-06` |
| `M_rho(1 us)` | `6.770665719e-03` | `7.941844883e-03` |
| diagonal fraction of the 1 us `gg` norm | `92.544871%` | `99.999179%` |

These are threshold crossings on the requested discrete time grid, not fitted exponential constants. Oscillations or multiple decay scales are retained; no forced single-exponential fit is made.

## Density-matrix diagnostics

| Metric | Forward | Reverse |
|---|---:|---:|
| maximum trace error | `1.022114476e-12` | `1.625003611e-13` |
| maximum Hermiticity error | `3.507186558e-13` | `2.935152154e-14` |
| minimum eigenvalue | `-6.433670793e-17` | `-6.700076118e-17` |

## Physical conclusion

1. Magnetic history exists immediately after the field switch: the atoms have the same current B, light, detuning and total illumination time but different `rho` and P. At `T_post=0`, the dominant blocks are the Hermitian-conjugate optical blocks `ge/eg` (their plotted curves overlap).
2. The requested grid brackets the persistent P-memory 1/e crossing between `50 and 100 ns`, and the 10% crossing between `100 and 200 ns`, in both field directions. The sampled curves are not forced into a single-exponential fit.
3. By `1 us`, `M_P` is only `4.167e-06` forward and `2.521e-06` reverse. Thus the current optical response has recovered to the local-field reference on a timescale much shorter than a typical microsecond transit.
4. Total `rho` does not reach numerical zero: it plateaus near `0.677%` forward and `0.794%` reverse. The surviving block is `gg`; its 1 us norm is `92.545%` / `99.999%` diagonal population, so the long internal-state tail is mainly weak residual optical-pumping population redistribution, not long-lived ground coherence.
5. Ground-population redistribution during the 1 us pre-stage is still small: below 1% of the initial full-history difference. It is negligible for the initial optical-memory signal but explains the much smaller long-time `rho` floor in this no-ground-relaxation model.
6. Under weak light and longitudinal B, the local-response approximation is therefore good on microsecond timescales for P, even though the complete density matrix retains a tiny population history. This conclusion must not be generalized to transverse/nonuniform B, stronger light, collisions or added ground coherence without new tests.

## Files

- `forward_0G_to_300G.npz`
- `reverse_300G_to_0G.npz`
- `magnetic_memory_decay.png`
- `rho_block_memory.png`
- `current_field_response_recovery.png`
