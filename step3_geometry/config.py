# -*- coding: utf-8 -*-
"""第三步统一几何与热运动参数，全部使用 SI 单位。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometryConfig:
    cell_length_m: float = 25.0e-3
    beam_radius_m: float = 0.5e-3
    n_slices: int = 100
    atoms_per_slice: int = 50
    temperature_C: float = 70.0
    seed: int = 20260807

    @property
    def dz_m(self):
        return self.cell_length_m / self.n_slices

    @property
    def n_atoms(self):
        return self.n_slices * self.atoms_per_slice


CONFIG = GeometryConfig()

# SI constants
K_B = 1.380649e-23
M_K39 = 38.96370668 * 1.66053906660e-27

