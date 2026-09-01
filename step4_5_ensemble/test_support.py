# -*- coding: utf-8 -*-
"""Step 4.5 测试共用的小型系综和切片场。"""

import numpy as np

import path_setup  # noqa: F401

from config import GeometryConfig
from geometry import generate_ensemble
from moving_atom import SliceFields


def make_config(n_slices=4, atoms_per_slice=1, seed=45001):
    return GeometryConfig(
        cell_length_m=25e-3,
        beam_radius_m=0.5e-3,
        n_slices=n_slices,
        atoms_per_slice=atoms_per_slice,
        temperature_C=70.0,
        seed=seed,
    )


def make_ensemble(n_slices=4, atoms_per_slice=1, seed=45001):
    config = make_config(n_slices, atoms_per_slice, seed)
    trajectories, edges, sigma = generate_ensemble(config)
    return config, trajectories, edges, sigma


def nonuniform_fields(config, saturation=0.03, detuning_MHz=40.0):
    return SliceFields(
        B_z_G=np.linspace(100.0, 500.0, config.n_slices),
        saturation=np.full(config.n_slices, saturation),
        polarization_cart=np.tile(
            np.array([1.0, 0.0, 0.0], dtype=complex),
            (config.n_slices, 1),
        ),
        laser_detuning_MHz=np.full(config.n_slices, detuning_MHz),
    )

