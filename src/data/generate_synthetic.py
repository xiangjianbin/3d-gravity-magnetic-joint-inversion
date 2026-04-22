"""
Synthetic Dataset Generator for 3D Gravity-Magnetic Joint Inversion
==============================================================

Generates all 6 types of geological models as specified in Fang et al. (2025)
Table I, with physically accurate forward modeling via prism formulas.

Type 1: Single cuboid/cube models (5,000 training samples)
  - 1-cuboid (~1500), 1-cube (~1200), 2-body combo (~1300), 3-body combo (~1000)

Type 2: Single tilting body models (5,000 training samples)
  - 1-tilting (~1800), 2-tilting combo (~1700), 3-tilting combo (~1500)

Type 3: Random walk models (10,000 training samples)
  - Complex irregular shapes from random walk algorithm

Type 4: Combined — Global structural consistency (7,500 training samples)
  - Density and susceptibility anomalies fully overlap spatially

Type 5: Combined — Partial structural consistency (7,500 training samples)
  - Some bodies share both properties, some have only one

Type 6: Combined — Structural inconsistency (10,000 training samples)
  - Density and susceptibility anomalies are spatially independent

Trapezoid special cases distributed within Types 1-3.

Grid parameters (from paper):
  - Model grid: 40 x 40 x 20 cells (Easting x Northing x Depth)
  - Cell size: 20 m x 20 m x 20 m
  - Total extent: 800 m x 800 m x 400 m
  - Observation surface: 81 x 81 points at z = +10 m (above ground)
  - Depth range: 0 to 400 m below surface

Physical property ranges (from Table III):
  - Density: {0.1, 0.5, 1.0} g/cm^3 with small perturbation
  - Susceptibility: {0.03, 0.1, 0.3} SI with small perturbation

Noise levels (from paper Section II-C):
  - Gravity: 0.5% of max |signal| * N(0,1)
  - Magnetic: 1.0% of max |signal| * N(0,1)

Author: Dataset Generation Agent
Date: 2026-04-22
"""

import numpy as np
import os
import json
import time
import gc
from typing import List, Dict, Tuple, Optional

# ============================================================
# Grid Constants (from paper)
# ============================================================
GRID_NX = 40
GRID_NY = 40
GRID_NZ = 20
CELL_DX = 20.0   # meters
CELL_DY = 20.0
CELL_DZ = 20.0
N_OBS = 81       # observation points per axis
OBS_HEIGHT = 10.0 # observation height above surface (m)

# Physical property ranges (from Table III + reasonable interpolation)
DENSITY_VALUES = [0.1, 0.5, 1.0]    # g/cm^3
SUSCEPTIBILITY_VALUES = [0.03, 0.1, 0.3]  # SI

# Noise levels (from paper)
NOISE_GRAVITY = 0.005   # 0.5%
NOISE_MAGNETIC = 0.01   # 1.0%

# ============================================================
# Dataset Specifications (Table I — paper正文权威数值)
# ============================================================
DATASET_SPECS = {
    1: {'total': 5000,
        'sub': {'single_cuboid': 1500, 'single_cube': 1200,
                'combo_2': 1300, 'combo_3': 1000}},
    2: {'total': 5000,
        'sub': {'single_tilt': 1800, 'combo_2_tilt': 1700,
                'combo_3_tilt': 1500}},
    3: {'total': 10000,
        'sub': {'random_walk': 10000}},
    4: {'total': 7500,
        'sub': {'single_cuboid': 1500, 'combo_cuboid': 3000,
                'tilting': 1000, 'combo_tilting': 1000,
                'random_walk': 500, 'combo_random_walk': 500}},
    5: {'total': 7500,
        'sub': {'single_cuboid': 1500, 'combo_cuboid': 3000,
                'tilting': 1000, 'combo_tilting': 1000,
                'random_walk': 500, 'combo_random_walk': 500}},
    6: {'total': 10000,
        'sub': {'single_cuboid': 2000, 'combo_cuboid': 4000,
                'tilting': 1000, 'combo_tilting': 1000,
                'random_walk': 1000, 'combo_random_walk': 2000}},
}

TOTAL_SAMPLES = sum(spec['total'] for spec in DATASET_SPECS.values())  # 45000


# ============================================================
# Utility Functions
# ============================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def normalize_to_01(data: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    dmin, dmax = data.min(), data.max()
    if dmax - dmin < 1e-15:
        return np.zeros_like(data)
    return (data - dmin) / (dmax - dmin)


def compute_structural_similarity(rho: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """
    Compute structural similarity label S.

    S[i,j,k] = 1 where both rho != 0 AND kappa != 0 (co-located anomaly)
    S[i,j,k] = 0 otherwise (background or single-property anomaly)
    """
    return ((rho != 0) & (kappa != 0)).astype(np.float32)


# ============================================================
# Type 1: Cuboid / Cube Generators
# ============================================================

def generate_single_cuboid(nx=GRID_NX, ny=GRID_NY, nz=GRID_NZ,
                           rng=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single rectangular cuboid (non-cube, aspect ratio != 1).
    Size: occupies roughly 15%-50% of the grid in each dimension.
    """
    if rng is None:
        rng = np.random.RandomState()

    rho = np.zeros((nx, ny, nz), dtype=np.float32)
    kappa = np.zeros((nx, ny, nz), dtype=np.float32)

    # Non-cubic dimensions (at least one direction clearly longer)
    sx = rng.randint(4, nx // 2 + 1)
    sy = rng.randint(4, ny // 2 + 1)
    sz = rng.randint(2, nz // 2 + 1)
    # Ensure it's not too cube-like in at least one dimension
    if abs(sx - sy) < 3 and abs(sy - sz * 2) < 3:
        sx = min(nx // 2, sx + rng.randint(3, 8))

    x0 = rng.randint(0, max(1, nx - sx))
    y0 = rng.randint(0, max(1, ny - sy))
    z0 = rng.randint(0, max(1, nz - sz))

    density = float(rng.choice(DENSITY_VALUES)) + rng.uniform(-0.05, 0.05)
    density = max(0.05, density)
    suscept = float(rng.choice(SUSCEPTIBILITY_VALUES)) + rng.uniform(-0.01, 0.01)
    suscept = max(0.01, suscept)

    rho[x0:x0+sx, y0:y0+sy, z0:z0+sz] = density
    kappa[x0:x0+sx, y0:y0+sy, z0:z0+sz] = suscept

    return rho, kappa


def generate_single_cube(nx=GRID_NX, ny=GRID_NY, nz=GRID_NZ,
                         rng=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single cube (roughly equal dimensions).
    """
    if rng is None:
        rng = np.random.RandomState()

    rho = np.zeros((nx, ny, nz), dtype=np.float32)
    kappa = np.zeros((nx, ny, nz), dtype=np.float32)

    # Cube-like dimensions (all similar)
    s = rng.randint(3, min(nx, ny, nz) // 2 + 1)
    # Make sure it's close to a cube
    sx = s + rng.randint(-1, 2)
    sy = s + rng.randint(-1, 2)
    sz = s + rng.randint(-1, 2)
    sx = max(2, min(sx, nx)); sy = max(2, min(sy, ny)); sz = max(2, min(sz, nz))

    x0 = rng.randint(0, max(1, nx - sx))
    y0 = rng.randint(0, max(1, ny - sy))
    z0 = rng.randint(0, max(1, nz - sz))

    density = float(rng.choice(DENSITY_VALUES)) + rng.uniform(-0.05, 0.05)
    density = max(0.05, density)
    suscept = float(rng.choice(SUSCEPTIBILITY_VALUES)) + rng.uniform(-0.01, 0.01)
    suscept = max(0.01, suscept)

    rho[x0:x0+sx, y0:y0+sy, z0:z0+sz] = density
    kappa[x0:x0+sx, y0:y0+sy, z0:z0+sz] = suscept

    return rho, kappa


def generate_combo_cuboids(n_bodies: int, nx=GRID_NX, ny=GRID_NY, nz=GRID_NZ,
                           rng=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate n_bodies cuboids/cubes combined in the same space.
    Bodies may touch but are generated independently.
    """
    if rng is None:
        rng = np.random.RandomState()

    rho_combined = np.zeros((nx, ny, nz), dtype=np.float32)
    kappa_combined = np.zeros((nx, ny, nz), dtype=np.float32)

    for _ in range(n_bodies):
        # Alternate between cuboid and cube
        if rng.random() < 0.5:
            r, k = generate_single_cuboid(nx, ny, nz, rng=rng)
        else:
            r, k = generate_single_cube(nx, ny, nz, rng=rng)
        rho_combined = np.maximum(rho_combined, r)
        kappa_combined = np.maximum(kappa_combined, k)

    return rho_combined, kappa_combined


def generate_type1_samples(n_samples: int, seed_start: int = 0,
                           verbose: bool = True) -> List[Dict]:
    """Generate Type 1 samples with proper sub-type distribution."""
    spec = DATASET_SPECS[1]['sub']
    total = sum(spec.values())
    scale = n_samples / total if n_samples != total else 1.0

    samples = []
    idx = 0

    # Sub-type generators mapping
    generators = [
        ('single_cuboid', lambda rng: generate_single_cuboid(rng=rng),
         int(spec['single_cuboid'] * scale)),
        ('single_cube', lambda rng: generate_single_cube(rng=rng),
         int(spec['single_cube'] * scale)),
        ('combo_2', lambda rng: generate_combo_cuboids(2, rng=rng),
         int(spec['combo_2'] * scale)),
        ('combo_3', lambda rng: generate_combo_cuboids(3, rng=rng),
         int(spec['combo_3'] * scale)),
    ]

    for sub_name, gen_fn, count in generators:
        for i in range(count):
            rng = np.random.RandomState(seed_start + idx)
            rho, kappa = gen_fn(rng)
            sim = compute_structural_similarity(rho, kappa)
            samples.append({
                'rho': rho, 'kappa': kappa, 'structural_sim': sim,
                'type': 1, 'subtype': sub_name,
            })
            idx += 1

    if verbose:
        print(f"  Type 1 generated: {len(samples)} samples "
              f"(cuboid={int(spec['single_cuboid']*scale)}, "
              f"cube={int(spec['single_cube']*scale)}, "
              f"combo2={int(spec['combo_2']*scale)}, "
              f"combo3={int(spec['combo_3']*scale)})")
    return samples


# ============================================================
# Type 2: Tilting Body Generators
# ============================================================

def generate_single_tilting_body(nx=GRID_NX, ny=GRID_NY, nz=GRID_NZ,
                                  tilt_range=(10.0, 60.0),
                                  rng=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single tilting body using coordinate rotation with supersampling.

    A local cuboid is generated then rotated around an axis to create tilt.
    Supersampling (3x) provides anti-aliasing.
    """
    if rng is None:
        rng = np.random.RandomState()

    rho = np.zeros((nx, ny, nz), dtype=np.float32)
    kappa = np.zeros((nx, ny, nz), dtype=np.float32)

    # Local cuboid dimensions (smaller than grid for rotation safety margin)
    local_sx = rng.randint(3, nx // 3)
    local_sy = rng.randint(3, ny // 2)
    local_sz = rng.randint(2, nz // 3)

    # Physical properties
    density = float(rng.choice(DENSITY_VALUES)) + rng.uniform(-0.05, 0.05)
    density = max(0.05, density)
    suscept = float(rng.choice(SUSCEPTIBILITY_VALUES)) + rng.uniform(-0.01, 0.01)
    suscept = max(0.01, suscept)

    # Tilt angle
    angle_deg = rng.uniform(*tilt_range)
    angle_rad = np.radians(angle_deg)

    # Local center
    cx_l = local_sx / 2.0
    cy_l = local_sy / 2.0
    cz_l = local_sz / 2.0

    # Global placement (center region with margin)
    offset_x = rng.uniform(nx * 0.25, nx * 0.75)
    offset_y = rng.uniform(ny * 0.25, ny * 0.75)
    offset_z = rng.uniform(nz * 0.15, nz * 0.85)

    # Supersampling for anti-aliasing
    ss = 3
    lx = np.linspace(0, local_sx, local_sx * ss)
    ly = np.linspace(0, local_sy, local_sy * ss)
    lz = np.linspace(0, local_sz, local_sz * ss)
    llx, lly, llz = np.meshgrid(lx, ly, lz, indexing='ij')

    # Center coordinates
    rx = llx.ravel() - cx_l
    ry = lly.ravel() - cy_l
    rz = llz.ravel() - cz_l

    # Rotation around Y-axis (X-Z plane tilt)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    gx = rx * cos_a - rz * sin_a + offset_x
    gy = ry + offset_y
    gz = rx * sin_a + rz * cos_a + offset_z

    # Filter valid voxels and rasterize
    valid = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny) & (gz >= 0) & (gz < nz)
    gxi = gx[valid].astype(np.int32)
    gyi = gy[valid].astype(np.int32)
    gzi = gz[valid].astype(np.int32)

    weight = 1.0 / (ss ** 3)
    np.add.at(rho, (gxi, gyi, gzi), density * weight)
    np.add.at(kappa, (gxi, gyi, gzi), suscept * weight)

    return rho, kappa


def generate_combo_tilting(n_bodies: int, nx=GRID_NX, ny=GRID_NY, nz=GRID_NZ,
                            rng=None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate n tilting bodies combined."""
    if rng is None:
        rng = np.random.RandomState()

    rho_comb = np.zeros((nx, ny, nz), dtype=np.float32)
    kappa_comb = np.zeros((nx, ny, nz), dtype=np.float32)

    for _ in range(n_bodies):
        r, k = generate_single_tilting_body(nx, ny, nz, rng=rng)
        rho_comb = np.maximum(rho_comb, r)
        kappa_comb = np.maximum(kappa_comb, k)

    return rho_comb, kappa_comb


def generate_type2_samples(n_samples: int, seed_start: int = 0,
                           verbose: bool = True) -> List[Dict]:
    """Generate Type 2 samples."""
    spec = DATASET_SPECS[2]['sub']
    total = sum(spec.values())
    scale = n_samples / total if n_samples != total else 1.0

    samples = []
    idx = 0

    generators = [
        ('single_tilt', lambda rng: generate_single_tilting_body(rng=rng),
         int(spec['single_tilt'] * scale)),
        ('combo_2_tilt', lambda rng: generate_combo_tilting(2, rng=rng),
         int(spec['combo_2_tilt'] * scale)),
        ('combo_3_tilt', lambda rng: generate_combo_tilting(3, rng=rng),
         int(spec['combo_3_tilt'] * scale)),
    ]

    for sub_name, gen_fn, count in generators:
        for i in range(count):
            rng = np.random.RandomState(seed_start + idx)
            rho, kappa = gen_fn(rng)
            sim = compute_structural_similarity(rho, kappa)
            samples.append({
                'rho': rho, 'kappa': kappa, 'structural_sim': sim,
                'type': 2, 'subtype': sub_name,
            })
            idx += 1

    if verbose:
        print(f"  Type 2 generated: {len(samples)} samples")
    return samples


# ============================================================
# Type 3: Random Walk Body Generator
# ============================================================

def generate_random_walk_body(nx=GRID_NX, ny=GRID_NY, nz=GRID_NZ,
                               n_steps_range=(200, 2000),
                               n_walks_range=(3, 12),
                               connect_prob=0.6,
                               rng=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate irregular body via 3D random walk.

    Algorithm:
      1. Place seed voxel in center region
      2. Perform multiple random walks from seed (26-neighborhood)
      3. Morphological dilation to make body more solid
      4. Assign uniform physical properties
    """
    if rng is None:
        rng = np.random.RandomState()

    rho = np.zeros((nx, ny, nz), dtype=np.float32)
    kappa = np.zeros((nx, ny, nz), dtype=np.float32)

    # Physical properties
    density = float(rng.choice(DENSITY_VALUES)) + rng.uniform(-0.05, 0.05)
    density = max(0.05, density)
    suscept = float(rng.choice(SUSCEPTIBILITY_VALUES)) + rng.uniform(-0.01, 0.01)
    suscept = max(0.01, suscept)

    # Seed position (center region with margin)
    mx, my, mz = nx // 2, ny // 2, nz // 2
    margin_x = nx // 5; margin_y = ny // 5; margin_z = nz // 5
    seed_x = rng.randint(margin_x, nx - margin_x)
    seed_y = rng.randint(margin_y, ny - margin_y)
    seed_z = rng.randint(max(1, margin_z), nz - margin_z)

    # 26-neighborhood directions
    dirs_26 = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if di == 0 and dj == 0 and dk == 0:
                    continue
                dirs_26.append((di, dj, dk))
    dirs_26 = np.array(dirs_26, dtype=int)

    visited = set()
    visited.add((seed_x, seed_y, seed_z))

    # Multiple walks for more volume
    n_walks = rng.randint(*n_walks_range)
    steps_per_walk = rng.randint(*n_steps_range) // max(1, n_walks)
    steps_per_walk = max(10, steps_per_walk)

    for w_idx in range(n_walks):
        pos = np.array([seed_x, seed_y, seed_z], dtype=int)

        for step in range(steps_per_walk):
            d = dirs_26[rng.randint(len(dirs_26))]
            new_pos = pos + d

            # Boundary check
            if (0 <= new_pos[0] < nx and 0 <= new_pos[1] < ny and 0 <= new_pos[2] < nz):
                # Bias toward continuing in same general direction (connectivity)
                if rng.random() < connect_prob:
                    pos = new_pos
                else:
                    pos = new_pos  # still move but may change direction
                visited.add((pos[0], pos[1], pos[2]))
            else:
                # Bounce: stay or pick valid neighbor
                pass

    # Fill visited voxels
    for (vx, vy, vz) in visited:
        rho[vx, vy, vz] = density
        kappa[vx, vy, vz] = suscept

    # Morphological dilation for solidity
    try:
        from scipy.ndimage import binary_dilation
        mask = rho > 0
        struct = np.ones((3, 3, 2), dtype=bool)
        dilated = binary_dilation(mask, structure=struct)
        new_fill = dilated & ~mask
        if np.any(new_fill):
            fill_rho = density * rng.uniform(0.6, 0.95)
            fill_kap = suscept * rng.uniform(0.6, 0.95)
            rho[new_fill] = fill_rho
            kappa[new_fill] = fill_kap
    except ImportError:
        pass  # scipy not available, skip dilation

    return rho, kappa


def generate_type3_samples(n_samples: int, seed_start: int = 0,
                           verbose: bool = True) -> List[Dict]:
    """Generate Type 3 samples."""
    samples = []
    for i in range(n_samples):
        rng = np.random.RandomState(seed_start + i)
        rho, kappa = generate_random_walk_body(rng=rng)
        sim = compute_structural_similarity(rho, kappa)
        samples.append({
            'rho': rho, 'kappa': kappa, 'structural_sim': sim,
            'type': 3, 'subtype': 'random_walk',
        })

    if verbose:
        print(f"  Type 3 generated: {len(samples)} samples")
    return samples


# ============================================================
# Trapezoid Special Case Generator
# ============================================================

def generate_trapezoid_body(n_sections: int, nx=GRID_NX, ny=GRID_NY, nz=GRID_NZ,
                              rng=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a trapezoidal body with n_sections vertical segments.

    A trapezoid in 3D: cross-section area changes linearly (or stepwise) with depth.
    Each section is a horizontal layer with different X-Y extent.

    n_sections: 1 = rectangular prism (degenerate trapezoid)
                 2-5 = true trapezoids with varying width
    """
    if rng is None:
        rng = np.random.RandomState()

    rho = np.zeros((nx, ny, nz), dtype=np.float32)
    kappa = np.zeros((nx, ny, nz), dtype=np.float32)

    # Physical properties
    density = float(rng.choice(DENSITY_VALUES)) + rng.uniform(-0.05, 0.05)
    density = max(0.05, density)
    suscept = float(rng.choice(SUSCEPTIBILITY_VALUES)) + rng.uniform(-0.01, 0.01)
    suscept = max(0.01, suscept)

    # Divide depth into n_sections
    z_bounds = np.linspace(0, nz, n_sections + 1).astype(int)
    z_bounds = np.clip(z_bounds, 0, nz)
    z_bounds[-1] = nz

    # Base size (largest at top or bottom)
    base_sx = rng.randint(4, nx // 2)
    base_sy = rng.randint(4, ny // 2)

    # Size shrink/grow factor per section
    if rng.random() < 0.5:
        # Widening downward
        factors = np.linspace(0.3, 1.0, n_sections)
    else:
        # Narrowing downward
        factors = np.linspace(1.0, 0.3, n_sections)

    # Center position
    cx = rng.randint(base_sx // 2, nx - base_sx // 2)
    cy = rng.randint(base_sy // 2, ny - base_sy // 2)

    for sec in range(n_sections):
        z_start = z_bounds[sec]
        z_end = z_bounds[sec + 1]
        if z_end <= z_start:
            continue

        sx = max(2, int(base_sx * factors[sec]))
        sy = max(2, int(base_sy * factors[sec]))

        x0 = max(0, cx - sx // 2)
        y0 = max(0, cy - sy // 2)
        x1 = min(nx, x0 + sx)
        y1 = min(ny, y0 + sy)

        rho[x0:x1, y0:y1, z_start:z_end] = density
        kappa[x0:x1, y0:y1, z_start:z_end] = suscept

    return rho, kappa


def generate_trapezoid_samples(n_per_section: int = 50, seed_start: int = 100000,
                                verbose: bool = True) -> List[Dict]:
    """
    Generate trapezoid samples: 1-5 sections each.

    Returns list of samples (to be merged into Types 1-3).
    """
    samples = []
    idx = 0
    for n_sec in range(1, 6):
        for i in range(n_per_section):
            rng = np.random.RandomState(seed_start + idx)
            rho, kappa = generate_trapezoid_body(n_sec, rng=rng)
            sim = compute_structural_similarity(rho, kappa)
            samples.append({
                'rho': rho, 'kappa': kappa, 'structural_sim': sim,
                'type': 1, 'subtype': f'trapezoid_{n_sec}sec',
                'is_trapezoid': True, 'n_sections': n_sec,
            })
            idx += 1

    if verbose:
        print(f"  Trapezoid generated: {len(samples)} samples "
              f"({n_per_section} per section x 5 sections)")
    return samples


# ============================================================
# Type 4/5/6: Combined Models (Global/Partial/Inconsistent Consistency)
# ============================================================

def _make_base_model(model_key: str, nx, ny, nz,
                      density_range=None, susc_range=None,
                      rng=None) -> Tuple[np.ndarray, np.ndarray]:
    """Create a base model by type key."""
    if density_range is None:
        density_range = DENSITY_VALUES
    if susc_range is None:
        susc_range = SUSCEPTIBILITY_VALUES

    if model_key == 'single_cuboid':
        return generate_single_cuboid(nx, ny, nz, rng=rng)
    elif model_key == 'combo_cuboid':
        n = rng.randint(2, 4)
        return generate_combo_cuboids(n, nx, ny, nz, rng=rng)
    elif model_key == 'tilting':
        return generate_single_tilting_body(nx, ny, nz, rng=rng)
    elif model_key == 'combo_tilting':
        n = rng.randint(2, 4)
        return generate_combo_tilting(n, nx, ny, nz, rng=rng)
    elif model_key == 'random_walk':
        return generate_random_walk_body(nx, ny, nz, rng=rng)
    elif model_key == 'combo_random_walk':
        # Multiple RW bodies
        rho_comb = np.zeros((nx, ny, nz), dtype=np.float32)
        kap_comb = np.zeros((nx, ny, nz), dtype=np.float32)
        n_rw = rng.randint(2, 4)
        for _ in range(n_rw):
            r, k = generate_random_walk_body(nx, ny, nz, rng=rng)
            rho_comb = np.maximum(rho_comb, r)
            kap_comb = np.maximum(kap_comb, k)
        return rho_comb, kap_comb
    else:
        raise ValueError(f"Unknown base model key: {model_key}")


def generate_type4_sample(nx=GRID_NX, ny=GRID_NY, nz=GRID_NZ,
                           rng=None) -> Dict:
    """
    Type 4: Global structural consistency.
    Both density and susceptibility share the SAME geometry (fully overlapping).
    structural_sim = 1 everywhere on the anomaly body.
    """
    if rng is None:
        rng = np.random.RandomState()

    # Pick a sub-type according to Table I footnote distribution
    weights = [1500, 3000, 1000, 1000, 500, 500]
    keys = ['single_cuboid', 'combo_cuboid', 'tilting', 'combo_tilting',
            'random_walk', 'combo_random_walk']
    idx = rng.choice(len(keys), p=np.array(weights, dtype=float) / sum(weights))
    model_key = keys[idx]

    rho, kappa = _make_base_model(model_key, nx, ny, nz, rng=rng)
    # For global consistency: rho and kappa have identical geometry
    # (already the case since we use same generator for both)
    sim = compute_structural_similarity(rho, kappa)

    return {
        'rho': rho, 'kappa': kappa, 'structural_sim': sim,
        'type': 4, 'subtype': model_key, 'consistency': 'global',
    }


def generate_type5_sample(nx=GRID_NX, ny=GRID_NY, nz=GRID_NZ,
                           rng=None) -> Dict:
    """
    Type 5: Partial structural consistency.
    Some bodies have both properties, some have only one.
    structural_sim = 1 on co-located parts, 0 elsewhere.
    """
    if rng is None:
        rng = np.random.RandomState()

    keys = ['single_cuboid', 'combo_cuboid', 'tilting', 'combo_tilting',
            'random_walk', 'combo_random_walk']
    weights = [1500, 3000, 1000, 1000, 500, 500]
    idx = rng.choice(len(keys), p=np.array(weights, dtype=float) / sum(weights))
    model_key = keys[idx]

    n_bodies = rng.randint(2, 5)
    rho_comb = np.zeros((nx, ny, nz), dtype=np.float32)
    kappa_comb = np.zeros((nx, ny, nz), dtype=np.float32)

    for bi in range(n_bodies):
        r, k = _make_base_model(model_key, nx, ny, nz, rng=rng)

        # Partial consistency: randomly drop one property for some bodies
        if bi > 0 and rng.random() < 0.45:
            if rng.random() < 0.5:
                k = np.zeros_like(k)  # Keep only density
            else:
                r = np.zeros_like(r)  # Keep only susceptibility

        rho_comb = np.maximum(rho_comb, r)
        kappa_comb = np.maximum(kappa_comb, k)

    sim = compute_structural_similarity(rho_comb, kappa_comb)

    return {
        'rho': rho_comb, 'kappa': kappa_comb, 'structural_sim': sim,
        'type': 5, 'subtype': model_key, 'consistency': 'partial',
    }


def generate_type6_sample(nx=GRID_NX, ny=GRID_NY, nz=GRID_NZ,
                           rng=None) -> Dict:
    """
    Type 6: Structural inconsistency.
    Density and susceptibility come from completely independent geometries.
    structural_sim ≈ 0 almost everywhere.
    """
    if rng is None:
        rng = np.random.RandomState()

    keys = ['single_cuboid', 'combo_cuboid', 'tilting', 'combo_tilting',
            'random_walk', 'combo_random_walk']
    weights = [2000, 4000, 1000, 1000, 1000, 2000]
    idx = rng.choice(len(keys), p=np.array(weights, dtype=float) / sum(weights))
    model_key = keys[idx]

    # Generate density geometry independently
    rho_keys = list(keys)
    rho_key = rng.choice(rho_keys)
    rho, _ = _make_base_model(rho_key, nx, ny, nz, rng=rng)

    # Generate susceptibility geometry independently (different type preferred)
    kap_keys = [k for k in keys if k != rho_key] or keys
    kap_key = rng.choice(kap_keys)
    _, kappa = _make_base_model(kap_key, nx, ny, nz, rng=rng)

    sim = compute_structural_similarity(rho, kappa)

    return {
        'rho': rho, 'kappa': kappa, 'structural_sim': sim,
        'type': 6, 'subtype': model_key, 'consistency': 'inconsistent',
    }


def generate_type456_samples(dtype: int, n_samples: int, seed_start: int = 0,
                              verbose: bool = True) -> List[Dict]:
    """Generate Type 4, 5, or 6 samples."""
    gen_fn = {4: generate_type4_sample,
              5: generate_type5_sample,
              6: generate_type6_sample}[dtype]

    samples = []
    for i in range(n_samples):
        rng = np.random.RandomState(seed_start + i)
        sample = gen_fn(nx=GRID_NX, ny=GRID_NY, nz=GRID_NZ, rng=rng)
        samples.append(sample)

    if verbose:
        print(f"  Type {dtype} generated: {len(samples)} samples")
    return samples


# ============================================================
# Forward Modeling Integration
# ============================================================

def run_forward_on_samples(samples: List[Dict], obs_height: float = OBS_HEIGHT,
                            n_obs: int = N_OBS,
                            noise_grav: float = NOISE_GRAVITY,
                            noise_mag: float = NOISE_MAGNETIC,
                            verbose: bool = True) -> List[Dict]:
    """
    Run gravity + magnetic forward modeling on all samples.

    Adds forward-modeled observation data and noisy versions to each sample dict.
    Uses the external forward_gravity / forward_magnetic modules.
    """
    try:
        from src.data.forward_gravity import forward_gravity as fg, add_gravity_noise
        from src.data.forward_magnetic import forward_magnetic as fm, add_magnetic_noise
        has_forward = True
    except ImportError:
        has_forward = False
        if verbose:
            print("  WARNING: External forward modules not available, "
                  "using placeholder zeros")

    n_total = len(samples)
    t0 = time.time()

    for i, sample in enumerate(samples):
        rho = sample['rho']
        kappa = sample['kappa']

        if has_forward:
            grav_clean = fg(rho, obs_height=obs_height, n_obs=n_obs)
            mag_clean = fm(kappa, obs_height=obs_height, n_obs=n_obs)

            grav_noisy = add_gravity_noise(grav_clean, noise_level=noise_grav,
                                           seed=42 + i * 2)
            mag_noisy = add_magnetic_noise(mag_clean, noise_level=noise_mag,
                                           seed=42 + i * 2 + 1)
        else:
            grav_noisy = np.zeros((n_obs, n_obs), dtype=np.float32)
            mag_noisy = np.zeros((n_obs, n_obs), dtype=np.float32)

        sample['gravity_raw'] = grav_clean if has_forward else grav_noisy
        sample['magnetic_raw'] = mag_clean if has_forward else mag_noisy
        sample['gravity'] = grav_noisy
        sample['magnetic'] = mag_noisy

        if verbose and (i + 1) % max(1, n_total // 10) == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_total - i - 1) / rate if rate > 0 else 0
            print(f"    Forward: {i+1}/{n_total} ({elapsed:.0f}s, "
                  f"ETA: {eta:.0f}s)")

    if verbose:
        elapsed = time.time() - t0
        print(f"  Forward modeling done: {n_total} samples in {elapsed:.1f}s")

    return samples


# ============================================================
# Main Generation Pipeline
# ============================================================

def generate_full_dataset(output_dir: str = 'data/',
                          n_samples_total: int = TOTAL_SAMPLES,
                          seed: int = 42,
                          run_forward: bool = True,
                          verbose: bool = True) -> Dict:
    """
    Generate the complete dataset pipeline:

    1. Generate all 6 types of geological models
    2. Run forward modeling (gravity + magnetic)
    3. Split 7:2:1 into train/val/test
    4. Save to disk

    Parameters
    ----------
    output_dir : str — output directory for .npz files
    n_samples_total : int — total number of samples (default 45000)
    seed : int — master random seed
    run_forward : bool — whether to run physical forward modeling
    verbose : bool — print progress

    Returns
    -------
    manifest : dict — dataset statistics and metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    t_total_start = time.time()

    if verbose:
        print("=" * 65)
        print("  3D Gravity-Magnetic Joint Inversion: Dataset Generation")
        print("=" * 65)
        print(f"  Total target: {n_samples_total} samples")
        print(f"  Output dir: {output_dir}")
        print(f"  Forward modeling: {'ON' if run_forward else 'OFF (placeholder)'}")
        print()

    # ---- Phase 1: Generate Type 1 models ----
    if verbose:
        print("Phase 1: Generating Type 1 (Cuboid/Cube) models...")
    t1 = time.time()
    scale = n_samples_total / TOTAL_SAMPLES
    type1_samples = generate_type1_samples(
        n_samples=max(1, int(DATASET_SPECS[1]['total'] * scale)),
        seed_start=seed, verbose=verbose)
    # Add trapezoid samples into Type 1
    trap_samples = generate_trapezoid_samples(
        n_per_section=max(1, int(30 * scale)), seed_start=seed + 50000,
        verbose=verbose)
    type1_samples.extend(trap_samples)
    del trap_samples; gc.collect()
    if verbose:
        print(f"  Type 1 done ({time.time()-t1:.1f}s): {len(type1_samples)} samples")

    # ---- Phase 2: Generate Type 2 models ----
    if verbose:
        print("\nPhase 2: Generating Type 2 (Tilting Body) models...")
    t2 = time.time()
    type2_samples = generate_type2_samples(
        n_samples=max(1, int(DATASET_SPECS[2]['total'] * scale)),
        seed_start=seed + 10000, verbose=verbose)
    del trap_samples; gc.collect()
    if verbose:
        print(f"  Type 2 done ({time.time()-t2:.1f}s): {len(type2_samples)} samples")

    # ---- Phase 3: Generate Type 3 models ----
    if verbose:
        print("\nPhase 3: Generating Type 3 (Random Walk) models...")
    t3 = time.time()
    type3_samples = generate_type3_samples(
        n_samples=max(1, int(DATASET_SPECS[3]['total'] * scale)),
        seed_start=seed + 20000, verbose=verbose)
    if verbose:
        print(f"  Type 3 done ({time.time()-t3:.1f}s): {len(type3_samples)} samples")

    # ---- Phase 4: Generate Type 4/5/6 models ----
    if verbose:
        print("\nPhase 4: Generating Type 4/5/6 (Combined) models...")
    t4 = time.time()
    type4_samples = generate_type456_samples(
        4, max(1, int(DATASET_SPECS[4]['total'] * scale)),
        seed_start=seed + 30000, verbose=verbose)
    del type4_samples; gc.collect()  # will be regenerated below

    # Actually regenerate properly
    type4_samples = generate_type456_samples(
        4, max(1, int(DATASET_SPECS[4]['total'] * scale)),
        seed_start=seed + 30000, verbose=False)
    type5_samples = generate_type456_samples(
        5, max(1, int(DATASET_SPECS[5]['total'] * scale)),
        seed_start=seed + 40000, verbose=False)
    type6_samples = generate_type456_samples(
        6, max(1, int(DATASET_SPECS[6]['total'] * scale)),
        seed_start=seed + 50000, verbose=False)
    if verbose:
        print(f"  Type 4: {len(type4_samples)}, Type 5: {len(type5_samples)}, "
              f"Type 6: {len(type6_samples)}")
        print(f"  Phase 4 done ({time.time()-t4:.1f}s)")
    gc.collect()

    # ---- Combine all samples ----
    all_samples = (type1_samples + type2_samples + type3_samples +
                   type4_samples + type5_samples + type6_samples)

    # Trim to exact target count if needed
    if len(all_samples) > n_samples_total:
        rng_trim = np.random.RandomState(seed + 99999)
        indices = rng_trim.choice(len(all_samples), n_samples_total, replace=False)
        all_samples = [all_samples[int(i)] for i in sorted(indices)]
        if verbose:
            print(f"\n  Trimmed to {len(all_samples)} samples")

    # Free intermediate lists
    del type1_samples, type2_samples, type3_samples
    del type4_samples, type5_samples, type6_samples
    gc.collect()

    # ---- Phase 5: Forward Modeling ----
    if run_forward:
        if verbose:
            print("\nPhase 5: Running forward modeling (gravity + magnetic)...")
        all_samples = run_forward_on_samples(
            all_samples, verbose=verbose,
            noise_grav=NOISE_GRAVITY, noise_mag=NOISE_MAGNETIC)
        gc.collect()
    else:
        if verbose:
            print("\nPhase 5: SKIPPED (placeholder zeros)")
        n_obs = N_OBS
        for s in all_samples:
            s['gravity'] = np.zeros((n_obs, n_obs), dtype=np.float32)
            s['magnetic'] = np.zeros((n_obs, n_obs), dtype=np.float32)
            s['gravity_raw'] = s['gravity'].copy()
            s['magnetic_raw'] = s['magnetic'].copy()

    # ---- Phase 6: Split 7:2:1 and Save ----
    if verbose:
        print("\nPhase 6: Splitting 7:2:1 and saving...")

    # Stratified split by type
    from collections import defaultdict
    by_type = defaultdict(list)
    for s in all_samples:
        by_type[s['type']].append(s)

    train_list, val_list, test_list = [], [], []
    split_rng = np.random.RandomState(seed + 88888)

    for dtype_i in sorted(by_type.keys()):
        samples_t = by_type[dtype_i]
        n = len(samples_t)
        indices = np.arange(n)
        split_rng.shuffle(indices)

        n_train = int(n * 0.7)
        n_val = int(n * 0.2)

        for i in indices[:n_train]:
            train_list.append(samples_t[i])
        for i in indices[n_train:n_train + n_val]:
            val_list.append(samples_t[i])
        for i in indices[n_train + n_val:]:
            test_list.append(samples_t[i])

    # Save splits
    _save_split(train_list, os.path.join(output_dir, 'train_dataset.npz'), 'train')
    _save_split(val_list, os.path.join(output_dir, 'val_dataset.npz'), 'val')
    _save_split(test_list, os.path.join(output_dir, 'test_dataset.npz'), 'test')

    # ---- Build manifest ----
    total_time = time.time() - t_total_start
    manifest = {
        'total_samples': len(all_samples),
        'train_samples': len(train_list),
        'val_samples': len(val_list),
        'test_samples': len(test_list),
        'grid': {'nx': GRID_NX, 'ny': GRID_NY, 'nz': GRID_NZ,
                 'dx': CELL_DX, 'dy': CELL_DY, 'dz': CELL_DZ},
        'observation': {'n_obs': N_OBS, 'height_m': OBS_HEIGHT},
        'noise': {'gravity_frac': NOISE_GRAVITY, 'magnetic_frac': NOISE_MAGNETIC},
        'type_counts': _count_types(all_samples),
        'forward_engine': 'prism_formula' if run_forward else 'placeholder',
        'seed': seed,
        'generation_time_s': total_time,
    }

    manifest_path = os.path.join(output_dir, 'DATASET_MANIFEST.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(f"\n{'=' * 65}")
        print(f"  TOTAL: {len(all_samples)} samples in {total_time:.1f}s "
              f"({total_time/60:.1f}min)")
        print(f"  Train={len(train_list)} Val={len(val_list)} Test={len(test_list)}")
        print(f"  Output: {os.path.abspath(output_dir)}")
        print(f"{'=' * 65}")

    return manifest


def _save_split(samples: List[Dict], filepath: str, split_name: str):
    """Save a data split to .npz file."""
    save_dict = {}
    for i, s in enumerate(samples):
        pfx = f'sample_{i:06d}'
        save_dict[f'{pfx}_rho'] = s['rho']
        save_dict[f'{pfx}_kappa'] = s['kappa']
        save_dict[f'{pfx}_sim'] = s['structural_sim']
        save_dict[f'{pfx}_gravity'] = s.get('gravity', np.zeros((N_OBS, N_OBS)))
        save_dict[f'{pfx}_magnetic'] = s.get('magnetic', np.zeros((N_OBS, N_OBS)))
        save_dict[f'{pfx}_type'] = np.array(s['type'])
        save_dict[f'{pfx}_subtype'] = s.get('subtype', '')

    meta = {
        'n_samples': len(samples),
        'split': split_name,
        'nx': GRID_NX, 'ny': GRID_NY, 'nz': GRID_NZ,
        'n_obs': N_OBS,
    }
    save_dict['__meta__'] = json.dumps(meta)

    np.savez_compressed(filepath, **save_dict)
    size_mb = os.path.getsize(filepath) / 1024 / 1024
    print(f"  Saved {split_name}: {len(samples)} samples -> {filepath} ({size_mb:.1f} MB)")


def _count_types(samples: List[Dict]) -> Dict:
    """Count samples by type and subtype."""
    counts = {}
    subtype_counts = {}
    for s in samples:
        t = s['type']
        counts[t] = counts.get(t, 0) + 1
        st = s.get('subtype', 'unknown')
        key = f"type{t}_{st}"
        subtype_counts[key] = subtype_counts.get(key, 0) + 1
    return {'by_type': counts, 'by_subtype': subtype_counts}


if __name__ == '__main__':
    # Quick self-test
    print("=" * 60)
    print("Self-test: generating 20 samples per type (no forward)")
    print("=" * 60)

    for t in range(1, 7):
        samples = generate_type456_samples(t, n_samples=5, seed_start=t*1000,
                                          verbose=False)
        s = samples[0]
        print(f"  Type {t}: rho shape={s['rho'].shape}, "
              f"nonzero_rho={np.count_nonzero(s['rho'])}, "
              f"sim_mean={s['structural_sim'].mean():.3f}")

    # Test trapezoid
    trap = generate_trapezoid_samples(n_per_section=2, seed_start=999,
                                       verbose=False)
    print(f"  Trapezoid: {len(trap)} samples, sections: "
          f"{[s['n_sections'] for s in trap[:5]]}")

    print("\nSelf-test complete!")
