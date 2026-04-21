# Dataset Generation Report

## Overview

This document describes the synthetic training dataset generated for reproducing
Fang et al. (2025) "Improved 3-D Joint Inversion of Gravity and Magnetic Data
Based on Deep Learning With a Multitask Learning Strategy", IEEE TGRS, Vol. 63.

## Dataset Specifications

### Grid Configuration

| Parameter | Value |
|-----------|-------|
| Model grid (nx, ny, nz) | 40 x 40 x 20 = 32,000 cells |
| Cell size (dx, dy, dz) | 20 m x 20 m x 20 m |
| Physical extent (X, Y, Z) | 800 m x 800 m x 400 m |
| Observation grid | 81 x 81 points |
| Observation height | 10 m above surface |
| Coordinate origin | Center of horizontal plane, surface z=0 |

### Physical Property Ranges

| Property | Values | Unit |
|----------|--------|------|
| Density | {0.1, 0.5, 1.0} | g/cm^3 (with +/-10% random variation) |
| Susceptibility | {0.03, 0.1, 0.3} | SI (with +/-10% random variation) |

### Geomagnetic Field Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Total field intensity (F) | 55,000 | nT |
| Declination | -7.0 | degrees (positive east) |
| Inclination | 60.0 | degrees (positive down) |

*Note: These values correspond to the Inner Mongolia study region from the paper.*

### Noise Model

| Data type | Noise level | Formula |
|-----------|-------------|---------|
| Gravity anomaly | 0.5% of max \|signal\| | d_obs = d_syn + 0.005 * max(\|d_syn\|) * N(0,1) |
| Magnetic anomaly | 1.0% of max \|signal\| | d_obs = d_syn + 0.01 * max(\|d_syn\|) * N(0,1) |

### Data Split

| Split | Count | Ratio |
|-------|-------|-------|
| Training | 31,500 | 70% |
| Validation | 9,000 | 20% |
| Test | 4,500 | 10% |
| **Total** | **45,000** | **100%** |

## Geological Model Types

### Type 1: Cuboid / Cube Models (5,000 samples)

Simple rectangular prisms representing basic geological bodies (intrusions,
ore bodies, basement blocks).

| Sub-type | Count | Description |
|----------|-------|-------------|
| single_cuboid | ~1,500 | Single rectangular prism with arbitrary aspect ratio |
| single_cube | ~1,200 | Single prism with equal X-Y dimensions |
| combined_2 | ~1,300 | Two overlapping/separate prisms in same subspace |
| combined_3 | ~1,000 | Three prisms in same subspace |

**Generation parameters:**
- Size: 2-12 cells per dimension (random)
- Position: random within grid bounds
- Density/susceptibility: sampled from paper's value ranges
- Structural similarity: 1.0 (fully consistent by definition)

### Type 2: Tilting Body Models (5,000 samples)

Dipping/inclined geological bodies that shift horizontally with depth,
simulating fault-controlled or stratiform deposits.

| Sub-type | Count | Description |
|----------|-------|-------------|
| single_tilted | ~1,800 | Single tilted body |
| combined_tilted_2 | ~1,700 | Two tilted bodies |
| combined_tilted_3 | ~1,500 | Three tilted bodies |

**Generation parameters:**
- Tilt angle: 10-75 degrees (uniform random)
- Azimuth: 0-360 degrees (uniform random)
- Size: 3-11 cells per horizontal dimension, 3-10 cells depth
- Horizontal shift per layer: sz * tan(tilt_angle)

### Type 3: Random Walk Models (10,000 samples)

Irregularly shaped geological bodies generated via 3D random walk algorithm.
These represent complex natural geometries that cannot be described by simple
prisms.

**Algorithm:**
1. Seed voxel placed near grid center (margin=5 cells from boundary)
2. At each step, choose one of 26 neighbors (3D connectivity)
3. With probability `walk_prob` (0.4-0.85), add neighbor to body
4. Direction bias (0.0-0.4) encourages continuation in same direction
5. Continue until target voxel count reached (200-4000 voxels)

**Parameters (per sample):**
- Target voxels: 200-4,000 (uniform integer random)
- Walk probability: 0.4-0.85 (uniform)
- Direction bias: 0.0-0.4 (uniform)

### Type 4: Global Consistency Combined Models (7,500 samples)

Multiple geological bodies where density and susceptibility distributions are
**fully coincident** -- every anomalous cell has both non-zero density AND
non-zero susceptibility. Structural similarity = 1.0 everywhere.

| Sub-type | Count |
|----------|-------|
| single_cuboid | 1,500 |
| combined_cuboid | 3,000 |
| single_tilted | 1,000 |
| combined_tilted | 1,000 |
| random_walk | 500 |
| combined_rw | 500 |

### Type 5: Partial Consistency Combined Models (7,500 samples)

Multiple bodies where some share geometry between density and susceptibility,
while others are independent. Structural similarity is mixed:
- Overlapping regions: s = 1.0
- Non-overlapping anomalous regions: s = 0.1

| Sub-type | Count |
|----------|-------|
| single_cuboid | 1,500 |
| combined_cuboid | 3,000 |
| single_tilted | 1,000 |
| combined_tilted | 1,000 |
| random_walk | 500 |
| combined_rw | 500 |

### Type 6: Structural Inconsistency Models (10,000 samples)

Density and susceptibility distributions are **independent** -- bodies appear
in different locations for each property. Structural similarity is near zero
for most of the model volume.

| Sub-type | Count |
|----------|-------|
| single_cuboid | 2,000 |
| combined_cuboid | 4,000 |
| single_tilted | 1,000 |
| combined_tilted | 1,000 |
| random_walk | 1,000 |
| combined_rw | 2,000 |

### Trapezoid Special Cases (5 samples)

Wedge-shaped bodies with linear width tapering across depth segments.
One sample each for 1 through 5 segments, included within the main type
distribution as special cases.

| Segments | Description |
|----------|-------------|
| 1 | Simplest wedge (abrupt width change) |
| 2 | Two-segment taper |
| 3 | Three-segment gradual taper |
| 4 | Four-segment gradual taper |
| 5 | Five-segment most gradual tapering |

## Forward Modeling Implementation

### Gravity Anomaly Computation

**Method:** Analytical Nagy et al. (2000) prism formula

The gravity effect of each rectangular prism cell is computed using the
closed-form Nagy formula:

```
dG = G * rho * sum_{i,j,k} mu_ijk *
     [z_k * arctan(x_i * y_j / (z_k * R))
    - x_i * arctan(y_j * z_k / (x_i * R))
    - y_j * arctan(x_i * z_k / (y_j * R))]
```

where mu_ijk = (-1)^(i+j+k), R = sqrt(x_i^2 + y_j^2 + z_k^2).

**Key implementation details:**
- Negation applied for z-down geophysics convention
- Vectorized over observation grid using numpy broadcasting
- Only non-zero cells iterated (sparse optimization)
- Units: output in mGal (1 mGal = 10^-5 m/s^2)
- Density converted from g/cm^3 to kg/m^3 internally (x1000)

### Magnetic Anomaly Computation

**Method:** Bhattacharyya (1964) total field anomaly formula

The total magnetic field anomaly for a uniformly magnetized prism:

```
Delta_T = (F / 4pi) * kappa * sum_{corners} mu_ijk *
          [ax*atan(yz/x) + ay*atan(xz/y) + az*atan(xy/z)
         - bxy*log(R+z) - bxz*log(R+y) - byz*log(R+x)]
```

where direction cosine products encode the geomagnetic field and
magnetization directions (assumed parallel for induced magnetization).

**Key implementation details:**
- Prefactor = F_intensity / (4*pi) nT per SI unit
- Geomagnetic direction cosines computed from declination/inclination
- Induced magnetization assumed (M parallel to F)
- Safe arctan2 and log with near-zero denominator handling
- Output in nanoTesla (nT)

## File Format

### Individual Sample Files

Each sample saved as compressed NumPy archive:

```
data/sample_XXXXXX.npz
```

Containing arrays:
- `density`: shape (40, 40, 20), float32, g/cm^3
- `susceptibility`: shape (40, 40, 20), float32, SI
- `structural_sim`: shape (40, 40, 20), float32, [0, 1]
- `gravity`: shape (81, 81), float64, mGal (noisy observation)
- `magnetic`: shape (81, 81), float64, nT (noisy observation)
- `type`: scalar int, model type (1-6)
- `subtype`: string, sub-type identifier

### Index Files

JSON files listing sample IDs per split:
- `data/train_index.json`: 31,500 entries
- `data/val_index.json`: 9,000 entries
- `data/test_index.json`: 4,500 entries

### Metadata Files

- `data/DATASET_MANIFEST.json`: Complete dataset specification
- `data/normalization_stats.json`: Min-max normalization parameters
  (computed from training set only)

## Normalization

Min-max normalization parameters are computed from the **training set only**
to prevent data leakage:

```json
{
  "input": {
    "method": "minmax",
    "min": [grav_min, mag_min],
    "max": [grav_max, mag_max],
    "mean": [grav_mean, mag_mean],
    "std": [grav_std, mag_std]
  },
  "density": { ... },
  "susceptibility": { ... }
}
```

## Quality Assurance

### Unit Test Suite (19/19 passing)

All tests in `tests/test_forward.py` verify:

**Gravity forward modeling (13 tests):**
- Prism coefficient positivity (mass below observer -> positive attraction)
- Coordinate symmetry (sign-flipped coordinates give identical result)
- Zero-density model produces zero anomaly
- Single-cell model produces physically reasonable signal (~0.001 mGal)
- Higher density produces proportionally larger signal (verified 10x ratio)
- Output units in reasonable mGal range (0.01-100 mGal for typical body)
- Noise addition preserves mean but increases variance
- Array version matches scalar version (vectorization correctness)
- Deeper prism produces weaker signal
- Larger prism produces stronger signal
- Bouguer slab approximation (order-of-magnitude check)
- Point mass limit (very small deep prism approaches point mass formula)
- Peak anomaly dominates over edge values

**Magnetic forward modeling (5 tests):**
- Zero susceptibility produces zero anomaly
- Single-cell model produces reasonable signal (<500 nT)
- Dipole character verified (positive center at 60 deg inclination)
- Higher susceptibility produces larger signal (verified 10x ratio)
- Noise addition works correctly

**End-to-end smoke test (10 samples):**
- Full pipeline: generate models -> forward modeling -> noise -> save -> reload
- Gravity range: approximately [-0.5, 1.0] mGal per sample
- Magnetic range: approximately [-150, 400] nT per sample
- Train/val/test split correctly populated
- Normalization stats computed and saved

## Visualization Figures

All figures saved in SVG format in `figures/` directory:

### Model Type Examples (depth slices)
- `dataset_type1_cuboid_1.svg` - Single cuboid density slice
- `dataset_type1_cube_1.svg` - Single cube density slice
- `dataset_type1_combined_2.svg` - Two-body combined
- `dataset_type1_combined_3.svg` - Three-body combined
- `dataset_type2_tilted_1.svg` - Single tilted body
- `dataset_type2_combined_2.svg` - Two tilted bodies
- `dataset_type2_combined_3.svg` - Three tilted bodies
- `dataset_type3_randomwalk.svg` (+ variants) - Random walk shapes
- `dataset_type4_global_consist.svg` - Global consistency example
- `dataset_type5_partial_consist.svg` - Partial consistency example
- `dataset_type6_inconsistent.svg` - Inconsistent example

### Special Cases
- `dataset_trapezoid_1seg.svg` through `dataset_trapezoid_5seg.svg`

### Observations & Statistics
- `dataset_combined_example_gravity.svg` - Example gravity anomaly map
- `dataset_combined_example_magnetic.svg` - Example magnetic anomaly map
- `dataset_distributions.svg` - Pie chart + bar chart of type distribution
- `dataset_structural_sim_stats.svg` - Histograms of structural similarity by type

### Documentation
- `FIGURE_README.md` - Complete figure index with descriptions

## Reproducibility

- Random seed: 42 (fixed for all generation)
- All RNG instances derived from master seed via `np.random.default_rng()`
- Forward modeling is deterministic (no stochasticity)
- Noise is applied after forward modeling (only source of non-determinism
  in observations; model generation is fully reproducible)

## Runtime Performance

| Phase | Estimated Time | Notes |
|-------|---------------|-------|
| Model generation (46,000 samples) | ~6-7 min | Pure numpy, CPU-bound |
| Forward modeling (45,000 samples) | ~2-6 hours | Main bottleneck: analytical formula per cell per obs point |
| Data saving (.npz compression) | ~5-10 min | Depends on disk I/O |
| SVG visualization generation | ~1-2 min | matplotlib Agg backend |
| **Total estimated** | **~2.5-6.5 hours** | Dominated by forward modeling |

## References

1. Fang, Y., et al. (2025). Improved 3-D Joint Inversion of Gravity and Magnetic
   Data Based on Deep Learning With a Multitask Learning Strategy. IEEE TGRS, 63.

2. Nagy, D., Papp, G., & Benedek, J. (2000). The gravitational potential and
   its derivatives for a prism. Journal of Geodesy, 74(7-8), 552-560.

3. Bhattacharyya, B. K. (1964). Magnetic anomalies due to prism-shaped bodies
   with arbitrary polarization. Geophysics, 29(4), 517-531.

4. Sharma, P. V. (1986). Computation of magnetic anomalies of three-dimensional
   bodies. Geoexploration, 24(2-3), 87-94.
