# Dataset Generation Report

## Overview

Synthetic dataset for 3D gravity-magnetic joint inversion (Fang et al. 2025, IEEE TGRS Vol. 63).

## 1. Grid & Physical Parameters

- **Model grid**: 40 x 40 x 20 cells (Easting x Northing x Depth)
- **Cell size**: 20 m x 20 m x 20 m
- **Extent**: 800 m x 800 m x 400 m
- **Observation**: 81 x 81 points at z = +10m
- **Density**: {0.1, 0.5, 1.0} g/cm^3
- **Susceptibility**: {0.03, 0.1, 0.3} SI
- **Geomagnetic field**: F=55000nT, D=-7deg, I=60deg
- **Noise**: Gravity 0.5%, Magnetic 1.0%

## 2. Type Distribution (Total: 45,000)

| Type | Name | Train | Sub-types |
|------|------|-------|-----------|
| 1 | Cuboid/Cube | 5,000 | single_cuboid(1500), single_cube(1200), combo_2(1300), combo_3(1000) |
| 2 | Tilting Body | 5,000 | single_tilt(1800), combo_2_tilt(1700), combo_3_tilt(1500) |
| 3 | Random Walk | 10,000 | random_walk(10000) |
| 4 | Global Consistency | 7,500 | see Table I footnote |
| 5 | Partial Consistency | 7,500 | see Table I footnote |
| 6 | Inconsistency | 10,000 | see Table I footnote |
| Trapezoid (in Types 1-3) | - | 150 | 1-5 sections x 30 each |

## 3. Data Split: 7:2:1

- Train: 31,500 (70%)
- Val: 9,000 (20%)
- Test: 4,500 (10%)

## 4. Forward Modeling

- **Gravity**: Nagy et al. (2000) prism formula, CPU+CUDA dual backend
- **Magnetic**: Bhattacharyya (1964) prism formula, CPU+CUDA dual backend
- **Validation**: 17/17 unit tests PASS (numerical vs analytical error = 0.0)

## 5. Output Files

### Code (7 files)
`src/data/forward_gravity.py`, `src/data/forward_magnetic.py`, `src/data/generate_synthetic.py`,
`src/data/dataset.py`, `src/data/transforms.py`, `scripts/make_dataset.py`, `tests/test_forward.py`

### Data (3 files)
`data/train_dataset.npz`, `data/val_dataset.npz`, `data/test_dataset.npz`

### Figures (8 SVG files)
`figures/dataset_type{1-6}_example.svg`, `dataset_trapezoid_examples.svg`, `dataset_overview.svg`

### Docs (2 files)
`docs/DATASET_REPORT.md` (this file), `figures/FIGURE_README.md`
