# Dataset Visualization Figures — README

Each figure is saved in **SVG format** (vector graphics, scalable without quality loss).

## Figure List

| Filename | Description |
|----------|-------------|
| `dataset_type1_example.svg` | Type 1: Single cuboid/cube model. Depth slice of density. Subtypes: single cuboid (rectangular), single cube (near-equal dims), 2-body combo, 3-body combo. Color: viridis (purple=low, yellow=high density). |
| `dataset_type2_example.svg` | Type 2: Single tilting body. Depth slice showing non-axis-aligned anomaly from coordinate rotation with supersampling anti-aliasing. |
| `dataset_type3_example.svg` | Type 3: Random walk body. Irregular complex shape from 3D random walk algorithm with morphological dilation. |
| `dataset_type4_example.svg` | Type 4: Global structural consistency. **Three panels**: Density (left), Susceptibility (middle), Structural Similarity (right). Note: density and susceptibility anomalies fully overlap; Sim ≈ 1 on body. |
| `dataset_type5_example.svg` | Type 5: Partial structural consistency. Three panels. Some regions show co-located anomalies (Sim=1), others show single-property anomalies (Sim=0). |
| `dataset_type6_example.svg` | Type 6: Structural inconsistency. Three panels. Density and susceptibility anomalies are spatially independent; Sim ≈ 0 everywhere. |
| `dataset_trapezoid_examples.svg` | Trapezoid special cases: **5 panels** showing 1-section through 5-section trapezoids. Cross-sectional width varies linearly with depth. |
| `dataset_overview.svg` | Dataset overview: **Left**: pie chart of type distribution (Types 1-6 with counts). **Right**: summary statistics table (total samples, split ratios, grid specs, property ranges, noise levels). |

## Coordinate System

- **X-axis**: Easting (cells 0-39, corresponding to -400m to +400m)
- **Y-axis**: Northing (cells 0-39, corresponding to -400m to +400m)
- **Z-axis**: Depth (cells 0-19, corresponding to 0m to 400m below surface)
- Observation grid: 81x81 centered on model, at z = +10m above surface

## Color Scheme

- Model property maps (density, susceptibility): `viridis` colormap
- Structural similarity: binary (0 = background or single-property, 1 = co-located anomaly)
- Observation data (gravity, magnetic): diverging colormaps (blue = negative, red = positive)
