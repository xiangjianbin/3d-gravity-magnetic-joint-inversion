# Figures README — 重磁联合反演论文复现

All figures saved in **SVG format** (vector graphics, scalable without quality loss).

---

## Paper Figures (Fig.5 — Fig.13)

Corresponding to Fang et al., IEEE TGRS Vol.63, 2025.

| Filename | Paper Fig | Description | Status |
|----------|-----------|-------------|--------|
| `fig5_training_curves.svg` | Fig.5 | Training & validation loss curves (3 panels: gravity, magnetic, structural sim) | Generated from actual training data |
| `fig6_base_models.svg` | Fig.6 | Base geological models: (a) cuboid, (b) tilting body, (c) random walk | Synthetic visualization |
| `fig7_combined_models.svg` | Fig.7 | Combined models: global/partial/inconsistent consistency (3×3 panels) | Synthetic visualization |
| `fig8_test_model.svg` | Fig.8 | Test model: density, susceptibility, gravity obs, magnetic obs | Synthetic visualization |
| `fig9_real_data.svg` | Fig.9 | Real field data application (placeholder — Qingchuan iron deposit) | Placeholder |
| `fig10_cross_sections.svg` | Fig.10 | Cross-section profiles of inversion results | Synthetic visualization |
| `fig11_depth_slices.svg` | Fig.11 | Depth slice comparison at 60m and 260m (multi-method comparison) | Synthetic visualization |
| `fig12_3d_reconstruction.svg` | Fig.12 | 3D reconstructed model isosurface visualization | Synthetic visualization |
| `fig13_prediction_vs_observed.svg` | Fig.13 | Predicted vs observed data (gravity + magnetic fitting check) | Synthetic visualization |

### Important Note on Figure Status

**Training did not converge** (see `docs/RESULT_COMPARISON.md`). The paper figures above were generated
during Phase 2-4 using synthetic/idealized data for illustration purposes. They do NOT reflect
actual model predictions from a converged training run.

For actual (untrained) model output visualizations, see `results/figures/`.

---

## Result Analysis Figures (Phase 7 Output)

Located in `results/figures/`. These reflect the **actual training results**.

| Filename | Description |
|----------|-------------|
| `training_curves.svg` | Actual training/validation loss curves (flat — model didn't learn) |
| `inversion_density_gt.svg` | Density ground truth (diagnostic) |
| `inversion_density_pred.svg` | Density prediction (untrained model) |
| `inversion_suscept_gt.svg` | Susceptibility ground truth (diagnostic) |
| `inversion_suscept_pred.svg` | Susceptibility prediction (untrained model) |
| `inversion_struct_sim_gt.svg` | Structural similarity GT (diagnostic) |
| `inversion_struct_sim_pred.svg` | Structural similarity pred (untrained) |
| `scatter_gt_vs_pred.svg` | Scatter plot: GT vs Prediction (all tasks + diagnosis panel) |
| `slice_depth_comparison_density.svg` | Multi-depth slice comparison (density) |
| `slice_depth_comparison_suscept.svg` | Multi-depth slice comparison (susceptibility) |
| `metrics_summary.svg` | Bar chart: MSE/MAE/R²/SSIM by task |

---

## Dataset Figures (Phase 2 Output)

| Filename | Description |
|----------|-------------|
| `dataset_type{1-6}_example.svg` | Example samples for each of 6 geological model types |
| `dataset_trapezoid_examples.svg` | Trapezoid special cases (1-5 sections) |
| `dataset_overview.svg` | Dataset overview pie chart + statistics table |
| `dataset_6types_samples.pdf` | All 6 types in one page (PDF) |
| `dataset_distributions.pdf` | Property value distributions (PDF) |
| `dataset_model_examples.pdf` | Model geometry examples (PDF) |
| `dataset_split_pie.pdf` | Train/val/test split (PDF) |
| `dataset_structural_sim_stats.pdf` | Structural similarity statistics (PDF) |

---

## Coordinate System

- **X-axis**: Easting (cells 0–39 → 0m to 800m)
- **Y-axis**: Northing (cells 0–39 → 0m to 800m)
- **Z-axis**: Depth (cells 0–19 → 0m to 400m below surface)
- **Observation grid**: 81×81 centered on model, at z = +10m above surface

## Color Scheme

- Model property maps (density, susceptibility): `viridis` / `RdYlBu_r`
- Structural similarity: `RdBu_r` (blue=0 background, red=1 co-located)
- Observation data (gravity, magnetic): diverging colormaps (blue=negative, red=positive)
- Training curves: task-specific colors (blue/orange/green/red/purple)

---

*Generated: 2026-04-22*
*Project: Fang et al. 2025 TGRS — Gravity-Magnetic Joint Inversion Reproduction*
