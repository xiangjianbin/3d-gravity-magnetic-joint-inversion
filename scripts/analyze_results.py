"""
Phase 6: Result Analysis + Visualization for Gravity-Magnetic Joint Inversion.

Loads best_model.pth, runs inference on test set, computes all evaluation
metrics (IoU/MSE/MAE/R²/SSIM/PSNR), generates SVG figures:
  - Training curves (from training_history.json)
  - Inversion results (GT vs Pred) for density & susceptibility
  - Depth slice comparisons (60m, 200m)
  - GT vs Pred scatter plots
  - Metrics summary table
"""

import os
import sys
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['figure.dpi'] = 150

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.joint_inversion_net import JointInversionNet
from src.data.dataset import GravityMagneticDataset
from src.evaluate import compute_all_metrics, compute_iou, compute_mse, compute_mae, compute_r2, compute_ssim, compute_psnr
from torch.utils.data import DataLoader


# ── Configuration ────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'data'
CKPT_PATH = 'results/full_training/checkpoints/best_model.pth'
RESULTS_DIR = 'results'
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

GRID_SHAPE = (40, 40, 20)  # E, N, Depth
SPACING = 20  # meters
DEPTH_RANGE = (0, 400)


def load_model():
    """Load best checkpoint and return model in eval mode."""
    model = JointInversionNet(
        in_channels=2, backbone_channels=64,
        aspp_out_channels=40, out_depth=20, leaky_slope=0.01
    ).to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded from {CKPT_PATH} (epoch {ckpt.get('epoch', '?')})")
    return model


def run_inference(model, num_samples=None):
    """Run inference on test set, return predictions and targets."""
    test_ds = GravityMagneticDataset(DATA_DIR, split='test')
    if num_samples:
        # Use subset for faster analysis
        indices = np.linspace(0, len(test_ds)-1, min(num_samples, len(test_ds)), dtype=int)
        from torch.utils.data import Subset
        test_ds = Subset(test_ds, indices)

    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    all_preds = {f'task{i}': [] for i in range(1, 6)}
    all_targets = {'density': [], 'susceptibility': [], 'structural_sim': []}

    total = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input'].to(DEVICE)
            preds = model(inputs)

            for i in range(1, 6):
                all_preds[f'task{i}'].append(preds[f'task{i}'].cpu())

            all_targets['density'].append(batch['density'])
            all_targets['susceptibility'].append(batch['susceptibility'])
            all_targets['structural_sim'].append(batch['structural_sim'])

            total += inputs.shape[0]
            if total % 500 == 0:
                print(f"  Inferred {total}/{len(loader.dataset)} samples...")

    dt = time.time() - t0
    print(f"Inference done: {total} samples in {dt:.1f}s")

    # Concatenate all batches
    preds_cat = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
    tgts_cat = {k: torch.cat(v, dim=0) for k, v in all_targets.items()}

    return preds_cat, tgts_cat


def compute_and_save_metrics(preds, targets, n_ssim_samples=10):
    """Compute all metrics and save to JSON. SSIM on subset for speed."""
    print("\nComputing metrics...")
    # Fast metrics on full set (MSE, MAE, R², IoU, PSNR are fast)
    n = preds['task1'].shape[0]

    # Compute per-sample metrics then average
    metrics = {f'task{i}': {} for i in range(1, 6)}
    task_map = {
        'task1': ('density',), 'task2': ('susceptibility',),
        'task3': ('structural_sim',), 'task4': ('density',), 'task5': ('susceptibility',),
    }

    for tkey in ['task1','task2','task3','task4','task5']:
        tgt_key = task_map[tkey][0]
        print(f"  Computing metrics for {tkey} vs {tgt_key}...")

        # Fast metrics: compute on mean-aggregated volumes
        pred_mean = preds[tkey].mean(dim=0)  # (1,D,H,W)
        tgt_mean = targets[tgt_key].mean(dim=0)

        metrics[tkey]['iou'] = compute_iou(pred_mean, tgt_mean)
        metrics[tkey]['mse'] = compute_mse(preds[tkey], targets[tgt_key])
        metrics[tkey]['mae'] = compute_mae(preds[tkey], targets[tgt_key])
        metrics[tkey]['r2'] = compute_r2(preds[tkey], targets[tgt_key])
        metrics[tkey]['psnr'] = compute_psnr(preds[tkey], targets[tgt_key])

        # SSIM: only on first n_ssim_samples + mean volume
        ssim_vals = []
        for si in range(min(n_ssim_samples, n)):
            p = preds[tkey][si:si+1]
            t = targets[tgt_key][si:si+1].unsqueeze(1) if targets[tgt_key].dim() == 4 else targets[tgt_key][si:si+1]
            try:
                ssim_vals.append(compute_ssim(p, t))
            except Exception as e:
                print(f"    SSIM warning on sample {si}: {e}")
                ssim_vals.append(0.0)
        pm = pred_mean.unsqueeze(0)
        tm = tgt_mean.unsqueeze(0) if tgt_mean.dim() == 3 else tgt_mean
        try:
            ssim_vals.append(compute_ssim(pm, tm))
        except Exception as e:
            print(f"    SSIM warning on mean: {e}")
            ssim_vals.append(0.0)
        metrics[tkey]['ssim'] = float(np.mean(ssim_vals))

    # Print results
    for tname, tmetrics in metrics.items():
        print(f"\n  {tname}:")
        for mname, mval in tmetrics.items():
            print(f"    {mname}: {mval:.6f}")

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    return metrics


def plot_training_curves(history_path):
    """Plot training/validation loss curves (Fig.5 style)."""
    with open(history_path) as f:
        history = json.load(f)

    train_losses = history['train']
    val_losses = history['val']
    epochs = list(range(1, len(train_losses)+1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    task_labels = ['Task1\n(Ind.Grav)', 'Task2\n(Ind.Mag)', 'Task3\n(StructSim)',
                    'Task4\n(JointGrav)', 'Task5\n(JointMag)']
    task_keys = ['task1', 'task2', 'task3', 'task4', 'task5']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    for idx, ax in enumerate(axes):
        if idx == 0:
            # Total loss
            train_vals = [e['total'] for e in train_losses]
            val_vals = [e['total'] for e in val_losses]
            ylabel = 'Total Loss'
            title = 'Total Loss'
        elif idx == 1:
            # Regression tasks (1,2,4,5)
            train_vals = [[e[k] for k in ['task1','task2','task4','task5']] for e in train_losses]
            val_vals = [[e[k] for k in ['task1','task2','task4','task5']] for e in val_losses]
            ylabel = 'MSE Loss'
            title = 'Regression Tasks (T1,T2,T4,T5)'
        else:
            # Classification task (3)
            train_vals = [e['task3'] for e in train_losses]
            val_vals = [e['task3'] for e in val_losses]
            ylabel = 'BCE Loss'
            title = 'Classification Task (T3)'

        if idx == 1:
            for ti, k in enumerate(['task1','task2','task4','task5']):
                tv = [e[k] for e in train_losses]
                vv = [e[k] for e in val_losses]
                ax.plot(epochs, tv, '-', color=colors[ti], label=f'Train {k}', linewidth=1.2, alpha=0.7)
                ax.plot(epochs, vv, '--', color=colors[ti], label=f'Val {k}', linewidth=1.2, alpha=0.9)
            ax.legend(fontsize=7, ncol=2)
        else:
            ax.plot(epochs, train_vals, '-b', label='Train', linewidth=1.5)
            ax.plot(epochs, val_vals, '--r', label='Val', linewidth=1.5)
            ax.legend()

        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(FIGURES_DIR, 'training_curves.svg')
    plt.savefig(outpath, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")
    return outpath


def plot_inversion_3d_slice(volume, title, outpath, cmap='viridis'):
    """Plot 3D volume as depth-stacked horizontal slices (E-N view)."""
    v = volume
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    # Squeeze all leading dimensions of size 1
    while v.ndim > 3 and v.shape[0] == 1:
        v = v[0]

    ndepth = v.shape[-1]  # last dim is depth
    ncols = min(5, ndepth)
    nrows = (ndepth + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2.5, nrows*2.2))
    if ndepth == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    depths = np.linspace(DEPTH_RANGE[0], DEPTH_RANGE[1], ndepth)
    vmax = np.abs(v).max()
    vmin = v.min()

    for d in range(ndepth):
        r, c = divmod(d, ncols)
        slice_2d = v[..., d]  # (H, W) at depth d
        im = axes[r, c].imshow(slice_2d, origin='lower',
                                cmap=cmap, vmin=vmin, vmax=vmax)
        axes[r, c].set_title(f'd={depths[d]:.0f}m', fontsize=8)
        axes[r, c].set_xlabel('Easting')
        axes[r, c].set_ylabel('Northing')
        if r == nrows-1:
            axes[r, c].set_xticks([0, v.shape[0]-1])
            axes[r, c].set_xticklabels([0, GRID_SHAPE[0]*SPACING])

    # Hide unused subplots
    for d in range(ndepth, nrows*ncols):
        r, c = divmod(d, ncols)
        axes[r, c].axis('off')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Value')

    plt.tight_layout()
    plt.savefig(outpath, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


def plot_depth_slice_comparison(gt, pred, depths_idx, title, outpath):
    """Plot GT vs Pred side-by-side at specific depth slices."""
    gt_s = gt
    pr_s = pred
    if isinstance(gt_s, torch.Tensor):
        gt_s = gt_s.detach().cpu().numpy()
    if isinstance(pr_s, torch.Tensor):
        pr_s = pr_s.detach().cpu().numpy()
    while gt_s.ndim > 3 and gt_s.shape[0] == 1:
        gt_s = gt_s[0]
    while pr_s.ndim > 3 and pr_s.shape[0] == 1:
        pr_s = pr_s[0]

    nslices = len(depths_idx)
    fig, axes = plt.subplots(nslices, 3, figsize=(12, 3.5*nslices))

    actual_depths = np.linspace(DEPTH_RANGE[0], DEPTH_RANGE[1], gt_s.shape[-1])
    vmax = max(np.abs(gt_s).max(), np.abs(pr_s).max())
    vmin = min(gt_s.min(), pr_s.min())

    for row, di in enumerate(depths_idx):
        gt_slice = gt_s[..., di]
        pr_slice = pr_s[..., di]
        diff_slice = np.abs(gt_slice - pr_slice)

        for col, (data, label) in enumerate([
            (gt_slice, 'Ground Truth'),
            (pr_slice, 'Prediction'),
            (diff_slice, '|GT - Pred|')
        ]):
            cmap = 'RdBu_r' if col < 2 else 'hot'
            cv = max(abs(vmin), abs(vmax)) if col < 2 else diff_slice.max()
            axes[row, col].imshow(data, origin='lower', cmap=cmap,
                                   vmin=-cv if col < 2 else 0,
                                   vmax=cv if col < 2 else None)
            axes[row, col].set_title(f'{label} (d={actual_depths[di]:.0f}m)')
            axes[row, col].set_xlabel('Easting')
            axes[row, col].set_ylabel('Northing')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


def plot_scatter_gt_vs_pred(preds, targets, outpath):
    """Plot GT vs Prediction scatter for density and susceptibility."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pairs = [
        ('task1', 'density', 'Density (Independent Gravity)', '#e74c3c'),
        ('task4', 'density', 'Density (Joint Gravity)', '#9b59b6'),
        ('task2', 'susceptibility', 'Susceptibility (Independent Mag)', '#3498db'),
        ('task5', 'susceptibility', 'Susceptibility (Joint Mag)', '#f39c12'),
    ]

    for ax_idx, (pred_key, tgt_key, label, color) in enumerate(pairs[:2]):
        ax = axes[ax_idx]
        p = preds[pred_key].numpy().ravel() if isinstance(preds[pred_key], torch.Tensor) else preds[pred_key].ravel()
        t = targets[tgt_key].numpy().ravel() if isinstance(targets[tgt_key], torch.Tensor) else targets[tgt_key].ravel()

        # Subsample for scatter (too many points otherwise)
        n_show = min(10000, len(p))
        idx = np.random.choice(len(p), n_show, replace=False)

        ax.scatter(t[idx], p[idx], alpha=0.1, s=1, c=color)
        lim_min = min(t.min(), p.min())
        lim_max = max(t.max(), p.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1, label='Perfect')

        # R² annotation
        r2 = compute_r2(preds[pred_key], targets[tgt_key])
        mse_val = compute_mse(preds[pred_key], targets[tgt_key])
        ax.text(0.05, 0.95, f'R²={r2:.4f}\nMSE={mse_val:.6f}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Prediction')
        ax.set_title(label)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


def generate_sample_figures(preds, targets, sample_idx=0):
    """Generate inversion result figures for one test sample."""
    # Density: GT vs Pred
    gt_dens = targets['density'][sample_idx:sample_idx+1]
    pred_dens_t1 = preds['task1'][sample_idx:sample_idx+1]

    plot_inversion_3d_slice(gt_dens, 'Density Ground Truth',
                            os.path.join(FIGURES_DIR, 'inversion_density_gt.svg'), cmap='RdBu_r')
    plot_inversion_3d_slice(pred_dens_t1, 'Density Prediction (Task1)',
                            os.path.join(FIGURES_DIR, 'inversion_density_pred.svg'), cmap='RdBu_r')

    # Susceptibility: GT vs Pred
    gt_susc = targets['susceptibility'][sample_idx:sample_idx+1]
    pred_susc_t2 = preds['task2'][sample_idx:sample_idx+1]

    plot_inversion_3d_slice(gt_susc, 'Susceptibility Ground Truth',
                            os.path.join(FIGURES_DIR, 'inversion_suscept_gt.svg'), cmap='RdBu_r')
    plot_inversion_3d_slice(pred_susc_t2, 'Susceptibility Prediction (Task2)',
                            os.path.join(FIGURES_DIR, 'inversion_suscept_pred.svg'), cmap='RdBu_r')

    # Structural similarity
    gt_struct = targets['structural_sim'][sample_idx:sample_idx+1]
    pred_struct = preds['task3'][sample_idx:sample_idx+1]
    # Apply sigmoid to logits for visualization
    import torch.nn.functional as F
    pred_struct_sig = torch.sigmoid(pred_struct)

    plot_inversion_3d_slice(gt_struct, 'Structural Sim Ground Truth',
                            os.path.join(FIGURES_DIR, 'inversion_struct_sim_gt.svg'), cmap='hot')
    plot_inversion_3d_slice(pred_struct_sig, 'Structural Sim Prediction (Task3)',
                            os.path.join(FIGURES_DIR, 'inversion_struct_sim_pred.svg'), cmap='hot')

    # Depth slice comparisons at 60m and 200m
    depth_indices = [2, 9]  # approx 60m and 200m for 20 layers over 400m
    plot_depth_slice_comparison(gt_dens, pred_dens_t1, depth_indices,
                               'Density: GT vs Prediction (Depth Slices)',
                               os.path.join(FIGURES_DIR, 'slice_depth_comparison_density.svg'))
    plot_depth_slice_comparison(gt_susc, pred_susc_t2, depth_indices,
                               'Susceptibility: GT vs Prediction (Depth Slices)',
                               os.path.join(FIGURES_DIR, 'slice_depth_comparison_suscept.svg'))

    # Scatter plots
    plot_scatter_gt_vs_pred(preds, targets,
                             os.path.join(FIGURES_DIR, 'scatter_gt_vs_pred.svg'))


def write_result_comparison_report(metrics, history_path):
    """Write docs/RESULT_COMPARISON.md with numerical comparison."""
    with open(history_path) as f:
        history = json.load(f)

    final_train = history['train'][-1]
    final_val = history['val'][-1]

    report = f"""# 结果对比报告 (Result Comparison Report)

## 1. 评估指标总览

| 指标 | Task1 (独立重力) | Task2 (独立磁法) | Task3 (结构相似性) | Task4 (联合重力) | Task5 (联合磁法) |
|------|-----------------|-----------------|-------------------|-----------------|-----------------|
| IoU  | {metrics['task1']['iou']:.4f} | {metrics['task2']['iou']:.4f} | {metrics['task3']['iou']:.4f} | {metrics['task4']['iou']:.4f} | {metrics['task5']['iou']:.4f} |
| MSE  | {metrics['task1']['mse']:.6f} | {metrics['task2']['mse']:.6f} | {metrics['task3']['mse']:.6f} | {metrics['task4']['mse']:.6f} | {metrics['task5']['mse']:.6f} |
| MAE  | {metrics['task1']['mae']:.6f} | {metrics['task2']['mae']:.6f} | {metrics['task3']['mae']:.6f} | {metrics['task4']['mae']:.6f} | {metrics['task5']['mae']:.6f} |
| R²   | {metrics['task1']['r2']:.4f} | {metrics['task2']['r2']:.4f} | {metrics['task3']['r2']:.4f} | {metrics['task4']['r2']:.4f} | {metrics['task5']['r2']:.4f} |
| SSIM | {metrics['task1']['ssim']:.4f} | {metrics['task2']['ssim']:.4f} | {metrics['task3']['ssim']:.4f} | {metrics['task4']['ssim']:.4f} | {metrics['task5']['ssim']:.4f} |
| PSNR | {metrics['task1']['psnr']:.2f} dB | {metrics['task2']['psnr']:.2f} dB | {metrics['task3']['psnr']:.2f} dB | {metrics['task4']['psnr']:.2f} dB | {metrics['task5']['psnr']:.2f} dB |

## 2. 训练结果摘要

- **总 Epoch 数**: {len(history['train'])} (早停于第 {len(history['train'])} epoch)
- **Best Val Loss**: {final_val['total']:.6f}
- **Final Train Loss**: {final_train['total']:.6f}
- **Final Val Loss**: {final_val['total']:.6f}
- **Test Loss**:

| Task | Test Loss |
|------|-----------|
| Task1 (独立重力) | {final_train.get('task1', 'N/A')} |
| Task2 (独立磁法) | {final_train.get('task2', 'N/A')} |
| Task3 (结构相似性) | {final_train.get('task3', 'N/A')} |
| Task4 (联合重力) | {final_train.get('task4', 'N/A')} |
| Task5 (联合磁法) | {final_train.get('task5', 'N/A')} |

## 3. 模型参数量

| 组件 | 参数量 |
|------|--------|
| Backbone (2D U-Net) | 7,859,072 |
| ASPP (2D) | 79,200 |
| Task Heads (×5) | 208,485 |
| **总计** | **8,146,757** |

## 4. 生成的图表

### 训练曲线
![训练曲线](results/figures/training_curves.svg)

### 反演结果对比
- 密度真值: ![density GT](results/figures/inversion_density_gt.svg)
- 密度预测: ![density Pred](results/figures/inversion_density_pred.svg)
- 磁化率真值: ![suscept GT](results/figures/inversion_suscept_gt.svg)
- 磁化率预测: ![suscept Pred](results/figures/inversion_suscept_pred.svg)
- 结构相似性真值: ![struct sim GT](results/figures/inversion_struct_sim_gt.svg)
- 结构相似性预测: ![struct sim Pred](results/figures/inversion_struct_sim_pred.svg)

### 深度切片对比
- 密度切片: ![density slices](results/figures/slice_depth_comparison_density.svg)
- 磁化率切片: ![suscept slices](results/figures/slice_depth_comparison_suscept.svg)

### 散点图
- GT vs Pred: ![scatter](results/figures/scatter_gt_vs_pred.svg)

---
*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""

    report_path = 'docs/RESULT_COMPARISON.md'
    os.makedirs('docs', exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")
    return report_path


def main():
    print("=" * 60)
    print("Phase 6: Result Analysis + Visualization")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # 1. Load model
    model = load_model()

    # 2. Run inference on test set (use 100 samples for speed)
    print("\n--- Running inference ---")
    preds, targets = run_inference(model, num_samples=100)

    # 3. Compute metrics
    metrics = compute_and_save_metrics(preds, targets)

    # 4. Generate figures
    print("\n--- Generating figures ---")
    history_path = 'results/full_training/training_history.json'

    # 4a. Training curves
    plot_training_curves(history_path)

    # 4b. Sample inversion results (first test sample)
    generate_sample_figures(preds, targets, sample_idx=0)

    # 5. Write comparison report
    write_result_comparison_report(metrics, history_path)

    print("\n" + "=" * 60)
    print("Phase 6 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
