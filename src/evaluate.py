"""
Evaluation Metrics for Gravity-Magnetic Joint Inversion.

Implements all required evaluation metrics:
  - IoU  (Intersection over Union)
  - MSE  (Mean Squared Error)
  - MAE  (Mean Absolute Error)
  - R^2  (Coefficient of Determination)
  - SSIM (Structural Similarity Index)
  - PSNR (Peak Signal-to-Noise Ratio)

All metrics operate on 3D volumes (B, 1, D, H, W) or (D, H, W).
"""

import numpy as np
import torch
from typing import Optional


def compute_iou(pred, target,
                threshold: float = 0.5) -> float:
    """Compute Intersection over Union (IoU) for binary segmentation.

    Both pred and target are thresholded to binary masks before computing IoU.
    This measures how well the predicted anomaly body overlaps with the ground
    truth anomaly body.

    Args:
        pred:      Predicted volume, shape (..., D, H, W) or (B, 1, D, H, W).
        target:    Ground truth volume, same shape as pred.
        threshold: Threshold for binarizing predictions.

    Returns:
        IoU score in [0, 1].  Returns 0.0 if both masks are empty.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Flatten and threshold
    pred_bin = (pred > threshold).astype(np.float64).ravel()
    target_bin = (target > threshold).astype(np.float64).ravel()

    intersection = np.sum(pred_bin * target_bin)
    union = np.sum(pred_bin) + np.sum(target_bin) - intersection

    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_mse(pred, target) -> float:
    """Compute Mean Squared Error between prediction and target.

    Args:
        pred:   Predicted volume.
        target: Ground truth volume.

    Returns:
        MSE value (lower is better).
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    return float(np.mean((pred - target) ** 2))


def compute_mae(pred, target) -> float:
    """Compute Mean Absolute Error between prediction and target.

    Args:
        pred:   Predicted volume.
        target: Ground truth volume.

    Returns:
        MAE value (lower is better).
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    return float(np.mean(np.abs(pred - target)))


def compute_r2(pred, target) -> float:
    """Compute R-squared (coefficient of determination).

    R^2 = 1 - SS_res / SS_tot
    where SS_res = sum((y_true - y_pred)^2)
          SS_tot = sum((y_true - mean(y_true))^2)

    Args:
        pred:   Predicted volume.
        target: Ground truth volume.

    Returns:
        R^2 value in (-inf, 1].  1.0 is perfect prediction.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)

    if ss_tot == 0:
        return 1.0  # constant target -> perfect fit

    return float(1.0 - ss_res / ss_tot)


def compute_ssim(pred, target,
                 data_range=None) -> float:
    """Compute Structural Similarity Index (SSIM).

    Uses skimage's structural_similarity for 3D data when available,
    with a fallback to a simplified implementation.

    Args:
        pred:        Predicted volume.
        target:      Ground truth volume.
        data_range:  Value range of the data. If None, computed from target.

    Returns:
        SSIM value in [-1, 1].  1.0 means identical.
    """
    try:
        from skimage.metrics import structural_similarity as _ssim
    except ImportError:
        # Fallback: use Pearson correlation as a simple structural similarity proxy
        return _pearson_correlation(pred, target)

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    if data_range is None:
        data_range = float(target.max() - target.min())
    if data_range == 0:
        data_range = 1.0

    # Remove channel dimension if present
    p = pred.squeeze() if pred.ndim == 5 else pred
    t = target.squeeze() if target.ndim == 5 else target

    # Use 3D SSIM with a reasonable window size
    # Choose window size that fits the smallest spatial dimension
    min_spatial = min(p.shape[-3:])
    win_size = min(7, min_spatial if min_spatial % 2 == 1 else min_spatial - 1)
    if win_size < 3:
        win_size = 3
    try:
        val = _ssim(p, t, data_range=data_range, win_size=win_size, channel=None)
    except (ValueError, NotImplementedError):
        # Fall back to per-slice 2D SSIM averaged over depth slices
        scores = []
        min_depth = min(p.shape[0], t.shape[0])
        for d in range(min_depth):
            slice_win = min(win_size, min(p.shape[1], p.shape[2]))
            if slice_win < 3:
                slice_win = 3
            s = _ssim(p[d], t[d], data_range=data_range, win_size=slice_win)
            scores.append(s)
        val = float(np.mean(scores)) if scores else 0.0

    return float(val)


def _pearson_correlation(pred, target) -> float:
    """Fallback similarity metric using Pearson correlation coefficient."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    p_flat = pred.ravel()
    t_flat = target.ravel()

    p_mean = np.mean(p_flat)
    t_mean = np.mean(t_flat)

    num = np.sum((p_flat - p_mean) * (t_flat - t_mean))
    den = np.sqrt(np.sum((p_flat - p_mean) ** 2) * np.sum((t_flat - t_mean) ** 2))

    if den == 0:
        return 0.0
    return float(num / den)


def compute_psnr(pred, target,
                 data_range=None) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR).

    PSNR = 10 * log10(MAX^2 / MSE)
    Higher is better. Typical values: 20-40 dB for good reconstructions.

    Args:
        pred:        Predicted volume.
        target:      Ground truth volume.
        data_range:  Maximum possible value in the data range.
                     If None, uses max(abs(target)).

    Returns:
        PSNR value in dB.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    mse_val = np.mean((pred - target) ** 2)

    if mse_val == 0:
        return float('inf')

    if data_range is None:
        data_range = float(np.max(np.abs(target)))
    if data_range == 0:
        data_range = 1.0

    return float(10.0 * np.log10(data_range ** 2 / mse_val))


def compute_all_metrics(predictions: dict, targets: dict,
                        iou_threshold: float = 0.5) -> dict:
    """Compute all metrics for each task and return a structured result dict.

    Args:
        predictions: Dict of model outputs {'task1'..'task5': tensor}.
        targets:     Dict of ground truth {
                        'density', 'susceptibility', 'structural_sim'
                      }.
        iou_threshold: Threshold for IoU computation.

    Returns:
        Nested dict: {task_name: {metric_name: value}}
    """
    results = {}

    # Task 1 & 4: gravity density prediction vs density target
    for tkey in ['task1', 'task4']:
        pred = predictions[tkey]
        tgt = targets['density']
        name = f'task{tkey[-1]}'
        results[name] = {
            'iou': compute_iou(pred, tgt, iou_threshold),
            'mse': compute_mse(pred, tgt),
            'mae': compute_mae(pred, tgt),
            'r2': compute_r2(pred, tgt),
            'ssim': compute_ssim(pred, tgt),
            'psnr': compute_psnr(pred, tgt),
        }

    # Task 2 & 5: magnetic susceptibility prediction vs susceptibility target
    for tkey in ['task2', 'task5']:
        pred = predictions[tkey]
        tgt = targets['susceptibility']
        name = f'task{tkey[-1]}'
        results[name] = {
            'iou': compute_iou(pred, tgt, iou_threshold),
            'mse': compute_mse(pred, tgt),
            'mae': compute_mae(pred, tgt),
            'r2': compute_r2(pred, tgt),
            'ssim': compute_ssim(pred, tgt),
            'psnr': compute_psnr(pred, tgt),
        }

    # Task 3: structural similarity prediction vs structural_sim target
    pred_t3 = predictions['task3']
    tgt_struct = targets['structural_sim']
    results['task3'] = {
        'iou': compute_iou(pred_t3, tgt_struct, iou_threshold),
        'mse': compute_mse(pred_t3, tgt_struct),
        'mae': compute_mae(pred_t3, tgt_struct),
        'r2': compute_r2(pred_t3, tgt_struct),
        'ssim': compute_ssim(pred_t3, tgt_struct),
        'psnr': compute_psnr(pred_t3, tgt_struct),
    }

    return results


# ---------------------------------------------------------------------------
# Quick smoke-test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    np.random.seed(42)
    pred = torch.rand(2, 1, 8, 8, 4)  # small test volume
    tgt = torch.rand(2, 1, 8, 8, 4)

    print("=== Individual metric tests ===")
    print(f"IoU  (thresh=0.5): {compute_iou(pred, tgt):.4f}")
    print(f"MSE:               {compute_mse(pred, tgt):.6f}")
    print(f"MAE:               {compute_mae(pred, tgt):.6f}")
    print(f"R^2:               {compute_r2(pred, tgt):.4f}")
    print(f"SSIM:              {compute_ssim(pred, tgt):.4f}")
    print(f"PSNR:              {compute_psnr(pred, tgt):.2f} dB")

    # Test batched evaluation
    preds = {
        'task1': pred.clone(),
        'task2': pred.clone(),
        'task3': (torch.sigmoid(torch.randn_like(pred)) > 0.5).float(),
        'task4': pred.clone(),
        'task5': pred.clone(),
    }
    tgts = {
        'density': tgt.clone(),
        'susceptibility': tgt.clone(),
        'structural_sim': (tgt > 0.5).float(),
    }
    all_metrics = compute_all_metrics(preds, tgts)

    print("\n=== All metrics (batched) ===")
    for task_name, metrics in all_metrics.items():
        print(f"{task_name}:")
        for mname, mval in metrics.items():
            print(f"  {mname}: {mval:.4f}")

    print("\nAll evaluate smoke tests PASSED.")
