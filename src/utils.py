"""
工具函数 — 随机种子、参数统计、检查点管理、指标追踪
====================================================

作者: Agent-MLTestEngineer
日期: 2026-04-21
"""

import torch
import numpy as np
import random
import os
import json


def set_seed(seed=42):
    """固定所有随机种子，保证实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    统计模型参数量。

    Args:
        model: nn.Module 实例

    Returns:
        dict: {'total': int, 'trainable': int}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    保存训练检查点。

    Args:
        model: 模型实例
        optimizer: 优化器实例
        epoch: 当前 epoch 编号
        loss: 当前验证损失
        path: 保存路径 (如 checkpoints/best_model.pt)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None):
    """
    加载训练检查点。

    Args:
        path: 检查点文件路径
        model: 模型实例 (将加载权重到此模型)
        optimizer: 可选，优化器实例 (将恢复优化器状态)

    Returns:
        dict: 包含 epoch, loss 等信息
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    info = {
        'epoch': checkpoint.get('epoch', -1),
        'loss': checkpoint.get('loss', float('inf')),
    }
    return info


class AverageMeter:
    """
    运行平均值追踪器（用于训练/验证过程中的损失和指标追踪）。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def compute_metrics(predictions, targets):
    """
    计算评估指标: MSE, RMSE, MAE, Pearson Correlation Coefficient。

    分别对密度(rho)和磁化率(kappa)计算。

    Args:
        predictions: dict 含 'rho_final' 和 'kappa_final' (B,1,D,H,W)
        targets: dict 含 'rho' 和 'kappa' (B,D,H,W) 或 (B,1,D,H,W)

    Returns:
        dict: 嵌套字典 {metric_name: {rho: float, kappa: float}}
    """
    metrics = {}

    for key in ['rho', 'kappa']:
        pred_key = f'{key}_final'
        pred = predictions[pred_key].detach().cpu().flatten()
        target = targets[key].detach().cpu().flatten()

        # MSE
        mse = torch.mean((pred - target) ** 2).item()

        # RMSE
        rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()

        # MAE
        mae = torch.mean(torch.abs(pred - target)).item()

        # Pearson Correlation Coefficient
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        corr_num = torch.sum(pred_centered * target_centered)
        corr_den = torch.sqrt(
            torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2)
        )
        correlation = (corr_num / corr_den).item() if corr_den.item() > 1e-8 else 0.0

        metrics[key] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Correlation': correlation,
        }

    return metrics
