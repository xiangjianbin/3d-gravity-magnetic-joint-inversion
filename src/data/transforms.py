"""
Data Transforms for Gravity-Magnetic Joint Inversion Dataset
===========================================================

Provides preprocessing transforms for the training pipeline:
  - NormalizeTransform: Min-Max or Z-Score normalization
  - AddNoiseTransform: Online Gaussian noise augmentation
  - RandomFlipTransform: Spatial flipping (data augmentation)
  - Compose: Sequential transform composition

Author: Dataset Agent
Date: 2026-04-22
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List


class Compose:
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, input_tensor: torch.Tensor,
                 output_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        for t in self.transforms:
            input_tensor, output_dict = t(input_tensor, output_dict)
        return input_tensor, output_dict


class NormalizeTransform:
    """
    Min-Max or Z-Score normalization.

    Applied per-channel to input and per-field to output.
    """

    def __init__(self, mode: str = 'minmax',
                 input_stats: Optional[Dict] = None,
                 output_stats: Optional[Dict] = None):
        assert mode in ('minmax', 'zscore')
        self.mode = mode
        self.input_stats = input_stats
        self.output_stats = output_stats

    def _norm(self, x: torch.Tensor, stats: Optional[Dict], ch: int = 0) -> torch.Tensor:
        if self.mode == 'minmax':
            if stats and ch in stats:
                dmin, dmax = stats[ch]['min'], stats[ch]['max']
            else:
                dmin, dmax = x.min(), x.max()
            if dmax - dmin < 1e-8:
                return torch.zeros_like(x)
            return (x - dmin) / (dmax - dmin)
        else:  # zscore
            if stats and ch in stats:
                mean, std = stats[ch]['mean'], stats[ch]['std']
            else:
                mean, std = x.mean(), x.std()
            if std < 1e-8:
                return torch.zeros_like(x)
            return (x - mean) / std

    def __call__(self, input_tensor: torch.Tensor,
                 output_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        c = input_tensor.shape[0]
        norm_in = torch.zeros_like(input_tensor)
        for ch in range(c):
            norm_in[ch] = self._norm(input_tensor[ch], self.input_stats, ch)

        norm_out = {}
        for key, val in output_dict.items():
            idx = ['rho', 'kappa', 'sim'].index(key) if key in ('rho', 'kappa', 'sim') else 0
            norm_out[key] = self._norm(val, self.output_stats, idx)

        return norm_in, norm_out


class AddNoiseTransform:
    """
    Add Gaussian noise to observation channels only (not labels).

    noise_level * max(|signal|) * N(0,1), clipped to valid range.
    """

    def __init__(self, noise_level_gravity: float = 0.005,
                 noise_level_magnetic: float = 0.01,
                 noise_prob: float = 1.0):
        self.noise_g = noise_level_gravity
        self.noise_m = noise_level_magnetic
        self.prob = noise_prob

    def __call__(self, input_tensor: torch.Tensor,
                 output_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        if torch.rand(1).item() > self.prob:
            return input_tensor, output_dict

        out = input_tensor.clone()
        g_noise = torch.randn_like(out[0]) * (self.noise_g / 3.0)
        out[0] = torch.clamp(out[0] + g_noise, 0.0, 1.0)

        m_noise = torch.randn_like(out[1]) * (self.noise_m / 3.0)
        out[1] = torch.clamp(out[1] + m_noise, 0.0, 1.0)

        return out, output_dict


class RandomFlipTransform:
    """
    Random spatial flip for data augmentation.

    Flips input and all outputs consistently along specified axes.
    """

    def __init__(self, p_horizontal: float = 0.5,
                 p_vertical: float = 0.5):
        self.p_h = p_horizontal
        self.p_v = p_vertical

    def __call__(self, input_tensor: torch.Tensor,
                 output_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        inp = input_tensor
        out = {k: v.clone() for k, v in output_dict.items()}

        # Horizontal flip (dim=1 for 2D obs, dim=1 for 3D model)
        if torch.rand(1).item() < self.p_h:
            inp = torch.flip(inp, dims=[1])
            out = {k: torch.flip(v, dims=[1]) for k, v in out.items()}

        # Vertical flip (dim=0)
        if torch.rand(1).item() < self.p_v:
            inp = torch.flip(inp, dims=[0])
            out = {k: torch.flip(v, dims=[0]) for k, v in out.items()}

        return inp, out


def get_train_transforms() -> Compose:
    """Standard transform pipeline for training."""
    return Compose([
        RandomFlipTransform(p_horizontal=0.5, p_vertical=0.5),
    ])


def get_eval_transforms() -> Compose:
    """Transform pipeline for evaluation/inference (no augmentation)."""
    return Compose([])


if __name__ == '__main__':
    print("Transforms module test")
    inp = torch.randn(2, 81, 81).float()
    inp = (inp - inp.min()) / (inp.max() - inp.min() + 1e-8)
    out = {
        'rho': torch.rand(40, 40, 20).float(),
        'kappa': torch.rand(40, 40, 20).float(),
        'sim': (torch.rand(40, 40, 20) > 0.5).float(),
    }

    tf = NormalizeTransform(mode='minmax')
    r1, o1 = tf(inp, out)
    print(f"Normalize: in [{r1.min():.4f}, {r1.max():.4f}]")

    tf2 = get_train_transforms()
    r2, o2 = tf2(inp, out)
    print(f"Train TF: shape={r2.shape}")
    print("OK")
