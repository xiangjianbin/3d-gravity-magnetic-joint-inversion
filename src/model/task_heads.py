"""
Task-Specific Heads — 2D->3D version.

Takes 2D ASPP features (B, 40, 40, 40) and produces 3D predictions
(B, 1, 40, 40, 20) by expanding in the depth dimension.

Tasks:
  Task 1: Independent gravity inversion (density)   — MSE loss
  Task 2: Independent magnetic inversion (suscept.) — MSE loss
  Task 3: Structural similarity                    — BCE + Sigmoid
  Task 4: Joint gravity inversion (density)          — MSE loss
  Task 5: Joint magnetic inversion (suscept.)        — MSE loss
"""

import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """
    2D task head: processes 2D features -> expands to 3D output.

    Architecture:
      Conv2d(40->64, 3x3) -> BN -> LeakyReLU
      Conv2d(64->32, 3x3) -> BN -> LeakyReLU
      Conv2d(32->1,   1x1)
      View as (B, 1, H, W) -> expand to (B, 1, D, H, W) via repeat/interpolate
    """

    def __init__(self, in_channels: int = 40, out_depth: int = 20,
                 use_sigmoid: bool = False, negative_slope: float = 0.01):
        super().__init__()
        self.out_depth = out_depth
        self.use_sigmoid = use_sigmoid

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            nn.Conv2d(32, 1, kernel_size=1, bias=True),
        )

        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 40, H, W) from ASPP.
        Returns:
            (B, 1, H, W, D) where D=out_depth.
            Note: when use_sigmoid=True, raw logits are returned (sigmoid applied
            in the loss function via BCEWithLogitsLoss for AMP compatibility).
        """
        feat = self.layers(x)  # (B, 1, H, W)
        # Expand 2D -> 3D: (B, 1, H, W) -> (B, 1, H, W, D) to match target format
        out = feat.unsqueeze(4).expand(-1, -1, -1, -1, self.out_depth).contiguous()
        return out


class TaskHeads(nn.Module):
    """All 5 task heads."""

    TASK_NAMES = [
        'independent_gravity', 'independent_magnetic',
        'structural_similarity', 'joint_gravity', 'joint_magnetic'
    ]

    def __init__(self, in_channels: int = 40, out_depth: int = 20,
                 negative_slope: float = 0.01):
        super().__init__()
        self.task1 = TaskHead(in_channels, out_depth, False, negative_slope)
        self.task2 = TaskHead(in_channels, out_depth, False, negative_slope)
        self.task3 = TaskHead(in_channels, out_depth, True,  negative_slope)
        self.task4 = TaskHead(in_channels, out_depth, False, negative_slope)
        self.task5 = TaskHead(in_channels, out_depth, False, negative_slope)
        self.heads = [self.task1, self.task2, self.task3, self.task4, self.task5]

    def forward(self, x: torch.Tensor) -> dict:
        """x: (B, 40, H, W). Returns dict of 5 tensors each (B, 1, 20, 40, 40)."""
        return {f'task{i+1}': head(x) for i, head in enumerate(self.heads)}

    def _num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = TaskHeads(in_channels=40, out_depth=20).to(device)
    print(f"Params: {heads._num_params()}")

    x = torch.randn(2, 40, 40, 40, device=device)
    with torch.no_grad():
        outs = heads(x)
    for k, v in outs.items():
        print(f"  {k}: {v.shape}  [{v.min().item:.3f}, {v.max().item:.3f}]")
    assert all(v.shape == (2, 1, 40, 40, 20) for v in outs.values())
    # task3 outputs raw logits (unbounded)
    print("PASSED")
