"""
Task-Specific Heads for Gravity-Magnetic Joint Inversion Network.

Each head is a small CNN that takes the 40-channel ASPP output feature map
and produces a task-specific prediction of shape (B, 1, 40, 40, 20).

Tasks:
  Task 1: Independent gravity inversion (density)   -- MSE loss
  Task 2: Independent magnetic inversion (suscept.) -- MSE loss
  Task 3: Structural similarity extraction            -- BCE + Sigmoid
  Task 4: Joint gravity inversion (density)          -- MSE loss
  Task 5: Joint magnetic inversion (suscept.)        -- MSE loss

Reproduced from:
  Fang et al., "Improved 3-D Joint Inversion of Gravity and Magnetic Data
  Based on Deep Learning With a Multitask Learning Strategy",
  IEEE TGRS, Vol. 63, 2025
"""

import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """Generic task head: a few Conv3d layers mapping ASPP features to output.

    Architecture (per paper Fig.2/Fig.3):
      Conv3d(40->64, 3x3x3) -> BN -> LeakyReLU(0.01)
      Conv3d(64->32, 3x3x3) -> BN -> LeakyReLU(0.01)
      Conv3d(32->1,   1x1x1)                    # final projection to single channel
      [optional Sigmoid for classification tasks]
    """

    def __init__(self, in_channels: int = 40, use_sigmoid: bool = False,
                 negative_slope: float = 0.01) -> None:
        """
        Args:
            in_channels:    Number of input channels from ASPP (default 40).
            use_sigmoid:    Whether to apply Sigmoid at the output (for Task 3).
            negative_slope: Negative slope for Leaky-ReLU (paper uses ~0.01).
        """
        super().__init__()
        self.use_sigmoid = use_sigmoid

        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            nn.Conv3d(32, 1, kernel_size=1, bias=True),  # 1x1x1 projection
        )

        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ASPP output features, shape (B, 40, D, H, W).

        Returns:
            Task prediction, shape (B, 1, D, H, W).  If use_sigmoid=True,
            values are in [0, 1].
        """
        x = self.layers(x)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x


class TaskHeads(nn.Module):
    """Container for all 5 task heads.

    Each head is an independent TaskHead instance sharing the same architecture
    but with separate weights.  All heads receive the same ASPP feature map.
    """

    # Task names for indexing / logging
    TASK_NAMES = [
        'independent_gravity',     # Task 1
        'independent_magnetic',    # Task 2
        'structural_similarity',   # Task 3
        'joint_gravity',           # Task 4
        'joint_magnetic',          # Task 5
    ]

    def __init__(self, in_channels: int = 40,
                 negative_slope: float = 0.01) -> None:
        """
        Args:
            in_channels:    ASPP output channels (default 40).
            negative_slope: Leaky-ReLU negative slope.
        """
        super().__init__()

        # Task 1: Independent gravity inversion (regression)
        self.task1 = TaskHead(in_channels, use_sigmoid=False,
                              negative_slope=negative_slope)

        # Task 2: Independent magnetic inversion (regression)
        self.task2 = TaskHead(in_channels, use_sigmoid=False,
                              negative_slope=negative_slope)

        # Task 3: Structural similarity (classification -> sigmoid)
        self.task3 = TaskHead(in_channels, use_sigmoid=True,
                              negative_slope=negative_slope)

        # Task 4: Joint gravity inversion (regression)
        self.task4 = TaskHead(in_channels, use_sigmoid=False,
                              negative_slope=negative_slope)

        # Task 5: Joint magnetic inversion (regression)
        self.task5 = TaskHead(in_channels, use_sigmoid=False,
                              negative_slope=negative_slope)

        # Store as list for easy iteration
        self.heads = [self.task1, self.task2, self.task3,
                      self.task4, self.task5]

    def forward(self, x: torch.Tensor) -> dict:
        """
        Run all 5 task heads on the same input feature map.

        Args:
            x: ASPP output, shape (B, 40, D, H, W).

        Returns:
            dict with keys 'task1'..'task5', each value is (B, 1, D, H, W).
        """
        return {
            'task1': self.task1(x),  # independent gravity density
            'task2': self.task2(x),  # independent magnetic susceptibility
            'task3': self.task3(x),  # structural similarity (sigmoid)
            'task4': self.task4(x),  # joint gravity density
            'task5': self.task5(x),  # joint magnetic susceptibility
        }

    def _num_params(self) -> int:
        """Return total number of trainable parameters across all heads."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick smoke-test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    heads = TaskHeads(in_channels=40).to(device)
    total_params = heads._num_params()
    print(f"TaskHeads total parameters: {total_params:,}")

    x = torch.randn(2, 40, 40, 40, 20, device=device)
    with torch.no_grad():
        outputs = heads(x)

    print(f"\nInput shape: {x.shape}")
    for key, out in outputs.items():
        print(f"  {key}: {out.shape}  range=[{out.min().item():.4f}, {out.max().item():.4f}]")

    # Verify shapes
    for key, out in outputs.items():
        assert out.shape == (2, 1, 40, 40, 20), f"{key} shape mismatch: {out.shape}"

    # Verify task3 output is in [0, 1] due to sigmoid
    assert outputs['task3'].min() >= 0.0, "task3 should be >= 0 (sigmoid)"
    assert outputs['task3'].max() <= 1.0, "task3 should be <= 1 (sigmoid)"

    print("\nAll smoke tests PASSED.")
