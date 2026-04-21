"""
Joint Inversion Network -- Main Model.

Combines the 2D U-Net backbone, 2D ASPP module, and 5 task-specific heads into
a single end-to-end trainable network for gravity-magnetic joint inversion.

Architecture (paper Fig.1/Fig.2):
  Input (B, 2, 81, 81)       [gravity_obs on 81x81 grid, magnetic_obs on 81x81 grid]
    -> 2D U-Net Backbone     (feature extraction: 81x81 -> 40x40)
    -> ASPP 2D               (multi-scale feature aggregation)
    -> 5 Task Heads           (2D->3D expansion + task-specific predictions)

Outputs:
  Task 1: Independent gravity density      (B, 1, 40, 40, 20)  MSE
  Task 2: Independent magnetic suscept.    (B, 1, 40, 40, 20)  MSE
  Task 3: Structural similarity             (B, 1, 40, 40, 20)  BCE+Sigmoid
  Task 4: Joint gravity density            (B, 1, 40, 40, 20)  MSE
  Task 5: Joint magnetic susceptibility    (B, 1, 40, 40, 20)  MSE

Reproduced from:
  Fang et al., "Improved 3-D Joint Inversion of Gravity and Magnetic Data
  Based on Deep Learning With a Multitask Learning Strategy",
  IEEE TGRS, Vol. 63, 2025
"""

import torch
import torch.nn as nn

try:
    from .backbone_unet3d import UNet2DBackbone
    from .aspp import ASPP2d
    from .task_heads import TaskHeads
except ImportError:
    # Standalone execution: add project root to path and retry
    import sys, os
    _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    from src.model.backbone_unet3d import UNet2DBackbone
    from src.model.aspp import ASPP2d
    from src.model.task_heads import TaskHeads


class JointInversionNet(nn.Module):
    """Full joint inversion network: 2D backbone + 2D ASPP + 5 task heads (2D->3D).

    The network takes concatenated gravity and magnetic observation data as
    input (2D observation surfaces at 81x81 resolution) and produces five 3D
    subsurface prediction outputs (40x40x20).
    """

    def __init__(self,
                 in_channels: int = 2,
                 backbone_channels: int = 64,
                 aspp_out_channels: int = 40,
                 out_depth: int = 20,
                 leaky_slope: float = 0.01) -> None:
        """
        Args:
            in_channels:       Number of input channels (2: gravity + magnetic).
            backbone_channels: Base channel count for U-Net encoder layer 1.
            aspp_out_channels: Number of output channels from ASPP (also input
                               to each task head).
            out_depth:         Depth dimension for 3D output (task heads expand
                               2D features to this depth).
            leaky_slope:       Negative slope for Leaky-ReLU in task heads.
        """
        super().__init__()

        # ---- Shared backbone: 2D U-Net feature extractor ----
        self.backbone = UNet2DBackbone(in_channels=in_channels)
        # Backbone output: (B, backbone_channels=64, 40, 40)

        # ---- ASPP: multi-scale feature aggregation (2D) ----
        self.aspp = ASPP2d(
            in_channels=backbone_channels,
            out_channels=aspp_out_channels,
        )
        # ASPP output: (B, aspp_out_channels=40, 40, 40)

        # ---- Task-specific heads (2D->3D expansion) ----
        self.task_heads = TaskHeads(
            in_channels=aspp_out_channels,
            out_depth=out_depth,
            negative_slope=leaky_slope,
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through the complete network.

        Args:
            x: Input tensor of shape (B, 2, 81, 81) where the 2 channels are
               [gravity_observation, magnetic_observation] on an 81x81 surface grid.

        Returns:
            Dict with keys 'task1'..'task5', each value is a tensor of shape
            (B, 1, 40, 40, 20) — 3D subsurface predictions.
        """
        # Shared feature extraction (2D)
        features = self.backbone(x)          # (B, 64, 40, 40)
        aspp_out = self.aspp(features)       # (B, 40, 40, 40)

        # Task-specific predictions (each head expands 2D->3D)
        outputs = self.task_heads(aspp_out)   # dict of 5 tensors, each (B, 1, 40, 40, 20)

        return outputs

    def get_num_params(self) -> dict:
        """Return parameter counts for each component."""
        return {
            'backbone': sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
            'aspp': sum(p.numel() for p in self.aspp.parameters() if p.requires_grad),
            'task_heads': self.task_heads._num_params(),
            'total': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


# ---------------------------------------------------------------------------
# Quick smoke-test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = JointInversionNet(
        in_channels=2,
        backbone_channels=64,
        aspp_out_channels=40,
        out_depth=20,
        leaky_slope=0.01,
    ).to(device)

    params = model.get_num_params()
    print("Parameter counts:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    # Forward pass test — input is now 2D observation surface (81x81)
    x = torch.randn(2, 2, 81, 81, device=device)
    with torch.no_grad():
        outputs = model(x)

    print(f"\nInput shape:  {x.shape}")
    for key, out in outputs.items():
        print(f"  {key}: {out.shape}")

    # Verify all output shapes
    for key, out in outputs.items():
        assert out.shape == (2, 1, 40, 40, 20), \
            f"{key} unexpected shape: {out.shape}"

    print("\nJointInversionNet smoke test PASSED.")
