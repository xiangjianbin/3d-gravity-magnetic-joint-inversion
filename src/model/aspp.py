"""
Atrous Spatial Pyramid Pooling (ASPP) Module for 3D Feature Maps.

Reproduced from:
  Fang et al., "Improved 3-D Joint Inversion of Gravity and Magnetic Data
  Based on Deep Learning With a Multitask Learning Strategy",
  IEEE TGRS, Vol. 63, 2025

Architecture (from paper Section II-A, Fig.4):
  - Input:  feature map from U-Net backbone  (B, C_in, D, H, W)
  - 5 parallel branches:
      Branch 1-4: Conv3d(1x1x1)->BN->ReLU -> DilatedConv3d(3x3x3, rate=r)->BN->ReLU
                  rates = [6, 12, 18, 24], each outputs 40 channels
      Branch 5:   Global Average Pooling -> Conv3d(1x1x1)->BN->ReLU -> Upsample -> 40 channels
  - Concatenate all 5 branches: 40 * 5 = 200 channels
  - Fusion: Conv3d(1x1x1, 200->40) -> BN -> ReLU
  - Output: (B, 40, D, H, W)  -- multi-scale features for task heads

Physical meaning (paper):
  ASPP captures multi-scale subsurface structural features through dilated
  convolutions with different receptive fields, enabling the network to
  simultaneously identify small-scale local anomalies and large-scale regional
  structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv3d(nn.Module):
    """Single ASPP branch: optional 1x1 projection + dilated 3x3 convolution.

    When rate=1 this reduces to a standard 3x3 convolution.
    For rate > 1 the dilation expands the effective receptive field without
    increasing parameter count.
    """

    def __init__(self, in_channels: int, out_channels: int, rate: int) -> None:
        """
        Args:
            in_channels:  Number of input channels.
            out_channels: Number of output channels per branch (40 per paper).
            rate:         Dilation rate for the 3x3x3 conv (6, 12, 18, or 24).
        """
        super().__init__()
        # 1x1x1 projection to reduce/adjust channel dimension first
        self.project = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Dilated 3x3x3 convolution with same padding (= rate)
        self.dilated_conv = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=rate,
                dilation=rate,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        x = self.dilated_conv(x)
        return x


class ASPPPooling(nn.Module):
    """Global Average Pooling branch (Branch 5 of ASPP).

    Applies global average pooling over spatial dimensions (DxHxW),
    then uses a 1x1x1 conv to produce `out_channels` feature maps,
    and bilinearly upsamples back to original spatial size.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_size = x.shape[2:]  # (D, H, W)
        x = F.adaptive_avg_pool3d(x, 1)   # (B, C_in, 1, 1, 1)
        x = self.conv(x)                   # (B, out_c, 1, 1, 1)
        x = F.interpolate(x, size=spatial_size, mode='trilinear', align_corners=False)
        return x


class ASPP3d(nn.Module):
    """Atrous Spatial Pyramid Pooling module for 3D feature maps.

    Structure (per paper Fig.4):
        Input (B, C_in, D, H, W)
          |
          +---> [ASPPConv r=6]  ---> (B, 40, D, H, W)
          +---> [ASPPConv r=12] ---> (B, 40, D, H, W)
          +---> [ASPPConv r=18] ---> (B, 40, D, H, W)
          +---> [ASPPConv r=24] ---> (B, 40, D, H, W)
          +---> [ASPPPool]       ---> (B, 40, D, H, W)
          |
          v  concat along channel dim  --> (B, 200, D, H, W)
          |
          +---> Conv3d(1x1x1, 200->40) -> BN -> ReLU
          |
          v
        Output (B, 40, D, H, W)

    Note on input spatial size:
        The largest dilation rate is 24.  For same padding to work correctly,
        every spatial dimension must be >= 25.  The backbone output is
        (40, 40, 20), which satisfies this constraint for Easting/Northing
        but NOT for Depth (20 < 25).  In practice PyTorch's same-padding
        with dilation still produces an output; the effective receptive field
        simply extends beyond the input boundary (zero-padded).  This matches
        common DeepLabv3+ implementations.
    """

    # Dilation rates from paper Fig.4
    RATES = [6, 12, 18, 24]
    # Output channels per branch (from paper Fig.4)
    BRANCH_OUT_CHANNELS = 40

    def __init__(self, in_channels: int, out_channels: int = 40) -> None:
        """
        Args:
            in_channels:  Number of input channels (e.g., 64 from backbone).
            out_channels: Final output channels after fusion (default 40).
        """
        super().__init__()

        # Branches 1-4: dilated convolutions at different rates
        self.branches = nn.ModuleList([
            ASPPConv3d(in_channels, self.BRANCH_OUT_CHANNELS, rate=r)
            for r in self.RATES
        ])

        # Branch 5: global average pooling
        self.global_pool = ASPPPooling(in_channels, self.BRANCH_OUT_CHANNELS)

        # Fusion layer: 1x1x1 conv to merge concatenated features
        fused_channels = self.BRANCH_OUT_CHANNELS * len(self.RATES) + self.BRANCH_OUT_CHANNELS  # 200
        self.fusion = nn.Sequential(
            nn.Conv3d(fused_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map from backbone, shape (B, C_in, D, H, W).

        Returns:
            Multi-scale aggregated features, shape (B, out_channels, D, H, W).
        """
        branch_outputs = [branch(x) for branch in self.branches]
        branch_outputs.append(self.global_pool(x))

        # Concatenate along channel dimension: (B, 200, D, H, W)
        cat = torch.cat(branch_outputs, dim=1)

        # Fuse: (B, 200, D, H, W) -> (B, out_channels, D, H, W)
        return self.fusion(cat)

    def _num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick smoke-test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test standalone ASPP
    aspp = ASPP3d(in_channels=64, out_channels=40).to(device)
    total_params = aspp._num_params()
    print(f"ASPP3d total parameters: {total_params:,}")
    print(f"ASPP:\n{aspp}")

    x = torch.randn(2, 64, 40, 40, 20, device=device)
    with torch.no_grad():
        out = aspp(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 40, 40, 40, 20), f"Unexpected output shape: {out.shape}"
    print("Standalone ASPP smoke test PASSED.")

    # Test combined Backbone + ASPP
    from backbone_unet3d import UNet3DBackbone
    backbone = UNet3DBackbone(in_channels=2).to(device)
    aspp2 = ASPP3d(in_channels=64, out_channels=40).to(device)

    x2 = torch.randn(2, 2, 40, 40, 20, device=device)
    with torch.no_grad():
        feat = backbone(x2)
        out2 = aspp2(feat)
    print(f"\n--- Combined test ---")
    print(f"Backbone output: {feat.shape}")
    print(f"ASPP output:     {out2.shape}")
    assert out2.shape == (2, 40, 40, 40, 20)
    print("Combined Backbone+ASPP smoke test PASSED.")
