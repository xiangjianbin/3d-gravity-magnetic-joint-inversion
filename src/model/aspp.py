"""
Atrous Spatial Pyramid Pooling (ASPP) Module -- 2D Version.

For 2D feature maps from U-Net backbone. Uses dilated convolutions
to capture multi-scale features at different receptive field sizes.

Based on Fang et al., IEEE TGRS Vol.63, 2025 (Fig.4).

Rates: [6, 12, 18, 24] + Global Average Pooling (5 branches total).
Each branch outputs 40 channels -> concat (200) -> 1x1 fusion conv -> out_channels.
Activation: LeakyReLU(0.01) throughout (paper Eq.8).

Input:  (B, C_in, H, W)
Output: (B, out_channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv2d(nn.Module):
    """Single ASPP branch: 1x1 projection + dilated 2D convolution."""

    def __init__(self, in_channels: int, out_channels: int, rate: int,
                 negative_slope: float = 0.01):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )
        self.dilated = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dilated(self.proj(x))


class ASPPPool2d(nn.Module):
    """Global Average Pooling branch (Branch 5)."""

    def __init__(self, in_channels: int, out_channels: int,
                 negative_slope: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        x = F.adaptive_avg_pool2d(x, 1)   # (B, C, 1, 1)
        # Use eval mode for BN in this branch to avoid batch_size=1 error
        # (global pool output has spatial size 1x1, BN needs >1 sample/channel)
        is_training = self.training
        if is_training:
            self.conv.eval()
        x = self.conv(x)
        if is_training:
            self.conv.train()
        return F.interpolate(x, size=(h, w), mode='bilinear',
                             align_corners=False)


class ASPP2d(nn.Module):
    """
    2D Atrous Spatial Pyramid Pooling module.

    5 branches: 4 dilated conv (rates 6,12,18,24) + global avg pool
    Concat -> 1x1 fusion conv -> out_channels.

    Args:
        in_channels:   Number of input channels (from backbone output).
        out_channels:  Number of output channels (default 40 per paper Fig.4).
        negative_slope: Negative slope for LeakyReLU (default 0.01).
    """

    RATES = [6, 12, 18, 24]
    BRANCH_OUT = 40

    def __init__(self, in_channels: int, out_channels: int = 40,
                 negative_slope: float = 0.01):
        super().__init__()
        self.branches = nn.ModuleList([
            ASPPConv2d(in_channels, self.BRANCH_OUT, rate=r,
                       negative_slope=negative_slope)
            for r in self.RATES
        ])
        self.global_pool = ASPPPool2d(in_channels, self.BRANCH_OUT,
                                      negative_slope=negative_slope)

        fused = self.BRANCH_OUT * len(self.RATES) + self.BRANCH_OUT  # 200
        self.fusion = nn.Sequential(
            nn.Conv2d(fused, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) feature map from backbone.
        Returns:
            (B, out_channels, H, W) fused multi-scale features.
        """
        branches_out = [b(x) for b in self.branches]
        branches_out.append(self.global_pool(x))
        cat = torch.cat(branches_out, dim=1)  # (B, 200, H, W)
        return self.fusion(cat)                # (B, out_channels, H, W)


# ---------------------------------------------------------------------------
# Quick smoke-test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aspp = ASPP2d(in_channels=64, out_channels=40).to(device)
    n_params = sum(p.numel() for p in aspp.parameters())
    print(f"ASPP params: {n_params:,}")

    x = torch.randn(2, 64, 40, 40, device=device)
    with torch.no_grad():
        out = aspp(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (2, 40, 40, 40)

    # Check no NaN/Inf
    assert not torch.isnan(out).any(), "NaN in ASPP output"
    assert not torch.isinf(out).any(), "Inf in ASPP output"

    print("PASSED")
