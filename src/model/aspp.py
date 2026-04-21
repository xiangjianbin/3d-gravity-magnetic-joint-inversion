"""
Atrous Spatial Pyramid Pooling (ASPP) Module — 2D Version.

For 2D feature maps from U-Net backbone. Uses dilated convolutions
to capture multi-scale features at different receptive field sizes.

Rates: [6, 12, 18, 24] + Global Average Pooling.
Output: (B, 40, H, W) fused multi-scale features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv2d(nn.Module):
    """Single ASPP branch: 1x1 projection + dilated 2D convolution."""
    def __init__(self, in_channels: int, out_channels: int, rate: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dilated = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                     stride=1, padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.dilated(self.proj(x))


class ASPPPool2d(nn.Module):
    """Global Average Pooling branch (Branch 5)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x = F.adaptive_avg_pool2d(x, 1)   # (B, C, 1, 1)
        x = self.conv(x)
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)


class ASPP2d(nn.Module):
    """
    2D Atrous Spatial Pyramid Pooling module.

    Input:  (B, C_in, H, W)
    Output: (B, 40, H, W)

    5 branches: 4 dilated conv (rates 6,12,18,24) + global avg pool
    Concat -> 1x1 fusion conv -> 40 channels.
    """

    RATES = [6, 12, 18, 24]
    BRANCH_OUT = 40

    def __init__(self, in_channels: int, out_channels: int = 40):
        super().__init__()
        self.branches = nn.ModuleList([
            ASPPConv2d(in_channels, self.BRANCH_OUT, rate=r) for r in self.RATES
        ])
        self.global_pool = ASPPPool2d(in_channels, self.BRANCH_OUT)

        fused = self.BRANCH_OUT * len(self.RATES) + self.BRANCH_OUT  # 200
        self.fusion = nn.Sequential(
            nn.Conv2d(fused, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [b(x) for b in self.branches]
        branches.append(self.global_pool(x))
        cat = torch.cat(branches, dim=1)  # (B, 200, H, W)
        return self.fusion(cat)  # (B, 40, H, W)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aspp = ASPP2d(in_channels=64, out_channels=40).to(device)
    print(f"ASPP params: {sum(p.numel() for p in aspp.parameters())}")

    x = torch.randn(2, 64, 40, 40, device=device)
    with torch.no_grad():
        out = aspp(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (2, 40, 40, 40)
    print("PASSED")
