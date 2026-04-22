"""
2D U-Net Backbone for Gravity-Magnetic Joint Inversion.

Takes 2D observation maps (gravity + magnetic anomalies on observation surface)
and produces feature maps for downstream ASPP and task heads.

Architecture based on Fang et al., IEEE TGRS Vol.63, 2025.
Uses 2D convolutions for encoder (processing observation data),
with decoder producing features for multi-task prediction.

Key design decisions vs paper:
  - Paper describes a 3D U-Net with input (2, 40, 40, 20). However the actual
    input data is 2D observation surfaces (81x81 gravity + magnetic grids).
    We use a 2D U-Net backbone that maps (B, 2, 81, 81) -> (B, 64, 40, 40),
    then task heads expand to (B, 1, 40, 40, 20). This is the "obs-to-subsurface"
    mapping approach established in phase4-fix.
  - Activation: LeakyReLU(0.01) throughout (paper specifies Leaky-ReLU regularizer).
  - Supports return_features=True for multi-scale output (needed by ASPP / task heads).

Input:  (B, 2, 81, 81) -- [gravity_obs, magnetic_obs] on 81x81 grid
Output: (B, 64, 40, 40) -- feature map for ASPP + task heads
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv2d(nn.Module):
    """Two consecutive (Conv2d -> BatchNorm2d -> LeakyReLU) blocks.

    Args:
        in_channels:   Number of input channels.
        out_channels:  Number of output channels.
        negative_slope: Negative slope for LeakyReLU (default 0.01 per paper).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 negative_slope: float = 0.01):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet2DBackbone(nn.Module):
    """
    2D U-Net backbone for gravity-magnetic joint inversion.

    Encoder: 4 layers of DoubleConv2d + MaxPool2d (downsampling).
    Decoder: 4 layers of Upsample + DoubleConv2d with skip connections (concat).

    Channel progression (encoder):
      Layer 1:  2  ->  64    spatial: 81x81
      Layer 2: 64 -> 128    spatial: 40x40
      Layer 3: 128-> 256    spatial: 20x20
      Layer 4: 256-> 512    spatial: 10x10  (bottleneck)

    Channel progression (decoder):
      Layer 1: 512+256=768 -> 256  spatial: 10x10 -> 20x20
      Layer 2: 256+128=384 -> 128  spatial: 20x20 -> 40x40
      Layer 3: 128+64 =192 ->  64  spatial: 40x40 -> 80x80
      Layer 4: 64          ->  64  spatial: 80x80 -> (crop to) 40x40

    Activation: LeakyReLU(0.01) everywhere (paper Eq.8).
    """

    ENCODER_CHANNELS = [64, 128, 256, 512]

    def __init__(self, in_channels: int = 2,
                 negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

        # ---- Encoder (4 layers) ----
        # Enc1: (B, 2, 81, 81) -> (B, 64, 81, 81)
        self.enc1 = DoubleConv2d(in_channels, self.ENCODER_CHANNELS[0],
                                 negative_slope)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 40x40

        # Enc2: (B, 64, 40, 40) -> (B, 128, 40, 40)
        self.enc2 = DoubleConv2d(self.ENCODER_CHANNELS[0],
                                 self.ENCODER_CHANNELS[1], negative_slope)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 20x20

        # Enc3: (B, 128, 20, 20) -> (B, 256, 20, 20)
        self.enc3 = DoubleConv2d(self.ENCODER_CHANNELS[1],
                                 self.ENCODER_CHANNELS[2], negative_slope)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 10x10

        # Enc4 (bottleneck): (B, 256, 10, 10) -> (B, 512, 10, 10)
        self.enc4 = DoubleConv2d(self.ENCODER_CHANNELS[2],
                                 self.ENCODER_CHANNELS[3], negative_slope)

        # ---- Decoder (4 layers) ----
        # Dec1: upsample(512) + skip(256) -> DoubleConv -> 256,  spatial 20x20
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=False)
        self.dec1 = DoubleConv2d(self.ENCODER_CHANNELS[3] + self.ENCODER_CHANNELS[2],
                                 self.ENCODER_CHANNELS[2], negative_slope)

        # Dec2: upsample(256) + skip(128) -> DoubleConv -> 128,  spatial 40x40
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=False)
        self.dec2 = DoubleConv2d(self.ENCODER_CHANNELS[2] + self.ENCODER_CHANNELS[1],
                                 self.ENCODER_CHANNELS[1], negative_slope)

        # Dec3: upsample(128) + skip(64)  -> DoubleConv -> 64,   spatial 80x80
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=False)
        self.dec3 = DoubleConv2d(self.ENCODER_CHANNELS[1] + self.ENCODER_CHANNELS[0],
                                 self.ENCODER_CHANNELS[0], negative_slope)

        # Dec4: upsample(64) -> DoubleConv -> 64,  spatial ~160x160 -> center crop to 40x40
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=False)
        self.dec4 = DoubleConv2d(self.ENCODER_CHANNELS[0],
                                 self.ENCODER_CHANNELS[0], negative_slope)

    def _upsample_match(self, x: torch.Tensor,
                        target_size: tuple) -> torch.Tensor:
        """Upsample x to match target_size (H, W) via bilinear interpolation."""
        if x.shape[2:] == target_size:
            return x
        return nn.functional.interpolate(x, size=target_size, mode='bilinear',
                                         align_corners=False)

    def forward(self, x: torch.Tensor,
                return_features: bool = False) -> torch.Tensor | dict:
        """
        Forward pass.

        Args:
            x: (B, 2, 81, 81) -- [gravity_obs, magnetic_obs].
            return_features: If True, return dict of multi-scale encoder/decoder
                             features instead of just the final output.

        Returns:
            If return_features=False: (B, 64, 40, 40) -- cropped feature map.
            If return_features=True: dict with keys:
                'enc1' .. 'enc4': encoder features at each scale
                'dec1' .. 'dec4': decoder features at each scale
                'out':           final output (same as non-features mode)
        """
        # === Encoder ===
        e1 = self.enc1(x)                          # (B,  64, 81, 81)
        e2 = self.enc2(self.pool1(e1))             # (B, 128, 40, 40)
        e3 = self.enc3(self.pool2(e2))             # (B, 256, 20, 20)
        e4 = self.enc4(self.pool3(e3))             # (B, 512, 10, 10)

        # === Decoder with skip connections (concat) ===
        d1_up = self._upsample_match(self.up4(e4), e3.shape[2:])  # (B, 256, 20, 20)
        d1 = self.dec1(torch.cat([d1_up, e3], dim=1))              # (B, 256, 20, 20)

        d2_up = self._upsample_match(self.up3(d1), e2.shape[2:])  # (B, 128, 40, 40)
        d2 = self.dec2(torch.cat([d2_up, e2], dim=1))              # (B, 128, 40, 40)

        d3_up = self._upsample_match(self.up2(d2), e1.shape[2:])  # (B,  64, 81, 81)
        d3 = self.dec3(torch.cat([d3_up, e1], dim=1))              # (B,  64, 81, 81)

        # Final decoder layer (no skip connection -- just refine)
        d4 = self.up1(d3)       # (B, 64, 162, 162)  (bilinear doubles 81->162)
        d4 = self.dec4(d4)      # (B, 64, 162, 162)

        # Center crop to target model grid size (40x40)
        out = self._center_crop(d4, (40, 40))

        if return_features:
            return {
                'enc1': e1, 'enc2': e2, 'enc3': e3, 'enc4': e4,
                'dec1': d1, 'dec2': d2, 'dec3': d3, 'dec4': d4,
                'out': out,
            }
        return out

    @staticmethod
    def _center_crop(x: torch.Tensor, size: tuple) -> torch.Tensor:
        """Center crop tensor x to (H, W)=size."""
        h, w = x.shape[2], x.shape[3]
        th, tw = size
        h_off = (h - th) // 2
        w_off = (w - tw) // 2
        return x[:, :, h_off:h_off + th, w_off:w_off + tw]

    def _num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick smoke-test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2DBackbone(in_channels=2).to(device)
    print(f"Parameters: {model._num_params():,}")

    # Standard forward pass
    x = torch.randn(2, 2, 81, 81, device=device)
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (2, 64, 40, 40), f"Shape mismatch: {out.shape}"

    # Multi-scale features forward pass
    with torch.no_grad():
        feats = model(x, return_features=True)
    print("Multi-scale features:")
    for k, v in feats.items():
        print(f"  {k}: {v.shape}")
    assert feats['out'].shape == (2, 64, 40, 40)

    print("ALL PASSED")
