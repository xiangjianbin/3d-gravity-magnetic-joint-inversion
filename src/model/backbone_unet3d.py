"""
3D U-Net Backbone for Gravity-Magnetic Joint Inversion.

Reproduced from:
  Fang et al., "Improved 3-D Joint Inversion of Gravity and Magnetic Data
  Based on Deep Learning With a Multitask Learning Strategy",
  IEEE TGRS, Vol. 63, 2025

Architecture (from paper Section II-A, Fig.2):
  - Input:  (batch, 2, 40, 40, 20)  -- [gravity, magnetic] x Easting x Northing x Depth
  - Encoder: 4 levels, each = Conv3d->BN->ReLU -> Conv3d->BN->ReLU -> MaxPool3d(2)
  - Decoder: 4 levels, each = UpSample(2) -> Concat(skip) -> Conv3d->BN->ReLU -> Conv3d->BN->ReLU
  - Channels: 64 -> 128 -> 256 -> 512 (bottleneck) -> 256 -> 128 -> 64
  - All Conv3d: kernel_size=3, padding=1
  - Activation: ReLU in encoder/decoder (Leaky-ReLU used as regularizer elsewhere)
"""

import torch
import torch.nn as nn


class DoubleConv3d(nn.Module):
    """Two consecutive (Conv3d -> BatchNorm3d -> ReLU) blocks.

    This is the standard U-Net "double convolution" building block.
    Each conv uses kernel_size=3, padding=1 to preserve spatial dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3DBackbone(nn.Module):
    """3D U-Net backbone for feature extraction from gravity+magnetic input.

    The network follows the classic encoder-decoder architecture with skip
    connections.  It takes a 5-D tensor of shape (B, 2, 40, 40, 20) and
    produces a feature map of shape (B, 64, 40, 40, 20) that is fed into
    the ASPP module and subsequently to task-specific heads.

    Channel progression per level:
        Enc1:   2  -> 64     spatial: 40x40x20
        Enc2:  64 -> 128    spatial: 20x20x10
        Enc3: 128 -> 256    spatial: 10x10x5
        Enc4: 256 -> 512    spatial: 5x5x2  (bottleneck)
        Dec1:  512+512=1024 -> 256  spatial: 10x10x5
        Dec2:  256+256=512  -> 128  spatial: 20x20x10
        Dec3:  128+128=256  -> 64   spatial: 40x40x20
        Dec4:   64+64=128   -> 64   spatial: 40x40x20
    """

    # Channel counts at each encoder level
    ENCODER_CHANNELS = [64, 128, 256, 512]
    # Initial input channels (gravity + magnetic)
    IN_CHANNELS = 2

    def __init__(self, in_channels: int = 2) -> None:
        """
        Args:
            in_channels: Number of input channels (default 2 for gravity + magnetic).
        """
        super().__init__()

        # ---- Encoder (down-sampling path) ----
        self.enc1 = DoubleConv3d(in_channels, self.ENCODER_CHANNELS[0])   # 2  -> 64
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)                # 40x40x20 -> 20x20x10

        self.enc2 = DoubleConv3d(self.ENCODER_CHANNELS[0], self.ENCODER_CHANNELS[1])  # 64 -> 128
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)                # 20x20x10 -> 10x10x5

        self.enc3 = DoubleConv3d(self.ENCODER_CHANNELS[1], self.ENCODER_CHANNELS[2])  # 128 -> 256
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)                # 10x10x5  -> 5x5x2 (or 5x5x3 with floor)

        self.enc4 = DoubleConv3d(self.ENCODER_CHANNELS[2], self.ENCODER_CHANNELS[3])  # 256 -> 512 (bottleneck)

        # ---- Decoder (up-sampling path) ----
        # UpSample via trilinear interpolation (mode='trilinear' for 5D tensors)
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec1 = DoubleConv3d(self.ENCODER_CHANNELS[3] + self.ENCODER_CHANNELS[2],
                                  self.ENCODER_CHANNELS[2])  # 512+256=768 -> 256

        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec2 = DoubleConv3d(self.ENCODER_CHANNELS[2] + self.ENCODER_CHANNELS[1],
                                  self.ENCODER_CHANNELS[1])  # 256+128=384 -> 128

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec3 = DoubleConv3d(self.ENCODER_CHANNELS[1] + self.ENCODER_CHANNELS[0],
                                  self.ENCODER_CHANNELS[0])  # 128+64=192  -> 64

        # Note: dec3 output already has spatial size 40x40x20 (same as input).
        # No further upsampling is needed.  The final DoubleConv refines features
        # at the original resolution before passing to ASPP.

    def _ensure_spatial_match(self, src: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Interpolate *src* to match *target_shape* if sizes differ."""
        if src.shape[2:] == target_shape:
            return src
        return nn.functional.interpolate(
            src, size=target_shape, mode='trilinear', align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 3D U-Net backbone.

        Args:
            x: Input tensor of shape (batch, 2, 40, 40, 20).

        Returns:
            Feature map of shape (batch, 64, 40, 40, 20), ready for ASPP.
        """
        # --- Encoder ---
        e1 = self.enc1(x)       # (B, 64, 40, 40, 20)
        e2 = self.enc2(self.pool1(e1))  # (B, 128, 20, 20, 10)
        e3 = self.enc3(self.pool2(e2))  # (B, 256, 10, 10, 5)
        e4 = self.enc4(self.pool3(e3))  # (B, 512, 5, 5, 2) or (B, 512, 5, 5, 3)

        # --- Decoder with skip connections ---
        # Dec level 1: upsample bottleneck -> concat enc3 skip
        d1 = self._ensure_spatial_match(self.up4(e4), e3.shape[2:])
        d1 = self.dec1(torch.cat([d1, e3], dim=1))  # (B, 256, 10, 10, 5)

        # Dec level 2: upsample -> concat enc2 skip
        d2 = self._ensure_spatial_match(self.up3(d1), e2.shape[2:])
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, 128, 20, 20, 10)

        # Dec level 3: upsample -> concat enc1 skip  (recovers full resolution)
        d3 = self._ensure_spatial_match(self.up2(d2), e1.shape[2:])
        d3 = self.dec3(torch.cat([d3, e1], dim=1))  # (B, 64, 40, 40, 20)

        return d3

    def _num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick smoke-test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3DBackbone(in_channels=2).to(device)

    # Print summary
    total_params = model._num_params()
    print(f"UNet3DBackbone total parameters: {total_params:,}")
    print(f"Model:\n{model}")

    # Forward pass test
    x = torch.randn(2, 2, 40, 40, 20, device=device)  # batch=2
    with torch.no_grad():
        out = model(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 64, 40, 40, 20), f"Unexpected output shape: {out.shape}"
    print("Smoke test PASSED.")
