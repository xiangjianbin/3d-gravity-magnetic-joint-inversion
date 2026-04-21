"""
2D U-Net Backbone for Gravity-Magnetic Joint Inversion.

Takes 2D observation maps (gravity + magnetic anomalies on 81×81 surface grid)
and produces feature maps for 3D model prediction (40×40×20 subsurface).

Architecture based on Fang et al., IEEE TGRS Vol.63, 2025.
Uses 2D convolutions for encoder (processing observation data),
with decoder producing features reshaped to 3D for task heads.

Input:  (B, 2, 81, 81) — [gravity_obs, magnetic_obs]
Output: (B, 64, 10, 10) — feature map (will be processed by ASPP + task heads)
"""

import torch
import torch.nn as nn


class DoubleConv2d(nn.Module):
    """Two consecutive (Conv2d -> BatchNorm2d -> ReLU) blocks."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet2DBackbone(nn.Module):
    """
    2D U-Net backbone for gravity-magnetic joint inversion.

    Encoder: 2D convolutions on (81, 81) observation grid.
    Decoder: Upsamples to recover spatial resolution.

    Channel progression:
        Enc: 2 -> 64 -> 128 -> 256 -> 512 (bottleneck)
        Dec: 512 -> 256 -> 128 -> 64

    Spatial progression (for 81x81 input):
        Enc: 81x81 -> 40x40 -> 20x20 -> 10x10
        Dec: 10x10 -> 20x20 -> 40x40 -> 80x80 -> 40x40 (crop to match model grid projection)
    """

    ENCODER_CHANNELS = [64, 128, 256, 512]

    def __init__(self, in_channels: int = 2):
        super().__init__()

        # Encoder (2D)
        self.enc1 = DoubleConv2d(in_channels, self.ENCODER_CHANNELS[0])   # 2 -> 64,   81x81
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                # -> 40x40

        self.enc2 = DoubleConv2d(self.ENCODER_CHANNELS[0], self.ENCODER_CHANNELS[1])  # 64->128,  40x40
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                # -> 20x20

        self.enc3 = DoubleConv2d(self.ENCODER_CHANNELS[1], self.ENCODER_CHANNELS[2])  # 128->256, 20x20
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)                # -> 10x10

        self.enc4 = DoubleConv2d(self.ENCODER_CHANNELS[2], self.ENCODER_CHANNELS[3])  # 256->512, 10x10 (bottleneck)

        # Decoder (2D)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = DoubleConv2d(self.ENCODER_CHANNELS[3] + self.ENCODER_CHANNELS[2],
                                  self.ENCODER_CHANNELS[2])  # 512+256->256

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = DoubleConv2d(self.ENCODER_CHANNELS[2] + self.ENCODER_CHANNELS[1],
                                  self.ENCODER_CHANNELS[1])  # 256+128->128

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = DoubleConv2d(self.ENCODER_CHANNELS[1] + self.ENCODER_CHANNELS[0],
                                  self.ENCODER_CHANNELS[0])  # 128+64->64

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Final conv: no skip connection at this level, just 64->64
        self.dec4 = DoubleConv2d(self.ENCODER_CHANNELS[0],
                                  self.ENCODER_CHANNELS[0])  # 64->64

        # Final center crop: 80x80 -> 40x40 (to match 40x40 model grid projection from 81x81 obs)
        # Using a convolution with appropriate padding/cropping, or just center crop

    def _upsample_match(self, x, target_size):
        """Upsample x to match target_size (H, W)."""
        if x.shape[2:] == target_size:
            return x
        return nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 2, 81, 81) — [gravity_obs, magnetic_obs]

        Returns:
            (B, 64, 40, 40) — feature map cropped to model grid size
        """
        # Encoder
        e1 = self.enc1(x)        # (B, 64, 81, 81)
        e2 = self.enc2(self.pool1(e1))  # (B, 128, 40, 40)
        e3 = self.enc3(self.pool2(e2))  # (B, 256, 20, 20)
        e4 = self.enc4(self.pool3(e3))  # (B, 512, 10, 10)

        # Decoder with skip connections
        d1 = self._upsample_match(self.up4(e4), e3.shape[2:])  # (B, 256, 20, 20)
        d1 = self.dec1(torch.cat([d1, e3], dim=1))

        d2 = self._upsample_match(self.up3(d1), e2.shape[2:])  # (B, 128, 40, 40)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d3 = self._upsample_match(self.up2(d2), e1.shape[2:])  # (B, 64, 81, 81)
        d3 = self.dec3(torch.cat([d3, e1], dim=1))

        # Final upsample to 80x80 then center crop to 40x40
        d4 = self.up1(d3)   # (B, 64, 162, 162) roughly (bilinear doubles 81->162)
        d4 = self.dec4(d4)     # (B, 64, 162, 162)

        # Center crop 162x162 -> 40x40 (crop 61 pixels from each side)
        # 162 - 40 = 122; 122/2 = 61
        crop_size = (40, 40)
        h_offset = (d4.shape[2] - crop_size[0]) // 2
        w_offset = (d4.shape[3] - crop_size[1]) // 2
        out = d4[:, :, h_offset:h_offset+crop_size[0], w_offset:w_offset+crop_size[1]]

        return out

    def _num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2DBackbone(in_channels=2).to(device)
    print(f"Parameters: {model._num_params():,}")

    x = torch.randn(2, 2, 81, 81, device=device)
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (2, 64, 40, 40), f"Shape mismatch: {out.shape}"
    print("PASSED")
