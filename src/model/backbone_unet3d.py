"""
3D U-Net 骨干网络 — 论文 Fig.2 严格实现

论文: Improved 3-D Joint Inversion of Gravity and Magnetic Data Based on
       Deep Learning With a Multitask Learning Strategy
       (IEEE TGRS, Vol.63, 2025)

架构说明:
    - 输入: (B, 2, 40, 40, 20) — [重力异常 Δg, 磁异常 ΔT]
    - 编码器: 4层下采样 (64→128→256→512→1024), MaxPool3d(2)
    - 瓶颈层: Conv3d(512→1024) @ [2,2,1]
    - 解码器: 4层上采样(F.interpolate trilinear) + Skip Connection
    - 输出: (B, out_channels=256, 40, 40, 20) — 送入 ASPP 的特征

关键设计决策 (参考 docs/ASSUMPTIONS_AND_GAPS.md):
    - Gap 1: 解码器采用对称通道 (1024→512→256→128→64→256)
    - Gap 2: LeakyReLU negative_slope=0.01
    - 上采样使用 F.interpolate(trilinear) 而非 ConvTranspose3d (避免棋盘格)
    - 支持 gradient checkpointing 节省显存
    - 兼容 torch.cuda.amp.autocast() 混合精度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint


class ConvBlock3d(nn.Module):
    """
    基础卷积块: Conv3d + BatchNorm3d + LeakyReLU

    对应论文 Fig.2 中每个编码/解码层的卷积单元。
    kernel_size=3, stride=1, padding=1 保持空间尺寸不变。
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # Gap 2: gamma=0.01
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DoubleConvBlock3d(nn.Module):
    """
    双卷积块: 两个连续的 ConvBlock3d (标准 U-Net 每层结构)

    论文 Fig.2 中每层编码/解码实际包含两层卷积，
    这与原始 U-Net 设计一致。用于增强特征提取能力。
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock3d(in_ch, out_ch),
            ConvBlock3d(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder3d(nn.Module):
    """
    编码器: 4 层下采样路径

    论文 Fig.2 编码器部分:
        Level 1: in_ch → 64,   空间 [40,40,20] → MaxPool → [20,20,10]
        Level 2: 64   → 128,  空间 [20,20,10] → MaxPool → [10,10,5]
        Level 3: 128  → 256,  空间 [10,10,5]  → MaxPool → [5,5,2]
        Level 4: 256  → 512,  空间 [5,5,2]    → MaxPool → [2,2,1]

    每层输出 skip feature 供解码器使用。
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 64):
        super().__init__()
        ch = base_channels
        # 每层用双卷积增强表达能力 (与原版 U-Net 一致)
        self.enc1 = DoubleConvBlock3d(in_channels, ch)       # 2  → 64
        self.enc2 = DoubleConvBlock3d(ch, ch * 2)             # 64 → 128
        self.enc3 = DoubleConvBlock3d(ch * 2, ch * 4)         # 128 → 256
        self.enc4 = DoubleConvBlock3d(ch * 4, ch * 8)         # 256 → 512
        self.pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            skips: list[Tensor] — 各层 skip feature (enc1~enc4 的池化前输出)
            bottleneck_input: Tensor — 编码器最终输出, shape (B, 512, 2, 2, 1)
        """
        # Level 1: [B, 2, 40, 40, 20] → [B, 64, 40, 40, 20]
        e1 = self.enc1(x)
        s1 = e1  # skip: pool 前的特征
        e1 = self.pool(e1)  # → [B, 64, 20, 20, 10]

        # Level 2: [B, 64, 20, 20, 10] → [B, 128, 20, 20, 10]
        e2 = self.enc2(e1)
        s2 = e2
        e2 = self.pool(e2)  # → [B, 128, 10, 10, 5]

        # Level 3: [B, 128, 10, 10, 5] → [B, 256, 10, 10, 5]
        e3 = self.enc3(e2)
        s3 = e3
        e3 = self.pool(e3)  # → [B, 256, 5, 5, 2]

        # Level 4: [B, 256, 5, 5, 2] → [B, 512, 5, 5, 2]
        e4 = self.enc4(e3)
        s4 = e4
        e4 = self.pool(e4)  # → [B, 512, 2, 2, 1]

        return [s1, s2, s3, s4], e4


class Decoder3d(nn.Module):
    """
    解码器: 4 层上采样 + Skip Connection

    论文 Fig.2 解码器部分 (对称结构, Gap 1):
        Level 1': 1024+512=1536 → 512, 上采样到 s4 尺寸 [5,5,2]
        Level 2': 512+256=768  → 256, 上采样到 s3 尺寸 [10,10,5]
        Level 3': 256+128=384  → 128, 上采样到 s2 尺寸 [20,20,10]
        Level 4': 128+64=192   → 64,  上采样到 s1 尺寸 [40,40,20]

    上采样方式: F.interpolate(mode='trilinear', align_corners=False)
    Skip 方式:   torch.cat([skip, up], dim=1)
    """

    def __init__(self, base_channels: int = 64, out_channels: int = 256):
        super().__init__()
        ch = base_channels
        # 解码通道: 512, 256, 128, 64 → 最终 out_channels
        # 输入通道 = 上采样输出通道(bottleneck或上一dec输出) + skip通道
        self.dec1 = DoubleConvBlock3d(ch * 16 + ch * 8, ch * 8)   # 1024+512 → 512
        self.dec2 = DoubleConvBlock3d(ch * 8 + ch * 4, ch * 4)    # 512+256  → 256
        self.dec3 = DoubleConvBlock3d(ch * 4 + ch * 2, ch * 2)    # 256+128  → 128
        self.dec4 = DoubleConvBlock3d(ch * 2 + ch, ch)            # 128+64   → 64
        # 最终投影: 64 → out_channels (ASPP 需要 256 通道输入)
        self.final_conv = nn.Sequential(
            nn.Conv3d(ch, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def _upsample(self, x: torch.Tensor, target_size) -> torch.Tensor:
        """三线性插值上采样到目标尺寸"""
        return F.interpolate(
            x, size=target_size, mode='trilinear', align_corners=False
        )

    def forward(self, bottleneck: torch.Tensor, skips: list) -> torch.Tensor:
        """
        Args:
            bottleneck: 瓶颈层输出 (B, 1024, 2, 2, 1)
            skips: 编码器各层 skip feature [s1, s2, s3, s4]

        Returns:
            decoded: (B, out_channels, 40, 40, 20)
        """
        s1, s2, s3, s4 = skips

        # Decoder Level 1': bottleneck(1024) + skip4(512) → 512
        up1 = self._upsample(bottleneck, s4.shape[2:])  # → [5,5,2]
        cat1 = torch.cat([s4, up1], dim=1)              # → (B, 1536, 5,5,2)
        d1 = self.dec1(cat1)                             # → (B, 512, 5,5,2)

        # Decoder Level 2': d1(512) + skip3(256) → 256
        up2 = self._upsample(d1, s3.shape[2:])           # → [10,10,5]
        cat2 = torch.cat([s3, up2], dim=1)               # → (B, 768, 10,10,5)
        d2 = self.dec2(cat2)                             # → (B, 256, 10,10,5)

        # Decoder Level 3': d2(256) + skip2(128) → 128
        up3 = self._upsample(d2, s2.shape[2:])           # → [20,20,10]
        cat3 = torch.cat([s2, up3], dim=1)               # → (B, 384, 20,20,10)
        d3 = self.dec3(cat3)                             # → (B, 128, 20,20,10)

        # Decoder Level 4': d3(128) + skip1(64) → 64
        up4 = self._upsample(d3, s1.shape[2:])           # → [40,40,20]
        cat4 = torch.cat([s1, up4], dim=1)               # → (B, 192, 40,40,20)
        d4 = self.dec4(cat4)                             # → (B, 64, 40,40,20)

        # 最终投影到 ASPP 所需通道数
        out = self.final_conv(d4)                        # → (B, 256, 40,40,20)
        return out


class BackboneUNet3d(nn.Module):
    """
    完整的 3D U-Net 骨干网络 (论文 Fig.2)

    输入: x shape (B, 2, 40, 40, 20) — [重力异常, 磁异常]
    输出: features shape (B, 256, 40, 40, 20) — ASPP 之前的特征

    架构流程:
        Input(B,2,40,40,20)
          → Encoder(4层DoubleConv + MaxPool) → (B,512,2,2,1)
          → Bottleneck(DoubleConv 512→1024)  → (B,1024,2,2,1)
          → Decoder(4层Upsample+SkipCat+DoubleConv) → (B,256,40,40,20)

    显存优化:
        - use_checkpoint=True 时启用 gradient checkpointing
        - 兼容 torch.cuda.amp.autocast() 混合精度训练
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        out_channels: int = 256,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        # 编码器: 2 → 64 → 128 → 256 → 512
        self.encoder = Encoder3d(in_channels, base_channels)

        # 瓶颈层: 512 → 1024 (论文明确给出)
        self.bottleneck = DoubleConvBlock3d(
            base_channels * 8, base_channels * 16  # 512 → 1024
        )

        # 解码器: 对称上采样, 最终输出 out_channels
        self.decoder = Decoder3d(base_channels, out_channels)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """Kaiming 正交初始化 — 适合 LeakyReLU"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu'
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _checkpointed_forward(self, x: torch.Tensor) -> torch.Tensor:
        """使用 gradient checkpointing 节省显存的 forward"""

        def _run_encoder(enc, inp):
            return enc(inp)

        def _run_bottleneck(bn, inp):
            return bn(inp)

        skips, enc_out = checkpoint.checkpoint(
            _run_encoder, self.encoder, x, use_reentrant=False
        )
        bott = checkpoint.checkpoint(
            _run_bottleneck, self.bottleneck, enc_out, use_reentrant=False
        )
        out = self.decoder(bott, skips)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量, shape (B, 2, 40, 40, 20)

        Returns:
            features: Backbone 输出特征, shape (B, out_channels, 40, 40, 20)
        """
        if self.use_checkpoint and self.training:
            return self._checkpointed_forward(x)

        # 标准 forward (无 checkpointing)
        skips, enc_out = self.encoder(x)     # 编码 + 收集 skip connections
        bott = self.bottleneck(enc_out)       # 瓶颈层
        features = self.decoder(bott, skips)  # 解码 + skip concatenation
        return features


# ============================================================
# 便捷函数
# ============================================================

def build_backbone(use_checkpoint: bool = False) -> BackboneUNet3d:
    """构建默认配置的 BackboneUNet3d"""
    model = BackboneUNet3d(
        in_channels=2,
        base_channels=64,
        out_channels=256,
        use_checkpoint=use_checkpoint,
    )
    return model


def count_parameters(model: nn.Module) -> dict:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
