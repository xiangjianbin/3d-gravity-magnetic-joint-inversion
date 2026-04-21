"""
3D ASPP (Atrous Spatial Pyramid Pooling) 模块 — 论文 Fig.4 严格实现

论文: Improved 3-D Joint Inversion of Gravity and Magnetic Data Based on
       Deep Learning With a Multitask Learning Strategy
       (IEEE TGRS, Vol.63, 2025)

架构说明:
    - 位置: Backbone U-Net 输出之后, Task 1/2 头之前
    - 输入: (B, 256, 40, 40, 20) — Backbone 特征
    - 输出: (B, 256, 40, 40, 20) — 多尺度特征

    分支结构 (4 并行分支):
        Branch 1-3: Dilated Conv3d(k=3, dilation={6,12,18}) + BN + ReLU
        Branch 4:   Global Average Pooling → Conv1x1 → BN → ReLU → Upsample

    融合: Concat(4 branches) → Conv1x1(1024→256) + BN + ReLU

Gap 参考:
    - Gap 5: 每分支输出通道 256 (与 DeepLab v3+ 一致)
    - 论文 Fig.4 显示 dilation rates = {6, 12, 18} + global pooling
"""

import torch
import torch.nn as nn


class ASPPConv3d(nn.Module):
    """
    ASPP 单个空洞卷积分支: Conv3d(dilated) + BN + ReLU

    对应论文 Fig.4 中 dilation=6/12/18 的三个并行卷积分支。
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        padding = dilation  # 空洞卷积 padding = dilation 保持尺寸
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=padding,
                dilation=dilation, bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ASPPPooling3d(nn.Module):
    """
    ASPP 全局平均池化分支

    对应论文 Fig.4 中的 Image Pooling 分支:
        AdaptiveAvgPool3d(1) -> Conv1x1 -> BN -> ReLU -> Upsample(原始尺寸)

    作用: 捕获全局上下文信息。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)  # 全局池化到 1x1x1
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            # Note: No BatchNorm here — spatial size is 1×1×1 after global pool,
            # BN requires >1 value per channel (PyTorch limitation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """全局池化 -> 1x1 卷积 -> 上采样回原始空间尺寸"""
        spatial_size = x.shape[2:]
        pooled = self.pool(x)           # (B, C, 1, 1, 1)
        projected = self.conv(pooled)   # (B, out_ch, 1, 1, 1)
        # 上采样回输入的空间尺寸
        upsampled = nn.functional.interpolate(
            projected, size=spatial_size,
            mode='trilinear', align_corners=False,
        )
        return upsampled


class ASPP3d(nn.Module):
    """
    3D Atrous Spatial Pyramid Pooling 模块 (论文 Fig.4)

    通过多尺度空洞卷积捕获不同感受野的特征，增强网络对多尺度地质体的
    表达能力。

    Args:
        in_channels:   输入通道数 (Backbone 输出, 默认 256)
        out_channels:  输出通道数 (每分支及最终输出, 默认 256)
        dilation_rates: 空洞卷积膨胀率列表, 默认 [6, 12, 18]

    输入形状:  (B, in_channels, D, H, W)
    输出形状:  (B, out_channels, D, H, W)

    内部流程:
        input
          |-> ASPPConv(d=6)   --|
          |-> ASPPConv(d=12)  -+|-> concat -> Conv1x1 -> output
          |-> ASPPConv(d=18)  -+|
          |-> ASPPPool(global)-|
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        dilation_rates: list = None,
    ):
        super().__init__()

        if dilation_rates is None:
            dilation_rates = [6, 12, 18]  # 论文 Fig.4 明确给出

        # 3 个不同 dilation rate 的空洞卷积分支
        # 对应论文 Fig.4 的三个并行 dilated convolution 路径
        self.branches = nn.ModuleList([
            ASPPConv3d(in_channels, out_channels, d)
            for d in dilation_rates
        ])

        # 全局平均池化分支 (第 4 个分支)
        # 对应论文 Fig.4 的 global average pooling 路径
        self.global_pool = ASPPPooling3d(in_channels, out_channels)

        # 融合层: 将 4 个分支拼接后用 1x1 卷积降维
        # 4 * out_channels -> out_channels
        num_branches = len(dilation_rates) + 1  # 3 dilated + 1 pooling = 4
        self.fusion = nn.Sequential(
            nn.Conv3d(
                out_channels * num_branches, out_channels,
                kernel_size=1, bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Backbone 输出特征, shape (B, in_channels, D, H, W)

        Returns:
            多尺度融合特征, shape (B, out_channels, D, H, W)
        """
        # 各分支并行计算
        branch_outputs = [branch(x) for branch in self.branches]
        pool_output = self.global_pool(x)
        branch_outputs.append(pool_output)

        # 拼接所有分支: (B, 4*out_channels, D, H, W)
        concatenated = torch.cat(branch_outputs, dim=1)

        # 1x1 卷积融合降维: (B, out_channels, D, H, W)
        out = self.fusion(concatenated)
        return out


# ============================================================
# 便捷函数
# ============================================================

def build_aspp(in_channels: int = 256, out_channels: int = 256) -> ASPP3d:
    """构建默认配置的 ASPP3d"""
    model = ASPP3d(
        in_channels=in_channels,
        out_channels=out_channels,
        dilation_rates=[6, 12, 18],
    )
    return model
