"""
5 任务头定义 — 多任务学习架构
=============================

包含论文中的全部 5 个任务模块:
- Task 1: 独立重力反演头 (TaskIndependentGravityHead)
- Task 2: 独立磁法反演头 (TaskIndependentMagneticHead)
- Task 3: 结构相似性提取模块 (TaskStructuralSimilarity)
- Task 4: 联合重力反演头 (TaskJointInversionHead)
- Task 5: 联合磁法反演头 (TaskJointInversionHead, 复用同一类)

参考论文:
- Fig.3: 独立反演子网络结构
- Fig.4: ASPP 模块位置
- Section 2.3: 多任务策略描述

数据流:
  Input(B,2,D,H,W) → Backbone → ASPP(256ch)
    ├→ Task1 → rho_pred (B,1,D,H,W)      [MSE loss]
    ├→ Task2 → kappa_pred (B,1,D,H,W)    [MSE loss]
    ↓
  [rho_pred, kappa_pred]
    ↓
  Task3 → structural_sim S (B,1,D,H,W)   [MSE loss, sigmoid→[0,1]]
    ↓
  [Input, S] → Task4 → rho_final (B,1,D,H,W)     [BCE loss]
  [Input, S] → Task5 → kappa_final (B,1,D,H,W)   [BCE loss]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 辅助组件
# =====================================================================

class _DecoderBlock(nn.Module):
    """
    单层解码器块: Conv3d → BN → LeakyReLU × 2

    用于 Task 1/2/4/5 的内部解码层。
    """

    def __init__(self, in_channels: int, out_channels: int,
                 leaky_slope: float = 0.01):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _MiniDecoder(nn.Module):
    """
    小型 4 层解码器，用于独立反演和联合反演头。

    每层: Conv3d×2 + BN + LeakyReLU
    内部通过通道递减实现空间特征精炼。
    """

    def __init__(
        self,
        in_channels: int,
        channel_list: list = None,
        leaky_slope: float = 0.01,
    ):
        """
        Args:
            in_channels: 输入通道数
            channel_list: 各层输出通道列表，长度=层数。
                          默认根据输入通道自动推断:
                          - 大输入 (>=128): [in//2, in//4, in//8, 1]
                          - 小输入 (<128):  [64, 32, 16, 1]
            leaky_slope: LeakyReLU 负斜率
        """
        super().__init__()

        if channel_list is None:
            if in_channels >= 128:
                # Task 1/2 路径: 256→128→64→32→1
                channel_list = [in_channels // 2, in_channels // 4,
                                in_channels // 8, 1]
            else:
                # Task 4/5 路径: 3→64→32→16→1
                channel_list = [64, 32, 16, 1]

        layers = []
        prev_ch = in_channels
        for out_ch in channel_list:
            layers.append(_DecoderBlock(prev_ch, out_ch, leaky_slope))
            prev_ch = out_ch

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


# =====================================================================
# Task 1: 独立重力反演头
# =====================================================================

class TaskIndependentGravityHead(nn.Module):
    """
    Task 1: 独立重力反演头。

    从 ASPP 特征中预测密度模型 M_ρ_pred。

    输入: ASPP 特征 (B, 256, D, H, W)，默认 (B, 256, 40, 40, 20)
    输出: 密度模型预测 (B, 1, D, H, W)

    结构: 4 层小型解码器
      - 通道: 256 → 128 → 64 → 32 → 1
      - 每层: Conv3d(3×3×3) → BN → LeakyReLU × 2
      - 无 Sigmoid（输出为连续值，用 MSE 监督）
    """

    def __init__(
        self,
        in_channels: int = 256,
        leaky_slope: float = 0.01,
    ):
        super().__init__()
        # 4 层解码器: 256→128→64→32→1
        self.decoder = _MiniDecoder(
            in_channels=in_channels,
            channel_list=[128, 64, 32, 1],
            leaky_slope=leaky_slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ASPP 特征 (B, 256, D, H, W)

        Returns:
            密度模型预测 (B, 1, D, H, W)
        """
        return self.decoder(x)


# =====================================================================
# Task 2: 独立磁法反演头
# =====================================================================

class TaskIndependentMagneticHead(nn.Module):
    """
    Task 2: 独立磁法反演头。

    从 ASPP 特征中预测磁化率模型 M_κ_pred。
    与 Task 1 结构完全相同，但参数独立（不共享权重）。

    输入: ASPP 特征 (B, 256, D, H, W)
    输出: 磁化率模型预测 (B, 1, D, H, W)

    结构: 同 Task 1 — 4 层小型解码器
      - 通道: 256 → 128 → 64 → 32 → 1
    """

    def __init__(
        self,
        in_channels: int = 256,
        leaky_slope: float = 0.01,
    ):
        super().__init__()
        # 与 Task 1 相同结构，独立参数
        self.decoder = _MiniDecoder(
            in_channels=in_channels,
            channel_list=[128, 64, 32, 1],
            leaky_slope=leaky_slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ASPP 特征 (B, 256, D, H, W)

        Returns:
            磁化率模型预测 (B, 1, D, H, W)
        """
        return self.decoder(x)


# =====================================================================
# Task 3: 结构相似性提取模块
# =====================================================================

class TaskStructuralSimilarity(nn.Module):
    """
    Task 3: 结构相似性提取模块。

    接收 Task 1 和 Task 2 的预测结果，提取密度与磁化率之间的
    结构一致性信息，生成二值相似图 S。

    物理含义:
      S[i,j,k] ≈ 1 表示该位置密度和磁化率同时非零（地质体一致）
      S[i,j,k] ≈ 0 表示该位置只有一种物理属性异常（结构不一致）

    输入: [M_ρ_pred, M_κ_pred] 各 (B, 1, D, H, W)
    输出: 结构相似图 S (B, 1, D, H, W), 值域 [0, 1] (Sigmoid)

    结构 (论文描述 + Gap 6 的推断实现):
      1. 分别对两个输入做 Conv3d(1→32, k=3) + LeakyReLU
      2. 拼接: (B, 64, D, H, W)
      3. MaxPool3d(2) × 2 次 → 下采样 4 倍
      4. "径向网格操作" → 用 Conv3d 替代 (Gap 6): Conv3d(64→32, k=3)
      5. 上采样回原始尺寸: F.interpolate(trilinear)
      6. Conv3d(32→1, k=1) + Sigmoid → [0, 1]
    """

    def __init__(self, leaky_slope: float = 0.01):
        super().__init__()

        # 步骤 1: 分别处理两个输入
        self.rho_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )
        self.kappa_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

        # 步骤 3: MaxPool3d(2) × 2
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # 步骤 4: 径向网格操作替代 — 标准卷积
        self.radial_grid = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

        # 步骤 6: 最终输出头
        self.output_head = nn.Sequential(
            nn.Conv3d(32, 1, kernel_size=1),
            nn.Sigmoid(),  # 输出值域 [0, 1]
        )

    def forward(
        self,
        rho_pred: torch.Tensor,
        kappa_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rho_pred: Task 1 输出的密度预测 (B, 1, D, H, W)
            kappa_pred: Task 2 输出的磁化率预测 (B, 1, D, H, W)

        Returns:
            结构相似图 S (B, 1, D, H, W)，值域 [0, 1]
        """
        # 记录原始空间尺寸用于后续上采样
        orig_size = rho_pred.shape[2:]

        # 步骤 1: 分别卷积编码
        rho_feat = self.rho_conv(rho_pred)       # (B, 32, D, H, W)
        kappa_feat = self.kappa_conv(kappa_pred)  # (B, 32, D, H, W)

        # 步骤 2: 拼接
        combined = torch.cat([rho_feat, kappa_feat], dim=1)  # (B, 64, D, H, W)

        # 步骤 3: MaxPool3d(2) × 2 次 (下采样 4 倍)
        pooled = self.pool(combined)             # (B, 64, D/2, H/2, W/2)
        pooled = self.pool(pooled)               # (B, 64, D/4, H/4, W/4)

        # 步骤 4: 径向网格操作替代
        radial_out = self.radial_grid(pooled)    # (B, 32, D/4, H/4, W/4)

        # 步骤 5: 上采样回原始尺寸
        upsampled = F.interpolate(
            radial_out, size=orig_size, mode='trilinear', align_corners=False
        )                                         # (B, 32, D, H, W)

        # 步骤 6: 输出头 + Sigmoid
        S = self.output_head(upsampled)           # (B, 1, D, H, W), [0, 1]

        return S


# =====================================================================
# Task 4 & 5: 联合反演头
# =====================================================================

class TaskJointInversionHead(nn.Module):
    """
    Task 4 (联合重力反演) / Task 5 (联合磁法反演) 共享类。

    利用原始观测数据和结构相似图 S 进行联合反演，
    得到最终的高质量预测结果。

    输入: 原始数据 (B, 2, D, H, W) + 结构相似图 S (B, 1, D, H, W)
         → concat → (B, 3, D, H, W)
    输出: 最终预测 (B, 1, D, H, W)

    结构: 4 层小型解码器
      - 第一层输入是 3 通道（原始 2ch + S 1ch），而非 Task 1/2 的 256 通道
      - 通道: 3 → 64 → 32 → 16 → 1
      - 无 Sigmoid（损失函数使用 BCEWithLogitsLoss）

    注意:
      - 输出不加 Sigmoid，由 BCEWithLogitsLoss 处理（数值更稳定）
      - Task 4 和 Task 5 使用此类各自的实例（参数独立）
    """

    def __init__(self, leaky_slope: float = 0.01):
        super().__init__()
        # 4 层解码器: 3→64→32→16→1
        self.decoder = _MiniDecoder(
            in_channels=3,
            channel_list=[64, 32, 16, 1],
            leaky_slope=leaky_slope,
        )

    def forward(
        self,
        input_data: torch.Tensor,
        structural_sim: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_data: 原始观测数据 (B, 2, D, H, W) — 重力+磁异常
            structural_sim: Task 3 输出的结构相似图 (B, 1, D, H, W)

        Returns:
            最终预测 (B, 1, D, H, W)，未经过 Sigmoid
        """
        # 拼接原始数据和结构相似图
        combined = torch.cat([input_data, structural_sim], dim=1)  # (B, 3, D, H, W)
        return self.decoder(combined)
