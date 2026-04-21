"""
完整多任务联合反演网络 — 网络组装
==================================

将 Backbone U-Net、ASPP、5 个任务头组装为完整的端到端网络。

数据流:
  Input (B, 2, D, H, W)
    ↓
  Backbone UNet → (B, 256, D, H, W)
    ↓
  ASPP → (B, 256, D, H, W)
    ├→ Task1 (IndependentGravity)   → rho_pred     (B, 1, D, H, W)
    ├→ Task2 (IndependentMagnetic)  → kappa_pred   (B, 1, D, H, W)
    ↓
  [rho_pred, kappa_pred]
    ↓
  Task3 (StructuralSimilarity)      → S            (B, 1, D, H, W), 值域[0,1]
    ↓
  [Input, S] → Task4 (JointGravity)       → rho_final    (B, 1, D, H, W)
  [Input, S] → Task5 (JointMagnetic)      → kappa_final  (B, 1, D, H, W)

训练模式: 返回所有 5 个任务的输出
推理模式: 只返回 Task 4 + Task 5 的输出
"""

import torch
import torch.nn as nn

from .backbone_unet3d import BackboneUNet3d
from .aspp import ASPP3d
from .task_heads import (
    TaskIndependentGravityHead,
    TaskIndependentMagneticHead,
    TaskStructuralSimilarity,
    TaskJointInversionHead,
)


class JointInversionNet(nn.Module):
    """
    完整的多任务联合反演网络。

    组装顺序:
      Input (B,2,D,H,W)
        ↓
      Backbone UNet → (B,256,D,H,W)
        ↓
      ASPP → (B,256,D,H,W)
        ├→ Task1 (IndependentGravity) → rho_pred (B,1,D,H,W)
        ├→ Task2 (IndependentMagnetic) → kappa_pred (B,1,D,H,W)
        ↓
      [rho_pred, kappa_pred]
        ↓
      Task3 (StructuralSimilarity) → S (B,1,D,H,W)
        ↓
      [Input, S] → Task4 (JointGravity) → rho_final (B,1,D,H,W)
      [Input, S] → Task5 (JointMagnetic) → kappa_final (B,1,D,H,W)

    Args:
        in_channels: 输入通道数（重力+磁异常），默认 2
        aspp_in_channels: ASPP 输入通道数，默认 256（与 backbone 输出一致）
        aspp_out_channels: ASPP 输出通道数，默认 256
        leaky_slope: LeakyReLU 负斜率，默认 0.01
        use_gradient_checkpointing: 是否启用梯度检查点以节省显存

    训练模式 (self.training=True 或 return_all=True):
        返回包含 5 个任务输出的字典

    推理模式 (self.training=False 且 return_all=False):
        只返回 Task 4 + Task 5 的最终预测结果
    """

    def __init__(
        self,
        in_channels: int = 2,
        aspp_in_channels: int = 256,
        aspp_out_channels: int = 256,
        leaky_slope: float = 0.01,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # ===== 骨干网络 =====
        self.backbone = BackboneUNet3d(
            in_channels=in_channels,
            out_channels=aspp_in_channels,
            use_checkpoint=use_gradient_checkpointing,
        )

        # ===== ASPP 模块 =====
        self.aspp = ASPP3d(
            in_channels=aspp_in_channels,
            out_channels=aspp_out_channels,
        )

        # ===== 任务头 =====
        # Task 1: 独立重力反演
        self.task1 = TaskIndependentGravityHead(
            in_channels=aspp_out_channels,
            leaky_slope=leaky_slope,
        )

        # Task 2: 独立磁法反演
        self.task2 = TaskIndependentMagneticHead(
            in_channels=aspp_out_channels,
            leaky_slope=leaky_slope,
        )

        # Task 3: 结构相似性提取
        self.task3 = TaskStructuralSimilarity(leaky_slope=leaky_slope)

        # Task 4: 联合重力反演
        self.task4 = TaskJointInversionHead(leaky_slope=leaky_slope)

        # Task 5: 联合磁法反演
        self.task5 = TaskJointInversionHead(leaky_slope=leaky_slope)

    def forward(self, x: torch.Tensor, return_all: bool = False) -> dict:
        """
        前向传播。

        Args:
            x: 输入张量 (B, 2, D, H, W)，包含归一化的重力和磁异常数据
            return_all: 是否返回全部 5 个任务的输出。
                        True 或训练模式下返回全部；
                        False 且推理模式下只返回 Task 4/5。

        Returns:
            dict，包含以下键:
              - 'rho_pred':      Task 1 输出 (B, 1, D, H, W)
              - 'kappa_pred':    Task 2 输出 (B, 1, D, H, W)
              - 'structural_sim': Task 3 输出 (B, 1, D, H, W), 值域[0,1]
              - 'rho_final':     Task 4 输出 (B, 1, D, H, W)
              - 'kappa_final':   Task 5 输出 (B, 1, D, H, W)
        """
        # === 阶段 1: 特征提取 ===
        backbone_feat = self.backbone(x)          # (B, 256, D, H, W)
        aspp_feat = self.aspp(backbone_feat)      # (B, 256, D, H, W)

        # === 阶段 2: 独立反演 (Task 1 & 2) ===
        rho_pred = self.task1(aspp_feat)           # (B, 1, D, H, W)
        kappa_pred = self.task2(aspp_feat)         # (B, 1, D, H, W)

        # === 阶段 3: 结构相似性提取 (Task 3) ===
        structural_sim = self.task3(rho_pred, kappa_pred)  # (B, 1, D, H, W), [0,1]

        # === 阶段 4: 联合反演 (Task 4 & 5) ===
        rho_final = self.task4(x, structural_sim)           # (B, 1, D, H, W)
        kappa_final = self.task5(x, structural_sim)         # (B, 1, D, H, W)

        # 根据模式决定返回内容
        if return_all or self.training:
            return {
                'rho_pred': rho_pred,
                'kappa_pred': kappa_pred,
                'structural_sim': structural_sim,
                'rho_final': rho_final,
                'kappa_final': kappa_final,
            }
        else:
            # 推理模式: 只返回最终预测
            return {
                'rho_final': rho_final,
                'kappa_final': kappa_final,
            }

    def get_param_summary(self) -> dict:
        """
        统计各模块的参数量。

        Returns:
            dict 包含各模块参数量和总参数量
        """
        summary = {}
        total_params = 0

        for name, module in [
            ('backbone', self.backbone),
            ('aspp', self.aspp),
            ('task1_independent_gravity', self.task1),
            ('task2_independent_magnetic', self.task2),
            ('task3_structural_similarity', self.task3),
            ('task4_joint_gravity', self.task4),
            ('task5_joint_magnetic', self.task5),
        ]:
            n_params = sum(p.numel() for p in module.parameters())
            n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            summary[name] = {
                'total': n_params,
                'trainable': n_trainable,
            }
            total_params += n_params

        summary['TOTAL'] = {'total': total_params, 'trainable': total_params}
        return summary
