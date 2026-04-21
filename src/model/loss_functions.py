"""
多任务损失函数组合
==================

实现论文 Eq.9 和 Eq.10 的多任务损失:

  Total_Loss = MSE(rho_pred, rho_gt)       # Task 1: 独立重力反演
            + MSE(kappa_pred, kappa_gt)     # Task 2: 独立磁法反演
            + MSE(S, S_gt)                   # Task 3: 结构相似性
            + BCE(rho_final, rho_gt)         # Task 4: 联合重力 (ω=1)
            + BCE(kappa_final, kappa_gt)     # Task 5: 联合磁法 (ω=1)

支持:
  - 固定权重版本（默认各任务权重=1.0）
  - 可学习权重版本（Uncertainty Weighting 风格）
  - 各任务损失的独立访问
"""

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数组合。

    Total_Loss = w1 * MSE(rho_pred, rho_gt)
               + w2 * MSE(kappa_pred, kappa_gt)
               + w3 * MSE(S, S_gt)
               + w4 * BCE(rho_final, rho_gt)
               + w5 * BCE(kappa_final, kappa_gt)

    关于 BCE 的说明:
      - Task 4/5 的输出头 (TaskJointInversionHead) 内部不加 Sigmoid
      - 本模块使用 BCEWithLogitsLoss（数值更稳定，内部包含 Sigmoid）
      - Task 3 的输出头 (TaskStructuralSimilarity) 内部已包含 Sigmoid
        因此 Task 3 使用标准 MSELoss

    Args:
        weights: 各任务权重字典，格式为:
                 {
                     'task1': 1.0,   # 独立重力反演 (MSE)
                     'task2': 1.0,   # 独立磁法反演 (MSE)
                     'task3': 1.0,   # 结构相似性 (MSE)
                     'task4': 1.0,   # 联合重力反演 (BCE)
                     'task5': 1.0,   # 联合磁法反演 (BCE)
                 }
                 默认全部为 1.0。
        learnable: 是否使用可学习的任务权重（基于对数方差）。
                   如果为 True，weights 参数被忽略，
                   改用 Kendall et al. (2018) 的不确定性加权方案。

    用法示例:
        >>> criterion = MultiTaskLoss()
        >>> predictions = model(x)          # JointInversionNet 的输出 dict
        >>> targets = {
        ...     'rho': rho_ground_truth,     # (B, 1, D, H, W)
        ...     'kappa': kappa_ground_truth, # (B, 1, D, H, W)
        ...     'sim': sim_ground_truth,     # (B, 1, D, H, W), 值域[0,1]
        ... }
        >>> total_loss, task_losses = criterion(predictions, targets)
        >>> total_loss.backward()
    """

    def __init__(
        self,
        weights: dict = None,
        learnable: bool = False,
    ):
        super().__init__()

        # 默认权重: 全部 1.0
        default_weights = {
            'task1': 1.0,
            'task2': 1.0,
            'task3': 1.0,
            'task4': 1.0,
            'task5': 1.0,
        }
        if weights is not None:
            default_weights.update(weights)

        self.fixed_weights = default_weights
        self.learnable = learnable

        # ===== 损失函数实例 =====
        # Task 1-3: MSE Loss
        self.mse_loss = nn.MSELoss()

        # Task 4-5: BCEWithLogitsLoss
        # 注意: 输入是 logits（未过 sigmoid），目标应在 [0, 1]
        self.bce_loss = nn.BCEWithLogitsLoss()

        # ===== 可学习权重参数 (可选) =====
        if learnable:
            # 基于 log(σ²) 的可学习权重，初始化为 0 → weight = exp(0) = 1.0
            # 使用 ParameterList 确保每个参数都是标量叶节点（方便梯度检查）
            self.log_vars = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0)) for _ in range(5)
            ])

    def forward(
        self,
        predictions: dict,
        targets: dict,
    ) -> tuple:
        """
        计算多任务总损失和各任务分项损失。

        Args:
            predictions: JointInversionNet.forward() 返回的字典，包含:
                - 'rho_pred'      (B, 1, D, H, W): Task 1 输出
                - 'kappa_pred'    (B, 1, D, H, W): Task 2 输出
                - 'structural_sim' (B, 1, D, H, W): Task 3 输出, [0,1]
                - 'rho_final'     (B, 1, D, H, W): Task 4 输出 (logits)
                - 'kappa_final'   (B, 1, D, H, W): Task 5 输出 (logits)
            targets: 真值字典，包含:
                - 'rho'  (B, 1, D, H, W): 密度真值
                - 'kappa'(B, 1, D, H, W): 磁化率真值
                - 'sim'  (B, 1, D, H, W): 结构相似性真值, 值域[0,1]

        Returns:
            (total_loss, task_losses):
              - total_loss: 标量张量，加权总损失
              - task_losses: dict，包含各任务的原始（未加权）损失值
        """
        # ===== 计算各任务原始损失 =====

        # 自动对齐 target 形状: (B,D,H,W) -> (B,1,D,H,W)
        def _align_target(pred, target):
            """如果 target 缺少通道维度，自动 unsqueeze"""
            if target.dim() == pred.dim() - 1:
                return target.unsqueeze(1)
            return target

        rho_t = _align_target(predictions['rho_pred'], targets['rho'])
        kappa_t = _align_target(predictions['kappa_pred'], targets['kappa'])
        sim_t = _align_target(predictions['structural_sim'], targets['sim'])

        # Task 1: 独立重力反演 — MSE
        loss_task1 = self.mse_loss(predictions['rho_pred'], rho_t)

        # Task 2: 独立磁法反演 — MSE
        loss_task2 = self.mse_loss(predictions['kappa_pred'], kappa_t)

        # Task 3: 结构相似性 — MSE
        # structural_sim 已经过 Sigmoid，直接与真值计算 MSE
        loss_task3 = self.mse_loss(predictions['structural_sim'], sim_t)

        # Task 4: 联合重力反演 — BCEWithLogits
        # rho_final 是 logits（未过 sigmoid），目标是密度真值（需在[0,1]）
        loss_task4 = self.bce_loss(predictions['rho_final'], rho_t)

        # Task 5: 联合磁法反演 — BCEWithLogits
        loss_task5 = self.bce_loss(predictions['kappa_final'], kappa_t)

        # 收集各任务损失
        task_losses = {
            'task1_gravity_mse': loss_task1.item() if loss_task1.requires_grad else loss_task1,
            'task2_magnetic_mse': loss_task2.item() if loss_task2.requires_grad else loss_task2,
            'task3_similarity_mse': loss_task3.item() if loss_task3.requires_grad else loss_task3,
            'task4_gravity_bce': loss_task4.item() if loss_task4.requires_grad else loss_task4,
            'task5_magnetic_bce': loss_task5.item() if loss_task5.requires_grad else loss_task5,
        }

        # ===== 计算加权总损失 =====
        if self.learnable and hasattr(self, 'log_vars'):
            # 不确定性加权: L_total = Σ (1/(2*σ_i²)) * L_i + log(σ_i²)
            lv0, lv1, lv2, lv3, lv4 = self.log_vars
            precision_1 = torch.exp(-lv0)
            precision_2 = torch.exp(-lv1)
            precision_3 = torch.exp(-lv2)
            precision_4 = torch.exp(-lv3)
            precision_5 = torch.exp(-lv4)

            total_loss = (
                precision_1 * loss_task1 + lv0 +
                precision_2 * loss_task2 + lv1 +
                precision_3 * loss_task3 + lv2 +
                precision_4 * loss_task4 + lv3 +
                precision_5 * loss_task5 + lv4
            )
        else:
            # 固定权重
            w = self.fixed_weights
            total_loss = (
                w['task1'] * loss_task1 +
                w['task2'] * loss_task2 +
                w['task3'] * loss_task3 +
                w['task4'] * loss_task4 +
                w['task5'] * loss_task5
            )

        return total_loss, task_losses


class MultiTaskLossWithGradNorm(nn.Module):
    """
    基于梯度归一化的多任务损失 (Chen et al., 2018 GradNorm)。

    动态调整任务权重，使各任务的梯度量级接近均衡。

    注意: 这是一个更高级的变体，通常固定权重的 MultiTaskLoss 已经足够。

    Args:
        alpha: 恢复超参数，控制权重恢复到均匀分布的速度 (默认 1.5)
    """

    def __init__(self, alpha: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.base_criterion = MultiTaskLoss()
        # 可学习权重，初始化为 1.0
        self.task_weights = nn.Parameter(torch.ones(5))

    def forward(self, predictions: dict, targets: dict) -> tuple:
        """
        与 MultiTaskLoss.forward 接口一致，但使用动态调整的权重。
        """
        # 先获取各任务原始损失
        _, raw_losses = self.base_criterion(predictions, targets)

        # 使用当前权重计算加权和
        # 这里简化处理: 实际 GradNorm 需要在反向传播后调整权重
        # 完整实现需要在训练循环中配合调用 adjust_weights()
        w = torch.sigmoid(self.task_weights)

        loss_task1 = self.base_criterion.mse_loss(predictions['rho_pred'], targets['rho'])
        loss_task2 = self.base_criterion.mse_loss(predictions['kappa_pred'], targets['kappa'])
        loss_task3 = self.base_criterion.mse_loss(predictions['structural_sim'], targets['sim'])
        loss_task4 = self.base_criterion.bce_loss(predictions['rho_final'], targets['rho'])
        loss_task5 = self.base_criterion.bce_loss(predictions['kappa_final'], targets['kappa'])

        total_loss = (
            w[0] * loss_task1 + w[1] * loss_task2 + w[2] * loss_task3 +
            w[3] * loss_task4 + w[4] * loss_task5
        )

        task_losses = {
            'task1_gravity_mse': loss_task1.item(),
            'task2_magnetic_mse': loss_task2.item(),
            'task3_similarity_mse': loss_task3.item(),
            'task4_gravity_bce': loss_task4.item(),
            'task5_magnetic_bce': loss_task5.item(),
        }

        return total_loss, task_losses
