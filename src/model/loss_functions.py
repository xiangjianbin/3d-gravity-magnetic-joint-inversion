"""
Loss Functions for Gravity-Magnetic Joint Inversion Network.

Implements:
  - MSELoss for regression tasks (Tasks 1, 2, 4, 5)
  - BCELoss for classification task (Task 3: structural similarity)
  - Leaky-ReLU regularizer (L2 penalty on weights)
  - Multi-task combined loss (weighted sum of all task losses)

Reproduced from:
  Fang et al., "Improved 3-D Joint Inversion of Gravity and Magnetic Data
  Based on Deep Learning With a Multitask Learning Strategy",
  IEEE TGRS, Vol. 63, 2025

Paper equations:
  (9) L_MSE = (1/N) * sum((m_i - Net(d_i; theta))^2)
  (10) L_BCE = -(1/N) * sum[omega*m_i*ln(Net) + (1-m_i)*ln(1-Net)]
  (8)  Leaky-ReLU(x) = x if x>=0 else nu*x   (nu ~ 0.01)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointInversionLoss(nn.Module):
    """Combined loss for all 5 tasks in the joint inversion network.

    The paper states that each task is trained independently, but since they
    share the backbone parameters, we compute all losses simultaneously and
    backpropagate the weighted sum.  This is the standard multi-task learning
    approach and matches the practical interpretation of "independent training"
    (each task has its own loss function and head, but parameter updates are
    shared through the backbone).

    Loss configuration:
      Task 1 (independent gravity):     MSE, weight=1.0
      Task 2 (independent magnetic):    MSE, weight=1.0
      Task 3 (structural similarity):  BCE with logits, weight=1.0
      Task 4 (joint gravity):           MSE, weight=1.0
      Task 5 (joint magnetic):          MSE, weight=1.0

    Optional Leaky-ReLU regularization: adds L2 penalty on weights to prevent
    overfitting (paper mentions this as a regularizer).
    """

    def __init__(self,
                 task_weights = None,
                 leaky_relu_lambda: float = 0.0,
                 bce_pos_weight: float = 1.0) -> None:
        """
        Args:
            task_weights:       Dict mapping task name to loss weight.
                                Default: all 1.0.
            leaky_relu_lambda:  Coefficient for L2 weight penalty (Leaky-ReLU
                                regularizer).  0 disables it.
            bce_pos_weight:     Positive sample weight for BCE (omega in eq.10).
        """
        super().__init__()

        # Per-task loss weights (default equal weight)
        self.task_weights = task_weights or {
            'task1': 1.0,  # independent gravity
            'task2': 1.0,  # independent magnetic
            'task3': 1.0,  # structural similarity
            'task4': 1.0,  # joint gravity
            'task5': 1.0,  # joint magnetic
        }

        # Loss functions
        self.mse_loss = nn.MSELoss()
        # Use BCEWithLogitsLoss for numerical stability (internal sigmoid, AMP-safe)
        self.bce_pos_weight = bce_pos_weight

        # Regularization strength
        self.leaky_relu_lambda = leaky_relu_lambda

    def forward(self, predictions: dict, targets: dict,
                model = None) -> tuple:
        """
        Compute per-task losses and total weighted loss.

        Args:
            predictions: Dict of model outputs {'task1'..'task5': tensor(B,1,D,H,W)}.
            targets:     Dict of ground truth {
                            'density':          (B,1,D,H,W),
                            'susceptibility':   (B,1,D,H,W),
                            'structural_sim':   (B,1,D,H,W)
                        }.
            model:       Optional model reference for computing weight regularization.

        Returns:
            Tuple of (per_task_losses dict, total_loss scalar tensor).
            per_task_losses keys: 'task1'..'task5', 'total', 'regularization'
        """
        device = next(iter(predictions.values())).device
        per_task = {}

        # --- Task 1: Independent gravity inversion (MSE) ---
        pred_t1 = predictions['task1']   # (B,1,D,H,W)
        tgt_density = targets['density']
        loss_t1 = self.mse_loss(pred_t1, tgt_density)
        per_task['task1'] = loss_t1

        # --- Task 2: Independent magnetic inversion (MSE) ---
        pred_t2 = predictions['task2']
        tgt_suscept = targets['susceptibility']
        loss_t2 = self.mse_loss(pred_t2, tgt_suscept)
        per_task['task2'] = loss_t2

        # --- Task 3: Structural similarity (BCE with logits) ---
        # task3 head outputs raw logits; BCEWithLogitsLoss is AMP-safe.
        pred_t3 = predictions['task3']   # raw logits (no sigmoid in head)
        tgt_struct = targets['structural_sim']
        bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.bce_pos_weight], device=device)
        )
        loss_t3 = bce_loss(pred_t3, tgt_struct)
        per_task['task3'] = loss_t3

        # --- Task 4: Joint gravity inversion (MSE) ---
        pred_t4 = predictions['task4']
        loss_t4 = self.mse_loss(pred_t4, tgt_density)
        per_task['task4'] = loss_t4

        # --- Task 5: Joint magnetic inversion (MSE) ---
        pred_t5 = predictions['task5']
        loss_t5 = self.mse_loss(pred_t5, tgt_suscept)
        per_task['task5'] = loss_t5

        # --- Weighted total loss ---
        total = sum(
            self.task_weights.get(f'task{i}', 1.0) * per_task[f'task{i}']
            for i in range(1, 6)
        )
        per_task['total'] = total.detach().item() if total.requires_grad else total.item()

        # --- Leaky-ReLU regularization (L2 penalty on weights) ---
        reg_loss = torch.tensor(0.0, device=device)
        if self.leaky_relu_lambda > 0.0 and model is not None:
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            reg_loss = self.leaky_relu_lambda * l2_norm
            total = total + reg_loss
        per_task['regularization'] = reg_loss.detach().item()

        return per_task, total


def get_criterion(task_weights=None,
                  leaky_relu_lambda: float = 1e-5) -> JointInversionLoss:
    """Factory: create a standard JointInversionLoss instance.

    Args:
        task_weights:       Optional per-task loss weights.
        leaky_relu_lambda:  L2 regularization coefficient.

    Returns:
        Configured JointInversionLoss module.
    """
    return JointInversionLoss(
        task_weights=task_weights,
        leaky_relu_lambda=leaky_relu_lambda,
        bce_pos_weight=1.0,
    )


# ---------------------------------------------------------------------------
# Quick smoke-test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = JointInversionLoss(leaky_relu_lambda=1e-5)

    # Create dummy predictions and targets
    B = 2
    preds = {
        'task1': torch.randn(B, 1, 40, 40, 20, device=device),   # independent grav
        'task2': torch.randn(B, 1, 40, 40, 20, device=device),   # independent mag
        'task3': torch.sigmoid(torch.randn(B, 1, 40, 40, 20, device=device)),  # struct sim
        'task4': torch.randn(B, 1, 40, 40, 20, device=device),   # joint grav
        'task5': torch.randn(B, 1, 40, 40, 20, device=device),   # joint mag
    }
    tgts = {
        'density':        torch.rand(B, 1, 40, 40, 20, device=device),
        'susceptibility': torch.rand(B, 1, 40, 40, 20, device=device),
        'structural_sim': (torch.rand(B, 1, 40, 40, 20, device=device) > 0.5).float(),
    }

    per_task, total_loss = criterion(preds, tgts, model=None)

    print("Per-task losses:")
    for key, val in per_task.items():
        print(f"  {key}: {val:.6f}")
    print(f"\nTotal loss: {total_loss.item():.6f}")

    assert total_loss.dim() == 0, "Total loss should be a scalar"
    print("\nSmoke test PASSED.")
