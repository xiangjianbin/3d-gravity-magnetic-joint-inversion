"""
Tests for Task Heads and Joint Inversion Network.

Validates:
  - TaskHead forward pass shapes (with/without sigmoid)
  - TaskHeads container: all 5 heads produce correct output
  - JointInversionNet end-to-end forward pass
  - Full pipeline: backbone -> ASPP -> task heads dimension chain
  - Loss function computation
"""

import sys
import os
import pytest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from src.model.task_heads import TaskHead, TaskHeads
from src.model.joint_inversion_net import JointInversionNet
from src.model.loss_functions import JointInversionLoss


class TestTaskHead:
    """Test suite for individual TaskHead."""

    def test_regression_head_shape(self):
        """Regression head output should be (B, 1, D, H, W)."""
        head = TaskHead(in_channels=40, use_sigmoid=False)
        x = torch.randn(2, 40, 8, 8, 4)
        out = head(x)
        assert out.shape == (2, 1, 8, 8, 4)

    def test_classification_head_range(self):
        """Classification head with sigmoid should output in [0, 1]."""
        head = TaskHead(in_channels=40, use_sigmoid=True)
        x = torch.randn(2, 40, 8, 8, 4)
        out = head(x)
        assert out.min() >= 0.0, f"Min={out.min()}"
        assert out.max() <= 1.0, f"Max={out.max()}"

    def test_gradient_flow(self):
        """Gradients should flow through head."""
        head = TaskHead(in_channels=40, use_sigmoid=False)
        x = torch.randn(2, 40, 8, 8, 4, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient on input"


class TestTaskHeads:
    """Test suite for TaskHeads container."""

    def test_all_heads_output_shapes(self):
        """All 5 heads should produce (B, 1, D, H, W)."""
        heads = TaskHeads(in_channels=40)
        x = torch.randn(2, 40, 10, 10, 5)
        outputs = heads(x)

        expected_keys = {'task1', 'task2', 'task3', 'task4', 'task5'}
        assert set(outputs.keys()) == expected_keys, \
            f"Missing keys: {expected_keys - set(outputs.keys())}"

        for key, out in outputs.items():
            assert out.shape == (2, 1, 10, 10, 5), \
                f"{key} shape {out.shape} != (2,1,10,10,5)"

    def test_task3_sigmoid_range(self):
        """Task 3 (structural similarity) should be in [0, 1]."""
        heads = TaskHeads(in_channels=40)
        x = torch.randn(2, 40, 10, 10, 5)
        outputs = heads(x)

        t3 = outputs['task3']
        assert t3.min() >= 0.0, f"task3 min={t3.min()}"
        assert t3.max() <= 1.0, f"task3 max={t3.max()}"

    def test_param_count_positive(self):
        """Total parameter count should be positive."""
        heads = TaskHeads(in_channels=40)
        n = heads._num_params()
        assert n > 0, f"Zero parameters"


class TestJointInversionNet:
    """Test suite for the full JointInversionNet."""

    def test_end_to_end_shape(self):
        """Full network: (B,2,40,40,20) -> 5x(B,1,40,40,20)."""
        device = torch.device("cpu")
        model = JointInversionNet()
        x = torch.randn(2, 2, 40, 40, 20)
        outputs = model(x)

        for key in ['task1', 'task2', 'task3', 'task4', 'task5']:
            assert key in outputs, f"Missing output: {key}"
            assert outputs[key].shape == (2, 1, 40, 40, 20), \
                f"{key} shape: {outputs[key].shape}"

    def test_task3_sigmoid_in_full_net(self):
        """Task 3 output from full net should be in [0, 1]."""
        model = JointInversionNet()
        x = torch.randn(2, 2, 40, 40, 20)
        outputs = model(x)

        t3 = outputs['task3']
        assert t3.min() >= 0.0
        assert t3.max() <= 1.0

    def test_full_network_gradients(self):
        """Gradients should flow through entire network."""
        model = JointInversionNet()
        x = torch.randn(2, 2, 40, 40, 20)
        outputs = model(x)

        # Sum all outputs to create scalar loss
        total = sum(o.sum() for o in outputs.values())
        total.backward()

        # Check backbone gradients
        for name, param in model.backbone.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad in backbone.{name}"

        # Check ASPP gradients
        for name, param in model.aspp.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad in aspp.{name}"

        # Check head gradients
        for i, head in enumerate(model.task_heads.heads, 1):
            for name, param in head.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"No grad in task{i}.{name}"

    def test_get_num_params(self):
        """get_num_params should return dict with positive values."""
        model = JointInversionNet()
        params = model.get_num_params()
        assert 'backbone' in params
        assert 'aspp' in params
        assert 'task_heads' in params
        assert 'total' in params
        assert params['total'] > 0
        # Total should equal sum of parts
        assert params['total'] == params['backbone'] + params['aspp'] + params['task_heads']


class TestLossFunctions:
    """Test suite for loss functions."""

    def test_mse_loss_computation(self):
        """MSE loss should compute correctly for regression tasks."""
        criterion = JointInversionLoss(leaky_relu_lambda=0.0)

        preds = {
            'task1': torch.zeros(2, 1, 4, 4, 2),
            'task2': torch.zeros(2, 1, 4, 4, 2),
            'task3': torch.ones(2, 1, 4, 4, 2) * 0.5,
            'task4': torch.zeros(2, 1, 4, 4, 2),
            'task5': torch.zeros(2, 1, 4, 4, 2),
        }
        targets = {
            'density': torch.ones(2, 1, 4, 4, 2),
            'susceptibility': torch.ones(2, 1, 4, 4, 2),
            'structural_sim': torch.ones(2, 1, 4, 4, 2),
        }

        per_task, total = criterion(preds, targets)

        # MSE of zeros vs ones = 1.0
        assert abs(per_task['task1'] - 1.0) < 0.01, \
            f"MSE task1: {per_task['task1']}"

        assert total.dim() == 0, "Total loss should be scalar"
        assert total.item() > 0, "Total loss should be positive"

    def test_bce_loss_for_task3(self):
        """Task 3 BCE loss should handle binary targets."""
        criterion = JointInversionLoss(leaky_relu_lambda=0.0)

        preds = {
            'task1': torch.rand(2, 1, 4, 4, 2),
            'task2': torch.rand(2, 1, 4, 4, 2),
            'task3': torch.sigmoid(torch.randn(2, 1, 4, 4, 2)),
            'task4': torch.rand(2, 1, 4, 4, 2),
            'task5': torch.rand(2, 1, 4, 4, 2),
        }
        targets = {
            'density': torch.rand(2, 1, 4, 4, 2),
            'susceptibility': torch.rand(2, 1, 4, 4, 2),
            'structural_sim': (torch.rand(2, 1, 4, 4, 2) > 0.5).float(),
        }

        per_task, total = criterion(preds, targets)
        assert per_task['task3'] > 0, "BCE loss should be positive"
        assert not torch.isnan(total), "Total loss is NaN"

    def test_loss_backward(self):
        """Loss backward should work through model."""
        model = JointInversionNet()
        criterion = JointInversionLoss(leaky_relu_lambda=0.0)

        x = torch.randn(2, 2, 40, 40, 20)
        preds = model(x)
        targets = {
            'density': torch.rand(2, 1, 40, 40, 20),
            'susceptibility': torch.rand(2, 1, 40, 40, 20),
            'structural_sim': (torch.rand(2, 1, 40, 40, 20) > 0.5).float(),
        }

        _, total_loss = criterion(preds, targets, model=model)
        total_loss.backward()

        # Verify at least some parameter has non-zero gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad, "No non-zero gradients found after backward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
