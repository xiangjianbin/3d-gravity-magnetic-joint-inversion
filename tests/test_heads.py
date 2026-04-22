"""
Tests for Task Heads and Joint Inversion Network (2D version).

Validates:
  - TaskHead forward pass shapes (2D input -> 3D output via expansion)
  - TaskHeads container: all 5 heads produce correct output
  - JointInversionNet end-to-end forward pass (2D obs -> 3D pred)
  - Full pipeline: backbone -> ASPP -> task heads dimension chain
  - Loss function computation (MSE, BCEWithLogitsLoss, Leaky-ReLU regularizer)
  - AMP compatibility
  - Resume checkpoint loading
"""

import sys
import os
import pytest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from src.model.task_heads import TaskHead, TaskHeads
from src.model.joint_inversion_net import JointInversionNet
from src.model.loss_functions import JointInversionLoss, get_criterion


class TestTaskHead:
    """Test suite for individual TaskHead."""

    def test_regression_head_shape(self):
        """Regression head: 2D (B,40,H,W) -> 3D (B,1,D,H,W)."""
        head = TaskHead(in_channels=40, out_depth=10, use_sigmoid=False)
        x = torch.randn(2, 40, 8, 8)
        out = head(x)
        assert out.shape == (2, 1, 8, 8, 10)

    def test_classification_head_logits(self):
        """Classification head outputs raw logits (sigmoid applied in loss)."""
        head = TaskHead(in_channels=40, out_depth=5, use_sigmoid=True)
        x = torch.randn(2, 40, 8, 8)
        out = head(x)  # raw logits, unbounded
        assert out.shape == (2, 1, 8, 8, 5)

    def test_gradient_flow(self):
        """Gradients should flow through head."""
        head = TaskHead(in_channels=40, out_depth=6, use_sigmoid=False)
        x = torch.randn(2, 40, 8, 8, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient on input"


class TestTaskHeads:
    """Test suite for TaskHeads container."""

    def test_all_heads_output_shapes(self):
        """All 5 heads should produce (B, 1, D, H, W)."""
        heads = TaskHeads(in_channels=40, out_depth=5)
        x = torch.randn(2, 40, 10, 10)
        outputs = heads(x)

        expected_keys = {'task1', 'task2', 'task3', 'task4', 'task5'}
        assert set(outputs.keys()) == expected_keys, \
            f"Missing keys: {expected_keys - set(outputs.keys())}"

        for key, out in outputs.items():
            assert out.shape == (2, 1, 10, 10, 5), \
                f"{key} shape {out.shape} != (2,1,10,10,5)"

    def test_task3_logits_range(self):
        """Task 3 (structural similarity) outputs raw logits."""
        heads = TaskHeads(in_channels=40, out_depth=5)
        x = torch.randn(2, 40, 10, 10)
        outputs = heads(x)

        t3 = outputs['task3']
        assert t3.shape == (2, 1, 10, 10, 5)

    def test_param_count_positive(self):
        """Total parameter count should be positive."""
        heads = TaskHeads(in_channels=40)
        n = heads._num_params()
        assert n > 0, f"Zero parameters"

    def test_different_out_depths(self):
        """TaskHeads should work with different depth dimensions."""
        for d in [5, 10, 20]:
            heads = TaskHeads(in_channels=40, out_depth=d)
            x = torch.randn(1, 40, 8, 8)
            outs = heads(x)
            for v in outs.values():
                assert v.shape[-1] == d


class TestJointInversionNet:
    """Test suite for the full JointInversionNet (2D->3D)."""

    def test_end_to_end_shape(self):
        """Full network: (B,2,81,81) -> 5x(B,1,20,40,40)."""
        device = torch.device("cpu")
        model = JointInversionNet()
        x = torch.randn(2, 2, 81, 81)
        outputs = model(x)

        for key in ['task1', 'task2', 'task3', 'task4', 'task5']:
            assert key in outputs, f"Missing output: {key}"
            assert outputs[key].shape == (2, 1, 40, 40, 20), \
                f"{key} shape: {outputs[key].shape}"

    def test_task3_logits_in_full_net(self):
        """Task 3 output from full net should be raw logits (unbounded)."""
        model = JointInversionNet()
        x = torch.randn(2, 2, 81, 81)
        outputs = model(x)

        t3 = outputs['task3']
        assert t3.shape == (2, 1, 40, 40, 20)

    def test_full_network_gradients(self):
        """Gradients should flow through entire network."""
        model = JointInversionNet()
        x = torch.randn(2, 2, 81, 81)
        outputs = model(x)

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

    def test_batch_size_1(self):
        """Network should work with batch size 1."""
        model = JointInversionNet()
        x = torch.randn(1, 2, 81, 81)
        outputs = model(x)
        for key, val in outputs.items():
            assert val.shape[0] == 1, f"{key} batch dim != 1"

    def test_cuda_forward_if_available(self):
        """Network should run on CUDA without errors (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        model = JointInversionNet().cuda()
        x = torch.randn(2, 2, 81, 81, device='cuda')
        with torch.no_grad():
            outputs = model(x)
        for key, val in outputs.items():
            assert val.device.type == 'cuda'
            assert val.shape == (2, 1, 40, 40, 20)


class TestLossFunctions:
    """Test suite for loss functions."""

    def test_mse_loss_computation(self):
        """MSE loss should compute correctly for regression tasks."""
        criterion = JointInversionLoss(leaky_relu_lambda=0.0)

        preds = {
            'task1': torch.zeros(2, 1, 4, 4, 2),
            'task2': torch.zeros(2, 1, 4, 4, 2),
            'task3': torch.zeros(2, 1, 4, 4, 2),  # raw logits for BCE
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

    def test_bce_loss_for_task3_with_logits(self):
        """Task 3 BCE loss should handle raw logits (not sigmoid output)."""
        criterion = JointInversionLoss(leaky_relu_lambda=0.0)

        # task3 gets RAW LOGITS (as the actual model does), not sigmoid
        preds = {
            'task1': torch.rand(2, 1, 4, 4, 2),
            'task2': torch.rand(2, 1, 4, 4, 2),
            'task3': torch.randn(2, 1, 4, 4, 2),  # raw logits!
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

    def test_bce_perfect_prediction(self):
        """BCE loss should be near-zero when predictions match targets perfectly."""
        criterion = JointInversionLoss(leaky_relu_lambda=0.0)

        # Large positive logits -> sigmoid ~1, target=1 => low BCE
        preds = {
            'task1': torch.zeros(2, 1, 4, 4, 2),
            'task2': torch.zeros(2, 1, 4, 4, 2),
            'task3': torch.full((2, 1, 4, 4, 2), 10.0),  # large positive logits
            'task4': torch.zeros(2, 1, 4, 4, 2),
            'task5': torch.zeros(2, 1, 4, 4, 2),
        }
        targets = {
            'density': torch.zeros(2, 1, 4, 4, 2),
            'susceptibility': torch.zeros(2, 1, 4, 4, 2),
            'structural_sim': torch.ones(2, 1, 4, 4, 2),  # all ones
        }

        per_task, _ = criterion(preds, targets)
        assert per_task['task3'] < 0.1, \
            f"BCE with good prediction should be small, got {per_task['task3']}"

    def test_leaky_relu_regularization(self):
        """Leaky-ReLU regularization should add L2 penalty to total loss."""
        model = JointInversionNet()
        criterion_low = JointInversionLoss(leaky_relu_lambda=0.0)
        criterion_high = JointInversionLoss(leaky_relu_lambda=1e-3)

        preds = {
            'task1': torch.randn(2, 1, 4, 4, 2),
            'task2': torch.randn(2, 1, 4, 4, 2),
            'task3': torch.randn(2, 1, 4, 4, 2),
            'task4': torch.randn(2, 1, 4, 4, 2),
            'task5': torch.randn(2, 1, 4, 4, 2),
        }
        targets = {
            'density': torch.randn(2, 1, 4, 4, 2),
            'susceptibility': torch.randn(2, 1, 4, 4, 2),
            'structural_sim': (torch.rand(2, 1, 4, 4, 2) > 0.5).float(),
        }

        _, total_low = criterion_low(preds, targets, model=model)
        _, total_high = criterion_high(preds, targets, model=model)

        assert total_high.item() > total_low.item(), \
            "Higher lambda should produce larger total loss"

    def test_loss_backward(self):
        """Loss backward should work through model."""
        model = JointInversionNet()
        criterion = JointInversionLoss(leaky_relu_lambda=0.0)

        x = torch.randn(2, 2, 81, 81)
        preds = model(x)
        targets = {
            'density': torch.rand(2, 1, 40, 40, 20),
            'susceptibility': torch.rand(2, 1, 40, 40, 20),
            'structural_sim': (torch.rand(2, 1, 40, 40, 20) > 0.5).float(),
        }

        _, total_loss = criterion(preds, targets, model=model)
        total_loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad, "No non-zero gradients found after backward"

    def test_get_criterion_factory(self):
        """get_criterion factory should produce working criterion."""
        c = get_criterion()
        assert isinstance(c, JointInversionLoss)

        preds = {f'task{i}': torch.randn(1, 1, 4, 4, 2) for i in range(1, 6)}
        tgts = {
            'density': torch.rand(1, 1, 4, 4, 2),
            'susceptibility': torch.rand(1, 1, 4, 4, 2),
            'structural_sim': (torch.rand(1, 1, 4, 4, 2) > 0.5).float(),
        }
        _, total = c(preds, tgts)
        assert total.dim() == 0

    def test_custom_task_weights(self):
        """Custom task weights should affect total loss proportionally."""
        c_equal = JointInversionLoss(task_weights={
            'task1': 1.0, 'task2': 1.0, 'task3': 1.0, 'task4': 1.0, 'task5': 1.0,
        }, leaky_relu_lambda=0.0)

        c_weighted = JointInversionLoss(task_weights={
            'task1': 2.0, 'task2': 1.0, 'task3': 1.0, 'task4': 1.0, 'task5': 1.0,
        }, leaky_relu_lambda=0.0)

        preds = {
            'task1': torch.ones(2, 1, 4, 4, 2),
            'task2': torch.zeros(2, 1, 4, 4, 2),
            'task3': torch.zeros(2, 1, 4, 4, 2),
            'task4': torch.zeros(2, 1, 4, 4, 2),
            'task5': torch.zeros(2, 1, 4, 4, 2),
        }
        targets = {
            'density': torch.zeros(2, 1, 4, 4, 2),
            'susceptibility': torch.zeros(2, 1, 4, 4, 2),
            'structural_sim': torch.zeros(2, 1, 4, 4, 2),
        }

        _, total_eq = c_equal(preds, targets)
        _, total_wt = c_weighted(preds, targets)

        # With task1 weight doubled and task1 loss = 1.0 while others = 0,
        # weighted total should be larger
        assert total_wt.item() > total_eq.item(), \
            "Weighted loss should differ from equal-weighted loss"


class TestAMPCapability:
    """Test AMP (Automatic Mixed Precision) compatibility."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_amp_forward_backward(self):
        """Full forward+backward should work under AMP autocast."""
        from torch.cuda.amp import autocast, GradScaler

        model = JointInversionNet().cuda()
        criterion = get_criterion(leaky_relu_lambda=1e-5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = GradScaler()

        x = torch.randn(2, 2, 81, 81, device='cuda')

        optimizer.zero_grad()
        with autocast():
            preds = model(x)
            targets = {
                'density': torch.rand(2, 1, 40, 40, 20, device='cuda'),
                'susceptibility': torch.rand(2, 1, 40, 40, 20, device='cuda'),
                'structural_sim': (torch.rand(2, 1, 40, 40, 20, device='cuda') > 0.5).float(),
            }
            per_task, total_loss = criterion(preds, targets, model=model)

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Should complete without errors
        assert total_loss.dim() == 0


class TestEvaluateMetrics:
    """Test evaluation metric functions."""

    def test_iou_identical(self):
        """IoU of identical masks should be 1.0."""
        from src.evaluate import compute_iou
        t = torch.ones(1, 1, 4, 4, 2)
        assert abs(compute_iou(t, t) - 1.0) < 0.001

    def test_iou_disjoint(self):
        """IoU of disjoint masks should be 0.0."""
        from src.evaluate import compute_iou
        a = torch.zeros(1, 1, 4, 4, 2)
        b = torch.ones(1, 1, 4, 4, 2)
        assert abs(compute_iou(a, b)) < 0.001

    def test_mse_zero(self):
        """MSE of identical tensors should be 0."""
        from src.evaluate import compute_mse
        t = torch.randn(4, 4, 2)
        assert abs(compute_mse(t, t)) < 1e-7

    def test_r2_perfect(self):
        """R^2 of identical tensors should be 1.0."""
        from src.evaluate import compute_r2
        t = torch.randn(4, 4, 2)
        assert abs(compute_r2(t, t) - 1.0) < 0.001

    def test_compute_all_metrics(self):
        """compute_all_metrics should return dict with all 6 metrics per task."""
        from src.evaluate import compute_all_metrics
        # Use larger volume so SSIM window size fits (need >=7 in spatial dims)
        preds = {
            'task1': torch.rand(2, 1, 8, 8, 4),
            'task2': torch.rand(2, 1, 8, 8, 4),
            'task3': (torch.rand(2, 1, 8, 8, 4) > 0.5).float(),
            'task4': torch.rand(2, 1, 8, 8, 4),
            'task5': torch.rand(2, 1, 8, 8, 4),
        }
        tgts = {
            'density': torch.rand(2, 1, 8, 8, 4),
            'susceptibility': torch.rand(2, 1, 8, 8, 4),
            'structural_sim': (torch.rand(2, 1, 8, 8, 4) > 0.5).float(),
        }
        result = compute_all_metrics(preds, tgts)
        expected_metrics = {'iou', 'mse', 'mae', 'r2', 'ssim', 'psnr'}
        for task_key in ['task1', 'task2', 'task3', 'task4', 'task5']:
            assert set(result[task_key].keys()) == expected_metrics, \
                f"{task_key}: missing metrics"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
