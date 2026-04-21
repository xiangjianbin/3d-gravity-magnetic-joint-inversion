"""
Tests for 2D U-Net Backbone module.

Validates:
  - Forward pass shape correctness (2D input 81x81 -> 2D output 40x40)
  - Parameter count sanity
  - Gradient flow
"""

import sys
import os
import pytest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from src.model.backbone_unet3d import UNet2DBackbone


class TestUNet2DBackbone:
    """Test suite for UNet2DBackbone."""

    def test_forward_shape(self):
        """Output shape should be (B, 64, 40, 40) for standard 81x81 input."""
        device = torch.device("cpu")
        model = UNet2DBackbone(in_channels=2)
        x = torch.randn(2, 2, 81, 81)
        out = model(x)
        assert out.shape == (2, 64, 40, 40), \
            f"Expected (2,64,40,40), got {out.shape}"

    def test_forward_shape_batch1(self):
        """Should work with batch size 1."""
        model = UNet2DBackbone(in_channels=2)
        x = torch.randn(1, 2, 81, 81)
        out = model(x)
        assert out.shape == (1, 64, 40, 40)

    def test_parameter_count(self):
        """Parameter count should be in a reasonable range."""
        model = UNet2DBackbone(in_channels=2)
        n_params = model._num_params()
        # 2D U-Net should be ~7.8M params; allow wide range
        assert 1_000_000 < n_params < 50_000_000, \
            f"Parameter count {n_params:,} seems abnormal"

    def test_gradient_flow(self):
        """Gradients should propagate through the network."""
        model = UNet2DBackbone(in_channels=2)
        x = torch.randn(2, 2, 81, 81, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient is None"
        assert not torch.isnan(x.grad).any(), "NaN in input gradients"

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN in grad of {name}"

    def test_different_input_channels(self):
        """Should work with different number of input channels."""
        model = UNet2DBackbone(in_channels=4)
        x = torch.randn(2, 4, 81, 81)
        out = model(x)
        assert out.shape == (2, 64, 40, 40)

    def test_deterministic_output(self):
        """Same input should produce same output (no randomness in forward)."""
        model = UNet2DBackbone(in_channels=2)
        model.eval()
        x = torch.randn(2, 2, 81, 81)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.equal(out1, out2), "Outputs differ for same input"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
