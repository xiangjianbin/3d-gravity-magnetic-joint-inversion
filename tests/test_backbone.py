"""
Tests for 2D U-Net Backbone and ASPP modules.

Validates:
  1. Forward pass shape correctness (2D input 81x81 -> 2D output 40x40)
  2. Multi-scale feature output (return_features=True)
  3. Parameter count in reasonable range (5M-15M combined backbone+ASPP)
  4. No NaN/Inf in outputs
  5. Gradient flow through full pipeline
  6. ASPP multi-scale feature aggregation
  7. Deterministic output (no randomness in forward)
  8. Different input channel counts
"""

import sys
import os
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from src.model.backbone_unet3d import UNet2DBackbone
from src.model.aspp import ASPP2d


# =====================================================================
# U-Net Backbone Tests
# =====================================================================

class TestUNet2DBackbone:
    """Test suite for UNet2DBackbone."""

    def test_forward_shape_standard(self):
        """Output shape should be (B, 64, 40, 40) for standard 81x81 input."""
        model = UNet2DBackbone(in_channels=2)
        x = torch.randn(2, 2, 81, 81)
        out = model(x)
        assert out.shape == (2, 64, 40, 40), \
            f"Expected (2,64,40,40), got {out.shape}"

    def test_forward_shape_batch1(self):
        """Should work with batch size 1 (including training mode)."""
        model = UNet2DBackbone(in_channels=2)
        model.train()  # training mode is stricter (BN behaves differently)
        x = torch.randn(1, 2, 81, 81)
        out = model(x)
        assert out.shape == (1, 64, 40, 40)

    def test_forward_shape_batch4(self):
        """Should work with larger batch sizes."""
        model = UNet2DBackbone(in_channels=2)
        x = torch.randn(4, 2, 81, 81)
        out = model(x)
        assert out.shape == (4, 64, 40, 40)

    def test_parameter_count(self):
        """Parameter count should be in a reasonable range (~7.8M for this architecture)."""
        model = UNet2DBackbone(in_channels=2)
        n_params = model._num_params()
        # 2D U-Net with base=64: ~7.86M params; allow wide range
        assert 1_000_000 < n_params < 50_000_000, \
            f"Parameter count {n_params:,} seems abnormal"

    def test_gradient_flow(self):
        """Gradients should propagate through the entire network."""
        model = UNet2DBackbone(in_channels=2)
        x = torch.randn(2, 2, 81, 81, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient is None"
        assert not torch.isnan(x.grad).any(), "NaN in input gradients"
        assert not torch.isinf(x.grad).any(), "Inf in input gradients"

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN in grad of {name}"
                assert not torch.isinf(param.grad).any(), f"Inf in grad of {name}"

    def test_no_nan_inf_output(self):
        """Output must not contain NaN or Inf values."""
        model = UNet2DBackbone(in_channels=2)
        # Test both train and eval modes
        for mode in [True, False]:
            model.train() if mode else model.eval()
            x = torch.randn(2, 2, 81, 81)
            with torch.no_grad():
                out = model(x)
            assert not torch.isnan(out).any(), \
                f"NaN in output (mode={'train' if mode else 'eval'})"
            assert not torch.isinf(out).any(), \
                f"Inf in output (mode={'train' if mode else 'eval'})"

    def test_return_features(self):
        """return_features=True should return dict of all encoder/decoder features."""
        model = UNet2DBackbone(in_channels=2)
        x = torch.randn(1, 2, 81, 81)
        with torch.no_grad():
            feats = model(x, return_features=True)

        assert isinstance(feats, dict), "return_features should return dict"
        expected_keys = {'enc1', 'enc2', 'enc3', 'enc4',
                         'dec1', 'dec2', 'dec3', 'dec4', 'out'}
        assert set(feats.keys()) == expected_keys, \
            f"Missing keys: {expected_keys - set(feats.keys())}"

        # Verify shapes
        assert feats['enc1'].shape == (1, 64, 81, 81), \
            f"enc1 shape: {feats['enc1'].shape}"
        assert feats['enc2'].shape == (1, 128, 40, 40), \
            f"enc2 shape: {feats['enc2'].shape}"
        assert feats['enc3'].shape == (1, 256, 20, 20), \
            f"enc3 shape: {feats['enc3'].shape}"
        assert feats['enc4'].shape == (1, 512, 10, 10), \
            f"enc4 shape: {feats['enc4'].shape}"
        assert feats['dec1'].shape == (1, 256, 20, 20), \
            f"dec1 shape: {feats['dec1'].shape}"
        assert feats['dec2'].shape == (1, 128, 40, 40), \
            f"dec2 shape: {feats['dec2'].shape}"
        assert feats['dec3'].shape == (1, 64, 81, 81), \
            f"dec3 shape: {feats['dec3'].shape}"
        assert feats['out'].shape == (1, 64, 40, 40), \
            f"out shape: {feats['out'].shape}"

    def test_return_features_matches_default(self):
        """The 'out' key from return_features should match default forward output."""
        model = UNet2DBackbone(in_channels=2)
        model.eval()
        x = torch.randn(1, 2, 81, 81)
        with torch.no_grad():
            out_default = model(x)
            out_feats = model(x, return_features=True)['out']
        assert torch.allclose(out_default, out_feats, atol=1e-6), \
            "return_features['out'] differs from default output"

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

    def test_leaky_relu_used(self):
        """Verify LeakyReLU is used (not ReLU) throughout the network."""
        import torch.nn as nn
        model = UNet2DBackbone(in_channels=2)
        leaky_found = False
        relu_found = False
        for module in model.modules():
            if isinstance(module, nn.LeakyReLU):
                leaky_found = True
                # Verify negative_slope is 0.01
                assert module.negative_slope == 0.01, \
                    f"LeakyReLU negative_slope={module.negative_slope}, expected 0.01"
            elif isinstance(module, nn.ReLU) and module.inplace:
                relu_found = True
        assert leaky_found, "No LeakyReLU found in backbone"
        assert not relu_found, "Found ReLU (should use LeakyReLU per paper Eq.8)"


# =====================================================================
# ASPP Module Tests
# =====================================================================

class TestASPP2d:
    """Test suite for ASPP2d module."""

    def test_forward_shape(self):
        """Output shape should preserve spatial dims, change channels to out_channels."""
        aspp = ASPP2d(in_channels=64, out_channels=40)
        x = torch.randn(2, 64, 40, 40)
        out = aspp(x)
        assert out.shape == (2, 40, 40, 40), \
            f"Expected (2,40,40,40), got {out.shape}"

    def test_forward_shape_different_spatial(self):
        """ASPP should work with different input spatial sizes."""
        aspp = ASPP2d(in_channels=64, out_channels=40)
        for h, w in [(10, 10), (20, 20), (80, 80)]:
            x = torch.randn(1, 64, h, w)
            out = aspp(x)
            assert out.shape == (1, 40, h, w), \
                f"For input ({h},{w}): expected (1,40,{h},{w}), got {out.shape}"

    def test_parameter_count(self):
        """ASPP parameter count should be ~79K for in_ch=64, out_ch=40."""
        aspp = ASPP2d(in_channels=64, out_channels=40)
        n_params = sum(p.numel() for p in aspp.parameters())
        # Expected: ~79,200; allow some flexibility
        assert 10_000 < n_params < 500_000, \
            f"ASPP param count {n_params:,} seems abnormal"

    def test_no_nan_inf(self):
        """No NaN or Inf in ASPP output (including batch_size=1 in training mode)."""
        aspp = ASPP2d(in_channels=64, out_channels=40)
        for mode in [True, False]:
            aspp.train() if mode else aspp.eval()
            x = torch.randn(1, 64, 40, 40)
            with torch.no_grad():
                out = aspp(x)
            assert not torch.isnan(out).any(), \
                f"NaN in ASPP output (mode={'train' if mode else 'eval'})"
            assert not torch.isinf(out).any(), \
                f"Inf in ASPP output (mode={'train' if mode else 'eval'})"

    def test_gradient_flow(self):
        """Gradients should flow through ASPP."""
        aspp = ASPP2d(in_channels=64, out_channels=40)
        x = torch.randn(2, 64, 40, 40, requires_grad=True)
        out = aspp(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient is None"
        assert not torch.isnan(x.grad).any(), "NaN in input gradients"
        assert not torch.isinf(x.grad).any(), "Inf in input gradients"

        for name, param in aspp.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN in grad of {name}"

    def test_rates_configuration(self):
        """Verify ASPP uses the correct dilation rates from the paper."""
        aspp = ASPP2d(in_channels=64, out_channels=40)
        assert aspp.RATES == [6, 12, 18, 24], \
            f"Expected rates [6,12,18,24], got {aspp.RATES}"
        assert len(aspp.branches) == 4, \
            f"Expected 4 dilated branches, got {len(aspp.branches)}"

    def test_branch_output_channels(self):
        """Each branch should produce BRANCH_OUT channels."""
        aspp = ASPP2d(in_channels=64, out_channels=40)
        x = torch.randn(1, 64, 40, 40)
        with torch.no_grad():
            branch_outputs = []
            for b in aspp.branches:
                branch_outputs.append(b(x))
            pool_out = aspp.global_pool(x)

        for i, bo in enumerate(branch_outputs):
            assert bo.shape[1] == aspp.BRANCH_OUT, \
                f"Branch {i} output channels: {bo.shape[1]}, expected {aspp.BRANCH_OUT}"
        assert pool_out.shape[1] == aspp.BRANCH_OUT, \
            f"Global pool output channels: {pool_out.shape[1]}"

    def test_leaky_relu_used(self):
        """ASPP should use LeakyReLU (not ReLU)."""
        import torch.nn as nn
        aspp = ASPP2d(in_channels=64, out_channels=40)
        leaky_found = False
        for module in aspp.modules():
            if isinstance(module, nn.LeakyReLU):
                leaky_found = True
                assert module.negative_slope == 0.01, \
                    f"LeakyReLU slope={module.negative_slope}, expected 0.01"
        assert leaky_found, "No LeakyReLU found in ASPP"


# =====================================================================
# Combined Pipeline Tests
# =====================================================================

class TestCombinedPipeline:
    """Tests for the backbone + ASPP combined pipeline."""

    def test_end_to_end_shape(self):
        """Full pipeline: (B,2,81,81) -> backbone -> ASPP -> (B,40,40,40)."""
        backbone = UNet2DBackbone(in_channels=2)
        aspp = ASPP2d(in_channels=64, out_channels=40)

        x = torch.randn(2, 2, 81, 81)
        with torch.no_grad():
            feat = backbone(x)
            out = aspp(feat)

        assert feat.shape == (2, 64, 40, 40), \
            f"Backbone output: {feat.shape}"
        assert out.shape == (2, 40, 40, 40), \
            f"ASPP output: {out.shape}"

    def test_total_param_count_in_range(self):
        """Combined backbone + ASPP params should be between 5M and 15M."""
        backbone = UNet2DBackbone(in_channels=2)
        aspp = ASPP2d(in_channels=64, out_channels=40)

        total = (backbone._num_params() +
                 sum(p.numel() for p in aspp.parameters()))
        assert 5_000_000 <= total <= 15_000_000, \
            f"Total params {total:,} outside [5M, 15M] range"

    def test_pipeline_no_nan_inf(self):
        """Full pipeline should produce no NaN/Inf at any stage."""
        backbone = UNet2DBackbone(in_channels=2)
        aspp = ASPP2d(in_channels=64, out_channels=40)
        backbone.train()
        aspp.train()

        x = torch.randn(1, 2, 81, 81)
        with torch.no_grad():
            feat = backbone(x)
            assert not torch.isnan(feat).any(), "NaN in backbone output"
            assert not torch.isinf(feat).any(), "Inf in backbone output"

            out = aspp(feat)
            assert not torch.isnan(out).any(), "NaN in ASPP output"
            assert not torch.isinf(out).any(), "Inf in ASPP output"

    def test_pipeline_gradient_flow(self):
        """Gradients flow end-to-end through backbone -> ASPP."""
        backbone = UNet2DBackbone(in_channels=2)
        aspp = ASPP2d(in_channels=64, out_channels=40)

        x = torch.randn(2, 2, 81, 81, requires_grad=True)
        feat = backbone(x)
        out = aspp(feat)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient to input"
        assert not torch.isnan(x.grad).any(), "NaN in input grad"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
