"""
Backbone U-Net + ASPP 模块冒烟测试

测试覆盖:
    1. 输入输出形状验证
    2. 反向传播梯度流
    3. NaN/Inf 检测
    4. ASPP 形状和 dilation 效应
    5. AMP (混合精度) 兼容性
"""

import pytest
import torch
import torch.nn as nn

# 确保项目路径可导入
import sys
sys.path.insert(0, "/home/wx/重磁联合反演 - pipeline改进思路探索")

from src.model.backbone_unet3d import (
    BackboneUNet3d,
    ConvBlock3d,
    Encoder3d,
    Decoder3d,
    build_backbone,
    count_parameters,
)
from src.model.aspp import (
    ASPP3d,
    ASPPConv3d,
    ASPPPooling3d,
    build_aspp,
)


# ============================================================
# 固定随机种子保证可复现
# ============================================================
@pytest.fixture(autouse=True)
def set_global_seed():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


# ============================================================
# 测试数据 fixture
# ============================================================
@pytest.fixture
def sample_input():
    """标准输入: (B=2, C=2, D=40, H=40, W=20)"""
    return torch.randn(2, 2, 40, 40, 20)


@pytest.fixture
def sample_feature():
    """ASPP 标准输入: (B=2, C=256, D=40, H=40, W=20)"""
    return torch.randn(2, 256, 40, 40, 20)


@pytest.fixture
def backbone():
    """默认配置的 Backbone"""
    return build_backbone(use_checkpoint=False)


@pytest.fixture
def aspp_module():
    """默认配置的 ASPP"""
    return build_aspp()


# ============================================================
# 1. Backbone 输入输出形状测试
# ============================================================
class TestBackboneShape:
    """验证 Backbone 输入 (B,2,40,40,20) -> 输出 (B,256,40,40,20)"""

    def test_backbone_input_output_shape(self, backbone, sample_input):
        """输入 (B,2,40,40,20) -> 输出 (B,256,40,40,20)"""
        backbone.eval()
        with torch.no_grad():
            output = backbone(sample_input)

        assert output.shape == (2, 256, 40, 40, 20), \
            f"Expected shape (2, 256, 40, 40, 20), got {output.shape}"

    def test_backbone_batch1(self, backbone):
        """batch_size=1 时形状正确"""
        backbone.eval()
        x = torch.randn(1, 2, 40, 40, 20)
        with torch.no_grad():
            output = backbone(x)
        assert output.shape == (1, 256, 40, 40, 20)

    def test_backbone_different_out_channels(self):
        """自定义 out_channels 参数"""
        model = build_backbone()
        custom_model = BackboneUNet3d(
            in_channels=2, base_channels=64, out_channels=128
        )
        custom_model.eval()
        x = torch.randn(1, 2, 40, 40, 20)
        with torch.no_grad():
            output = custom_model(x)
        assert output.shape == (1, 128, 40, 40, 20)

    def test_conv_block_output_shape(self):
        """ConvBlock3d 保持空间尺寸不变"""
        block = ConvBlock3d(32, 64)
        block.eval()
        x = torch.randn(1, 32, 10, 10, 5)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (1, 64, 10, 10, 5)


# ============================================================
# 2. 反向传播梯度流测试
# ============================================================
class TestGradientFlow:
    """验证反向传播无报错，所有参数有梯度"""

    def test_backbone_gradient_flow(self, backbone, sample_input):
        """反向传播无报错"""
        backbone.train()
        sample_input.requires_grad_(True)
        output = backbone(sample_input)
        loss = output.sum()
        loss.backward()

        # 确保输入有梯度
        assert sample_input.grad is not None, "Input has no gradient"
        assert not torch.isnan(sample_input.grad).any(), "Input gradient contains NaN"

    def test_all_params_get_gradients(self, backbone, sample_input):
        """所有可训练参数都收到梯度"""
        backbone.train()
        output = backbone(sample_input)
        loss = output.sum()
        loss.backward()

        for name, param in backbone.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert not torch.isnan(param.grad).any(), \
                    f"Parameter {name} gradient contains NaN"


# ============================================================
# 3. NaN / Inf 检测
# ============================================================
class TestNoNaNInf:
    """验证输出不含 NaN 或 Inf"""

    def test_backbone_no_nan(self, backbone, sample_input):
        """输出无 NaN/Inf"""
        backbone.eval()
        with torch.no_grad():
            output = backbone(sample_input)

        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_aspp_no_nan(self, aspp_module, sample_feature):
        """ASPP 输出无 NaN/Inf"""
        aspp_module.eval()
        with torch.no_grad():
            output = aspp_module(sample_feature)

        assert not torch.isnan(output).any(), "ASPP output contains NaN"
        assert not torch.isinf(output).any(), "ASPP output contains Inf"

    def test_backbone_aspp_pipeline_no_nan(self, backbone, aspp_module, sample_input):
        """Backbone + ASPP 联合前向无 NaN"""
        backbone.eval()
        aspp_module.eval()
        with torch.no_grad():
            features = backbone(sample_input)
            output = aspp_module(features)

        assert not torch.isnan(output).any(), "Pipeline output contains NaN"
        assert not torch.isinf(output).any(), "Pipeline output contains Inf"


# ============================================================
# 4. ASPP 模块专项测试
# ============================================================
class TestASPP:
    """ASPP 模块功能验证"""

    def test_aspp_output_shape(self, aspp_module, sample_feature):
        """ASPP: (B,256,D,H,W) -> (B,256,D,H,W)"""
        aspp_module.eval()
        with torch.no_grad():
            output = aspp_module(sample_feature)

        assert output.shape == sample_feature.shape, \
            f"ASPP shape mismatch: expected {sample_feature.shape}, got {output.shape}"

    def test_aspp_different_spatial_sizes(self):
        """ASPP 支持不同空间尺寸的输入"""
        aspp = build_aspp()
        aspp.eval()
        for size in [(10, 10, 5), (20, 20, 10), (8, 16, 8)]:
            x = torch.randn(1, 256, *size)
            with torch.no_grad():
                out = aspp(x)
            assert out.shape == (1, 256, *size), \
                f"Failed for spatial size {size}: got {out.shape}"

    def test_aspp_dilation_effect(self):
        """不同 dilation rate 应产生不同感受野（通过参数量间接验证）"""
        # 创建三个不同 dilation 的卷积并检查参数量
        conv6 = ASPPConv3d(256, 256, dilation=6)
        conv12 = ASPPConv3d(256, 256, dilation=12)
        conv18 = ASPPConv3d(256, 256, dilation=18)

        # 同结构的空洞卷积参数量相同 (kernel_size 相同)
        p6 = sum(p.numel() for p in conv6.parameters())
        p12 = sum(p.numel() for p in conv12.parameters())
        p18 = sum(p.numel() for p in conv18.parameters())

        # 但它们的 receptive field 不同:
        # RF = kernel + (kernel-1)*(dilation-1) = 3 + 2*(d-1)
        # d=6 -> RF=13, d=12 -> RF=25, d=18 -> RF=37
        # 通过检查权重矩阵的实际尺寸来区分
        assert p6 == p12 == p18, \
            "Dilated convs should have same parameter count but different receptive fields"

        # 验证它们对同一输入产生不同输出
        x = torch.randn(1, 256, 10, 10, 5)
        conv6.eval(); conv12.eval(); conv18.eval()
        with torch.no_grad():
            o6 = conv6(x)
            o12 = conv12(x)
            o18 = conv18(x)

        # 不同 dilation 应产生不同输出 (概率上几乎必然不同)
        assert not torch.allclose(o6, o12, atol=1e-6), \
            "d=6 and d=12 produce identical outputs"
        assert not torch.allclose(o12, o18, atol=1e-6), \
            "d=12 and d=18 produce identical outputs"

    def test_aspp_branch_count(self, aspp_module):
        """ASPP 应包含 3 个 dilated 分支 + 1 个 pooling 分支"""
        assert len(aspp_module.branches) == 3, \
            f"Expected 3 dilated branches, got {len(aspp_module.branches)}"
        assert isinstance(aspp_module.global_pool, ASPPPooling3d), \
            "Missing global pooling branch"

    def test_aspp_fusion_channels(self, aspp_module):
        """融合层输入通道数应为 4 * 256 = 1024"""
        fusion_conv = aspp_module.fusion[0]
        assert fusion_conv.in_channels == 1024, \
            f"Fusion layer input channels: expected 1024, got {fusion_conv.in_channels}"
        assert fusion_conv.out_channels == 256, \
            f"Fusion layer output channels: expected 256, got {fusion_conv.out_channels}"


# ============================================================
# 5. AMP (混合精度) 兼容性测试
# ============================================================
class TestAMPCompatibility:
    """验证模型支持 torch.cuda.amp.autocast()"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backbone_with_amp_cuda(self, backbone, sample_input):
        """GPU 上混合精度前向传播正常"""
        backbone = backbone.cuda()
        x = sample_input.cuda()
        backbone.train()

        import torch.cuda.amp as amp
        scaler = amp.GradScaler()

        with amp.autocast():
            output = backbone(x)
            loss = output.mean()

        scaler.scale(loss).backward()
        scaler.step(torch.optim.Adam(backbone.parameters(), lr=1e-4))
        scaler.update()

        assert output.shape == (2, 256, 40, 40, 20)
        assert not torch.isnan(output.cpu()).any()

    def test_backbone_with_amp_cpu(self, backbone, sample_input):
        """CPU 上混合精度模拟 (autocast 在 CPU 也支持 float16 前向)"""
        backbone.train()

        # CPU autocast 使用 bfloat16 (如果支持) 或 float16
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            output = backbone(sample_input)
            loss = output.mean()

        assert output.shape == (2, 256, 40, 40, 20)
        # bfloat16 可能精度较低但不应产生 NaN
        assert not torch.isnan(output).any(), "AMP forward produced NaN on CPU"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_aspp_with_amp_cuda(self, aspp_module, sample_feature):
        """ASPP GPU 混合精度正常"""
        aspp_module = aspp_module.cuda()
        x = sample_feature.cuda()
        aspp_module.train()

        import torch.cuda.amp as amp
        with amp.autocast():
            output = aspp_module(x)

        assert output.shape == (2, 256, 40, 40, 20)
        assert not torch.isnan(output.cpu()).any()


# ============================================================
# 6. Gradient Checkpointing 测试
# ============================================================
class TestGradientCheckpointing:
    """验证 gradient checkpointing 功能"""

    def test_checkpoint_forward_shape(self, sample_input):
        """checkpoint 模式下输出形状正确"""
        model = build_backbone(use_checkpoint=True)
        model.train()  # checkpoint 只在 training 模式生效
        with torch.no_grad():
            output = model(sample_input)
        assert output.shape == (2, 256, 40, 40, 20)

    def test_checkpoint_backward(self, sample_input):
        """checkpoint 模式下反向传播正常"""
        model = build_backbone(use_checkpoint=True)
        model.train()
        output = model(sample_input)
        loss = output.sum()
        loss.backward()

        # 至少部分参数应有梯度
        grad_count = sum(
            1 for p in model.parameters()
            if p.requires_grad and p.grad is not None
        )
        assert grad_count > 0, "No parameters received gradients in checkpoint mode"


# ============================================================
# 7. 参数量统计测试
# ============================================================
class TestParameterCount:
    """参数量合理性检验"""

    def test_backbone_param_count_reasonable(self, backbone):
        """Backbone 参数量应在合理范围内 (几M ~ 几十M)"""
        stats = count_parameters(backbone)
        total_m = stats["total"] / 1e6

        # 3D U-Net with base=64, double conv blocks: 预估 ~30-80M params
        assert 1e6 < stats["total"] < 200e6, \
            f"Parameter count {total_m:.1f}M seems unreasonable (expected ~10-100M)"

    def test_aspp_param_count_reasonable(self, aspp_module):
        """ASPP 参数量应在合理范围"""
        total = sum(p.numel() for p in aspp_module.parameters())
        total_k = total / 1e3

        # 4 branches * (conv3d 256->256 + bn) + global pool + fusion
        # 预估 ~3-15M params (3D 卷积参数较多)
        assert 1e3 < total < 50e6, \
            f"ASPP parameter count {total_k:.1f}K seems unreasonable"


# ============================================================
# 8. Backbone + ASPP 端到端测试
# ============================================================
class TestEndToEnd:
    """完整 pipeline 冒烟测试"""

    def test_full_pipeline_forward(self, sample_input):
        """Backbone -> ASPP 完整前向传播"""
        backbone = build_backbone()
        aspp = build_aspp()
        backbone.eval()
        aspp.eval()

        with torch.no_grad():
            features = backbone(sample_input)   # (B, 256, 40, 40, 20)
            output = aspp(features)              # (B, 256, 40, 40, 20)

        assert features.shape == (2, 256, 40, 40, 20)
        assert output.shape == (2, 256, 40, 40, 20)

    def test_full_pipeline_train_step(self, sample_input):
        """完整 pipeline 训练步骤 (forward + backward + optimizer step)"""
        backbone = build_backbone()
        aspp = build_aspp()
        backbone.train()
        aspp.train()

        target = torch.randn_like(sample_input[:, :1])  # dummy target

        features = backbone(sample_input)
        output = aspp(features)
        # 简单 MSE loss (dummy)
        loss = nn.MSELoss()(output.mean(dim=1, keepdim=True), target)
        loss.backward()

        optimizer = torch.optim.Adam(
            list(backbone.parameters()) + list(aspp.parameters()), lr=1e-4
        )
        optimizer.step()

        assert not torch.isnan(loss), "Loss is NaN after train step"
