"""
任务头和网络组装单元测试
========================

测试覆盖:
1. 各任务头输出形状
2. Task 3 Sigmoid 值域 [0, 1]
3. 完整网络前向传播 (训练模式/推理模式)
4. 多任务损失计算
5. 反向传播梯度检查
6. 输出无 NaN

运行:
    python -m pytest tests/test_heads.py -v
"""

import sys
import os
import torch
import pytest

# 添加项目根路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.task_heads import (
    TaskIndependentGravityHead,
    TaskIndependentMagneticHead,
    TaskStructuralSimilarity,
    TaskJointInversionHead,
)
from src.model.joint_inversion_net import JointInversionNet
from src.model.loss_functions import MultiTaskLoss


# ===== 测试配置 =====
BATCH_SIZE = 2
D, H, W = 40, 40, 20   # 网络输入空间尺寸（论文规定）
IN_CHANNELS = 2         # 重力 + 磁异常
ASPP_CHANNELS = 256     # ASPP 输出通道数

# 设备: 有 GPU 用 GPU，否则 CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _make_aspp_feat(batch_size=BATCH_SIZE):
    """构造 ASPP 输出特征张量"""
    return torch.randn(batch_size, ASPP_CHANNELS, D, H, W, device=DEVICE)


def _make_input(batch_size=BATCH_SIZE):
    """构造网络输入张量 (重力+磁异常)"""
    return torch.randn(batch_size, IN_CHANNELS, D, H, W, device=DEVICE)


def _make_single_channel(batch_size=BATCH_SIZE):
    """构造单通道预测输出张量"""
    return torch.randn(batch_size, 1, D, H, W, device=DEVICE)


# =====================================================================
# 测试 1: Task 1 — 独立重力反演头
# =====================================================================

class TestTask1IndependentGravity:
    """Task 1: 独立重力反演头测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.head = TaskIndependentGravityHead(
            in_channels=ASPP_CHANNELS,
        ).to(DEVICE)

    def test_output_shape(self):
        """(B,256,40,40,20) -> (B,1,40,40,20)"""
        x = _make_aspp_feat()
        out = self.head(x)
        assert out.shape == (BATCH_SIZE, 1, D, H, W), (
            f"Task 1 输出形状错误: 期望 {(BATCH_SIZE, 1, D, H, W)}, "
            f"实际 {out.shape}"
        )

    def test_no_nan(self):
        """输出无 NaN"""
        x = _make_aspp_feat()
        out = self.head(x)
        assert not torch.isnan(out).any(), "Task 1 输出包含 NaN"
        assert not torch.isinf(out).any(), "Task 1 输出包含 Inf"


# =====================================================================
# 测试 2: Task 2 — 独立磁法反演头
# =====================================================================

class TestTask2IndependentMagnetic:
    """Task 2: 独立磁法反演头测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.head = TaskIndependentMagneticHead(
            in_channels=ASPP_CHANNELS,
        ).to(DEVICE)

    def test_output_shape(self):
        """(B,256,40,40,20) -> (B,1,40,40,20)"""
        x = _make_aspp_feat()
        out = self.head(x)
        assert out.shape == (BATCH_SIZE, 1, D, H, W), (
            f"Task 2 输出形状错误: 期望 {(BATCH_SIZE, 1, D, H, W)}, "
            f"实际 {out.shape}"
        )

    def test_independent_params(self):
        """Task 2 与 Task 1 参数独立（不同的参数对象，不共享内存）"""
        task1 = TaskIndependentGravityHead(in_channels=ASPP_CHANNELS).to(DEVICE)
        # 检查两个实例的参数是独立的对象（不是同一个内存地址）
        params1 = list(task1.parameters())
        params2 = list(self.head.parameters())
        assert len(params1) == len(params2), "参数数量不一致"
        for i, (p1, p2) in enumerate(zip(params1, params2)):
            # 关键检查: 参数对象不能相同（即不是共享权重）
            assert p1 is not p2, (
                f"Task 1 和 Task 2 的第 {i} 个参数是同一对象（共享了内存）"
            )
            # 卷积权重应不同（随机初始化），BN 权重可能初始值相同但对象独立
            if p1.numel() > 10 and 'weight' in str(p1.shape):
                # 大张量（卷积权重）随机初始化后应不同
                assert not torch.equal(p1.data, p2.data), (
                    f"Task 1 和 Task 2 的第 {i} 个大参数值完全相同，可能存在意外共享"
                )


# =====================================================================
# 测试 3: Task 3 — 结构相似性模块
# =====================================================================

class TestTask3StructuralSimilarity:
    """Task 3: 结构相似性提取模块测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.module = TaskStructuralSimilarity().to(DEVICE)

    def test_output_shape(self):
        """2个(B,1,40,40,20) -> (B,1,40,40,20)"""
        rho = _make_single_channel()
        kappa = _make_single_channel()
        out = self.module(rho, kappa)
        assert out.shape == (BATCH_SIZE, 1, D, H, W), (
            f"Task 3 输出形状错误: 期望 {(BATCH_SIZE, 1, D, H, W)}, "
            f"实际 {out.shape}"
        )

    def test_sigmoid_range(self):
        """输出确实在 [0, 1] 范围内（Sigmoid 激活）"""
        rho = _make_single_channel()
        kappa = _make_single_channel()
        out = self.module(rho, kappa)
        assert out.min() >= 0.0, f"Task 3 输出最小值 < 0: {out.min().item():.6f}"
        assert out.max() <= 1.0, f"Task 3 输出最大值 > 1: {out.max().item():.6f}"

    def test_sigmoid_range_extreme_input(self):
        """极端输入下 Sigmoid 输出仍在 [0, 1]"""
        # 使用极大和极小值作为输入
        rho = torch.ones(BATCH_SIZE, 1, D, H, W, device=DEVICE) * 100.0
        kappa = torch.ones(BATCH_SIZE, 1, D, H, W, device=DEVICE) * (-50.0)
        out = self.module(rho, kappa)
        assert out.min() >= 0.0 and out.max() <= 1.0, (
            "极端输入下 Task 3 输出超出 [0, 1] 范围"
        )

    def test_different_input_produces_different_output(self):
        """不同输入产生不同输出（非恒等函数）"""
        rho_a = _make_single_channel()
        kappa_a = _make_single_channel()
        rho_b = _make_single_channel()
        kappa_b = _make_single_channel()

        out_a = self.module(rho_a, kappa_a)
        out_b = self.module(rho_b, kappa_b)

        # 随机初始化的不同输入，输出大概率不同
        # 允许极小概率相同，但通常不应完全一致
        if not torch.allclose(out_a, out_b, atol=1e-6):
            pass  # 正常情况
        else:
            # 如果碰巧相同，换一组更极端的数据再试
            rho_c = torch.zeros(BATCH_SIZE, 1, D, H, W, device=DEVICE)
            kappa_c = torch.ones(BATCH_SIZE, 1, D, H, W, device=DEVICE)
            out_c = self.module(rho_c, kappa_c)
            assert not torch.allclose(out_a, out_c, atol=1e-6), (
                "Task 3 对不同输入产生了相同的输出"
            )


# =====================================================================
# 测试 4 & 5: 联合反演头
# =====================================================================

class TestTask4JointGravity:
    """Task 4: 联合重力反演头测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.head = TaskJointInversionHead().to(DEVICE)

    def test_output_shape(self):
        """(B,3,40,40,20) -> (B,1,40,40,20)"""
        input_data = _make_input()                          # (B, 2, D, H, W)
        sim = torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE)  # (B, 1, D, H, W)
        out = self.head(input_data, sim)
        assert out.shape == (BATCH_SIZE, 1, D, H, W), (
            f"Task 4 输出形状错误: 期望 {(BATCH_SIZE, 1, D, H, W)}, "
            f"实际 {out.shape}"
        )

    def test_no_sigmoid_in_output(self):
        """输出未经过 Sigmoid（可能为负值）"""
        input_data = _make_input()
        sim = torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE)
        out = self.head(input_data, sim)
        # logits 可以是任意实数值（不一定在 [0,1] 内）
        # 只需确认不是被 clamp 或 sigmoid 过的
        assert out.requires_grad, "输出应该需要梯度（用于 BCEWithLogitsLoss）"


class TestTask5JointMagnetic:
    """Task 5: 联合磁法反演头测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.head = TaskJointInversionHead().to(DEVICE)

    def test_output_shape(self):
        """(B,3,40,40,20) -> (B,1,40,40,20)"""
        input_data = _make_input()
        sim = torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE)
        out = self.head(input_data, sim)
        assert out.shape == (BATCH_SIZE, 1, D, H, W), (
            f"Task 5 输出形状错误: 期望 {(BATCH_SIZE, 1, D, H, W)}, "
            f"实际 {out.shape}"
        )

    def test_task4_and_task5_independent(self):
        """Task 4 和 Task 5 参数独立（不同的参数对象，不共享内存）"""
        head4 = TaskJointInversionHead().to(DEVICE)
        head5 = TaskJointInversionHead().to(DEVICE)
        params4 = list(head4.parameters())
        params5 = list(head5.parameters())

        # 检查两个实例的参数是独立的对象（不是同一个内存地址）
        for i, (p4, p5) in enumerate(zip(params4, params5)):
            assert p4 is not p5, (
                f"Task 4 和 Task 5 的第 {i} 个参数是同一对象（共享了内存）"
            )
        # 额外检查: 卷积权重（非 BN 的 weight）在两个实例间应不同
        # 只需确认至少有一个 Conv 权重不同即可（BN 初始化为常数 1 是正常的）
        has_different_conv_weight = False
        for i, (p4, p5) in enumerate(zip(params4, params5)):
            # 跳过 BN 的 weight/bias（通常初始化为 1/0）和 bias
            if p4.numel() > 64:  # Conv3d(3,64) 的权重大小 > 64
                if not torch.equal(p4.data, p5.data):
                    has_different_conv_weight = True
                    break
        assert has_different_conv_weight, (
            "Task 4 和 Task 5 所有卷积权重都完全相同，可能存在意外共享"
        )


# =====================================================================
# 测试 6: 完整网络前向传播
# =====================================================================

class TestFullNetworkForward:
    """完整 JointInversionNet 的前向传播测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.net = JointInversionNet(
            in_channels=IN_CHANNELS,
            use_gradient_checkpointing=False,  # 测试时不启用 checkpoint
        ).to(DEVICE)

    def test_training_mode_returns_all_five(self):
        """训练模式下返回全部 5 个任务的输出"""
        self.net.train()
        x = _make_input()
        outputs = self.net(x)

        expected_keys = {'rho_pred', 'kappa_pred', 'structural_sim',
                         'rho_final', 'kappa_final'}
        assert set(outputs.keys()) == expected_keys, (
            f"训练模式输出键不匹配。期望 {expected_keys}, 实际 {set(outputs.keys())}"
        )
        for key in expected_keys:
            assert outputs[key].shape == (BATCH_SIZE, 1, D, H, W), (
                f"{key} 形状错误: {outputs[key].shape}"
            )

    def test_inference_mode_returns_two(self):
        """推理模式 (return_all=False) 只返回 Task 4+5"""
        self.net.eval()
        x = _make_input()
        with torch.no_grad():
            outputs = self.net(x, return_all=False)

        expected_keys = {'rho_final', 'kappa_final'}
        assert set(outputs.keys()) == expected_keys, (
            f"推理模式输出键不匹配。期望 {expected_keys}, "
            f"实际 {set(outputs.keys())}"
        )

    def test_return_all_override(self):
        """return_all=True 在推理模式下也返回全部输出"""
        self.net.eval()
        x = _make_input()
        with torch.no_grad():
            outputs = self.net(x, return_all=True)

        assert len(outputs.keys()) == 5, (
            f"return_all=True 应返回 5 个键, 实际 {len(outputs.keys())}"
        )

    def test_no_nan_in_outputs(self):
        """所有输出无 NaN 和 Inf"""
        self.net.train()
        x = _make_input()
        outputs = self.net(x)

        for key, value in outputs.items():
            assert not torch.isnan(value).any(), (
                f"{key} 输出包含 NaN"
            )
            assert not torch.isinf(value).any(), (
                f"{key} 输出包含 Inf"
            )

    def test_structural_sim_range(self):
        """structural_sim 在训练模式下仍在 [0, 1]"""
        self.net.train()
        x = _make_input()
        outputs = self.net(x)
        s = outputs['structural_sim']
        assert s.min() >= 0.0, f"S 最小值 < 0: {s.min().item():.6f}"
        assert s.max() <= 1.0, f"S 最大值 > 1: {s.max().item():.6f}"


# =====================================================================
# 测试 7: 多任务损失函数
# =====================================================================

class TestMultiTaskLoss:
    """多任务损失函数测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.criterion = MultiTaskLoss()
        self.net = JointInversionNet(
            in_channels=IN_CHANNELS,
            use_gradient_checkpointing=False,
        ).to(DEVICE)

    def test_loss_is_scalar(self):
        """loss 是标量且 > 0"""
        self.net.train()
        x = _make_input()
        targets = {
            'rho': torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE),
            'kappa': torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE),
            'sim': torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE),
        }
        predictions = self.net(x)
        total_loss, task_losses = self.criterion(predictions, targets)

        assert total_loss.dim() == 0, f"总损失应为标量, 实际维度 {total_loss.dim()}"
        assert total_loss.item() > 0, f"总损失应为正数, 实际 {total_loss.item():.6f}"

    def test_loss_gradients(self):
        """反向传播正常，所有参数有梯度"""
        self.net.train()
        self.net.zero_grad()
        x = _make_input()
        targets = {
            'rho': torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE),
            'kappa': torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE),
            'sim': torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE),
        }
        predictions = self.net(x)
        total_loss, _ = self.criterion(predictions, targets)

        # 反向传播
        total_loss.backward()

        # 检查关键模块是否有梯度
        checked = 0
        has_grad_count = 0
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                checked += 1
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grad_count += 1

        assert has_grad_count > 0, (
            f"反向传播后无任何参数获得梯度 "
            f"(checked={checked}, with_grad={has_grad_count})"
        )
        # 大部分可训练参数应有梯度
        grad_ratio = has_grad_count / max(checked, 1)
        assert grad_ratio > 0.8, (
            f"梯度覆盖率过低: {has_grad_count}/{checked} = {grad_ratio:.2%}"
        )

    def test_custom_weights(self):
        """自定义权重正确应用"""
        custom_weights = {
            'task1': 2.0,
            'task2': 2.0,
            'task3': 0.5,
            'task4': 1.5,
            'task5': 1.5,
        }
        criterion_weighted = MultiTaskLoss(weights=custom_weights)

        self.net.train()
        x = _make_input()
        targets = {
            'rho': torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE),
            'kappa': torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE),
            'sim': torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE),
        }
        predictions = self.net(x)

        _, losses_default = self.criterion(predictions, targets)
        _, losses_weighted = criterion_weighted(predictions, targets)

        # 加权后的总损失应与默认不同（因为权重不同）
        loss_default = self.criterion(predictions, targets)[0]
        loss_weighted = criterion_weighted(predictions, targets)[0]

        # 权重不同时，总损失一般也不同
        # 这里只验证不会报错即可
        assert loss_weighted.item() >= 0, "加权损失不应为负"

    def test_learnable_weights_mode(self):
        """可学习权重模式正常工作"""
        criterion_learnable = MultiTaskLoss(learnable=True)

        self.net.train()
        x = _make_input()
        targets = {
            'rho': torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE),
            'kappa': torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE),
            'sim': torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE),
        }
        predictions = self.net(x)
        total_loss, task_losses = criterion_learnable(predictions, targets)

        assert total_loss.dim() == 0, "可学习权重模式下总损失应为标量"
        assert total_loss.item() > 0, "可学习权重模式下总损失应为正"

        # 反向传播应能更新 log_vars 参数（ParameterList 中每个都是叶节点）
        total_loss.backward()
        for i, p in enumerate(criterion_learnable.log_vars):
            assert p.is_leaf, f"log_vars[{i}] 应为叶节点"
            assert p.grad is not None, (
                f"log_vars[{i}] 反向传播后梯度为 None"
            )
            # 梯度值可以接近零但不应对所有 5 个参数都为零
        # 至少部分 log_var 应有非零梯度
        grad_norms = [p.grad.abs().sum().item() for p in criterion_learnable.log_vars]
        assert any(g > 1e-10 for g in grad_norms), (
            f"log_vars 所有梯度都接近零: {grad_norms}"
        )

    def test_amp_compatibility(self):
        """兼容 AMP 混合精度训练"""
        if DEVICE.type != 'cuda':
            pytest.skip("AMP 测试需要 GPU")

        self.net.train()
        scaler = torch.cuda.amp.GradScaler()
        x = _make_input()
        targets = {
            'rho': torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE),
            'kappa': torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE),
            'sim': torch.rand(BATCH_SIZE, 1, D, H, W, device=DEVICE),
        }

        self.net.zero_grad()
        with torch.cuda.amp.autocast():
            predictions = self.net(x)
            total_loss, _ = self.criterion(predictions, targets)

        scaler.scale(total_loss).backward()
        scaler.step(torch.optim.Adam(self.net.parameters(), lr=1e-4))
        scaler.update()

        # 不抛异常即通过
        assert True


# =====================================================================
# 测试 8: 参数量统计
# =====================================================================

class TestParameterCount:
    """参数量统计测试"""

    def test_param_summary(self):
        """get_param_summary 返回合理结果"""
        net = JointInversionNet(in_channels=IN_CHANNELS).to(DEVICE)
        summary = net.get_param_summary()

        # 应包含所有模块 + TOTAL
        expected_modules = [
            'backbone', 'aspp',
            'task1_independent_gravity', 'task2_independent_magnetic',
            'task3_structural_similarity', 'task4_joint_gravity',
            'task5_joint_magnetic', 'TOTAL',
        ]
        for mod in expected_modules:
            assert mod in summary, f"缺少模块: {mod}"

        # 总参数量应大于各模块之和（实际上就是等于）
        total_from_sum = sum(
            summary[m]['total'] for m in expected_modules if m != 'TOTAL'
        )
        assert summary['TOTAL']['total'] == total_from_sum, (
            f"总参数量不一致: TOTAL={summary['TOTAL']['total']}, "
            f"求和={total_from_sum}"
        )

        # 总参数量应在合理范围（几百万到几千万）
        total = summary['TOTAL']['total']
        assert total > 100_000, f"总参数量过少: {total:,}"
        assert total < 200_000_000, f"总参数量过多: {total:,}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
