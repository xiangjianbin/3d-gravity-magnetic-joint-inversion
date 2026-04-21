# 模型架构实现报告: 5 任务头 + 完整网络组装

> **Agent**: 模型架构专家 (Task 3B)
> **日期**: 2026-04-21
> **状态**: 全部测试通过

---

## 1. 实现文件清单

| 文件 | 说明 | 行数 |
|------|------|------|
| `src/model/task_heads.py` | 5 个任务头定义 + 辅助组件 | ~310 |
| `src/model/joint_inversion_net.py` | 完整网络组装 | ~200 |
| `src/model/loss_functions.py` | 多任务损失函数（固定/可学习权重） | ~220 |
| `tests/test_heads.py` | 单元测试 (23 个用例) | ~470 |
| `src/model/backbone_unet3d.py` | Backbone U-Net (Agent 3A 实现) | ~250 |
| `src/model/aspp.py` | ASPP 模块 (Agent 3A 实现) | ~180 |

## 2. 参数量统计

| 模块 | 参数量 | 占比 | 说明 |
|------|--------|------|------|
| **backbone** (BackboneUNet3d) | 94,578,048 (94.58M) | 90.7% | 4 层编解码器，参数主体 |
| **aspp** (ASPP3d) | 5,638,656 (5.64M) | 5.4% | 4 分支空洞卷积 + 融合 |
| **task1** (IndependentGravity) | 1,743,615 (1.74M) | 1.7% | 独立重力反演解码器 |
| **task2** (IndependentMagnetic) | 1,743,615 (1.74M) | 1.7% | 独立磁法反演解码器 |
| **task3** (StructuralSimilarity) | 57,249 (0.06M) | 0.05% | 结构相似性提取模块 |
| **task4** (JointGravity) | 220,367 (0.22M) | 0.2% | 联合重力反演解码器 |
| **task5** (JointMagnetic) | 220,367 (0.22M) | 0.2% | 联合磁法反演解码器 |
| **TOTAL** | **104,201,917 (104.20M)** | 100% | — |

### 显存估算

- FP32 权重显存: ~795 MB
- 激活值显存估算 (batch_size=2): ~3180 MB
- RTX 3060 6GB 可用显存 (~4GB): 需要使用 AMP + gradient checkpointing 才能训练

## 3. 各任务头详细设计

### Task 1 & 2: 独立反演头

```
输入: ASPP 特征 (B, 256, 40, 40, 20)
结构: 4 层小型解码器 (_MiniDecoder)
  Layer 1: Conv3d(256→128) ×2 + BN + LeakyReLU
  Layer 2: Conv3d(128→64)  ×2 + BN + LeakyReLU
  Layer 3: Conv3d(64→32)   ×2 + BN + LeakyReLU
  Layer 4: Conv3d(32→1)    ×2 + BN + LeakyReLU
输出: (B, 1, 40, 40, 20), 连续值, MSE 监督
参数量: 每个 1.74M (两个共 3.48M)
```

### Task 3: 结构相似性模块

```
输入: [rho_pred, kappa_pred] 各 (B, 1, 40, 40, 20)
结构:
  Step 1: 分别 Conv3d(1→32, k=3) + BN + LeakyReLU → (B, 32, D, H, W) × 2
  Step 2: Concat → (B, 64, D, H, W)
  Step 3: MaxPool3d(2) × 2 → (B, 64, D/4, H/4, W/4)
  Step 4: 径向网格替代 Conv3d(64→32, k=3) + BN + LeakyReLU (Gap 6)
  Step 5: F.interpolate(trilinear) 回原始尺寸 → (B, 32, D, H, W)
  Step 6: Conv3d(32→1, k=1) + Sigmoid → (B, 1, D, H, W), 值域 [0, 1]
输出: S ∈ [0, 1], MSE 监督
参数量: 57K (最轻量的模块)
物理含义: S≈1 表示密度和磁化率同时非零（结构一致）
```

### Task 4 & 5: 联合反演头

```
输入: concat[原始数据(B,2,D,H,W), S(B,1,D,H,W)] → (B, 3, D, H, W)
结构: 4 层小型解码器 (_MiniDecoder)
  Layer 1: Conv3d(3→64)   ×2 + BN + LeakyReLU
  Layer 2: Conv3d(64→32)  ×2 + BN + LeakyReLU
  Layer 3: Conv3d(32→16)  ×2 + BN + LeakyReLU
  Layer 4: Conv3d(16→1)   ×2 + BN + LeakyReLU
输出: (B, 1, 40, 40, 20), logits (未过 Sigmoid), BCEWithLogitsLoss 监督
参数量: 每个 220K (两个共 440K)
注意: 推理时只使用 Task 4/5 的输出作为最终预测结果
```

## 4. 损失函数设计

### 固定权重版本 (默认)

```python
Total_Loss = 1.0 * MSE(rho_pred, rho_gt)        # Task 1
           + 1.0 * MSE(kappa_pred, kappa_gt)     # Task 2
           + 1.0 * MSE(S, S_gt)                  # Task 3
           + 1.0 * BCE(rho_final, rho_gt)        # Task 4 (ω=1)
           + 1.0 * BCE(kappa_final, kappa_gt)    # Task 5 (ω=1)
```

### 可选: 不确定性加权版本 (learnable=True)

基于 Kendall et al. (2018) 的 homoscedastic uncertainty:
```
L_total = Σ (exp(-log_var_i) * L_i + log_var_i)
```
每个任务的 log_var_i 为可学习标量参数。

### 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| Task 4/5 输出 Sigmoid? | 否 (输出 logits) | BCEWithLogitsLoss 数值更稳定 |
| Task 3 输出 Sigmoid? | 是 (内部含 Sigmoid) | 物理意义要求值域 [0,1] |
| Gap 6 (径向网格)? | Conv3d 替代 | 先跑通 pipeline，后续可优化 |

## 5. 测试结果

```
========================= test session starts ==========================
tests/test_heads.py::TestTask1IndependentGravity::test_output_shape        PASSED
tests/test_heads.py::TestTask1IndependentGravity::test_no_nan             PASSED
tests/test_heads.py::TestTask2IndependentMagnetic::test_output_shape      PASSED
tests/test_heads.py::TestTask2IndependentMagnetic::test_independent_params PASSED
tests/test_heads.py::TestTask3StructuralSimilarity::test_output_shape     PASSED
tests/test_heads.py::TestTask3StructuralSimilarity::test_sigmoid_range    PASSED
tests/test_heads.py::TestTask3StructuralSimilarity::test_sigmoid_range_extreme_input PASSED
tests/test_heads.py::TestTask3StructuralSimilarity::test_different_input_produces_different_output PASSED
tests/test_heads.py::TestTask4JointGravity::test_output_shape            PASSED
tests/test_heads.py::TestTask4JointGravity::test_no_sigmoid_in_output     PASSED
tests/test_heads.py::TestTask5JointMagnetic::test_output_shape            PASSED
tests/test_heads.py::TestTask5JointMagnetic::test_task4_and_task5_independent PASSED
tests/test_heads.py::TestFullNetworkForward::test_training_mode_returns_all_five PASSED
tests/test_heads.py::TestFullNetworkForward::test_inference_mode_returns_two PASSED
tests/test_heads.py::TestFullNetworkForward::test_return_all_override      PASSED
tests/test_heads.py::TestFullNetworkForward::test_no_nan_in_outputs       PASSED
tests/test_heads.py::TestFullNetworkForward::test_structural_sim_range    PASSED
tests/test_heads.py::TestMultiTaskLoss::test_loss_is_scalar               PASSED
tests/test_heads.py::TestMultiTaskLoss::test_loss_gradients              PASSED
tests/test_heads.py::TestMultiTaskLoss::test_custom_weights              PASSED
tests/test_heads.py::TestMultiTaskLoss::test_learnable_weights_mode      PASSED
tests/test_heads.py::TestMultiTaskLoss::test_amp_compatibility           SKIPPED (无GPU)
tests/test_heads.py::TestParameterCount::test_param_summary               PASSED

================= 22 passed, 1 skipped in 132.62s ===================
```

## 6. 与论文的对齐情况

| 论文要素 | 实现状态 | 备注 |
|----------|----------|------|
| Fig.2 Backbone U-Net | 已实现 (Agent 3A) | 4 层编解码 + skip connection |
| Fig.3 独立反演子网络 | 已实现 | 4 层解码器, 256→128→64→32→1 |
| Fig.4 ASPP 模块 | 已实现 (Agent 3A) | dilation {6,12,18} + global pool |
| Task 3 结构相似性 | 已实现 | Gap 6 用 Conv3d 替代径向网格 |
| Task 4/5 联合反演 | 已实现 | 3ch 输入 → 4 层解码器 |
| Eq.9 MSE Loss (T1-T3) | 已实现 | nn.MSELoss |
| Eq.10 BCE Loss (T4-T5) | 已实现 | nn.BCEWithLogitsLoss |
| 训练/推理双模式 | 已实现 | return_all 参数控制 |

## 7. 下一步建议

1. **数据管道对接**: 确保 dataset.py 输出的 tensor shape 与网络输入匹配 `(B, 2, 40, 40, 20)`
2. **AMP 训练验证**: 在 GPU 上运行 `test_amp_compatibility` 测试
3. **Gradient Checkpointing**: 启用 `use_gradient_checkpointing=True` 后重新跑冒烟测试
4. **Gap 6 优化**: 如果 Task 3 效果不佳，考虑实现真正的径向网格操作
5. **学习率调度**: 实现论文 Eq.12 的自定义调度器或使用 CosineAnnealingLR
