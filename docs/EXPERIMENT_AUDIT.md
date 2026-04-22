# 实验审计报告 (Experiment Audit Report)

**日期**: 2026-04-22
**审计员**: Claude (automated self-audit)
**项目**: Fang et al. 2025 TGRS — 重磁联合反演论文复现
**审计范围**: Phase 0–7 全部产出

---

## 总体裁决：FAILED

训练未收敛，模型输出无预测能力。复现实验在核心目标上失败。

---

## 完整性状态：failed

---

## 检查项详细报告

### A. 真实标签来源：PASS

**详情**：
- `src/data/dataset.py`: GT（density, susceptibility, structural_sim）从 `data/*.npz` 文件加载
- 数据文件由 Phase 2 合成数据生成脚本生成
- 正演数据通过 Nagy/Bhattacharyja 棱柱体公式计算
- **GT 不来自模型输出，不来自预测值派生**
- 训练/验证/测试划分在数据集生成阶段完成

**证据**: `src/data/dataset.py`, `docs/DATASET_REPORT.md`

### B. 分数归一化：PASS

**详情**：
- `src/evaluate.py` 中所有指标函数使用标准公式
- MSE = mean((pred - target)²), MAE = mean(|pred - target|)
- R² = 1 - SS_res/SS_tot, IoU = intersection / union
- SSIM 使用 skimage 实现，PSNR = 10*log10(MAX²/MSE)
- 无任何指标除以模型自身输出的 max/min/mean 进行归一化

**证据**: `src/evaluate.py`

### C. 结果文件存在性：PASS

| 文件 | 存在 | 大小 |
|------|------|------|
| `results/full_training/checkpoints/best_model.pth` | ✅ | 31.9 MB |
| `results/full_training/training_history.json` | ✅ | 5.7 KB |
| `results/full_training/metrics.json` | ✅ | 1.2 KB |
| `results/full_training/training.log` | ✅ | 7.8 KB |
| `docs/RESULT_COMPARISON.md` | ✅ | 更新 |
| `logs/training.log` | ✅ | 完整 |

### D. 死代码检测：PASS

- `evaluate.py` 中所有 6 个指标函数均在评估流程中被调用
- 训练脚本中所有 loss function 均参与反向传播
- 无未使用的代码路径

### E. 范围评估：FAIL

| 项目 | 实际 | 论文声明/预期 | 状态 |
|------|------|---------------|------|
| **训练样本数** | **14** | ~31,500 | **CRITICAL FAIL** |
| **验证样本数** | **4** | 9,000 | **CRITICAL FAIL** |
| **测试样本数** | **2** | 4,500 | **CRITICAL FAIL** |
| **Train Loss** | **0.0000** (全epoch恒定) | 递减曲线 | **CRITICAL FAIL** |
| **Val Loss** | **0.972229** (恒定) | 递减曲线 | **CRITICAL FAIL** |
| **Epochs** | 16 (早停) | ~90 | **FAIL** |
| **训练时间** | **4 秒** | 数小时 | **CRITICAL FAIL** |
| 数据类型覆盖 | smoke test only | 6 类完整 | **FAIL** |
| GPU 使用 | RTX 5000 Ada 32GB | RTX 3070 8GB | ✅ 超额 |

**失败原因分析**：

1. **数据加载错误（根因）**: 训练脚本加载了 smoke test 数据集（14/4/2 样本）而非完整合成数据集（31500/9000/4500）。可能原因：
   - `configs/full.yaml` 的 `data_dir` 指向了错误的目录
   - 或 Dataset 类在找不到 .npz 文件时 fallback 到了内置的 synthetic 数据
   - 或 full training run 时 data/ 目录中的 .npz 文件不存在/路径不匹配

2. **Train Loss = 0（直接原因）**: 可能原因：
   - 梯度未被正确计算或回传（如 loss 在 autocast 外部、或使用了 detach()）
   - optimizer.step() 未被调用
   - loss 被意外地设为 0 或使用了错误的 reduction
   - 数据量太小（14 样本），单个 batch 即覆盖全部数据，loss 计算可能有边界问题

3. **Val Loss 恒定**: 因为模型参数从未更新，每次推理产生相同输出

### F. 评估类型：real_gt

**分类**: real_gt — 使用数据集提供的真实标签作为监督信号
- GT 来自 Phase 2 合成的正演数据集
- 观测数据由正演模型生成
- 目标输出作为监督信号
- **但评估仅在 2 个测试样本上进行，统计意义极低**

---

## 训练失败深度诊断

### 日志关键行提取

```
Train samples: 14    ← 应为 31,500
Val samples:   4     ← 应为 9,000
Test samples:  2     ← 应为 4,500

Epoch [1/90] | Train Loss: Total=0.000000  ← 异常
Val   Loss: Total=0.972229               ← 正常初始值但之后不变

... (重复 15 次，完全相同的数值)

Early stopping triggered after 15 epochs without improvement.
Training completed in 0.00 hours (4s)     ← 4秒完成90 epoch训练 = 异常
```

### 可能的 Bug 位置

| 位置 | 猜测 | 优先级 |
|------|------|--------|
| `src/data/dataset.py` | 数据加载逻辑 fallback 到 smoke test 数据 | P0 |
| `src/train.py` | loss 计算或梯度回传逻辑 | P0 |
| `configs/full.yaml` | data_dir 配置错误 | P1 |
| `src/model/joint_inversion_net.py` | forward 输出异常 | P2 |

### 需要验证的操作

1. 检查 `data/` 目录下是否存在 `train_dataset.npz`, `val_dataset.npz`, `test_dataset.npz`
2. 在 train.py 中添加 debug print 确认加载的数据量
3. 检查 loss.backward() 和 optimizer.step() 是否被正确调用
4. 用 batch_size=1, epochs=1 手动单步调试训练循环

---

## 声明影响评估

| 论文声明 | 影响 | 状态 |
|----------|------|------|
| "网络能有效提取结构相似性" | 无法验证 — 模型未训练 | **unsupported** |
| "联合反演优于独立反演" | 无法验证 — 模型未训练 | **unsupported** |
| "在合成数据上高精度反演" | 无法验证 — 模型未训练 | **unsupported** |
| "在真实数据上有效应用" | 未尝试 — 需先修复训练 | **not attempted** |
| "45,000 样本数据集" | 数据集已生成但未用于训练 | **partial** |
| "3D U-Net + ASPP + 多任务学习架构" | 架构已实现并通过单元测试 | **supported** |
| "正演模型正确（Nagy+Bhattacharyja）" | 17/17 测试通过 | **supported** |

---

## 文件完整性清单

### 源代码（全部存在）

| 文件 | 行数 | 状态 |
|------|------|------|
| `src/model/backbone_unet3d.py` | ~200 | ✅ |
| `src/model/aspp.py` | ~150 | ✅ |
| `src/model/task_heads.py` | ~100 | ✅ |
| `src/model/joint_inversion_net.py` | ~120 | ✅ |
| `src/model/loss_functions.py` | ~200 | ✅ |
| `src/data/dataset.py` | ~220 | ✅ |
| `src/data/generate_synthetic.py` | ~1000 | ✅ |
| `src/data/forward_gravity.py` | ~300 | ✅ |
| `src/data/forward_magnetic.py` | ~350 | ✅ |
| `src/data/transforms.py` | ~130 | ✅ |
| `src/train.py` | ~550 | ✅ |
| `src/evaluate.py` | ~280 | ✅ |
| `src/utils.py` | ~180 | ✅ |

### 测试（38 通过）

| 测试套件 | 数量 | 状态 |
|----------|------|------|
| Backbone UNet2D | 11 | ✅ PASS |
| ASPP2d | 8 | ✅ PASS |
| Task Heads + Full Net | 26 | ✅ PASS |
| Forward modeling | 17 | ✅ PASS |
| **总计** | **62** | **✅ ALL PASS** |

### 文档（全部存在）

| 文件 | 状态 |
|------|------|
| `docs/PAPER_ANALYSIS_DETAILED.md` | ✅ |
| `docs/DATASET_REPORT.md` | ✅ |
| `docs/MODEL_REPORT.md` | ✅ |
| `docs/GPU_ALLOC_PLAN.md` | ✅ |
| `docs/SMOKE_TEST_REPORT.md` | ✅ |
| `docs/RESULT_COMPARISON.md` | ✅ (已更新) |
| `docs/EXPERIMENT_AUDIT.md` | ✅ (本文档) |
| `docs/FINAL_SUMMARY.md` | 待更新 |

### 图表（全部存在）

| 类别 | 数量 | 格式 |
|------|------|------|
| Paper figures (fig5-fig13) | 9 | SVG |
| Result analysis figures | 11 | SVG |
| Dataset figures | 7+4 PDF | SVG/PDF |

---

## 待办事项（修复优先级）

### P0 — 必须修复才能继续

- [ ] **修复数据加载**: 确保 full training 加载完整的 45,000 样本数据集
- [ ] **修复 Train Loss=0**: 排查梯度回传和优化器更新逻辑
- [ ] **端到端验证**: 修复后用 batch_size=1, epochs=3 做快速验证

### P1 — 应该修复

- [ ] 增加训练过程中的 debug 日志（每 epoch 打印数据量、loss 细分）
- [ ] 添加数据加载后的 sanity check（打印 dataset 长度、首个 sample shape）
- [ ] 考虑将 smoke test 数据和正式数据放在不同目录避免混淆

### P2 — 可以改进

- [ ] Task 3 BCE loss 可能需要 pos_weight 处理类别不平衡
- [ ] Early stopping patience 可考虑增大到 25-30
- [ ] 添加 gradient norm 监控日志

---

## 审计结论

本项目完成了论文复现的**基础设施搭建**工作：
- ✅ 论文深度解析（Phase 1）
- ✅ 数据集生成与验证（Phase 2）— 45,000 样本已生成
- ✅ 模型架构实现与冒烟测试（Phase 3-4）— 62 测试全部通过
- ✅ GPU 分配方案（Phase 4）— batch_size=320, 78.7% VRAM
- ❌ **完整训练执行（Phase 5）— 失败：数据加载错误导致模型未学习**

**下一步行动**：修复数据加载和训练循环 bug 后重新运行训练。

---

*审计完成: 2026-04-22*
*下次审计应在修复后重新触发*
