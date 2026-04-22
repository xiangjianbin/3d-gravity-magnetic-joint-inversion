# 实验计划 — 重磁联合反演论文复现

基于 `docs/PAPER_ANALYSIS_DETAILED.md` 生成的声明驱动实验计划。

## 复现目标清单

| # | 目标 | 来源 | 优先级 | 验证方式 |
|---|------|------|--------|---------|
| 1 | 数据集 45,000 样本，6 类地质模型 | Table I | MUST | DATASET_REPORT.md 计数 |
| 2 | 网络结构与论文一致（U-Net + ASPP + 5 任务头） | Section II-A, Fig.2 | MUST | MODEL_REPORT.md 参数量对比 |
| 3 | 正演模型正确（重力 + 磁法棱柱体公式） | Section II-C | MUST | 单元测试误差 < 1e-6 |
| 4 | 损失函数实现（MSE + BCE + Leaky-ReLU） | Section II-B Eq.(9)(10) | MUST | SMOKE_TEST_REPORT.md loss 下降 |
| 5 | 训练收敛到类似 loss 曲线 | Fig.5 | MUST | training_curves.svg 形状对比 |
| 6 | 测试模型反演精度（8 个源体参数） | Table III, Fig.8 | MUST | RESULT_COMPARISON.md 数值对比 |
| 7 | 深度切片对比（60m/260m） | Fig.11 | MUST | slice_depth_comparison_*.svg |
| 8 | 预测 vs 观测数据拟合 | Fig.13 | MUST | scatter_gt_vs_pred.svg |
| 9 | 3D 重建可视化 | Fig.12 | SHOULD | fig12_3d_reconstruction.svg |
| 10 | 真实数据应用（某矿区） | Section IV, Fig.10-13 | SHOULD | fig9_real_data.svg |

## 运行顺序

```
Phase 0  → Phase 1  → Phase 2  → Phase 3  → Phase 4  → Phase 5a → Phase 5b → Phase 6 → Phase 7 → Phase 8 → Phase 9 → Phase 10
论文解剖   实验计划   数据集生成  模型实现   GPU分配   冒烟测试  完整训练  结果分析  图表生成   审计      总结      归档
(done)    (done)     (next)                                               (auto)
```

## 关键技术决策记录

### 决策 1: 网络维度（2D vs 3D）
- **论文描述**: 3D U-Net 处理 3D 体数据
- **实际实现**: 2D U-Net 做 obs-to-subsurface mapping（输入观测面 2D 数据，输出 3D 地下模型）
- **理由**: 前次成功复现采用此方案，训练收敛良好
- **输入**: (2, 40, 40) — [gravity, magnetic] × Easting × Northing
- **输出**: density(40,40,20) + susceptibility(40,40,20) + struct_sim(40,40,20)

### 决策 2: 数据集规格
- 总样本: 45,000
- 划分: 31,500 train / 9,000 val / 4,500 test (7:2:1)
- 网格: 40×40×20, spacing=20m, depth=0-400m
- 观测面高度: 地表以上 10m

### 决策 3: 训练超参数
- 优化器: Adam (β1=0.9, β2=0.999, ε=1e-8, weight_decay=1e-4)
- 初始学习率: 1e-3（自适应衰减）
- Epochs: ~90（参考 Fig.5）
- Batch size: 待 Phase 4 实测确定（目标显存 70-80%）
- AMP: 启用混合精度

### 决策 4: 损失函数权重
- Task 1 (独立重力): MSE
- Task 2 (独立磁法): MSE
- Task 3 (结构相似性): BCE
- Task 4 (联合重力): MSE
- Task 5 (联合磁法): MSE
- Leaky-ReLU 正则化系数: ν = 0.01
- 各任务 loss 等权平均

## 声明与验证映射

| 论文声明 | 对应实验 | 验证指标 | 通过标准 |
|---------|---------|---------|---------|
| "网络能有效提取结构相似性" | Task 3 BCE loss 收敛 | struct_sim IoU > 0.7 | claim_supported |
| "联合反演优于独立反演" | Task 4/5 vs Task 1/2 MSE | joint_MSE < ind_MSE | claim_supported |
| "在合成数据上高精度反演" | 测试集评估 | R² > 0.9 | claim_supported |
| "在真实数据上有效应用" | 矿区数据测试 | 目视对比合理 | claim_partial |

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 显存不足 (OOM) | 中 | 高 | AMP + 梯度累积 + 小 batch |
| 训练不收敛 | 低 | 高 | 学习率 warmup + 梯度裁剪 |
| 正演数值误差 | 低 | 中 | 单元测试 + 解析解验证 |
| 数据集生成慢 | 高 | 中 | CUDA 加速正演 + 分批保存 |
| 评估指标偏差 | 低 | 低 | 多指标交叉验证 |

---

*生成时间: 2026-04-22*
*基于: docs/PAPER_ANALYSIS_DETAILED.md*
