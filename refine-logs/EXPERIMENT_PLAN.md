# 实验计划

基于 PAPER_ANALYSIS_DETAILED.md 生成的声明驱动实验计划。

## 复现目标清单
| # | 目标 | 来源 | 优先级 |
|---|------|------|--------|
| 1 | 数据集 45000 样本，6 类地质模型 | Table I | MUST |
| 2 | 网络结构与论文一致：3D U-Net + ASPP + 5 任务头 | Section II-A | MUST |
| 3 | 训练收敛到类似 loss 曲线 | Fig. 5 | MUST |
| 4 | 测试模型反演精度（8 个合成模型） | Table III, Fig.8 | MUST |
| 5 | 深度切片对比（60m, 200m） | Fig. 11 | MUST |
| 6 | 预测 vs 观测异常对比 | Fig. 13 | MUST |
| 7 | IoU/MSE/MAE/R²/SSIM/PSNR 全指标计算 | 自定义评估 | MUST |
| 8 | 全部 SVG 可视化图（数据集+结果+论文 Figure 复现） | CLAUDE.md | MUST |

## 运行顺序
```
Phase 0  →  论文深度解剖                          →  docs/PAPER_ANALYSIS_DETAILED.md     ✓ DONE
Phase 1  →  实验计划生成                          →  refine-logs/EXPERIMENT_PLAN.md       ← 当前
Phase 2  →  数据集生成 + 全面可视化（SVG）         →  notebooks/make_dataset.ipynb          ← 下一步
Phase 3  →  模型代码实现（并行 Agents ≤ 2）        →  docs/MODEL_REPORT.md + train.ipynb
Phase 4  →  GPU 检查与分配                        →  docs/GPU_ALLOC_PLAN.md
Phase 5a →  冒烟测试                              →  SMOKE_TEST_REPORT.md
Phase 5b →  完整训练                              →  results/checkpoints/best_model.pth
Phase 5c →  按需修复                              →  BUG_FIX_LOG.md
Phase 6  →  结果分析 + 全面可视化                  →  docs/RESULT_COMPARISON.md
Phase 7  →  图表生成（论文 Figure 复现 SVG）        →  figures/fig*.svg
Phase 8  →  实验审计                              →  EXPERIMENT_AUDIT.md
Phase 9  →  实验总结报告                          →  docs/FINAL_SUMMARY.md
Phase 10 →  Git 归档                              →  tag v1.0-reproduction-complete
```

## 关键参数汇总（从论文提取）

### 网络架构
- 输入: (2, 81, 81) — [重力异常, 磁力异常] × 观测面网格
- 骨干: 3D U-Net, 4层编码器 + 4层解码器 + skip connections
- ASPP: rates=[6,12,18,24] + GlobalAvgPool, 融合后 40ch
- 5 个任务头: 独立重力(MSE), 独立磁法(MSE), 结构相似性(BCE+Sigmoid), 联合重力(MSE), 联合磁法(MSE)
- 输出: 每个 (40, 40, 20) — 地下网格密度/磁化率/结构相似性

### 训练超参
- 优化器: Adam (β₁=0.9, β₂=0.999)
- 学习率: 初始 ~1e-3（自适应衰减）
- Epochs: ~90
- Batch size: 待 GPU 实测确定（目标显存 70~80%）
- Loss: MSE(回归) + BCE(Task3分类) + Leaky-ReLU正则化

### 数据集
- 总样本: 45,000 (训练 31,500 / 验证 9,000 / 测试 4,500)
- 网格: 40×40×20, 间距 20m, 深度 0-400m
- 观测面: 81×81, 高度地表+10m
- 噪声: 重力 0.5%, 磁法 1%
- 6 类模型: Type1(5000) + Type2(5000) + Type3(10000) + Type4(7500) + Type5(7500) + Type6(10000)

### 正演公式
- 重力: Talwani et al. (1959) 棱柱体公式
- 磁法: Bhattacharyya (1964) 或 Sharma (1986) 棱柱体磁异常公式
- 磁化方向: 地磁场倾角/偏角按论文设置

## 声明映射（Claim → 实验）

| Claim ID | 声明内容 | 验证方法 | 通过标准 |
|----------|---------|---------|---------|
| C1 | 网络能有效提取结构相似性 | Task 3 BCE loss 收敛 | val BCE < 0.15 |
| C2 | 联合反演优于独立反演 | 对比 Task 1/4 和 Task 2/5 的 loss | joint loss < independent loss |
| C3 | 在合成数据上达到高精度 | 8 个测试模型的 IoU > 0.7 | IoU ≥ 0.70 |
| C4 | 深度切片吻合良好 | Fig.11 切片视觉对比 + SSIM | SSIM > 0.85 |
| C5 | 预测异常与观测一致 | Fig.13 散点图 R² | R² > 0.95 |
