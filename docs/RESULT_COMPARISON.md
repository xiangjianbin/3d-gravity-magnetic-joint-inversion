# 结果对比报告 (Result Comparison Report)

## 1. 训练状态警告

**训练未收敛**。本次完整训练存在严重问题，详见下方分析。

## 2. 评估指标总览（基于实际 full_training 结果）

| 指标 | Task1 (独立重力) | Task2 (独立磁法) | Task3 (结构相似性) | Task4 (联合重力) | Task5 (联合磁法) |
|------|-----------------|-----------------|-------------------|-----------------|-----------------|
| IoU  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| MSE  | 0.176204 | 0.064779 | 0.320773 | 0.033453 | 0.012268 |
| MAE  | 0.394363 | 0.247912 | 0.336695 | 0.149944 | 0.094762 |
| R²   | -7.5198 | -18.5212 | -0.5466 | -0.6175 | -2.6970 |
| SSIM | -0.0094 | -0.0100 | -0.0013 | 0.0091 | 0.0012 |
| PSNR | 1.52 dB | -2.09 dB | 4.94 dB | 8.73 dB | 5.13 dB |

## 3. 训练结果摘要

- **总 Epoch 数**: 16 (早停于第 16 epoch, patience=15)
- **Best Val Loss**: 0.972229
- **Final Train Loss**: 0.000000
- **Final Val Loss**: 0.972229
- **训练耗时**: ~4 秒（异常快速）
- **Train Loss 全为 0**: 模型权重未被更新

### Test Loss

| Task | Test Loss |
|------|-----------|
| T1(独立重力) | 0.176204 |
| T2(独立磁法) | 0.064779 |
| T3(结构相似性) | 0.684491 |
| T4(联合重力) | 0.033453 |
| T5(联合磁法) | 0.012268 |

## 4. 训练问题诊断

### 关键异常指标

| 项目 | 实际值 | 预期值 | 状态 |
|------|--------|--------|------|
| 训练样本数 | **14** | 31,500 | **CRITICAL** |
| 验证样本数 | **4** | 9,000 | **CRITICAL** |
| 测试样本数 | **2** | 4,500 | **CRITICAL** |
| Train Loss | **0.0000** (所有epoch) | 递减曲线 | **CRITICAL** |
| Val Loss | **恒定** 0.9722 | 递减曲线 | **CRITICAL** |
| 训练时间 | **4秒** | 数小时 | **CRITICAL** |

### 根因分析

1. **数据集路径错误**: 训练脚本加载了 smoke test 数据（14/4/2 样本）而非完整数据集（31500/9000/4500 样本）
2. **Train Loss = 0 的原因**: 可能是梯度未正确回传、loss 计算在 autocast 外部、或 optimizer.step() 未被调用
3. **Val Loss 恒定**: 因为模型参数从未更新，每次推理输出相同结果

### 模型参数量

| 组件 | 参数量 |
|------|--------|
| Backbone (2D U-Net) | 7,859,072 |
| ASPP (2D) | 79,200 |
| Task Heads (×5) | 208,485 |
| **总计** | **8,146,757** |

## 5. 生成的图表

> 注：以下图表基于**未收敛模型**的输出，仅作诊断参考。

### 训练曲线
![训练曲线](results/figures/training_curves.svg)

### 反演结果对比
- 密度真值: ![density GT](results/figures/inversion_density_gt.svg)
- 密度预测: ![density Pred](results/figures/inversion_density_pred.svg)
- 磁化率真值: ![suscept GT](results/figures/inversion_suscept_gt.svg)
- 磁化率预测: ![suscept Pred](results/figures/inversion_suscept_pred.svg)
- 结构相似性真值: ![struct sim GT](results/figures/inversion_struct_sim_gt.svg)
- 结构相似性预测: ![struct sim Pred](results/figures/inversion_struct_sim_pred.svg)

### 深度切片对比
- 密度切片: ![density slices](results/figures/slice_depth_comparison_density.svg)
- 磁化率切片: ![suscept slices](results/figures/slice_depth_comparison_suscept.svg)

### 散点图
- GT vs Pred: ![scatter](results/figures/scatter_gt_vs_pred.svg)

### 指标汇总
![metrics summary](results/figures/metrics_summary.svg)

---

*报告生成时间: 2026-04-22 (基于实际 full_training 结果)*
*状态: [FAILED] — 训练未收敛，需要修复数据加载和训练循环*
