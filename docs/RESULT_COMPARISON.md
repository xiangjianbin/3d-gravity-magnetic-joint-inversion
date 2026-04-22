# 结果对比报告 (Result Comparison Report)

## 1. 评估指标总览

| 指标 | Task1 (独立重力) | Task2 (独立磁法) | Task3 (结构相似性) | Task4 (联合重力) | Task5 (联合磁法) |
|------|-----------------|-----------------|-------------------|-----------------|-----------------|
| IoU  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| MSE  | 0.002477 | 0.001531 | 59.400005 | 0.002438 | 0.001584 |
| MAE  | 0.013279 | 0.011545 | 7.130451 | 0.012464 | 0.012640 |
| R²   | -112.5066 | -111.2549 | -296888.7188 | -110.6954 | -115.1391 |
| SSIM | 0.6146 | 0.2832 | -0.0028 | 0.6527 | 0.2616 |
| PSNR | 24.57 dB | 23.47 dB | -17.74 dB | 24.64 dB | 23.32 dB |

## 2. 训练结果摘要

- **总 Epoch 数**: 35 (早停于第 35 epoch)
- **Best Val Loss**: 0.080569
- **Final Train Loss**: 0.077017
- **Final Val Loss**: 0.080569
- **Test Loss**:

| Task | Test Loss |
|------|-----------|
| Task1 (独立重力) | 0.001642487808034852 |
| Task2 (独立磁法) | 0.0013554080911430864 |
| Task3 (结构相似性) | 0.06985342205072398 |
| Task4 (联合重力) | 0.001643897316407822 |
| Task5 (联合磁法) | 0.0013551621979742067 |

## 3. 模型参数量

| 组件 | 参数量 |
|------|--------|
| Backbone (2D U-Net) | 7,859,072 |
| ASPP (2D) | 79,200 |
| Task Heads (×5) | 208,485 |
| **总计** | **8,146,757** |

## 4. 生成的图表

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

---
*报告生成时间: 2026-04-22 09:45:05*
