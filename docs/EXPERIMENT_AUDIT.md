# 实验审计报告 (Experiment Audit Report)

**日期**：2026-04-22
**审计员**：Claude (self-audit, cross-referenced with skill checklist)
**项目**：Fang et al. 2025 TGRS — 重磁联合反演论文复现

## 总体裁决：WARN

## 完整性状态：warn

## 检查项

### A. 真实标签来源：PASS

**详情**：
- `src/data/dataset.py:77-81`：GT（density, susceptibility, structural_sim）从 `data/*.npz` 文件加载
- `data/*.npz` 文件由 Phase 2 数据集生成脚本（`scripts/make_dataset.py` / `notebooks/make_dataset.ipynb`）生成
- 正演数据（gravity, magnetic）同样来自 .npz 文件，通过 Nagy/Bhattacharyja 公式计算
- **GT 不来自模型输出，不来自预测值派生**
- 训练/验证/测试划分在数据集生成阶段完成（`data/{train,val,test}_index.json`）

**证据**：`src/data/dataset.py:70-109`, `docs/DATASET_REPORT.md`

### B. 分数归一化：PASS

**详情**：
- `src/evaluate.py`：所有指标函数（IoU, MSE, MAE, R², SSIM, PSNR）均使用标准公式
- 无任何指标除以模型自身输出的 max/min/mean
- MSE = mean((pred - target)²) — 标准定义
- MAE = mean(|pred - target|) — 标准定义
- R² = 1 - SS_res/SS_tot — 标准定义
- IoU = intersection / union — 标准二值化定义
- 指标值未经过任何后处理"美化"

**证据**：`src/evaluate.py:20-227`

### C. 结果文件存在性：PASS

**详情**：

| 文件 | 存在 | 大小 |
|------|------|------|
| `results/full_training/checkpoints/best_model.pth` | ✅ | 93,469 KB |
| `results/full_training/training_history.json` | ✅ | 16.1 KB |
| `results/full_training/test_results.json` | ✅ | 0.3 KB |
| `results/metrics.json` | ✅ | 0.9 KB |
| `docs/RESULT_COMPARISON.md` | ✅ | 2.2 KB |
| `logs/training.log` | ✅ | 完整 |

**Test Results 数值一致性**：
- Test Total Loss: 0.082497 (log 中记录) ≈ 0.0825 (test_results.json) ✅
- Best Val Loss: 0.074656 (log 中记录)

### D. 死代码检测：PASS

**详情**：
- `evaluate.py` 中所有 6 个指标函数均在 `compute_all_metrics()` 中被调用
- `scripts/analyze_results.py` 在推理后调用 `compute_and_save_metrics()`
- 无未使用的评估代码路径

**证据**：`src/evaluate.py:230-286`, `scripts/analyze_results.py:115-155`

### E. 范围评估：WARN

**详情**：

| 项目 | 实际 | 论文声明 | 状态 |
|------|------|----------|------|
| 训练样本数 | 31,499 | ~31,500 | ✅ 接近 |
| 验证样本数 | 9,000 | 9,000 | ✅ 匹配 |
| 测试样本数 | 4,501 | 4,500 | ✅ 接近 |
| 指标计算样本 | 100 (子集) | 全量测试集 | ⚠️ 子集 |
| 训练 Epochs | 35 (早停) | ~90 | ⚠️ 早停 |
| 数据类型覆盖 | 6 类 | 6 类 | ✅ |

**警告原因**：
1. 指标计算仅使用 100 个测试样本（为加速 SSIM 计算），非完整 4500 样本
2. 早停于第 35 epoch（patience=15），论文训练约 90 epoch
3. Task 3 (BCE loss) 值偏高 (~0.07)，可能需要更多训练或学习率调整

### F. 评估类型：real_gt

**分类**：real_gt — 使用数据集提供的真实标签（密度、磁化率、结构相似性）
- GT 来自 Phase 2 合成的正演数据集（Nagy 重力公式 + Bhattacharyja 磁法公式）
- 观测数据（gravity, magnetic）由正演模型生成，作为网络输入
- 目标输出（density, susceptibility, structural_sim）作为监督信号

## 待办事项
- [ ] 用完整测试集（4500 样本）重新计算指标以获得更准确的数值
- [ ] 考虑调整早停 patience 或学习率策略以训练更多 epochs
- [ ] Task 3 BCE loss 偏高，检查结构相似性目标分布是否平衡

## 声明影响

| 声明 | 影响 | 状态 |
|------|------|------|
| 模型可训练并收敛 | supported | ✅ |
| 5 个任务均有独立损失 | supported | ✅ |
| 反演结果与 GT 可视化对比 | needs qualifier (仅 100 样本) | ⚠️ |
| 指标达到论文水平 | unsupported (需全量测试集验证) | ⚠️ |
| 训练曲线收敛趋势正确 | supported (loss 下降) | ✅ |
