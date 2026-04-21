# 3D 重磁联合反演论文复现项目

## 项目概述

复现论文: **Improved 3D Joint Inversion of Gravity and Magnetic Data Based on Deep Learning With a Multitask Learning Strategy**

核心方法: 3D U-Net + 多任务学习 (Multitask Learning) 联合反演重力异常和磁异常数据。

## GPU 环境

- **GPU**: NVIDIA RTX 5000 Ada Generation (32GB VRAM)
- **可用显存**: ~30GB (当前占用 ~1.3GB)
- **CUDA**: 12.2
- **驱动**: 535.230.02
- **关键约束**: 使用混合精度(AMP)，batch_size 可用 4-8（显存充裕）

## 项目结构

```
src/
├── model/           # 模型定义 (3D U-Net + 多任务头)
│   ├── unet3d.py   # 3D U-Net 编解码器
│   ├── multitask_head.py  # 双任务输出头
│   └── loss_functions.py  # 损失函数
├── data/            # 数据处理模块
│   ├── dataset.py   # PyTorch Dataset
│   ├── generate_synthetic.py  # 合成数据生成器
│   └── transforms.py # 数据变换
├── train.py         # 训练入口
├── evaluate.py      # 评估脚本
└── utils.py         # 工具函数
data/                # 数据集 (.gitignore)
results/             # 实验结果
figures/             # 图表输出
checkpoints/         # 模型权重 (.gitignore)
configs/             # YAML 配置文件
docs/                # Agent 间文档传递目录
```

## 关键技术参数（来自论文）

### 模型架构
- 骨干网络: 3D U-Net
- 输入: 2通道 3D 体数据 [重力异常 Δg, 磁异常 ΔT]
- 输出: 2通道 3D 体数据 [密度模型 ρ, 磁化率模型 κ]
- 多任务策略: 共享编码器 + 独立解码器/任务头

### 数据规格
- 合成数据: 随机地质体模型 + 正演计算
- 正演公式: 论文 Section 2/3 的重力/磁法正演方程
- 网格: 按论文规定的 3D 网格分辨率
- 噪声: 高斯噪声，多个 σ 水平

### 训练配置
- 优化器: Adam (按论文)
- 损失: MSE + 多任务加权
- 评估指标: MSE, RMSE, MAE, Correlation Coefficient

## Pipeline 执行流程（8 Phase）

本项目采用 **Skill + Task Agent 混合调度** 方式执行:

1. **Phase 0**: 项目初始化 + GitHub 连接 → `REPRODUCTION_STATE.json`
2. **Phase 1**: 论文深度解析 → `docs/PAPER_ANALYSIS.md` + `docs/ASSUMPTIONS_AND_GAPS.md`
3. **Phase 2**: 数据集生成+验证+可视化 → `data/` + `docs/DATASET_REPORT.md` + `figures/dataset_*.pdf`
4. **Phase 3**: 模型架构实现+冒烟测试 → `src/` + `docs/MODEL_REPORT.md` + `docs/SMOKE_TEST_REPORT.md`
5. **Phase 4**: GPU检查+配置优化 → `docs/GPU_ALLOC_PLAN.md` + `configs/full.yaml`
6. **Phase 5**: 训练执行+自动修复 → `checkpoints/` + `docs/TRAINING_LOG.md`
7. **Phase 6**: 结果对齐+图表生成 → `docs/RESULT_COMPARISON.md` + `figures/result_*.pdf`
8. **Phase 7**: 实验审计+最终报告 → `docs/EXPERIMENT_AUDIT.md` + `docs/FINAL_REPORT.md`

详细方案见: `论文复现项目级技能使用方案.md`

---

## 主会话应该做的事

```
✅ 读每个阶段输出的 .md 报告（摘要级别）
✅ 判断当前阶段是否通过（pass/fail/need-fix）
✅ 决定下一个阶段启动哪些 Agent、几个 Agent
✅ 在 Agent 之间传递关键信息（通过告诉 Agent 读哪个文件）
✅ 处理需要人工决策的异常情况
✅ 维护全局进度状态（REPRODUCTION_STATE.json）
✅ 每个 Phase 结束后 git commit
```

## 主会话不应该做的事

```
❌ 读完整的源代码文件（让 Agent 自己读）
❌ 读完整的训练日志（让 Agent 读日志文件）
❌ 读 Agent 的中间调试输出（只要最终报告）
❌ 同时管理超过 3 个活跃 Agent
❌ 替 Agent 做具体的代码编写/调试
❌ 向 Agent 返回大段代码或日志
```

## Agent 调度规则

### Agent 职责
- 接收输入文档 → 完成工作 → 写输出文档 → 返回精简摘要
- 代码错误自动修复（最多 3 次重试）
- 不向主会话返回大段代码或日志

### 文档传递规则
- Agent 之间通过 `docs/*.md` 文件通信
- 每个 Agent 完成后必须写 REPORT.md
- 下一个 Agent 读上一个的 REPORT.md 作为输入

## 代码规范

### 显存优化（强制）
```python
# 所有训练代码必须包含:
import torch.cuda.amp as amp
scaler = amp.GradScaler()

# 混合精度前向传播
with amp.autocast():
    output = model(input)
    loss = criterion(output, target)

# 梯度缩放反向传播
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 随机性控制（强制）
```python
# 所有脚本开头必须包含:
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 结果保存格式（强制）
```python
# 评估结果必须保存为 JSON:
import json
results = {
    "metrics": {...},
    "config": {...},
    "timestamp": "...",
    "gpu_info": "RTX 5000 Ada 32GB"
}
with open("results/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
```

## 常用 Skill 映射

| 任务 | Skill | 触发词 |
|------|-------|--------|
| 分析实验结果 | /analyze-results | "分析结果" |
| 绘制图表 | /paper-figure | "画图"、"作图" |
| 运行实验 | /run-experiment | "跑实验" |
| 训练监控 | /training-check | "检查训练" |
| 实验审计 | /experiment-audit | "审计实验" |
| 结果到声明 | /result-to-claim | "判定结果" |

## 注意事项

1. **显存充裕但不要浪费** — RTX 5000 Ada 32GB，batch_size 可用 4-8，但仍使用 AMP 保持效率
2. **不追求完全一致** — 复现允许 ±10% 的偏差（随机种子、硬件差异）
3. **每个 Phase 都 commit** — 保证可回滚
4. **文档优先** — 先读文档再行动，不要凭记忆
5. **出错不慌** — 有自动修复机制，最多重试 3 次
