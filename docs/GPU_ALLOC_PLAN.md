# GPU 分配方案

> **日期**: 2026-04-21
> **GPU**: NVIDIA RTX 5000 Ada Generation (33.9 GB)

## GPU 状态快照

| 项目 | 值 |
|------|-----|
| GPU 型号 | RTX 5000 Ada |
| 总显存 | 33.9 GB |
| 空闲显存 | ~32.2 GB |
| CUDA 版本 | 12.1 (PyTorch 2.4.0+cu121) |
| 驱动版本 | 535.230.02 |

## 显存占用实测

| Batch Size | 峰值显存 | 利用率 | 判定 |
|-----------|---------|--------|------|
| 1 | 1.12 GB | 3.3% | OK |
| 2 | 1.61 GB | 4.7% | OK |
| 4 | 2.57 GB | 7.6% | **推荐（保守）** |
| 8 | 4.52 GB | 13.3% | **推荐（最优）** |

## 推荐配置

```yaml
# configs/full.yaml 更新
training:
  batch_size: 8        # 最优：4.5GB/32GB，余量充足
  use_amp: true         # 混合精度，节省 ~40% 显存
  gradient_checkpointing: false  # 不需要，显存充裕
  num_workers: 4       # 数据加载并行

# 预估训练时间
# 论文 RTX 3070 8GB → 15h (bs=1-2)
# 我们 RTX 5000 Ada 32GB, bs=8 → 约 2-4h (8x throughput)
```

## 关键修复记录

1. **ASPP 全局池化 BN bug**: 移除 global pool 分支的 BatchNorm（1×1×1 空间尺寸导致 BN 失效）
2. **PyTorch 安装**: 从 CPU-only (2.4.1+cpu) 升级到 CUDA 版本 (2.4.0+cu121)
