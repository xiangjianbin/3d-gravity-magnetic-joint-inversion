# GPU 分配方案

## 硬件信息
- GPU: NVIDIA RTX 5000 Ada Generation
- 总显存: 31.6 GB (32362 MiB)
- 可用显存: ~31 GB (系统预留约 1 GB)
- 目标显存占用: 70%~80% = 22.1 ~ 25.3 GB

## 模型信息
- 模型: JointInversionNet (2D U-Net + ASPP + 5 Task Heads)
- 总参数量: 8,146,757 (~31 MB in FP32)
- 输入: (B, 2, 81, 81) float32 — gravity + magnetic observation maps
- 输出: 5 x (B, 1, 40, 40, 20) float32 — density / susceptibility / structural_sim

## 显存测试结果（FP32，forward + backward）

| Batch Size | 峰值显存(GB) | 占比% | 状态 |
|------------|-------------|-------|------|
| 1          | 0.113       | 0.4   | OK   |
| 2          | 0.194       | 0.6   | OK   |
| 4          | 0.351       | 1.1   | OK   |
| 8          | 0.663       | 2.1   | OK   |
| 16         | 1.285       | 4.1   | OK   |
| 32         | 2.540       | 8.0   | OK   |
| 48         | 3.767       | 11.9  | OK   |
| 64         | 5.017       | 15.9  | OK   |
| 96         | 7.504       | 23.7  | OK   |
| 128        | 9.981       | 31.6  | OK   |
| 192        | 14.950      | 47.3  | OK   |
| 256        | 19.906      | 63.0  | OK   |
| **320**    | **24.881**  | **78.7** | **OK** (目标区间) |
| 384        | 29.669      | 93.9  | OOM  |

## AMP 混合精度测试结果（FP16 forward + FP32 backward）

| Batch Size | 峰值显存(GB) | 占比% | 状态 |
|------------|-------------|-------|------|
| 256        | 12.359      | 39.1  | OK   |
| 320        | 15.438      | 48.8  | OK   |
| 384        | 18.508      | 58.6  | OK   |
| 448        | 21.585      | 68.3  | OK   |
| **512**    | **24.655**  | **78.0** | **OK** (目标区间) |
| 576        | 25.937      | 82.1  | OK   |
| 640        | —           | —     | OOM  |

## 推荐 Batch Size

### 方案 A：FP32 全精度训练（推荐用于论文对齐）
- 训练 batch_size: **320** （显存占用 78.7%，在 70-80% 目标区间内）
- 推理 batch_size: **320**
- 启用 AMP: 否（使用全精度）
- 使用梯度累积: 不需要
- 有效 batch size: 320

### 方案 B：AMP 混合精度训练（推荐用于快速迭代）
- 训练 batch_size: **512** （显存占用 78.0%，在 70-80% 目标区间内）
- 推理 batch_size: **576** （82.1%，推理无梯度可稍大）
- 启用 AMP: 是（torch.amp.autocast('cuda', dtype=torch.float16)）
- 使用梯度累积: 不需要
- 有效 batch size: 512

## 最终推荐

**采用方案 A（FP32, batch_size=320）作为默认训练配置**，理由：
1. 论文未明确提及使用混合精度训练，FP32 更利于结果复现对齐
2. 78.7% 显存利用率处于最佳区间，充分利用 GPU 资源
3. RTX 5000 Ada 的 Tensor Core 在 FP32 下性能依然强劲
4. 如需加速可随时切换到方案 B（AMP, batch_size=512）

## 显存预算分解（估算 @ BS=320, FP32）

| 组件 | 预估显存 |
|------|---------|
| 模型参数 (8.15M x 4 bytes) | ~0.03 GB |
| 输入张量 (320x2x81x81) | ~0.02 GB |
| 中间激活值 (U-Net skip connections) | ~18 GB |
| 梯度张量 | ~5 GB |
| 优化器状态 (Adam: 2x params) | ~0.06 GB |
| 输出张量 (5 x 320x1x40x40x20) | ~0.5 GB |
| **总计** | **~24.9 GB** |
| **剩余** | **~6.7 GB** (给 CUDA context / 数据加载器预取) |

## 测试方法说明

测试脚本 `scripts/gpu_memory_test.py` 执行以下步骤：
1. 将 JointInversionNet 加载到 CUDA 设备
2. 对每个 batch_size 生成随机输入 (B, 2, 81, 81) 和目标 (B, 1, 40, 40, 20)
3. 执行完整的前向传播 + 反向传播（含 5 个任务的 MSE loss）
4. 通过 `torch.cuda.max_memory_allocated()` 记录峰值显存
5. 遇到 OOM 时停止并记录失败点
6. 分别测试 FP32 和 AMP (FP16) 两种模式
