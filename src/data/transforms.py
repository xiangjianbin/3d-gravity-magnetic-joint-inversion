"""
数据变换模块
============

提供合成数据的预处理变换，用于 PyTorch Dataset pipeline。

可用变换:
  - NormalizeTransform: 归一化 (Min-Max 或 Z-Score)
  - AddNoiseTransform: 添加高斯噪声
  - ToTensorTransform: numpy → torch tensor (通常在 Dataset 内部已完成)
  - RandomFlipTransform: 随机翻转 (数据增强)
  - Compose: 组合多个变换

作者: Agent-DataEngineering
日期: 2026-04-21
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List


class Compose:
    """组合多个变换，顺序执行。"""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, input_tensor: torch.Tensor,
                 output_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        for t in self.transforms:
            input_tensor, output_dict = t(input_tensor, output_dict)
        return input_tensor, output_dict


class NormalizeTransform:
    """
    归一化变换。

    支持两种模式:
      - 'minmax': Min-Max 归一化到 [0, 1] (默认)
      - 'zscore': Z-Score 标准化 (均值0, 标准差1)

    分别对输入的每个 channel 和输出的每个 field 进行归一化。
    注意: 如果数据已在生成时归一化过，此变换主要用于推理时
          与训练时保持一致的统计量。
    """

    def __init__(self, mode: str = 'minmax',
                 input_stats: Optional[Dict] = None,
                 output_stats: Optional[Dict] = None):
        """
        参数:
            mode: 'minmax' 或 'zscore'
            input_stats: 输入统计量 {channel_idx: {'mean':, 'std':, 'min':, 'max':}}
                        为 None 时从数据在线计算
            output_stats: 输出统计量 同上格式
        """
        assert mode in ('minmax', 'zscore'), f"未知 mode: {mode}"
        self.mode = mode
        self.input_stats = input_stats
        self.output_stats = output_stats

    def _normalize(self, x: torch.Tensor, stats: Optional[Dict],
                   channel_idx: int = 0) -> torch.Tensor:
        """对单个张量执行归一化"""
        if self.mode == 'minmax':
            if stats is not None and channel_idx in stats:
                s = stats[channel_idx]
                dmin, dmax = s['min'], s['max']
            else:
                dmin, dmax = x.min(), x.max()

            if dmax - dmin < 1e-8:
                return torch.zeros_like(x)
            return (x - dmin) / (dmax - dmin)

        elif self.mode == 'zscore':
            if stats is not None and channel_idx in stats:
                s = stats[channel_idx]
                mean, std = s['mean'], s['std']
            else:
                mean, std = x.mean(), x.std()

            if std < 1e-8:
                return torch.zeros_like(x)
            return (x - mean) / std

    def __call__(self, input_tensor: torch.Tensor,
                 output_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        # 归一化输入 (2 channels)
        c, nx, ny, nz = input_tensor.shape
        normalized_input = torch.zeros_like(input_tensor)
        for ch in range(c):
            normalized_input[ch] = self._normalize(input_tensor[ch],
                                                    self.input_stats, ch)

        # 归一化输出
        normalized_output = {}
        for key, value in output_dict.items():
            idx = ['rho', 'kappa', 'sim'].index(key) if key in ('rho', 'kappa', 'sim') else 0
            normalized_output[key] = self._normalize(value,
                                                      self.output_stats, idx)

        return normalized_input, normalized_output


class AddNoiseTransform:
    """
    添加高斯噪声变换。

    主要用于训练时数据增强 (在线加噪)。
    通常数据在离线生成时已添加固定噪声，
    此变换可用于动态噪声水平训练。

    噪声仅在输入 (观测数据) 上添加，不影响输出标签。
    """

    def __init__(self,
                 noise_level_gravity: float = 0.005,
                 noise_level_magnetic: float = 0.108,
                 noise_prob: float = 1.0):
        """
        参数:
            noise_level_gravity: 重力通道最大噪声幅度
            noise_level_magnetic: 磁通道最大噪声幅度
            noise_prob: 以该概率添加噪声 (1.0=总是添加)
        """
        self.noise_g = noise_level_gravity
        self.noise_m = noise_level_magnetic
        self.noise_prob = noise_prob

    def __call__(self, input_tensor: torch.Tensor,
                 output_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        if torch.rand(1).item() > self.noise_prob:
            return input_tensor, output_dict

        noisy_input = input_tensor.clone()

        # 对 gravity channel (index 0) 加噪
        g_noise = torch.randn_like(noisy_input[0]) * (self.noise_g / 3.0)
        noisy_input[0] = torch.clamp(noisy_input[0] + g_noise, 0.0, 1.0)

        # 对 magnetic channel (index 1) 加噪
        m_noise = torch.randn_like(noisy_input[1]) * (self.noise_m / 3.0)
        noisy_input[1] = torch.clamp(noisy_input[1] + m_noise, 0.0, 1.0)

        return noisy_input, output_dict


class ToTensorTransform:
    """
    numpy → torch tensor 变换。

    注意: JointInversionDataset 已内部处理了 tensor 转换，
    此变换主要供外部调用或原始 numpy 数据使用。
    """

    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype

    def __call__(self, input_tensor: torch.Tensor,
                 output_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor).to(self.dtype)

        new_output = {}
        for k, v in output_dict.items():
            if isinstance(v, np.ndarray):
                new_output[k] = torch.from_numpy(v).to(self.dtype)
            else:
                new_output[k] = v

        return input_tensor, new_output


class RandomFlipTransform:
    """
    随机空间翻转 (数据增强)。

    同时翻转输入和输出以保持一致性。
    支持水平翻转 (左右)、垂直翻转 (前后)、深度翻转 (上下)。
    """

    def __init__(self,
                 p_horizontal: float = 0.5,
                 p_vertical: float = 0.5,
                 p_depth: float = 0.5):
        """
        参数:
            p_horizontal: 水平翻转概率 (dim=1)
            p_vertical: 垂直翻转概率 (dim=0)
            p_depth: 深度翻转概率 (dim=2, 仅影响 3D 部分)
        """
        self.p_h = p_horizontal
        self.p_v = p_vertical
        self.p_d = p_depth

    def __call__(self, input_tensor: torch.Tensor,
                 output_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        inp = input_tensor
        out = {k: v.clone() for k, v in output_dict.items()}

        # 水平翻转 (Northing 方向, dim=1)
        if torch.rand(1).item() < self.p_h:
            inp = torch.flip(inp, dims=[1])
            out = {k: torch.flip(v, dims=[1]) for k, v in out.items()}

        # 垂直翻转 (Easting 方向, dim=0)
        if torch.rand(1).item() < self.p_v:
            inp = torch.flip(inp, dims=[0])
            out = {k: torch.flip(v, dims=[0]) for k, v in out.items()}

        # 深度翻转 (Depth 方向, dim=3 for input, dim=2 for output)
        if torch.rand(1).item() < self.p_d:
            inp = torch.flip(inp, dims=[3])
            out = {k: torch.flip(v, dims=[2]) for k, v in out.items()}

        return inp, out


class MaskedOutputTransform:
    """
    掩码输出变换 — 用于部分一致样本的训练。

    将结构不一致区域 (S=0) 的输出梯度掩蔽，
    让网络专注于结构一致区域的学习。
    这是一种软性的课程学习策略。
    """

    def __init__(self, mask_value: float = 0.0):
        """
        参数:
            mask_value: 掩蔽区域的填充值
        """
        self.mask_value = mask_value

    def __call__(self, input_tensor: torch.Tensor,
                 output_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        sim = output_dict.get('sim', None)
        if sim is None:
            return input_tensor, output_dict

        # S=0 的区域将 rho 和 kappa 设为 mask_value
        mask = (sim == 0)
        new_output = dict(output_dict)
        if 'rho' in new_output:
            new_output['rho'] = new_output['rho'].clone()
            new_output['rho'][mask] = self.mask_value
        if 'kappa' in new_output:
            new_output['kappa'] = new_output['kappa'].clone()
            new_output['kappa'][mask] = self.mask_value

        return input_tensor, new_output


# ============================================================
# 预设变换组合
# ============================================================

def get_train_transforms() -> Compose:
    """获取训练时的标准变换组合"""
    return Compose([
        RandomFlipTransform(p_horizontal=0.5, p_vertical=0.5, p_depth=0.3),
        # AddNoiseTransform(noise_prob=0.3),  # 可选: 动态噪声增强
    ])


def get_eval_transforms() -> Compose:
    """获取评估/推理时的变换组合 (无增强)"""
    return Compose([])


if __name__ == '__main__':
    # 快速测试
    print("=" * 60)
    print("Transforms 模块快速测试")
    print("=" * 60)

    # 创建假数据
    input_t = torch.randn(2, 40, 40, 20).float()
    input_t = (input_t - input_t.min()) / (input_t.max() - input_t.min() + 1e-8)
    output_d = {
        'rho': torch.rand(40, 40, 20).float(),
        'kappa': torch.rand(40, 40, 20).float(),
        'sim': (torch.rand(40, 40, 20) > 0.5).float(),
    }

    print(f"\n输入: {input_t.shape}, range=[{input_t.min():.4f}, {input_t.max():.4f}]")

    # 测试 NormalizeTransform
    norm_tf = NormalizeTransform(mode='minmax')
    inp_n, out_n = norm_tf(input_t, output_d)
    print(f"Min-Max 归一化后: input range=[{inp_n.min():.4f}, {inp_n.max():.4f}]")

    # 测试 AddNoiseTransform
    noise_tf = AddNoiseTransform(noise_level_gravity=0.01, noise_level_magnetic=0.05)
    inp_noise, _ = noise_tf(inp_n, out_n)
    print(f"加噪后: ch0 range=[{inp_noise[0].min():.4f}, {inp_noise[0].max():.4f}], "
          f"ch1 range=[{inp_noise[1].min():.4f}, {inp_noise[1].max():.4f}]")

    # 测试 RandomFlipTransform
    flip_tf = RandomFlipTransform(p_horizontal=1.0, p_vertical=0.0, p_depth=0.0)
    inp_flip, out_flip = flip_tf(input_t, output_d)
    print(f"水平翻转后: 输出 rho 与原 rho 一致? "
          f"{torch.equal(out_flip['rho'], output_d['rho'].flip(dims=[1]))}")

    # 测试 Compose
    composed = get_train_transforms()
    inp_c, out_c = composed(input_t, output_d)
    print(f"Compose 组合变换: 输入 shape={inp_c.shape}, 输出 keys={list(out_c.keys())}")

    print("\n" + "=" * 60)
    print("Transforms 测试通过!")
    print("=" * 60)
