"""
PyTorch Dataset 类 — 3D 重磁联合反演数据集
============================================

输入 (x): [gravity, magnetic] 2通道, shape (2, 40, 40, 20)
  设计决策: 论文 Fig.1 显示输入是 2ch 3D volume (40x40x20)。
  但重力/磁异常观测数据本质上是 2D (40x40)。
  本实现采用 **深度方向复制 (depth-wise repeat)** 策略:
    将 2D 观测数据在 nz=20 的深度维度上复制，
    使每个深度切片都包含完整的观测信息。
  这等价于告诉网络"每个深度层都看到相同的地面观测信号"。
  替代方案可以是线性插值或学习到的上采样，但复制最简单且可解释。

输出 (y): [rho, kappa, structural_sim] 3通道, shape (3, 40, 40, 20)

作者: Agent-DataEngineering
日期: 2026-04-21
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json


class JointInversionDataset(Dataset):
    """
    3D 重磁联合反演 PyTorch 数据集。

    从 .npz 文件加载预生成的合成数据样本。

    Args:
        data_dir: 包含 .npz 数据文件的目录路径
        split: 'train' | 'val' | 'test'
        transform: 可选的数据变换 (torchvision.transforms.Compose)
        input_expansion: 2D->3D 输入扩展方式:
            - 'repeat': 深度方向复制 (默认)
            - 'interpolate': 线性插值
            - 'learnable': 占位符 (需配合网络修改)
    """

    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 transform=None,
                 input_expansion: str = 'repeat'):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.input_expansion = input_expansion

        # 加载数据文件
        data_file = os.path.join(data_dir, f'{split}_dataset.npz')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件不存在: {data_file}")

        print(f"[Dataset] 加载 {split} 集: {data_file}")
        data = np.load(data_file, allow_pickle=True)

        # 读取元数据
        meta_str = str(data['__meta__'])
        self.meta = json.loads(meta_str)
        self.n_samples = self.meta['n_samples']
        self.nx = self.meta['nx']
        self.ny = self.meta['ny']
        self.nz = self.meta['nz']

        # 预加载所有数据到内存 (对于 ~45000 样本可能较大,
        # 但 RTX 3060 6GB 的机器通常有足够系统内存)
        self.samples = []
        for i in range(self.n_samples):
            prefix = f'sample_{i:06d}'
            sample = {
                'rho': torch.from_numpy(data[f'{prefix}_rho'].copy()).float(),
                'kappa': torch.from_numpy(data[f'{prefix}_kappa'].copy()).float(),
                'sim': torch.from_numpy(data[f'{prefix}_sim'].copy()).float(),
                'gravity': torch.from_numpy(data[f'{prefix}_gravity'].copy()).float(),
                'magnetic': torch.from_numpy(data[f'{prefix}_magnetic'].copy()).float(),
                'type': int(data[f'{prefix}_type']),
                'consistency': str(data[f'{prefix}_consistency']),
            }
            self.samples.append(sample)

        data.close()
        print(f"[Dataset] {split} 集加载完成: {self.n_samples} 样本, "
              f"shape=({self.nx},{self.ny},{self.nz})")

    def __len__(self) -> int:
        return self.n_samples

    def _expand_to_3d(self, data_2d: torch.Tensor) -> torch.Tensor:
        """
        将 2D 观测数据扩展为 3D volume。

        策略说明 (input_expansion):
          'repeat' (默认): 在深度维度直接复制
            shape (nx, ny) -> (nx, ny, nz)
            物理含义: 假设每个深度的源体都对地表观测有贡献,
                      网络需要自己学会从相同的 2D 观测中反推出 3D 结构

          'interpolate': 使用线性插值平滑扩展
            先 repeat 再做 1D 卷积平滑
        """
        if self.input_expansion == 'repeat':
            # 直接在深度维度复制
            nx, ny = data_2d.shape
            return data_2d.unsqueeze(-1).expand(-1, -1, self.nz).contiguous()

        elif self.input_expansion == 'interpolate':
            # 复制 + 轻微深度方向平滑
            nx, ny = data_2d.shape
            repeated = data_2d.unsqueeze(-1).expand(-1, -1, self.nz).contiguous()
            # 用简单的 1D 平均池化模拟平滑 (kernel_size=3, pad=1)
            smooth = repeated.unsqueeze(0).unsqueeze(0)  # (1,1,nx,ny,nz)
            smooth = torch.nn.functional.avg_pool3d(smooth, kernel_size=3,
                                                     stride=1, padding=1)
            return smooth.squeeze(0).squeeze(0)

        else:
            raise ValueError(f"未知 input_expansion: {self.input_expansion}")

    def __getitem__(self, idx: int) -> tuple:
        """
        返回单个样本。

        Returns:
            input_tensor: (2, nx, ny, nz) float32
              [channel_0: gravity expanded to 3D]
              [channel_1: magnetic expanded to 3D]

            output_dict: dict
              'rho': (nx, ny, nz) 密度模型
              'kappa': (nx, ny, nz) 磁化率模型
              'sim': (nx, ny, nz) 结构相似性标签
        """
        sample = self.samples[idx]

        # 构建输入: 2ch 3D volume
        gravity_3d = self._expand_to_3d(sample['gravity'])   # (nx, ny, nz)
        magnetic_3d = self._expand_to_3d(sample['magnetic'])  # (nx, ny, nz)
        input_tensor = torch.stack([gravity_3d, magnetic_3d], dim=0)  # (2, nx, ny, nz)

        # 输出字典
        output_dict = {
            'rho': sample['rho'],
            'kappa': sample['kappa'],
            'sim': sample['sim'],
        }

        # 应用变换 (如果有)
        if self.transform is not None:
            input_tensor, output_dict = self.transform(input_tensor, output_dict)

        return input_tensor, output_dict


class JointInversionInMemoryDataset(Dataset):
    """
    内存版数据集 — 直接从 Python 列表构建，无需 .npz 文件。

    适用于 generate_synthetic.py 生成后立即使用、或小规模实验场景。
    """

    def __init__(self,
                 samples: list,
                 transform=None,
                 input_expansion: str = 'repeat'):
        """
        参数:
            samples: generate_synthetic.py 生成的样本列表
                   每个 element 是 dict 含 rho/kappa/sim/gravity/magnetic/type/consistency_type
            transform: 可选变换
            input_expansion: 2D->3D 扩展方式
        """
        self.samples = samples
        self.transform = transform
        self.input_expansion = input_expansion

        if len(samples) > 0:
            s = samples[0]
            self.nx, self.ny, self.nz = s['rho'].shape
        else:
            self.nx = self.ny = self.nz = 0

    def __len__(self) -> int:
        return len(self.samples)

    def _expand_to_3d(self, data_2d: torch.Tensor) -> torch.Tensor:
        """同 JointInversionDataset._expand_to_3d"""
        if self.input_expansion == 'repeat':
            nx, ny = data_2d.shape
            return data_2d.unsqueeze(-1).expand(-1, -1, self.nz).contiguous()
        elif self.input_expansion == 'interpolate':
            nx, ny = data_2d.shape
            repeated = data_2d.unsqueeze(-1).expand(-1, -1, self.nz).contiguous()
            smooth = repeated.unsqueeze(0).unsqueeze(0)
            smooth = torch.nn.functional.avg_pool3d(smooth, kernel_size=3,
                                                     stride=1, padding=1)
            return smooth.squeeze(0).squeeze(0)
        else:
            raise ValueError(f"未知 input_expansion: {self.input_expansion}")

    def __getitem__(self, idx: int) -> tuple:
        s = self.samples[idx]

        gravity_3d = self._expand_to_3d(
            torch.from_numpy(s['gravity']).float())
        magnetic_3d = self._expand_to_3d(
            torch.from_numpy(s['magnetic']).float())
        input_tensor = torch.stack([gravity_3d, magnetic_3d], dim=0)

        output_dict = {
            'rho': torch.from_numpy(s['rho']).float(),
            'kappa': torch.from_numpy(s['kappa']).float(),
            'sim': torch.from_numpy(s['structural_sim']).float(),
        }

        if self.transform is not None:
            input_tensor, output_dict = self.transform(input_tensor, output_dict)

        return input_tensor, output_dict


def create_dataloaders(data_dir: str,
                       batch_size: int = 4,
                       num_workers: int = 2,
                       pin_memory: bool = True,
                       input_expansion: str = 'repeat') -> dict:
    """
    创建 train / val / test DataLoader。

    参数:
        data_dir: 数据目录 (含 train_dataset.npz, val_dataset.npz, test_dataset.npz)
        batch_size: 批大小 (RTX 3060 6GB 建议 1-2)
        num_workers: 数据加载进程数
        pin_memory: 是否使用锁页内存 (加速 CPU→GPU 传输)
        input_expansion: 2D->3D 输入扩展方式

    返回:
        dataloaders: dict {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        dataset = JointInversionDataset(
            data_dir=data_dir,
            split=split,
            input_expansion=input_expansion,
        )
        shuffle = (split == 'train')

        # 测试集不需要 shuffle 且可以更大 batch (仅推理)
        dl_batch = 1 if split == 'test' else batch_size

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=dl_batch,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train'),  # 训练时丢弃不完整最后batch
        )

    total_train = len(dataloaders['train'].dataset)
    total_val = len(dataloaders['val'].dataset)
    total_test = len(dataloaders['test'].dataset)
    print(f"[DataLoader] train={total_train}, val={total_val}, test={total_test}, "
          f"batch_size={batch_size}")

    return dataloaders


def split_and_save_datasets(all_samples: list,
                            data_dir: str,
                            train_ratio: float = 0.70,
                            val_ratio: float = 0.20,
                            seed: int = 42):
    """
    将全部样本按比例划分为 train/val/test 并保存为独立 .npz 文件。

    划分策略: 分层抽样 (stratified)，保证每类数据集在各划分中比例一致。

    参数:
        all_samples: 全部样本列表
        data_dir: 输出目录
        train_ratio: 训练集比例 (默认 0.70)
        val_ratio: 验证集比例 (默认 0.20), 测试集 = 1 - train - val
        seed: 随机种子
    """
    os.makedirs(data_dir, exist_ok=True)

    # 按 type 分组后分层抽样
    from collections import defaultdict
    by_type = defaultdict(list)
    for s in all_samples:
        by_type[s['type']].append(s)

    train_samples = []
    val_samples = []
    test_samples = []

    rng = np.random.RandomState(seed)

    for dtype, samples in sorted(by_type.items()):
        n = len(samples)
        indices = np.arange(n)
        rng.shuffle(indices)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        for i in train_idx:
            train_samples.append(samples[i])
        for i in val_idx:
            val_samples.append(samples[i])
        for i in test_idx:
            test_samples.append(samples[i])

    # 保存
    from src.data.generate_synthetic import save_dataset
    save_dataset(train_samples, os.path.join(data_dir, 'train_dataset.npz'))
    save_dataset(val_samples, os.path.join(data_dir, 'val_dataset.npz'))
    save_dataset(test_samples, os.path.join(data_dir, 'test_dataset.npz'))

    print(f"\n[划分完成] train={len(train_samples)} ({100*len(train_samples)/len(all_samples):.1f}%), "
          f"val={len(val_samples)} ({100*len(val_samples)/len(all_samples):.1f}%), "
          f"test={len(test_samples)} ({100*len(test_samples)/len(all_samples):.1f}%)")


if __name__ == '__main__':
    # 快速测试 Dataset 类
    print("=" * 60)
    print("Dataset 类快速测试")
    print("=" * 60)

    # 生成少量测试数据
    from src.data.generate_synthetic import generate_dataset
    test_samples = generate_dataset(dataset_type=1, n_samples=10, seed=42, verbose=False)

    # 测试 InMemoryDataset
    print("\n--- InMemoryDataset ---")
    mem_ds = JointInversionInMemoryDataset(test_samples)
    print(f"长度: {len(mem_ds)}")
    x, y = mem_ds[0]
    print(f"输入 shape: {x.shape}, dtype: {x.dtype}")
    print(f"输出 keys: {list(y.keys())}")
    print(f"  rho: {y['rho'].shape}, kappa: {y['kappa'].shape}, sim: {y['sim'].shape}")
    print(f"输入范围: [{x.min():.4f}, {x.max():.4f}]")

    # 测试 DataLoader
    print("\n--- DataLoader ---")
    loader = DataLoader(mem_ds, batch_size=2, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"batch 输入 shape: {batch_x.shape}")
    print(f"batch 输出 rho shape: {batch_y['rho'].shape}")

    print("\n" + "=" * 60)
    print("Dataset 测试通过!")
    print("=" * 60)
