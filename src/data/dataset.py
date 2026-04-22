"""
PyTorch Dataset Class for 3D Gravity-Magnetic Joint Inversion
=============================================================

Input (x):  [gravity, magnetic] 2 channels, shape (2, 81, 81)
  - Observation data on the surface (81 x 81 grid)
  - Gravity and magnetic anomaly maps

Output (y): [density, susceptibility, structural_sim], shape (3, 40, 40, 20)
  - density: (40, 40, 20) density model in g/cm^3
  - susceptibility: (40, 40, 20) susceptibility model in SI
  - structural_sim: (40, 40, 20) binary similarity label {0, 1}

The network input is 2D observation data (81x81), which gets expanded/processed
internally by the backbone to produce 3D output (40x40x20).

Author: Dataset Agent
Date: 2026-04-22
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json


class JointInversionDataset(Dataset):
    """
    PyTorch Dataset for gravity-magnetic joint inversion.

    Loads pre-generated .npz files with synthetic training data.

    Args:
        data_dir: directory containing train_dataset.npz, val_dataset.npz, test_dataset.npz
        split: 'train', 'val', or 'test'
        transform: optional transform pipeline
        normalize_input: whether to normalize input to [0,1]
    """

    def __init__(self, data_dir: str, split: str = 'train',
                 transform=None, normalize_input: bool = True):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.normalize_input = normalize_input

        data_file = os.path.join(data_dir, f'{split}_dataset.npz')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        print(f"[Dataset] Loading {split} set from {data_file}")
        data = np.load(data_file, allow_pickle=True)

        # Metadata
        meta_str = str(data['__meta__'])
        self.meta = json.loads(meta_str)
        self.n_samples = self.meta['n_samples']
        self.nx = self.meta.get('nx', 40)
        self.ny = self.meta.get('ny', 40)
        self.nz = self.meta.get('nz', 20)
        self.n_obs = self.meta.get('n_obs', 81)

        # Preload all samples into memory
        self.samples = []
        for i in range(self.n_samples):
            pfx = f'sample_{i:06d}'
            sample = {
                'rho': torch.from_numpy(data[f'{pfx}_rho'].copy()).float(),
                'kappa': torch.from_numpy(data[f'{pfx}_kappa'].copy()).float(),
                'sim': torch.from_numpy(data[f'{pfx}_sim'].copy()).float(),
                'gravity': torch.from_numpy(data[f'{pfx}_gravity'].copy()).float(),
                'magnetic': torch.from_numpy(data[f'{pfx}_magnetic'].copy()).float(),
                'type': int(data[f'{pfx}_type']),
                'subtype': str(data[f'{pfx}_subtype']),
            }
            self.samples.append(sample)

        data.close()
        print(f"[Dataset] {split}: {self.n_samples} samples loaded, "
              f"model=({self.nx},{self.ny},{self.nz}), obs=({self.n_obs},{self.n_obs})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Input: 2-channel observation data (2, n_obs, n_obs)
        gravity_2d = s['gravity']     # (n_obs, n_obs)
        magnetic_2d = s['magnetic']   # (n_obs, n_obs)

        if self.normalize_input:
            g_min, g_max = gravity_2d.min(), gravity_2d.max()
            m_min, m_max = magnetic_2d.min(), magnetic_2d.max()
            if g_max - g_min > 1e-8:
                gravity_2d = (gravity_2d - g_min) / (g_max - g_min)
            if m_max - m_min > 1e-8:
                magnetic_2d = (magnetic_2d - m_min) / (m_max - m_min)

        # Stack into 2-channel input
        input_tensor = torch.stack([gravity_2d, magnetic_2d], dim=0)  # (2, n_obs, n_obs)

        # Output dict with model volumes
        output_dict = {
            'rho': s['rho'],
            'kappa': s['kappa'],
            'sim': s['sim'],
        }

        # Apply transforms
        if self.transform is not None:
            input_tensor, output_dict = self.transform(input_tensor, output_dict)

        return input_tensor, output_dict


class JointInversionInMemoryDataset(Dataset):
    """
    In-memory dataset from a Python list of sample dicts.

    Useful for immediate use after generation without saving to disk.
    """

    def __init__(self, samples: list, transform=None, normalize_input: bool = True):
        self.samples = samples
        self.transform = transform
        self.normalize_input = normalize_input

        if len(samples) > 0:
            s = samples[0]
            self.nx, self.ny, self.nz = s['rho'].shape
            self.n_obs = s.get('gravity', np.zeros((1, 1))).shape[0]
        else:
            self.nx = self.ny = self.nz = 0
            self.n_obs = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        g = torch.from_numpy(s['gravity']).float() if isinstance(s['gravity'], np.ndarray) else s['gravity']
        m = torch.from_numpy(s['magnetic']).float() if isinstance(s['magnetic'], np.ndarray) else s['magnetic']

        if self.normalize_input:
            g_min, g_max = g.min(), g.max()
            m_min, m_max = m.min(), m.max()
            if g_max - g_min > 1e-8:
                g = (g - g_min) / (g_max - g_min)
            if m_max - m_min > 1e-8:
                m = (m - m_min) / (m_max - m_min)

        input_tensor = torch.stack([g, m], dim=0)

        output_dict = {
            'rho': (torch.from_numpy(s['rho']).float() if isinstance(s['rho'], np.ndarray) else s['rho']),
            'kappa': (torch.from_numpy(s['kappa']).float() if isinstance(s['kappa'], np.ndarray) else s['kappa']),
            'sim': (torch.from_numpy(s['structural_sim']).float() if isinstance(s['structural_sim'], np.ndarray) else s['structural_sim']),
        }

        if self.transform is not None:
            input_tensor, output_dict = self.transform(input_tensor, output_dict)

        return input_tensor, output_dict


def create_dataloaders(data_dir: str,
                       batch_size: int = 8,
                       num_workers: int = 4,
                       pin_memory: bool = True,
                       normalize_input: bool = True) -> dict:
    """
    Create train/val/test DataLoaders.

    Args:
        data_dir: directory with .npz files
        batch_size: training batch size
        num_workers: data loading workers
        pin_memory: use pinned memory for faster GPU transfer
        normalize_input: normalize observations to [0,1]

    Returns:
        dict of {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        ds = JointInversionDataset(
            data_dir=data_dir, split=split,
            normalize_input=normalize_input,
        )
        shuffle = (split == 'train')
        dl_batch = 1 if split == 'test' else batch_size

        dataloaders[split] = DataLoader(
            ds, batch_size=dl_batch, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=(split == 'train'),
        )

    info = {k: len(v.dataset) for k, v in dataloaders.items()}
    print(f"[DataLoader] train={info['train']}, val={info['val']}, "
          f"test={info['test']}, batch_size={batch_size}")
    # Return as tuple for easy unpacking: train_loader, val_loader, test_loader
    return dataloaders['train'], dataloaders['val'], dataloaders['test']


if __name__ == '__main__':
    print("Quick dataset class test")
    from src.data.generate_synthetic import generate_type456_samples

    samples = generate_type456_samples(1, n_samples=3, seed_start=42, verbose=False)
    ds = JointInversionInMemoryDataset(samples)
    print(f"Length: {len(ds)}")
    x, y = ds[0]
    print(f"Input: {x.shape}, dtype={x.dtype}")
    print(f"Output keys: {list(y.keys())}")
    print(f"  rho: {y['rho'].shape}, kappa: {y['kappa'].shape}, sim: {y['sim'].shape}")
    print("OK")
