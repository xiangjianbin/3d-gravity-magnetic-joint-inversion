"""
3D 重磁联合反演网络训练脚本
============================

用法:
    python src/train.py --config configs/smoke.yaml
    python src/train.py --config configs/full.yaml

关键特性:
    - AMP (torch.cuda.amp) 混合精度训练
    - GradScaler 梯度缩放
    - clip_grad_norm_ 梯度裁剪
    - 每个 epoch 打印各 task 的 loss
    - 保存 best 和 final 两个 checkpoint

作者: Agent-MLTestEngineer
日期: 2026-04-21
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader, Subset, random_split
import sys
import os
import time

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    set_seed,
    count_parameters,
    save_checkpoint,
    AverageMeter,
)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch):
    """
    训练一个 epoch。

    Args:
        model: JointInversionNet 实例
        loader: 训练 DataLoader
        criterion: MultiTaskLoss 实例
        optimizer: Adam 优化器
        device: torch.device
        scaler: torch.cuda.amp.GradScaler 或 None
        epoch: 当前 epoch 编号

    Returns:
        dict: 各 task 的平均损失 {'total': float, 'task1': ..., ...}
    """
    model.train()

    # 初始化各 task 的 AverageMeter
    loss_meters = {
        'total': AverageMeter(),
        'task1_gravity_mse': AverageMeter(),
        'task2_magnetic_mse': AverageMeter(),
        'task3_similarity_mse': AverageMeter(),
        'task4_gravity_bce': AverageMeter(),
        'task5_magnetic_bce': AverageMeter(),
    }

    for batch_idx, (inputs, targets_dict) in enumerate(loader):
        inputs = inputs.to(device)
        # 将 targets 移动到 device (targets 是 dict of tensors)
        targets_device = {k: v.to(device) for k, v in targets_dict.items()}

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(inputs, return_all=True)
                total_loss, task_losses = criterion(outputs, targets_device)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs, return_all=True)
            total_loss, task_losses = criterion(outputs, targets_device)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 更新 meters
        loss_meters['total'].update(total_loss.item(), inputs.size(0))
        for key in ['task1_gravity_mse', 'task2_magnetic_mse',
                     'task3_similarity_mse', 'task4_gravity_bce', 'task5_magnetic_bce']:
            val = task_losses[key]
            if isinstance(val, (int, float)):
                loss_meters[key].update(val, inputs.size(0))

    # 返回各 task 平均值
    return {k: m.avg for k, m in loss_meters.items()}


def validate(model, loader, criterion, device):
    """
    验证一个 epoch。

    Args:
        model: JointInversionNet 实例
        loader: 验证 DataLoader
        criterion: MultiTaskLoss 实例
        device: torch.device

    Returns:
        tuple: (losses_dict, metrics_dict)
               losses_dict: 各 task 平均损失
               metrics_dict: MSE/RMSE/MAE/Correlation 指标
    """
    from src.utils import compute_metrics

    model.eval()

    loss_meters = {
        'total': AverageMeter(),
        'task1_gravity_mse': AverageMeter(),
        'task2_magnetic_mse': AverageMeter(),
        'task3_similarity_mse': AverageMeter(),
        'task4_gravity_bce': AverageMeter(),
        'task5_magnetic_bce': AverageMeter(),
    }

    all_predictions = {'rho_final': [], 'kappa_final': []}
    all_targets = {'rho': [], 'kappa': [], 'sim': []}

    with torch.no_grad():
        for inputs, targets_dict in loader:
            inputs = inputs.to(device)
            targets_device = {k: v.to(device) for k, v in targets_dict.items()}

            outputs = model(inputs, return_all=True)
            total_loss, task_losses = criterion(outputs, targets_device)

            loss_meters['total'].update(total_loss.item(), inputs.size(0))
            for key in ['task1_gravity_mse', 'task2_magnetic_mse',
                         'task3_similarity_mse', 'task4_gravity_bce', 'task5_magnetic_bce']:
                val = task_losses[key]
                if isinstance(val, (int, float)):
                    loss_meters[key].update(val, inputs.size(0))

            # 收集预测和真值用于指标计算
            all_predictions['rho_final'].append(outputs['rho_final'])
            all_predictions['kappa_final'].append(outputs['kappa_final'])
            all_targets['rho'].append(targets_device['rho'])
            all_targets['kappa'].append(targets_device['kappa'])
            all_targets['sim'].append(targets_device['sim'])

    losses = {k: m.avg for k, m in loss_meters.items()}

    # 计算评估指标
    pred_concat = {
        'rho_final': torch.cat(all_predictions['rho_final'], dim=0),
        'kappa_final': torch.cat(all_predictions['kappa_final'], dim=0),
    }
    target_concat = {
        'rho': torch.cat(all_targets['rho'], dim=0),
        'kappa': torch.cat(all_targets['kappa'], dim=0),
        'sim': torch.cat(all_targets['sim'], dim=0),
    }
    metrics = compute_metrics(pred_concat, target_concat)

    return losses, metrics


def main():
    parser = argparse.ArgumentParser(description='3D Gravity-Magnetic Joint Inversion Training')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    # 加载 config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("3D Gravity-Magnetic Joint Inversion Training")
    print(f"Config: {args.config}")
    print("=" * 60)

    # 设置 seed
    seed = config['training']['seed']
    set_seed(seed)
    print(f"Random seed: {seed}")

    # 设备
    device_str = config.get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # ===== 数据准备 =====
    from src.data.generate_synthetic import generate_dataset
    from src.data.dataset import JointInversionInMemoryDataset

    data_dir = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']

    # 如果数据集文件不存在，先生成小规模数据
    npz_exists = os.path.exists(os.path.join(data_dir, 'train_dataset.npz'))

    if not npz_exists:
        print("\n[Data] No pre-generated dataset found. Generating small synthetic dataset...")
        samples = generate_dataset(dataset_type=1, n_samples=60, seed=seed, verbose=True)
        print(f"[Data] Generated {len(samples)} samples")

        # 创建内存数据集并划分
        full_dataset = JointInversionInMemoryDataset(samples)
        n_total = len(full_dataset)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        n_test = n_total - n_train - n_val

        train_ds, val_ds, test_ds = random_split(
            full_dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(seed)
        )
        print(f"[Data] Split: train={n_train}, val={n_val}, test={n_test}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=config['data']['num_workers'],
                                  drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                               num_workers=config['data']['num_workers'])
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                                 num_workers=config['data']['num_workers'])
    else:
        from src.data.dataset import create_dataloaders
        dataloaders = create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=config['data']['num_workers'],
        )
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']

    # ===== 模型 =====
    from src.model.joint_inversion_net import JointInversionNet

    model_config = config.get('model', {})
    use_gc = config['training'].get('gradient_checkpointing', False)
    model = JointInversionNet(
        in_channels=model_config.get('in_channels', 2),
        use_gradient_checkpointing=use_gc,
    ).to(device)

    param_info = count_parameters(model)
    print(f"\nModel parameters: {param_info['trainable']:,} trainable / "
          f"{param_info['total']:,} total")

    # ===== 损失函数 =====
    from src.model.loss_functions import MultiTaskLoss
    criterion = MultiTaskLoss().to(device)

    # ===== 优化器 =====
    lr = config['training']['lr']
    weight_decay = config['training'].get('weight_decay', 0)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # 学习率调度器 (full.yaml 有 scheduler 配置时使用)
    scheduler = None
    if 'scheduler' in config['training']:
        sched_type = config['training']['scheduler']
        sched_params = config['training'].get('scheduler_params', {})
        if sched_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **sched_params
            )
        elif sched_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, **sched_params
            )
        if scheduler is not None:
            print(f"LR Scheduler: {sched_type} ({sched_params})")

    # ===== AMP =====
    use_amp = config['training'].get('use_amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"AMP enabled: {use_amp}")

    # ===== 输出目录 =====
    ckpt_dir = config['output']['checkpoint_dir']
    result_dir = config['output']['result_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # ===== 训练循环 =====
    epochs = config['training']['epochs']
    best_loss = float('inf')
    best_epoch = 0
    history = []

    print(f"\n{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>10} | "
          f"T1(MSE):>10 | T2(MSE):>10 | T3(Sim):>9 | T4(BCE):>9 | T5(BCE):>9 | LR")
    print("-" * 110)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )

        # Validate
        val_losses, val_metrics = validate(
            model, val_loader, criterion, device
        )

        # LR step
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        # 打印
        print(f"{epoch+1:>6} | {train_losses['total']:>12.6f} | "
              f"{val_losses['total']:>10.6f} | "
              f"{train_losses['task1_gravity_mse']:>10.6f} | "
              f"{train_losses['task2_magnetic_mse']:>10.6f} | "
              f"{train_losses['task3_similarity_mse']:>9.6f} | "
              f"{train_losses['task4_gravity_bce']:>9.6f} | "
              f"{train_losses['task5_magnetic_bce']:>9.6f} | "
              f"{current_lr:.2e} ({epoch_time:.1f}s)")

        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_losses['total'],
            'val_loss': val_losses['total'],
            'lr': current_lr,
            **{f'train_{k}': v for k, v in train_losses.items()},
            **{f'val_{k}': v for k, v in val_losses.items()},
        })

        # 保存最佳模型
        if val_losses['total'] < best_loss:
            best_loss = val_losses['total']
            best_epoch = epoch + 1
            save_checkpoint(
                model, optimizer, epoch, best_loss,
                os.path.join(ckpt_dir, 'best_model.pt')
            )

    total_time = time.time() - start_time

    # 保存最终模型
    final_loss = val_losses['total']
    save_checkpoint(
        model, optimizer, epochs - 1, final_loss,
        os.path.join(ckpt_dir, 'final_model.pt')
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Best validation loss: {best_loss:.6f} at epoch {best_epoch}")
    print(f"Final validation loss: {final_loss:.6f}")
    print(f"Checkpoints saved to: {ckpt_dir}/")
    print("=" * 60)

    # 保存训练历史
    history_path = os.path.join(result_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")


if __name__ == '__main__':
    main()
