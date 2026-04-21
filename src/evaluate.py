"""
评估脚本 — 加载 checkpoint，在测试集上评估
==========================================

输出: results/metrics.json

用法:
    python src/evaluate.py --config configs/full.yaml --checkpoint checkpoints/best_model.pt

评估指标 (分别对密度 rho 和磁化率 kappa 计算):
    - MSE  (Mean Squared Error)
    - RMSE (Root Mean Square Error)
    - MAE  (Mean Absolute Error)
    - Correlation Coefficient (Pearson's r)

作者: Agent-MLTestEngineer
日期: 2026-04-21
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader, Subset, random_split
import sys
import os
import json
import time

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, count_parameters, compute_metrics


def evaluate(model, loader, criterion, device):
    """
    在测试集上完整评估。

    Args:
        model: JointInversionNet 实例 (已加载权重)
        loader: 测试 DataLoader
        criterion: MultiTaskLoss 实例
        device: torch.device

    Returns:
        dict: 包含 losses 和 metrics 的完整结果字典
    """
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

    from src.utils import AverageMeter as _AM

    with torch.no_grad():
        for batch_idx, (inputs, targets_dict) in enumerate(loader):
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

            # 收集预测和真值
            all_predictions['rho_final'].append(outputs['rho_final'].cpu())
            all_predictions['kappa_final'].append(outputs['kappa_final'].cpu())
            all_targets['rho'].append(targets_device['rho'].cpu())
            all_targets['kappa'].append(targets_device['kappa'].cpu())
            all_targets['sim'].append(targets_device['sim'].cpu())

    # 汇总损失
    losses = {k: m.avg for k, m in loss_meters.items()}

    # 拼接所有批次计算指标
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

    return {
        'losses': losses,
        'metrics': metrics,
        'n_samples': len(all_predictions['rho_final']),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Joint Inversion Model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: checkpoints/best_model.pt)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    args = parser.parse_args()

    # 加载 config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("3D Gravity-Magnetic Joint Inversion Evaluation")
    print(f"Config: {args.config}")
    print("=" * 60)

    # 设置 seed
    set_seed(config['training']['seed'])

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ===== 数据准备 =====
    from src.data.generate_synthetic import generate_dataset
    from src.data.dataset import JointInversionInMemoryDataset

    data_dir = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    seed = config['training']['seed']

    npz_exists = os.path.exists(os.path.join(data_dir, 'train_dataset.npz'))

    if not npz_exists:
        print("\n[Data] No pre-generated dataset. Generating synthetic data...")
        samples = generate_dataset(dataset_type=1, n_samples=60, seed=seed, verbose=True)
        full_dataset = JointInversionInMemoryDataset(samples)
        n_total = len(full_dataset)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        n_test = n_total - n_train - n_val

        train_ds, val_ds, test_ds = random_split(
            full_dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(seed)
        )

        split_map = {'train': train_ds, 'val': val_ds, 'test': test_ds}
        eval_loader = DataLoader(split_map[args.split], batch_size=1, shuffle=False,
                                 num_workers=config['data']['num_workers'])
    else:
        from src.data.dataset import create_dataloaders
        dataloaders = create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=config['data']['num_workers'],
        )
        eval_loader = dataloaders[args.split]

    # ===== 模型 =====
    from src.model.joint_inversion_net import JointInversionNet
    from src.model.loss_functions import MultiTaskLoss

    model_config = config.get('model', {})
    use_gc = config['training'].get('gradient_checkpointing', False)
    model = JointInversionNet(
        in_channels=model_config.get('in_channels', 2),
        use_gradient_checkpointing=use_gc,
    ).to(device)

    param_info = count_parameters(model)
    print(f"\nModel parameters: {param_info['trainable']:,} trainable / "
          f"{param_info['total']:,} total")

    # 加载检查点
    ckpt_path = args.checkpoint or os.path.join(
        config['output']['checkpoint_dir'], 'best_model.pt'
    )
    if os.path.exists(ckpt_path):
        from src.utils import load_checkpoint
        info = load_checkpoint(ckpt_path, model)
        print(f"Loaded checkpoint: {ckpt_path} (epoch={info['epoch']}, loss={info['loss']:.6f})")
    else:
        print(f"[WARNING] Checkpoint not found: {ckpt_path}, using random weights")

    # 损失函数
    criterion = MultiTaskLoss().to(device)

    # ===== 评估 =====
    print(f"\nEvaluating on {args.split} split...")
    start_time = time.time()
    results = evaluate(model, eval_loader, criterion, device)
    elapsed = time.time() - start_time

    # ===== 打印结果 =====
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)

    print(f"\n--- Losses ({args.split} set, N={results['n_samples']}) ---")
    for name, value in results['losses'].items():
        print(f"  {name:>25s}: {value:.6f}")

    print("\n--- Metrics ---")
    for prop in ['rho', 'kappa']:
        m = results['metrics'][prop]
        print(f"  [{prop.upper():5s}] MSE={m['MSE']:.6f}  RMSE={m['RMSE']:.6f}  "
              f"MAE={m['MAE']:.6f}  Corr={m['Correlation']:.6f}")

    print(f"\nEvaluation time: {elapsed:.2f}s")

    # ===== 保存结果 =====
    result_dir = config['output']['result_dir']
    os.makedirs(result_dir, exist_ok=True)

    output = {
        "metrics": {
            "rho": results['metrics']['rho'],
            "kappa": results['metrics']['kappa'],
        },
        "losses": results['losses'],
        "config": {
            "checkpoint": ckpt_path,
            "split": args.split,
            "n_samples": results['n_samples'],
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_info": (
            f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available()
            else "CPU"
        ),
        "model_params": param_info,
    }

    output_path = os.path.join(result_dir, 'metrics.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    print("=" * 50)


if __name__ == '__main__':
    main()
