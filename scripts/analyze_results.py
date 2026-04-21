"""
结果分析脚本 — 加载训练好的模型，在测试集上运行完整推理和指标计算
=====================================================================

功能:
  1. 加载最佳模型 checkpoint (checkpoints/best_model.pt)
  2. 在测试集上运行推理 (Task 4 + Task 5 输出)
  3. 计算完整评估指标:
     - MSE, RMSE, MAE (分别对密度 rho 和磁化率 kappa)
     - Pearson Correlation Coefficient
     - Structural Accuracy (S_pred vs S_true 的分类准确率)
  4. 与论文目标值对比
  5. 生成对比表格 Markdown

输出:
  - results/metrics.json: 完整指标 JSON
  - results/comparison_table.md: Markdown 对比表

用法:
  python scripts/analyze_results.py --config configs/full.yaml
  python scripts/analyze_results.py --config configs/full.yaml --checkpoint checkpoints/best_model.pt

作者: Agent-ResultAnalysis
日期: 2026-04-21
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# 添加项目根目录到 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import set_seed, count_parameters, load_checkpoint, compute_metrics


# ---------------------------------------------------------------------------
# 论文目标值 (从 Fig.9 / Table II 等处读取的占位符)
# Phase 6 执行时需根据论文图表精确填充实际数值
# ---------------------------------------------------------------------------
PAPER_TARGETS = {
    "DL_Joint": {
        "rho": {
            "MSE": None,       # TODO: 从论文 Fig/Table 精确读取
            "RMSE": None,
            "MAE": None,
            "Correlation": None,  # 预期 > 0.95
        },
        "kappa": {
            "MSE": None,
            "RMSE": None,
            "MAE": None,
            "Correlation": None,  # 预期 > 0.90
        },
        "structural_accuracy": None,  # 预期 > 85%
    },
    "Independent": {
        "rho": {
            "MSE": None,       # TODO: 从论文独立反演结果读取
            "RMSE": None,
            "MAE": None,
            "Correlation": None,
        },
        "kappa": {
            "MSE": None,
            "RMSE": None,
            "MAE": None,
            "Correlation": None,
        },
        "structural_accuracy": None,
    },
    "CrossGradient": {
        "rho": {"MSE": None, "RMSE": None, "MAE": None, "Correlation": None},
        "kappa": {"MSE": None, "RMSE": None, "MAE": None, "Correlation": None},
        "structural_accuracy": None,
    },
}


def compute_structural_accuracy(s_pred: torch.Tensor, s_true: torch.Tensor,
                                threshold: float = 0.5) -> float:
    """
    计算结构相似性分类准确率。

    将连续值 S 二值化后计算 pixel-wise accuracy。

    Args:
        s_pred: 预测的结构相似图 (N,) 或 (B,1,D,H,W) 展平后
        s_true: 真实的结构相似标签 (N,)
        threshold: 二值化阈值

    Returns:
        float: 分类准确率 [0, 1]
    """
    pred_binary = (s_pred > threshold).float()
    true_binary = (s_true > threshold).float()
    accuracy = (pred_binary == true_binary).float().mean().item()
    return accuracy


def run_inference(model, loader, device):
    """
    在数据集上运行推理，收集所有预测和真值。

    Args:
        model: JointInversionNet 实例 (已加载权重)
        loader: DataLoader
        device: torch.device

    Returns:
        dict: 包含 predictions 和 targets 的字典
    """
    model.eval()

    all_preds = {
        'rho_final': [],
        'kappa_final': [],
        'rho_pred': [],      # Task 1 独立重力输出
        'kappa_pred': [],    # Task 2 独立磁法输出
        'structural_sim': [], # Task 3 结构相似性
    }
    all_targets = {
        'rho': [],
        'kappa': [],
        'sim': [],
    }

    with torch.no_grad():
        for inputs, targets_dict in loader:
            inputs = inputs.to(device)
            targets_device = {k: v.to(device) for k, v in targets_dict.items()}

            # 推理模式返回全部5个任务输出 (return_all=True)
            outputs = model(inputs, return_all=True)

            # 收集预测结果
            all_preds['rho_final'].append(outputs['rho_final'].cpu())
            all_preds['kappa_final'].append(outputs['kappa_final'].cpu())
            all_preds['rho_pred'].append(outputs['rho_pred'].cpu())
            all_preds['kappa_pred'].append(outputs['kappa_pred'].cpu())
            all_preds['structural_sim'].append(outputs['structural_sim'].cpu())

            # 收集真值
            all_targets['rho'].append(targets_device['rho'].cpu())
            all_targets['kappa'].append(targets_device['kappa'].cpu())
            all_targets['sim'].append(targets_device['sim'].cpu())

    # 拼接所有批次
    result = {
        'predictions': {k: torch.cat(v, dim=0) for k, v in all_preds.items()},
        'targets': {k: torch.cat(v, dim=0) for k, v in all_targets.items()},
    }
    return result


def analyze_full_results(inference_result):
    """
    基于推理结果计算完整的评估指标。

    Args:
        inference_result: run_inference() 返回的结果字典

    Returns:
        dict: 包含所有指标的嵌套字典
    """
    preds = inference_result['predictions']
    targets = inference_result['targets']

    results = {}

    # --- Task 4/5 联合反演最终输出指标 ---
    final_pred = {'rho_final': preds['rho_final'], 'kappa_final': preds['kappa_final']}
    final_target = {'rho': targets['rho'], 'kappa': targets['kappa']}
    results['joint_inversion'] = compute_metrics(final_pred, final_target)

    # --- Task 1/2 独立反演输出指标 (用于与联合方法对比) ---
    indep_pred = {'rho_final': preds['rho_pred'], 'kappa_final': preds['kappa_pred']}
    results['independent_inversion'] = compute_metrics(indep_pred, final_target)

    # --- 结构相似性分类准确率 ---
    s_pred_flat = preds['structural_sim'].flatten()
    s_true_flat = targets['sim'].flatten()
    struct_acc = compute_structural_accuracy(s_pred_flat, s_true_flat, threshold=0.5)
    results['structural_similarity'] = {
        'accuracy': struct_acc,
        'n_pixels': int(s_pred_flat.numel()),
        'pred_mean': s_pred_flat.mean().item(),
        'pred_std': s_pred_flat.std().item(),
        'true_positive_ratio': (s_true_flat > 0.5).float().mean().item(),
    }

    return results


def compare_with_paper(our_metrics, paper_targets=PAPER_TARGETS):
    """
    将我们的实验结果与论文声称的目标值进行对比。

    Args:
        our_metrics: analyze_full_results() 返回的指标字典
        paper_targets: 论文目标值字典

    Returns:
        dict: 对比结果
    """
    comparison = {}

    joint_ours = our_metrics.get('joint_inversion', {})
    indep_ours = our_metrics.get('independent_inversion', {})
    struct_ours = our_metrics.get('structural_similarity', {})

    # 联合反演 vs 论文 DL Joint
    comparison['DL_Joint'] = {}
    for prop in ['rho', 'kappa']:
        comparison['DL_Joint'][prop] = {}
        for metric_name in ['MSE', 'RMSE', 'MAE', 'Correlation']:
            our_val = joint_ours.get(prop, {}).get(metric_name)
            paper_val = paper_targets.get('DL_Joint', {}).get(prop, {}).get(metric_name)

            entry = {'ours': our_val}
            if paper_val is not None:
                entry['paper'] = paper_val
                if our_val is not None and paper_val != 0:
                    deviation_pct = ((our_val - paper_val) / abs(paper_val)) * 100
                    entry['deviation_pct'] = round(deviation_pct, 2)
                    entry['within_tolerance'] = abs(deviation_pct) <= 10.0
            else:
                entry['paper'] = 'TBD'
                entry['deviation_pct'] = None
                entry['within_tolerance'] = None

            comparison['DL_Joint'][prop][metric_name] = entry

    # 独立反演 vs 论文 Independent
    comparison['Independent'] = {}
    for prop in ['rho', 'kappa']:
        comparison['Independent'][prop] = {}
        for metric_name in ['MSE', 'RMSE', 'MAE', 'Correlation']:
            our_val = indep_ours.get(prop, {}).get(metric_name)
            paper_val = paper_targets.get('Independent', {}).get(prop, {}).get(metric_name)

            entry = {'ours': our_val}
            if paper_val is not None:
                entry['paper'] = paper_val
                if our_val is not None and paper_val != 0:
                    deviation_pct = ((our_val - paper_val) / abs(paper_val)) * 100
                    entry['deviation_pct'] = round(deviation_pct, 2)
            else:
                entry['paper'] = 'TBD'
                entry['deviation_pct'] = None

            comparison['Independent'][prop][metric_name] = entry

    # 结构相似性准确率
    comparison['Structural_Accuracy'] = {
        'ours': struct_ours.get('accuracy'),
        'paper': paper_targets.get('DL_Joint', {}).get('structural_accuracy'),
    }
    if comparison['Structural_Accuracy']['paper'] is not None:
        oa = comparison['Structural_Accuracy']['ours']
        op = comparison['Structural_Accuracy']['paper']
        if oa is not None and op != 0:
            comparison['Structural_Accuracy']['deviation_pct'] = round(
                ((oa - op) / abs(op)) * 100, 2
            )
    else:
        comparison['Structural_Accuracy']['paper'] = 'TBD'

    return comparison


def generate_comparison_markdown(comparison, output_path):
    """
    生成 Markdown 格式的对比表格。

    Args:
        comparison: compare_with_paper() 返回的对比字典
        output_path: 输出文件路径
    """
    lines = []
    lines.append("# Results Comparison with Paper")
    lines.append("")
    lines.append(f"> Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 表1: DL Joint Inversion 指标对比
    lines.append("## Table 1: DL Joint Inversion Metrics")
    lines.append("")
    lines.append("| Property | Metric | Ours | Paper | Deviation (%) | Within +/-10% |")
    lines.append("|----------|--------|------|-------|----------------|---------------|")
    for prop in ['rho', 'kappa']:
        for mname in ['MSE', 'RMSE', 'MAE', 'Correlation']:
            entry = comparison.get('DL_Joint', {}).get(prop, {}).get(mname, {})
            ours = f"{entry.get('ours', 'N/A'):.6f}" if isinstance(entry.get('ours'), float) else str(entry.get('ours', 'N/A'))
            paper = str(entry.get('paper', 'TBD'))
            dev = f"{entry.get('deviation_pct', 'N/A')}" if entry.get('deviation_pct') is not None else 'N/A'
            within = "YES" if entry.get('within_tolerance') is True else ("NO" if entry.get('within_tolerance') is False else "N/A")
            lines.append(f"| {prop} | {mname} | {ours} | {paper} | {dev} | {within} |")
    lines.append("")

    # 表2: Independent Inversion 指标对比
    lines.append("## Table 2: Independent Inversion Metrics")
    lines.append("")
    lines.append("| Property | Metric | Ours | Paper | Deviation (%) |")
    lines.append("|----------|--------|------|-------|----------------|")
    for prop in ['rho', 'kappa']:
        for mname in ['MSE', 'RMSE', 'MAE', 'Correlation']:
            entry = comparison.get('Independent', {}).get(prop, {}).get(mname, {})
            ours = f"{entry.get('ours', 'N/A'):.6f}" if isinstance(entry.get('ours'), float) else str(entry.get('ours', 'N/A'))
            paper = str(entry.get('paper', 'TBD'))
            dev = f"{entry.get('deviation_pct', 'N/A')}" if entry.get('deviation_pct') is not None else 'N/A'
            lines.append(f"| {prop} | {mname} | {ours} | {paper} | {dev} |")
    lines.append("")

    # 表3: Structural Accuracy
    sa = comparison.get('Structural_Accuracy', {})
    lines.append("## Table 3: Structural Similarity Accuracy")
    lines.append("")
    lines.append("| Metric | Ours | Paper | Deviation (%) |")
    lines.append("|--------|------|-------|----------------|")
    acc_ours = f"{sa.get('ours', 'N/A'):.4f}" if isinstance(sa.get('ours'), float) else str(sa.get('ours', 'N/A'))
    acc_paper = str(sa.get('paper', 'TBD'))
    acc_dev = f"{sa.get('deviation_pct', 'N/A')}" if sa.get('deviation_pct') is not None else 'N/A'
    lines.append(f"| Accuracy | {acc_ours} | {acc_paper} | {acc_dev} |")
    lines.append("")

    # 写入文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Comparison table saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Analyze Joint Inversion Results')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: checkpoints/best_model.pt)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    args = parser.parse_args()

    # 加载 config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("  3D Gravity-Magnetic Joint Inversion — Result Analysis")
    print("=" * 70)
    print(f"Config : {args.config}")
    print(f"Split  : {args.split}")

    # 设置 seed
    seed = config['training']['seed']
    set_seed(seed)
    print(f"Seed   : {seed}")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ===== 数据准备 =====
    from src.data.generate_synthetic import generate_dataset
    from src.data.dataset import JointInversionInMemoryDataset

    data_dir = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']

    npz_exists = os.path.exists(os.path.join(data_dir, 'train_dataset.npz'))

    if not npz_exists:
        print("\n[Data] No pre-generated dataset found. Generating synthetic data...")
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

    print(f"\n[Data] Evaluating on {args.split} set ({len(eval_loader.dataset)} samples)")

    # ===== 模型加载 =====
    from src.model.joint_inversion_net import JointInversionNet

    model_config = config.get('model', {})
    use_gc = config['training'].get('gradient_checkpointing', False)
    model = JointInversionNet(
        in_channels=model_config.get('in_channels', 2),
        use_gradient_checkpointing=use_gc,
    ).to(device)

    param_info = count_parameters(model)
    print(f"[Model] Parameters: {param_info['trainable']:,} trainable / "
          f"{param_info['total']:,} total")

    # 加载 checkpoint
    ckpt_path = args.checkpoint or os.path.join(
        config['output']['checkpoint_dir'], 'best_model.pt'
    )
    if os.path.exists(ckpt_path):
        info = load_checkpoint(ckpt_path, model)
        print(f"[Checkpoint] Loaded: {ckpt_path} (epoch={info['epoch']}, loss={info['loss']:.6f})")
    else:
        print(f"[WARNING] Checkpoint not found: {ckpt_path}, using random weights!")

    # ===== 推理 =====
    print("\n[Inference] Running inference on test set...")
    start_time = time.time()
    inference_result = run_inference(model, eval_loader, device)
    elapsed = time.time() - start_time
    print(f"[Inference] Completed in {elapsed:.2f}s")

    # ===== 指标计算 =====
    print("\n[Metrics] Computing evaluation metrics...")
    metrics = analyze_full_results(inference_result)

    # 打印联合反演指标
    print("\n" + "-" * 60)
    print("  Joint Inversion (Task 4+5) — Final Output Metrics")
    print("-" * 60)
    joint_m = metrics['joint_inversion']
    for prop in ['rho', 'kappa']:
        m = joint_m[prop]
        print(f"  [{prop.upper():5s}] MSE={m['MSE']:.6f}  RMSE={m['RMSE']:.6f}  "
              f"MAE={m['MAE']:.6f}  Corr={m['Correlation']:.6f}")

    # 打印独立反演指标
    print("\n" + "-" * 60)
    print("  Independent Inversion (Task 1+2) — Baseline Metrics")
    print("-" * 60)
    indep_m = metrics['independent_inversion']
    for prop in ['rho', 'kappa']:
        m = indep_m[prop]
        print(f"  [{prop.upper():5s}] MSE={m['MSE']:.6f}  RMSE={m['RMSE']:.6f}  "
              f"MAE={m['MAE']:.6f}  Corr={m['Correlation']:.6f}")

    # 打印结构相似性指标
    print("\n" + "-" * 60)
    print("  Structural Similarity (Task 3)")
    print("-" * 60)
    sm = metrics['structural_similarity']
    print(f"  Classification Accuracy: {sm['accuracy']:.4f} ({sm['accuracy']*100:.2f}%)")
    print(f"  Total pixels evaluated: {sm['n_pixels']:,}")
    print(f"  Pred S mean/std: {sm['pred_mean']:.4f} / {sm['pred_std']:.4f}")
    print(f"  True positive ratio: {sm['true_positive_ratio']:.4f}")

    # ===== 与论文对比 =====
    print("\n[Comparison] Comparing with paper target values...")
    comparison = compare_with_paper(metrics)

    # ===== 保存结果 =====
    result_dir = config['output']['result_dir']
    os.makedirs(result_dir, exist_ok=True)

    # 1) JSON 格式完整指标
    output_json = {
        "metrics": {
            "joint_inversion": {
                k: {mk: float(mv) for mk, mv in v.items()}
                for k, v in metrics['joint_inversion'].items()
            },
            "independent_inversion": {
                k: {mk: float(mv) for mk, mv in v.items()}
                for k, v in metrics['independent_inversion'].items()
            },
            "structural_similarity": {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in metrics['structural_similarity'].items()
            },
        },
        "comparison_with_paper": comparison,
        "config": {
            "checkpoint": ckpt_path,
            "split": args.split,
            "seed": seed,
            "n_samples": len(eval_loader.dataset),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_info": (
            f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available()
            else "CPU"
        ),
        "model_params": param_info,
    }

    json_path = os.path.join(result_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2, default=str)
    print(f"\n[Save] Full metrics JSON: {json_path}")

    # 2) Markdown 对比表
    md_path = os.path.join(result_dir, 'comparison_table.md')
    generate_comparison_markdown(comparison, md_path)

    # ===== 总结 =====
    print("\n" + "=" * 70)
    print("  Analysis Complete!")
    print("=" * 70)
    print(f"  Output files:")
    print(f"    - {json_path}")
    print(f"    - {md_path}")
    print(f"  GPU used: {output_json['gpu_info']}")
    print(f"  Time: {elapsed:.2f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
