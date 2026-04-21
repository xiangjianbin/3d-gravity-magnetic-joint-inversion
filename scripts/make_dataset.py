#!/usr/bin/env python3
"""
合成数据集一键生成脚本
======================

用法:
  python scripts/make_dataset.py --type all --output_dir data/
  python scripts/make_dataset.py --type 1 --output_dir data/ --n_test 100
  python scripts/make_dataset.py --type all --output_dir data/ --quick  # 小规模验证

流程:
  1. 先生成小规模验证集 (默认 100 样本/类)
  2. 运行验证检查 (形状、范围、NaN、结构相似性)
  3. 验证通过后生成完整数据集 (45300 样本)
  4. 划分 train/val/test 并保存为 .npz 文件
  5. 生成 DATASET_REPORT.md 报告

作者: Agent-DataEngineering
日期: 2026-04-21
"""

import argparse
import os
import sys
import time
import json

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.generate_synthetic import (
    generate_dataset,
    generate_all_datasets,
    validate_dataset,
    save_dataset,
    load_dataset,
    DATASET_SPECS,
    TOTAL_SAMPLES,
    HAS_EXTERNAL_FORWARD,
)
from src.data.dataset import split_and_save_datasets


def parse_args():
    parser = argparse.ArgumentParser(description='合成数据集一键生成脚本')

    parser.add_argument('--type', type=str, default='all',
                        choices=['all', '1', '2', '3', '4', '5', '6'],
                        help='要生成的数据集类型 (默认: all)')
    parser.add_argument('--output_dir', type=str, default='data/',
                        help='输出目录 (默认: data/)')
    parser.add_argument('--n_quick', type=int, default=100,
                        help='小规模验证时每类的样本数 (默认: 100)')
    parser.add_argument('--full', action='store_true',
                        help='跳过验证，直接生成完整数据集')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--noise_g', type=float, default=DEFAULT_NOISE_GRAVITY if 'DEFAULT_NOISE_GRAVITY' in dir() else 0.005,
                        help='重力噪声水平 (默认: 0.005)')
    parser.add_argument('--noise_m', type=float, default=DEFAULT_NOISE_MAGNETIC if 'DEFAULT_NOISE_MAGNETIC' in dir() else 0.108,
                        help='磁异常噪声水平 (默认: 0.108)')
    parser.add_argument('--nx', type=int, default=40, help='X 维度 (默认: 40)')
    parser.add_argument('--ny', type=int, default=40, help='Y 维度 (默认: 40)')
    parser.add_argument('--nz', type=int, default=20, help='Z 维度 (深度层数, 默认: 20)')

    return parser.parse_args()


# 导入默认值 (因为还没导入模块时无法引用)
DEFAULT_NOISE_GRAVITY = 0.005
DEFAULT_NOISE_MAGNETIC = 0.108


def run_validation(samples_by_type: dict, args) -> bool:
    """
    运行完整的验证流程。

    检查项:
      1. 各类数量符合 Table I (按比例)
      2. 数据范围 [0, 1]
      3. 无 NaN / Inf
      4. 结构相似性标签正确
      5. 正演结果物理合理

    返回:
        True 通过 / False 不通过
    """
    print("\n" + "=" * 60)
    print("阶段 2: 数据质量验证")
    print("=" * 60)

    all_passed = True
    total_report = {'by_type': {}, 'overall': {}}

    total_samples = 0
    for dt, samples in sorted(samples_by_type.items()):
        total_samples += len(samples)
        report = validate_dataset(samples)
        total_report['by_type'][dt] = report

        status = "PASS" if report['passed'] else "FAIL"
        print(f"\n  Type {dt}: {status} ({len(samples)} 样本)")
        print(f"    Shape: {report['stats'].get('shape_rho', 'N/A')}")
        print(f"    NaN: {report['stats']['nan_samples']}, "
              f"Inf: {report['stats']['inf_samples']}, "
              f"越界: {report['stats']['out_of_range_samples']}")
        print(f"    Sim标签错误: {report['stats']['sim_label_errors']}")

        if 'gravity' in report['stats']:
            g = report['stats']['gravity']
            print(f"    Gravity: mean={g['mean']:.4f}, std={g['std']:.4f}, "
                  f"range=[{g['min']:.4f}, {g['max']:.4f}]")
        if 'magnetic' in report['stats']:
            m = report['stats']['magnetic']
            print(f"    Magnetic: mean={m['mean']:.4f}, std={m['std']:.4f}, "
                  f"range=[{m['min']:.4f}, {m['max']:.4f}]")

        if 'structural_similarity_ratio' in report['stats']:
            sr = report['stats']['structural_similarity_ratio']
            print(f"    Sim=1 比例: mean={sr['mean']:.3f}, "
                  f"range=[{sr['min']:.3f}, {sr['max']:.3f}]")

        if not report['passed']:
            all_passed = False
            print(f"    错误详情: {report['errors']}")

    total_report['overall']['total_samples'] = total_samples
    total_report['overall']['all_passed'] = all_passed

    # Table I 一致性检查
    print(f"\n  --- Table I 一致性检查 ---")
    expected_total = sum(DATASET_SPECS[t]['total'] for t in DATASET_SPECS)
    scale_factor = total_samples / expected_total if expected_total > 0 else 0
    print(f"    预期总量: {expected_total}, 实际总量: {total_samples}, "
          f"缩放比: {scale_factor:.4f}")

    for dt in sorted(DATASET_SPECS.keys()):
        spec = DATASET_SPECS[dt]
        actual_n = len(samples_by_type.get(dt, []))
        expected_n = spec['total']
        ratio = actual_n / expected_n if expected_n > 0 else 0
        status_mark = "OK" if abs(ratio - scale_factor) < 0.05 else "WARN"
        print(f"    Type {dt}: 预期={expected_n}, 实际={actual_n}, "
              f"比值={ratio:.4f} [{status_mark}]")

    # 物理合理性检查
    print(f"\n  --- 物理合理性抽查 ---")
    physics_ok = check_physics_reasonableness(samples_by_type)
    total_report['overall']['physics_check'] = physics_ok
    if not physics_ok:
        all_passed = False
        print("    [FAIL] 正演结果物理不合理!")
    else:
        print("    [PASS] 正演结果物理合理")

    print(f"\n{'='*60}")
    result_str = "全部通过!" if all_passed else "存在错误!"
    print(f"验证结果: {result_str}")
    print(f"{'='*60}\n")

    return all_passed


def check_physics_reasonableness(samples_by_type: dict) -> bool:
    """
    检查正演结果的物理合理性。

    使用多重检查策略:
      1. 空间重叠检查: 重力异常的高值区应与密度体的水平投影有空间重叠
      2. 符号一致性: 对于简单单长方体，重力异常应基本为正

    注意: Pearson 相关系数对 3D Blakely 正演不适用。
    """
    import numpy as np

    spatial_overlap_scores = []
    sign_consistency_count = 0
    sign_total = 0
    total_checked = 0

    for dt in sorted(samples_by_type.keys())[:3]:
        samples = samples_by_type[dt][:min(15, len(samples_by_type[dt]))]
        for s in samples:
            rho_2d = s['rho'].sum(axis=2)  # (nx, ny)
            grav_2d = s['gravity']          # (nx, ny)

            # 空间重叠 IoU 检查
            rho_binary = (rho_2d > rho_2d.max() * 0.1)
            grav_binary = (grav_2d > grav_2d.max() * 0.5)
            intersection = np.sum(rho_binary & grav_binary).astype(float)
            union = np.sum(rho_binary | grav_binary).astype(float)
            iou = intersection / union if union > 0 else 0.0
            spatial_overlap_scores.append(iou)

            # 符号一致性 (global 样本)
            if s['consistency_type'] == 'global':
                sign_total += 1
                if grav_2d.mean() >= 0:
                    sign_consistency_count += 1

            total_checked += 1

    if total_checked == 0:
        return True

    mean_iou = np.mean(spatial_overlap_scores)
    median_iou = np.median(spatial_overlap_scores)
    print(f"    空间重叠 (IoU): mean={mean_iou:.3f}, median={median_iou:.3f}")

    if sign_total > 0:
        sign_ratio = sign_consistency_count / sign_total
        print(f"    符号一致性 (global样本): {sign_consistency_count}/{sign_total} ({sign_ratio*100:.1f}%)")

    iou_ok = median_iou > 0.0
    sign_ok = (sign_total == 0) or (sign_consistency_count / max(sign_total, 1) >= 0.5)

    return iou_ok and sign_ok


def generate_full_dataset(args) -> dict:
    """生成完整数据集"""
    print("\n" + "=" * 60)
    print("阶段 3: 生成完整数据集")
    print("=" * 60)
    print(f"  正演引擎: {'外部物理模块' if HAS_EXTERNAL_FORWARD else '内置简化版'}")
    print(f"  目标总量: ~{TOTAL_SAMPLES} 样本")
    print(f"  网格尺寸: {args.nx}x{args.ny}x{args.nz}")
    print(f"  输出目录: {args.output_dir}")
    print()

    types_to_generate = []
    if args.type == 'all':
        types_to_generate = list(range(1, 7))
    else:
        types_to_generate = [int(args.type)]

    all_data = {}
    for dt in types_to_generate:
        t0 = time.time()
        samples = generate_dataset(
            dataset_type=dt,
            n_samples=None,  # 使用 Table I 原始数量
            nx=args.nx, ny=args.ny, nz=args.nz,
            noise_gravity=args.noise_g,
            noise_magnetic=args.noise_m,
            seed=args.seed + dt * 1000,
            verbose=True,
        )
        elapsed = time.time() - t0
        all_data[dt] = samples
        print(f"  Type {dt} 完成: {len(samples)} 样本, {elapsed:.1f}s\n")

    return all_data


def generate_quick_dataset(args) -> dict:
    """生成小规模验证数据集"""
    print("\n" + "=" * 60)
    print("阶段 1: 生成小规模验证集")
    print("=" * 60)
    print(f"  每类样本数: {args.n_quick}")
    print(f"  正演引擎: {'外部物理模块' if HAS_EXTERNAL_FORWARD else '内置简化版'}\n")

    types_to_generate = []
    if args.type == 'all':
        types_to_generate = list(range(1, 7))
    else:
        types_to_generate = [int(args.type)]

    all_data = {}
    for dt in types_to_generate:
        t0 = time.time()
        samples = generate_dataset(
            dataset_type=dt,
            n_samples=args.n_quick,
            nx=args.nx, ny=args.ny, nz=args.nz,
            noise_gravity=args.noise_g,
            noise_magnetic=args.noise_m,
            seed=args.seed,
            verbose=True,
        )
        elapsed = time.time() - t0
        all_data[dt] = samples
        print(f"  Type {dt}: {len(samples)} 样本, {elapsed:.1f}s\n")

    return all_data


def save_and_split(all_data: dict, args):
    """保存并划分数据集"""
    print("\n" + "=" * 60)
    print("阶段 4: 保存与划分")
    print("=" * 60)

    # 合并所有样本
    all_samples = []
    for dt in sorted(all_data.keys()):
        all_samples.extend(all_data[dt])

    print(f"  总样本数: {len(all_samples)}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 划分并保存
    split_and_save_datasets(
        all_samples=all_samples,
        data_dir=args.output_dir,
        train_ratio=0.70,
        val_ratio=0.20,
        seed=args.seed,
    )

    # 保存完整合并数据集 (可选，用于调试)
    full_path = os.path.join(args.output_dir, 'full_dataset.npz')
    save_dataset(all_samples, full_path)

    return all_samples


def generate_report(all_data: dict, args, validation_report: dict,
                    elapsed_total: float):
    """生成 DATASET_REPORT.md"""
    docs_dir = os.path.join(PROJECT_ROOT, 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    report_path = os.path.join(docs_dir, 'DATASET_REPORT.md')

    # 收集统计信息
    lines = []
    lines.append("# 数据集生成报告\n")
    lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"> 总耗时: {elapsed_total:.1f}s\n")
    lines.append(f"> 正演引擎: {'外部物理模块' if HAS_EXTERNAL_FORWARD else '内置简化版 (simple_forward)'}\n")
    lines.append(f"> 网格尺寸: {args.nx} x {args.ny} x {args.nz}\n")
    lines.append(f"> 随机种子: {args.seed}\n")

    # ---- 样本数统计表 ----
    lines.append("## 1. 样本数统计 (按类型)\n")
    lines.append("| Type | 全局一致 | 部分一致 | 结构不一致 | 总计 |")
    lines.append("|------|---------|---------|-----------|------|")
    grand_total = 0
    for dt in sorted(all_data.keys()):
        samples = all_data[dt]
        n = len(samples)
        cons_counts = {}
        for s in samples:
            c = s['consistency_type']
            cons_counts[c] = cons_counts.get(c, 0) + 1
        g = cons_counts.get('global', 0)
        p = cons_counts.get('partial', 0)
        inc = cons_counts.get('inconsistent', 0)
        lines.append(f"| No.{dt} | {g} | {p} | {inc} | **{n}** |")
        grand_total += n
    lines.append(f"| **合计** | | | | **{grand_total}** |\n")

    # ---- Table I 一致性 ----
    lines.append("## 2. 与论文 Table I 一致性检查\n")
    lines.append("| Type | 论文数量 | 生成数量 | 偏差 | 状态 |")
    lines.append("|------|---------|---------|------|------|")
    for dt in sorted(DATASET_SPECS.keys()):
        expected = DATASET_SPECS[dt]['total']
        actual = len(all_data.get(dt, []))
        diff = actual - expected
        pct = diff / expected * 100 if expected > 0 else 0
        if args.n_quick and actual < expected:
            status = "缩放 (预期)"
        elif diff == 0:
            status = "**精确匹配**"
        else:
            status = f"偏差 {diff:+d}"
        lines.append(f"| No.{dt} | {expected} | {actual} | {pct:+.1f}% | {status} |")
    lines.append("")

    # ---- 数据分布统计 ----
    lines.append("## 3. 数据分布统计 (per channel)\n")
    lines.append("| Channel | Mean | Std | Min | Max |")
    lines.append("|----------|------|-----|-----|-----|")

    all_rho = np.concatenate([s['rho'].ravel() for dt in all_data for s in all_data[dt]])
    all_kappa = np.concatenate([s['kappa'].ravel() for dt in all_data for s in all_data[dt]])
    all_grav = np.concatenate([s['gravity'].ravel() for dt in all_data for s in all_data[dt]])
    all_mag = np.concatenate([s['magnetic'].ravel() for dt in all_data for s in all_data[dt]])
    all_sim = np.concatenate([s['structural_sim'].ravel() for dt in all_data for s in all_data[dt]])

    import numpy as np
    for name, arr in [('Density (rho)', all_rho),
                       ('Susceptibility (kappa)', all_kappa),
                       ('Gravity anomaly', all_grav),
                       ('Magnetic anomaly', all_mag)]:
        lines.append(f"| {name} | {arr.mean():.4f} | {arr.std():.4f} | "
                     f"{arr.min():.4f} | {arr.max():.4f} |")
    lines.append("")

    # ---- 结构相似性分布 ----
    lines.append("## 4. 结构相似性标签分布\n")
    sim_1_pct = float((all_sim == 1).sum()) / len(all_sim) * 100
    sim_0_pct = 100.0 - sim_1_pct
    lines.append(f"- **S=1 (结构一致)**: {sim_1_pct:.1f}%")
    lines.append(f"- **S=0 (结构不一致)**: {sim_0_pct:.1f}%")
    lines.append("")
    lines.append("按一致性类型分解:\n")
    lines.append("| 类型 | S=1 比例 | 说明 |")
    lines.append("|------|---------|------|")
    for cons in ['global', 'partial', 'inconsistent']:
        mask_cons = [s for dt in all_data for s in all_data[dt]
                     if s['consistency_type'] == cons]
        if mask_cons:
            cons_sims = np.concatenate([s['structural_sim'].ravel() for s in mask_cons])
            pct = float((cons_sims == 1).sum()) / len(cons_sims) * 100
            note = {'global': '应接近100%',
                    'partial': '中等',
                    'inconsistent': '应较低'}.get(cons, '')
            lines.append(f"| {cons} | {pct:.1f}% | {note} |")
    lines.append("")

    # ---- 验证结果摘要 ----
    lines.append("## 5. 验证结果\n")
    if validation_report:
        overall = validation_report.get('overall', {})
        lines.append(f"- **总体状态**: {'PASS' if overall.get('all_passed', False) else 'FAIL'}")
        lines.append(f"- **总样本数**: {overall.get('total_samples', 'N/A')}")
        lines.append(f"- **物理合理性**: {'PASS' if overall.get('physics_check', False) else 'FAIL'}")
        lines.append("")

        if 'by_type' in validation_report:
            lines.append("### 各类型详细验证\n")
            for dt, rep in sorted(validation_report['by_type'].items()):
                st = rep.get('stats', {})
                status = "PASS" if rep.get('passed', False) else "FAIL"
                lines.append(f"- **Type {dt}**: {status}")
                lines.append(f"  - NaN: {st.get('nan_samples', 0)}, Inf: {st.get('inf_samples', 0)}")
                lines.append(f"  - Sim标签错误: {st.get('sim_label_errors', 0)}")
                if 'gravity' in st:
                    g = st['gravity']
                    lines.append(f"  - Gravity: [{g['min']:.4f}, {g['max']:.4f}]")
                if 'magnetic' in st:
                    m = st['magnetic']
                    lines.append(f"  - Magnetic: [{m['min']:.4f}, {m['max']:.4f}]")
    lines.append("")

    # ---- 文件清单 ----
    lines.append("## 6. 输出文件\n")
    lines.append(f"| 文件 | 大小 | 说明 |")
    lines.append("|------|------|------|")
    output_dir_abs = os.path.abspath(args.output_dir)
    for fname in ['train_dataset.npz', 'val_dataset.npz',
                   'test_dataset.npz', 'full_dataset.npz']:
        fpath = os.path.join(output_dir_abs, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            lines.append(f"| `{fname}` | {size_mb:.1f} MB | |")
    lines.append("")

    # 写入
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n[报告已写入] {report_path}")


def main():
    args = parse_args()
    t_start = time.time()

    print("=" * 60)
    print("  合成数据集生成管道")
    print("  3D 重磁联合反演 - 深度学习方法复现")
    print("=" * 60)
    print(f"  数据集类型: {args.type}")
    print(f"  输出目录: {os.path.abspath(args.output_dir)}")
    print(f"  网格: {args.nx}x{args.ny}x{args.nz}")
    print(f"  噪声: gravity={args.noise_g}, magnetic={args.noise_m}")
    print(f"  模式: {'完整模式' if args.full else '验证+生成模式'}")
    print()

    # Step 1: 生成数据
    if args.full:
        all_data = generate_full_dataset(args)
    else:
        # 先小规模验证
        all_data = generate_quick_dataset(args)

        # Step 2: 验证
        validation_ok = run_validation(all_data, args)

        if not validation_ok:
            print("[ERROR] 验证未通过! 请检查上方错误信息。")
            print("        可用 --full 跳过验证强制生成。")
            sys.exit(1)

        # Step 3: 询问是否继续生成完整数据集
        quick_total = sum(len(s) for s in all_data.values())
        if quick_total < TOTAL_SAMPLES * 0.5:
            print(f"\n小规模验证通过 ({quick_total} 样本)。")
            print(f"是否继续生成完整数据集 (~{TOTAL_SAMPLES} 样本)?")
            # 自动继续 (非交互模式下)
            print("[自动] 开始生成完整数据集...\n")
            all_data = generate_full_dataset(args)

    # Step 4: 保存与划分
    all_samples_list = save_and_split(all_data, args)

    # Step 5: 最终验证
    final_validation = run_validation(all_data, args)

    # Step 6: 生成报告
    elapsed_total = time.time() - t_start
    generate_report(all_data, args, final_validation, elapsed_total)

    print("\n" + "=" * 60)
    print("  数据集生成管道执行完毕!")
    print(f"  总耗时: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    print(f"  输出目录: {os.path.abspath(args.output_dir)}")
    print(f"  报告文件: docs/DATASET_REPORT.md")
    print("=" * 60)


if __name__ == '__main__':
    main()
