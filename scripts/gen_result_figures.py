"""
结果图表生成脚本 — 匹配论文图表风格的 publication-quality 图表
==============================================================

生成的图表:
  Fig 1: figures/result_training_curves.pdf
    训练/验证 loss 曲线 (5条任务线 + 总loss)，匹配论文 Fig.5 风格

  Fig 2: figures/result_test_model.pdf
    测试模型展示 (密度/磁化率/重力异常/磁异常)，匹配论文 Fig.8 布局

  Fig 3: figures/result_comparison_slices.pdf
    反演结果切片对比，匹配论文 Fig.9 布局:
    3行 x 4列: 真实模型 / DL联合反演 / Independent反演
    每行4列: 密度z=10 / 密度z=5 / 磁化率z=10 / 磁化率z=5

  Fig 4: figures/result_metrics_table.tex
    LaTeX 格式的指标对比表

  Fig 5: figures/result_prediction_vs_observed.pdf
    预测观测数据 vs 实际观测数据 scatter plot

技术要求:
  - Publication 风格: Times New Roman, 300dpi, PDF 输出
  - 统一色谱
  - 所有脚本可独立运行 (不依赖训练正在进行)

用法:
  python scripts/gen_result_figures.py --config configs/full.yaml

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

# ---------------------------------------------------------------------------
# Matplotlib 全局设置 — Publication Quality
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')  # 非 GUI 后端
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 字体: Times New Roman (或回退到 serif)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['mathtext.fontset'] = 'stix'  # 数学字体与 Times 兼容
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.05
rcParams['axes.linewidth'] = 0.8
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8

# ---------------------------------------------------------------------------
# 统一色谱 (匹配论文风格)
# ---------------------------------------------------------------------------
COLORS = {
    'total':        '#1f77b4',  # 蓝色 — 总 loss
    'task1':        '#ff7f0e',  # 橙色 — Task1 重力 MSE
    'task2':        '#2ca02c',  # 绿色 — Task2 磁法 MSE
    'task3':        '#d62728',  # 红色 — Task3 结构相似性 MSE
    'task4':        '#9467bd',  # 紫色 — Task4 联合重力 BCE
    'task5':        '#8c564b',  # 棕色 — Task5 联合磁法 BCE
    'train':        '#1f77b4',  # 训练集
    'val':          '#d62728',  # 验证集
    'true_model':   '#2c3e50',  # 深灰 — 真实模型
    'dl_joint':     '#e74c3c',  # 红 — DL联合反演
    'independent':  '#3498db',  # 蓝 — Independent 反演
    'density':      '#e74c3c',  # 密度用暖色调
    'susceptibility': '#3498db',  # 磁化率用冷色调
    'gravity':      '#f39c12',  # 重力异常
    'magnetic':     '#9b59b6',  # 磁异常
}

LINESTYLES = {
    'train': '-',
    'val': '--',
}

MARKERS = {
    'train': '',
    'val': 'o',
    'marker_size': 3,
}


# ===========================================================================
# Figure 1: Training Curves (论文 Fig.5 风格)
# ===========================================================================

def plot_training_curves(history_path, output_path):
    """
    绘制训练/验证 loss 曲线。

    从 results/training_history.json 读取数据。
    包含 5 条任务 loss + 总 loss 的 train/val 曲线。

    Args:
        history_path: training_history.json 路径
        output_path: 输出 PDF 路径
    """
    if not os.path.exists(history_path):
        print(f"  [SKIP] Training history not found: {history_path}")
        return None

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = [h['epoch'] for h in history]

    # 要绘制的 loss 键名映射
    loss_keys = [
        ('total',               'Total Loss',           COLORS['total']),
        ('task1_gravity_mse',   'Task1 Gravity MSE',    COLORS['task1']),
        ('task2_magnetic_mse',  'Task2 Magnetic MSE',   COLORS['task2']),
        ('task3_similarity_mse','Task3 Similarity MSE', COLORS['task3']),
        ('task4_gravity_bce',   'Task4 Joint Gravity BCE', COLORS['task4']),
        ('task5_magnetic_bce',  'Task5 Joint Magnetic BCE', COLORS['task5']),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, (key, label, color) in enumerate(loss_keys):
        ax = axes[idx]

        train_vals = [h.get(f'train_{key}', h.get(key, None)) for h in history]
        val_vals = [h.get(f'val_{key}', None) for h in history]

        # 过滤掉 None 值
        valid_epochs_train = [(e, v) for e, v in zip(epochs, train_vals) if v is not None]
        valid_epochs_val = [(e, v) for e, v in zip(epochs, val_vals) if v is not None]

        if valid_epochs_train:
            et, vt = zip(*valid_epochs_train)
            ax.plot(et, vt, color=color, linestyle='-', linewidth=1.2,
                    label='Train')

        if valid_epochs_val:
            ev, vv = zip(*valid_epochs_val)
            ax.plot(ev, vv, color=color, linestyle='--', linewidth=1.2,
                    marker='o', markersize=MARKERS['marker_size'], markerfacecolor='none',
                    label='Val')

        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(loc='upper right', framealpha=0.8)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xlim(left=1)

    # 隐藏多余的子图
    for idx in range(len(loss_keys), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Training and Validation Loss Curves\n(Multitask Learning Strategy)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()

    print(f"  [DONE] {output_path}")
    return output_path


# ===========================================================================
# Figure 2: Test Model Display (论文 Fig.8 风格)
# ===========================================================================

def plot_test_model(sample_input, sample_target, output_path,
                    sample_pred=None, sample_s=None):
    """
    展示测试样本的输入和输出模型。

    布局 (2x2 或 2x3):
      Row 1: 密度模型 (真实 + 预测)
      Row 2: 磁化率模型 (真实 + 预测)
      可选: 重力异常 / 磁异常 观测数据

    Args:
        sample_input: (2, D, H, W) 输入张量 [gravity, magnetic]
        sample_target: dict 含 rho, kappa
        output_path: 输出 PDF 路径
        sample_pred: 可选, dict 含 rho_final, kappa_final
        sample_s: 可选, structural_sim 张量
    """
    gravity_2d = sample_input[0].numpy()[:, :, sample_input.shape[-1] // 2]  # 中间深度切片
    magnetic_2d = sample_input[1].numpy()[:, :, sample_input.shape[-1] // 2]
    rho_true = sample_target['rho'].numpy()
    kappa_true = sample_target['kappa'].numpy()

    n_cols = 3 if sample_pred is not None else 2
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7))

    mid_z = rho_true.shape[0] // 2  # 中间深度层

    # --- Row 1: Density ---
    # 真实密度
    im00 = axes[0, 0].imshow(rho_true[mid_z], cmap='RdBu_r', origin='lower',
                              vmin=0, vmax=max(rho_true.max(), 0.01))
    axes[0, 0].set_title('(a) True Density\nz={}'.format(mid_z), fontsize=10)
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

    if sample_pred is not None:
        rho_pred = sample_pred['rho_final'].numpy()[0, 0]  # (D,H,W)
        im01 = axes[0, 1].imshow(rho_pred[mid_z], cmap='RdBu_r', origin='lower',
                                  vmin=0, vmax=max(rho_pred.max(), 0.01))
        axes[0, 1].set_title('(b) Predicted Density\nDL Joint (z={})'.format(mid_z), fontsize=10)
        plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # 差异图
        diff_rho = np.abs(rho_pred[mid_z] - rho_true[mid_z])
        im02 = axes[0, 2].imshow(diff_rho, cmap='hot', origin='lower',
                                  vmin=0, vmax=max(diff_rho.max(), 0.001))
        axes[0, 2].set_title('(c) |Error| Density\n(z={})'.format(mid_z), fontsize=10)
        plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)
    else:
        # 无预测时显示重力异常
        im01 = axes[0, 1].imshow(gravity_2d, cmap='viridis', origin='lower')
        axes[0, 1].set_title('(b) Gravity Anomaly\n(observed)', fontsize=10)
        plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # --- Row 2: Susceptibility ---
    im10 = axes[1, 0].imshow(kappa_true[mid_z], cmap='RdBu_r', origin='lower',
                              vmin=0, vmax=max(kappa_true.max(), 0.001))
    axes[1, 0].set_title('(d) True Susceptibility\nz={}'.format(mid_z), fontsize=10)
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

    if sample_pred is not None:
        kappa_pred = sample_pred['kappa_final'].numpy()[0, 0]
        im11 = axes[1, 1].imshow(kappa_pred[mid_z], cmap='RdBu_r', origin='lower',
                                  vmin=0, vmax=max(kappa_pred.max(), 0.001))
        axes[1, 1].set_title('(e) Predicted Suscept.\nDL Joint (z={})'.format(mid_z), fontsize=10)
        plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

        diff_kappa = np.abs(kappa_pred[mid_z] - kappa_true[mid_z])
        im12 = axes[1, 2].imshow(diff_kappa, cmap='hot', origin='lower',
                                  vmin=0, vmax=max(diff_kappa.max(), 0.0001))
        axes[1, 2].set_title('(f) |Error| Suscept.\n(z={})'.format(mid_z), fontsize=10)
        plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)
    else:
        im11 = axes[1, 1].imshow(magnetic_2d, cmap='inferno', origin='lower')
        axes[1, 1].set_title('(e) Magnetic Anomaly\n(observed)', fontsize=10)
        plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 设置统一的坐标轴标签
    for row in axes:
        for ax in row:
            ax.set_xlabel('Easting (m)')
            ax.set_ylabel('Northing (m)')

    plt.suptitle('Test Model: Inversion Results', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()

    print(f"  [DONE] {output_path}")
    return output_path


# ===========================================================================
# Figure 3: Comparison Slices (论文 Fig.9 风格)
# ===========================================================================

def plot_comparison_slices(true_data, pred_joint, pred_indep, output_path,
                           z_indices=None):
    """
    反演结果切片对比图。

    布局 (3 x 4):
      Row 1: (a) True Model — density z_idx[0] / density z_idx[1] /
                               susceptibility z_idx[0] / susceptibility z_idx[1]
      Row 2: (d) DL Joint Inversion — 同上 4 列
      Row 3: (g) Independent Inversion — 同上 4 列

    Args:
        true_data: dict 含 rho (D,H,W), kappa (D,H,W) numpy 数组
        pred_joint: dict 含 rho_final (1,D,H,W), kappa_final (1,D,H,W)
        pred_indep: dict 含 rho_pred (1,D,H,W), kappa_pred (1,D,H,W)
        output_path: 输出 PDF 路径
        z_indices: list of 2 depth indices to slice, 默认 [中间层, 近表层]
    """
    if z_indices is None:
        nz = true_data['rho'].shape[0]
        z_indices = [nz // 2, max(nz // 4, 2)]  # 中间层 + 上部层

    rho_true = true_data['rho']
    kappa_true = true_data['kappa']

    # 提取预测 (去掉 batch 维度)
    rho_joint = pred_joint['rho_final'][0, 0].numpy() if pred_joint is not None else None
    kappa_joint = pred_joint['kappa_final'][0, 0].numpy() if pred_joint is not None else None
    rho_indep = pred_indep['rho_pred'][0, 0].numpy() if pred_indep is not None else None
    kappa_indep = pred_indep['kappa_pred'][0, 0].numpy() if pred_indep is not None else None

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    col_titles = [
        f'Density z={z_indices[0]}',
        f'Density z={z_indices[1]}',
        f'Susceptibility z={z_indices[0]}',
        f'Susceptibility z={z_indices[1]}',
    ]
    row_labels = ['(a-c) True Model', '(d-f) DL Joint Inversion', '(g-i) Independent']

    data_grid = [
        # Row 0: True
        [rho_true[z_indices[0]], rho_true[z_indices[1]],
         kappa_true[z_indices[0]], kappa_true[z_indices[1]]],
        # Row 1: DL Joint
        [rho_joint[z_indices[0]] if rho_joint is not None else None,
         rho_joint[z_indices[1]] if rho_joint is not None else None,
         kappa_joint[z_indices[0]] if kappa_joint is not None else None,
         kappa_joint[z_indices[1]] if kappa_joint is not None else None],
        # Row 2: Independent
        [rho_indep[z_indices[0]] if rho_indep is not None else None,
         rho_indep[z_indices[1]] if rho_indep is not None else None,
         kappa_indep[z_indices[0]] if kappa_indep is not None else None,
         kappa_indep[z_indices[1]] if kappa_indep is not None else None],
    ]

    # 使用真实数据的全局范围作为 colormap 范围
    rho_vmax = max(rho_true.max(), 0.01)
    kappa_vmax = max(kappa_true.max(), 0.001)

    cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r']
    vmaxs = [rho_vmax, rho_vmax, kappa_vmax, kappa_vmax]
    vmins = [0, 0, 0, 0]

    for row_idx in range(3):
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            data = data_grid[row_idx][col_idx]

            if data is not None:
                im = ax.imshow(data, cmap=cmaps[col_idx], origin='lower',
                               vmin=vmins[col_idx], vmax=vmaxs[col_idx])
                # 只在每行的第一个子图加 colorbar
                if col_idx == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14, color='gray')

            # 列标题 (只在第一行)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10, fontweight='bold')

            # 行标签 (只在第一列)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=10, fontweight='bold')

            ax.set_xlabel('Easting')
            if col_idx == 0:
                ax.set_ylabel('Northing')

            ax.tick_params(labelsize=8)

    plt.suptitle('Inversion Result Comparison: Slice View',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()

    print(f"  [DONE] {output_path}")
    return output_path


# ===========================================================================
# Figure 4: LaTeX Metrics Table
# ===========================================================================

def generate_latex_metrics_table(metrics_json_path, output_path):
    """
    生成 LaTeX 格式的指标对比表。

    Table 结构:
      \begin{table}[t]
      \caption{...}
      ...

    Args:
        metrics_json_path: results/metrics.json 路径
        output_path: 输出 .tex 文件路径
    """
    if not os.path.exists(metrics_json_path):
        print(f"  [SKIP] Metrics JSON not found: {metrics_json_path}")
        # 生成模板表格 (占位符)
        latex_content = _generate_placeholder_latex_table()
    else:
        with open(metrics_json_path, 'r') as f:
            metrics_data = json.load(f)

        joint_m = metrics_data.get('metrics', {}).get('joint_inversion', {})
        indep_m = metrics_data.get('metrics', {}).get('independent_inversion', {})
        struct_acc = metrics_data.get('metrics', {}).get('structural_similarity', {}).get('accuracy')
        comparison = metrics_data.get('comparison_with_paper', {})

        timestamp = metrics_data.get('timestamp', 'N/A')
        gpu_info = metrics_data.get('gpu_info', 'N/A')

        latex_content = _build_latex_table(
            joint_m, indep_m, struct_acc, comparison, timestamp, gpu_info
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex_content)

    print(f"  [DONE] {output_path}")
    return output_path


def _generate_placeholder_latex_table():
    """生成占位符 LaTeX 表格 (训练前使用)。"""
    return r"""%% LaTeX Metrics Comparison Table
%% Generated by gen_result_figures.py (placeholder — fill after training)
%%
%% Usage in paper:
%%   \input{figures/result_metrics_table.tex}
%%

\begin{table}[tb]
\centering
\caption{Quantitative Comparison of Inversion Methods}
\label{tab:metrics_comparison}
\begin{tabular}{llcccc}
\toprule
\multirow{2}{*}{Method} & \multirow{2}{*}{Property} &
\multicolumn{2}{c}{MSE} & \multicolumn{2}{c}{Correlation} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
 & & Ours & Paper & Ours & Paper \\
\midrule
\multirow{2}{*}{DL Joint (Ours)}
 & $\rho$ (density)       & TBD & TBD & TBD & TBD \\
 & $\kappa$ (suscept.)   & TBD & TBD & TBD & TBD \\
\midrule
\multirow{2}{*}{Independent}
 & $\rho$ (density)       & TBD & TBD & TBD & TBD \\
 & $\kappa$ (suscept.)   & TBD & TBD & TBD & TBD \\
\bottomrule
\end{tabular}
\end{table}
"""


def _build_latex_table(joint_m, indep_m, struct_acc, comparison, timestamp, gpu_info):
    """根据实际指标构建 LaTeX 表格。"""

    def fmt_val(val, fmt='.6f'):
        """格式化数值。"""
        if val is None:
            return 'TBD'
        try:
            return f'{float(val):{fmt}}'
        except (ValueError, TypeError):
            return str(val)

    def get_paper_val(comparison_dict, method, prop, metric):
        """从对比字典中获取论文值。"""
        entry = (comparison_dict or {}).get(method, {}).get(prop, {}).get(metric, {})
        paper = entry.get('paper', 'TBD')
        if isinstance(paper, float):
            return f'{paper:.6f}'
        return str(paper)

    lines = []
    lines.append(r"""%% LaTeX Metrics Comparison Table
%% Auto-generated by gen_result_figures.py
%% Timestamp: %s
%% GPU: %s
%%
%% Usage in paper:
%%   \input{figures/result_metrics_table.tex}

\begin{table}[tb]
\centering
\caption{Quantitative Comparison of Inversion Methods}
\label{tab:metrics_comparison}
\begin{tabular}{llcccc}
\toprule
\multirow{2}{*}{Method} & \multirow{2}{*}{Property} &
\multicolumn{2}{c}{MSE ($\\times 10^{-3}$)} & \multicolumn{2}{c}{Pearson $r$} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
 & & Ours & Paper & Ours & Paper \\
\midrule""" % (timestamp, gpu_info))

    # DL Joint 行
    for prop, prop_name in [('rho', '$\\rho$ (density)'),
                             ('kappa', '$\\kappa$ (suscept.)')]:
        jv = joint_m.get(prop, {})
        mse_ours = fmt_val(jv.get('MSE')) if jv else 'TBD'
        corr_ours = fmt_val(jv.get('Correlation')) if jv else 'TBD'
        mse_paper = get_paper_val(comparison, 'DL_Joint', prop, 'MSE')
        corr_paper = get_paper_val(comparison, 'DL_Joint', prop, 'Correlation')
        lines.append(
            r" & %s & %s & %s & %s & %s \\" %
            (prop_name, mse_ours, mse_paper, corr_ours, corr_paper)
        )

    lines.append(r"\midrule")

    # Independent 行
    for prop, prop_name in [('rho', '$\\rho$ (density)'),
                             ('kappa', '$\\kappa$ (suscept.)')]:
        iv = indep_m.get(prop, {})
        mse_ours = fmt_val(iv.get('MSE')) if iv else 'TBD'
        corr_ours = fmt_val(iv.get('Correlation')) if iv else 'TBD'
        mse_paper = get_paper_val(comparison, 'Independent', prop, 'MSE')
        corr_paper = get_paper_val(comparison, 'Independent', prop, 'Correlation')
        lines.append(
            r" & %s & %s & %s & %s & %s \\" %
            (prop_name, mse_ours, mse_paper, corr_ours, corr_paper)
        )

    # Structural Accuracy 行
    sa_ours = fmt_val(struct_acc, '.4f') if struct_acc is not None else 'TBD'
    sa_entry = (comparison or {}).get('Structural_Accuracy', {})
    sa_paper = sa_entry.get('paper', 'TBD')
    if isinstance(sa_paper, float):
        sa_paper = f'{sa_paper:.4f}'
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{2}{l}{Structural Accuracy (\%)} "
        r"& %.2f & %s & — & — \\" % (
            (float(struct_acc) * 100 if struct_acc is not None else 0),
            sa_paper
        )
    )

    lines.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
    return '\n'.join(lines)


# ===========================================================================
# Figure 5: Prediction vs Observed Scatter Plot
# ===========================================================================

def plot_prediction_vs_observed(inference_result, output_path):
    """
    绘制预测值 vs 真实值的散点图。

    分别对密度和磁化率绘制:
      - 散点图 (带密度着色)
      - identity line (y=x)
      - R^2 和 Pearson r 标注

    Args:
        inference_result: run_inference() 返回的结果字典
        output_path: 输出 PDF 路径
    """
    preds = inference_result['predictions']
    targets = inference_result['targets']

    rho_pred = preds['rho_final'].flatten().numpy()
    kappa_pred = preds['kappa_final'].flatten().numpy()
    rho_true = targets['rho'].flatten().numpy()
    kappa_true = targets['kappa'].flatten().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- 左图: Density ----
    ax = axes[0]
    # 2D histogram 作为散点图的替代 (大数据量时更清晰)
    h = ax.hist2d(rho_true, rho_pred, bins=100, cmap='Blues',
                   norm=mcolors.LogNorm(), vmin=1)
    plt.colorbar(h[3], ax=ax, label='Count')

    # Identity line
    lim_max = max(rho_true.max(), rho_pred.max())
    ax.plot([0, lim_max], [0, lim_max], 'r--', linewidth=1.0, label='y=x', alpha=0.8)

    # Pearson r
    from scipy.stats import pearsonr
    r_rho, p_rho = pearsonr(rho_true, rho_pred)
    ss_res = np.sum((rho_true - rho_pred) ** 2)
    ss_tot = np.sum((rho_true - rho_true.mean()) ** 2)
    r2_rho = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    ax.set_xlabel('True Density $\\rho$')
    ax.set_ylabel('Predicted Density $\\hat{\\rho}$')
    ax.set_title('(a) Density Model', fontweight='bold')
    ax.text(0.05, 0.95, f'$r$ = {r_rho:.4f}\n$R^2$ = {r2_rho:.4f}',
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    ax.legend(loc='lower right', framealpha=0.8)

    # ---- 右图: Susceptibility ----
    ax = axes[1]
    h = ax.hist2d(kappa_true, kappa_pred, bins=100, cmap='Greens',
                   norm=mcolors.LogNorm(), vmin=1)
    plt.colorbar(h[3], ax=ax, label='Count')

    lim_max = max(kappa_true.max(), kappa_pred.max())
    ax.plot([0, lim_max], [0, lim_max], 'r--', linewidth=1.0, label='y=x', alpha=0.8)

    r_kappa, p_kappa = pearsonr(kappa_true, kappa_pred)
    ss_res = np.sum((kappa_true - kappa_pred) ** 2)
    ss_tot = np.sum((kappa_true - kappa_true.mean()) ** 2)
    r2_kappa = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    ax.set_xlabel('True Susceptibility $\\kappa$')
    ax.set_ylabel('Predicted Susceptibility $\\hat{\\kappa}$')
    ax.set_title('(b) Susceptibility Model', fontweight='bold')
    ax.text(0.05, 0.95, f'$r$ = {r_kappa:.4f}\n$R^2$ = {r2_kappa:.4f}',
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    ax.legend(loc='lower right', framealpha=0.8)

    plt.suptitle('Prediction vs Ground Truth', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()

    print(f"  [DONE] {output_path}")
    return output_path


# 导入 matplotlib.colors 用于 hist2d
from matplotlib import colors as mcolors


# ===========================================================================
# 主函数
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Result Figures')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--sample-index', type=int, default=0,
                        help='Test sample index for model display (default: 0)')
    args = parser.parse_args()

    # 加载 config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("  3D Gravity-Magnetic Joint Inversion — Result Figure Generator")
    print("=" * 70)

    # 目录设置
    result_dir = config['output']['result_dir']
    figures_dir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    ckpt_dir = config['output']['checkpoint_dir']

    seed = config['training']['seed']
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ===== 数据准备 =====
    from src.data.generate_synthetic import generate_dataset
    from src.data.dataset import JointInversionInMemoryDataset
    from src.utils import load_checkpoint

    data_dir = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']

    npz_exists = os.path.exists(os.path.join(data_dir, 'train_dataset.npz'))

    if not npz_exists:
        print("\n[Data] Generating synthetic dataset...")
        samples = generate_dataset(dataset_type=1, n_samples=60, seed=seed, verbose=False)
        full_dataset = JointInversionInMemoryDataset(samples)
        n_total = len(full_dataset)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        n_test = n_total - n_train - n_val

        from torch.utils.data import random_split as _rs
        _, _, test_ds = _rs(
            full_dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(seed)
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                                 num_workers=config['data']['num_workers'])
    else:
        from src.data.dataset import create_dataloaders
        dataloaders = create_dataloaders(
            data_dir=data_dir, batch_size=batch_size,
            num_workers=config['data']['num_workers'],
        )
        test_loader = dataloaders['test']

    print(f"[Data] Test set: {len(test_loader.dataset)} samples")

    # ===== 尝试加载模型 (可选 — 部分图表不需要模型) =====
    model = None
    inference_result = None
    ckpt_path = args.checkpoint or os.path.join(ckpt_dir, 'best_model.pt')

    if os.path.exists(ckpt_path):
        from src.model.joint_inversion_net import JointInversionNet
        model_config = config.get('model', {})
        model = JointInversionNet(
            in_channels=model_config.get('in_channels', 2),
            use_gradient_checkpointing=config['training'].get('gradient_checkpointing', False),
        ).to(device)
        info = load_checkpoint(ckpt_path, model)
        print(f"[Model] Loaded checkpoint: epoch={info['epoch']}")

        # 运行推理收集所有预测
        model.eval()
        all_preds = {
            'rho_final': [], 'kappa_final': [],
            'rho_pred': [], 'kappa_pred': [], 'structural_sim': [],
        }
        all_targets = {'rho': [], 'kappa': [], 'sim': []}

        with torch.no_grad():
            for inputs, targets_dict in test_loader:
                inputs = inputs.to(device)
                td = {k: v.to(device) for k, v in targets_dict.items()}
                outputs = model(inputs, return_all=True)
                all_preds['rho_final'].append(outputs['rho_final'].cpu())
                all_preds['kappa_final'].append(outputs['kappa_final'].cpu())
                all_preds['rho_pred'].append(outputs['rho_pred'].cpu())
                all_preds['kappa_pred'].append(outputs['kappa_pred'].cpu())
                all_preds['structural_sim'].append(outputs['structural_sim'].cpu())
                all_targets['rho'].append(td['rho'].cpu())
                all_targets['kappa'].append(td['kappa'].cpu())
                all_targets['sim'].append(td['sim'].cpu())

        inference_result = {
            'predictions': {k: torch.cat(v, dim=0) for k, v in all_preds.items()},
            'targets': {k: torch.cat(v, dim=0) for k, v in all_targets.items()},
        }
        print(f"[Inference] Collected predictions for {len(test_loader.dataset)} samples")
    else:
        print(f"[Model] No checkpoint found at {ckpt_path}")
        print("        Some figures will be generated with placeholder data.")

    # ===== 获取一个测试样本用于可视化 =====
    sample_input, sample_target = test_loader.dataset[args.sample_index]

    # ===== 生成所有图表 =====
    print("\n" + "-" * 50)
    print("Generating figures...")
    print("-" * 50)

    t_start = time.time()

    # Fig 1: Training curves
    history_path = os.path.join(result_dir, 'training_history.json')
    fig1_path = os.path.join(figures_dir, 'result_training_curves.pdf')
    plot_training_curves(history_path, fig1_path)

    # Fig 2: Test model display
    fig2_path = os.path.join(figures_dir, 'result_test_model.pdf')
    sample_pred_for_fig2 = None
    if inference_result is not None:
        # 取推理结果的第一个样本
        sample_pred_for_fig2 = {
            'rho_final': inference_result['predictions']['rho_final'][args.sample_index:args.sample_index+1],
            'kappa_final': inference_result['predictions']['kappa_final'][args.sample_index:args.sample_index+1],
        }
    plot_test_model(sample_input, sample_target, fig2_path,
                    sample_pred=sample_pred_for_fig2)

    # Fig 3: Comparison slices
    fig3_path = os.path.join(figures_dir, 'result_comparison_slices.pdf')
    if inference_result is not None:
        true_data = {
            'rho': inference_result['targets']['rho'][args.sample_index].numpy(),
            'kappa': inference_result['targets']['kappa'][args.sample_index].numpy(),
        }
        pred_joint = {
            'rho_final': inference_result['predictions']['rho_final'][args.sample_index:args.sample_index+1],
            'kappa_final': inference_result['predictions']['kappa_final'][args.sample_index:args.sample_index+1],
        }
        pred_indep = {
            'rho_pred': inference_result['predictions']['rho_pred'][args.sample_index:args.sample_index+1],
            'kappa_pred': inference_result['predictions']['kappa_pred'][args.sample_index:args.sample_index+1],
        }
        plot_comparison_slices(true_data, pred_joint, pred_indep, fig3_path)
    else:
        print(f"  [SKIP] {fig3_path} (requires trained model)")

    # Fig 4: LaTeX metrics table
    fig4_path = os.path.join(figures_dir, 'result_metrics_table.tex')
    metrics_json_path = os.path.join(result_dir, 'metrics.json')
    generate_latex_metrics_table(metrics_json_path, fig4_path)

    # Fig 5: Prediction vs observed
    fig5_path = os.path.join(figures_dir, 'result_prediction_vs_observed.pdf')
    if inference_result is not None:
        plot_prediction_vs_observed(inference_result, fig5_path)
    else:
        print(f"  [SKIP] {fig5_path} (requires trained model)")

    elapsed = time.time() - t_start

    # ===== 完成 =====
    print("\n" + "=" * 70)
    print("  All figures generated!")
    print("=" * 70)
    print(f"  Output directory: {figures_dir}/")
    print(f"  Files:")
    print(f"    1. result_training_curves.pdf")
    print(f"    2. result_test_model.pdf")
    print(f"    3. result_comparison_slices.pdf" + (" (generated)" if inference_result else " (skipped)"))
    print(f"    4. result_metrics_table.tex")
    print(f"    5. result_prediction_vs_observed.pdf" + (" (generated)" if inference_result else " (skipped)"))
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
