#!/usr/bin/env python3
"""
Dataset Visualization Script for 3D Gravity-Magnetic Joint Inversion Project.

Generates 5 publication-quality PDF figures:
  1. dataset_6types_samples.pdf   -- 6 types x 2 samples, 5 columns each
  2. dataset_distributions.pdf    -- Value histograms (rho/kappa/gravity/magnetic)
  3. dataset_split_pie.pdf        -- Train/Val/Test split pie chart
  4. dataset_structural_sim_stats.pdf -- Structural similarity statistics
  5. dataset_model_examples.pdf   -- Geological model examples (Fig.6/7 style)

Usage:
    python scripts/visualize_dataset.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from collections import Counter, defaultdict
import os
import sys

# ── Publication style settings ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.pad_inches': 0.05,
})

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FIG_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Colormaps ───────────────────────────────────────────────────────────────
CMAP_RHO = 'viridis'
CMAP_KAPPA = 'plasma'
CMAP_GRAVITY = 'seismic'
CMAP_MAGNETIC = 'RdBu_r'
CMAP_SIM = 'cividis'


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_npz(filepath):
    """Load .npz and return dict of sample data."""
    d = np.load(filepath)
    samples = {}
    keys = [k for k in d.files if k.startswith('sample_')]
    # Group by sample index
    sample_indices = sorted(set(k.split('_')[1] for k in keys))
    for idx in sample_indices:
        prefix = f'sample_{idx}_'
        samples[idx] = {
            'rho': d[prefix + 'rho'],
            'kappa': d[prefix + 'kappa'],
            'sim': d[prefix + 'sim'],
            'gravity': d[prefix + 'gravity'],
            'magnetic': d[prefix + 'magnetic'],
            'type': int(d[prefix + 'type']),
            'consistency': str(d.get(prefix + 'consistency', 'unknown')),
        }
    return samples


def load_all_splits():
    """Load train/val/test splits."""
    splits = {}
    for name in ['train', 'val', 'test']:
        path = os.path.join(DATA_DIR, f'{name}_dataset.npz')
        if os.path.exists(path):
            splits[name] = load_npz(path)
    return splits


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: 6 Types Samples
# ══════════════════════════════════════════════════════════════════════════════

def fig1_6types_samples(all_data):
    """6 types x 2 samples, 5 columns: rho(z=10), kappa(z=10), sim(z=10), gravity, magnetic."""
    print("Generating Figure 1: 6 types samples ...")

    # Collect samples by type from training set
    by_type = defaultdict(list)
    for idx, s in all_data['train'].items():
        by_type[s['type']].append(idx)

    z_slice = 10  # middle slice

    fig, axes = plt.subplots(6, 5, figsize=(18, 22))
    col_titles = ['Density $\\rho$ (z=10)', 'Susceptibility $\\kappa$ (z=10)',
                  'Structural Sim. S (z=10)', 'Gravity Anomaly $\\Delta g$', 'Magnetic Anomaly $\\Delta T$']
    col_cmaps = [CMAP_RHO, CMAP_KAPPA, CMAP_SIM, CMAP_GRAVITY, CMAP_MAGNETIC]

    type_consistency_map = {
        1: 'global', 2: 'global', 3: 'partial',
        4: 'partial', 5: 'inconsistent', 6: 'inconsistent'
    }

    for row, t in enumerate(range(1, 7)):
        indices = by_type[t][:2]
        cons_label = type_consistency_map.get(t, '?')
        axes[row, 0].set_ylabel(f'Type {t}: {cons_label}', fontsize=10, fontweight='bold')

        for col in range(5):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, pad=8)

            for si, idx in enumerate(indices):
                s = all_data['train'][idx]
                if col < 3:
                    data_keys = ['rho', 'kappa', 'sim']
                    img = s[data_keys[col]][:, :, z_slice].T
                else:
                    data_keys = ['', '', '', 'gravity', 'magnetic']
                    img = s[data_keys[col]]

                im = ax.imshow(img, cmap=col_cmaps[col], aspect='auto',
                               origin='lower')
                # Add thin border between the two samples
                if si == 0:
                    ax.axvline(x=img.shape[1] - 0.5, color='white', linewidth=1.5)

            ax.set_xticks([])
            ax.set_yticks([])

            # Colorbar on the rightmost column only
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()
    outpath = os.path.join(FIG_DIR, 'dataset_6types_samples.pdf')
    fig.savefig(outpath, format='pdf')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Distributions
# ══════════════════════════════════════════════════════════════════════════════

def fig2_distributions(all_data):
    """Histograms of rho, kappa, gravity, magnetic values across all splits."""
    print("Generating Figure 2: distributions ...")

    # Collect all values
    all_rho, all_kappa, all_gravity, all_magnetic = [], [], [], []
    for split_name, split_data in all_data.items():
        for idx, s in split_data.items():
            all_rho.append(s['rho'].ravel())
            all_kappa.append(s['kappa'].ravel())
            all_gravity.append(s['gravity'].ravel())
            all_magnetic.append(s['magnetic'].ravel())

    all_rho = np.concatenate(all_rho)
    all_kappa = np.concatenate(all_kappa)
    all_gravity = np.concatenate(all_gravity)
    all_magnetic = np.concatenate(all_magnetic)

    datasets = [
        ('Density $\\rho$', all_rho, CMAP_RHO),
        ('Susceptibility $\\kappa$', all_kappa, CMAP_KAPPA),
        ('Gravity Anomaly $\\Delta g$', all_gravity, CMAP_GRAVITY),
        ('Magnetic Anomaly $\\Delta T$', all_magnetic, CMAP_MAGNETIC),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, (title, data, cmap) in enumerate(datasets):
        ax = axes[i]
        mean_val = np.mean(data)
        std_val = np.std(data)

        n, bins, patches = ax.hist(data, bins=80, color='steelblue',
                                    edgecolor='white', linewidth=0.3, alpha=0.85)

        # Use gradient coloring based on colormap
        norm = Normalize(vmin=data.min(), vmax=data.max())
        for patch, left in zip(patches, bins[:-1]):
            patch.set_facecolor(plt.colormaps[cmap](norm(left)))

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean={mean_val:.4f}')
        ax.axvline(mean_val + std_val, color='darkorange', linestyle=':', linewidth=1.2, label=f'+1$\\sigma$={mean_val+std_val:.4f}')
        ax.axvline(mean_val - std_val, color='darkorange', linestyle=':', linewidth=1.2, label=f'-1$\\sigma$={mean_val-std_val:.4f}')

        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    outpath = os.path.join(FIG_DIR, 'dataset_distributions.pdf')
    fig.savefig(outpath, format='pdf')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Split Pie Chart
# ══════════════════════════════════════════════════════════════════════════════

def fig3_split_pie(all_data):
    """Train / Val / Test split pie chart."""
    print("Generating Figure 3: split pie chart ...")

    sizes = [len(all_data[split]) for split in ['train', 'val', 'test']]
    labels = [
        f'Train ({sizes[0]} samples, {100*sizes[0]/sum(sizes):.1f}%)',
        f'Validation ({sizes[1]} samples, {100*sizes[1]/sum(sizes):.1f}%)',
        f'Test ({sizes[2]} samples, {100*sizes[2]/sum(sizes):.1f}%)',
    ]
    colors = ['#4e79a7', '#f28e2b', '#e15759']
    explode = (0.02, 0.02, 0.04)

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct='', startangle=90, textprops={'fontsize': 11},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for t in texts:
        t.set_fontsize(11)

    ax.set_title('Dataset Split: Train / Validation / Test', fontsize=13, fontweight='bold', pad=15)

    # Add center text with total
    total = sum(sizes)
    ax.text(0, 0, f'Total\n{total}\nsamples', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#333333')

    plt.tight_layout()
    outpath = os.path.join(FIG_DIR, 'dataset_split_pie.pdf')
    fig.savefig(outpath, format='pdf')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Structural Similarity Statistics
# ══════════════════════════════════════════════════════════════════════════════

def fig4_structural_sim_stats(all_data):
    """Structural similarity statistics: global ratio, per-type ratio, example slices."""
    print("Generating Figure 4: structural similarity stats ...")

    # Collect stats
    s_counts = {'S=0': 0, 'S=1': 0}
    s_by_type = defaultdict(lambda: {'S=0': 0, 'S=1': 0})
    s1_example_idx = None
    s0_example_idx = None

    for split_name, split_data in all_data.items():
        for idx, s in split_data.items():
            has_sim = float(s['sim'].max()) > 0.5
            key = 'S=1' if has_sim else 'S=0'
            s_counts[key] += 1
            s_by_type[s['type']][key] += 1

            if has_sim and s1_example_idx is None:
                s1_example_idx = (split_name, idx)
            if not has_sim and s0_example_idx is None:
                s0_example_idx = (split_name, idx)

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

    # ── Subplot 1: Global S=0 vs S=1 bar ──
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(['S=0 (No Similarity)', 'S=1 (Has Similarity)'],
                   [s_counts['S=0'], s_counts['S=1']],
                   color=['#e15759', '#59a14f'], edgecolor='white', linewidth=1.5)
    total_s = s_counts['S=0'] + s_counts['S=1']
    for bar, val in zip(bars, [s_counts['S=0'], s_counts['S=1']]):
        height = bar.get_height()
        pct = 100 * val / total_s
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5 * max(s_counts.values()),
                 f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('(a) Global Structural Similarity Distribution', fontsize=11, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Subplot 2: Per-type S=1 proportion grouped bar ──
    ax2 = fig.add_subplot(gs[0, 1])
    types = sorted(s_by_type.keys())
    x = np.arange(len(types))
    width = 0.35
    s0_vals = [s_by_type[t]['S=0'] for t in types]
    s1_vals = [s_by_type[t]['S=1'] for t in types]

    bars_s0 = ax2.bar(x - width/2, s0_vals, width, label='S=0', color='#e15759', edgecolor='white')
    bars_s1 = ax2.bar(x + width/2, s1_vals, width, label='S=1', color='#59a14f', edgecolor='white')

    ax2.set_xlabel('Model Type')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('(b) Structural Similarity by Type', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Type {t}' for t in types])
    ax2.legend(loc='upper right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add count labels on bars
    for bars in [bars_s0, bars_s1]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., h,
                         str(int(h)), ha='center', va='bottom', fontsize=7)

    # ── Subplot 3: S=1 example multi-slice view ──
    ax3 = fig.add_subplot(gs[1, 0])
    if s1_example_idx:
        sp, si = s1_example_idx
        ex = all_data[sp][si]
        nz = ex['sim'].shape[2]
        ncols = min(7, nz)
        nrows = int(np.ceil(nz / ncols))

        # Show as a grid of slices
        for zi in range(nz):
            r = zi // ncols
            c = zi % ncols
            # We'll draw within this axis using inset-like approach
            pass

        # Instead, show a composite: pick representative slices
        slice_indices = list(range(0, nz, max(1, nz // 7)))[:7]
        n_show = len(slice_indices)

        for j, zi in enumerate(slice_indices):
            ax_inner = fig.add_axes([0.06 + j*0.095, 0.08, 0.085, 0.25])  # manual positioning
            img = ax_inner.imshow(ex['sim'][:, :, zi].T, cmap=CMAP_SIM, aspect='auto', origin='lower')
            ax_inner.set_title(f'z={zi}', fontsize=7)
            ax_inner.set_xticks([])
            ax_inner.set_yticks([])
        ax3.set_visible(False)
        fig.text(0.38, 0.36, '(c) S=1 Example: Structural Similarity Multi-Slice View',
                 fontsize=11, fontweight='bold', ha='center')
    else:
        ax3.text(0.5, 0.5, 'No S=1 sample found', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('(c) S=1 Example', fontsize=11, fontweight='bold')

    # ── Subplot 4: S=0 example multi-slice view ──
    ax4 = fig.add_subplot(gs[1, 1])
    if s0_example_idx:
        sp, si = s0_example_idx
        ex = all_data[sp][si]
        nz = ex['sim'].shape[2]

        slice_indices = list(range(0, nz, max(1, nz // 7)))[:7]
        n_show = len(slice_indices)

        for j, zi in enumerate(slice_indices):
            ax_inner = fig.add_axes([0.57 + j*0.095, 0.08, 0.085, 0.25])
            img = ax_inner.imshow(ex['sim'][:, :, zi].T, cmap=CMAP_SIM, aspect='auto', origin='lower')
            ax_inner.set_title(f'z={zi}', fontsize=7)
            ax_inner.set_xticks([])
            ax_inner.set_yticks([])
        ax4.set_visible(False)
        fig.text(0.88, 0.36, '(d) S=0 Example: No Structural Similarity',
                 fontsize=11, fontweight='bold', ha='center')
    else:
        ax4.text(0.5, 0.5, 'No S=0 sample found', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('(d) S=0 Example', fontsize=11, fontweight='bold')

    outpath = os.path.join(FIG_DIR, 'dataset_structural_sim_stats.pdf')
    fig.savefig(outpath, format='pdf')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: Model Examples (Paper Fig.6/Fig.7 Style)
# ══════════════════════════════════════════════════════════════════════════════

def fig5_model_examples(all_data):
    """
    Geological model examples in paper Fig.6/Fig.7 style.
    3 rows: cuboid (Type 1-2), tilting (Type 3-4), random_walk (Type 5-6)
    Each row: density(yz-slice) | susceptibility(yz-slice) | 3D isometric view
    """
    print("Generating Figure 5: model examples ...")

    # Map types to model categories
    # Based on consistency: types 1-2 are global (cuboid-like), 3-4 partial (tilting), 5-6 inconsistent (random_walk)
    category_types = {
        'cuboid': [1, 2],
        'tilting': [3, 4],
        'random_walk': [5, 6],
    }

    # Pick one representative sample per category
    representatives = {}
    for cat, t_list in category_types.items():
        for t in t_list:
            for idx, s in all_data['train'].items():
                if s['type'] == t:
                    representatives[cat] = (idx, s)
                    break
            if cat in representatives:
                break

    fig, axes = plt.subplots(3, 3, figsize=(16, 17))
    row_labels = ['Cuboid (Global Consistency)',
                  'Tilting (Partial Consistency)',
                  'Random Walk (Inconsistent)']

    y_slice = 20  # middle y-slice for yz view
    x_slice = 20  # middle x-slice for xz view

    for row, (cat, label) in enumerate(zip(['cuboid', 'tilting', 'random_walk'], row_labels)):
        idx, s = representatives[cat]
        axes[row, 0].set_ylabel(label, fontsize=11, fontweight='bold', rotation=0,
                                 labelpad=80, va='center')

        # Col 1: Density yz-slice (x=mid)
        ax = axes[row, 0]
        im0 = ax.imshow(s['rho'][x_slice, :, :].T, cmap=CMAP_RHO, aspect='auto', origin='lower')
        if row == 0:
            ax.set_title('Density $\\rho$ (yz-slice)', fontsize=11, pad=8)
        ax.set_xlabel('z')
        ax.set_ylabel('y')
        fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

        # Col 2: Susceptibility yz-slice
        ax = axes[row, 1]
        im1 = ax.imshow(s['kappa'][x_slice, :, :].T, cmap=CMAP_KAPPA, aspect='auto', origin='lower')
        if row == 0:
            ax.set_title('Susceptibility $\\kappa$ (yz-slice)', fontsize=11, pad=8)
        ax.set_xlabel('z')
        ax.set_ylabel('y')
        fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

        # Col 3: 3D isometric-style view using multiple z-slices arranged as depth stack
        ax = axes[row, 2]
        if row == 0:
            ax.set_title('3D Isometric View (density, multi-z)', fontsize=11, pad=8)

        # Create pseudo-3D effect by offsetting successive z-slices
        rho = s['rho']
        nz = rho.shape[2]
        n_display = min(nz, 15)
        step = max(1, nz // n_display)
        display_z = list(range(0, nz, step))[:n_display]

        offset_x = 0.35
        offset_y = 0.35
        alpha_base = 0.4
        alpha_step = (1.0 - alpha_base) / max(len(display_z) - 1, 1)

        for i, zi in enumerate(display_z):
            alpha = alpha_base + i * alpha_step
            ox = i * offset_x
            oy = i * offset_y
            extent = [ox, ox + rho.shape[0], oy, oy + rho.shape[1]]
            ax.imshow(rho[:, :, zi].T, cmap=CMAP_RHO, extent=extent,
                      origin='lower', alpha=alpha, aspect='auto')

        ax.set_xlabel('x (offset)')
        ax.set_ylabel('y (offset)')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_RHO,
                     norm=Normalize(vmin=rho.min(), vmax=rho.max())),
                     ax=ax, fraction=0.046, pad=0.04, label='$\\rho$').ax.tick_params(labelsize=7)

    plt.tight_layout()
    outpath = os.path.join(FIG_DIR, 'dataset_model_examples.pdf')
    fig.savefig(outpath, format='pdf')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" Dataset Visualization Generator")
    print("=" * 60)
    print(f" Project root: {PROJECT_ROOT}")
    print(f" Output dir:   {FIG_DIR}")
    print()

    # Load all data
    print("Loading datasets ...")
    all_data = load_all_splits()
    for name, sd in all_data.items():
        print(f"  {name}: {len(sd)} samples")
    print()

    # Generate figures
    fig1_6types_samples(all_data)
    fig2_distributions(all_data)
    fig3_split_pie(all_data)
    fig4_structural_sim_stats(all_data)
    fig5_model_examples(all_data)

    print()
    print("=" * 60)
    print(" All figures generated successfully!")
    print("=" * 60)

    # Verify outputs
    expected = [
        'dataset_6types_samples.pdf',
        'dataset_distributions.pdf',
        'dataset_split_pie.pdf',
        'dataset_structural_sim_stats.pdf',
        'dataset_model_examples.pdf',
    ]
    print("\nVerification:")
    all_ok = True
    for fname in expected:
        path = os.path.join(FIG_DIR, fname)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "OK" if exists and size > 1000 else "EMPTY or MISSING"
        if status != "OK":
            all_ok = False
        print(f"  [{status:>15}] {fname} ({size:,} bytes)")

    if all_ok:
        print("\nAll 5 PDF files generated and verified.")
        return 0
    else:
        print("\nWARNING: Some files may be empty or missing!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
