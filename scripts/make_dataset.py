#!/usr/bin/env python3
"""
Entry Point Script for Dataset Generation (Multi-process Optimized)

Generates the complete 45,000-sample synthetic dataset using multiprocessing
for forward modeling acceleration. Uses all available CPU cores.

Usage:
    python scripts/make_dataset.py [--output-dir data/] [--seed 42] [--n-samples 45000] [--workers N]
"""

import argparse
import os
import sys
import time
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.generate_synthetic import (
    generate_all_datasets,
    GRID_NX, GRID_NY, GRID_NZ, N_OBS, OBS_HEIGHT,
    CELL_DX, CELL_DY, CELL_DZ,
)
from src.data.forward_gravity import forward_gravity, add_gravity_noise
from src.data.forward_magnetic import forward_magnetic, add_magnetic_noise
from src.data.dataset import save_dataset_samples


def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset (multiprocess)')
    parser.add_argument('--output-dir', type=str, default='data/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-forward', action='store_true')
    parser.add_argument('--n-samples', type=int, default=45000)
    parser.add_argument('--grav-noise', type=float, default=0.005)
    parser.add_argument('--mag-noise', type=float, default=0.01)
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of worker processes (0=auto, use all cores)')
    return parser.parse_args()


def _process_sample(args_tuple):
    """Process a single sample: run forward modeling + noise. For multiprocessing."""
    idx, sample, grav_noise, mag_noise = args_tuple
    density = sample['density']
    susceptibility = sample['susceptibility']

    gravity_clean = forward_gravity(density, obs_height=OBS_HEIGHT)
    magnetic_clean = forward_magnetic(susceptibility, obs_height=OBS_HEIGHT)

    sample['gravity'] = add_gravity_noise(gravity_clean, noise_level=grav_noise)
    sample['magnetic'] = add_magnetic_noise(magnetic_clean, noise_level=mag_noise)
    return idx, sample


def run_forward_modeling_multiprocess(samples, n_workers, grav_noise=0.005,
                                      mag_noise=0.01):
    """Run forward modeling using multiprocessing Pool."""
    from multiprocessing import Pool, cpu_count

    n_total = len(samples)
    if n_workers <= 0:
        n_workers = cpu_count()
    # Limit workers to avoid memory overload (each worker needs ~1-2GB for arrays)
    n_workers = min(n_workers, 12)  # cap at 12 to leave room for OS

    print(f"  Using {n_workers} worker processes...")
    start_time = time.time()

    # Prepare work items
    work_items = [(i, s, grav_noise, mag_noise) for i, s in enumerate(samples)]

    # Process in batches to limit memory usage
    batch_size = max(100, n_total // (n_workers * 4))  # ~4 batches per worker
    results = [None] * n_total

    processed = 0
    for batch_start in range(0, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch = work_items[batch_start:batch_end]

        with Pool(processes=n_workers) as pool:
            for idx, result in pool.imap_unordered(_process_sample, batch,
                                                     chunksize=max(1, len(batch)//(n_workers*2))):
                results[idx] = result
                processed += 1

                if processed % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    eta = (n_total - processed) / rate if rate > 0 else 0
                    print(f"  Forward modeling: {processed}/{n_total} "
                          f"({elapsed:.0f}s, ETA: {eta:.0f}s, {rate:.1f} samples/s)")

    # Reconstruct ordered list
    final_samples = [r for r in results if r is not None]

    total_time = time.time() - start_time
    print(f"  Forward modeling complete: {n_total} samples in {total_time:.1f}s "
          f"({total_time/n_total:.2f}s/sample, {n_workers} workers)")

    return final_samples


def run_forward_modeling_serial(samples, grav_noise=0.005, mag_noise=0.01):
    """Fallback serial implementation."""
    n_total = len(samples)
    start_time = time.time()

    for i, sample in enumerate(samples):
        density = sample['density']
        susceptibility = sample['susceptibility']
        sample['gravity'] = add_gravity_noise(
            forward_gravity(density, obs_height=OBS_HEIGHT), noise_level=grav_noise)
        sample['magnetic'] = add_magnetic_noise(
            forward_magnetic(susceptibility, obs_height=OBS_HEIGHT), noise_level=mag_noise)

        if (i+1) % 100 == 0 or i == n_total - 1:
            elapsed = time.time() - start_time
            rate = (i+1) / elapsed if elapsed > 0 else 0
            print(f"  Serial forward: {i+1}/{n_total} ({elapsed:.0f}s, {rate:.2f}/s)")

    total_time = time.time() - start_time
    print(f"  Serial complete: {total_time:.1f}s ({total_time/n_total:.2f}s/sample)")
    return samples


def main():
    args = parse_args()
    output_dir = args.output_dir

    print("=" * 60)
    print("  Gravity-Magnetic Joint Inversion: Dataset Generation")
    print("=" * 60)
    print(f"  Output: {output_dir}")
    print(f"  Seed: {args.seed}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Workers: {'auto' if args.workers==0 else args.workers}")
    print()

    # Step 1: Generate models
    t0 = time.time()
    print("Step 1: Generating geological models...")
    datasets = generate_all_datasets(seed=args.seed)
    all_samples = datasets['all']

    if len(all_samples) > args.n_samples:
        np.random.seed(args.seed + 999)
        indices = np.random.choice(len(all_samples), args.n_samples, replace=False)
        all_samples = [all_samples[i] for i in sorted(indices)]
        print(f"  Trimmed to {len(all_samples)} samples")

    print(f"  Model generation: {time.time()-t0:.1f}s")
    print()

    # Step 2: Forward modeling
    if not args.no_forward:
        print("Step 2: Running forward modeling (gravity + magnetic)...")

        # Try multiprocessing; fall back to serial on failure
        try:
            all_samples = run_forward_modeling_multiprocess(
                all_samples, args.workers, args.grav_noise, args.mag_noise)
        except Exception as e:
            print(f"  Multiprocessing failed ({e}), falling back to serial...")
            all_samples = run_forward_modeling_serial(
                all_samples, args.grav_noise, args.mag_noise)
        print()
    else:
        print("Step 2: SKIPPED (--no-forward)")
        for s in all_samples:
            s['gravity'] = np.zeros((N_OBS, N_OBS), dtype=np.float32)
            s['magnetic'] = np.zeros((N_OBS, N_OBS), dtype=np.float32)
        print()

    # Step 3: Save
    print("Step 3: Saving dataset with 7:2:1 split...")
    save_dataset_samples(all_samples, output_dir=output_dir)

    # Step 4: Manifest
    manifest = {
        'total_samples': len(all_samples),
        'grid': {'nx': GRID_NX, 'ny': GRID_NY, 'nz': GRID_NZ,
                 'dx': CELL_DX, 'dy': CELL_DY, 'dz': CELL_DZ},
        'observation': {'n_obs': N_OBS, 'height_m': OBS_HEIGHT},
        'split': {'train_ratio': 0.7, 'val_ratio': 0.2, 'test_ratio': 0.1},
        'noise': {'gravity_fraction': args.grav_noise, 'magnetic_fraction': args.mag_noise},
        'type_counts': {
            'type1': len(datasets.get('type1', [])),
            'type2': len(datasets.get('type2', [])),
            'type3': len(datasets.get('type3', [])),
            'type4': len(datasets.get('type4', [])),
            'type5': len(datasets.get('type5', [])),
            'type6': len(datasets.get('type6', [])),
            'trapezoids': len(datasets.get('trapezoids', [])),
        },
        'seed': args.seed,
        'generation_time_s': time.time() - t0,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'DATASET_MANIFEST.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  TOTAL TIME: {total:.1f}s ({total/60:.1f}min)")
    print(f"  Output: {os.path.abspath(output_dir)}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
