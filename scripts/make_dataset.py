#!/usr/bin/env python3
"""
Entry Point Script for Dataset Generation

Generates the complete 45,000-sample synthetic dataset for gravity-magnetic
joint inversion, following Fang et al. (2025) Table I specifications.

Usage:
    python scripts/make_dataset.py [--output-dir data/] [--seed 42] [--n-samples 45000] [--no-forward]
"""

import argparse
import os
import sys
import time
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset')
    parser.add_argument('--output-dir', type=str, default='data/',
                        help='Output directory for .npz files (default: data/)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--n-samples', type=int, default=45000,
                        help='Total number of samples (default: 45000)')
    parser.add_argument('--no-forward', action='store_true',
                        help='Skip forward modeling (use placeholder zeros)')
    parser.add_argument('--grav-noise', type=float, default=0.005,
                        help='Gravity noise level as fraction of max signal (default: 0.005)')
    parser.add_argument('--mag-noise', type=float, default=0.01,
                        help='Magnetic noise level as fraction of max signal (default: 0.01)')
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir

    print("=" * 65)
    print("  Gravity-Magnetic Joint Inversion: Dataset Generation")
    print("=" * 65)
    print(f"  Output directory : {os.path.abspath(output_dir)}")
    print(f"  Seed             : {args.seed}")
    print(f"  Target samples   : {args.n_samples}")
    print(f"  Forward modeling : {'OFF' if args.no_forward else 'ON'}")
    print()

    # Import and run the generation pipeline
    from src.data.generate_synthetic import (
        generate_full_dataset,
        NOISE_GRAVITY, NOISE_MAGNETIC,
    )

    # Override noise levels from CLI
    import src.data.generate_synthetic as gen_mod
    gen_mod.NOISE_GRAVITY = args.grav_noise
    gen_mod.NOISE_MAGNETIC = args.mag_noise

    t_start = time.time()
    manifest = generate_full_dataset(
        output_dir=output_dir,
        n_samples_total=args.n_samples,
        seed=args.seed,
        run_forward=not args.no_forward,
        verbose=True,
    )
    total_time = time.time() - t_start

    print(f"\n{'=' * 65}")
    print(f"  TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Output: {os.path.abspath(output_dir)}")
    print(f"{'=' * 65}")

    return manifest


if __name__ == '__main__':
    main()
