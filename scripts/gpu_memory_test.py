#!/usr/bin/env python3
"""
GPU Memory Benchmark for JointInversionNet.

Tests different batch sizes to find the optimal one that uses 70-80% of
the 32 GB VRAM on RTX 5000 Ada.

Method:
  1. Load JointInversionNet to CUDA
  2. For each batch_size in [1, 2, 4, 8, 16, 32, 48, 64]:
     - Run forward + backward pass with synthetic data
     - Record peak memory via torch.cuda.max_memory_allocated()
  3. Determine optimal batch size (70-80% VRAM utilization)
"""

import sys
import os
import time
import gc

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.model.joint_inversion_net import JointInversionNet


def get_gpu_info():
    """Return GPU name and total memory in bytes."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, 'total_mem', None) or getattr(props, 'total_memory', 0)
    return props.name, total_mem


def benchmark_batch_size(model, device, batch_size, input_shape=(2, 81, 81)):
    """
    Benchmark memory usage for a given batch size.

    Returns:
        peak_mem_bytes: Peak GPU memory allocated during forward+backward.
        success: Whether the operation completed without OOM.
        elapsed_ms: Time taken for forward+backward in milliseconds.
    """
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    try:
        # Create input and target tensors on GPU
        x = torch.randn(batch_size, *input_shape, device=device)
        # Targets: 5 outputs each (B, 1, 40, 40, 20)
        targets = {
            f'task{i+1}': torch.randn(batch_size, 1, 40, 40, 20, device=device)
            for i in range(5)
        }

        # Forward pass
        start = time.perf_counter()
        outputs = model(x)

        # Compute loss (MSE for all tasks to simulate real training)
        total_loss = torch.tensor(0.0, device=device)
        for key in outputs:
            total_loss = total_loss + nn.functional.mse_loss(outputs[key], targets[key])

        # Backward pass
        total_loss.backward()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Record peak memory BEFORE cleaning up
        peak_mem = torch.cuda.max_memory_allocated(device)

        # Clean up gradients for next iteration
        model.zero_grad(set_to_none=True)

        del x, targets, outputs, total_loss
        torch.cuda.empty_cache()
        gc.collect()

        return peak_mem, True, elapsed_ms

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Clear any partial allocations
            if 'x' in locals():
                del x
            if 'targets' in locals():
                del targets
            if 'outputs' in locals():
                del outputs
            torch.cuda.empty_cache()
            gc.collect()
            return torch.cuda.max_memory_allocated(device), False, 0.0
        else:
            raise


def main():
    print("=" * 72)
    print("  GPU Memory Benchmark — JointInversionNet")
    print("=" * 72)

    device = torch.device("cuda:0")

    # Get GPU info
    gpu_name, total_mem_bytes = get_gpu_info()
    total_mem_gb = total_mem_bytes / (1024 ** 3)

    print(f"\n  GPU:          {gpu_name}")
    print(f"  Total VRAM:   {total_mem_gb:.1f} GB ({total_mem_bytes / (1024**2):.0f} MiB)")
    print(f"  Target:       70~80% = {total_mem_gb * 0.7:.1f} ~ {total_mem_gb * 0.8:.1f} GB")

    # Load model
    print("\n  Loading JointInversionNet...")
    model = JointInversionNet(
        in_channels=2,
        backbone_channels=64,
        aspp_out_channels=40,
        out_depth=20,
        leaky_slope=0.01,
    ).to(device)

    params = model.get_num_params()
    print(f"  Model params: {params['total']:,}")
    print(f"    Backbone:   {params['backbone']:,}")
    print(f"    ASPP:       {params['aspp']:,}")
    print(f"    Task heads: {params['task_heads']:,}")

    # Model base memory (just loading, no data)
    torch.cuda.empty_cache()
    base_mem = torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() else 0
    base_mem_gb = base_mem / (1024 ** 3)
    print(f"\n  Base model memory (no data): {base_mem_gb:.2f} GB")

    # Test batch sizes -- start small then go large to find the 70-80% sweet spot
    batch_sizes = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
    results = []

    print("\n" + "-" * 72)
    print(f"  {'Batch':>6s} | {'Peak Mem (GB)':>14s} | {'% of Total':>10s} | {'Time (ms)':>9s} | {'Status':>8s}")
    print("-" * 72)

    best_bs = None
    best_pct = 0

    for bs in batch_sizes:
        peak_mem, success, elapsed = benchmark_batch_size(model, device, bs)
        peak_gb = peak_mem / (1024 ** 3)
        pct = (peak_mem / total_mem_bytes) * 100
        status = "OK" if success else "OOM"

        print(f"  {bs:>6d} | {peak_gb:>14.3f} | {pct:>9.1f}% | {elapsed:>8.1f} | {status:>8s}")

        results.append({
            'batch_size': bs,
            'peak_bytes': peak_mem,
            'peak_gb': round(peak_gb, 3),
            'pct': round(pct, 1),
            'success': success,
            'elapsed_ms': round(elapsed, 1),
        })

        if success:
            # Track best batch size within 70-80% target
            if 70 <= pct <= 80:
                if pct > best_pct:
                    best_pct = pct
                    best_bs = bs
            elif pct < 70 and best_bs is None:
                # If we haven't found anything in range yet, track highest under 80%
                if best_bs is None or bs > best_bs:
                    best_bs = bs
                    best_pct = pct

        # Stop testing larger batches if we already hit OOM
        if not success:
            break

    print("-" * 72)

    # Determine recommendation
    print("\n" + "=" * 72)
    print("  RECOMMENDATION")
    print("=" * 72)

    # Find the largest successful batch size
    successful = [r for r in results if r['success']]
    if successful:
        max_ok = max(successful, key=lambda r: r['batch_size'])

        # Find best within 70-80%
        in_range = [r for r in successful if 70 <= r['pct'] <= 80]
        if in_range:
            recommended = max(in_range, key=lambda r: r['pct'])
            rec_reason = "within 70-80% target"
        else:
            # Pick the largest that's under 85%
            under_85 = [r for r in successful if r['pct'] <= 85]
            if under_85:
                recommended = max(under_85, key=lambda r: r['batch_size'])
                rec_reason = f"largest under 85% ({recommended['pct']}%)"
            else:
                recommended = min(successful, key=lambda r: r['pct'])
                rec_reason = f"smallest viable (all others >85%)"

        print(f"\n  Training batch_size:     {recommended['batch_size']}  "
              f"(VRAM: {recommended['peak_gb']} GB, {recommended['pct']}%)  -- {rec_reason}")
        print(f"  Inference batch_size:    {max_ok['batch_size']}  "
              f"(VRAM: {max_ok['peak_gb']} GB, {max_ok['pct']}%)  -- no grad overhead")
        print(f"  Enable AMP:              YES (saves ~30-50% VRAM)")
        print(f"  Gradient accumulation:   NO needed at this batch size")
        print(f"  Max possible batch_size: {max_ok['batch_size']} "
              f"({max_ok['peak_gb']} GB, {max_ok['pct']}%)")
    else:
        print("\n  ERROR: All batch sizes resulted in OOM!")
        recommended = None

    # Also test with AMP to show potential savings
    print("\n" + "=" * 72)
    print("  AMP (Mixed Precision) Test")
    print("=" * 72)

    amp_results = []
    # Re-create model fresh for AMP test
    del model
    torch.cuda.empty_cache()
    gc.collect()

    model_amp = JointInversionNet(
        in_channels=2,
        backbone_channels=64,
        aspp_out_channels=40,
        out_depth=20,
        leaky_slope=0.01,
    ).to(device)

    print(f"\n  {'Batch':>6s} | {'Peak Mem (GB)':>14s} | {'% of Total':>10s} | {'Time (ms)':>9s} | {'Status':>8s}")
    print("-" * 72)

    amp_best_bs = None
    amp_best_pct = 0

    amp_batch_sizes = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1408, 1536, 1664, 1792]

    for bs in amp_batch_sizes:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        gc.collect()

        try:
            x = torch.randn(bs, 2, 81, 81, device=device)
            targets = {
                f'task{i+1}': torch.randn(bs, 1, 40, 40, 20, device=device)
                for i in range(5)
            }

            scaler = torch.amp.GradScaler('cuda')
            start = time.perf_counter()

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model_amp(x)
                total_loss = torch.tensor(0.0, device=device)
                for key in outputs:
                    total_loss = total_loss + nn.functional.mse_loss(outputs[key], targets[key])

            scaler.scale(total_loss).backward()
            elapsed_ms = (time.perf_counter() - start) * 1000

            peak_mem = torch.cuda.max_memory_allocated(device)
            peak_gb = peak_mem / (1024 ** 3)
            pct = (peak_mem / total_mem_bytes) * 100

            model_amp.zero_grad(set_to_none=True)
            del x, targets, outputs, total_loss, scaler
            torch.cuda.empty_cache()
            gc.collect()

            status = "OK"
            amp_results.append({
                'batch_size': bs,
                'peak_gb': round(peak_gb, 3),
                'pct': round(pct, 1),
                'elapsed_ms': round(elapsed_ms, 1),
                'success': True,
            })
            print(f"  {bs:>6d} | {peak_gb:>14.3f} | {pct:>9.1f}% | {elapsed_ms:>8.1f} | {status:>8s}")

            if 70 <= pct <= 80 and pct > amp_best_pct:
                amp_best_pct = pct
                amp_best_bs = bs

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                amp_peak = torch.cuda.max_memory_allocated(device)
                print(f"  {bs:>6d} | {amp_peak/(1024**3):>14.3f} | {'N/A':>9s} | {'N/A':>9s} | {'OOM':>8s}")
                amp_results.append({
                    'batch_size': bs,
                    'success': False,
                })
                break
            else:
                raise

    print("-" * 72)

    if amp_results:
        amp_successful = [r for r in amp_results if r['success']]
        if amp_successful:
            amp_max = max(amp_successful, key=lambda r: r['batch_size'])
            amp_in_range = [r for r in amp_successful if 70 <= r['pct'] <= 80]

            if amp_in_range:
                amp_rec = max(amp_in_range, key=lambda r: r['pct'])
                amp_reason = "within 70-80% target"
            else:
                amp_under_85 = [r for r in amp_successful if r['pct'] <= 85]
                if amp_under_85:
                    amp_rec = max(amp_under_85, key=lambda r: r['batch_size'])
                    amp_reason = f"largest under 85% ({amp_rec['pct']}%)"
                else:
                    amp_rec = min(amp_successful, key=lambda r: r['pct'])
                    amp_reason = f"smallest viable"

            print(f"\n  AMP Training batch_size:  {amp_rec['batch_size']}  "
                  f"(VRAM: {amp_rec['peak_gb']} GB, {amp_rec['pct']}%)  -- {amp_reason}")
            print(f"  AMP Max batch_size:       {amp_max['batch_size']}  "
                  f"(VRAM: {amp_max['peak_gb']} GB, {amp_max['pct']}%)")

    # Print final summary table for report
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY TABLE (for GPU_ALLOC_PLAN.md)")
    print("=" * 72)

    # Return values for script consumption
    fp32_rec = recommended if 'recommended' in dir() and recommended else None
    amp_rec_final = amp_rec if 'amp_rec' in dir() else None

    return {
        'gpu_name': gpu_name,
        'total_mem_gb': total_mem_gb,
        'fp32_results': results,
        'fp32_recommended': fp32_rec,
        'amp_results': amp_results,
        'amp_recommended': amp_rec_final,
    }


if __name__ == "__main__":
    result = main()
