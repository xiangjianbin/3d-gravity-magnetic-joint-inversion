"""
Training Script for Gravity-Magnetic Joint Inversion Network.

Implements the complete training loop:
  - Adam optimizer (beta1=0.9, beta2=0.999)
  - Learning rate: initial 1e-3, with StepLR or CosineAnnealingLR
  - Training loop with validation each epoch
  - Best model saving based on validation loss
  - Mixed precision (AMP) support
  - Gradient clipping
  - Early stopping

Usage:
    python src/train.py --config configs/full.yaml
    python src/train.py --config configs/smoke.yaml

Reproduced from:
  Fang et al., "Improved 3-D Joint Inversion of Gravity and Magnetic Data
  Based on Deep Learning With a Multitask Learning Strategy",
  IEEE TGRS, Vol. 63, 2025
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.joint_inversion_net import JointInversionNet
from src.model.loss_functions import get_criterion
from src.data.dataset import create_dataloaders
from src.utils import (
    set_seed, save_checkpoint, load_config,
    setup_logger, get_device, gpu_memory_summary, save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Joint Inversion Network')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory from config')
    return parser.parse_args()


def train_one_epoch(model, train_loader, criterion, optimizer, device,
                     scaler=None, grad_clip=1.0):
    """Run one training epoch.

    Returns:
        Dict of average per-task losses over the epoch.
    """
    model.train()
    task_loss_sums = {f'task{i}': 0.0 for i in range(1, 6)}
    task_loss_counts = {f'task{i}': 0 for i in range(1, 6)}
    total_loss_sum = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        inputs = batch['input'].to(device)
        targets = {
            'density': batch['density'].unsqueeze(1).to(device),
            'susceptibility': batch['susceptibility'].unsqueeze(1).to(device),
            'structural_sim': batch['structural_sim'].unsqueeze(1).to(device),
        }

        # Forward pass with optional mixed precision
        optimizer.zero_grad()

        if scaler is not None:
            with autocast(enabled=True):
                predictions = model(inputs)
                per_task, total_loss = criterion(predictions, targets, model=model)
            # Scale loss for AMP
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(inputs)
            per_task, total_loss = criterion(predictions, targets, model=model)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        # Accumulate losses
        for i in range(1, 6):
            key = f'task{i}'
            task_loss_sums[key] += per_task[key].item() if isinstance(per_task[key], torch.Tensor) else per_task[key]
            task_loss_counts[key] += 1
        total_loss_sum += total_loss.item()
        num_batches += 1

    # Compute averages
    avg_losses = {f'task{i}': task_loss_sums[f'task{i}'] / max(task_loss_counts[f'task{i}'], 1)
                   for i in range(1, 6)}
    avg_losses['total'] = total_loss_sum / max(num_batches, 1)

    return avg_losses


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Run one validation epoch.

    Returns:
        Dict of average per-task losses over the validation set.
    """
    model.eval()
    task_loss_sums = {f'task{i}': 0.0 for i in range(1, 6)}
    task_loss_counts = {f'task{i}': 0 for i in range(1, 6)}
    total_loss_sum = 0.0
    num_batches = 0

    for batch in val_loader:
        inputs = batch['input'].to(device)
        targets = {
            'density': batch['density'].unsqueeze(1).to(device),
            'susceptibility': batch['susceptibility'].unsqueeze(1).to(device),
            'structural_sim': batch['structural_sim'].unsqueeze(1).to(device),
        }

        predictions = model(inputs)
        per_task, total_loss = criterion(predictions, targets, model=None)

        for i in range(1, 6):
            key = f'task{i}'
            task_loss_sums[key] += per_task[key].item() if isinstance(per_task[key], torch.Tensor) else per_task[key]
            task_loss_counts[key] += 1
        total_loss_sum += total_loss.item()
        num_batches += 1

    avg_losses = {f'task{i}': task_loss_sums[f'task{i}'] / max(task_loss_counts[f'task{i}'], 1)
                   for i in range(1, 6)}
    avg_losses['total'] = total_loss_sum / max(num_batches, 1)

    return avg_losses


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # ---- Setup output directory ----
    output_dir = args.output_dir or cfg.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    # ---- Logging ----
    log_file = os.path.join(output_dir, 'training.log')
    logger = setup_logger('train', log_file=log_file)
    logger.info("=" * 60)
    logger.info("Gravity-Magnetic Joint Inversion Training")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output dir: {output_dir}")

    # ---- Reproducibility ----
    seed = cfg.get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # ---- Device ----
    device = get_device()
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(gpu_memory_summary())

    # ---- Data loaders ----
    data_dir = cfg.get('data_dir', 'data')
    batch_size = cfg.get('batch_size', 8)
    num_workers = cfg.get('num_workers', 4)

    logger.info(f"Loading dataset from: {data_dir}")
    logger.info(f"Batch size: {batch_size}, Workers: {num_workers}")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples:   {len(val_loader.dataset)}")
    logger.info(f"Test samples:  {len(test_loader.dataset)}")

    # ---- Model ----
    model = JointInversionNet(
        in_channels=2,
        backbone_channels=cfg.get('backbone_channels', 64),
        aspp_out_channels=cfg.get('aspp_out_channels', 40),
        leaky_slope=cfg.get('leaky_slope', 0.01),
    ).to(device)

    param_info = model.get_num_params()
    logger.info(f"Model parameters:")
    for k, v in param_info.items():
        logger.info(f"  {k}: {v:,}")

    # ---- Loss function ----
    criterion = get_criterion(
        task_weights=cfg.get('task_weights'),
        leaky_relu_lambda=cfg.get('leaky_relu_lambda', 1e-5),
    )

    # ---- Optimizer ----
    lr = cfg.get('lr', 1e-3)
    betas = cfg.get('betas', [0.9, 0.999])
    weight_decay = cfg.get('weight_decay', 1e-5)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=tuple(betas),
        weight_decay=weight_decay,
    )
    logger.info(f"Optimizer: Adam lr={lr}, betas={betas}, weight_decay={weight_decay}")

    # ---- LR scheduler ----
    scheduler_name = cfg.get('scheduler', 'cosine')
    epochs = cfg.get('epochs', 90)
    if scheduler_name == 'step':
        step_size = cfg.get('step_size', 30)
        gamma = cfg.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
        logger.info(f"Scheduler: StepLR(step={step_size}, gamma={gamma})")
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        logger.info(f"Scheduler: CosineAnnealingLR(T_max={epochs})")
    elif scheduler_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        logger.info("Scheduler: ReduceLROnPlateau(factor=0.5, patience=10)")
    else:
        scheduler = None
        logger.info("No LR scheduler.")

    # ---- AMP ----
    use_amp = cfg.get('use_amp', True) and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    logger.info(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    # ---- Gradient clipping ----
    grad_clip = cfg.get('grad_clip', 1.0)

    # ---- Resume from checkpoint ----
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, device=str(device))
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch-1}, best_val_loss={best_val_loss:.6f}")

    # ---- Training loop ----
    early_stopping_patience = cfg.get('early_stopping_patience', 15)
    epochs_no_improve = 0
    history = {'train': [], 'val': []}

    logger.info(f"\nStarting training for {epochs} epochs...")
    logger.info("-" * 60)

    total_start_time = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Train
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, grad_clip=grad_clip,
        )

        # Validate
        val_losses = validate(model, val_loader, criterion, device)

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_losses['total'])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        # Log
        history['train'].append(train_losses)
        history['val'].append(val_losses)

        task_names = ['Task1(IndGrav)', 'Task2(IndMag)', 'Task3(StructSim)',
                      'Task4(JointGrav)', 'Task5(JointMag)']

        log_msg = (f"Epoch [{epoch+1}/{epochs}] | "
                   f"Time: {epoch_time:.1f}s | LR: {current_lr:.2e}\n"
                   f"  Train Loss: Total={train_losses['total']:.6f}" +
                   "".join([f", T{i+1}={train_losses[f'task{i+1}']:.4f}"
                             for i in range(5)]) + "\n" +
                   f"  Val   Loss: Total={val_losses['total']:.6f}" +
                   "".join([f", T{i+1}={val_losses[f'task{i+1}']:.4f}"
                             for i in range(5)]))

        logger.info(log_msg)

        # Save best model based on total validation loss
        is_best = val_losses['total'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['total']
            epochs_no_improve = 0
            logger.info(f"  ** New best val loss: {best_val_loss:.6f} **")
        else:
            epochs_no_improve += 1

        # Save checkpoint every N epochs or when best
        save_every = cfg.get('save_every', 10)
        if is_best or (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(output_dir, 'checkpoints',
                                     f'checkpoint_epoch{epoch+1:03d}.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': cfg,
            }, ckpt_path, is_best=is_best)

        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            logger.info(f"\nEarly stopping triggered after {epochs_no_improve} epochs "
                       f"without improvement.")
            break

    total_time = time.time() - total_start_time
    logger.info(f"\nTraining completed in {total_time/3600:.2f} hours ({total_time:.0f}s)")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

    # ---- Save training history ----
    history_path = os.path.join(output_dir, 'training_history.json')
    save_json(history, history_path)
    logger.info(f"Training history saved to {history_path}")

    # ---- Final evaluation on test set ----
    logger.info("\nRunning final test evaluation...")
    test_losses = validate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: Total={test_losses['total']:.6f}" +
                "".join([f", T{i+1}={test_losses[f'task{i+1}']:.4f}" for i in range(5)]))

    # Save test results
    test_results_path = os.path.join(output_dir, 'test_results.json')
    save_json({'test_losses': test_losses, 'best_val_loss': best_val_loss},
              test_results_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
