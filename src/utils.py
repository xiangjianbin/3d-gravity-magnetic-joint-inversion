"""
Utility functions for the gravity-magnetic joint inversion training pipeline.

Provides:
  - Random seed setting (for reproducibility)
  - Checkpoint save/load
  - Logging setup
  - Configuration loading from YAML
"""

import os
import sys
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (may impact performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, filepath: str, is_best: bool = False) -> None:
    """Save a training checkpoint to disk.

    Args:
        state:     Dict containing model state_dict, optimizer state_dict,
                   epoch, best_val_loss, etc.
        filepath:  Path to save the checkpoint file.
        is_best:   If True, also save as 'best_model.pth' in the same dir.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(os.path.dirname(filepath), 'best_model.pth')
        torch.save(state, best_path)


def load_checkpoint(filepath: str, model: torch.nn.Module,
                    optimizer=None,
                    device: str = 'cpu') -> dict:
    """Load a training checkpoint from disk.

    Args:
        filepath: Path to the checkpoint file.
        model:    Model to load weights into.
        optimizer: Optional optimizer to load state into.
        device:   Device to map tensors to.

    Returns:
        The loaded checkpoint dict (with epoch, best_val_loss, etc.).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str = 'train', log_file=None,
                 level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with optional file output.

    Args:
        name:     Logger name.
        log_file: Optional path to log file. If None, only console output.
        level:    Logging level.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Avoid duplicate handlers

    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _coerce_types(cfg):
    """Recursively coerce string values that look like numbers/bools to native types."""
    if isinstance(cfg, dict):
        return {k: _coerce_types(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [_coerce_types(v) for v in cfg]
    elif isinstance(cfg, str):
        # Try int, then float, then bool
        try:
            return int(cfg)
        except ValueError:
            pass
        try:
            return float(cfg)
        except ValueError:
            pass
        if cfg.lower() == 'true':
            return True
        if cfg.lower() == 'false':
            return False
        if cfg.lower() in ('null', 'none', '~'):
            return None
        return cfg
    return cfg


def load_config(config_path: str) -> dict:
    """Load YAML configuration file with automatic type coercion.

    Args:
        config_path: Path to .yaml file.

    Returns:
        Parsed configuration dict with proper types.
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return _coerce_types(cfg)


def save_json(data: dict, filepath: str) -> None:
    """Save a dictionary as JSON to disk.

    Args:
        data:     Dictionary to save.
        filepath: Output path.
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> dict:
    """Load JSON file into a dictionary.

    Args:
        filepath: Path to JSON file.

    Returns:
        Loaded dictionary.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def gpu_memory_summary() -> str:
    """Return a string summarizing current GPU memory usage."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    cached = torch.cuda.memory_reserved(0) / (1024 ** 3)
    return (
        f"GPU: {torch.cuda.get_device_name(0)}\n"
        f"  Total: {total:.1f} GB\n"
        f"  Allocated: {allocated:.2f} GB\n"
        f"  Cached: {cached:.2f} GB"
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Utils smoke test ===")
    set_seed(42)
    print("Seed set.")

    device = get_device()
    print(f"Device: {device}")
    print(gpu_memory_summary())

    logger = setup_logger('test')
    logger.info("Logger works.")
    print("\nAll utils tests PASSED.")
