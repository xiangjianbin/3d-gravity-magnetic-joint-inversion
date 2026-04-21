# Smoke Test Report -- Training Pipeline (Phase 5a)

**Date**: 2026-04-21
**Status**: ALL PASS

## 1. Test Summary

| Test Suite | File | Tests | Passed | Failed | Status |
|-----------|------|-------|--------|--------|--------|
| Backbone UNet3D | `tests/test_backbone.py` | 6 | 6 | 0 | PASS |
| Task Heads + Full Net + Loss | `tests/test_heads.py` | 13 | 13 | 0 | PASS |
| **Total** | | **19** | **19** | **0** | **PASS** |

## 2. Module Standalone Smoke Tests

| Module | File | Status |
|--------|------|--------|
| TaskHeads (5 heads) | `src/model/task_heads.py` | PASS |
| Loss Functions (MSE+BCE+Reg) | `src/model/loss_functions.py` | PASS |
| JointInversionNet (end-to-end) | `src/model/joint_inversion_net.py` | PASS |
| Evaluate Metrics (6 metrics) | `src/evaluate.py` | PASS |
| Utils (seed, GPU, logger) | `src/utils.py` | PASS |

## 3. Verified Dimensions

```
Input:   (B, 2, 40, 40, 20)  [gravity_obs + magnetic_obs]
           |
    3D U-Net Backbone        -> (B, 64, 40, 40, 20)
           |
    ASPP (rates=6,12,18,24)  -> (B, 40, 40, 40, 20)
           |
    5 x Task Heads          -> each (B, 1, 40, 40, 20)

Task 1: Independent Gravity Density      (B, 1, 40, 40, 20)  MSE
Task 2: Independent Magnetic Suscept.    (B, 1, 40, 40, 20)  MSE
Task 3: Structural Similarity             (B, 1, 40, 40, 20)  BCE+Sigmoid [0,1]
Task 4: Joint Gravity Density            (B, 1, 40, 40, 20)  MSE
Task 5: Joint Magnetic Susceptibility     (B, 1, 40, 40, 20)  MSE
```

## 4. Model Parameter Count

| Component | Parameters |
|-----------|-----------|
| Backbone (UNet3D) | 23,344,000 |
| ASPP | 194,400 |
| Task Heads (x5) | 623,205 |
| **Total** | **24,161,605** (~24.2M) |

## 5. Key Implementation Details

### 5.1 Network Architecture
- **Backbone**: 4-level 3D U-Net with skip connections (64->128->256->512 channels)
- **ASPP**: 4 dilated conv branches (r=6,12,18,24) + global avg pool, fused to 40ch
- **Task Heads**: Each head = Conv3d(40->64)->BN->LeakyReLU(0.01)->Conv3d(64->32)->BN->LeakyReLU->Conv3d(32->1)
- **Task 3** has Sigmoid activation for BCE loss (output in [0,1])

### 5.2 Loss Functions
- Tasks 1,2,4,5: MSELoss (regression)
- Task 3: Binary Cross Entropy (classification with sigmoid output)
- Optional L2 weight penalty (Leaky-ReLU regularizer): lambda=1e-5
- All tasks equally weighted (1.0)

### 5.3 Training Configuration
- Optimizer: Adam (lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)
- LR Scheduler: CosineAnnealingLR (T_max=90 epochs, eta_min=1e-6)
- Mixed Precision: AMP enabled (GradScaler)
- Gradient Clipping: max_norm=1.0
- Early Stopping: patience=15 epochs

### 5.4 Evaluation Metrics
All 6 metrics implemented and tested:
- IoU (threshold=0.5), MSE, MAE, R^2, SSIM (skimage, adaptive window size), PSNR

## 6. Files Produced

| File | Purpose |
|------|---------|
| `src/model/joint_inversion_net.py` | Main network (backbone + ASPP + 5 heads) |
| `src/model/task_heads.py` | 5 task-specific CNN heads |
| `src/model/loss_functions.py` | MSE/BCE losses + regularization |
| `src/train.py` | Full training script (CLI) |
| `src/evaluate.py` | 6 evaluation metric functions |
| `src/utils.py` | Seed, checkpoint, logging, GPU utilities |
| `configs/smoke.yaml` | Smoke test config (3 epochs, batch=2) |
| `configs/full.yaml` | Full training config (90 epochs, batch=16) |
| `notebooks/train.ipynb` | Jupyter notebook (11 cells) |
| `tests/test_backbone.py` | Backbone unit tests (6 tests) |
| `tests/test_heads.py` | Heads/net/loss tests (13 tests) |
| `requirements.txt` | Python dependencies |

## 7. Compatibility Notes
- Python 3.8 compatible (no PEP 604 union type syntax)
- PyTorch >= 2.0 required
- scikit-image >= 0.21 for SSIM computation
- RTX 5000 Ada 32GB detected and confirmed working

## 8. Next Steps
- Phase 5b: Run full training with `configs/full.yaml`
- Phase 5c: Fix any issues found during full training
- Phase 6: Result analysis and visualization
