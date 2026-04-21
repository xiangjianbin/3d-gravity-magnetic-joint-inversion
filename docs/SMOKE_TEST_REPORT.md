# Smoke Test Report

**Date**: 2026-04-21
**Tester**: Agent-MLTestEngineer
**Verdict**: **PASS**

---

## 1. Files Created

| File | Status |
|------|--------|
| `src/utils.py` | Created |
| `configs/smoke.yaml` | Created |
| `configs/full.yaml` | Created |
| `src/train.py` | Created |
| `src/evaluate.py` | Created |

## 2. Bug Fix Applied

**Issue**: `MultiTaskLoss.forward()` raised `ValueError: Target size (B,D,H,W) must be the same as input size (B,1,D,H,W)` for BCEWithLogitsLoss.

**Root cause**: Dataset provides targets as `(B, D, H, W)` (no channel dim), but model outputs are `(B, 1, D, H, W)`. MSE loss tolerates broadcasting but BCE does not.

**Fix**: Added `_align_target()` helper in `src/model/loss_functions.py` that auto-unsqueezes target tensors from `(B,D,H,W)` to `(B,1,D,H,W)` when dimension mismatch is detected. Applied to all 5 task losses.

**File modified**: `src/model/loss_functions.py`

## 3. Smoke Test Results

### Step 1: Import Test
```
Result: PASSED
Details: All modules imported successfully:
  - src.model.joint_inversion_net.JointInversionNet
  - src.model.loss_functions.MultiTaskLoss
  - src.utils.set_seed, count_parameters
```

### Step 2: Forward Pass Test
```
Result: PASSED
Input:  torch.randn(2, 2, 40, 40, 20)
Output keys: ['rho_pred', 'kappa_pred', 'structural_sim', 'rho_final', 'kappa_final']
Shapes verified:
  rho_pred:       (2, 1, 40, 40, 20)  OK
  kappa_pred:     (2, 1, 40, 40, 20)  OK
  structural_sim: (2, 1, 40, 40, 20)  OK
  rho_final:      (2, 1, 40, 40, 20)  OK
  kappa_final:    (2, 1, 40, 40, 20)  OK
NaN check: No NaN in any output tensor
```

### Step 3: Backward Pass Test
```
Result: PASSED
Total loss: 4.9995
Task losses:
  task1_gravity_mse:    1.4368
  task2_magnetic_mse:   1.5814
  task3_similarity_mse: 0.0893
  task4_gravity_bce:    0.9365
  task5_magnetic_bce:   0.9555
Mean gradient norm: 1.129609
NaN check: Loss is not NaN
Backward propagation successful
```

### Step 4: Optimizer Step Test
```
Result: PASSED
Optimizer: Adam(lr=1e-3)
AMP: GradScaler enabled (CPU fallback mode)
Gradient clipping: clip_grad_norm_(max_norm=1.0)
Initial loss: ~4.9995
Loss after 1 step: 4.5701 (decreased as expected)
```

## 4. Model Statistics

| Metric | Value |
|--------|-------|
| Total parameters | 104,201,917 (~104M) |
| Trainable parameters | 104,201,917 |
| Input shape | (B, 2, 40, 40, 20) |
| Output shapes | (B, 1, 40, 40, 20) x 5 tasks |

## 5. GPU Memory Test

**Status**: SKIPPED (CUDA not available on this machine)

Note: GPU memory test requires CUDA-capable device. Will be re-run when training on RTX 3060.

Expected VRAM usage estimate (batch_size=4): ~4-6 GB based on model size (104M params).

## 6. API Compatibility Notes

- Used `torch.amp.GradScaler('cuda')` and `torch.amp.autocast('cuda')` instead of deprecated `torch.cuda.amp.*` APIs to avoid FutureWarnings.
- Both old and new APIs are functionally equivalent; the new API was introduced in PyTorch >= 1.13.

## 7. Summary

All 4 smoke test steps passed successfully. The model can:
1. Import without errors
2. Forward-propagate with correct output shapes
3. Backward-propagate with valid gradients
4. Complete a full optimizer step with AMP + gradient clipping

The pipeline is ready for training execution (Phase 4).
