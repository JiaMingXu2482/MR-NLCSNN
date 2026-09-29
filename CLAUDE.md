# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

NLCSNN (Nonlinear Continuous-time State-Space Neural Network) for CDC damper force prediction.
Based on Eischens et al. (2025) "State space neural network with nonlinear physics for mechanical system modeling" (Reliability Engineering and System Safety 259:110946).

The model learns a continuous-time state-space dynamics model with:
- A nonlinear feature library (NFL) providing physics-aware basis functions
- A parallel linear path (learnable NFL coefficients) bypassing the nonlinear MLP
- L1 sparsity on NFL linear-path weights
- RK4 integration (dt=0.001)
- Dissipation decay for bounded hidden states

Target: NRMSE < 3% on all validation cases.

## Current best result & deployment recipe (2026-06-22)

The physics-NFL valve model is the best deliverable and is **at the data ceiling**:

- **Best checkpoint:** `training_outputs/legacy_repro_180_epochs_20260620_105245/nlcsnn_legacy_best_all_val.pth` (63-dim physics NFL: fluid-dynamics laminar/turbulent/current-mod/inertia/gas/friction + relief-valve + state h).
- **Result (v≥0.05 envelope, 28 cases, RMS-NRMSE):** mean 17.7% / max 25.4% / MAE 34.4 N (beats paper SOTA MAE 44.9 N).
- **Deployment recipe** (reproduce via `validate_legacy_repro.py`): `CDC_MIN_VPEAK_MS=0.05` (drop quasi-static cases) + `CDC_BASELINE_CAL_MS=1000` (first-second zero-point calibration, removes the constant per-file offset). Input clamp (`input_clamp=5.0`, on by default) guards against draw-wire sensor glitches.
- **Data ceiling:** oracle de-bias gives 17.6%/25.1% — the model is essentially there. RMS 10%/20% is NOT reachable with this data; remaining error is draw-wire velocity jitter + small-signal SNR on RMS5mm cases. Lower error needs better sensors (magnetostrictive/LVDT) + consistent mounting, re-measured.
- **Known bad files (exclude):** `RLF_RMS10mm_2Hz_r2` (displacement sensor read 721 mm — impossible), `RMS10mm_1Hz` (not physically real).

## Environment

- **Python:** `C:\Users\user\.conda\envs\myenvPINN_test\python.exe` (Conda `myenvPINN_test`)
- **GPU:** RTX 4060 Ti 8 GB — CUDA required
- **Data root:** `D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Processed_Groups` (override via `train_config.py` or `CDC_PROCESSED_GROUPS` env var)

## Commands

```bash
# Training (from repo root)
& C:\Users\user\.conda\envs\myenvPINN_test\python.exe "d:/OneDrive - mail.scut.edu.cn/Python/MR-NLCSNN/AA02Train/TrainFinalVersion.py"

# Quick hyperparameter sweep (10-15 epochs, subset of data)
& C:\Users\user\.conda\envs\myenvPINN_test\python.exe "d:/OneDrive - mail.scut.edu.cn/Python/MR-NLCSNN/AA02Train/sweep.py" --sweep h_dim

# Validation
& C:\Users\user\.conda\envs\myenvPINN_test\python.exe "d:/OneDrive - mail.scut.edu.cn/Python/MR-NLCSNN/AA03Validate/Validate.py"
```

## Architecture

**Model** (`mr_nlcsnn/model.py`):
- 6 inputs: rod_length, velocity, accel, current, current_dot, temp
- `_get_nfl()` constructs ~188-dim nonlinear feature library:
  - Polynomial terms up to 3rd order (paper emphasizes low-order polynomials)
  - Directional dynamics (velocity-gated terms)
  - Current-velocity interactions, cross-coupling, hysteresis terms
- `input_proj` → 2× `ResidualMLPBlock` (LayerNorm + FF with GELU, dropout 0.05)
- Parallel linear path: `nn_x_linear` (NFL→h_dim, bias=False) + `nn_y_linear` (NFL→1, bias=False)
  - These learnable weights ARE the NFL coefficients (L1 penalty for sparsity)
- Nonlinear path: `nn_x_nonlinear` + `nn_y_nonlinear` for residual dynamics
- `state_derivative = dh_linear + dh_nonlinear - decay`
- `predict_force = f_linear + f_nonlinear` (tanh-bounded output)
- `rk4_step()` integrates state with fixed dt=0.001

**Training** (`AA02Train/TrainFinalVersion.py`):
- Training from t=0 with h=0 (paper Algorithm 1): always uses first `seq_len` samples of each file
- `ParallelDataManager`: stratified train/local-val split, file preloading to GPU, per-category sampling weights
- Train categories: all 4 (`1_Steady_Harmonic`, `2_Steady_Random`, `4_Random_LF_Current`, `5_Step_Current_Triangle`)
- HF validation: `3_Random_HF_Current`
- Loss: MSE (paper's RMS) + optional slope/curvature regularization + L1 NFL sparsity
- Mixed precision, AdamW + CosineAnnealingLR, gradient clipping 1.0, EMA

## Training configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `CDC_EPOCHS` | 400 | Total epochs |
| `CDC_BATCH_SIZE` | 32 | Sequences per step |
| `CDC_SEQ_LEN` | 512 | Timesteps per sequence (from t=0) |
| `CDC_STEPS_PER_EPOCH` | 48 | Gradient updates per epoch |
| `CDC_WARMUP` | 0 | Timesteps to skip (0 = train full sequence) |
| `CDC_HIDDEN` | 128 | Trunk width |
| `CDC_STATE_DIM` | 8 | Latent state dimension |
| `CDC_LR` | 8e-4 | Peak learning rate |
| `CDC_TARGET_NRMSE` | 3.0 | Early-exit threshold (%) |
| `CDC_SLOPE_WEIGHT` | 0.03 | Force slope regularization |
| `CDC_CURV_WEIGHT` | 0.012 | Force curvature regularization |
| `CDC_NFL_L1_WEIGHT` | 1e-6 | L1 sparsity on NFL linear-path weights |
| `CDC_USE_EMA` | 1 | Enable exponential moving average |
| `CDC_COMPILE` | 1 | Enable torch.compile |
| `CDC_VAL_RATIO` | 0.07 | Fraction for local validation split |
| `CDC_LOSS` | mse | "mse" or "smooth_l1" |
| `CDC_TRAIN_CATS` | all 4 | Comma-separated training categories |
| `CDC_RESUME_CHECKPOINT` | (empty) | Path to checkpoint .pth to resume |

## Data pipeline

1. `AA01DataProcess/AA1_DataCleaner.py` → `AA2_DataStats.py` → `AA3_DatasetSplitter.py`
2. Training reads per-category CSV files from `Processed_Groups/` subfolders
3. Validation (`AA03Validate/Validate.py`) loads a `.pth` checkpoint and runs open-loop simulation

## Training policy

- 200 epochs is not a hard limit — increase budget if validation error is still improving
- Plot sparingly (not every epoch); monitoring plots show 3 subplots per figure (5s slices)
- If training is too slow, stop and improve GPU utilization first
- If it becomes clear the target cannot be reached, stop and improve architecture/hyperparameters
- If all 3 representative cases reach below target, test all validation cases before declaring success
- Run sweeps (`sweep.py`) to understand hyperparameter influence before committing to long runs
