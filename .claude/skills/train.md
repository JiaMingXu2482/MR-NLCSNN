# Train NLCSNN

Start or configure an NLCSNN training session.

## Training command

Always use the Conda environment Python:

```powershell
& C:\Users\user\.conda\envs\myenvPINN_test\python.exe "d:/OneDrive - mail.scut.edu.cn/Python/MR-NLCSNN/AA02Train/TrainFinalVersion.py"
```

## Key environment variables

Before launching, set CDC_* env vars to override defaults. Common overrides:

- `CDC_EPOCHS` — total epochs (default 400)
- `CDC_BATCH_SIZE` — sequences per step (default 32)
- `CDC_SEQ_LEN` — timesteps per sequence (default 512)
- `CDC_HIDDEN` — trunk width (default 128)
- `CDC_STATE_DIM` — latent state dim (default 8)
- `CDC_LR` — peak learning rate (default 8e-4)
- `CDC_WARMUP` — warmup steps (default 0, full sequence from t=0)
- `CDC_TARGET_NRMSE` — early-exit threshold in % (default 3.0)
- `CDC_COMPILE` — set to 0 to disable torch.compile
- `CDC_RESUME_CHECKPOINT` — path to .pth to resume from
- `CDC_TRAIN_CATS` — comma-separated training categories (default all 4)
- `CDC_LOSS` — "mse" or "smooth_l1" (default mse)
- `CDC_NFL_L1_WEIGHT` — L1 sparsity on NFL linear path (default 1e-6)

## Memory constraints (RTX 4060 Ti 8 GB)

Current model (~200k params) uses <2 GB. Plenty of headroom.

## Launch method

For foreground (visible) training, run the python command directly.
For background (detached), use `start_adaptive_training.py` which logs to `training_outputs/adaptive_training.log`.

## After training

Check `training_outputs/<session>/val_metrics.csv` for per-epoch metrics and `nlcsnn_best.pth` for the best checkpoint.
