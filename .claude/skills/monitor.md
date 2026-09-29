# Monitor NLCSNN Training

Check training progress for a running or completed session.

## Check training status

```bash
ls -lt training_outputs/
```

## Read metrics

The main metrics file is `training_outputs/<session>/val_metrics.csv`. Key columns:
- `epoch`, `train_loss`
- `rep_mean_nrmse_pct` — mean of 3 representative cases
- `rep1_nrmse_pct` (harmonic), `rep2_nrmse_pct` (LF random), `rep3_nrmse_pct` (HF random)
- `epoch_seconds`, `torch_peak_gb`, `nvidia_used_gb`, `gpu_util_pct`

## View monitoring plots

Plots are in `training_outputs/<session>/val_plots/epoch_XXXX_monitor_cases.png`.
Each plot shows 3 subplots (harmonic, LF random, HF random) with measured vs predicted force.

## Check if training is still running

Look for the Python process:
```powershell
Get-Process python* | Where-Object {$_.CommandLine -like "*TrainFinalVersion*"}
```

Or check `training_outputs/adaptive_training.pid` if launched via `start_adaptive_training.py`.
