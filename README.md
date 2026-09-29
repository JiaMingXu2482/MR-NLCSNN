# MR-NLCSNN

Nonlinear continuous-time state-space neural network for CDC damper force prediction, based on a physics-aware nonlinear feature library (NFL), parallel linear/nonlinear paths, and RK4 state integration.

The implementation follows the modeling ideas in Eischens et al. (2025), *State space neural network with nonlinear physics for mechanical system modeling*.

## Repository structure

- `mr_nlcsnn/` — model implementations and nonlinear feature libraries
- `AA01DataProcess/` — CDC experiment-data cleaning, filtering, auditing, and splitting
- `AA02Train/` — training, checkpoint averaging, and hyperparameter sweeps
- `AA03Validate/` — validation, diagnostics, calibration, and reports
- `AA04References/` — reference material
- `training_outputs/` — trained checkpoints, logs, and monitoring artifacts
- `train_config.py` — shared training configuration
- `CLAUDE.md` — detailed architecture, workflow, and experiment notes

## Current best checkpoint

```text
training_outputs/legacy_repro_180_epochs_20260620_105245/
nlcsnn_legacy_best_all_val.pth
```

This checkpoint uses the 63-dimensional physics NFL with fluid-dynamics, current-modulation, inertia, gas/friction, relief-valve, and latent-state features.

## Typical commands

Run from the repository root in the configured Python/PyTorch environment.

```bash
# Train
python AA02Train/TrainFinalVersion.py

# Quick sweep
python AA02Train/sweep.py --sweep h_dim

# Validate the best legacy checkpoint
python AA03Validate/validate_legacy_repro.py \
  --checkpoint training_outputs/legacy_repro_180_epochs_20260620_105245/nlcsnn_legacy_best_all_val.pth \
  --data-root /path/to/processed/data
```

The experimental dataset is configured separately through `train_config.py`, command-line arguments, or the `CDC_PROCESSED_GROUPS` environment variable.

See [`CLAUDE.md`](CLAUDE.md) and [`training_work_log.md`](training_work_log.md) for detailed architecture, deployment settings, data-quality findings, and experiment history.
