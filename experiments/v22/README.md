# V22-AE Hybrid experiment archive

Snapshot: 2026-10-30. Imported from the completed local V22 experiment without modifying its training code, checkpoints or result files.

## Runs and results

All runs use **seed 7 only**. The five-model matrix uses 130 epochs. E0 and EGTA have separate 100-epoch extensions (epochs 131–230); the other models were not extended.

| Model | Parameters | 130-epoch validation pooled NRMSE | Extended to 230 epochs |
|---|---:|---:|---:|
| AGT | 3220 | 13.031% | — |
| E0 | 2073 | 10.595% | 10.293% |
| EG | 2187 | 12.449% | — |
| EGT | 2188 | 11.778% | — |
| EGTA | 2189 | 11.340% | 10.765% |

These are validation results, not final-test results. E0 is a forward-only baseline. The extension uses a recomputed learning-rate schedule and must not be described as an uninterrupted 230-epoch schedule or mixed with the 130-epoch matrix. Single-seed comparisons are exploratory, not evidence of cross-seed statistical significance.

## Contents

- `v22_models.py`, `train_v22.py`, `run_v22_all.py`, `test_v22.py`: five-model implementation, training and acceptance tests.
- `v20_data.py`: original data preprocessing used for these runs.
- `runs_seed7/`: five complete 130-epoch runs, best/last checkpoints, epoch metrics, grouped/per-file metrics, plots, resource reports, launcher logs and GPU telemetry.
- `runs_ext230/`: separate E0/EGTA extension artifacts.
- `smoke_seed7/`: initial 2-epoch smoke evidence.
- `fig_*/`, plotting/diagnostic scripts and root-level JSON/CSV files: later diagnostics.
- [V22_FINAL_REPORT.md](V22_FINAL_REPORT.md): original report, preserved verbatim.
- `specs/`: original V22 task and architecture/protocol documentation.
- `archive_manifest.json`: source paths and byte-level SHA256 for imported files. `.gitattributes` disables line-ending normalization in this archive to preserve these hashes.

## Reproduction context

The original environment used Python 3.11, PyTorch 2.11.0+cu128, NumPy, Matplotlib and an RTX 4060 Ti. Source scripts and configurations retain their original Windows paths, notably `D:/v22_ae_hybrid` and `D:/v20_data/Parsed_v19_F42_I42`; some diagnostic scripts therefore need the original layout or explicit path edits on another machine. Code is archived unchanged so the recorded training-source hashes remain meaningful.

From this directory, run `python -B test_v22.py` with that Python/PyTorch environment to check model invariants; it does not train or open final_test. Consult `run_v22_all.py --help` and `train_v22.py --help` before starting a new experiment, and always choose a fresh output directory.

The dataset is external and not duplicated here. Manifest SHA256: `0766763e7fb77880ffa04f2d93bcf629bfeb817e51c571188eee1ea83262e2ff`; train 259 / validation 16. Archived configs and summaries record `final_test_read=false`.

## Diagnostic provenance

Preserved reports describe different measurement passes. In particular, later `inverse_accuracy_fixed.json` records P99 values around 33–41 ms, unlike the earlier 4–5 ms values in the main report. They are retained with their originating scripts, not merged into one benchmark or used here to claim real-time acceptance. Python timings and CUDA per-sequence amortized timings are not ECU WCET. No training, scientific conclusion revision or final-test evaluation was performed during this Git archival task.
