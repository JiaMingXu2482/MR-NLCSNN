# CDC_FWD_V22_E0 validation plots — best 3 s window per condition

Model: CDC_FWD_V22_E0, frozen release v1.0.0 (CDC_Model_Release, main, commit 7dadd76),
best checkpoint ep127, SHA dbfb844f…351b4, 2073 parameters, seed 7.
Data: Parsed_v19_F42_I42, validation set only (16 files), final_test not opened.
Method: model run full file (GRU state reset per file, 1 kHz), sliding 3 s window,
step 250 ms, long files start after the first 1 s (GRU warm-up guard);
per condition the window with the lowest NRMSE is plotted.
NRMSE = 100*sqrt(sum((pred-meas)^2)/sum(meas^2)), force in N.
Upper panel: measured (black) vs predicted (blue). Lower panel: measured current (A).

| Condition | File | Window in file | Window NRMSE | File guarded NRMSE |
|---|---|---:|---:|---:|
| SH | SH_10mm_6.62Hz_0.4A_r1 | 0.00–2.61 s (whole file, <3 s) | 3.25% | 2.92% |
| SR | SR_RMS10mm_1.66Hz_0A | 13.50–16.50 s | 25.52% | 34.98% |
| RHF | RHF_RMS10mm_4Hz | 1.00–4.00 s | 13.50% | 15.21% |
| RLF | RLF_RMS15mm_3Hz | 5.25–8.25 s | 10.84% | 16.47% |
| ST | ST_10mm_Vel0.13_step5_0_1.6A_r1 | 2.50–5.50 s | 7.87% | 11.59% |

Note: SH validation files are all shorter than 3 s, so the SH figure shows the
whole best file (2.61 s), not a 3 s window. Best-window figures are not a
replacement for whole-file metrics; SR remains the weakest condition.
