# CDC_FWD_V22_E0 step-delay plots — validation ST events

Model: CDC_FWD_V22_E0, frozen release v1.0.0 (CDC_Model_Release, main, commit 7dadd76),
best checkpoint ep127, SHA dbfb844f…351b4, 2073 parameters, seed 7.
Data: Parsed_v19_F42_I42, validation ST files only (ST_10mm_Vel0.13_step5_0_1.6A_r1/r2/r3),
final_test not opened.

## Definition (V22, must be read with the numbers)

Lag = current reaching 10% -> force reaching 90% (I10->F90),
velocity-drift corrected, |dF| >= 250 N.
It is NOT the force 10-90% rise time, and NOT t63.
Figures show raw measured/predicted force; the V22 report itself warns that
the visual order on raw curves may not match the drift-corrected table lag.
Use the table lag values, not a ruler reading on the raw plot.

## Overall delay accuracy (validation ST, 250 N table, 20 events)

- Measured lag: median 22 ms, range 1-75 ms. Predicted lag: median 33 ms.
- Lag error: median +0.5 ms; 19/20 events within +/-5 ms; 1 outlier at +30 ms
  (r1 6.494 s, not plotted here).
- Measured lag is strongly direction-dependent in this definition:
  rising current median 51 ms, falling current median 6.5 ms (250 N table).
  So this lag is not a single physical tau; it mixes motion direction and
  drift correction. Do not quote it as a pure current lag.
- For the 50 N threshold (25 events): median error 0.0 ms, 22/25 within +/-5 ms,
  outliers -29, -20, +30 ms.

## Plotted events (window: -100 ms to +200 ms around the table step edge)

| Figure | File | t_edge | dI | dF | Lag meas | Lag pred | Lag err |
|---|---|---:|---:|---:|---:|---:|---:|
| delay_A_falling_fast | ST_10mm_Vel0.13_step5_0_1.6A_r2 | 1.622 s | -1.59 A | +458 N | 6 ms | 5 ms | -1 ms |
| delay_B_rising_slow | ST_10mm_Vel0.13_step5_0_1.6A_r2 | 6.020 s | +1.59 A | -474 N | 75 ms | 76 ms | +1 ms |
| delay_C_rising_normal | ST_10mm_Vel0.13_step5_0_1.6A_r1 | 3.038 s | +0.95 A | -268 N | 28 ms | 28 ms | 0 ms |

Upper panel: measured force (black) vs predicted force (blue), dashed line = table step edge.
Lower panel: measured current (A).
Note: in some windows the force swings again about 50-150 ms after the edge;
that is subsequent motion reversal, not the current step itself.
