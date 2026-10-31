# Recovered V25 facts used by this package

These facts were recovered from the saved V25 training specification and V25 delay audit:

1. V25-RI0 used GRU 2 layers × hidden 8, 31 input features, skip concat 8+31=39, MLP16, linear delta head, 2073 trainable parameters.
2. V25's old residual base was same-file previous 100 Hz measured current `I_meas[k-1]`; target was `I_meas[k]`. This required measured-current feedback at deployment.
3. V25's old 61-tap, 40 Hz Hamming FIR at 1 kHz had a measured group delay of about 29.82 ms (theoretical 30 ms).
4. The old 100 Hz decimation used `[::10]` phase 0.
5. V25 delay audit decomposed approximately 29.82 ms FIR + 3.77 ms sampling-phase quantization + 4.91 ms model/base contribution = 38.75 ms measured total.
6. The audit identified a current-history feature in the 31-D input. The actual source code must be inspected to locate every current-derived feature and its lag; do not hardcode an index based only on this note.
7. The historical manifest hash is `0766763e7fb77880ffa04f2d93bcf629bfeb817e51c571188eee1ea83262e2ff`, with 259 train / 16 validation / 10 final_test files.
8. The dataset's current column is measured coil current, not a logged ECU command. Consequently, V25.2 labels are proxy commands based on the user's one-step response assumption.
9. Previous V25 training used Baseline sampling, seed 7, CUDA, 130 epochs. The previous run was still slightly improving at epoch 130; this package sets a maximum budget of 180 epochs while inheriting the original optimizer/scheduler.
10. Validation SR and RHF each have one file. Per-family findings on those conditions have high uncertainty.
