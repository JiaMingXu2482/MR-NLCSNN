# V25.2-RI0-CMD — 100 Hz command-proxy residual inverse model

Version: V25.2  
Run: `V25_2_RI0_CMD_PROXY_seed7`  
Purpose: remove dependence on measured-current feedback at deployment while retaining the V25 residual parameterization and the same compact network.

---

## 1. Deployment contract

The target ECU executes the inverse model at 100 Hz (10 ms per control step).

Available at decision step `k`:
- current desired damping force / force request at step `k`;
- mechanical state and other causal input signals at step `k` and their permitted history;
- the previous command actually issued by the ECU, `I_cmd[k-1]`.

Not available:
- measured coil current feedback;
- any future signal;
- any future command.

Engineering assumption supplied by the user:

\[
I_{\mathrm{act}}[k]\approx I_{\mathrm{cmd}}[k-1]
\]

because the current-driver/coil response is expected to be within one 10 ms control interval. Treat this as an **engineering approximation to be tested**, not as a universally guaranteed identity.

Runtime model:

\[
I_{\mathrm{base}}[k]=I_{\mathrm{cmd}}[k-1]
\]

\[
\Delta \hat I[k]=f_\theta(X[k])
\]

\[
I_{\mathrm{cmd}}[k]=
\operatorname{clip}\left(I_{\mathrm{base}}[k]+\Delta\hat I[k],0,1.6\right)
\]

The previous command is a known ECU state and may be used as the residual base. This is not measured-current feedback. It is also not an uncontrolled accumulation of an estimated physical current: it is the last command value stored by the ECU.

At file start, initialize command state using the agreed test initialization (normally zero); reset at every file boundary. Never carry state between files.

---

## 2. Offline proxy-label construction — exact indexing

The historical data contains measured coil current `I_meas`, not `I_cmd`. Under the agreed one-step assumption:

\[
I_{\mathrm{meas}}[k]\approx I_{\mathrm{cmd}}[k-1]
\]

Therefore, for a decision at control index `k`:

\[
\boxed{I_{\mathrm{base,proxy}}[k]
=\operatorname{clip}(|I_{\mathrm{meas}}[k]|,0,1.6)}
\]

\[
\boxed{I_{\mathrm{target,proxy}}[k]
=\operatorname{clip}(|I_{\mathrm{meas}}[k+1]|,0,1.6)}
\]

\[
\Delta I_{\mathrm{target}}[k]
=I_{\mathrm{target,proxy}}[k]-I_{\mathrm{base,proxy}}[k]
\]

The sign-corrected measured current is converted to the nonnegative command convention using absolute value and clipped to the physical command interval `[0, 1.6] A`, matching the previous V25 label convention. Record raw signed current separately for diagnostics.

**Time alignment rule:**
- Input mechanical/force features describe the decision state at `k`.
- The previous-command input and residual base at decision `k` use the proxy `I_meas[k]`.
- The proxy command label for the command issued at `k` is `I_meas[k+1]`.
- `I_meas[k+1]` is a label only. It must never appear in `X[k]`, normalization statistics, recurrent state, or any other input at decision `k`.
- Drop the final 100 Hz sample in each file because it has no `k+1` proxy label.
- Do not wrap, roll, pad from the next file, or join files.
- The first sample may be retained if all required causal features and the initial command state are valid.

This one-step target is intentionally different from V25's old `base=I_meas[k-1]`, `target=I_meas[k]`. Do not reuse the old index pair.

### Current-history feature mapping
The old V25 feature audit identified a historical-current feature and a residual base derived from measured current. The local agent must inspect the actual V24/V25 feature builder and identify **every** current-derived input.

At runtime, every current-history feature must be populated from stored ECU command history, not measured current. Offline proxy mapping must obey:

\[
I_{\mathrm{cmd}}[k-j]\approx I_{\mathrm{meas}}[k-j+1]
\]

for each lag `j` that actually exists in the feature map. Do not guess the feature index or lag convention. If the source code cannot establish it unambiguously, STOP and report.

For the one-step residual base specifically, use `I_meas[k]` as the proxy for `I_cmd[k-1]`.

---

## 3. Preprocessing: zero added digital filter delay

This version deliberately has **no FIR or IIR low-pass filter** in the inverse-model time-varying signal path.

- Do not use the old 61-tap / 40 Hz FIR.
- Do not substitute one of the six V25.1 filter candidates.
- Do not use `filtfilt`, `sosfiltfilt`, centered smoothing, or future-sample compensation.
- Convert the synchronized 1 kHz records to the 100 Hz control grid by direct sampling with the established phase (`[::10]`, phase 0), applied consistently to every synchronized channel.
- Preserve units, sign correction, feature order, and normalization definitions unless the source audit proves they depend on the old FIR. If normalization statistics must be recomputed, use training files only and save the exact statistics.
- Any derivative or window feature must be causal. A derivative must use current/past samples (e.g. backward difference), never a centered difference that consumes `k+1`.
- Apply identical sample indices and time stamps to force, velocity, displacement, and current proxy channels.

**No-filter does not mean no risk.** Direct 1 kHz-to-100 Hz sampling can alias energy above 50 Hz. The local agent must calculate and report above-Nyquist energy for each relevant channel/family on train and validation. Do not silently add a filter if aliasing is high. If the alias audit indicates that the raw 100 Hz representation is unusable, stop before formal training and report the evidence for approval.

The digital low-pass filter contribution is 0 ms by construction. This does **not** mean total system delay is 0 ms: the 10 ms control period, sampling phase, driver/coil response, communication, and computation remain.

---

## 4. Network — keep V25 compact architecture

Keep the V25 architecture unchanged:
- GRU: 2 layers, hidden size 8;
- 31 input features, after source-code feature-map audit;
- skip concatenation: GRU output 8 + input 31 = 39;
- Linear(39, 16) + ReLU;
- Linear(16, 1), linear output for `delta_current_A`;
- expected trainable parameter count: **2073**.

Do not deepen/widen the GRU, add branches, add trainable filters, or change the MLP. Parameter count mismatch is a stop condition until explained from source code.

Train from scratch; do not warm-start from V24-I0 or V25 RI0. Keep the old checkpoints frozen.

---

## 5. Training target and loss

For each valid control index:

\[
I_{\mathrm{raw}}[k]
=I_{\mathrm{base,proxy}}[k]+\Delta\hat I[k]
\]

\[
L=\operatorname{MSE}\left(
\frac{I_{\mathrm{raw}}[k]-I_{\mathrm{target,proxy}}[k]}{1.6}
\right)
\]

- Compute loss on the **unclipped** raw prediction.
- Apply `[0, 1.6] A` clipping only for deployed output and clipped metrics.
- Do not add zero-current weighting, smoothness penalties, dead zones, monotonicity penalties, force-cascade losses, or other regularizers in this experiment.
- Do not use future labels as inputs.
- Reset GRU state at every file boundary.

---

## 6. Dataset and protocol

Use the established dataset and split:
- Data root: `D:\\v20_data\\Parsed_v19_F42_I42`
- Manifest SHA256: `0766763e7fb77880ffa04f2d93bcf629bfeb817e51c571188eee1ea83262e2ff`
- Train / validation / final_test: **259 / 16 / 10 files**
- Train family counts: SH/ST/RLF/SR/RHF = **186/30/18/17/8**
- Validation family counts: **8/3/3/1/1**
- Sampling: original Baseline; no Balanced sampling or family reweighting.
- Seed: 7.
- Device: CUDA.
- One formal model only: `V25_2_RI0_CMD_PROXY_seed7`.
- Maximum budget: 180 epochs. Preserve V25's optimizer, scheduler, batch/sequence organization, gradient clipping, and other training settings unless the local source audit identifies a bug. Record every deviation.
- Keep `final_test` sealed. Training, validation, diagnostics, plotting, feature scaling, and model selection must read 0 final_test files.

Because the previous V25 run was still improving at epoch 130, report convergence honestly. Do not silently extend beyond 180 or change the scheduler.

---

## 7. Mandatory Phase 0 audit — before training

Produce these files:
1. `data_manifest_audit.json`: manifest hash, split counts, file hashes, family counts.
2. `feature_timing_audit.json`: all 31 feature names/indices, signal source, timestamp, lag, causal availability, and runtime source.
3. `current_proxy_alignment_audit.json`: prove base `I_meas[k]`, target `I_meas[k+1]`, delta, file-boundary handling, and no leakage.
4. `preprocessing_audit.json`: prove FIR/IIR removed from all dynamic channels, direct sample phase 0, synchronized time stamps, causal derivatives.
5. `alias_energy_audit.json`: energy above 50 Hz by signal/family and train/validation.
6. `architecture_audit.json`: GRU/MLP dimensions and 2073 trainable parameters.
7. `deployment_contract.json`: runtime inputs available on ECU and stored command-state update.
8. `final_test_access.json`: access count must be 0.

**Stop conditions:**
- manifest/split/hash mismatch;
- unclear current sign or physical meaning;
- unclear feature lag/current-history mapping;
- `I_meas[k]` / `I_meas[k+1]` indexing cannot be proven;
- any input at `k` uses `k+1` or later;
- model parameter count differs from 2073 without an explained source-level reason;
- no-filter decimation produces unacceptable aliasing (report; do not silently filter);
- final_test is read.

The user's one-step response assumption authorizes this proxy experiment, but does not turn the proxy into measured ECU command ground truth.

---

## 8. Phase 1 smoke test

Run 2 epochs on the official train/validation split after Phase 0 passes.

Required tests:
- direct sample phase and timestamp equality across channels;
- proxy base equals `clip(abs(I_meas_100Hz[k]), 0, 1.6)`;
- proxy label equals `clip(abs(I_meas_100Hz[k+1]), 0, 1.6)`;
- delta target equals target minus base;
- current-history feature uses previous command proxy, not old measured-current feedback slot;
- changing `I_meas[k+1]` changes only the label for decision `k`, never `X[k]`;
- no cross-file state or wraparound;
- output delta can be positive and negative and can be exactly zero;
- no clipping inside loss;
- finite/nonzero gradients, no NaN/Inf;
- checkpoint save/load equivalence;
- 2073 parameters;
- `final_test` access count 0.

If any test fails, do not launch formal training.

---

## 9. Evaluation

All metrics must use the same valid validation mask and the new proxy-label alignment.

### Primary proxy-command metrics
- MAE, RMSE, median absolute error, P95, P99, mean bias (A);
- pooled NRMSE with explicit formula (secondary only);
- file-macro MAE and sample-micro MAE;
- SH/ST/RLF/SR/RHF MAE, RMSE, P95, bias;
- raw out-of-range rate and clip activation rate;
- zero-proxy-current bias for target <0.05 A;
- false turn-on rates >0.05 A and >0.10 A;
- delta-current MAE/RMSE/P95;
- command total variation and per-step command changes;
- rising/falling edge 10–90% delay measured on the 100 Hz grid, with the grid resolution explicitly stated.

### Baselines
- Persistence: `I_cmd_pred[k] = I_base_proxy[k]`.
- V24-I0 and V25 RI0 may be included only as historical references. Their old filtered labels and old timing definition are not directly comparable to the V25.2 proxy-command target. If recomputed, label the comparison as non-equivalent or separately re-evaluate from raw data under a clearly defined common target.

SR NRMSE may be degenerate because current is near zero; MAE/RMSE and false-turn-on metrics take priority. Validation SR and RHF each have only one file, so per-family conclusions are directional, not statistically conclusive.

### Real-time
Report CPU single-step P50/P95/P99/max and parameter count. CUDA timing is supplementary. Python/CUDA timing is not MCU WCET; measure generated/ported C on the target ECU separately.

### HIL deployment gate
Before claiming the command model is validated, log on HIL/bench:
- actual `I_cmd[k]` sent by ECU;
- measured coil `I_meas[k]`;
- synchronized timestamps;
- force and mechanical state.
Then compare `I_meas[k]` with `I_cmd[k-1]` directly. This is the only way to quantify the one-step command-to-current approximation on the real driver/coil.

---

## 10. Deliverables

Output root: `D:\\v25_2_cmd_proxy\\`

```text
D:\\v25_2_cmd_proxy\\
├── task\\V25_2_TRAINING_PACKAGE.md
├── configs\\V25_2_RI0_CMD_PROXY_seed7.json
├── audits\\
│   ├── data_manifest_audit.json
│   ├── feature_timing_audit.json
│   ├── current_proxy_alignment_audit.json
│   ├── preprocessing_audit.json
│   ├── alias_energy_audit.json
│   ├── architecture_audit.json
│   └── deployment_contract.json
├── runs\\V25_2_RI0_CMD_PROXY_seed7\\
│   ├── run_config.json
│   ├── metrics.jsonl
│   ├── best.pth
│   ├── last.pth
│   ├── summary.json
│   ├── per_file_metrics.csv
│   ├── per_condition_metrics.csv
│   ├── proxy_command_metrics.json
│   ├── persistence_comparison.csv
│   └── figures\\
├── logs\\gpu_1s.csv
├── V25_2_FINAL_REPORT.md
└── release_manifest.sha256
```

---

## 11. Final report must answer

1. Did V25.2 improve proxy-command MAE/RMSE/P95/P99 over persistence?
2. Does it improve all five operating families, especially SR without harming RHF?
3. What is the zero-current bias and false turn-on rate?
4. What are raw range violations and clip activation rate?
5. What is the command total variation / jitter?
6. Is the model converged by the 180-epoch budget?
7. Is there zero added digital filter delay, and what is the direct-decimation alias energy?
8. Are all inputs causal and available on ECU?
9. Is the previous-command proxy aligned and does it pass all indexing tests?
10. What is the CPU single-step inference latency?
11. Which results are only proxy-label results and which have been verified against actual HIL command/current logs?
12. Is it ready for HIL closed-loop testing, or only for an offline proxy experiment?

Never claim true command accuracy without actual command/current pair validation.
