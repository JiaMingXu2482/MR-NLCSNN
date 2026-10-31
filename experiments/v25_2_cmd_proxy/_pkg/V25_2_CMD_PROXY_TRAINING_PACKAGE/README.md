# V25.2-RI0-CMD — Training handoff package

## Purpose
Build one 100 Hz inverse-current model that can run on the target ECU without measured coil-current feedback. At runtime it uses the previous issued current command as the current-state proxy.

This package is a **local-agent handoff**, not a claim that training has already run. The local agent must adapt the existing V25/V24 training code and local dataset; it must not invent missing feature mappings or silently change timing.

## Files
- `task/V25_2_TRAINING_PACKAGE.md`: full experiment specification and stop gates.
- `prompts/HANDOFF_TO_LOCAL_AGENT.txt`: paste this into the local agent.
- `configs/V25_2_RI0_CMD_PROXY_seed7.json`: machine-readable configuration.
- `scripts/proxy_alignment.py`: explicit one-step proxy-label construction.
- `scripts/test_proxy_alignment.py`: unit tests for the alignment helper.
- `audits/KNOWN_V25_BASELINE_FACTS.md`: facts recovered from the previous V25 package and delay audit.

## First steps
1. Extract the ZIP.
2. Run `python scripts/test_proxy_alignment.py` (requires NumPy).
3. Give the entire extracted folder to the local agent.
4. Ask it to run Phase 0 audit and 2-epoch smoke before formal training.

## Important limitation
The historical dataset contains measured coil current, not the ECU command signal. Therefore `I_meas[k+1]` is used as a **proxy label** for `I_cmd[k]` under the agreed one-control-step response assumption. This is not ground-truth command supervision. The final report must call it proxy-command error and must not claim true ECU command accuracy until command/current pairs are recorded on HIL or the bench.
