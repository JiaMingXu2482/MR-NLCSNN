# GitHub archive note — 2026-10-27

The user authorized publishing this completed local A/B/D experiment after the results were delivered. Statements such as “未推送GitHub” in the frozen reports describe their status when generated, before this upload.

Start with [the comparison report](results_summary_ABD/README.md).

- Base training source: cac96e8f85491e731a0030c268f8c9f814b06953, plus the supplied runner patch in the archived source snapshot.
- Each arm: 30 epochs, cold start, seed 7; 259 training / 16 validation files.
- A: 21,041 parameters; best epoch 30; validation pooled NRMSE 9.870482697184269%; corresponding 6.62 Hz 5.708410605409629%.
- B: 19,121 parameters; best epoch 30; validation pooled NRMSE 10.470538436299973%; corresponding 6.62 Hz 8.071124197877758%.
- D: 18,401 parameters; best epoch 22; validation pooled NRMSE 9.43658142458234%; corresponding 6.62 Hz 6.255000973673345%.
- Checkpoints are Stage 1 forward-trained; inverse heads are untrained. Timing is local PC CPU Python/PyTorch, not S32/NI measurements.
- Final-test CSV files were not extracted or read. No raw or parsed training/test CSV data are included in this archive.
- Configs and checkpoints retain original local paths as provenance; archived source copies are under results_summary_ABD/code_snapshot. The source snapshot is archival, not a replacement for mainline files.
- delivery_file_hashes.json verifies the 65 original payload files. This note, the hash manifest itself and .gitattributes are additional archive files. The directory disables Git text conversion to preserve payload hashes.
