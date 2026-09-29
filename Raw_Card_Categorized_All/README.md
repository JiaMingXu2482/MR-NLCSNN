# Raw_Card_Categorized_All

This directory is a categorized archive of raw acquisition-card `auto_1.csv` files. Files are copied byte-for-byte from the source experiment folders after removing exact duplicate copies within the same normalized condition. They are not filtered, zeroed, converted to physical quantities, or otherwise processed.

## Categories

- `1_SH`: steady harmonic displacement with fixed current
- `2_SR`: steady random displacement with fixed current
- `3_RHF`: random high-frequency current plus random displacement
- `4_RLF`: random low-frequency current plus random displacement
- `5_ST`: standard velocity or step-current cases

## File Name Format

`{CAT}_{readable_setting}.csv` when only one file remains for a condition.
`{CAT}_{readable_setting}_r{N}.csv` when multiple distinct files remain for the same condition.

Examples:

- `SH_10mm_1.66Hz_0.8A_r1.csv`
- `SR_RMS10mm_4.97Hz_0.2A.csv`
- `RLF_RMS15mm_3Hzslow.csv`
- `ST_30mm_Vel0.13.csv`
- `ST_50mm_Vel0.13_step5_0_1.6A_r3.csv`

## Abbreviations

- `SH`: steady harmonic displacement with fixed current
- `SR`: steady random displacement with fixed current
- `RHF`: random high-frequency current plus random displacement
- `RLF`: random low-frequency current plus random displacement
- `ST`: standard velocity or step-current case
- `10mm`: displacement amplitude 10 mm
- `RMS15mm`: displacement RMS 15 mm
- `1.66Hz`: frequency 1.66 Hz
- `3Hzslow`: 3 Hz slow condition
- `0.8A`: current 0.8 A
- `1.6A`: current or peak current 1.6 A
- `Vel0.13`: velocity setting 0.13
- `step5_0`: step5_0 step-current program
- `r1`, `r2`, ...: repeat number used only when multiple distinct files remain within the same normalized condition key after exact-duplicate removal

The source root is intentionally not included in file names. Use `manifest.csv` to trace each archived file back to its original folder and full source path.

Exact duplicates are detected by `(condition_key, sha256)`. When multiple source files have the same normalized condition and identical content, only the first source in stable source order is copied, then the remaining files are renumbered. Omitted duplicate source files are recorded in `duplicates_removed.csv`.

## Manifest

`manifest.csv` records:

- original source path and output path
- source root and original source folder
- category and short file name
- normalized condition key and repeat number
- file size, last write time, and SHA256
- special flags for legacy copies, `step430`, reversed-current correction requirements, manual-zero cases, and legacy `1.4A` step files

## Important Notes

- `step430` files collected on 2026-06-11 have reversed current wiring. The raw files here are not changed; the manifest flag `is_reversed_current_required` tells later processing to multiply current by `-1`.
- `AMP30mm_Vel0.13_step5_0_1.6A` requires manual zero points during physical processing. The raw file here is not changed.
- This archive is for raw data organization only. Use the CDC processing skill before converting these files into physical model-ready datasets.
