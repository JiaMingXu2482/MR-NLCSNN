# Validate NLCSNN

Run validation on a trained NLCSNN checkpoint.

## Validate command

```powershell
& C:\Users\user\.conda\envs\myenvPINN_test\python.exe "d:/OneDrive - mail.scut.edu.cn/Python/MR-NLCSNN/AA03Validate/Validate.py"
```

## How it works

- `Validate.py` loads a checkpoint and runs open-loop simulation (RK4 integration with zero initial hidden state)
- Excludes first 200 ms of data from metric computation
- Saves per-case CSV (time, force_measured, force_predicted, abs_error), PNG plot, and summary CSV to `AA03Validate/results/run_<timestamp>/`

## Customization

Edit the `VAL_CASES` list in `Validate.py` to change which test cases are evaluated.
