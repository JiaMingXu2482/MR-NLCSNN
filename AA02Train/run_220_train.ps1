# 220-epoch FINE-TUNE launcher — run in YOUR OWN PowerShell terminal so it survives
# independent of any agent session. Warm-starts from the best 0620 checkpoint (weights
# only, fresh cosine) instead of training from scratch, so it refines the proven model
# (24.26% all-val / 17.7% v>=0.05+cal) rather than gambling on a new random seed.
# Standard config: compile ON, batch 48 (~90% GPU util / ~2.5GB), valve NFL, all train data.
$best = "D:\OneDrive - mail.scut.edu.cn\Python\MR-NLCSNN\training_outputs\legacy_repro_180_epochs_20260620_105245\nlcsnn_legacy_best_all_val.pth"

$env:CDC_RESUME_CHECKPOINT  = $best
$env:CDC_RESUME_WEIGHTS_ONLY= "1"
$env:CDC_USE_RESUME_NORM    = "1"
$env:CDC_LR                 = "3e-4"

$env:CDC_EPOCHS         = "220"
$env:CDC_HIDDEN         = "256"
$env:CDC_STATE_DIM      = "8"
$env:CDC_PHYSICS_NFL    = "1"
$env:CDC_FILE_BATCH_SIZE= "48"
$env:CDC_CHUNK_LEN      = "1000"
$env:CDC_COSINE_FINETUNE= "1"
$env:CDC_COSINE_ETA_MIN = "1e-5"
$env:CDC_COMPILE        = "1"
$env:CDC_TRAIN_CATS     = "1_Steady_Harmonic,2_Steady_Random,5_Step_Current_Triangle"
$env:CDC_VAL_CATS       = "4_Random_LF_Current"
$env:CDC_PROCESSED_GROUPS = "D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Parsed_Card_force_wide_zerofix"

$py = "C:\Users\user\.conda\envs\myenvPINN_test\python.exe"
$train = "D:\OneDrive - mail.scut.edu.cn\Python\MR-NLCSNN\AA02Train\train_legacy_repro.py"
Write-Host "[launcher] FINE-TUNE from 0620 best (weights-only, lr 3e-4, 220 epochs). Keep this window open." -ForegroundColor Cyan
& $py $train
Write-Host "[launcher] training process exited." -ForegroundColor Yellow
