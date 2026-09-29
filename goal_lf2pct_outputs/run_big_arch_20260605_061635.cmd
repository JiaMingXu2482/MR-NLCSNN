@echo off
cd /d "D:\OneDrive - mail.scut.edu.cn\Python\MR-NLCSNN"
set CDC_OUTPUT_ROOT=D:\OneDrive - mail.scut.edu.cn\Python\MR-NLCSNN\goal_lf2pct_outputs
set CDC_PROCESSED_GROUPS=D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Processed_Groups_new_zerophase_force_wide
set CDC_VAL_MODE=lf_holdout
set CDC_LF_VAL_RATIO=0.25
set CDC_LF_VAL_SEED=20260516
set CDC_NORM_USES_VAL=0
set CDC_INCLUDE_STEP=1
set CDC_INCLUDE_HF=1
set CDC_EXTRA_DYNAMIC_NFL=1
set CDC_DIRECT_NFL_OUTPUT=1
set CDC_SLOW_GAIN_NFL=1
set CDC_EPOCHS=240
set CDC_HIDDEN=256
set CDC_STATE_DIM=12
set CDC_LR=8e-4
set CDC_FILE_BATCH_SIZE=32
set CDC_LENGTH_BUCKETS=1
set CDC_CHUNK_LEN=500
set CDC_WARMUP=50
set CDC_VAL_WARMUP_MS=200
set CDC_MONITOR_EVERY=0
set CDC_VAL_ALL_EVERY=0
set CDC_VALIDATE_ALL_ON_EPOCH1=1
set CDC_VAL_EVERY_SECONDS=7200
set CDC_VAL_START_SECONDS=15
set CDC_VAL_MAX_SECONDS=5
set CDC_VAL_ALL_MAX_FILES=0
set CDC_PERIODIC_EVERY=10
set CDC_PLOT_EVERY=0
set CDC_TQDM=0
set CDC_HIDDEN_NORM_WEIGHT=0
set CDC_DH_NORM_WEIGHT=0
set CDC_P2P_LOSS_POWER=2.0
set CDC_NEG_FORCE_WEIGHT=1.5
set CDC_HIGH_FORCE_ABS_THRESHOLD_N=0
set CDC_HIGH_FORCE_WEIGHT=1.0
set CDC_NEAR_ZERO_V_THRESHOLD_MM_S=0
set CDC_NEAR_ZERO_V_WEIGHT=1.0
set CDC_HIGH_FORCE_NEAR_ZERO_WEIGHT=1.0
set CDC_WEIGHT_LOW_AMP=1.0
set CDC_WEIGHT_HIGH_FREQ=1.0
set CDC_WEIGHT_10MM=1.0
set CDC_WEIGHT_HIGH_AMP=1.0
set CDC_WEIGHT_HIGH_AMP_LOW_FREQ=1.0
echo Started %DATE% %TIME% > "D:\OneDrive - mail.scut.edu.cn\Python\MR-NLCSNN\goal_lf2pct_outputs\big_arch_20260605_061635.log"
"C:\Users\user\.conda\envs\myenvPINN_test\python.exe" "D:\OneDrive - mail.scut.edu.cn\Python\MR-NLCSNN\AA02Train\train_legacy_repro.py" >> "D:\OneDrive - mail.scut.edu.cn\Python\MR-NLCSNN\goal_lf2pct_outputs\big_arch_20260605_061635.log" 2>&1
