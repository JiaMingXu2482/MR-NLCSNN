from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AA03Validate.validate_legacy_repro import LegacyCDCValidator, get_device

CHECKPOINT = REPO_ROOT / "training_outputs/legacy_repro_180_epochs_20260620_105245/nlcsnn_legacy_best_all_val.pth"
CASE_FILE = Path(r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Parsed_Card_force_wide_zerofix\4_Random_LF_Current\RLF_RMS15mm_2Hz11.csv")
OUTPUT_DIR = Path(__file__).resolve().parent

validator = LegacyCDCValidator(CHECKPOINT, get_device("cuda"))
row = validator.validate_case(
    CASE_FILE,
    warmup_ms=200,
    baseline_cal_samples=1000,
)

mask = np.asarray(row["time"]) >= row["warmup_s"]
time = np.asarray(row["time"])[mask]
measured = np.asarray(row["measured"])[mask]
predicted = np.asarray(row["predicted"])[mask]

pd.DataFrame(
    {"time_s": time, "measured_force_N": measured, "predicted_force_N": predicted}
).to_csv(OUTPUT_DIR / "best_case_prediction.csv", index=False)

fig, ax = plt.subplots(figsize=(13.2, 6.2))
ax.plot(
    time,
    measured,
    color="#686868",
    linewidth=2.7,
    linestyle="-",
    label="Measured force",
    zorder=2,
)
ax.plot(
    time,
    predicted,
    color="#d62728",
    linewidth=2.0,
    linestyle=(0, (3.2, 2.2)),
    label="Predicted force",
    zorder=3,
)
ax.set_xlabel("Time (s)", fontsize=13)
ax.set_ylabel("Damper force (N)", fontsize=13)
ax.set_title("RLF RMS 15 mm, 2 Hz (11) — Best 10–30 mm validation case", fontsize=15, pad=12)
ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
ax.legend(loc="upper right", frameon=True, fontsize=12)
ax.tick_params(labelsize=11)
ax.margins(x=0.01)
metrics = (
    f"NRMSE = {row['NRMSE (%)']:.2f}%\n"
    f"MAE = {row['MAE (N)']:.2f} N\n"
    f"RMSE = {row['RMSE (N)']:.2f} N"
)
ax.text(
    0.015,
    0.97,
    metrics,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=11.5,
    bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#bcbcbc", "alpha": 0.92},
)
fig.tight_layout()
out_path = OUTPUT_DIR / "best_case_measured_vs_predicted.png"
fig.savefig(out_path, dpi=220, bbox_inches="tight")
plt.close(fig)

print(f"case={row['Case']}")
print(f"nrmse_pct={row['NRMSE (%)']:.6f}")
print(f"mae_n={row['MAE (N)']:.6f}")
print(f"rmse_n={row['RMSE (N)']:.6f}")
print(f"scored_samples={int(mask.sum())}")
print(f"plot={out_path}")
