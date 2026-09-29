from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent
SOURCE = OUTPUT_DIR / "best_case_prediction.csv"
START_S = 4.2
END_S = 7.0

full = pd.read_csv(SOURCE)
data = full[(full["time_s"] >= START_S) & (full["time_s"] <= END_S)].copy()
if data.empty:
    raise RuntimeError("Selected segment contains no samples")

y = data["measured_force_N"].to_numpy(float)
y_hat = data["predicted_force_N"].to_numpy(float)
error = y_hat - y
rmse = float(np.sqrt(np.mean(error**2)))
mae = float(np.mean(np.abs(error)))
y_rms = float(np.sqrt(np.mean(y**2)))
y_range = float(np.ptp(y))
nrmse_rms = 100.0 * rmse / y_rms
nrmse_range = 100.0 * rmse / y_range
r2 = 1.0 - float(np.sum(error**2) / np.sum((y - np.mean(y)) ** 2))

out_csv = OUTPUT_DIR / "late_segment_4p2_to_7p0_prediction.csv"
data.to_csv(out_csv, index=False)

fig, ax = plt.subplots(figsize=(13.2, 6.2))
ax.plot(
    data["time_s"], y,
    color="#686868", linewidth=2.7, linestyle="-",
    label="Measured force", zorder=2,
)
ax.plot(
    data["time_s"], y_hat,
    color="#d62728", linewidth=2.0, linestyle=(0, (3.2, 2.2)),
    label="Predicted force", zorder=3,
)
ax.set_xlabel("Time (s)", fontsize=13)
ax.set_ylabel("Damper force (N)", fontsize=13)
ax.set_title(
    "RLF RMS 15 mm, 2 Hz (11) — Later segment (4.2–7.0 s)",
    fontsize=15, pad=12,
)
ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
ax.legend(loc="upper right", frameon=True, fontsize=12)
ax.tick_params(labelsize=11)
ax.margins(x=0.01)
metrics = (
    f"RMSE = {rmse:.2f} N\n"
    f"NRMSE (RMS-normalized) = {nrmse_rms:.2f}%\n"
    f"NRMSE (range-normalized) = {nrmse_range:.2f}%\n"
    f"R² = {r2:.4f}"
)
ax.text(
    0.015, 0.97, metrics,
    transform=ax.transAxes, ha="left", va="top", fontsize=11.5,
    bbox={
        "boxstyle": "round,pad=0.45", "facecolor": "white",
        "edgecolor": "#bcbcbc", "alpha": 0.92,
    },
)
fig.tight_layout()
out_png = OUTPUT_DIR / "late_segment_4p2_to_7p0_measured_vs_predicted.png"
fig.savefig(out_png, dpi=220, bbox_inches="tight")
plt.close(fig)

print(f"segment={START_S:.1f}-{END_S:.1f} s")
print(f"samples={len(data)}")
print(f"rmse_n={rmse:.6f}")
print(f"mae_n={mae:.6f}")
print(f"measured_rms_n={y_rms:.6f}")
print(f"measured_range_n={y_range:.6f}")
print(f"nrmse_rms_pct={nrmse_rms:.6f}")
print(f"nrmse_range_pct={nrmse_range:.6f}")
print(f"r2={r2:.8f}")
print(f"plot={out_png}")
print(f"csv={out_csv}")
