"""4-panel validation plots for the physics(+valve) NFL model.

Per validation case, one figure with 4 stacked subplots (top->bottom):
  1. 减震器位移 displacement (zeroed at the MTS start point = first sample)
  2. 电流 current
  3. 预测 vs 真实 force
  4. 绝对误差 absolute error

Time zero is set 1 s AFTER the MTS start (the first second is the h=0 cold-start /
zero-point-calibration window and is dropped from display). Each plot spans 5 s,
i.e. the original [1 s, 6 s] window shown as x in [0, 5] s.

Predictions use first-second baseline (zero-point) calibration — the deployment
recipe — estimated on the dropped [0, 1 s] window.
"""
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mr_nlcsnn.legacy_model import LegacyNLCSNN, rk4_step

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CKPT = REPO / "training_outputs" / "legacy_repro_180_epochs_20260620_105245" / "nlcsnn_legacy_best_all_val.pth"
LF_DIR = Path(r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Parsed_Card_force_wide_zerofix\4_Random_LF_Current")
OUT = REPO / "AA03Validate" / "reports" / "Validation_4panel"

CAL_S = 1.0      # first-second zero-point calibration / cold-start drop
PLOT_S = 5.0     # plotted span after time zero
FS = 1000        # Hz


def load_model(device):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    mc = ck["model_cfg"]
    m = LegacyNLCSNN(
        u_dim=6, h_dim=int(mc["h_dim"]), hidden=int(mc["hidden"]),
        physics_nfl=bool(mc.get("physics_nfl", False)),
    ).to(device)
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["state_dict"].items()})
    m.eval()
    return m, ck["norm_cfg"]


@torch.no_grad()
def rollout(m, cfg, df, n, device):
    u = np.stack([
        (df.rod_length.values - cfg["x_ref"]) / cfg["x_scale"],
        df.velocity.values / cfg["v_scale"],
        df.accel.values / cfg["a_scale"],
        df.current.values / cfg["i_scale"],
        df.current_dot.values / cfg["di_scale"],
        (df.temp.values - cfg["t_ref"]) / cfg["t_scale"],
    ], axis=1).astype(np.float32)[:n]
    u = torch.from_numpy(u).to(device)
    h = torch.zeros(1, m.h_dim, device=device)
    pred = np.empty(n, dtype=np.float32)
    for t in range(n):
        pred[t] = m.predict_force(u[t:t + 1], h).item()
        h = rk4_step(m, u[t:t + 1], h)
    return pred * cfg["f_scale"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, cfg = load_model(device)
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(LF_DIR.glob("*.csv"))
    print(f"[*] model loaded (NFL={m.nfl_dim}), {len(files)} validation cases -> {OUT}")

    cal_n = int(CAL_S * FS)
    win_n = int(PLOT_S * FS)
    need = cal_n + win_n  # 6000

    rows = []
    for path in files:
        df = pd.read_csv(path)
        n = min(len(df), need)
        if n < cal_n + FS:  # need >=1s after cal window
            print(f"  skip {path.stem}: too short ({len(df)})")
            continue
        pred = rollout(m, cfg, df, n, device)
        true = df.force.values[:n].astype(np.float32)

        offset = float(np.mean(pred[:cal_n] - true[:cal_n]))   # first-second zero-point cal
        pred_c = pred - offset

        s, e = cal_n, n                                        # scored/plotted window [1s, n]
        t = np.arange(s, e) / FS - CAL_S                       # time zero at 1s -> x in [0,5]
        disp = (df.rod_length.values[:n] - df.rod_length.values[0])[s:e]   # zero at MTS start
        curr = df.current.values[:n][s:e]
        meas, pp = true[s:e], pred_c[s:e]
        err = np.abs(pp - meas)
        nrmse = float(np.sqrt(np.mean((pp - meas) ** 2)) / (np.sqrt(np.mean(meas ** 2)) + 1e-6) * 100)
        rows.append((path.stem, nrmse))

        fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        ax[0].plot(t, disp, color="#1D9E75", lw=1.0)
        ax[0].set_ylabel("位移 (mm)")
        ax[0].set_title(f"{path.stem}  |  RMS-NRMSE = {nrmse:.1f}%  (首秒零点标定, 时间零点=启动后1s)")
        ax[0].grid(True, alpha=0.25)

        ax[1].plot(t, curr, color="#BA7517", lw=1.0)
        ax[1].set_ylabel("电流 (A)")
        ax[1].grid(True, alpha=0.25)

        ax[2].plot(t, meas, color="black", alpha=0.45, lw=1.2, label="真实值 Measured")
        ax[2].plot(t, pp, color="red", ls="--", lw=1.1, label="预测值 NLCSNN")
        ax[2].set_ylabel("力 (N)")
        ax[2].legend(loc="best")
        ax[2].grid(True, alpha=0.25)

        ax[3].plot(t, err, color="purple", lw=1.0)
        ax[3].set_ylabel("绝对误差 (N)")
        ax[3].set_xlabel("时间 (s)  — 零点为 MTS 启动后 1s")
        ax[3].grid(True, alpha=0.25)
        ax[3].set_xlim(0, PLOT_S)

        fig.tight_layout()
        fig.savefig(OUT / f"Result4_{path.stem}.png", dpi=130)
        plt.close(fig)
        print(f"  {path.stem}: NRMSE={nrmse:.1f}%")

    if rows:
        rows.sort(key=lambda r: r[1])
        print("\n=== summary (sorted) ===")
        for name, v in rows:
            print(f"  {name:28s} {v:5.1f}%")
        vals = np.array([v for _, v in rows])
        print(f"\nN={len(vals)}  mean={vals.mean():.1f}%  max={vals.max():.1f}%")


if __name__ == "__main__":
    main()
