"""E0 step-lag surface over |velocity| x |Delta I|.

Same detector as the main matrix (train_v22.step_events): current 10% -> force
90%, velocity-drift corrected, |dF| >= 250 N.  Uses every ST file (train +
validation); final_test is never opened.  Single seed 7 checkpoint.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = r"D:/v22_ae_hybrid"
sys.path.insert(0, HERE)
from v20_data import normalization, read_dataset            # noqa: E402
from v22_models import V22Model                             # noqa: E402
from train_v22 import evaluate, step_events                 # noqa: E402

ROOT = os.path.join(HERE, "runs_seed7")
DATA = r"D:/v20_data/Parsed_v19_F42_I42"
OUT = os.path.join(HERE, "fig_E0_lag_surface")
os.makedirs(OUT, exist_ok=True)
CAT = ["5_Step_Current_Triangle"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = read_dataset(DATA, "train")
norm = normalization(train)
st = read_dataset(DATA, "train", categories=CAT) + \
     read_dataset(DATA, "validation", categories=CAT)
print("ST 文件数:", len(st))

net = V22Model("E0")
ck = torch.load(os.path.join(ROOT, "E0", "best.pth"), map_location="cpu", weights_only=True)
net.load_state_dict(ck["state_dict"])
net.to(dev).eval()

_, keep = evaluate(net, st, norm, dev, keep=True)
ev = step_events(keep, 250.0)
print("合格事件:", len(ev))

vel = {}
for it in st:
    vel[it["name"]] = it["values"][:, 1].numpy()
rows = []
for e in ev:
    k = int(round(e["t_edge"] * 1000))
    v = vel[e["file"]]
    rows.append(dict(file=e["file"], t_edge=e["t_edge"], dI=e["dI"], dF=e["dF"],
                     v=float(np.median(v[max(0, k - 80):k - 10])),
                     lag_true=e["lag_true"], lag_pred=e["lag_pred"]))
json.dump(rows, open(os.path.join(OUT, "e0_step_events.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print("速度范围 %.0f..%.0f, |dI| 取值 %s" % (
    min(abs(r["v"]) for r in rows), max(abs(r["v"]) for r in rows),
    sorted({round(abs(r["dI"]), 2) for r in rows})))

# ---------------------------------------------------------------- plotting
HV, HD = 55.0, 0.14
VMIN, VMAX = 40.0, 540.0
DMIN, DMAX = 0.50, 1.70
SUPPORT = 1.6
ZLO, ZHI = -12.0, 80.0


def nw(x, y, z, gx, gy):
    X, Y = np.meshgrid(gx, gy)
    w = np.exp(-((X[..., None] - x) ** 2) / (2 * HV ** 2)
               - ((Y[..., None] - y) ** 2) / (2 * HD ** 2))
    sw = w.sum(-1)
    zz = np.where(sw > 1e-12, (w * z).sum(-1) / np.maximum(sw, 1e-12), np.nan)
    neff = np.where(sw > 1e-12, sw ** 2 / np.maximum((w ** 2).sum(-1), 1e-12), 0.0)
    return zz, neff


gx = np.linspace(VMIN, VMAX, 30)
gy = np.linspace(DMIN, DMAX, 26)
X, Y = np.meshgrid(gx, gy)
groups = [("current-down  ($\\Delta I < 0$)", [r for r in rows if r["dI"] < 0]),
          ("current-up  ($\\Delta I > 0$)", [r for r in rows if r["dI"] > 0])]

fig = plt.figure(figsize=(13.2, 10.4))
last = None
for ri, (gl, sub) in enumerate(groups):
    x = np.abs([r["v"] for r in sub])
    y = np.abs([r["dI"] for r in sub])
    for ci, (key, ttl) in enumerate((("lag_true", "measured"), ("lag_pred", "E0 prediction"))):
        z = np.array([r[key] for r in sub], float)
        s, neff = nw(x, y, z, gx, gy)
        s[neff < SUPPORT] = np.nan
        ax = fig.add_subplot(2, 2, ri * 2 + ci + 1, projection="3d")
        last = ax.plot_surface(X, Y, np.clip(s, ZLO, ZHI), cmap="viridis", vmin=ZLO, vmax=ZHI,
                               linewidth=0.25, edgecolor="0.25", alpha=0.97,
                               antialiased=True, rstride=1, cstride=1)
        ax.scatter(x, y, np.clip(z, ZLO, ZHI), c="w", edgecolors="0.2", s=12,
                   linewidths=0.4, depthshade=False, alpha=0.85, zorder=5)
        n_out = int(((z < ZLO) | (z > ZHI)).sum())
        ax.set_zlim(ZLO, ZHI); ax.set_xlim(VMIN, VMAX); ax.set_ylim(DMIN, DMAX)
        ax.set_xlabel("|velocity| (mm/s)", labelpad=4, fontsize=9)
        ax.set_ylabel("|$\\Delta I$| (A)", labelpad=4, fontsize=9)
        ax.set_zlabel("10-90% lag (ms)", labelpad=4, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title("%s — %s\nn=%d%s" % (gl.split("  ")[0], ttl, len(sub),
                                          "  (%d clipped)" % n_out if n_out else ""),
                     fontsize=10)
        ax.view_init(elev=26, azim=-125)

cax = fig.add_axes([0.925, 0.30, 0.014, 0.42])
cb = fig.colorbar(last, cax=cax, ticks=np.arange(ZLO, ZHI + 1, 10))
cb.set_label("10-90% lag (ms)", fontsize=9)
cb.ax.tick_params(labelsize=8)
fig.suptitle("V22 E0 step lag over |velocity| x |$\\Delta I$|   "
             "(all 33 ST files, |$\\Delta F$| $\\geq$ 250 N, seed 7)\n"
             "E0 has no explicit current gain: this is how well its GRU/NFL path follows "
             "current steps; surface = kernel regression, dots = events",
             fontsize=11)
fig.subplots_adjust(left=0.03, right=0.90, top=0.87, bottom=0.04, wspace=0.06, hspace=0.22)
p = os.path.join(OUT, "E0_lag_surface_3d.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
print("[saved]", p)

print("\n格点摘要（实测 / E0）：")
for gl, sub in groups:
    if not sub:
        continue
    x = np.abs([r["v"] for r in sub]); y = np.abs([r["dI"] for r in sub])
    zt = np.array([r["lag_true"] for r in sub], float)
    zp = np.array([r["lag_pred"] for r in sub], float)
    m, _ = nw(x, y, zt, np.array([130.0, 260.0]), np.array([0.95, 1.27, 1.59]))
    q, _ = nw(x, y, zp, np.array([130.0, 260.0]), np.array([0.95, 1.27, 1.59]))
    print("  %s" % gl)
    for i, dd in enumerate((0.95, 1.27, 1.59)):
        print("    |dI|=%.2f  |v|=130: %5.1f / %5.1f   |v|=260: %5.1f / %5.1f"
              % (dd, m[i, 0], q[i, 0], m[i, 1], q[i, 1]))
