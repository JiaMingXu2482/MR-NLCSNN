"""A vs E nonlinear time-delay behaviour under current-step (ST) conditions,
compared against the measured force.

row 0 : two step transients (up / down) - current, measured force, model A, E
row 1 : rolling timing lag vs time + a thin current strip sharing the x axis
row 2 : lag vs current magnitude (nonlinear drift) and the lag histogram

A lag value is only shown where a time shift explains >=5% of that window's MSE
(otherwise "lag" is meaningless: the prediction is simply wrong, not late).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 150
rcParams["savefig.dpi"] = 150

NPZ = np.load(os.path.join(RES, "st_delay_preds.npz"))
LAG = json.load(open(os.path.join(RES, "st_rolling_lag.json"), encoding="utf-8"))
ST = ["ST_10mm_Vel0.13_step5_0_1.6A_r1", "ST_10mm_Vel0.13_step5_0_1.6A_r2",
      "ST_10mm_Vel0.13_step5_0_1.6A_r3"]
COL = {"A": "#2E6FB7", "E": "#D97706"}
LAB = {"A": "A 原始输入·GRU16", "E": "E NFL·GRU8"}
ICOL = "#8fb8d0"

fig = plt.figure(figsize=(17.0, 14.6))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.05, 0.95],
              hspace=0.42, wspace=0.20, left=0.058, right=0.975, top=0.908, bottom=0.072)

# ================= row 0: transients =================
name = ST[0]
I = NPZ[f"{name}|I"]
truth = NPZ[f"{name}|truth"]
pa, pe = NPZ[f"{name}|A"], NPZ[f"{name}|E"]
ZOOM = [(2.105, "上跳  dI = +1.29 A   v = +125 mm/s"),
        (2.568, "下跳  dI = −1.27 A   v = −117 mm/s")]
for k, (te, title) in enumerate(ZOOM):
    ax = fig.add_subplot(gs[0, k])
    c = int(te * 1000)
    lo, hi = c - 100, c + 200
    tt = np.arange(lo, hi) - c
    ax.plot(tt, truth[lo:hi], color="#111", lw=2.7, label="实测力", zorder=4)
    ax.plot(tt, pa[lo:hi], color=COL["A"], lw=1.7, label="A", zorder=5)
    ax.plot(tt, pe[lo:hi], color=COL["E"], lw=1.7, ls="--", label="E", zorder=6)
    ax.axvline(0, color="#999", ls=":", lw=1.2)
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.set_xlabel("相对电流跳变时刻 (ms)", fontsize=10)
    ax.set_ylabel("阻尼力 (N)", fontsize=10.5)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax2 = ax.twinx()
    ax2.step(tt, I[lo:hi], where="mid", color=ICOL, lw=1.4, label="电流")
    ax2.set_ylabel("电流 (A)", color="#5a92b5", fontsize=9.5)
    ax2.tick_params(axis="y", colors="#5a92b5", labelsize=9)
    ax2.set_ylim(-0.15, 1.95)
    if k == 0:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=9.5, frameon=False,
                  loc="upper left", ncol=2, columnspacing=1.0)

# ================= row 1: rolling lag + current strip =================
sub = gs[1, :].subgridspec(2, 1, height_ratios=[0.74, 0.26], hspace=0.10)
ax = fig.add_subplot(sub[0])
for m in ("A", "E"):
    rows = LAG[name][m]
    t = np.array([r["t"] for r in rows])
    lag = np.array([r["lag_ms"] for r in rows], float)
    ok = np.array([r["meaningful"] for r in rows])
    ax.plot(t, np.where(ok, lag, np.nan), color=COL[m], ls="none", marker="o", ms=2.8,
            label=LAB[m], alpha=0.92)
ax.axhline(0, color="#333", lw=1.2)
ax.axhspan(-2, 2, color="#999", alpha=0.22, zorder=0)
ax.set_ylim(-16, 16)
ax.set_xlim(1.0, 9.8)
ax.set_ylabel("对实测的时延 (ms)\n正 = 模型滞后", fontsize=10.5)
ax.set_title("沿整个文件的滚动时延（仅显示平移可解释 ≥5% 误差的窗口；灰带 = ±2 ms；"
             "超出 ±16 ms 的点被裁切显示）", fontsize=12, fontweight="bold")
ax.grid(alpha=0.25, linestyle="--")
ax.set_axisbelow(True)
ax.legend(fontsize=9.5, frameon=False, ncol=2, loc="upper right")
ax.tick_params(labelbottom=False)

axc = fig.add_subplot(sub[1], sharex=ax)
axc.step(np.arange(len(I)) / 1000.0, I, where="mid", color=ICOL, lw=1.3)
axc.set_ylabel("电流 (A)", fontsize=9.5, color="#4a7f9e")
axc.tick_params(axis="y", labelsize=9, colors="#4a7f9e")
axc.set_xlabel("时间 (s)", fontsize=10)
axc.grid(alpha=0.22, linestyle="--")
axc.set_axisbelow(True)

# ================= row 2 left: lag vs |I| =================
ax = fig.add_subplot(gs[2, 0])
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 2.0]
for k, m in enumerate(("A", "E")):
    rows = [r for nm in ST for r in LAG[nm][m] if r["meaningful"]]
    xs = np.array([r["i_mean"] for r in rows])
    ys = np.array([r["lag_ms"] for r in rows], float)
    jit = (np.random.RandomState(k).rand(len(xs)) - 0.5) * 0.05
    ax.scatter(xs + jit, ys, s=9, color=COL[m], alpha=0.28, linewidths=0, zorder=2)
    cx, med = [], []
    for a, b in zip(bins, bins[1:]):
        q = (xs >= a) & (xs < b)
        if q.sum() >= 5:
            cx.append(xs[q].mean())
            med.append(np.median(ys[q]))
    ax.plot(cx, med, "-o", color=COL[m], lw=2.4, ms=7, zorder=5,
            markeredgecolor="white", markeredgewidth=1.2, label=LAB[m] + " 中位数")
ax.axhline(0, color="#333", lw=1.2)
ax.axhspan(-2, 2, color="#999", alpha=0.22, zorder=0)
ax.set_ylim(-18, 12)
ax.set_xlabel("窗口内平均 |电流| (A)", fontsize=10)
ax.set_ylabel("时延 (ms)", fontsize=10.5)
ax.set_title("非线性时延：时延随电流漂移（散点 = 各窗口，线 = 分箱中位数）",
             fontsize=11.5, fontweight="bold")
ax.grid(alpha=0.25, linestyle="--")
ax.set_axisbelow(True)
ax.legend(fontsize=9.5, frameon=False, loc="lower left")

# ================= row 2 right: histogram =================
ax = fig.add_subplot(gs[2, 1])
edges = np.arange(-31, 32, 1)
tot = len(LAG[ST[0]]["A"]) * 3
for m in ("A", "E"):
    rows = [r for nm in ST for r in LAG[nm][m] if r["meaningful"]]
    ys = np.array([r["lag_ms"] for r in rows], float)
    ax.hist(ys, bins=edges, histtype="step", lw=1.9, color=COL[m],
            label="%s  n=%d（有效窗 %.0f%%）" % (LAB[m], len(ys), 100 * len(ys) / tot))
ax.axvline(0, color="#333", lw=1.2)
ax.axvspan(-2, 2, color="#999", alpha=0.22, zorder=0)
ax.set_xlabel("时延 (ms)；正 = 模型滞后于真实", fontsize=10)
ax.set_ylabel("窗口数", fontsize=10.5)
ax.set_title("时延分布：主体都落在 ±2 ms", fontsize=11.5, fontweight="bold")
ax.grid(alpha=0.25, linestyle="--")
ax.set_axisbelow(True)
ax.legend(fontsize=9, frameon=False, loc="upper left")
ax.set_xlim(-30, 30)
ax.set_yscale("log")
ax.text(0.98, 0.55, "纵轴对数", transform=ax.transAxes, ha="right", fontsize=8.5, color="#666")

fig.suptitle("电流阶跃（ST）工况下的非线性时延：A vs E 与实测对比"
             "（seed 7；3 个重复文件；ST 10 mm / Vel 0.13 / 0→1.6 A 阶梯）",
             fontsize=15.5, fontweight="bold", y=0.960)
out = os.path.join(FIG, "ST_nonlinear_delay_A_vs_E.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved:", out)
