"""A vs E on random-excitation / random-displacement validation files.
Row 1: per-file NRMSE for ALL 5 random files (3-seed mean +- sd).
Row 2: time series (seed 7).  Row 3: signed error vs truth, same window."""
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

M = json.load(open(os.path.join(RES, "a_vs_e_random_metrics.json"), encoding="utf-8"))
NPZ = np.load(os.path.join(RES, "a_vs_e_random_preds.npz"))
SEEDS = (7, 17, 27)

# all five random-excitation validation files (SR = pure random displacement, 0 A)
BARS = [
    ("SR_RMS10mm_1.66Hz_0A", "SR 随机位移\n(0 A 电流)"),
    ("RLF_RMS10mm_2Hz_r1", "RLF 10mm 2Hz"),
    ("RLF_RMS10mm_4Hz", "RLF 10mm 4Hz"),
    ("RLF_RMS15mm_3Hz", "RLF 15mm 3Hz"),
    ("RHF_RMS10mm_4Hz", "RHF 10mm 4Hz"),
]
# three of them get time-series panels
TS = [
    ("SR_RMS10mm_1.66Hz_0A", 2.0, "SR 随机位移 · 0 A"),
    ("RLF_RMS10mm_2Hz_r1", 2.0, "RLF 10mm 2Hz · 随机电流"),
    ("RHF_RMS10mm_4Hz", 1.0, "RHF 10mm 4Hz · 随机电流"),
]
COL = {"A": "#2E6FB7", "E": "#D97706"}
LAB = {"A": "A  原始输入 · GRU16", "E": "E  NFL 输入 · GRU8"}

fig = plt.figure(figsize=(17.0, 14.2))
gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 0.8],
              hspace=0.55, wspace=0.22, left=0.058, right=0.985, top=0.905, bottom=0.085)

# ---------- row 0: per-file NRMSE, all five files ----------
ax = fig.add_subplot(gs[0, :])
xb = np.arange(len(BARS))
w = 0.36
for k, model in enumerate(("A", "E")):
    mus = [np.mean([M[f"{model}|{s}"][f[0]]["nrmse_pct"] for s in SEEDS]) for f in BARS]
    sds = [np.std([M[f"{model}|{s}"][f[0]]["nrmse_pct"] for s in SEEDS], ddof=1) for f in BARS]
    bars = ax.bar(xb + (k - 0.5) * w, mus, w, yerr=sds, capsize=4, color=COL[model],
                  label=LAB[model], edgecolor="white", linewidth=0.8,
                  error_kw=dict(ecolor="#444", lw=1.2))
    for b, mu, sd in zip(bars, mus, sds):
        ax.text(b.get_x() + b.get_width() / 2, mu + sd + 0.8, "%.1f" % mu,
                ha="center", va="bottom", fontsize=9.5, color=COL[model], fontweight="bold")
ax.set_xticks(xb)
ax.set_xticklabels([f[1] for f in BARS], fontsize=10.5)
ax.set_xlim(-0.6, len(BARS) - 0.4)
ax.set_ylabel("NRMSE (%)  越低越好", fontsize=11)
ax.set_title("随机激励 / 随机位移工况：逐文件归一化 RMSE（5 个文件，3 seed 均值 ± 标准差）",
             fontsize=14, pad=10, fontweight="bold")
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.set_axisbelow(True)
ax.set_ylim(0, 45)
ax.legend(fontsize=11.5, frameon=False, ncol=2, loc="upper right")

# ---------- rows 1-2 ----------
for idx, (name, dur, subtitle) in enumerate(TS):
    truth = NPZ[f"A|{name}|truth"]
    valid = NPZ[f"A|{name}|valid"].astype(bool)
    pa, pe = NPZ[f"A|{name}|pred"], NPZ[f"E|{name}|pred"]
    n = int(dur * 1000)
    sl = slice(2000, 2000 + n)
    t = np.arange(n) / 1000.0
    m = valid[sl]

    ax = fig.add_subplot(gs[1, idx])
    ax.plot(t, np.where(m, truth[sl], np.nan), color="#222", lw=2.6, label="实测", zorder=3)
    ax.plot(t, np.where(m, pa[sl], np.nan), color=COL["A"], lw=1.6, label="A", zorder=4)
    ax.plot(t, np.where(m, pe[sl], np.nan), color=COL["E"], lw=1.6, ls="--", label="E", zorder=5)
    ax.set_title(subtitle, fontsize=12, fontweight="bold")
    ax.set_ylabel("阻尼力 (N)", fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(labelbottom=False)

    ax2 = fig.add_subplot(gs[2, idx])
    ea, ee = (pa - truth)[sl], (pe - truth)[sl]
    ax2.plot(t, np.where(m, ea, np.nan), color=COL["A"], lw=1.4, label="A − 实测")
    ax2.plot(t, np.where(m, ee, np.nan), color=COL["E"], lw=1.4, ls="--", label="E − 实测")
    rms_a = float(np.sqrt((ea[m] ** 2).mean()))
    rms_e = float(np.sqrt((ee[m] ** 2).mean()))
    ax2.axhline(0, color="#999", lw=0.9)
    for v, c in ((rms_a, COL["A"]), (-rms_a, COL["A"]), (rms_e, COL["E"]), (-rms_e, COL["E"])):
        ax2.axhline(v, color=c, lw=0.8, ls=":", alpha=0.85)
    ax2.set_title("本段 RMSE   A %.1f N   |   E %.1f N" % (rms_a, rms_e),
                  fontsize=10.5, color="#333")
    ax2.set_xlabel("时间 (s)", fontsize=10)
    ax2.set_ylabel("误差 (N)", fontsize=10)
    ax2.grid(alpha=0.22, linestyle="--")
    ax2.set_axisbelow(True)

handles, labels = [], []
for a in fig.axes[1:]:
    h, l = a.get_legend_handles_labels()
    for hh, ll in zip(h, l):
        if ll not in labels:
            handles.append(hh)
            labels.append(ll)
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
           fontsize=11, bbox_to_anchor=(0.5, 0.008))
fig.suptitle("A 与 E 在随机激励/随机位移工况下的表现（曲线取 seed 7 的 best 模型，"
             "误差面板点线为 ±RMSE）", fontsize=15, fontweight="bold", y=0.958)
out = os.path.join(FIG, "A_vs_E_random_conditions.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved:", out)
