"""V25.2 figures."""
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = r"D:/v25_2_cmd_proxy"
sys.path.insert(0, HERE)
from v20_data import normalization, read_dataset                     # noqa: E402
from v252_data import build_v252                                     # noqa: E402

_s = importlib.util.spec_from_file_location("mdl", os.path.join(HERE, "model_def.py"))
M = importlib.util.module_from_spec(_s)
_s.loader.exec_module(M)

RUN = os.path.join(HERE, "runs", "V25_2_RI0_CMD_PROXY_seed7")
FIG = os.path.join(HERE, "figures")
ROOT = r"D:/v20_data/Parsed_v19_F42_I42"
CATS = ("SH", "ST", "RLF", "SR", "RHF")
train = read_dataset(ROOT, "train")
val = read_dataset(ROOT, "validation")
nrm = normalization(train)
net = M.ResidualI0()
net.load_state_dict(torch.load(os.path.join(RUN, "best.pth"), map_location="cpu",
                               weights_only=True)["state_dict"])
net.eval()

hist = [json.loads(l) for l in open(os.path.join(RUN, "metrics.jsonl"), encoding="utf-8")]
p4 = json.load(open(os.path.join(HERE, "audits", "phase4_recompute.json"), encoding="utf-8"))
per = p4["per_condition"]

# ---------------------------------------------------------------- 1 condition bars
fig, ax = plt.subplots(figsize=(9, 4.0))
x = np.arange(len(CATS))
w = 0.38
a = [per[c]["mae"] for c in CATS]
b = [per[c]["pers_mae"] for c in CATS]
ax.bar(x - w / 2, a, w, label="V25.2", color="#4c72b0")
ax.bar(x + w / 2, b, w, label="Persistence（保持上一拍指令）", color="#c44e52")
for i in range(len(CATS)):
    ax.text(i - w / 2, a[i], "%.4f" % a[i], ha="center", va="bottom", fontsize=7.5)
    ax.text(i + w / 2, b[i], "%.4f" % b[i], ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(CATS)
ax.set_ylabel("电流 MAE (A)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")
ax.set_title("V25.2 vs Persistence 分工况电流 MAE（整体 %.4f vs %.4f，−%.1f%%）"
             % (p4["recomputed"]["mae"], p4["persistence"]["mae"],
                100 * (1 - p4["recomputed"]["mae"] / p4["persistence"]["mae"])),
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "per_condition_vs_persistence.png"), dpi=150)
plt.close(fig)
print("-> figures/per_condition_vs_persistence.png")

# ---------------------------------------------------------------- 2 training curve
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
ep = [h["epoch"] for h in hist]
axs[0].plot(ep, [h["val_mae"] for h in hist], label="V25.2 val MAE")
axs[0].axhline(hist[0]["persistence_mae"], ls="--", color="#c44e52",
               label="Persistence MAE")
axs[0].set_xlabel("epoch")
axs[0].set_ylabel("电流 MAE (A)")
axs[0].legend(fontsize=8)
axs[0].grid(alpha=0.25)
axs[0].set_title("验证 MAE 训练曲线", fontsize=10)
axs[1].plot(ep, [h["val_nrmse"] for h in hist], color="#2ca02c")
axs[1].set_xlabel("epoch")
axs[1].set_ylabel("pooled NRMSE (%)")
axs[1].grid(alpha=0.25)
axs[1].set_title("验证 pooled NRMSE", fontsize=10)
fig.suptitle("V25.2 训练过程（180 轮，best_ep %d）" % json.load(
    open(os.path.join(RUN, "summary.json"), encoding="utf-8"))["best_epoch"], fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "training_curve.png"), dpi=150)
plt.close(fig)
print("-> figures/training_curve.png")

# ---------------------------------------------------------------- 3 ST edge
it = next(x for x in val if x["name"].startswith("ST_"))
v = it["values"].numpy().astype(np.float64)
raw_i = np.abs(v[:, 3])
x_, base, tgt, mask = build_v252(it, nrm)
with torch.no_grad():
    delta = net(x_)[0].numpy()
cmd = np.clip(base[0].numpy() + delta, 0, 1.6)
tgrid = np.arange(len(cmd)) * 10 / 1000.0
# pick the strongest step edge
d = np.zeros_like(raw_i)
d[:-10] = raw_i[10:] - raw_i[:-10]
te = int(np.argmax(np.abs(d)))
lo, hi = max(0, (te - 150)), min(len(raw_i), te + 250)

fig, ax = plt.subplots(figsize=(11, 4.0))
ax.plot(np.arange(lo, hi) / 1000.0, raw_i[lo:hi], color="black", lw=1.3,
        label="原始实测电流 (1 kHz)")
ax.plot(np.arange(lo, hi) / 1000.0, np.repeat(tgt[0].numpy(), 10)[lo:hi],
        color="#2ca02c", lw=1.2, ls="-", label="训练标签 = 代理指令 clip(|I[k+1]|)")
ax.step(tgrid, cmd, where="post", color="#d62728", lw=1.4,
        label="V25.2 输出 I_cmd（ZOH）")
ax.step(tgrid, base[0].numpy(), where="post", color="#7f7f7f", lw=1.0, ls="--",
        label="Persistence 基准 clip(|I[k]|)")
ax.axvline(te / 1000.0, color="gray", ls=":", lw=0.9)
ax.text(te / 1000.0, ax.get_ylim()[1] * 0.97, " 阶跃起始", color="gray", fontsize=8,
        va="top")
ax.set_xlim(lo / 1000.0, hi / 1000.0)
ax.set_xlabel("时间 (s)")
ax.set_ylabel("|I| (A)")
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(alpha=0.25)
ax.set_title("%s：阶跃处 实测 / 代理标签 / V25.2 指令 / Persistence（无 FIR）" % it["name"],
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "st_edge_proxy_vs_cmd.png"), dpi=150)
plt.close(fig)
print("-> figures/st_edge_proxy_vs_cmd.png")
