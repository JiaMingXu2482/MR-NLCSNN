"""Validation figures for E0, one per condition family (SH/ST/SR/RLF/RHF).

Each panel is measured vs predicted force over the valid window of one validation
file, with that file's NRMSE in the title.  Uses the 230-epoch E0 checkpoint.
final_test is never touched.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch

HERE = r"D:/v22_ae_hybrid"
sys.path.insert(0, HERE)
from v20_data import normalization, read_dataset, sample     # noqa: E402
from v22_models import V22Model                              # noqa: E402
from train_v22 import evaluate                               # noqa: E402

CKPT = r"D:/v22_ae_hybrid/runs_ext230/E0"
DATA = r"D:/v20_data/Parsed_v19_F42_I42"
OUT = r"D:/v22_ae_hybrid/fig_E0_validation"
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = read_dataset(DATA, "train")
val = read_dataset(DATA, "validation")
norm = normalization(train)

net = V22Model("E0")
net.load_state_dict(torch.load(os.path.join(CKPT, "best.pth"), map_location="cpu",
                               weights_only=True)["state_dict"])
net.to(dev).eval()
final, keep = evaluate(net, val, norm, dev, keep=True)
pct = {r["name"]: r["nrmse"] for r in final["rows"]}

FAMS = [("SH", "SH_", "正弦稳态 SH"),
        ("ST", "ST_", "阶跃/三角电流 ST"),
        ("SR", "SR_", "零电流随机位移 SR"),
        ("RLF", "RLF_", "低频随机电流 RLF"),
        ("RHF", "RHF_", "高频随机电流 RHF")]

summary = {}
for tag, pre, label in FAMS:
    names = [n for n in keep if n.startswith(pre)]
    names.sort()
    show = names[:4]
    cols = 2 if len(show) > 1 else 1
    rows = max(1, (len(show) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 3.0 * rows), squeeze=False)
    for k, n in enumerate(show):
        ax = axes[k // cols][k % cols]
        d = keep[n]
        m = d["valid"].astype(bool)
        idx = np.where(m)[0]
        step = max(1, len(idx) // 4000)
        ii = idx[::step]
        ax.plot(ii / 1000.0, d["truth"][ii], "k", lw=1.1, label="measured")
        ax.plot(ii / 1000.0, d["pred"][ii], color="#d62728", lw=1.0, alpha=0.85, label="E0")
        ax.set_title("%s   NRMSE %.1f%%" % (n.replace("_r", " r"), pct[n]), fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        ax.set_ylabel("force (N)", fontsize=8)
        if k // cols == rows - 1:
            ax.set_xlabel("time (s)", fontsize=8)
    for k in range(len(show), rows * cols):
        axes[k // cols][k % cols].set_visible(False)
    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle("E0 (230 epoch) — %s  (%d/%d files shown, family pooled NRMSE %.2f%%)"
                 % (label, len(show), len(names), final["per_type"][pre]), fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, "E0_validation_%s.png" % tag)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    summary[tag] = dict(files=len(names), shown=len(show),
                        pooled_pct=final["per_type"][pre],
                        per_file={n: pct[n] for n in names})
    print("%-4s %d 文件, pooled %.2f%%  -> %s" % (tag, len(names), final["per_type"][pre], p),
          flush=True)

json.dump(dict(checkpoint=CKPT, pooled=final["pooled"], families=summary),
          open(os.path.join(OUT, "E0_validation_summary.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("\n整体 pooled %.3f%%  最差文件 %.2f%%"
      % (final["pooled"], final["worst"]))
