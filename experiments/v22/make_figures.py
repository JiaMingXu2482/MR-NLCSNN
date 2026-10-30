"""Generate the five per-model figures required by the V22 task book section 4:
  1. prediction time series          2. force-velocity cloud
  3. force-displacement loop         4. step-response overlay
  5. acceleration-coloured F-v plot
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = r"D:/v22_ae_hybrid"
sys.path.insert(0, HERE)
from v20_data import normalization, read_dataset, sample      # noqa: E402
from v22_models import MODELS, V22Model                        # noqa: E402
from train_v22 import evaluate, step_events                    # noqa: E402

ROOT = os.path.join(HERE, "runs_seed7")
DATA = r"D:/v20_data/Parsed_v19_F42_I42"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = read_dataset(DATA, "train")
val = read_dataset(DATA, "validation")
norm = normalization(train)


def load(m):
    ck = torch.load(os.path.join(ROOT, m, "best.pth"), map_location="cpu", weights_only=True)
    net = V22Model(m)
    net.load_state_dict(ck["state_dict"])
    return net.to(dev).eval()


for m in MODELS:
    net = load(m)
    _, keep = evaluate(net, val, norm, dev, keep=True)
    out = os.path.join(ROOT, m, "figures")
    os.makedirs(out, exist_ok=True)

    sh = next(k for k in keep if k.startswith("SH_") and "1.66Hz" in k)
    sr = next(k for k in keep if k.startswith("SR_"))
    st = next(k for k in keep if k.startswith("ST_"))
    d = keep[sh]
    v_ = d["valid"].astype(bool)

    # (1) prediction time series -------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 4))
    w = np.where(v_)[0][:6000]
    ax.plot(w / 1000.0, d["truth"][w], "k", lw=1.3, label="measured")
    ax.plot(w / 1000.0, d["pred"][w], color="#d62728", lw=1.1, alpha=0.85, label=m)
    ax.set_xlabel("time (s)"); ax.set_ylabel("force (N)")
    ax.set_title("%s — prediction vs measurement (%s)" % (m, sh))
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out, "1_prediction_timeseries.png"), dpi=140)
    plt.close(fig)

    # (2) F-v cloud ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(d["vel"][v_][::3], d["truth"][v_][::3], s=2, c="0.75", label="measured")
    ax.scatter(d["vel"][v_][::3], d["pred"][v_][::3], s=2, c="#d62728", alpha=0.45, label=m)
    ax.set_xlabel("velocity (mm/s)"); ax.set_ylabel("force (N)")
    ax.set_title("%s — force-velocity cloud" % m)
    ax.legend(markerscale=6); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out, "2_force_velocity_cloud.png"), dpi=140)
    plt.close(fig)

    # (3) F-x loop ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(d["pos"][v_], d["truth"][v_], "k", lw=1.0, label="measured")
    ax.plot(d["pos"][v_], d["pred"][v_], color="#d62728", lw=1.0, alpha=0.8, label=m)
    ax.set_xlabel("displacement (mm)"); ax.set_ylabel("force (N)")
    ax.set_title("%s — force-displacement loop" % m)
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out, "3_force_displacement_loop.png"), dpi=140)
    plt.close(fig)

    # (4) step response -----------------------------------------------------
    ev = [e for e in step_events(keep, 250.0) if e["file"] == st]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    dd = keep[st]
    if ev:
        e0 = max(ev, key=lambda e: abs(e["dF"]))
        a0 = int(round(e0["t_edge"] * 1000))
        w2 = np.arange(max(0, a0 - 250), min(len(dd["cur"]), a0 + 400))
        ax.plot((w2 - a0) / 1000.0, dd["truth"][w2], "k", lw=1.5, label="measured")
        ax.plot((w2 - a0) / 1000.0, dd["pred"][w2], color="#d62728", lw=1.3, label=m)
        ax.axvline(0, color="0.5", lw=0.9, ls=":")
        ax.set_title("%s — raw step response (shown event dF=%.0f N, %d qualifying events)\n"
                     "note: the reported lag metric is velocity-drift corrected, "
                     "this panel shows the raw force" % (m, e0["dF"], len(ev)),
                     fontsize=9.5)
    else:
        ax.text(0.5, 0.5, "no qualifying step event", ha="center", transform=ax.transAxes)
        ax.set_title("%s — step response" % m)
    ax.set_xlabel("time from current step (s)"); ax.set_ylabel("force (N)")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out, "4_step_response.png"), dpi=140)
    plt.close(fig)

    # (5) acceleration-coloured F-v -----------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    e = np.abs(d["pred"][v_] - d["truth"][v_])
    sc = ax.scatter(d["vel"][v_][::3], d["truth"][v_][::3], c=d["acc"][v_][::3], s=3,
                    cmap="coolwarm")
    fig.colorbar(sc, ax=ax, label="acceleration (mm/s^2)")
    ax.set_xlabel("velocity (mm/s)"); ax.set_ylabel("force (N)")
    ax.set_title("%s — measured F-v coloured by acceleration (mean |err| = %.1f N)"
                 % (m, e.mean()))
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out, "5_fv_acceleration_coloured.png"), dpi=140)
    plt.close(fig)

    print("%-5s figures -> %s" % (m, out), flush=True)

print("done")
