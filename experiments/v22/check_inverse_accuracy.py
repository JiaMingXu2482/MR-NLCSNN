"""Corrected inverse-model accuracy for the models that have a current path.

Fixes the unit bug in train_v22.inverse_100hz: trial_force returns NORMALISED
force while the target was in newtons, so every step was flagged saturated.

Reports, at 100 Hz with fixed observer state and pure trials:
  * saturation fraction (target outside [F(0A), F(1.6A)])
  * residual after bisection, in newtons
  * recovered current vs the measured current actually used in the file
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = r"D:/v22_ae_hybrid"
sys.path.insert(0, HERE)
from v20_data import normalization, read_dataset, sample      # noqa: E402
from v22_models import V22Model                               # noqa: E402

DATA = r"D:/v20_data/Parsed_v19_F42_I42"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train = read_dataset(DATA, "train")
val = read_dataset(DATA, "validation")
norm = normalization(train)
FS = norm["f_scale"]

CK = {"EG": r"D:/v22_ae_hybrid/runs_seed7/EG",
      "EGT": r"D:/v22_ae_hybrid/runs_seed7/EGT",
      "AGT": r"D:/v22_ae_hybrid/runs_seed7/AGT",
      "EGTA": r"D:/v22_ae_hybrid/runs_ext230/EGTA"}
ITERS = 24


def solve(net, item, dt=0.01):
    u, f, valid = sample(item, norm)
    u = u.to(dev)
    truth_N = (f[0] * FS).cpu().numpy()
    cur_A = item["values"][:, 3].numpy()
    v = item["values"][:, 1].numpy()
    a = item["values"][:, 2].numpy()
    with torch.no_grad():
        z_full, obs = net.observe(u)
    z_np = z_full[0].cpu().numpy()
    idx = np.arange(0, len(truth_N), int(round(dt * 1000)))
    q_prev = 1.0
    sat = 0
    resid, lat, rec = [], [], []
    for k in idx:
        z = torch.from_numpy(z_np[k]).float().view(1, 1, 16).to(dev)
        vh = torch.tensor([[float(v[k] / 1000.0)]], device=dev)
        ah = torch.tensor([[float(a[k] / 50000.0)]], device=dev)
        qp = torch.tensor([[q_prev]], device=dev)
        tgt = float(truth_N[k]) / FS              # <-- normalised, matches trial_force
        t0 = time.perf_counter()
        with torch.no_grad():
            flo = float(net.trial_force(z, vh, ah, qp, torch.tensor([[0.0]], device=dev))[0])
            fhi = float(net.trial_force(z, vh, ah, qp, torch.tensor([[1.0]], device=dev))[0])
        if not (min(flo, fhi) <= tgt <= max(flo, fhi)):
            sat += 1
            lat.append((time.perf_counter() - t0) * 1000)
            continue
        lo, hi = 0.0, 1.0                          # normalised current range
        for _ in range(ITERS):
            mid = 0.5 * (lo + hi)
            with torch.no_grad():
                fm, qm = net.trial_force(z, vh, ah, qp, torch.tensor([[mid]], device=dev))
            if (float(fm) > tgt) == (flo > fhi):
                lo, flo = mid, float(fm)
            else:
                hi = mid
        mid = 0.5 * (lo + hi)
        with torch.no_grad():
            fm, qm = net.trial_force(z, vh, ah, qp, torch.tensor([[mid]], device=dev))
        lat.append((time.perf_counter() - t0) * 1000)
        q_prev = float(qm)
        resid.append(abs(float(fm) - tgt) * FS)
        rec.append(abs(mid * 1.6 - abs(float(cur_A[k]))))
    return dict(steps=int(len(idx)), n_saturated=sat,
                sat_frac=float(sat / max(len(idx), 1)),
                residual_median_N=float(np.median(resid)) if resid else None,
                residual_p95_N=float(np.percentile(resid, 95)) if resid else None,
                current_err_median_A=float(np.median(rec)) if rec else None,
                current_err_p95_A=float(np.percentile(rec, 95)) if rec else None,
                latency_p50_ms=float(np.percentile(lat, 50)),
                latency_p99_ms=float(np.percentile(lat, 99)))


print("100 Hz 逆解（固定 observer 状态、纯试算、24 次二分）\n")
SH = "SH_10mm_1.66Hz_1A_r1"
item = next(x for x in val if x["name"] == SH)
print("文件 %s，实测电流 %.2f .. %.2f A\n" % (SH,
      item["values"][:, 3].abs().min(), item["values"][:, 3].abs().max()))
print("%-6s %8s %10s %12s %12s %13s %11s" % (
    "模型", "控制步", "饱和率", "残差中位N", "残差P95N", "电流误差中位A", "P99ms"))
out = {}
for tag, d in CK.items():
    net = V22Model(tag)
    net.load_state_dict(torch.load(os.path.join(d, "best.pth"), map_location="cpu",
                                   weights_only=True)["state_dict"])
    net.to(dev).eval()
    r = solve(net, item)
    out[tag] = r
    print("%-6s %8d %9.1f%% %12s %12s %13s %11.2f" % (
        tag, r["steps"], 100 * r["sat_frac"],
        "%.2f" % r["residual_median_N"] if r["residual_median_N"] is not None else "-",
        "%.2f" % r["residual_p95_N"] if r["residual_p95_N"] is not None else "-",
        "%.4f" % r["current_err_median_A"] if r["current_err_median_A"] is not None else "-",
        r["latency_p99_ms"]))

json.dump(out, open(r"D:/v22_ae_hybrid/inverse_accuracy_fixed.json", "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("\n[saved] inverse_accuracy_fixed.json")
