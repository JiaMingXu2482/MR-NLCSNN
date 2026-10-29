"""Effective current->force delay on the ST (step-current) validation files,
compared between measurement, model A and model E.

Method: near a current transition the velocity is quasi-static (|dv/dt| gives
~4 mm/s change over 50 ms vs 110-160 mm/s), so the force step is driven by the
current step. The delay is the difference in 50%-crossing times of the current
step and the force step, both measured on the same window.

Outputs analysis/results/st_delay.json and st_delay_preds.npz for plotting.
"""
import csv
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
from v20_data import normalization, read_dataset, sample   # noqa: E402
from v20_models import V20Model                              # noqa: E402

DATA = r"D:/v20_data/Parsed_v19_F42_I42"
RUNS = {7: r"D:/v20_data/v20_runs_20261029", 17: r"D:/v20_data/v20_seed17",
        27: r"D:/v20_data/v20_seed27"}
ST = ["ST_10mm_Vel0.13_step5_0_1.6A_r1", "ST_10mm_Vel0.13_step5_0_1.6A_r2",
      "ST_10mm_Vel0.13_step5_0_1.6A_r3"]
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val = read_dataset(DATA, "validation")
norm = normalization(read_dataset(DATA, "train"))
fs = norm["f_scale"]
items = {it["name"]: it for it in val}


def predict(model, it):
    ck = torch.load(os.path.join(RUNS[7], model, "best.pth"), map_location="cpu", weights_only=True)
    m = V20Model(arch=ck["arch"])
    m.load_state_dict(ck["state_dict"])
    m.to(device).eval()
    u, f, valid = sample(it, norm)
    with torch.no_grad():
        p = m(u.to(device))
    return (p[0] * fs).cpu().numpy(), (f[0] * fs).cpu().numpy(), valid[0].numpy()


def detect_edges(i, th=0.04, gap=40, guard=1000):
    hit = np.where(np.abs(np.diff(i)) > th)[0]
    out = []
    if not len(hit):
        return out
    a = b = hit[0]
    for e in hit[1:]:
        if e - b <= gap:
            b = e
        else:
            out.append((a, b))
            a = b = e
    out.append((a, b))
    return [(a, b) for a, b in out if a >= guard and b < len(i) - guard]


def t50(sig, lo, hi, before, after):
    """first index in [lo,hi] where sig crosses the midpoint toward `after`."""
    mid = 0.5 * (before + after)
    seg = sig[lo:hi]
    if after > before:
        idx = np.where(seg >= mid)[0]
    else:
        idx = np.where(seg <= mid)[0]
    return lo + int(idx[0]) if len(idx) else None


def delays(curve, i, edges):
    res = []
    for a, b in edges:
        ib = float(np.mean(i[max(0, a - 30):a]))
        ia = float(np.mean(i[b + 30:b + 80]))
        fb = float(np.mean(curve[max(0, a - 30):a]))
        fa = float(np.mean(curve[b + 30:b + 80]))
        if abs(ia - ib) < 0.3 or abs(fa - fb) < 30:
            continue
        ti = t50(i, a - 5, b + 60, ib, ia)
        tf = t50(curve, a - 5, a + 120, fb, fa)
        if ti is None or tf is None:
            continue
        res.append(dict(t_edge=float(a) / 1000.0, dI=float(ia - ib), I_after=float(ia),
                        F_before=float(fb), F_after=float(fa), dF=float(fa - fb),
                        t50_I=float(ti) / 1000.0, t50_F=float(tf) / 1000.0,
                        delay_ms=int(tf - ti)))
    return res


all_res, arrays = {}, {}
for name in ST:
    it = items[name]
    pa, truth, valid = predict("A", it)
    pe, _, _ = predict("E", it)
    v = it["values"][:, 1].numpy()
    i = it["values"][:, 3].numpy()
    edges = detect_edges(i)
    all_res[name] = dict(n_edges=len(edges),
                         truth=delays(truth, i, edges),
                         A=delays(pa, i, edges),
                         E=delays(pe, i, edges))
    arrays[f"{name}|truth"] = truth
    arrays[f"{name}|A"] = pa
    arrays[f"{name}|E"] = pe
    arrays[f"{name}|I"] = i
    arrays[f"{name}|v"] = v
    arrays[f"{name}|valid"] = valid
    print("%-32s edges=%2d  truth n=%2d  A n=%2d  E n=%2d" % (
        name, len(edges), len(all_res[name]["truth"]), len(all_res[name]["A"]),
        len(all_res[name]["E"])))

np.savez_compressed(os.path.join(OUT, "st_delay_preds.npz"), **arrays)
with open(os.path.join(OUT, "st_delay.json"), "w", encoding="utf-8") as fh:
    json.dump(all_res, fh, indent=2, ensure_ascii=False)

print()
print("=== 时延（ms），按跳变列出（文件 r1）===")
r = all_res[ST[0]]
print("%7s %7s %8s %10s %10s %10s" % ("t_s", "dI_A", "dF_N", "truth", "A", "E"))
for k in range(min(len(r["truth"]), len(r["A"]), len(r["E"]))):
    t, a, e = r["truth"][k], r["A"][k], r["E"][k]
    print("%7.3f %7.2f %8.1f %10d %10d %10d" % (t["t_edge"], t["dI"], t["dF"],
                                                t["delay_ms"], a["delay_ms"], e["delay_ms"]))
for lab in ("truth", "A", "E"):
    ds = [x["delay_ms"] for nm in ST for x in all_res[nm][lab]]
    print("%-6s n=%2d  mean=%6.1f ms  median=%5.1f  min=%d max=%d" % (
        lab, len(ds), np.mean(ds), np.median(ds), min(ds), max(ds)))
