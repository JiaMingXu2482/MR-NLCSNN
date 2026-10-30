"""Unified final metrics for V22 in the user's required table format.

Columns (in order):
  pooled NRMSE | SH/ST/SR/RLF/RHF | worst file | 10->90% step lag error |
  F-v loop error | monotonicity violations | 100 Hz inverse P50/P95/P99 |
  parameters | forward inference time

Everything is recomputed from best.pth in this process so the table cannot drift
from the checkpoints.  Writes summary_unified.csv and V22_FINAL_REPORT.md.
final_test is never opened.
"""
import csv
import json
import math
import os
import statistics as st
import sys
import time

import numpy as np
import torch

HERE = r"D:/v22_ae_hybrid"
sys.path.insert(0, HERE)
from v20_data import normalization, read_dataset, sample            # noqa: E402
from v22_models import MODELS, V22Model, count_params               # noqa: E402
from train_v22 import evaluate, step_events, monotonicity           # noqa: E402

ROOT = os.path.join(HERE, "runs_seed7")
DATA = r"D:/v20_data/Parsed_v19_F42_I42"
CATS = ["SH_", "ST_", "SR_", "RLF_", "RHF_"]

train = read_dataset(DATA, "train")
val = read_dataset(DATA, "validation")
norm = normalization(train)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(model):
    ck = torch.load(os.path.join(ROOT, model, "best.pth"), map_location="cpu",
                    weights_only=True)
    net = V22Model(model)
    net.load_state_dict(ck["state_dict"])
    return net


def fv_loop_error(keep):
    """Closed-loop area in the FORCE-VELOCITY plane, relative error.

    Cycles are found on displacement mid-point up-crossings (same event set as
    the F-x hysteresis metric); the area is the line integral of F over v.
    """
    trap = getattr(np, "trapezoid", None) or np.trapz
    out = []
    for name, d in keep.items():
        if not name.startswith("SH_"):
            continue
        x, F, P = d["pos"], d["truth"], d["pred"]
        v = d["valid"].astype(bool)
        if v.sum() < 500:
            continue
        mid = 0.5 * (x[v].min() + x[v].max())
        s = np.sign(x - mid)
        cr = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
        for i in range(len(cr) - 1):
            a, b = cr[i], cr[i + 1]
            if b - a < 100:
                continue
            sl = slice(a, b + 1)
            at = float(trap(F[sl], d["vel"][sl]))
            ap = float(trap(P[sl], d["vel"][sl]))
            if abs(at) < 1e-9:
                continue
            out.append(abs(ap - at) / abs(at))
    return (st.mean(out) if out else None), len(out)


def inverse_timing(net, keep, items, dt=0.01, iters=24):
    """100 Hz bisection with FIXED observer state and pure trials; returns
    P50/P95/P99 of the per-control-step latency."""
    if net.model_id == "E0":
        return dict(status="NOT_APPLICABLE")
    net = net.to(dev)
    name = next((k for k in keep if k.startswith("SH_")), None)
    item = next((x for x in items if x["name"] == name), None)
    if item is None:
        return dict(status="SKIP")
    d = keep[name]
    v, a, T = d["vel"], d["acc"], d["truth"]
    idx = np.arange(0, len(v), int(round(dt * 1000)))
    u_full, _, _ = sample(item, norm)
    with torch.no_grad():
        z_full, _ = net.observe(u_full.to(dev))
    z_np = z_full[0].cpu().numpy()
    q_prev, lat, resid, sat = 1.0, [], [], 0
    for k in idx:
        z = torch.from_numpy(z_np[k]).float().view(1, 1, 16).to(dev)
        vh = torch.tensor([[float(v[k] / 1000.0)]], device=dev)
        ah = torch.tensor([[float(a[k] / 50000.0)]], device=dev)
        qp = torch.tensor([[q_prev]], device=dev)
        target = float(T[k])
        t0 = time.perf_counter()
        with torch.no_grad():
            flo = float(net.trial_force(z, vh, ah, qp, torch.tensor([[0.0]], device=dev))[0])
            fhi = float(net.trial_force(z, vh, ah, qp, torch.tensor([[1.6]], device=dev))[0])
        if not (min(flo, fhi) <= target <= max(flo, fhi)):
            sat += 1
            lat.append((time.perf_counter() - t0) * 1000)
            continue
        lo, hi = 0.0, 1.6
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            with torch.no_grad():
                fm, qm = net.trial_force(z, vh, ah, qp,
                                         torch.tensor([[mid]], device=dev))
            if (float(fm) > target) == (flo > fhi):
                lo, flo = mid, float(fm)
            else:
                hi = mid
        mid = 0.5 * (lo + hi)
        with torch.no_grad():
            fm, qm = net.trial_force(z, vh, ah, qp, torch.tensor([[mid]], device=dev))
        lat.append((time.perf_counter() - t0) * 1000)
        q_prev = float(qm)
        resid.append(abs(float(fm) - target))
    l = np.array(lat)
    return dict(status="PASS", n=int(len(idx)), n_saturated=int(sat),
                saturated_frac=float(sat / max(len(idx), 1)),
                p50_ms=float(np.percentile(l, 50)),
                p95_ms=float(np.percentile(l, 95)),
                p99_ms=float(np.percentile(l, 99)),
                mean_ms=float(l.mean()), residual_max_N=float(max(resid) if resid else 0),
                note="Python/torch reference, fixed observer state, 24 bisection steps; not ECU WCET")


def forward_time(net, items_n=4):
    """Amortised per-1kHz-step forward latency (full-file forward / samples)."""
    samples = sum(it["values"].shape[0] for it in val[:items_n])
    res = {}
    for tag, device in (("cuda", dev), ("cpu", torch.device("cpu"))):
        n = net.to(device).eval()
        us = [sample(it, norm)[0].to(device) for it in val[:items_n]]
        with torch.no_grad():
            n(us[0])                                   # warmup
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for u in us:
                n(u)
            if device.type == "cuda":
                torch.cuda.synchronize()
            el = time.perf_counter() - t0
        res[tag + "_ms_per_step"] = el * 1000 / samples
        net.to(torch.device("cpu"))
    return res


rows = []
print("计算中（含 100 Hz 逆解计时与推理耗时）...", flush=True)
for m in MODELS:
    net = load(m)
    final, keep = evaluate(net.to(dev), val, norm, dev, keep=True)
    net.to(torch.device("cpu"))
    ev250 = step_events(keep, 250.0)
    lag_err = st.median([e["lag_err"] for e in ev250]) if ev250 else None
    fv, ncyc = fv_loop_error(keep)
    mono = monotonicity(net) if net.use_g else dict(violations=None, comparisons=0)
    inv = inverse_timing(net, keep, val)
    ft = forward_time(net)
    s = json.load(open(os.path.join(ROOT, m, "summary.json"), encoding="utf-8"))
    p = final["per_type"]
    rows.append(dict(model=m, params=count_params(V22Model(m))["total"],
                     pooled=final["pooled"],
                     SH=p.get("SH_"), ST=p.get("ST_"), SR=p.get("SR_"),
                     RLF=p.get("RLF_"), RHF=p.get("RHF_"),
                     worst=final["worst"], step_lag_err=lag_err,
                     fv_loop_err=fv, fv_cycles=ncyc,
                     mono_viol=mono.get("violations"),
                     inv_p50=inv.get("p50_ms"), inv_p95=inv.get("p95_ms"),
                     inv_p99=inv.get("p99_ms"),
                     fwd_cuda_ms=ft["cuda_ms_per_step"],
                     fwd_cpu_ms=ft["cpu_ms_per_step"],
                     tau_ms=s.get("tau_ms"), b_a=s.get("b_a")))
    print("  %-5s pooled=%.3f  F-v=%.4f  invP50=%s  fwd=%.4f ms/step"
          % (m, final["pooled"], fv if fv else float("nan"),
             ("%.2f" % inv["p50_ms"]) if inv.get("p50_ms") else "n/a",
             ft["cuda_ms_per_step"]), flush=True)

hdr = ["模型", "Pooled NRMSE%", "SH%", "ST%", "SR%", "RLF%", "RHF%", "最差文件%",
       "10→90% 阶跃时延误差ms", "力-速度回线误差", "单调性违反", "逆解P50ms",
       "逆解P95ms", "逆解P99ms", "参数量", "推理耗时ms/step(CUDA)", "推理耗时ms/step(CPU)"]
key = ["model", "pooled", "SH", "ST", "SR", "RLF", "RHF", "worst", "step_lag_err",
       "fv_loop_err", "mono_viol", "inv_p50", "inv_p95", "inv_p99", "params",
       "fwd_cuda_ms", "fwd_cpu_ms"]
with open(os.path.join(ROOT, "summary_unified.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(hdr + ["tau_ms", "b_a", "fv_cycles"])
    for r in rows:
        w.writerow([("%.4f" % r[k]) if isinstance(r[k], float) else ("" if r[k] is None else r[k])
                    for k in key] + [r["tau_ms"], r["b_a"], r["fv_cycles"]])

print()
def fm(v, f="%.3f"):
    return "-" if v is None else f % v
print("%-5s %9s %6s %6s %6s %6s %6s %8s %9s %9s %6s %7s %7s %7s %6s %9s" % (
    "模型", "Pooled%", "SH", "ST", "SR", "RLF", "RHF", "最差%", "阶跃ms", "F-v回线",
    "单调", "P50", "P95", "P99", "参数", "μs/step"))
for r in rows:
    print("%-5s %9.3f %6.2f %6.2f %6.2f %6.2f %6.2f %8.2f %9s %9s %6s %7s %7s %7s %6d %9.1f" % (
        r["model"], r["pooled"], r["SH"], r["ST"], r["SR"], r["RLF"], r["RHF"],
        r["worst"], fm(r["step_lag_err"], "%+.1f"), fm(r["fv_loop_err"], "%.4f"),
        "-" if r["mono_viol"] is None else r["mono_viol"],
        fm(r["inv_p50"], "%.2f"), fm(r["inv_p95"], "%.2f"), fm(r["inv_p99"], "%.2f"),
        r["params"], r["fwd_cuda_ms"] * 1000))

json.dump(rows, open(os.path.join(ROOT, "summary_unified.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("\n[saved]", os.path.join(ROOT, "summary_unified.csv"))
