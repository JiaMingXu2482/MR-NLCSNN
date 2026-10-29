"""Rolling (time-local) lag between model prediction and MEASURED force on the
step-current validation files. This is the honest 'nonlinear delay' measure:
the delay is estimated per window, so it can vary with the operating point.

lag > 0  -> model output must be shifted forward, i.e. the model LAGS reality
lag < 0  -> model runs ahead of reality
lag = 0  -> timing matches the measurement

Reads results/st_delay_preds.npz (written by st_delay.py).
Writes results/st_rolling_lag.json for plotting.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = np.load(os.path.join(HERE, "results", "st_delay_preds.npz"))
OUT = os.path.join(HERE, "results", "st_rolling_lag.json")

ST = ["ST_10mm_Vel0.13_step5_0_1.6A_r1", "ST_10mm_Vel0.13_step5_0_1.6A_r2",
      "ST_10mm_Vel0.13_step5_0_1.6A_r3"]
WIN = 200          # ms
STEP = 25          # ms
MAXS = 30          # +- ms; beyond this it is a wrong prediction, not a delay
MIN_GAIN = 0.05    # only report a lag where shifting explains >=5% of the MSE


def rolling_lag(pred, truth, valid, cur, edges):
    n = len(truth)
    out = []
    for c in range(1000 + WIN // 2, n - 1000 - WIN // 2, STEP):
        lo, hi = c - WIN // 2, c + WIN // 2
        m = valid[lo:hi]
        if m.sum() < 50:
            continue
        mse0 = float(((pred[lo:hi] - truth[lo:hi])[m] ** 2).mean())
        best = (None, float("inf"))
        for s in range(-MAXS, MAXS + 1, 1):
            a = pred[lo + s:hi + s]
            b = truth[lo:hi]
            if len(a) != len(b):
                continue
            mse = float(((a - b)[m] ** 2).mean())
            if mse < best[1]:
                best = (s, mse)
        if best[0] is None:
            continue
        gain = 1.0 - best[1] / max(mse0, 1e-12)
        seg_i = cur[lo:hi]
        near = min(edges, key=lambda e: abs(e - c)) if edges else None
        out.append(dict(
            t=float(c) / 1000.0,
            lag_ms=int(best[0]),
            mse=best[1],
            mse0=mse0,
            gain=float(gain),
            meaningful=bool(gain >= MIN_GAIN),
            i_mean=float(np.abs(seg_i).mean()),
            i_max=float(np.abs(seg_i).max()),
            dt_edge_ms=float(c - near) if near is not None else None,
        ))
    return out


def detect_edges(i, th=0.04, gap=40, guard=1000):
    hit = np.where(np.abs(np.diff(i)) > th)[0]
    if not len(hit):
        return []
    groups, a, b = [], hit[0], hit[0]
    for e in hit[1:]:
        if e - b <= gap:
            b = e
        else:
            groups.append((a, b))
            a = b = e
    groups.append((a, b))
    return [a for a, b in groups if a >= guard and b < len(i) - guard]


res = {}
for name in ST:
    truth = NPZ[f"{name}|truth"]
    valid = NPZ[f"{name}|valid"].astype(bool)
    cur = NPZ[f"{name}|I"]
    edges = detect_edges(cur)
    d = {}
    for m in ("A", "E"):
        d[m] = rolling_lag(NPZ[f"{name}|{m}"], truth, valid, cur, edges)
    res[name] = d

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1, ensure_ascii=False)

print("=== 每个文件：有意义时延的窗口占比与幅值 ===")
print("%-32s %-3s %5s %7s %9s %9s %8s" % ("file", "M", "n", "useful%", "median_ms", "p10", "p90"))
for name in ST:
    for m in ("A", "E"):
        rows = res[name][m]
        use = [r for r in rows if r["meaningful"]]
        v = np.array([r["lag_ms"] for r in use]) if use else np.array([0])
        print("%-32s %-3s %5d %6.1f%% %9d %9d %8d" % (
            name, m, len(rows), 100 * len(use) / len(rows),
            int(np.median(v)), int(np.percentile(v, 10)), int(np.percentile(v, 90))))

print()
print("=== 三个重复合并 ===")
for m in ("A", "E"):
    rows = [r for nm in ST for r in res[nm][m]]
    use = [r for r in rows if r["meaningful"]]
    v = np.array([r["lag_ms"] for r in use])
    allv = np.array([r["lag_ms"] for r in rows])
    print(" %s: 窗口 %d，其中平移有效 %d (%.1f%%)；有效窗时延 median=%d ms p10=%d p90=%d"
          % (m, len(rows), len(use), 100 * len(use) / len(rows),
             int(np.median(v)), int(np.percentile(v, 10)), int(np.percentile(v, 90))))
    print("    全部窗（含无效）median=%d ms，|lag|>10ms 的窗口 %d 个"
          % (int(np.median(allv)), int((np.abs(allv) > 10).sum())))

print()
print("=== 时延 vs 电流幅值（只用有效窗口，分箱中位数）===")
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 2.0]
for m in ("A", "E"):
    rows = [r for nm in ST for r in res[nm][m] if r["meaningful"]]
    xs = np.array([r["i_mean"] for r in rows])
    ys = np.array([r["lag_ms"] for r in rows])
    parts = []
    for lo, hi in zip(bins, bins[1:]):
        k = (xs >= lo) & (xs < hi)
        parts.append("%.1f-%.1f:%s(%d)" % (lo, hi, "%+.0f" % np.median(ys[k]) if k.sum() else "n/a", k.sum()))
    print(" %s  %s" % (m, "  ".join(parts)))
    print("    corr(lag,|I|) = %+.3f" % np.corrcoef(xs, ys)[0, 1])
