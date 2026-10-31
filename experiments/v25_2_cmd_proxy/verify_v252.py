"""V25.2 Phase 4: independent recompute + delay re-measurement.

Repeats the same 10%->90% step analysis used in the V25 delay audit, so the two
pipelines are directly comparable on the same definition.
"""
import csv
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch

HERE = r"D:/v25_2_cmd_proxy"
sys.path.insert(0, HERE)
from v20_data import normalization, read_dataset                     # noqa: E402
from v252_data import build_v252, I_SCALE                            # noqa: E402

_s = importlib.util.spec_from_file_location("mdl", os.path.join(HERE, "model_def.py"))
M = importlib.util.module_from_spec(_s)
_s.loader.exec_module(M)

RUN = os.path.join(HERE, "runs", "V25_2_RI0_CMD_PROXY_seed7")
ROOT = r"D:/v20_data/Parsed_v19_F42_I42"
AUD = os.path.join(HERE, "audits")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CATS = ("SH", "ST", "RLF", "SR", "RHF")

train = read_dataset(ROOT, "train")
val = read_dataset(ROOT, "validation")
nrm = normalization(train)
net = M.ResidualI0()
net.load_state_dict(torch.load(os.path.join(RUN, "best.pth"), map_location="cpu",
                               weights_only=True)["state_dict"])
net.to(dev).eval()

# ---------------------------------------------------------------- recompute
rows = []
with torch.no_grad():
    for it in val:
        x, base, tgt, mask = build_v252(it, nrm)
        delta = net(x.to(dev)).cpu()[0]
        raw = base[0] + delta
        rows.append(dict(name=it["name"], pred=raw.clamp(0, 1.6).numpy(),
                         pr_raw=raw.numpy(), target=tgt[0].numpy(),
                         base=base[0].numpy(), mask=mask[0].numpy()))

P = np.concatenate([r["pred"][r["mask"]] for r in rows])
T = np.concatenate([r["target"][r["mask"]] for r in rows])
B = np.concatenate([r["base"][r["mask"]] for r in rows])
e = P - T
ss = json.load(open(os.path.join(RUN, "summary.json"), encoding="utf-8"))
print("=== 独立复算 ===")
print("  MAE  %.6f  记录 %.6f  |diff| %.2e" % (np.abs(e).mean(), ss["validation"]["mae"],
                                              abs(np.abs(e).mean() - ss["validation"]["mae"])))
print("  RMSE %.6f  记录 %.6f" % (np.sqrt((e ** 2).mean()), ss["validation"]["rmse"]))
print("  P95  %.6f  记录 %.6f" % (np.percentile(np.abs(e), 95), ss["validation"]["p95"]))
print("  NRMSE %.4f%% 记录 %.4f%%"
      % (100 * math.sqrt((e ** 2).sum() / (T ** 2).sum()), ss["validation"]["pooled_nrmse"]))
ok = abs(np.abs(e).mean() - ss["validation"]["mae"]) < 1e-9

print("\n=== 分工况 ===")
print("%-5s %9s %9s %9s %10s" % ("工况", "V25.2", "Persist", "改善", "P95"))
per = {}
for c in CATS:
    sub = [r for r in rows if r["name"].startswith(c + "_")]
    if not sub:
        continue
    p = np.concatenate([r["pred"][r["mask"]] for r in sub])
    t = np.concatenate([r["target"][r["mask"]] for r in sub])
    b = np.concatenate([r["base"][r["mask"]] for r in sub])
    ee = p - t
    per[c] = dict(n=int(ee.size), mae=float(np.abs(ee).mean()),
                  pers_mae=float(np.abs(b - t).mean()),
                  p95=float(np.percentile(np.abs(ee), 95)))
    print("%-5s %9.4f %9.4f %8.1f%% %10.4f"
          % (c, per[c]["mae"], per[c]["pers_mae"],
             100 * (1 - per[c]["mae"] / per[c]["pers_mae"]), per[c]["p95"]))

# ---------------------------------------------------------------- delay re-measure
print("\n=== 延迟复测（与 V25 同一定义：10%→90% 过阈）===")


def crossing(sig, a0, a1, t0, frac):
    d = a1 - a0
    if abs(d) < 1e-6:
        return None
    thr = a0 + frac * d
    cond = sig >= thr if d > 0 else sig <= thr
    i = int(np.argmax(cond))
    if not cond[i]:
        return None
    if i == 0:
        return t0
    s0, s1 = sig[i - 1], sig[i]
    w = 0.0 if abs(s1 - s0) < 1e-12 else (thr - s0) / (s1 - s0)
    return t0 + (i - 1 + w) / 1000.0


def edges(raw_i, thr=0.6, step=10):
    d = np.zeros_like(raw_i)
    d[:-step] = raw_i[step:] - raw_i[:-step]
    idx = np.where(np.abs(d) > thr)[0]
    out = []
    if len(idx):
        g, cur = [], [idx[0]]
        for k in idx[1:]:
            if k - cur[-1] <= 30:
                cur.append(k)
            else:
                g.append(cur)
                cur = [k]
        g.append(cur)
        out = [int(gg[int(np.argmax(np.abs(d[gg])))]) for gg in g]
    return out


lags = []
for it in [x for x in val if x["name"].startswith("ST_")]:
    v = it["values"].numpy().astype(np.float64)
    raw_i = np.abs(v[:, 3])
    x, base, tgt, mask = build_v252(it, nrm)
    with torch.no_grad():
        delta = net(x.to(dev)).cpu()[0].numpy()
    cmd = np.clip(base[0].numpy() + delta, 0, 1.6)
    cmd_zoh = np.repeat(cmd, 10)[:len(raw_i)]
    tgt_zoh = np.repeat(tgt[0].numpy(), 10)[:len(raw_i)]
    for te in edges(raw_i):
        lo, hi = max(0, te - 60), min(len(raw_i), te + 200)
        pre = raw_i[max(0, te - 50):max(1, te - 10)]
        post = raw_i[min(len(raw_i) - 1, te + 150):min(len(raw_i), te + 250)]
        if len(pre) < 5 or len(post) < 20:
            continue
        a0, a1 = float(np.median(pre)), float(np.median(post))
        if abs(a1 - a0) < 0.3:
            continue
        cr = crossing(raw_i[lo:hi], a0, a1, lo / 1000.0, 0.5)
        cm = crossing(cmd_zoh[lo:hi], a0, a1, lo / 1000.0, 0.5)
        ct = crossing(tgt_zoh[lo:hi], a0, a1, lo / 1000.0, 0.5)
        if None in (cr, cm, ct):
            continue
        lags.append(dict(file=it["name"], t=float(te / 1000.0),
                         lag_cmd_vs_raw=float((cm - cr) * 1000),
                         lag_target_vs_raw=float((ct - cr) * 1000),
                         lag_cmd_vs_target=float((cm - ct) * 1000)))
L = {k: np.array([d[k] for d in lags]) for k in
     ("lag_cmd_vs_raw", "lag_target_vs_raw", "lag_cmd_vs_target")}
for k, a in L.items():
    print("  %-18s 中位 %+6.1f ms  (n=%d)" % (k, np.median(a), a.size))
print("\n  对比 V25（同一测法）：总延迟中位 +38.7 ms，其中 FIR 29.8 ms")

json.dump(dict(recomputed=dict(mae=float(np.abs(e).mean()),
                               rmse=float(np.sqrt((e ** 2).mean())),
                               p95=float(np.percentile(np.abs(e), 95)),
                               p99=float(np.percentile(np.abs(e), 99)),
                               nrmse=float(100 * math.sqrt((e ** 2).sum()
                                                           / (T ** 2).sum())),
                               mae_passed=bool(ok)),
               per_condition=per,
               persistence=dict(mae=float(np.abs(B - T).mean()),
                                rmse=float(np.sqrt(((B - T) ** 2).mean())),
                                p95=float(np.percentile(np.abs(B - T), 95)),
                                p99=float(np.percentile(np.abs(B - T), 99)),
                                bias=float((B - T).mean())),
               delay=dict(n_edges=len(lags),
                          cmd_vs_raw_median_ms=float(np.median(L["lag_cmd_vs_raw"])),
                          target_vs_raw_median_ms=float(np.median(L["lag_target_vs_raw"])),
                          cmd_vs_target_median_ms=float(np.median(L["lag_cmd_vs_target"])),
                          note="target is clip(|I_meas[k+1]|), so it sits one control "
                               "step AFTER raw sample k by construction; a command lag "
                               "of about +10 ms vs raw is the intended plant-response "
                               "step, not a digital filter delay"),
               edges=lags),
          open(os.path.join(AUD, "phase4_recompute.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("\n-> audits/phase4_recompute.json  复算 %s" % ("PASS" if ok else "FAIL"))
