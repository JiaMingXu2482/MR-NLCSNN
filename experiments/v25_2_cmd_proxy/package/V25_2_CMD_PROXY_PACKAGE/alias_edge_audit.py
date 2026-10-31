"""V25.2 alias audit supplement: edge-local aliasing, and what dropping the FIR does.

Whole-file energy fractions understate transient aliasing, because a step edge's
energy is tiny next to a long low-frequency record. This measures the >50 Hz
fraction inside short windows around every detected current step, and quantifies
how much the 100 Hz label actually changes when the FIR is removed.
"""
import json
import os
import sys

import numpy as np

HERE = r"D:/v25_2_cmd_proxy"
sys.path.insert(0, HERE)
import v23_data as D23                                              # noqa: E402
from v20_data import normalization, read_dataset                     # noqa: E402

ROOT = r"D:/v20_data/Parsed_v19_F42_I42"
AUD = os.path.join(HERE, "audits")
train = read_dataset(ROOT, "train")
val = read_dataset(ROOT, "validation")
nrm = normalization(train)
FAMS = ["SH", "ST", "RLF", "SR", "RHF"]


def frac_above(sig, fs=1000.0, cut=50.0):
    s = np.asarray(sig, dtype=np.float64)
    s = s - s.mean()
    if s.size < 32 or np.ptp(s) < 1e-12:
        return None
    w = np.hanning(s.size)
    P = np.abs(np.fft.rfft(s * w)) ** 2
    f = np.fft.rfftfreq(s.size, 1.0 / fs)
    tot = P.sum()
    return float(P[f > cut].sum() / max(tot, 1e-30))


def edges(raw_i, thr=0.5, step=10):
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
        out = [int(gg[int(np.argmax(np.abs(d[gg])))] ) for gg in g]
    return out


res = {}
print("=== 边沿局部混叠（±100 ms 窗内 >50 Hz 能量占比）===")
for c in FAMS:
    win_i, win_f, diffs = [], [], []
    for it in [x for x in train if x["name"].startswith(c + "_")]:
        v = it["values"].numpy().astype(np.float64)
        raw_i = np.abs(v[:, 3])
        for te in edges(raw_i):
            lo, hi = max(0, te - 100), min(len(v), te + 100)
            if hi - lo < 64:
                continue
            a = frac_above(v[lo:hi, 3])
            b = frac_above(v[lo:hi, 5])
            if a is not None:
                win_i.append(a)
            if b is not None:
                win_f.append(b)
        # what dropping the FIR does to the 100 Hz label, at the same grid
        fir = np.stack([np.convolve(v[:, j], D23.H, mode="full")[:len(v)]
                        for j in range(6)], 1)
        d_direct = np.clip(np.abs(v[::10, 3]), 0, 1.6)
        d_fir = np.clip(np.abs(fir[::10, 3]), 0, 1.6)
        diffs.append(np.abs(d_direct - d_fir))
    din = np.concatenate(diffs) if diffs else np.array([0.0])
    res[c] = dict(
        n_edges=len(win_i),
        edge_local_I_above_50hz_median=float(np.median(win_i)) if win_i else None,
        edge_local_I_above_50hz_p90=float(np.percentile(win_i, 90)) if win_i else None,
        edge_local_F_above_50hz_median=float(np.median(win_f)) if win_f else None,
        label_shift_direct_vs_fir_A_median=float(np.median(din)),
        label_shift_direct_vs_fir_A_p95=float(np.percentile(din, 95)),
        label_shift_direct_vs_fir_A_max=float(din.max()))
    r = res[c]
    print("  %-4s 边沿 %3d | I>50Hz 中位 %6.2f%% P90 %6.2f%% | F>50Hz 中位 %6.2f%% | "
          "去掉FIR后标签变化 中位 %.4f A P95 %.4f"
          % (c, r["n_edges"], 100 * (r["edge_local_I_above_50hz_median"] or 0),
             100 * (r["edge_local_I_above_50hz_p90"] or 0),
             100 * (r["edge_local_F_above_50hz_median"] or 0),
             r["label_shift_direct_vs_fir_A_median"],
             r["label_shift_direct_vs_fir_A_p95"]))

worst = max((res[c]["edge_local_I_above_50hz_median"] or 0) for c in FAMS)
print("\n最差边沿局部混叠（I）: %.2f%%（中位）" % (100 * worst))
print("判据：若边沿局部 >50 Hz 能量中位超过 20%%，视为混叠风险不可接受。")
print("结论: %s" % ("可接受" if worst < 0.20 else "不可接受 —— 必须停止并报告"))

old = json.load(open(os.path.join(AUD, "alias_energy_audit.json"), encoding="utf-8"))
old["edge_local"] = res
old["edge_local_worst_median_frac"] = float(worst)
old["acceptance_criterion"] = ("edge-local >50 Hz energy median must stay below 20%; "
                               "whole-file averages are reported too but understate "
                               "transient aliasing")
old["verdict"] = "ACCEPTABLE" if worst < 0.20 else "UNACCEPTABLE"
json.dump(old, open(os.path.join(AUD, "alias_energy_audit.json"), "w",
                    encoding="utf-8"), indent=2, ensure_ascii=False)
print("-> audits/alias_energy_audit.json (updated)")
