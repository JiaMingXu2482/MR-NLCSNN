"""Phase 0 audit for V22: manifest, split counts, sampling, and the G direction
check the task book requires before training.

Direction question: in matched-motion SH pairs that contain both 0 A and 1.6 A,
does the F-v slope DECREASE with current magnitude?  If it increases, the
spec-mandated g = 1 - G_pos(i) has the wrong sign and training must stop.
"""
import hashlib
import json
import os
from collections import Counter

import numpy as np
import torch

from v20_data import normalization, read_dataset

DATA = r"D:/v20_data/Parsed_v19_F42_I42"
EXPECT = "0766763e7fb77880ffa04f2d93bcf629bfeb817e51c571188eee1ea83262e2ff"
OUT = r"D:/v22_ae_hybrid/audit_v22_prerequisites.json"

raw = open(os.path.join(DATA, "alignment_manifest.json"), "rb").read()
h = hashlib.sha256(raw).hexdigest()
rows = json.loads(raw)
cnt = Counter(r["split"] for r in rows)
print("=== 1. manifest ===")
print("SHA256 : %s" % h)
print("期望    : %s" % EXPECT)
print("匹配    : %s" % ("是" if h == EXPECT else "否 —— 停止"))
print("条目    : %s" % dict(cnt))

train = read_dataset(DATA, "train")
val = read_dataset(DATA, "validation")
def fam(n):
    for p in ("SH_", "ST_", "RLF_", "RHF_", "SR_"):
        if n.startswith(p):
            return p[:2]
    return "??"
print()
print("train      : %d 文件  %s" % (len(train), dict(Counter(fam(i["name"]) for i in train))))
print("validation : %d 文件  %s" % (len(val), dict(Counter(fam(i["name"]) for i in val))))
print("(期望 train 259 = SH186/ST30/RLF18/SR17/RHF8 ; val 16 = SH8/ST3/RLF3/SR1/RHF1)")

ntrain = sum(i["values"].shape[0] for i in train)
nval = sum(i["values"].shape[0] for i in val)
print("样本数     : train %d, validation %d" % (ntrain, nval))

norm = normalization(train)
print("归一化     : f_scale=%.3f N  x_scale=%.2f mm  v_scale=%.0f  a_scale=%.0f  i_scale=%.1f"
      % (norm["f_scale"], norm["x_scale"], norm["v_scale"], norm["a_scale"], norm["i_scale"]))

print()
print("=== 2. G 方向审计（F-v 斜率 vs 电流幅值）===")
# group SH files by (aperture, frequency, phase) and find groups holding both 0 A and 1.6 A
def key_of(n):
    parts = n.split("_")
    try:
        return (parts[1], parts[2])          # aperture, frequency
    except IndexError:
        return None

groups = {}
for it in train + val:
    if not it["name"].startswith("SH_"):
        continue
    k = key_of(it["name"])
    if k is None:
        continue
    cur = float(it["values"][:, 3].abs().max())
    groups.setdefault(k, []).append((it, cur))

rows_out = []
for k, items in sorted(groups.items()):
    lo = [x for x in items if x[1] < 0.1]
    hi = [x for x in items if x[1] > 1.5]
    if not lo or not hi:
        continue
    for a_it, a_c in lo:
        for b_it, b_c in hi:
            def slope(it):
                v = it["values"]
                vv = v[:, 1].numpy()
                ff = v[:, 5].numpy()
                m = np.abs(vv) > 20.0
                if m.sum() < 200:
                    return None
                A = np.vstack([vv[m], np.zeros(m.sum())]).T
                return float(np.linalg.lstsq(A, ff[m], rcond=None)[0][0])
            s0, s1 = slope(a_it), slope(b_it)
            if s0 is None or s1 is None:
                continue
            rows_out.append(dict(group=list(k), low_name=a_it["name"], high_name=b_it["name"],
                                 low_max_A=a_c, high_max_A=b_c,
                                 slope_0A=s0, slope_16A=s1, decreases=s1 < s0))

print("匹配对（同孔径/频率，含 0 A 与 1.6 A 各一）: %d 组" % len(rows_out))
print("%-22s %-22s %10s %10s %s" % ("0 A 文件", "1.6 A 文件", "斜率@0A", "斜率@1.6A", "下降"))
for r in rows_out:
    print("%-22s %-22s %10.2f %10.2f %s" % (
        r["low_name"][:22], r["high_name"][:22], r["slope_0A"], r["slope_16A"],
        "是" if r["decreases"] else "**否**"))

n_dec = sum(1 for r in rows_out if r["decreases"])
print()
if rows_out:
    frac = n_dec / len(rows_out)
    print("斜率随电流下降的比例: %d/%d = %.0f%%" % (n_dec, len(rows_out), 100 * frac))
    verdict = "支持 g = 1 - G_pos(i)（g 随电流单调不增）" if frac >= 0.8 else \
        "**方向不一致，按任务书 §4 停止并报告**"
    print("判定:", verdict)
else:
    frac, verdict = None, "无匹配对，无法审计"

json.dump(dict(manifest_sha256=h, manifest_ok=h == EXPECT, split_counts=dict(cnt),
               n_train=len(train), n_validation=len(val),
               train_families=dict(Counter(fam(i["name"]) for i in train)),
               val_families=dict(Counter(fam(i["name"]) for i in val)),
               n_train_samples=ntrain, n_val_samples=nval, norm=norm,
               direction=dict(pairs=rows_out, n_pairs=len(rows_out),
                              n_decreasing=n_dec, fraction=frac, verdict=verdict)),
          open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("[saved]", OUT)
