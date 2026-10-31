"""Phase 2a — independent recompute of CDC_FWD_V22_E0 in a fresh process.

Reads the ORIGINAL training checkpoint; no training, no weight change.
Metric convention: pooled NRMSE = 100*sqrt(sum(e^2)/sum(y^2)), sst UNCENTRED,
in newtons, over the validation edge-guard mask — i.e. V22's evaluate() rules.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:/v22_ae_hybrid")
from v20_data import normalization, read_dataset, sample                 # noqa: E402
from v22_models import V22Model, count_params                             # noqa: E402

CK = r"D:/v22_ae_hybrid/runs_seed7/E0/best.pth"
ROOT = r"D:/v20_data/Parsed_v19_F42_I42"
OUT = r"D:/CDC_Model_Release/audits"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CATS = ("SH", "ST", "RLF", "SR", "RHF")

train = read_dataset(ROOT, "train")
val = read_dataset(ROOT, "validation")
norm = normalization(train)
net = V22Model("E0")
ck = torch.load(CK, map_location="cpu", weights_only=True)
net.load_state_dict(ck["state_dict"])
net.to(dev).eval()
fs = float(norm["f_scale"])

rows = []
with torch.no_grad():
    for it in val:
        u, f, valid = sample(it, norm)
        pred = net(u.to(dev))
        v = valid[0]
        e = (pred[0][v] * fs).cpu().numpy() - (f[0][v] * fs).numpy()
        y = (f[0][v] * fs).numpy()
        rows.append(dict(name=it["name"], n=int(v.sum()),
                         sse=float((e ** 2).sum()), sst=float((y ** 2).sum()),
                         rmse=float(np.sqrt((e ** 2).mean())),
                         mae=float(np.abs(e).mean())))

pooled = 100 * math.sqrt(sum(r["sse"] for r in rows) / sum(r["sst"] for r in rows))
per = {}
for c in CATS:
    sub = [r for r in rows if r["name"].startswith(c + "_")]
    if sub:
        per[c] = 100 * math.sqrt(sum(r["sse"] for r in sub) / sum(r["sst"] for r in sub))

RECORDED = 10.594997446967472
print("=== CDC_FWD_V22_E0 独立复算 ===")
print("  checkpoint  : %s" % CK)
print("  sha256      : %s" % hashlib.sha256(open(CK, "rb").read()).hexdigest())
print("  epoch saved : %s" % ck.get("epoch"))
print("  params      : %d %s" % (count_params(net)["total"], count_params(net)["groups"]))
print("  files       : %d validation" % len(rows))
print("  pooled NRMSE: %.6f%%   recorded %.6f%%   |diff| %.8f pp"
      % (pooled, RECORDED, abs(pooled - RECORDED)))
print("  分工况:", {k: round(v, 4) for k, v in per.items()})
print("  判定: %s" % ("PASS" if abs(pooled - RECORDED) < 0.001 else "FAIL"))

json.dump(dict(release_name="CDC_FWD_V22_E0", checkpoint=CK,
               sha256=hashlib.sha256(open(CK, "rb").read()).hexdigest(),
               epoch_in_checkpoint=ck.get("epoch"),
               params=count_params(net),
               n_validation_files=len(rows),
               pooled_nrmse=pooled, recorded_pooled_nrmse=RECORDED,
               abs_diff_pp=abs(pooled - RECORDED),
               per_condition=per,
               metric_convention="100*sqrt(sum y^2 uncentred), newtons, edge guard",
               verdict="PASS" if abs(pooled - RECORDED) < 0.001 else "FAIL",
               final_test_read=False,
               per_file=rows),
          open(os.path.join(OUT, "phase2_fwd_recompute.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("  -> audits/phase2_fwd_recompute.json")
