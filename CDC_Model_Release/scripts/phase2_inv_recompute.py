"""Phase 2b — independent recompute of CDC_INV_V25_RI0 in a fresh process.

Reads the ORIGINAL training checkpoint; no training, no weight change.
Recomputes MAE / RMSE / P95 / P99, per-condition MAE, zero-current bias and
false turn-on, using the same validation mask as the training run.
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:/v25_residual")
from v20_data import normalization, read_dataset                        # noqa: E402
from v25_data import build_residual                                     # noqa: E402
from v25_models import ResidualI0, count_params                         # noqa: E402

CK = r"D:/v25_residual/runs/V25_RI0_RESIDUAL_seed7/best.pth"
ROOT = r"D:/v20_data/Parsed_v19_F42_I42"
OUT = r"D:/CDC_Model_Release/audits"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CATS = ("SH", "ST", "RLF", "SR", "RHF")

train = read_dataset(ROOT, "train")
val = read_dataset(ROOT, "validation")
norm = normalization(train)
net = ResidualI0()
ck = torch.load(CK, map_location="cpu", weights_only=True)
net.load_state_dict(ck["state_dict"])
net.to(dev).eval()

rows = []
with torch.no_grad():
    for it in val:
        x, y, base, mask = build_residual(it, norm)
        d = net(x.to(dev)).cpu()[0]
        raw = base[0] + d
        rows.append(dict(name=it["name"], pred=raw.clamp(0, 1.6).numpy(),
                         raw=raw.numpy(), y=y[0].numpy(),
                         base=base[0].numpy(), mask=mask[0].numpy()))

P = np.concatenate([r["pred"][r["mask"]] for r in rows])
R = np.concatenate([r["raw"][r["mask"]] for r in rows])
T = np.concatenate([r["y"][r["mask"]] for r in rows])
e = P - T
s = json.load(open(r"D:/v25_residual/runs/V25_RI0_RESIDUAL_seed7/summary.json",
                   encoding="utf-8"))["validation"]

print("=== CDC_INV_V25_RI0 独立复算 ===")
print("  checkpoint  : %s" % CK)
print("  sha256      : %s" % hashlib.sha256(open(CK, "rb").read()).hexdigest())
print("  epoch saved : %s" % ck.get("epoch"))
print("  params      : %d %s" % (count_params(net)["total"], count_params(net)["groups"]))
print("  samples     : %d" % e.size)
res = dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
           median=float(np.median(np.abs(e))),
           p95=float(np.percentile(np.abs(e), 95)),
           p99=float(np.percentile(np.abs(e), 99)), bias=float(e.mean()))
rec = dict(mae=s["mae"], rmse=s["rmse"], p95=s["p95"], p99=s["p99"])
for k in ("mae", "rmse", "p95", "p99"):
    print("  %-6s %.6f   recorded %.6f   |diff| %.8f" % (k, res[k], rec[k],
                                                          abs(res[k] - rec[k])))
ok = all(abs(res[k] - rec[k]) < 1e-6 for k in ("mae", "rmse", "p95", "p99"))

per = {}
print("  分工况 MAE:")
for c in CATS:
    sub = [r for r in rows if r["name"].startswith(c + "_")]
    if not sub:
        continue
    p = np.concatenate([r["pred"][r["mask"]] for r in sub])
    t = np.concatenate([r["y"][r["mask"]] for r in sub])
    per[c] = dict(n=int(p.size), mae=float(np.abs(p - t).mean()),
                  rmse=float(np.sqrt(((p - t) ** 2).mean())),
                  p95=float(np.percentile(np.abs(p - t), 95)),
                  bias=float((p - t).mean()))
    print("    %-4s n=%5d  MAE %.4f  RMSE %.4f  P95 %.4f  bias %+.4f"
          % (c, per[c]["n"], per[c]["mae"], per[c]["rmse"], per[c]["p95"],
             per[c]["bias"]))

z = T < 0.05
zero = dict(n=int(z.sum()), pred_mean=float(P[z].mean()),
            pred_median=float(np.median(P[z])), pred_p95=float(np.percentile(P[z], 95)),
            false_turn_on_005=float((P[z] > 0.05).mean()),
            above_010=float((P[z] > 0.10).mean()))
print("  零电流 (%d 点): mean %.4f  median %.4f  P95 %.4f  >0.05A %.2f%%"
      % (zero["n"], zero["pred_mean"], zero["pred_median"], zero["pred_p95"],
         100 * zero["false_turn_on_005"]))
clip_rate = float((np.abs(R - P) > 1e-9).mean())
oor = float(((R < 0) | (R > 1.6)).mean())
print("  raw 越界 %.2f%%  clip 触发 %.2f%%" % (100 * oor, 100 * clip_rate))
print("  判定: %s" % ("PASS" if ok else "FAIL"))

json.dump(dict(release_name="CDC_INV_V25_RI0", checkpoint=CK,
               sha256=hashlib.sha256(open(CK, "rb").read()).hexdigest(),
               epoch_in_checkpoint=ck.get("epoch"), params=count_params(net),
               n_samples=int(e.size), metrics=res, recorded=rec,
               per_condition=per, zero_current=zero,
               raw_out_of_range=oor, clip_activation=clip_rate,
               mask_note="validation edge guard + k=0 excluded (same mask as training)",
               verdict="PASS" if ok else "FAIL", final_test_read=False),
          open(os.path.join(OUT, "phase2_inv_recompute.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("  -> audits/phase2_inv_recompute.json")
