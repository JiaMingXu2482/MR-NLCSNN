"""How much force authority does the candidate-current path actually have?

For each model with a G branch, at every 100 Hz control step of an SH validation
file, evaluate F at I=0 and I=1.6 A with everything else fixed.  The gap
  authority = |F(1.6A) - F(0A)|
is the entire range the inverse solver can reach.  If the measured force falls
outside [F(0), F(1.6)] the solve saturates.
"""
import json
import os
import sys

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

SH = "SH_10mm_1.66Hz_1A_r1"
item = next(x for x in val if x["name"] == SH)
u, f, valid = sample(item, norm)
u = u.to(dev)
target = (f[0] * FS).cpu().numpy()
v_hat = (item["values"][:, 1] / 1000.0).numpy()
a_hat = (item["values"][:, 2] / 50000.0).numpy()

CK = {"EG": r"D:/v22_ae_hybrid/runs_seed7/EG",
      "EGT": r"D:/v22_ae_hybrid/runs_seed7/EGT",
      "AGT": r"D:/v22_ae_hybrid/runs_seed7/AGT",
      "EGTA_dis": r"D:/v22_ae_hybrid/runs_seed7/EGTA",
      "EGTA": r"D:/v22_ae_hybrid/runs_ext230/EGTA"}

print("文件 %s，%d 采样，目标力 %.0f .. %.0f N" % (SH, len(target), target.min(), target.max()))
print()
print("%-10s %12s %12s %12s %12s %10s" % (
    "模型", "F(0A)中位N", "F(1.6A)中位N", "权限中位N", "权限/力幅值", "达标率"))
res = {}
for tag, ck_dir in CK.items():
    mid = "EGTA" if tag.startswith("EGTA") else tag
    net = V22Model(mid)
    net.load_state_dict(torch.load(os.path.join(ck_dir, "best.pth"), map_location="cpu",
                                   weights_only=True)["state_dict"])
    net.to(dev).eval()
    z, obs = net.observe(u)
    with torch.no_grad():
        base, dgain = net.head_parts(z)
        g0 = net.current_map(torch.zeros(1, u.shape[1], device=dev))
        g1 = net.current_map(torch.ones(1, u.shape[1], device=dev))
        if net.use_tau:
            from v22_models import lag_scan
            q0 = lag_scan(g0, net.tau_s()); q1 = lag_scan(g1, net.tau_s())
        else:
            q0, q1 = g0, g1
        vv = obs[..., 1]
        f0 = (base + vv * dgain * q0)
        f1 = (base + vv * dgain * q1)
        if net.use_acc:
            acc = net.b_a * obs[..., 2]
            f0 = f0 + acc; f1 = f1 + acc
    F0 = (f0[0] * FS).cpu().numpy()
    F1 = (f1[0] * FS).cpu().numpy()
    idx = np.arange(0, len(F0), 10)
    auth = np.abs(F1[idx] - F0[idx])
    lo = np.minimum(F0[idx], F1[idx]); hi = np.maximum(F0[idx], F1[idx])
    ok = (target[idx] >= lo) & (target[idx] <= hi)
    scale = float(np.abs(target[idx]).mean())
    res[tag] = dict(authority_median=float(np.median(auth)),
                    authority_max=float(auth.max()),
                    reachable_frac=float(ok.mean()),
                    f0_median=float(np.median(F0[idx])), f1_median=float(np.median(F1[idx])),
                    target_median=float(np.median(target[idx])))
    print("%-10s %12.0f %12.0f %12.0f %12.2f %10.1f%%" % (
        tag, np.median(F0[idx]), np.median(F1[idx]), np.median(auth),
        np.median(auth) / max(scale, 1e-9), 100 * ok.mean()))

json.dump(res, open(r"D:/v22_ae_hybrid/inverse_authority.json", "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print()
print("权限/力幅值 = 电流可调范围的中位数 / 目标力的平均幅值；达标率 = 目标力落在可达范围内的控制步比例")
