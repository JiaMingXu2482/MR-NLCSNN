"""Phase 5a: independent load test — load ONLY from the release directory.

The model definitions come from each release folder's own model_def.py; the data
reader is the shared (non-model) preprocessing code. Verifies the released
weights reproduce the recorded metrics and the golden vectors.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

REL = r"D:/CDC_Model_Release"
FWD_DIR = os.path.join(REL, "Forward_Model", "CDC_FWD_V22_E0")
INV_DIR = os.path.join(REL, "Inverse_Model", "CDC_INV_V25_RI0")
ROOT = r"D:/v20_data/Parsed_v19_F42_I42"
OUT = os.path.join(REL, "audits")

sys.path.insert(0, r"D:/v25_residual")          # data adapters only (non-model)
sys.path.insert(0, r"D:/v22_ae_hybrid")
from v20_data import normalization, read_dataset, sample                     # noqa: E402

train = read_dataset(ROOT, "train")
val = read_dataset(ROOT, "validation")
norm = normalization(train)
fs = float(norm["f_scale"])
rec = {}

# ---------------------------------------------------------------- FWD
print("=== 从发布目录独立加载：CDC_FWD_V22_E0 ===")
sys.path.insert(0, FWD_DIR)
for m in list(sys.modules):
    if m in ("model_def", "v22_models"):
        del sys.modules[m]
from model_def import V22Model                                              # noqa: E402
print("  model_def 来源:", V22Model.__module__, sys.modules[V22Model.__module__].__file__)
net = V22Model("E0")
ck = torch.load(os.path.join(FWD_DIR, "best.pth"), map_location="cpu", weights_only=True)
net.load_state_dict(ck["state_dict"], strict=True)
net.eval()
sse = sst = 0.0
with torch.no_grad():
    for it in val:
        u, f, valid = sample(it, norm)
        pr = net(u)[0][valid[0]] * fs
        t = f[0][valid[0]] * fs
        sse += float(((pr - t) ** 2).sum())
        sst += float((t ** 2).sum())
pooled = 100 * math.sqrt(sse / sst)

g = np.load(os.path.join(FWD_DIR, "golden_vectors.npz"))
uin, exp = torch.from_numpy(g["input_raw"]).float()[None], g["expected_force_N"]
with torch.no_grad():
    got = (net(uin)[0] * fs).numpy()
gv = float(np.abs(got - exp).max())

rec["fwd"] = dict(release="CDC_FWD_V22_E0",
                  loaded_from=os.path.join(FWD_DIR, "best.pth"),
                  sha256=hashlib.sha256(open(os.path.join(FWD_DIR, "best.pth"),
                                             "rb").read()).hexdigest(),
                  strict_load=True, params=sum(p.numel() for p in net.parameters()),
                  pooled_nrmse=pooled, expected=10.594997446967472,
                  abs_diff_pp=abs(pooled - 10.594997446967472),
                  golden_max_abs_force_error_N=gv, golden_n=int(exp.size),
                  verdict=("PASS" if abs(pooled - 10.594997446967472) < 0.001
                           and gv < 1e-3 else "FAIL"))
print("  pooled %.6f%%  expected 10.594997%%  diff %.8f pp"
      % (pooled, rec["fwd"]["abs_diff_pp"]))
print("  golden vectors: n=%d  max|dF| = %.3e N" % (exp.size, gv))
print("  -> %s" % rec["fwd"]["verdict"])

# ---------------------------------------------------------------- INV
print("\n=== 从发布目录独立加载：CDC_INV_V25_RI0 ===")
for m in list(sys.modules):
    if m in ("model_def", "v25_models", "v25_data"):
        del sys.modules[m]
sys.path.insert(0, INV_DIR)
from model_def import ResidualI0                                            # noqa: E402
print("  model_def 来源:", ResidualI0.__module__,
      sys.modules[ResidualI0.__module__].__file__)
net2 = ResidualI0()
ck2 = torch.load(os.path.join(INV_DIR, "best.pth"), map_location="cpu", weights_only=True)
net2.load_state_dict(ck2["state_dict"], strict=True)
net2.eval()

# rebuild the inverse inputs with the archived feature code
sys.path.insert(0, INV_DIR)
import importlib.util
spec = importlib.util.spec_from_file_location("rel_feature_def",
                                              os.path.join(INV_DIR, "feature_def.py"))
fd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fd)

E = []
with torch.no_grad():
    for it in val:
        m_, c_, ysig, valid, _raw = fd.build_inv(it, norm)
        d = net2(torch.cat((m_, c_), -1))[0]
        y = (ysig.abs().clamp(0, 1)) * 1.6
        base = torch.cat((torch.zeros_like(y[:, :1]), y[:, :-1]), 1)
        mask = valid.clone(); mask[:, 0] = False
        pred = (base[0] + d).clamp(0, 1.6)
        E.append((pred[mask[0]].numpy() - y[0][mask[0]].numpy()))
e = np.concatenate(E)
mae = float(np.abs(e).mean())
rmse = float(np.sqrt((e ** 2).mean()))

gi = np.load(os.path.join(INV_DIR, "golden_vectors.npz"))
xi = torch.from_numpy(gi["input_features"]).float()[None]
basei = torch.from_numpy(gi["base_A"])
with torch.no_grad():
    di = net2(xi)[0]
    cmdi = (basei + di).clamp(0, 1.6)
gvd = float((di.numpy() - gi["expected_delta_A"]).__abs__().max())
gvc = float((cmdi.numpy() - gi["expected_I_cmd_A"]).__abs__().max())

rec["inv"] = dict(release="CDC_INV_V25_RI0",
                  loaded_from=os.path.join(INV_DIR, "best.pth"),
                  sha256=hashlib.sha256(open(os.path.join(INV_DIR, "best.pth"),
                                             "rb").read()).hexdigest(),
                  strict_load=True, params=sum(p.numel() for p in net2.parameters()),
                  mae=mae, rmse=rmse, expected_mae=0.0300069,
                  abs_diff_A=abs(mae - 0.0300069),
                  golden_max_abs_delta_error_A=gvd,
                  golden_max_abs_cmd_error_A=gvc,
                  golden_n=int(gi["expected_delta_A"].size),
                  verdict=("PASS" if abs(mae - 0.0300069) < 1e-4
                           and max(gvd, gvc) < 1e-3 else "FAIL"))
print("  MAE %.6f A  expected 0.030007  diff %.2e" % (mae, abs(mae - 0.0300069)))
print("  RMSE %.6f A" % rmse)
print("  golden vectors: n=%d  max|dDelta| = %.3e  max|dCmd| = %.3e"
      % (gi["expected_delta_A"].size, gvd, gvc))
print("  -> %s" % rec["inv"]["verdict"])

rec["environment"] = dict(python=sys.version.split()[0], torch=torch.__version__,
                          cuda_available=torch.cuda.is_available(),
                          note="models loaded from the release directory only; "
                               "no file from the original training project was "
                               "imported for the model definition")
rec["final_test_read"] = False
json.dump(rec, open(os.path.join(OUT, "load_test.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("\n-> audits/load_test.json")
print("总判定:", "PASS" if rec["fwd"]["verdict"] == "PASS"
      and rec["inv"]["verdict"] == "PASS" else "FAIL")
