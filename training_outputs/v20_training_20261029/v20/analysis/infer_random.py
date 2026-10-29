"""A / E inference on the random-excitation validation files, 3 seeds.

Saves per-file metrics and the prediction curves used by plot_ae.py.

IMPORTANT (bug fixed 2026-10-29): the first version saved `pred` in NORMALISED
units while `truth` was already multiplied by f_scale, so a plot of the two on
one axis collapsed the predictions to a flat line near zero. `pred` is now
multiplied by f_scale here, so both arrays are in newtons. Do not "fix" this
downstream again.

Run from anywhere:
    python infer_random.py
Outputs (in the repo):
    analysis/results/a_vs_e_random_metrics.json
    analysis/results/a_vs_e_random_preds.npz
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)                       # .../v20
sys.path.insert(0, CODE)
from v20_data import normalization, read_dataset, sample   # noqa: E402
from v20_models import V20Model                              # noqa: E402

DATA = r"D:/v20_data/Parsed_v19_F42_I42"
SEEDS = (7, 17, 27)
RUNS = {
    7: r"D:/v20_data/v20_runs_20261029",
    17: r"D:/v20_data/v20_seed17",
    27: r"D:/v20_data/v20_seed27",
}
# G lives in its own root because it was added after the batch dirs existed
RANDOM_PREFIX = ("SR_", "RLF_", "RHF_")

OUT_RES = os.path.join(HERE, "results")
os.makedirs(OUT_RES, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val = read_dataset(DATA, "validation")
train = read_dataset(DATA, "train")
norm = normalization(train)
f_scale = norm["f_scale"]
print("validation files:", len(val), " f_scale =", f_scale)
rand = [it for it in val if it["name"].startswith(RANDOM_PREFIX)]
print("random-excitation files:", [it["name"] for it in rand])


def run_model(state_path, items):
    ck = torch.load(state_path, map_location="cpu", weights_only=True)
    model = V20Model(arch=ck["arch"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    out = {}
    for it in items:
        u, f, valid = sample(it, norm)
        u, f, valid = u.to(device), f.to(device), valid.to(device)
        with torch.no_grad():
            pred = model(u)
            err = (pred - f)[valid] * f_scale
            y = f[valid] * f_scale
            sse, sst = float(err.square().sum()), float(y.square().sum())
        out[it["name"]] = dict(
            nrmse_pct=100 * (sse / max(sst, 1e-12)) ** 0.5,
            sse=sse, sst=sst, samples=int(valid.sum()), best_epoch=ck["epoch"],
            # physical units, same as truth
            pred=(pred[0] * f_scale).cpu().numpy().astype(np.float32),
            truth=(f[0] * f_scale).cpu().numpy().astype(np.float32),
            valid=valid[0].cpu().numpy(),
        )
    return out


results, arrays = {}, {}
for model in ("A", "E"):
    for s in SEEDS:
        r = run_model(os.path.join(RUNS[s], model, "best.pth"), val)
        results[f"{model}|{s}"] = {
            k: {kk: vv for kk, vv in v.items() if kk not in ("pred", "truth", "valid")}
            for k, v in r.items()
        }
        if s == 7:  # curves for plotting (seed 7 best model)
            for k, v in r.items():
                if k.startswith(RANDOM_PREFIX):
                    arrays[f"{model}|{k}|pred"] = v["pred"]
                    arrays[f"{model}|{k}|truth"] = v["truth"]
                    arrays[f"{model}|{k}|valid"] = v["valid"]
    print("done model", model)

np.savez_compressed(os.path.join(OUT_RES, "a_vs_e_random_preds.npz"), **arrays)
with open(os.path.join(OUT_RES, "a_vs_e_random_metrics.json"), "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)

print()
print("%-34s %8s %8s %8s %8s %8s %8s" % ("file", "A_s7", "A_s17", "A_s27", "E_s7", "E_s17", "E_s27"))
for it in rand:
    n = it["name"]
    row = [results[f"{m}|{s}"][n]["nrmse_pct"] for m in ("A", "E") for s in SEEDS]
    print("%-34s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f" % (n, *row))
