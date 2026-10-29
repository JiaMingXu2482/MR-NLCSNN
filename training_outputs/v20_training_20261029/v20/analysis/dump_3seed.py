"""Dump the 3-seed comparison to analysis/results/results_3seed.json so the
tables in EXPERIMENT_LOG_v20.md can be regenerated without retraining."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "results_3seed.json")

SEEDS = (7, 17, 27)
MODELS = ("A", "B", "D", "E", "G")
PATHS = {}
for s in SEEDS:
    root = {7: "v20_runs_20261029", 17: "v20_seed17", 27: "v20_seed27"}[s]
    for m in ("A", "B", "D", "E"):
        PATHS[(s, m)] = rf"D:/v20_data/{root}/{m}/summary.json"
    PATHS[(s, "G")] = rf"D:/v20_data/v20_G/seed{s}/summary.json"


def mean(v):
    return sum(v) / len(v)


def sd(v):
    mu = mean(v)
    return math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))


S = {k: json.load(open(p, encoding="utf-8")) for k, p in PATHS.items()}
out = {"seeds": list(SEEDS), "epochs": 130,
       "metric": "best validation pooled NRMSE % (lower is better)",
       "models": {}, "paired": {}, "factorial": {}}

for m in MODELS:
    pooled = [S[(s, m)]["best_pooled_nrmse_pct"] for s in SEEDS]
    hf = [S[(s, m)]["best_hf_nrmse_pct"] for s in SEEDS]
    wf = [S[(s, m)]["best_worst_file_pct"] for s in SEEDS]
    out["models"][m] = dict(
        arch=S[(7, m)]["arch"],
        parameter_count=S[(7, m)]["parameter_count"],
        per_seed_pooled=dict(zip(map(str, SEEDS), [round(x, 4) for x in pooled])),
        pooled_mean=round(mean(pooled), 4), pooled_sd=round(sd(pooled), 4),
        per_seed_hf=dict(zip(map(str, SEEDS), [round(x, 4) for x in hf])),
        hf_mean=round(mean(hf), 4), hf_sd=round(sd(hf), 4),
        worst_mean=round(mean(wf), 4),
        best_epochs={str(s): S[(s, m)]["best_epoch"] for s in SEEDS},
    )


def paired(a, b):
    d = [S[(s, a)]["best_pooled_nrmse_pct"] - S[(s, b)]["best_pooled_nrmse_pct"] for s in SEEDS]
    mu, s_ = mean(d), sd(d)
    se = s_ / math.sqrt(len(d))
    return dict(per_seed={str(s): round(x, 4) for s, x in zip(SEEDS, d)},
                mean=round(mu, 4), sd=round(s_, 4), se=round(se, 4),
                t_df2=round(mu / se, 3) if se else None,
                t_crit_p05_df2=4.303, significant_at_p05=bool(abs(mu / se) > 4.303) if se else False)


for name, (a, b) in {"A-B": ("A", "B"), "G-E": ("G", "E"),
                     "A-G": ("A", "G"), "B-E": ("B", "E")}.items():
    out["paired"][name] = paired(a, b)

inp16 = out["paired"]["A-B"]["mean"]
inp8 = out["paired"]["G-E"]["mean"]
wid_raw = out["paired"]["A-G"]["mean"]
wid_nfl = out["paired"]["B-E"]["mean"]
out["factorial"] = dict(
    cells={"A": "(raw,hidden16)", "G": "(raw,hidden8)",
           "B": "(NFL,hidden16)", "E": "(NFL,hidden8)"},
    input_effect_at_h16_A_minus_B=inp16,
    input_effect_at_h8_G_minus_E=inp8,
    width_effect_on_raw_A_minus_G=wid_raw,
    width_effect_on_nfl_B_minus_E=wid_nfl,
    interaction=round(inp16 - inp8, 4),
    note=("sign convention: value = first minus second; negative means the first "
          "model has lower NRMSE. Every cell confounds input type with parameter "
          "count, so these effects cannot separate feature value from capacity."),
)
out["note"] = ("n=3 seeds; df=2; critical t at p=0.05 is 4.303. No comparison here "
               "reaches significance. Model spread is comparable to within-model "
               "seed sd, so no ranking should be declared from this run.")

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2, ensure_ascii=False)
print("saved:", OUT)
for m in MODELS:
    d = out["models"][m]
    print("%-2s %-16s mean=%.3f sd=%.3f  hf_mean=%.3f" % (
        m, d["arch"], d["pooled_mean"], d["pooled_sd"], d["hf_mean"]))
