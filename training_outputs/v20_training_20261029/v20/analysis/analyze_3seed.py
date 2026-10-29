import json
import math

SEEDS = (7, 17, 27)
MODELS = ("A", "B", "D", "E", "G")

PATHS = {
    (7, "A"): r"D:/v20_data/v20_runs_20261029/A/summary.json",
    (7, "B"): r"D:/v20_data/v20_runs_20261029/B/summary.json",
    (7, "D"): r"D:/v20_data/v20_runs_20261029/D/summary.json",
    (7, "E"): r"D:/v20_data/v20_runs_20261029/E/summary.json",
    (7, "G"): r"D:/v20_data/v20_G/seed7/summary.json",
    (17, "A"): r"D:/v20_data/v20_seed17/A/summary.json",
    (17, "B"): r"D:/v20_data/v20_seed17/B/summary.json",
    (17, "D"): r"D:/v20_data/v20_seed17/D/summary.json",
    (17, "E"): r"D:/v20_data/v20_seed17/E/summary.json",
    (17, "G"): r"D:/v20_data/v20_G/seed17/summary.json",
    (27, "A"): r"D:/v20_data/v20_seed27/A/summary.json",
    (27, "B"): r"D:/v20_data/v20_seed27/B/summary.json",
    (27, "D"): r"D:/v20_data/v20_seed27/D/summary.json",
    (27, "E"): r"D:/v20_data/v20_seed27/E/summary.json",
    (27, "G"): r"D:/v20_data/v20_G/seed27/summary.json",
}

S = {}
for (s, m), p in PATHS.items():
    S[(s, m)] = json.load(open(p, encoding="utf-8"))


def mean(v):
    return sum(v) / len(v)


def sd(v):
    mu = mean(v)
    return math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))


print("=== 逐 seed pooled NRMSE % ===")
print("%-4s %8s %8s %8s %10s %8s" % ("M", "seed7", "seed17", "seed27", "mean", "sd"))
for m in MODELS:
    v = [S[(s, m)]["best_pooled_nrmse_pct"] for s in SEEDS]
    print("%-4s %8.2f %8.2f %8.2f %10.3f %8.3f" % (m, *v, mean(v), sd(v)))
print()
print("C (nfl_mlp, 仅 seed7): %.2f  [不参与多 seed 比较]" % S[(7, "A")]["best_pooled_nrmse_pct"] if False else "")
c7 = json.load(open(r"D:/v20_data/v20_runs_20261029/C/summary.json", encoding="utf-8"))
print("C (nfl_mlp, 仅 seed7): %.2f %%" % c7["best_pooled_nrmse_pct"])

print()
print("=== 逐 seed 6.62Hz HF % ===")
print("%-4s %8s %8s %8s %10s %8s" % ("M", "seed7", "seed17", "seed27", "mean", "sd"))
for m in MODELS:
    v = [S[(s, m)]["best_hf_nrmse_pct"] for s in SEEDS]
    print("%-4s %8.2f %8.2f %8.2f %10.3f %8.3f" % (m, *v, mean(v), sd(v)))

print()
print("=== 配对差值（同 seed 内，单位 pp；负 = 前者更好）===")


def paired(a, b, label):
    d = [S[(s, a)]["best_pooled_nrmse_pct"] - S[(s, b)]["best_pooled_nrmse_pct"] for s in SEEDS]
    mu, s_ = mean(d), sd(d)
    se = s_ / math.sqrt(len(d))
    t = mu / se if se > 0 else float("nan")
    print("%-10s per-seed: %s   mean=%+.3f  sd=%.3f  se=%.3f  t(df=2)=%+.2f"
          % (label, "  ".join("%+.2f" % x for x in d), mu, s_, se, t))
    return mu, s_


res = {}
res["A-B"] = paired("A", "B", "A-B")
res["G-E"] = paired("G", "E", "G-E")
res["A-G"] = paired("A", "G", "A-G")
res["B-E"] = paired("B", "E", "B-E")

print()
print("=== 2x2 因子效应 ===")
print("格子: (raw,16)=A  (raw,8)=G  (NFL,16)=B  (NFL,8)=E")
inp16 = res["A-B"][0]
inp8 = res["G-E"][0]
wid_raw = res["A-G"][0]
wid_nfl = res["B-E"][0]
print("输入效应 @hidden16 (B-A) = %+.3f pp" % inp16)
print("输入效应 @hidden8  (E-G) = %+.3f pp" % inp8)
print("  两者同向? %s   交互(差值之差) = %+.3f pp" % ("是" if inp16 * inp8 > 0 else "否", inp16 - inp8))
print("宽度效应 @raw  (G-A) = %+.3f pp" % wid_raw)
print("宽度效应 @NFL  (E-B) = %+.3f pp" % wid_nfl)
print("  两者同向? %s   交互(差值之差) = %+.3f pp" % ("是" if wid_raw * wid_nfl > 0 else "否", wid_raw - wid_nfl))

print()
print("=== 每模型均值±sd 的极差 vs seed 波动 ===")
mus = {m: mean([S[(s, m)]["best_pooled_nrmse_pct"] for s in SEEDS]) for m in MODELS}
print("五模型均值极差 = %.3f pp (best %s=%.3f, worst %s=%.3f)"
      % (max(mus.values()) - min(mus.values()),
         min(mus, key=mus.get), min(mus.values()), max(mus, key=mus.get), max(mus.values())))
within = [sd([S[(s, m)]["best_pooled_nrmse_pct"] for s in SEEDS]) for m in MODELS]
print("模型内 seed 标准差 = %.3f ~ %.3f pp (中位 %.3f)"
      % (min(within), max(within), sorted(within)[len(within) // 2]))
