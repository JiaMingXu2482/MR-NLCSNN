import json
import os
import csv
import statistics as st

EXT = r"D:/v22_ae_hybrid/runs_ext230"
BASE = r"D:/v22_ae_hybrid/runs_seed7"

print("=== 130 轮 vs 230 轮（同 seed 7，热重启续训）===")
print("%-5s %10s %10s %9s %9s %9s %9s" % (
    "模型", "130轮best", "230轮best", "差值", "best_ep", "最差文件%", "tau_ms"))
data = {}
for m in ("E0", "EGTA"):
    e = json.load(open(os.path.join(EXT, m, "summary.json"), encoding="utf-8"))
    b = json.load(open(os.path.join(BASE, m, "summary.json"), encoding="utf-8"))
    data[m] = (e, b)
    print("%-5s %10.3f %10.3f %+9.3f %9d %9.2f %9s" % (
        m, b["validation"]["pooled_nrmse_pct"], e["validation"]["pooled_nrmse_pct"],
        e["delta_vs_base"], e["best_epoch"], e["validation"]["worst_file_pct"],
        ("%.2f" % e["tau_ms"]) if e.get("tau_ms") else "-"))

print()
print("=== 分工况（230 轮，括号为 130 轮）===")
cats = ["SH_", "ST_", "RLF_", "RHF_", "SR_"]
print("%-5s %s" % ("模型", "".join("%18s" % c.rstrip("_") for c in cats)))
for m in ("E0", "EGTA"):
    e, b = data[m]
    cells = []
    for c in cats:
        ve = e["validation"]["per_type_pct"].get(c)
        vb = b["validation"]["per_type_pct"].get(c)
        cells.append("%8.2f (%6.2f)" % (ve, vb))
    print("%-5s %s" % (m, "".join(cells)))

print()
print("=== 动态与实时性（230 轮 vs 130 轮）===")
print("%-5s %12s %12s %12s %12s %10s" % (
    "模型", "阶跃250误差", "(130轮)", "逆解P99", "(130轮)", "单调违反"))
for m in ("E0", "EGTA"):
    e, b = data[m]
    print("%-5s %12s %12s %12s %12s %10s" % (
        m,
        ("%+.1f" % e["step_250N"]["lag_err_median"]) if e["step_250N"]["lag_err_median"] is not None else "-",
        ("%+.1f" % b["step_250N"]["lag_err_median"]) if b["step_250N"]["lag_err_median"] is not None else "-",
        ("%.2f" % e["inverse"]["time_p99_ms"]) if e["inverse"].get("status") == "PASS" else "n/a",
        ("%.2f" % b["inverse"]["time_p99_ms"]) if b["inverse"].get("status") == "PASS" else "n/a",
        e["monotonicity"].get("violations", "n/a")))

print()
print("=== 关键对照：E0 vs EGTA 的精度差如何变化 ===")
for tag, idx in (("130 轮", 1), ("230 轮", 0)):
    a = data["E0"][idx]["validation"]["pooled_nrmse_pct"]
    b = data["EGTA"][idx]["validation"]["pooled_nrmse_pct"]
    print("  %s: E0 %.3f  EGTA %.3f  差 %+.3f pp  %s" % (
        tag, a, b, b - a, "在 0.5 pp 选型带内" if abs(b - a) <= 0.5 else "超出 0.5 pp 选型带"))
