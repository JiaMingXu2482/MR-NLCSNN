"""Launch the five V22 models, one process each (task book: default 5-way parallel).

Writes:
  <out>/logs/<MODEL>.log            per-process stdout
  <out>/gpu_parallel_log/gpu_1s.csv per-second GPU telemetry (util/mem/temp/power)
  <out>/launch_config.json, batch_status.json
  <out>/summary_all.csv / .json, model_parameter_counts.csv,
        paired_file_differences.csv, failed_or_missing_runs.csv

If a run fails with CUDA OOM the launcher retries the remaining runs with one
fewer worker (5 -> 4 -> 3 -> 2 -> 1), never switching to CPU or shortening data.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v22_models import MODELS  # noqa: E402

PY = sys.executable
REQUIRED = ("run_config.json", "metrics.jsonl", "best.pth", "last.pth", "summary.json",
            "per_file_metrics.csv", "per_condition_metrics.csv",
            "step_response_metrics_250N.csv", "step_response_metrics_50N.csv",
            "hysteresis_metrics.csv", "acceleration_residual_diagnostics.json",
            "monotonicity_report.json", "inverse_100hz_report.json", "resource_report.json")


def nvidia(query):
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=" + query,
                                     "--format=csv,noheader,nounits"],
                                    stderr=subprocess.DEVNULL, text=True)
        return [x.strip() for x in o.strip().split("\n")[0].split(",")]
    except Exception:
        return None


class GpuLogger(threading.Thread):
    def __init__(self, path, stop):
        super().__init__(daemon=True)
        self.path, self.stop = path, stop

    def run(self):
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", "gpu_util_pct", "mem_used_MiB", "mem_total_MiB",
                        "temp_C", "power_W"])
            t0 = time.time()
            while not self.stop.is_set():
                try:
                    v = nvidia("utilization.gpu,memory.used,memory.total,"
                               "temperature.gpu,power.draw")
                    if v and len(v) >= 5:
                        w.writerow(["%.0f" % (time.time() - t0)] + v[:5])
                        fh.flush()
                except Exception:
                    pass
                time.sleep(1.0)


def validate(out, model, epochs):
    d = os.path.join(out, model)
    missing = [f for f in REQUIRED if not os.path.isfile(os.path.join(d, f))]
    if missing:
        raise RuntimeError("%s: missing %s" % (model, missing))
    rows = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"), encoding="utf-8")]
    if [r["epoch"] for r in rows] != list(range(1, epochs + 1)):
        raise RuntimeError("%s: epoch log incomplete (%d)" % (model, len(rows)))
    s = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
    c = json.load(open(os.path.join(d, "run_config.json"), encoding="utf-8"))
    if c["final_test_read"] is not False or s["final_test_read"] is not False:
        raise RuntimeError("%s: final_test_read flag is not False" % model)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=r"D:/v20_data/Parsed_v19_F42_I42")
    ap.add_argument("--epochs", type=int, default=130)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-parallel", type=int, default=5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        raise SystemExit("refusing to overwrite " + out)
    os.makedirs(os.path.join(out, "logs"))
    os.makedirs(os.path.join(out, "gpu_parallel_log"))
    stop = threading.Event()
    GpuLogger(os.path.join(out, "gpu_parallel_log", "gpu_1s.csv"), stop).start()

    workers = a.max_parallel
    json.dump(dict(tag=a.tag, models=list(MODELS), seed=a.seed, epochs=a.epochs,
                   max_parallel_start=workers, python=PY, data=a.data,
                   final_test_read=False,
                   started=datetime.now(timezone(timedelta(hours=8))).isoformat()),
              open(os.path.join(out, "launch_config.json"), "w", encoding="utf-8"), indent=2)

    pending = list(MODELS)
    done, failed, active = [], {}, []
    t0 = time.time()
    try:
        while pending or active:
            while pending and len(active) < workers:
                m = pending.pop(0)
                log = os.path.join(out, "logs", m + ".log")
                fh = open(log, "w", encoding="utf-8")
                proc = subprocess.Popen(
                    [PY, "-B", "-u", os.path.join(HERE, "train_v22.py"),
                     "--model", m, "--seed", str(a.seed), "--epochs", str(a.epochs),
                     "--output", os.path.join(out, m), "--data", a.data,
                     "--execute-training"],
                    cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
                active.append(dict(m=m, p=proc, fh=fh, pos=0, log=log))
                print("START %s (parallel=%d)" % (m, len(active)), flush=True)
            for j in list(active):
                with open(j["log"], encoding="utf-8", errors="replace") as s:
                    s.seek(j["pos"]); add = s.read(); j["pos"] = s.tell()
                for line in add.splitlines():
                    if "/130 val=" in line:
                        print("[%s] %s" % (j["m"], line), flush=True)
                if j["p"].poll() is None:
                    continue
                j["fh"].close(); active.remove(j)
                if j["p"].returncode == 0:
                    try:
                        validate(out, j["m"], a.epochs)
                        done.append(j["m"])
                        print("COMPLETE %s (%d/%d)" % (j["m"], len(done), len(MODELS)), flush=True)
                    except Exception as e:
                        failed[j["m"]] = str(e)[:200]
                        print("INVALID %s: %s" % (j["m"], e), flush=True)
                else:
                    txt = open(j["log"], encoding="utf-8", errors="replace").read()
                    oom = "out of memory" in txt.lower() or "CUDA error" in txt
                    failed[j["m"]] = "exit %d%s" % (j["p"].returncode,
                                                    " (CUDA OOM)" if oom else "")
                    print("FAILED %s: %s" % (j["m"], failed[j["m"]]), flush=True)
                    if oom and workers > 1:
                        workers -= 1
                        print("  -> reducing parallelism to %d" % workers, flush=True)
            if active:
                time.sleep(2)
    finally:
        stop.set()
        for j in active:
            if j["p"].poll() is None:
                j["p"].terminate()

    wall = time.time() - t0
    # ---- aggregate -------------------------------------------------------
    rows, params = [], []
    for m in MODELS:
        sp = os.path.join(out, m, "summary.json")
        if not os.path.exists(sp):
            continue
        s = json.load(open(sp, encoding="utf-8"))
        r = json.load(open(os.path.join(out, m, "resource_report.json"), encoding="utf-8"))
        rows.append(dict(model=m, params=s["params"]["total"], seed=s["seed"],
                         epochs=s["completed_epochs"], best_epoch=s["best_epoch"],
                         pooled=s["validation"]["pooled_nrmse_pct"],
                         worst=s["validation"]["worst_file_pct"],
                         hf=s["validation"]["hf_nrmse_pct"],
                         tau_ms=s.get("tau_ms"), b_a=s.get("b_a"),
                         step250_err=s["step_250N"]["lag_err_median"],
                         step50_err=s["step_50N"]["lag_err_median"],
                         hyst_err=s.get("hysteresis_rel_error_mean"),
                         inv_p99=(s["inverse"].get("time_p99_ms")
                                  if s["inverse"].get("status") == "PASS" else None),
                         mono_viol=s.get("monotonicity", {}).get("violations", 0),
                         peak_alloc_MiB=r["cuda_peak_allocated_MiB"],
                         sec_per_epoch=r["seconds_per_epoch"]))
        params.append(dict(model=m, total=s["params"]["total"],
                           **{k: v for k, v in s["params"]["groups"].items()}))
        for c, v in s["validation"]["per_type_pct"].items():
            rows[-1]["cond_" + c[:2]] = v
    if rows:
        with open(os.path.join(out, "summary_all.csv"), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        json.dump(rows, open(os.path.join(out, "summary_all.json"), "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        with open(os.path.join(out, "model_parameter_counts.csv"), "w", newline="",
                  encoding="utf-8-sig") as fh:
            keys = sorted({k for p in params for k in p})
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(params)
        # per-file paired differences (descriptive only)
        perf = {}
        for m in [r["model"] for r in rows]:
            p = os.path.join(out, m, "per_file_metrics.csv")
            perf[m] = {r["file"]: float(r["nrmse_pct"])
                       for r in csv.DictReader(open(p, encoding="utf-8"))}
        names = sorted(set.intersection(*[set(v) for v in perf.values()])) if perf else []
        with open(os.path.join(out, "paired_file_differences.csv"), "w", newline="",
                  encoding="utf-8-sig") as fh:
            ms = list(perf)
            w = csv.writer(fh)
            w.writerow(["file"] + ms + ["max-min"])
            for n in names:
                vals = [perf[m][n] for m in ms]
                w.writerow([n] + ["%.4f" % v for v in vals] + ["%.4f" % (max(vals) - min(vals))])
    with open(os.path.join(out, "failed_or_missing_runs.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["model", "status", "detail"])
        for m in MODELS:
            w.writerow([m, "OK" if m in done else "FAILED", failed.get(m, "")])
    json.dump(dict(status="COMPLETE" if len(done) == len(MODELS) else "PARTIAL",
                   completed=done, failed=failed, wall_seconds=wall,
                   workers_final=workers, final_test_read=False),
              open(os.path.join(out, "batch_status.json"), "w", encoding="utf-8"), indent=2)
    print("\nDONE %d/%d in %.1f min (workers=%d)" % (len(done), len(MODELS), wall / 60, workers))
    for r in rows:
        print("%-5s pool=%7.3f worst=%7.2f params=%d step250=%s"
              % (r["model"], r["pooled"], r["worst"], r["params"], r["step250_err"]))


if __name__ == "__main__":
    main()
