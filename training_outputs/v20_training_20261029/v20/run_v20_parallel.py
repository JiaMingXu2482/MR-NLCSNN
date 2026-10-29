"""Launch several v20 models in parallel (separate processes, one per model)
on the user's local machine. Windows-compatible: no fork, no shell, only
subprocess + pathlib. All models share the one GPU; the models are small
(GRU 16, MLP 16, except F), so sharing is workable. If you hit out-of-memory
or the runs crawl, rerun with --sequential: same commands, one at a time.

Usage:
  python run_v20_parallel.py --data <Parsed_v19_F42_I42> ^
      --output-root <v20_runs> --epochs 30 --execute-training
  python run_v20_parallel.py --data <...> --output-root <...> ^
      --models A,B,C,D,E --epochs 30 --sequential --execute-training

Without --execute-training this only prints the plan and exits.
Each model writes its own log, metrics.jsonl, best.pth and summary.json
under <output-root>/<model_letter>/, and this script writes
<output-root>/summary_all.json at the end. A failed model does not stop
the others; its entry in summary_all.json is marked failed.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = ("A", "B", "C", "D", "E")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS),
                   help="comma list, e.g. A,B,C,D,E or A,B,C,D,E,F (F is the optional c/s/b control)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=7,
                   help="forwarded to train_v20.py --seed for every model in this batch")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--sequential", action="store_true")
    p.add_argument("--execute-training", action="store_true")
    a = p.parse_args()
    models = [m.strip().upper() for m in a.models.split(",") if m.strip()]
    if not models or any(len(m) != 1 or m not in "ABCDEFG" for m in models):
        p.error("--models must be letters from A..G")
    if a.output_root.exists():
        p.error("output-root exists; use a new directory (never overwrite a run)")
    print("PLAN (no training started):")
    for m in models:
        print(f"  model {m} -> {a.output_root / m}  epochs={a.epochs} seed={a.seed} "
              f"device={a.device} data={a.data} sequential={a.sequential}")
    if not a.execute_training:
        print("Add --execute-training to launch. This was only a plan.")
        return
    a.output_root.mkdir(parents=True)
    procs = []
    for m in models:
        outdir = a.output_root / m
        log = (a.output_root / f"log_{m}.txt").open("w", encoding="utf-8", errors="replace")
        cmd = [
            sys.executable,
            str(HERE / "train_v20.py"),
            "--data", str(a.data),
            "--model", m,
            "--output", str(outdir),
            "--epochs", str(a.epochs),
            "--seed", str(a.seed),
            "--device", a.device,
            "--execute-training",
        ]
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE))
        procs.append((m, proc, log))
        print(f"launched model {m} (pid {proc.pid})", flush=True)
        if a.sequential:
            rc = proc.wait()
            log.close()
            print(f"model {m} finished rc={rc}", flush=True)
    if not a.sequential:
        for m, proc, log in procs:
            rc = proc.wait()
            log.close()
            print(f"model {m} finished rc={rc}", flush=True)
    summary = {}
    for m, proc, _log in procs:
        sfile = a.output_root / m / "summary.json"
        if proc.returncode == 0 and sfile.exists():
            summary[m] = json.loads(sfile.read_text(encoding="utf-8"))
            summary[m]["status"] = "ok"
        else:
            summary[m] = dict(status="failed", returncode=proc.returncode,
                              log=str(a.output_root / f"log_{m}.txt"))
    (a.output_root / "summary_all.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    def fmt(x):
        return f"{x:.4f}%" if x is not None else "n/a"

    print("\nSUMMARY (validation only, final_test never opened):")
    for m in models:
        s = summary[m]
        if s.get("status") == "ok":
            print(f"  {m}: pooled={fmt(s['best_pooled_nrmse_pct'])} hf={fmt(s['best_hf_nrmse_pct'])} "
                  f"best_epoch={s['best_epoch']} params={s['parameter_count']}")
        else:
            print(f"  {m}: FAILED rc={s.get('returncode')} log={s.get('log')}")
    if any(s.get("status") != "ok" for s in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
