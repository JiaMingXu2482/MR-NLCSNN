"""Extend a finished V22 run by N more epochs, continuing from its last.pth.

Starting point is epoch 130 of runs_seed7/<MODEL>.  The LR schedule is recomputed
against the EXTENDED horizon (epochs_total = 230), which means the LR jumps from
3e-6 (end of the 130-epoch cosine) back up to ~1.24e-4 at epoch 131 - a deliberate
warm restart.  The discontinuity is recorded in run_config.json so the result can
never be mistaken for a single continuous 230-epoch schedule.

The base run is NOT touched; everything is written to a new directory and the
base best metric is carried into summary.json for direct comparison.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v20_data import digest, normalization, read_dataset, sample        # noqa: E402
from v22_models import V22Model, count_params                           # noqa: E402
from train_v22 import (accel_diagnostics, code_hash, evaluate, hysteresis,  # noqa: E402
                       inverse_100hz, monotonicity, step_events, write_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["E0", "EGTA"])
    ap.add_argument("--base", required=True, help="finished run dir (runs_seed7/<MODEL>)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--extra", type=int, default=100)
    ap.add_argument("--epochs-total", type=int, default=230)
    ap.add_argument("--data", default=r"D:/v20_data/Parsed_v19_F42_I42")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - refusing CPU fallback")
    dev = torch.device("cuda:0")
    out = a.output
    if os.path.exists(out):
        raise SystemExit("exists: " + out)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)

    base_cfg = json.load(open(os.path.join(a.base, "run_config.json"), encoding="utf-8"))
    base_sum = json.load(open(os.path.join(a.base, "summary.json"), encoding="utf-8"))
    start_epoch = base_sum["completed_epochs"]

    random.seed(base_cfg["seed"]); torch.manual_seed(base_cfg["seed"])
    np.random.seed(base_cfg["seed"]); torch.cuda.manual_seed_all(base_cfg["seed"])

    train = read_dataset(a.data, "train")
    val = read_dataset(a.data, "validation")
    norm = normalization(train)
    man = digest(os.path.join(a.data, "alignment_manifest.json"))

    model = V22Model(a.model)
    ck = torch.load(os.path.join(a.base, "last.pth"), map_location="cpu", weights_only=True)
    model.load_state_dict(ck["state_dict"])
    model.to(dev)
    pc = count_params(model)
    print("%s resume from ep%d -> ep%d  params=%d" % (a.model, start_epoch, a.epochs_total, pc["total"]),
          flush=True)

    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    lr_at = lambda e: 3e-4 * (min(e / 5, 1.0) if e <= 5 else
                              0.01 + 0.99 * (1 + math.cos(math.pi * (e - 5) /
                                                          max(a.epochs_total - 5, 1))) / 2)
    cfg = dict(model=a.model, seed=base_cfg["seed"], run_type="EXTENSION",
               base_run=a.base, resumed_from_epoch=start_epoch,
               epochs_total=a.epochs_total, extra_epochs=a.extra,
               data=a.data, manifest_sha256=man, norm=norm, params=pc,
               optimizer=dict(name="AdamW", lr=3e-4, weight_decay=1e-4),
               schedule="5-epoch warmup + cosine recomputed over epochs_total=%d" % a.epochs_total,
               lr_at_131=lr_at(start_epoch + 1), lr_at_130_old_schedule=lr_at(130),
               note="warm restart: LR jumps at the resume point by design",
               base_best_pooled_pct=base_sum["validation"]["pooled_nrmse_pct"],
               grad_clip=1.0, device="cuda:0", final_test_read=False, code=code_hash())
    cfg["config_sha256"] = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    json.dump(cfg, open(os.path.join(out, "run_config.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    best = dict(score=float("inf"), epoch=0, metrics=None)
    t0_all = time.perf_counter()
    for epoch in range(start_epoch + 1, a.epochs_total + 1):
        te = time.perf_counter()
        model.train()
        random.shuffle(train)
        lr = lr_at(epoch)
        for g in opt.param_groups:
            g["lr"] = lr
        tot = 0.0
        for it in train:
            u, f, valid = sample(it, norm)
            u, f, valid = u.to(dev), f.to(dev), valid.to(dev)
            opt.zero_grad(set_to_none=True)
            pred = model(u)
            loss = ((pred - f)[valid]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("non-finite loss at ep%d" % epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss.detach())
        m, _ = evaluate(model, val, norm, dev)
        m.update(epoch=epoch, train_loss=tot / len(train),
                 seconds=time.perf_counter() - te, lr=lr)
        if model.use_tau:
            m["tau_ms"] = float(model.tau_ms().detach().cpu())
        if model.use_acc:
            m["b_a"] = float(model.b_a.detach().cpu())
        with open(os.path.join(out, "metrics.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(m) + "\n")
        if m["pooled"] < best["score"]:
            best = dict(score=m["pooled"], epoch=epoch, metrics=m)
            torch.save(dict(model=a.model, state_dict=copy.deepcopy(model.state_dict()),
                            norm=norm, epoch=epoch, params=pc), os.path.join(out, "best.pth"))
        if (epoch - start_epoch) % 10 == 0 or epoch == a.epochs_total:
            print("ep%d/%d val=%.3f%% lr=%.2e (%.1fs)" % (epoch, a.epochs_total,
                                                          m["pooled"], lr, m["seconds"]),
                  flush=True)
    torch.save(dict(model=a.model, state_dict=copy.deepcopy(model.state_dict()),
                    norm=norm, epoch=a.epochs_total, params=pc),
               os.path.join(out, "last.pth"))

    bm = V22Model(a.model)
    bm.load_state_dict(torch.load(os.path.join(out, "best.pth"), map_location="cpu",
                                  weights_only=True)["state_dict"])
    bm.to(dev).eval()
    final, keep = evaluate(bm, val, norm, dev, keep=True)
    write_csv(os.path.join(out, "per_file_metrics.csv"),
              [["file", "samples", "nrmse_pct", "rmse_N", "sse", "sst"]] +
              [[r["name"], r["samples"], "%.6f" % r["nrmse"], "%.3f" % r["rmse"],
                "%.6e" % r["sse"], "%.6e" % r["sst"]]
               for r in sorted(final["rows"], key=lambda z: z["name"])])
    per = [[c, "%.6f" % final["per_type"][c] if final["per_type"][c] is not None else ""]
           for c in ("SH_", "ST_", "RLF_", "RHF_", "SR_")]
    write_csv(os.path.join(out, "per_condition_metrics.csv"),
              [["condition", "pooled_nrmse_pct"]] + per)
    ev_main = step_events(keep, 250.0)
    ev_supp = step_events(keep, 50.0)
    hdr = ["file", "t_edge_s", "dI_A", "dF_N", "v_sign", "i_dir", "lag_true_ms",
           "lag_pred_ms", "lag_err_ms", "amp_err_N"]
    for tag, evs in (("250N", ev_main), ("50N", ev_supp)):
        write_csv(os.path.join(out, "step_response_metrics_%s.csv" % tag), [hdr] +
                  [[e["file"], "%.3f" % e["t_edge"], "%.4f" % e["dI"], "%.1f" % e["dF"],
                    e["v_sign"], e["i_dir"], "%.1f" % e["lag_true"], "%.1f" % e["lag_pred"],
                    "%.1f" % e["lag_err"], "%.1f" % e["amp_err"]] for e in evs])
    hyst = hysteresis(keep)
    write_csv(os.path.join(out, "hysteresis_metrics.csv"),
              [["file", "start_sample", "end_sample", "measured_N_mm", "predicted_N_mm",
                "abs_error", "rel_error", "status"]] +
              [[h["file"], h["start_sample"], h["end_sample"], "%.4f" % h["measured_N_mm"],
                "%.4f" % h["predicted_N_mm"], "%.4f" % h["abs_error"],
                "%.6f" % h["rel_error"], h["status"]] for h in hyst])
    acc = accel_diagnostics(keep, bm, norm, dev, val)
    json.dump(acc, open(os.path.join(out, "acceleration_residual_diagnostics.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
    mono = monotonicity(bm) if bm.use_g else dict(status="NOT_APPLICABLE")
    json.dump(mono, open(os.path.join(out, "monotonicity_report.json"), "w",
                         encoding="utf-8"), indent=2, ensure_ascii=False)
    inv = inverse_100hz(bm, keep, val, norm, dev)
    json.dump(inv, open(os.path.join(out, "inverse_100hz_report.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
    wall = time.perf_counter() - t0_all
    json.dump(dict(params=pc, weight_bytes=os.path.getsize(os.path.join(out, "best.pth")),
                   train_seconds=wall, seconds_per_epoch=wall / a.extra,
                   cuda_peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2 ** 20,
                   throughput_samples_per_s=sum(i["values"].shape[0] for i in train)
                   * a.extra / wall),
              open(os.path.join(out, "resource_report.json"), "w", encoding="utf-8"), indent=2)
    med = lambda v: float(np.median(v)) if v else None
    summ = dict(model=a.model, seed=base_cfg["seed"], run_type="EXTENSION",
                completed_epochs=a.epochs_total, epochs_run=a.extra,
                base_run=a.base, status="PASS", best_epoch=best["epoch"], params=pc,
                validation=dict(pooled_nrmse_pct=best["score"],
                                per_type_pct=best["metrics"]["per_type"],
                                hf_nrmse_pct=best["metrics"]["hf"],
                                worst_file_pct=best["metrics"]["worst"]),
                base_validation_pooled=base_sum["validation"]["pooled_nrmse_pct"],
                delta_vs_base=best["score"] - base_sum["validation"]["pooled_nrmse_pct"],
                step_250N=dict(n=len(ev_main),
                               lag_err_median=med([e["lag_err"] for e in ev_main])),
                step_50N=dict(n=len(ev_supp),
                              lag_err_median=med([e["lag_err"] for e in ev_supp])),
                hysteresis_rel_error_mean=(float(np.mean([h["rel_error"] for h in hyst]))
                                           if hyst else None),
                monotonicity=mono, inverse=inv, acceleration=acc,
                tau_ms=(float(bm.tau_ms().detach().cpu()) if bm.use_tau else None),
                b_a=(float(bm.b_a.detach().cpu()) if bm.use_acc else None),
                final_test_read=False,
                limitations=["extension of a single-seed run; warm restart at the resume point",
                             "not comparable to the 130-epoch matrix as a separate architecture"])
    json.dump(summ, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("%s EXT DONE ep%d best=%.3f%% (base %.3f%%, delta %+.3f)"
          % (a.model, best["epoch"], best["score"], base_sum["validation"]["pooled_nrmse_pct"],
             summ["delta_vs_base"]), flush=True)


if __name__ == "__main__":
    main()
