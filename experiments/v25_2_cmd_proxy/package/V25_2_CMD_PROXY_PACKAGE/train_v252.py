"""V25.2-RI0-CMD training: residual inverse model on previous ECU command.

    I_cmd[k] = clip(I_cmd[k-1] + delta_hat[k], 0, 1.6)
Offline the previous command is the proxy clip(|I_meas[k]|,0,1.6) and the label is
clip(|I_meas[k+1]|,0,1.6).  No FIR / IIR anywhere; 1 kHz -> 100 Hz is x[::10].

Inherited from V25: AdamW 3e-4 / wd 1e-4, 5-epoch warmup + cosine to 0.01*base_lr,
grad clip 1.0, one full file per step, baseline sampling, zero head init,
best checkpoint by validation pooled NRMSE.  Budget raised to 180 epochs.
"""
import argparse
import copy
import csv
import importlib.util
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
from v20_data import normalization, read_dataset                     # noqa: E402
from v252_data import build_v252, I_SCALE                            # noqa: E402

_spec = importlib.util.spec_from_file_location("mdl", os.path.join(HERE, "model_def.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
ResidualI0 = M.ResidualI0

ROOT = r"D:/v20_data/Parsed_v19_F42_I42"
CATS = ("SH", "ST", "RLF", "SR", "RHF")
ZERO_TH = 0.05


def fam(n):
    for c in CATS:
        if n.startswith(c + "_"):
            return c
    return "?"


def write_csv(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)


def lr_at(ep, total, base=3e-4, warm=5):
    if ep <= warm:
        return base * ep / warm
    p = (ep - warm) / max(total - warm, 1)
    return base * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p)))


def summarize(rows):
    """rows: name, pred_clip, pred_raw, target, base, mask"""
    P = np.concatenate([r["pred"][r["mask"]] for r in rows])
    R = np.concatenate([r["pred_raw"][r["mask"]] for r in rows])
    T = np.concatenate([r["target"][r["mask"]] for r in rows])
    B = np.concatenate([r["base"][r["mask"]] for r in rows])
    e = P - T
    per = {}
    for c in CATS:
        sub = [r for r in rows if r["name"].startswith(c + "_")]
        if not sub:
            per[c] = None
            continue
        p = np.concatenate([r["pred"][r["mask"]] for r in sub])
        t = np.concatenate([r["target"][r["mask"]] for r in sub])
        b = np.concatenate([r["base"][r["mask"]] for r in sub])
        ee = p - t
        per[c] = dict(n=int(ee.size), mae=float(np.abs(ee).mean()),
                      rmse=float(np.sqrt((ee ** 2).mean())),
                      p95=float(np.percentile(np.abs(ee), 95)),
                      bias=float(ee.mean()),
                      persistence_mae=float(np.abs(b - t).mean()),
                      persistence_rmse=float(np.sqrt(((b - t) ** 2).mean())))
    z = T < ZERO_TH
    d = np.concatenate([np.diff(r["pred"][r["mask"]]) for r in rows if r["mask"].sum() > 1])
    dref = np.concatenate([np.diff(r["target"][r["mask"]]) for r in rows
                           if r["mask"].sum() > 1])
    tv = float(sum(np.abs(np.diff(r["pred"][r["mask"]])).sum() for r in rows
                   if r["mask"].sum() > 1))
    tvr = float(sum(np.abs(np.diff(r["target"][r["mask"]])).sum() for r in rows
                    if r["mask"].sum() > 1))
    return dict(
        n=int(e.size),
        mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
        median=float(np.median(np.abs(e))), p95=float(np.percentile(np.abs(e), 95)),
        p99=float(np.percentile(np.abs(e), 99)), bias=float(e.mean()),
        macro_mae=float(np.mean([np.abs(r["pred"][r["mask"]] - r["target"][r["mask"]]).mean()
                                 for r in rows if r["mask"].sum()])),
        pooled_nrmse=100 * math.sqrt(float((e ** 2).sum()) / max(float((T ** 2).sum()), 1e-12)),
        raw_mae=float(np.abs(R - T).mean()),
        raw_out_of_range=float(((R < 0) | (R > 1.6)).mean()),
        clip_activation=float((np.abs(R - P) > 1e-9).mean()),
        dI_mean=float(d.mean()), dI_rms=float(np.sqrt((d ** 2).mean())),
        dI_p95=float(np.percentile(np.abs(d), 95)), dI_max=float(np.abs(d).max()),
        dI_ref_p95=float(np.percentile(np.abs(dref), 95)),
        total_variation=tv, total_variation_ref=tvr,
        tv_ratio=float(tv / max(tvr, 1e-9)),
        per_type=per,
        zero_current=dict(
            n=int(z.sum()), pred_mean=float(P[z].mean()) if z.sum() else None,
            pred_median=float(np.median(P[z])) if z.sum() else None,
            pred_p95=float(np.percentile(P[z], 95)) if z.sum() else None,
            false_turn_on_005=float((P[z] > 0.05).mean()) if z.sum() else None,
            false_turn_on_010=float((P[z] > 0.10).mean()) if z.sum() else None),
        persistence=dict(
            mae=float(np.abs(B - T).mean()), rmse=float(np.sqrt(((B - T) ** 2).mean())),
            median=float(np.median(np.abs(B - T))),
            p95=float(np.percentile(np.abs(B - T), 95)),
            p99=float(np.percentile(np.abs(B - T), 99)),
            bias=float((B - T).mean())))


def run_eval(net, items, dev):
    net.eval()
    rows = []
    with torch.no_grad():
        for it in items:
            x, base, tgt, mask = build_v252(it, NORM)
            delta = net(x.to(dev)).cpu()[0]
            raw = base[0] + delta
            rows.append(dict(name=it["name"], pred_raw=raw.numpy(),
                             pred=raw.clamp(0, 1.6).numpy(), target=tgt[0].numpy(),
                             base=base[0].numpy(), mask=mask[0].numpy()))
    return summarize(rows), rows


def timing(net, x, dev="cpu", n=400):
    net = net.to(dev).eval()
    st = torch.zeros(2, 1, 8, device=dev)
    xt = x[0].to(dev)
    lat = []
    for i in range(min(n, xt.shape[0] - 1)):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            h, st = net.gru(xt[i].view(1, 1, -1), st)
            z = net.trunk(torch.cat((h[:, 0], xt[i].view(1, -1)), -1))
            _ = net.head(z)
        if dev == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1e3)
    a = np.array(lat)
    return dict(device=dev, n=int(a.size), mean_ms=float(a.mean()),
                p50_ms=float(np.percentile(a, 50)), p95_ms=float(np.percentile(a, 95)),
                p99_ms=float(np.percentile(a, 99)), max_ms=float(a.max()),
                note="Python/PyTorch single-step reference; NOT an MCU WCET")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--execute-training", action="store_true")
    a = ap.parse_args()
    out = a.output
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = read_dataset(ROOT, "train")
    val = read_dataset(ROOT, "validation")
    global NORM
    NORM = normalization(train)
    net = ResidualI0().to(dev)
    pc = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("V25.2 params=%d seed=%d train=%d val=%d device=%s" %
          (pc, a.seed, len(train), len(val), dev), flush=True)
    if pc != 2073:
        raise SystemExit("parameter count %d != 2073" % pc)

    import hashlib
    man = os.path.join(ROOT, "alignment_manifest.json")
    cfg = dict(experiment_id="V25_2_RI0_CMD_PROXY_seed7", seed=a.seed,
               epochs_total=a.epochs, data=ROOT,
               n_train_files=len(train), n_validation_files=len(val),
               manifest_sha256=hashlib.sha256(open(man, "rb").read()).hexdigest(),
               architecture="identical to V25 (GRU(31,8,2) + skip 39 + MLP16 + "
                            "linear delta head)", params=pc,
               timing=dict(control_rate_hz=100,
                           base="clip(|I_meas[k]|,0,1.6) proxy of I_cmd[k-1]",
                           target="clip(|I_meas[k+1]|,0,1.6) proxy of I_cmd[k]",
                           runtime="I_cmd[k]=clip(I_cmd[k-1]+delta,0,1.6)",
                           drop_last_sample=True, no_cross_file=True),
               preprocessing=dict(method="x[::10] direct decimation, phase 0",
                                  fir=False, iir=False, filter_delay_added_ms=0.0),
               optimizer=dict(name="AdamW", lr=3e-4, weight_decay=1e-4),
               schedule="5-epoch linear warmup + cosine to 0.01*base_lr over %d" % a.epochs,
               batch="one full file per step, GRU state reset per file",
               grad_clip=1.0, device=str(dev), norm=NORM,
               loss="MSE((base+delta-target)/1.6) on the unclipped raw prediction",
               sampling=dict(mode="baseline"),
               best_rule="lowest validation pooled NRMSE (inherited from V25/V24-I0)",
               warm_start=False, final_test_read=False)
    cfg["config_sha256"] = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    json.dump(cfg, open(os.path.join(out, "run_config.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    if not a.execute_training:
        print("dry run ok")
        return

    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    fh = open(os.path.join(out, "metrics.jsonl"), "w", encoding="utf-8")
    best = dict(score=1e18, epoch=0)
    best_mae = dict(score=1e18, epoch=0)
    t0 = time.perf_counter()
    for ep in range(1, a.epochs + 1):
        te = time.perf_counter()
        net.train()
        random.shuffle(train)
        lr = lr_at(ep, a.epochs)
        for g in opt.param_groups:
            g["lr"] = lr
        tot = 0.0
        for it in train:
            x, base, tgt, mask = build_v252(it, NORM)
            x, base, tgt, mask = x.to(dev), base.to(dev), tgt.to(dev), mask.to(dev)
            opt.zero_grad(set_to_none=True)
            i_raw = base + net(x)
            loss = (((i_raw - tgt) / I_SCALE) ** 2)[mask].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
        vm, _ = run_eval(net, val, dev)
        rec = dict(epoch=ep, lr=lr, train_loss=tot / len(train), val_nrmse=vm["pooled_nrmse"],
                   val_mae=vm["mae"], val_rmse=vm["rmse"], val_p95=vm["p95"],
                   val_p99=vm["p99"], persistence_mae=vm["persistence"]["mae"],
                   zero_pred=vm["zero_current"]["pred_mean"],
                   seconds=time.perf_counter() - te)
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        if vm["pooled_nrmse"] < best["score"]:
            best = dict(score=vm["pooled_nrmse"], epoch=ep)
            torch.save(dict(state_dict=copy.deepcopy(net.state_dict()), norm=NORM,
                            epoch=ep, params=pc), os.path.join(out, "best.pth"))
        if vm["mae"] < best_mae["score"]:
            best_mae = dict(score=vm["mae"], epoch=ep)
            torch.save(dict(state_dict=copy.deepcopy(net.state_dict()), norm=NORM,
                            epoch=ep, params=pc),
                       os.path.join(out, "best_by_val_mae.pth"))
        if ep % 10 == 1 or ep == a.epochs:
            print("ep%3d/%d nrmse=%.3f%% mae=%.4f p95=%.4f pers=%.4f loss=%.6f (%.1fs)"
                  % (ep, a.epochs, vm["pooled_nrmse"], vm["mae"], vm["p95"],
                     vm["persistence"]["mae"], rec["train_loss"], rec["seconds"]),
                  flush=True)
    fh.close()
    wall = time.perf_counter() - t0
    torch.save(dict(state_dict=copy.deepcopy(net.state_dict()), norm=NORM,
                    epoch=a.epochs, params=pc), os.path.join(out, "last.pth"))

    ck = torch.load(os.path.join(out, "best.pth"), map_location="cpu", weights_only=True)
    bm = ResidualI0()
    bm.load_state_dict(ck["state_dict"])
    bm.to(dev).eval()
    final, rows = run_eval(bm, val, dev)

    write_csv(os.path.join(out, "per_file_metrics.csv"),
              [["file", "condition", "samples", "mae_A", "rmse_A", "p95_A", "bias_A",
                "persistence_mae_A"]] +
              [[r["name"], fam(r["name"]), int(r["mask"].sum()),
                "%.6f" % np.abs(r["pred"][r["mask"]] - r["target"][r["mask"]]).mean(),
                "%.6f" % np.sqrt(((r["pred"][r["mask"]] - r["target"][r["mask"]]) ** 2).mean()),
                "%.6f" % np.percentile(np.abs(r["pred"][r["mask"]]
                                              - r["target"][r["mask"]]), 95),
                "%.6f" % (r["pred"][r["mask"]] - r["target"][r["mask"]]).mean(),
                "%.6f" % np.abs(r["base"][r["mask"]] - r["target"][r["mask"]]).mean()]
               for r in sorted(rows, key=lambda z: z["name"])])

    write_csv(os.path.join(out, "per_condition_metrics.csv"),
              [["condition", "samples", "mae_A", "rmse_A", "p95_A", "bias_A",
                "persistence_mae_A", "persistence_rmse_A", "mae_improvement_pct"]] +
              [[c, final["per_type"][c]["n"],
                "%.6f" % final["per_type"][c]["mae"],
                "%.6f" % final["per_type"][c]["rmse"],
                "%.6f" % final["per_type"][c]["p95"],
                "%.6f" % final["per_type"][c]["bias"],
                "%.6f" % final["per_type"][c]["persistence_mae"],
                "%.6f" % final["per_type"][c]["persistence_rmse"],
                "%.1f" % (100 * (1 - final["per_type"][c]["mae"]
                                 / max(final["per_type"][c]["persistence_mae"], 1e-9)))]
               for c in CATS if final["per_type"][c]])

    write_csv(os.path.join(out, "persistence_comparison.csv"),
              [["model", "mae_A", "rmse_A", "median_A", "p95_A", "p99_A", "bias_A"]] +
              [["V25_2_RI0_CMD_PROXY", "%.6f" % final["mae"], "%.6f" % final["rmse"],
                "%.6f" % final["median"], "%.6f" % final["p95"], "%.6f" % final["p99"],
                "%+.6f" % final["bias"]],
               ["PERSISTENCE", "%.6f" % final["persistence"]["mae"],
                "%.6f" % final["persistence"]["rmse"],
                "%.6f" % final["persistence"]["median"],
                "%.6f" % final["persistence"]["p95"],
                "%.6f" % final["persistence"]["p99"],
                "%+.6f" % final["persistence"]["bias"]]])

    json.dump(dict(threshold_A=ZERO_TH, base="clip(|I_meas[k]|,0,1.6) proxy",
                   **final["zero_current"]),
              open(os.path.join(out, "proxy_command_metrics.json"), "w",
                   encoding="utf-8"), indent=2, ensure_ascii=False)

    x0 = build_v252(val[0], NORM)[0]
    tm = dict(cpu=timing(bm, x0, "cpu"), rate_hz=100, budget_ms=10.0)
    if torch.cuda.is_available():
        tm["cuda"] = timing(bm, x0, "cuda")
    mem = sum(p.numel() * p.element_size() for p in bm.parameters())

    summary = dict(experiment_id="V25_2_RI0_CMD_PROXY_seed7", seed=a.seed,
                   completed_epochs=a.epochs, best_epoch=best["epoch"],
                   best_by_val_mae_epoch=best_mae["epoch"], params=pc,
                   checkpoint_bytes=os.path.getsize(os.path.join(out, "best.pth")),
                   param_memory_bytes=mem,
                   validation=final, timing=tm, train_seconds=wall,
                   final_test_read=False,
                   limitations=[
                       "labels are PROXY commands derived from measured current under "
                       "I_act[k] ~= I_cmd[k-1]; real ECU command accuracy is unverified",
                       "single seed 7",
                       "SR and RHF validation each have exactly 1 file",
                       "no digital filter: the physical system delay is unchanged"])
    json.dump(summary, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print("V25.2 DONE mae=%.4f rmse=%.4f p95=%.4f p99=%.4f bias=%+.4f nrmse=%.3f%% "
          "pers_mae=%.4f zero_pred=%.4f clip=%.2f%% best_ep=%d wall=%.1f min"
          % (final["mae"], final["rmse"], final["p95"], final["p99"], final["bias"],
             final["pooled_nrmse"], final["persistence"]["mae"],
             final["zero_current"]["pred_mean"], 100 * final["clip_activation"],
             best["epoch"], wall / 60), flush=True)


if __name__ == "__main__":
    main()
