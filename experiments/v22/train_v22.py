"""Train one V22 model and emit every artifact the task book section 4 requires.

    python train_v22.py --model EGT --seed 7 --output D:/v22_ae_hybrid/runs/EGT_seed7

Fixed public config: 130 epochs one-shot, AdamW lr 3e-4 wd 1e-4, 5-epoch linear
warmup then cosine, grad clip 1.0, plain MSE on f/f_scale, one full file per step,
best.pth by validation pooled NRMSE.  final_test is never opened.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v20_data import digest, normalization, read_dataset, sample        # noqa: E402
from v22_models import MODELS, V22Model, count_params                   # noqa: E402

CATS = ("SH_", "ST_", "RLF_", "RHF_", "SR_")
MAIN_DF, SUPP_DF = 250.0, 50.0
TH_DI = 0.30


# ---------------------------------------------------------------- helpers
def code_hash():
    out = {}
    for f in ("v22_models.py", "v20_data.py", "train_v22.py"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            out[f] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    try:
        out["git"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                             stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        out["git"] = None
    return out


def write_csv(path, rows, header=None):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


def evaluate(model, items, norm, device, keep=False):
    model.eval()
    rows, keepers = [], {}
    fs = norm["f_scale"]
    for it in items:
        u, f, valid = sample(it, norm)
        with torch.no_grad():
            pred = model(u.to(device))
        err = (pred - f.to(device))[0][valid[0]].cpu() * fs
        y = f[0][valid[0]] * fs
        sse, sst = float(err.square().sum()), float(y.square().sum())
        rows.append(dict(name=it["name"], samples=int(valid.sum()), sse=sse, sst=sst,
                         nrmse=100 * math.sqrt(sse / max(sst, 1e-12)),
                         rmse=math.sqrt(sse / max(int(valid.sum()), 1))))
        if keep:
            keepers[it["name"]] = dict(pred=(pred[0] * fs).cpu().numpy(),
                                       truth=(f[0] * fs).numpy(),
                                       valid=valid[0].numpy(),
                                       cur=it["values"][:, 3].numpy(),
                                       vel=it["values"][:, 1].numpy(),
                                       acc=it["values"][:, 2].numpy(),
                                       pos=it["values"][:, 0].numpy())

    def pooled(sub):
        if not sub:
            return None
        return 100 * math.sqrt(sum(r["sse"] for r in sub) / max(sum(r["sst"] for r in sub), 1e-12))

    by = {c: pooled([r for r in rows if r["name"].startswith(c)]) for c in CATS}
    hf = pooled([r for r in rows if r["name"].startswith("SH_") and "6.62Hz" in r["name"]])
    return dict(pooled=pooled(rows), per_type=by, hf=hf,
                worst=max((r["nrmse"] for r in rows), default=None), rows=rows), keepers


# ---------------------------------------------------------- step response
def step_events(keep, df_min):
    """Same detector for every model: current step, velocity-drift corrected,
    10% current -> 90% force (the V20-compatible definition)."""
    out = []
    for name, d in keep.items():
        if not name.startswith("ST_"):
            continue
        I, V, T, P = d["cur"], d["vel"], d["truth"], d["pred"]
        hit = np.where(np.abs(np.diff(I)) > 0.04)[0]
        if not len(hit):
            continue
        groups, a, b = [], hit[0], hit[0]
        for q in hit[1:]:
            if q - b <= 40:
                b = q
            else:
                groups.append((a, b)); a = b = q
        groups.append((a, b))
        for a, b in groups:
            if a < 1200 or b > len(I) - 1200:
                continue
            ip = float(np.median(I[a - 80:a - 10]))
            iq = float(np.median(I[b + 40:b + 120]))
            if abs(iq - ip) < TH_DI:
                continue
            fw, vw = T[a - 260:a - 20], V[a - 260:a - 20]
            beta = 0.0
            if len(fw) > 20 and np.ptp(vw) >= 5:
                A = np.vstack([vw, np.ones_like(vw)]).T
                beta = float(np.linalg.lstsq(A, fw, rcond=None)[0][0])
            vp = float(np.median(V[a - 80:a - 10]))
            vq = float(np.median(V[b + 40:b + 120]))

            def one(curve):
                corr = curve - beta * (V - vp)
                fb = float(np.median(corr[a - 80:a - 10]))
                fa = float(np.median(corr[b + 40:b + 120]))
                dF = fa - fb
                if abs(dF) < df_min:
                    return None
                sgn = np.sign(dF)

                def cross(base_v, target, frac):
                    thr = base_v + sgn * frac * abs(target - base_v)
                    seg = corr[a - 20:min(len(corr), b + 400)] - thr
                    idx = np.where(seg * sgn >= 0)[0]
                    return a - 20 + int(idx[0]) if len(idx) else None
                t10 = cross(ip, iq, 0.10)
                t90 = cross(fb, fa, 0.90)
                if t10 is None or t90 is None:
                    return None
                return dict(dF=dF, fb=fb, fa=fa, lag=t90 - t10)

            tr, pr = one(T), one(P)
            if tr is None or pr is None:
                continue
            out.append(dict(file=name, t_edge=a / 1000.0, dI=iq - ip, dF=tr["dF"],
                            i_dir=int(np.sign(iq - ip)) or 1,
                            v_sign=int(np.sign(vp)) or 1,
                            lag_true=float(tr["lag"]), lag_pred=float(pr["lag"]),
                            lag_err=float(pr["lag"] - tr["lag"]),
                            amp_err=abs(pr["fa"] - tr["fa"])))
    out.sort(key=lambda r: (r["file"], r["t_edge"]))
    return out


# ------------------------------------------------------------ hysteresis
def hysteresis(keep):
    rows = []
    for name, d in keep.items():
        if not name.startswith("SH_"):
            continue
        x, F, T, P = d["pos"], d["truth"], d["truth"], d["pred"]
        v = d["valid"].astype(bool)
        if v.sum() < 500:
            continue
        mid = 0.5 * (x[v].min() + x[v].max())
        s = np.sign(x - mid)
        cross = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
        if len(cross) < 2:
            continue
        for i in range(len(cross) - 1):
            a, b = cross[i], cross[i + 1]
            if b - a < 100:
                continue
            sl = slice(a, b + 1)
            _trap = getattr(np, "trapezoid", None) or np.trapz
            area_t = float(_trap(T[sl], x[sl]))
            area_p = float(_trap(P[sl], x[sl]))
            rows.append(dict(file=name, start_sample=int(a), end_sample=int(b),
                             measured_N_mm=area_t, predicted_N_mm=area_p,
                             abs_error=abs(area_p - area_t),
                             rel_error=abs(area_p - area_t) / max(abs(area_t), 1e-9),
                             status="complete_SH_cycle"))
    return rows


# ------------------------------------------- acceleration residual diagnostics
def accel_diagnostics(keep, model, norm, device, items):
    """PCC of the residual with a, v, x (task book section 6 requires recomputing
    these on this data instead of quoting the paper)."""
    res, aa, vv, xx = [], [], [], []
    for it in items:
        u, f, valid = sample(it, norm)
        with torch.no_grad():
            p = model(u.to(device))
        e = ((p - f.to(device))[0][valid[0]]).cpu().numpy()
        res.append(e)
        aa.append(it["values"][:, 2].numpy()[valid[0].numpy()])
        vv.append(it["values"][:, 1].numpy()[valid[0].numpy()])
        xx.append(it["values"][:, 0].numpy()[valid[0].numpy()])
    r = np.concatenate(res); a = np.concatenate(aa)
    v = np.concatenate(vv); x = np.concatenate(xx)
    def pcc(p_, q_):
        if np.std(p_) < 1e-12 or np.std(q_) < 1e-12:
            return None
        return float(np.corrcoef(p_, q_)[0, 1])
    return dict(residual_rms=float(np.sqrt((r ** 2).mean())),
                pcc_with_acceleration=pcc(r, a),
                pcc_with_velocity=pcc(r, v),
                pcc_with_displacement=pcc(r, x),
                n_samples=int(len(r)),
                note="residual = prediction - measurement, normalised force units")


# ---------------------------------------------------------- monotonicity
def monotonicity(model, n_state=36, n_current=101):
    model.eval()
    dev = next(model.parameters()).device
    rng = torch.Generator().manual_seed(0)
    def rnd(g):
        return float(torch.rand(1, generator=g))
    viol = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_state):
            z = (torch.randn(1, 1, 16, generator=rng) * 0.5).to(dev)
            v_hat = torch.tensor([[(rnd(rng) - 0.3) * 0.4]], device=dev)
            a_hat = torch.tensor([[(rnd(rng) - 0.5) * 0.2]], device=dev)
            q_prev = torch.rand(1, 1, generator=rng).to(dev)
            prev = None
            for i in np.linspace(0, 1.6, n_current):
                f, _ = model.trial_force(z, v_hat, a_hat, q_prev,
                                         torch.tensor([[float(i)]], device=dev))
                cur = float(f)
                if prev is not None:
                    total += 1
                    expect_non_increasing = float(v_hat) > 0
                    if expect_non_increasing and cur > prev + 1e-5:
                        viol += 1
                    if (not expect_non_increasing) and cur < prev - 1e-5:
                        viol += 1
                prev = cur
    return dict(violations=viol, comparisons=total,
                tolerance=1e-5, n_states=n_state, n_current_points=n_current)


# --------------------------------------------------------- 100 Hz inverse
def inverse_100hz(model, keep, items, norm, device, dt=0.01, iters=24):
    if model.model_id == "E0":
        return dict(status="NOT_APPLICABLE",
                    note="E0 has no candidate-current path; forward baseline only")
    model.eval()
    name = next((k for k in keep if k.startswith("SH_")), None)
    item = next((x for x in items if x["name"] == name), None) if name else None
    if item is None:
        return dict(status="SKIP", note="no SH file for inverse check")
    d = keep[name]
    v, a, T = d["vel"], d["acc"], d["truth"]
    step = int(round(dt * 1000))
    idx = np.arange(0, len(v), step)
    # observer state is computed once at 1 kHz and then HELD FIXED across the
    # 100 Hz step, so no candidate trial can advance it.
    u_full, _, _ = sample(item, norm)
    with torch.no_grad():
        z_full, _ = model.observe(u_full.to(device))
    z_np = z_full[0].cpu().numpy()
    q_prev = 1.0
    resid, times, saturated = [], [], 0
    for k in idx:
        z = torch.from_numpy(z_np[k]).float().view(1, 1, 16).to(device)
        vh = torch.tensor([[float(v[k] / 1000.0)]], device=device)
        ah = torch.tensor([[float(a[k] / 50000.0)]], device=device)
        target = float(T[k])
        qp = torch.tensor([[q_prev]], device=device)
        t0 = time.perf_counter()
        lo, hi = 0.0, 1.6
        with torch.no_grad():
            f_lo, _ = model.trial_force(z, vh, ah, qp, torch.tensor([[lo]], device=device))
            f_hi, _ = model.trial_force(z, vh, ah, qp, torch.tensor([[hi]], device=device))
        flo, fhi = float(f_lo), float(f_hi)
        if not (min(flo, fhi) <= target <= max(flo, fhi)):
            saturated += 1
            times.append((time.perf_counter() - t0) * 1000)
            continue
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            with torch.no_grad():
                fm, qm = model.trial_force(z, vh, ah, qp,
                                           torch.tensor([[mid]], device=device))
            if (float(fm) > target) == (flo > fhi):
                lo = mid
                flo = float(fm)
            else:
                hi = mid
        mid = 0.5 * (lo + hi)
        with torch.no_grad():
            fm, qm = model.trial_force(z, vh, ah, qp,
                                       torch.tensor([[mid]], device=device))
        times.append((time.perf_counter() - t0) * 1000)
        q_prev = float(qm)
        resid.append(abs(float(fm) - target))
    times = np.array(times) if times else np.array([0.0])
    resid = np.array(resid) if resid else np.array([0.0])
    return dict(status="PASS", file=name, dt_s=dt, iterations=iters,
                n_steps=int(len(idx)), n_saturated=int(saturated),
                saturated_fraction=float(saturated / max(len(idx), 1)),
                residual_mean_N=float(resid.mean()), residual_max_N=float(resid.max()),
                time_mean_ms=float(times.mean()), time_p95_ms=float(np.percentile(times, 95)),
                time_p99_ms=float(np.percentile(times, 99)),
                benchmark="Python reference (torch on CUDA, 1 sample per trial); "
                          "not an ECU WCET measurement")


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--data", default=r"D:/v20_data/Parsed_v19_F42_I42")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=130)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--execute-training", action="store_true")
    a = ap.parse_args()
    if not a.execute_training:
        print("dry: pass --execute-training"); return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - refusing to fall back to CPU")
    device = torch.device("cuda:0")
    out = a.output
    if os.path.exists(out):
        raise SystemExit("output exists, refusing to overwrite: " + out)
    os.makedirs(out)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)

    random.seed(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    train = read_dataset(a.data, "train")
    val = read_dataset(a.data, "validation")
    norm = normalization(train)
    man = digest(os.path.join(a.data, "alignment_manifest.json"))
    model = V22Model(a.model).to(device)
    pc = count_params(model)
    if pc["total"] > 3500:
        raise SystemExit("parameter budget exceeded: %d" % pc["total"])
    print("%s params=%d seed=%d train=%d val=%d device=%s"
          % (a.model, pc["total"], a.seed, len(train), len(val), device), flush=True)

    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    cfg = dict(model=a.model, backbone=model.backbone, use_g=model.use_g,
               use_tau=model.use_tau, use_acc=model.use_acc, seed=a.seed,
               epochs_total=a.epochs, data=a.data, manifest_sha256=man,
               n_train_files=len(train), n_validation_files=len(val), norm=norm,
               params=pc, optimizer=dict(name="AdamW", lr=3e-4, weight_decay=1e-4),
               schedule="5-epoch linear warmup + cosine to 0.01*base_lr",
               grad_clip=1.0, loss="plain (f/f_scale - pred)^2 MSE, edge mask",
               batch="one full file per step, states reset per file",
               device="cuda:0", gpu=torch.cuda.get_device_name(0),
               final_test_read=False, code=code_hash(),
               causality="observer uses I[t-1]; candidate current only via G->q->gain")
    cfg["config_sha256"] = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    json.dump(cfg, open(os.path.join(out, "run_config.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    best = dict(score=float("inf"), epoch=0, metrics=None)
    fs = norm["f_scale"]
    for epoch in range(1, a.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        random.shuffle(train)
        lr = 3e-4 * (min(epoch / 5, 1.0) if epoch <= 5 else
                     0.01 + 0.99 * (1 + math.cos(math.pi * (epoch - 5) / max(a.epochs - 5, 1))) / 2)
        for g in opt.param_groups:
            g["lr"] = lr
        tot = 0.0
        for it in train:
            u, f, valid = sample(it, norm)
            u, f, valid = u.to(device), f.to(device), valid.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(u)
            loss = ((pred - f)[valid]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("non-finite loss at epoch %d" % epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss.detach())
        m, _ = evaluate(model, val, norm, device)
        m.update(epoch=epoch, train_loss=tot / len(train), seconds=time.perf_counter() - t0,
                 lr=lr)
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
        if epoch % 10 == 1 or epoch == a.epochs:
            extra = ""
            if model.use_tau:
                extra += " tau=%.2fms" % m["tau_ms"]
            if model.use_acc:
                extra += " b_a=%+.4f" % m["b_a"]
            print("epoch=%d/130 val=%.3f%% train_loss=%.6f (%.1fs)%s"
                  % (epoch, m["pooled"], m["train_loss"], m["seconds"], extra), flush=True)
    train_seconds = time.perf_counter() - t_start
    torch.save(dict(model=a.model, state_dict=copy.deepcopy(model.state_dict()),
                    norm=norm, epoch=a.epochs, params=pc), os.path.join(out, "last.pth"))

    # ---- final evaluation with the best checkpoint ----------------------
    ck = torch.load(os.path.join(out, "best.pth"), map_location="cpu", weights_only=True)
    bm = V22Model(a.model); bm.load_state_dict(ck["state_dict"]); bm.to(device).eval()
    final, keep = evaluate(bm, val, norm, device, keep=True)

    write_csv(os.path.join(out, "per_file_metrics.csv"),
              [["file", "samples", "nrmse_pct", "rmse_N", "sse", "sst"]] +
              [[r["name"], r["samples"], "%.6f" % r["nrmse"], "%.3f" % r["rmse"],
                "%.6e" % r["sse"], "%.6e" % r["sst"]] for r in sorted(final["rows"],
                                                                     key=lambda z: z["name"])])
    per_cond = [[c, "%.6f" % final["per_type"][c]] if final["per_type"][c] is not None
                else [c, ""] for c in CATS]
    write_csv(os.path.join(out, "per_condition_metrics.csv"),
              [["condition", "pooled_nrmse_pct"]] + per_cond +
              [["hf_6.62Hz", "%.6f" % final["hf"] if final["hf"] else ""],
               ["worst_file", "%.6f" % final["worst"]]])

    ev_main = step_events(keep, MAIN_DF)
    ev_supp = step_events(keep, SUPP_DF)
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

    acc_diag = accel_diagnostics(keep, bm, norm, device, val)
    json.dump(acc_diag, open(os.path.join(out, "acceleration_residual_diagnostics.json"),
                             "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    mono = monotonicity(bm) if bm.use_g else dict(status="NOT_APPLICABLE",
                                                  note="E0 has no current gain path")
    json.dump(mono, open(os.path.join(out, "monotonicity_report.json"), "w",
                         encoding="utf-8"), indent=2, ensure_ascii=False)
    inv = inverse_100hz(bm, keep, val, norm, device)
    json.dump(inv, open(os.path.join(out, "inverse_100hz_report.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)

    peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
    peak_res = torch.cuda.max_memory_reserved() / 2 ** 20
    res = dict(params=pc, weight_bytes=os.path.getsize(os.path.join(out, "best.pth")),
               train_seconds=train_seconds,
               seconds_per_epoch=train_seconds / a.epochs,
               cuda_peak_allocated_MiB=peak_alloc, cuda_peak_reserved_MiB=peak_res,
               gpu=torch.cuda.get_device_name(0),
               throughput_samples_per_s=None,
               note="throughput = train samples x epochs / wall clock")
    res["throughput_samples_per_s"] = (sum(i["values"].shape[0] for i in train) * a.epochs
                                       / train_seconds)
    json.dump(res, open(os.path.join(out, "resource_report.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    def med(v):
        return float(np.median(v)) if v else None
    summary = dict(model=a.model, seed=a.seed, completed_epochs=a.epochs,
                   required_epochs=a.epochs, status="PASS",
                   best_epoch=best["epoch"], params=pc,
                   validation=dict(pooled_nrmse_pct=best["score"],
                                   per_type_pct=best["metrics"]["per_type"],
                                   hf_nrmse_pct=best["metrics"]["hf"],
                                   worst_file_pct=best["metrics"]["worst"]),
                   step_250N=dict(n=len(ev_main),
                                  lag_true_median=med([e["lag_true"] for e in ev_main]),
                                  lag_pred_median=med([e["lag_pred"] for e in ev_main]),
                                  lag_err_median=med([e["lag_err"] for e in ev_main])),
                   step_50N=dict(n=len(ev_supp),
                                 lag_true_median=med([e["lag_true"] for e in ev_supp]),
                                 lag_pred_median=med([e["lag_pred"] for e in ev_supp]),
                                 lag_err_median=med([e["lag_err"] for e in ev_supp])),
                   hysteresis_cycles=len(hyst),
                   hysteresis_rel_error_mean=(float(np.mean([h["rel_error"] for h in hyst]))
                                              if hyst else None),
                   monotonicity=mono, inverse=inv, acceleration=acc_diag,
                   tau_ms=(float(bm.tau_ms().detach().cpu()) if bm.use_tau else None),
                   b_a=(float(bm.b_a.detach().cpu()) if bm.use_acc else None),
                   final_test_read=False,
                   limitations=["single seed 7: architecture screening only",
                                "step metric is I10->F90 (V20-compatible), not force rise time",
                                "inverse timing is a Python CPU reference, not ECU WCET"])
    json.dump(summary, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in ("model", "best_epoch", "tau_ms", "b_a")}, ensure_ascii=False))
    print("POOLED %.4f  WORST %.2f  STEP250 %s  INV %s"
          % (summary["validation"]["pooled_nrmse_pct"],
             summary["validation"]["worst_file_pct"],
             summary["step_250N"]["lag_err_median"],
             inv.get("time_p99_ms")), flush=True)


if __name__ == "__main__":
    main()
