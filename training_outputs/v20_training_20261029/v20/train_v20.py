"""Train one v20 forward model. No training on import or without
--execute-training. Full-file batches of one, exactly like the v19 runner:
no invented current prehistory at random chunk starts, no final_test data.

Usage (Windows cmd / PowerShell, and Linux/macOS alike):
  python train_v20.py --data <Parsed_v19_F42_I42> --model B ^
      --output <run_dir> --epochs 30 --execute-training

Resume a session to a larger TOTAL epoch count (same model, data, seed):
  python train_v20.py --resume <run_dir> --epochs 60 --execute-training
  --epochs is the TOTAL count, not additional epochs, and the cosine LR
  schedule is recomputed from (epoch, total), as in the v19 runner.

Smoke test only (never use for selection): add --max-train-files 2
--max-val-files 2 and make the output directory name contain 'smoke'.
"""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch

from v20_data import digest, edge_guard, normalization, read_dataset, sample
from v20_models import ARCHES, V20Model


def evaluate(model, items, norm, device):
    model.eval()
    rows = []
    for it in items:
        u, f, valid = sample(it, norm)
        u, f, valid = u.to(device), f.to(device), valid.to(device)
        with torch.no_grad():
            pred = model(u)
            err = (pred - f)[valid] * norm["f_scale"]
            y = f[valid] * norm["f_scale"]
            sse, sst = float(err.square().sum()), float(y.square().sum())
        rows.append(
            dict(
                name=it["name"],
                samples=int(valid.sum()),
                SSE=sse,
                SST=sst,
                nrmse_pct=100 * math.sqrt(sse / max(sst, 1e-12)),
            )
        )

    def pooled(subset):
        # None (not NaN) when a type is absent, so summary JSON stays valid.
        return (
            100 * math.sqrt(sum(r["SSE"] for r in subset) / max(sum(r["SST"] for r in subset), 1e-12))
            if subset
            else None
        )

    def by_prefix(prefix):
        return [r for r in rows if r["name"].startswith(prefix)]

    result = dict(
        pooled_nrmse_pct=pooled(rows),
        hf_nrmse_pct=pooled([r for r in rows if r["name"].startswith("SH_") and "6.62Hz" in r["name"]]),
        per_type_pct={k: pooled(by_prefix(k)) for k in ("SH_", "ST_", "RLF_", "RHF_", "SR_")},
        worst_file_pct=max((r["nrmse_pct"] for r in rows), default=None),
        rows=rows,
        history_protocol="causal GRU over the full file; no tau, no lag module",
    )
    return result


def validation_lag(model, items, norm, device, max_shift=100):
    """Per-file best lag (ms) for the selected model only, not a training
    signal. lag_ms is the shift s applied to the prediction that minimises
    MSE against the true force on the guarded (valid) samples; s > 0 means
    the prediction must be shifted forward to match, i.e. it lags the truth.
    Computed once at the end, on validation only."""
    model.eval()
    out = {}
    for it in items:
        u, f, valid = sample(it, norm)
        u, f = u.to(device), f.to(device)
        with torch.no_grad():
            pred = model(u)[0] * norm["f_scale"]
        true = f[0] * norm["f_scale"]
        valid1 = valid[0]
        best = (None, float("inf"))
        for s in range(-max_shift, max_shift + 1):
            if s >= 0:
                a, b = pred[: len(pred) - s] if s else pred, true[s:]
                keep = valid1[: len(a)] if s else valid1
            else:
                a, b = pred[-s:], true[: len(true) + s]
                keep = valid1[-s:]
            if keep.sum() < 100:
                continue
            mse = float(((a - b)[keep]).square().mean())
            if mse < best[1]:
                best = (s, mse)
        out[it["name"]] = dict(best_lag_ms=best[0], best_lag_mse_N2=best[1])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, help="Parsed_v19_F42_I42 directory")
    p.add_argument("--model", choices=sorted(ARCHES), help="v20 letter: A,B,C,D,E,F")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--max-train-files", type=int, default=0, help="smoke only")
    p.add_argument("--max-val-files", type=int, default=0, help="smoke only")
    p.add_argument("--execute-training", action="store_true")
    a = p.parse_args()
    if not a.execute_training:
        print("NO TRAINING. Add --execute-training to run. Review README_v20.md first.")
        return
    if a.resume is not None:
        if a.output is not None or a.model is not None or a.data is not None:
            p.error("--resume takes the session directory; do not pass --output/--model/--data")
        if not (1 <= a.epochs <= 500):
            p.error("--epochs must be a total count in 1..500")
        a.output = a.resume
    else:
        if a.data is None or a.model is None or a.output is None or not (1 <= a.epochs <= 500):
            p.error("--data, --model, --output and --epochs (total, 1..500) are required")
    smoke = bool(a.max_train_files or a.max_val_files)
    if smoke and "smoke" not in str(a.output):
        p.error("smoke runs (limited files) must have 'smoke' in the output directory name")
    device = torch.device("cuda" if (a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())) else "cpu")
    print(f"device={device} model={a.model} epochs(total)={a.epochs} smoke={smoke}", flush=True)

    if a.resume is not None:
        orig = json.loads((a.output / "run_config.json").read_text(encoding="utf-8"))
        if orig.get("smoke"):
            raise ValueError("smoke sessions (limited files) cannot be resumed; start a full run")
        if orig.get("seed") != a.seed:
            raise ValueError("resume seed must match the original session seed")
        a.data = Path(orig["data"])
        a.model = orig["model"]
        if a.max_train_files or a.max_val_files:
            raise ValueError("do not combine --resume with smoke file limits")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    train = read_dataset(a.data, "train")
    validation = read_dataset(a.data, "validation")
    if a.max_train_files:
        train = train[: a.max_train_files]
    if a.max_val_files:
        validation = validation[: a.max_val_files]
    norm = normalization(read_dataset(a.data, "train") if smoke else train)
    # Note: for a full run, norm is computed from the full train set (train
    # above). For a smoke run, norm still uses the full train set, so smoke
    # numbers are on the same scale but are NOT comparable to full runs.
    manifest_hash = digest(Path(a.data) / "alignment_manifest.json")
    arch = ARCHES[a.model]

    if a.resume is None:
        if a.output.exists():
            raise ValueError("output exists; never overwrite a training session")
        model = V20Model(arch=arch)
        start_epoch, best_score, transfer = 0, float("inf"), {}
        latest = None
    else:
        latest = torch.load(a.output / "latest.pth", map_location="cpu", weights_only=True)
        if latest.get("model") != a.model or latest.get("data_manifest_sha256") != manifest_hash:
            raise ValueError("resume model/data mismatch")
        if latest.get("norm_cfg") != norm:
            raise ValueError("resume normalisation mismatch")
        model = V20Model(arch=latest["arch"])
        model.load_state_dict(latest["state_dict"])
        start_epoch = int(latest["epoch"])
        if start_epoch >= a.epochs:
            raise ValueError("session already at total epochs; increase --epochs")
        best_score = float(latest.get("best_score", float("inf")))
        transfer = {}
    model.to(device)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-4)
    if a.resume is not None:
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        random.setstate(latest["random_state"])
        torch.set_rng_state(latest["torch_rng_state"])
        if "train_order" not in latest:
            raise ValueError("latest.pth predates train-order capture; cannot resume strictly")
        by_name = {it["name"]: it for it in train}
        if set(by_name) != set(latest["train_order"]):
            raise ValueError("train-order names do not match the current dataset")
        train = [by_name[n] for n in latest["train_order"]]
    elif True:
        a.output.mkdir(parents=True)
        config = dict(
            model=a.model,
            arch=arch,
            model_cfg=model.config,
            parameter_count=sum(q.numel() for q in parameters),
            data=str(a.data),
            data_manifest_sha256=manifest_hash,
            norm_cfg=norm,
            seed=a.seed,
            epochs_total=a.epochs,
            batch="one full file",
            augmentation="none",
            delay="GRU hidden state only; no tau, no CurrentLag, no TauNet",
            inputs="[x, v, a, I, T]; dI dropped",
            smoke=smoke,
            device=str(device),
        )
        (a.output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for epoch in range(start_epoch + 1, a.epochs + 1):
        start = time.perf_counter()
        model.train()
        random.shuffle(train)
        factor = min(epoch / 5, 1.0) if epoch <= 5 else 0.01 + 0.99 * (1 + math.cos(math.pi * (epoch - 5) / max(a.epochs - 5, 1))) / 2
        for group in optimizer.param_groups:
            group["lr"] = 3e-4 * factor
        losses = []
        for it in train:
            u, f, valid = sample(it, norm)
            u, f, valid = u.to(device), f.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(u)
            loss = ((pred - f)[valid]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, validation, norm, device)
        metrics.update(epoch=epoch, train_loss=sum(losses) / len(losses), seconds=time.perf_counter() - start)
        with (a.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        score = metrics["pooled_nrmse_pct"]
        if score < best_score:
            best_score = score
            torch.save(
                dict(model=a.model, arch=arch, state_dict=copy.deepcopy(model.state_dict()),
                     model_cfg=model.config, norm_cfg=norm, epoch=epoch, metrics=metrics,
                     data_manifest_sha256=manifest_hash),
                a.output / "best.pth",
            )
        torch.save(
            dict(model=a.model, arch=arch, state_dict=copy.deepcopy(model.state_dict()),
                 model_cfg=model.config, norm_cfg=norm, epoch=epoch, metrics=metrics,
                 data_manifest_sha256=manifest_hash, optimizer_state_dict=optimizer.state_dict(),
                 random_state=random.getstate(), torch_rng_state=torch.get_rng_state(),
                 train_order=[it["name"] for it in train], best_score=best_score),
            a.output / "latest.pth",
        )
        hf_txt = f"{metrics['hf_nrmse_pct']:.4f}%" if metrics["hf_nrmse_pct"] is not None else "n/a"
        print(f"epoch={epoch} validation_pooled={metrics['pooled_nrmse_pct']:.4f}% "
              f"hf={hf_txt} train_loss={metrics['train_loss']:.6f} "
              f"({metrics['seconds']:.1f}s)", flush=True)

    # End of run: best-model validation predictions + lag, validation only.
    best = torch.load(a.output / "best.pth", map_location="cpu", weights_only=True)
    best_model = V20Model(arch=best["arch"])
    best_model.load_state_dict(best["state_dict"])
    best_model.to(device)
    lag = validation_lag(best_model, validation, norm, device)
    summary = dict(
        model=a.model,
        arch=arch,
        best_epoch=best["epoch"],
        best_pooled_nrmse_pct=best["metrics"]["pooled_nrmse_pct"],
        best_hf_nrmse_pct=best["metrics"]["hf_nrmse_pct"],
        best_per_type_pct=best["metrics"]["per_type_pct"],
        best_worst_file_pct=best["metrics"]["worst_file_pct"],
        parameter_count=sum(q.numel() for q in parameters),
        validation_lag_ms=lag,
        smoke=smoke,
        note="validation only; final_test was never opened",
    )
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
