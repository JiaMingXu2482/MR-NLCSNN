"""
Hyperparameter sweep for NLCSNN — Plan.md §III protocol:
descending-order stress test, VRAM guard, RWS sampling.

Usage:
  python sweep.py --sweep h_dim --values 16,12,8
  python sweep.py --sweep batch_size --values 64,48,32
  python sweep.py --sweep hidden --values 256,128,64
  python sweep.py --sweep h_dim --values 16,12,8 --parallel
"""
import os
import sys
import time
import argparse
import random
import signal
import traceback
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from mr_nlcsnn.model import NLCSNN, rk4_step
from AA02Train.TrainFinalVersion import ParallelDataManager

try:
    from train_config import PROCESSED_GROUPS
except ImportError:
    PROCESSED_GROUPS = r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Processed_Groups"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def _trainable_module(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def estimate_vram_gb(batch_size, seq_len, hidden, n_params):
    """Rough VRAM estimate: model params + batch tensors overhead."""
    param_gb = n_params * 4 / 1e9
    tensor_gb = (batch_size * seq_len * hidden * 4) / 1e9
    return param_gb + tensor_gb * 3  # activations, grads, etc.


def check_vram():
    return torch.cuda.memory_reserved() / (1024 ** 3)


# ── Loss (matches TrainFinalVersion.rws_loss) ──

def rws_loss(model, criterion, u_seq, f_seq, p2p_sq):
    batch_size = u_seq.shape[1]
    h_dim = _trainable_module(model).h_dim
    h = torch.zeros(batch_size, h_dim, device=device)
    force_loss = 0.0
    l_latent = 0.0
    for t in range(u_seq.shape[0]):
        u_t = u_seq[t]
        pred = model.predict_force(u_t, h).squeeze(-1)
        h = rk4_step(model, u_t, h)
        target = f_seq[t]
        sign_w = torch.where(target < 0, 1.5, 1.0)
        step_mse = criterion(pred, target)
        force_loss = force_loss + (step_mse * sign_w / p2p_sq).mean()
        l_latent = l_latent + torch.mean(torch.norm(h, dim=-1) ** 2)
    seq_len = u_seq.shape[0]
    return force_loss / seq_len, l_latent / seq_len


# ── Validation ──

@torch.no_grad()
def validate(model, manager, val_files, max_samples=1500, exclude=50):
    """Return overall mean NRMSE + per-category breakdown."""
    nrmse_list = []
    cat_nrmse = {}
    h_dim = _trainable_module(model).h_dim
    for item in val_files:
        h = torch.zeros(1, h_dim, device=device)
        n = min(len(item["u"]), max_samples)
        preds = []
        for t in range(n):
            preds.append(model.predict_force(item["u"][t:t + 1], h).item())
            h = rk4_step(model, item["u"][t:t + 1], h)
        p = np.asarray(preds, dtype=np.float32)
        y = item["f"][:n].cpu().numpy()
        start = min(exclude, max(0, n // 4))
        rmse = np.sqrt(np.mean((p[start:] - y[start:]) ** 2))
        nrmse = rmse / (y[start:].max() - y[start:].min() + 1e-6) * 100
        nrmse_list.append(nrmse)
        cat = item.get("category", "unknown")
        cat_nrmse.setdefault(cat, []).append(nrmse)
    overall = float(np.mean(nrmse_list))
    breakdown = {c: float(np.mean(v)) for c, v in sorted(cat_nrmse.items())}
    return overall, breakdown


# ── Anti-crash protections ──

_CRASH_LOG = None


def _crash_handler(signum, frame):
    """Log signal termination reason."""
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    msg = f"\n[!!] Process killed by signal {signum} ({sig_name})\n"
    sys.stderr.write(msg)
    sys.stderr.flush()
    if _CRASH_LOG is not None:
        with open(_CRASH_LOG, "a") as f:
            f.write(msg)
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _crash_handler)
signal.signal(signal.SIGINT, _crash_handler)


def _set_crash_log(path):
    global _CRASH_LOG
    _CRASH_LOG = str(path)


# ── Training ──

def run_one(model_cfg, train_cfg, manager, val_items, val_every=5):
    """Returns (epoch_losses, epoch_nrmses) for a training run.

    Adaptive validation frequency: every 5 epochs (first 50), then every 20.
    Wrapped in try/except to log crashes.
    """
    try:
        return _run_one_impl(model_cfg, train_cfg, manager, val_items, val_every)
    except Exception:
        tb = traceback.format_exc()
        sys.stderr.write(f"\n[!!] CRASH in run_one:\n{tb}\n")
        sys.stderr.flush()
        try:
            print(f"\n[!!] CRASH in run_one:\n{tb}\n", flush=True)
        except Exception:
            pass
        raise


def _run_one_impl(model_cfg, train_cfg, manager, val_items, val_every):
    model = NLCSNN(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    vram_est = estimate_vram_gb(train_cfg["batch_size"], train_cfg["seq_len"],
                                model_cfg.get("hidden", 128), n_params)
    vram_used = check_vram()
    print(f"[*] params={n_params / 1e3:.0f}k | VRAM used={vram_used:.2f} GB | est={vram_est:.2f} GB")

    if vram_used > 7.0:
        raise RuntimeError(f"VRAM {vram_used:.2f} GB exceeds 7 GB limit. Reduce params.")

    epochs = train_cfg["epochs"]
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=1e-4)
    sched_tmax = max(epochs, 40)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sched_tmax, eta_min=2e-5)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.MSELoss(reduction="none")
    nfl_l1_w = train_cfg.get("nfl_l1_weight", 0)

    epoch_losses = []
    epoch_nrmses = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for _ in range(train_cfg["steps_per_epoch"]):
            u_seq, f_seq, p2p_sq = manager.sample_rws_batch(
                train_cfg["batch_size"], train_cfg["seq_len"])
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                force_loss, l_latent = rws_loss(model, criterion, u_seq, f_seq, p2p_sq)
                l_latent_w = train_cfg.get("latent_weight", 0)
                if nfl_l1_w > 0:
                    loss = force_loss + l_latent_w * l_latent + nfl_l1_w * _trainable_module(model).nfl_l1_loss()
                else:
                    loss = force_loss + l_latent_w * l_latent
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append((float(force_loss.detach().cpu()), float(l_latent.detach().cpu())))
        scheduler.step()
        epoch_losses.append(float(np.mean([x[0] for x in batch_losses])))

        # Adaptive validation: every 5 (<=50) or every 20 (>50)
        val_interval = 5 if epoch <= 50 else 20
        do_val = (epoch % val_interval == 0) or (epoch == epochs)
        if do_val:
            model.eval()
            overall, breakdown = validate(model, manager, val_items, max_samples=1500)
        else:
            overall, breakdown = float("nan"), {}
        epoch_nrmses.append((overall, breakdown))
        if do_val:
            cat_str = " | ".join(f"{c.split('_')[0]}:{v:.1f}" for c, v in sorted(breakdown.items()))
            fl = float(np.mean([x[0] for x in batch_losses]))
            ll = float(np.mean([x[1] for x in batch_losses]))
            print(f"  Epoch {epoch:2d}/{epochs} | loss={fl:.5f} (L_lat={ll:.5f}) | nrmse={overall:.1f}% | {cat_str}", flush=True)
        else:
            fl = float(np.mean([x[0] for x in batch_losses]))
            print(f"  Epoch {epoch:2d}/{epochs} | loss={fl:.5f}", flush=True)

        # Clean CUDA cache periodically to prevent memory fragmentation
        if epoch % 5 == 0:
            torch.cuda.empty_cache()

    return epoch_losses, epoch_nrmses


# ── Main ──

SWEEP_CONFIGS = {
    "h_dim":       {"values": [16, 12, 8],         "param": "h_dim"},
    "hidden":      {"values": [256, 128, 64],       "param": "hidden"},
    "seq_len":     {"values": [1024, 512, 256],     "param": "seq_len"},
    "lr":          {"values": [4e-3, 2e-3, 8e-4, 4e-4, 1e-4], "param": "lr"},
    "nfl_l1":      {"values": [1e-3, 1e-4, 1e-5, 0], "param": "nfl_l1_weight"},
    "batch_size":  {"values": [64, 48, 32],         "param": "batch_size"},
}


def launch_parallel_process(sweep_name, value, epochs, steps, batch, data_root, output_dir):
    """Launch a single sweep value as an independent subprocess."""
    cmd = (
        f'"{sys.executable}" "{__file__}" '
        f'--sweep {sweep_name} --values {value} '
        f'--epochs {epochs} --steps {steps} --batch {batch} '
        f'--data_root "{data_root}" --output_dir "{output_dir}"'
    )
    print(f"[*] Launching: {cmd}")
    return subprocess.Popen(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, required=True, choices=list(SWEEP_CONFIGS))
    parser.add_argument("--values", type=str, default=None, help="Comma-separated override values")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--val_every", type=int, default=5, help="Base validation interval")
    parser.add_argument("--parallel", action="store_true", help="Split values across 2 processes")
    parser.add_argument("--data_root", type=str, default=None, help="Override data root")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    sweep_def = SWEEP_CONFIGS[args.sweep]
    values = [float(x) if "." in x or "e" in x.lower() else int(x)
              for x in (args.values or "").split(",")] if args.values else sweep_def["values"]
    # Descending order (Plan.md §I.2)
    values = sorted(values, reverse=True)
    param_name = sweep_def["param"]

    # Base config
    base_model = {"h_dim": 8, "hidden": 128}
    base_train = {
        "epochs": args.epochs, "batch_size": args.batch, "seq_len": 512,
        "lr": 8e-4, "steps_per_epoch": args.steps, "nfl_l1_weight": 0, "latent_weight": 1e-4,
    }

    if param_name in base_model:
        del base_model[param_name]
    if param_name in base_train:
        del base_train[param_name]

    print(f"{'='*80}")
    print(f"Sweep: {args.sweep} ({param_name}) | values={values} (descending)")
    print(f"Base model: {base_model} | Base train: {base_train}")
    print(f"Data: all train cats | {args.epochs} epochs x {args.steps} steps | parallel={args.parallel}")
    print(f"{'='*80}")

    # Set up crash log for diagnostics
    ts = time.strftime("%Y%m%d_%H%M%S")
    crash_dir = REPO_ROOT / "training_outputs" / "sweeps" / "crash_logs"
    crash_dir.mkdir(parents=True, exist_ok=True)
    _set_crash_log(crash_dir / f"crash_{ts}.log")

    # Resolve data root
    data_root = Path(args.data_root) if args.data_root else Path(PROCESSED_GROUPS)
    fallback = Path(r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\TrainData")
    if not data_root.exists() and fallback.exists():
        data_root = fallback

    # Parallel mode: split values and launch subprocesses
    if args.parallel and len(values) >= 2:
        mid = len(values) // 2
        vals_a = values[:mid]
        vals_b = values[mid:]
        print(f"[*] Parallel: GPU0 → {vals_a} | GPU1 → {vals_b}")
        out_dir = REPO_ROOT / "training_outputs" / "sweeps"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        p1 = launch_parallel_process(
            args.sweep, ",".join(str(v) for v in vals_a),
            args.epochs, args.steps, args.batch,
            str(data_root), str(out_dir / f"sweep_{args.sweep}_A_{ts}.csv"))
        p2 = launch_parallel_process(
            args.sweep, ",".join(str(v) for v in vals_b),
            args.epochs, args.steps, args.batch,
            str(data_root), str(out_dir / f"sweep_{args.sweep}_B_{ts}.csv"))
        print(f"[*] Waiting for parallel processes (PIDs: {p1.pid}, {p2.pid})...")
        p1.wait()
        p2.wait()
        print(f"[*] Parallel sweep complete. Outputs: *_A_{ts}.csv, *_B_{ts}.csv")
        return

    # Sequential mode
    manager = ParallelDataManager(str(data_root))
    print(f"[*] Data: {len(manager._offsets)} train files | {len(manager._val_offsets)} local val files")

    results = []
    for val in values:
        if param_name in ["h_dim", "hidden"]:
            model_cfg = {**base_model, param_name: int(val)}
            train_cfg = {**base_train}
        elif param_name == "seq_len":
            model_cfg = {**base_model}
            train_cfg = {**base_train, param_name: int(val)}
        else:
            model_cfg = {**base_model}
            train_cfg = {**base_train, param_name: float(val) if param_name in ["lr", "nfl_l1_weight"] else int(val)}

        print(f"\n--- Testing {param_name}={val} ---")
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        t0 = time.perf_counter()
        losses, nrmse_list = run_one(model_cfg, train_cfg, manager,
                                     manager.local_val_tensors + manager.hf_val_tensors,
                                     val_every=args.val_every)
        elapsed = time.perf_counter() - t0

        valid_nrmses = [(i + 1, n[0], n[1]) for i, n in enumerate(nrmse_list) if not np.isnan(n[0])]
        if not valid_nrmses:
            valid_nrmses = [(1, float("inf"), {})]
        final_epoch, final_nrmse, final_breakdown = valid_nrmses[-1]
        best_epoch, best_nrmse, _ = min(valid_nrmses, key=lambda x: x[1])
        first_nrmse = valid_nrmses[0][1]
        trend = "↓" if final_nrmse < first_nrmse else ("↑" if final_nrmse > first_nrmse else "→")

        results.append({
            "param": param_name, "value": val,
            "final_loss": f"{losses[-1]:.5f}",
            "final_nrmse": f"{final_nrmse:.1f}",
            "best_nrmse": f"{best_nrmse:.1f}",
            "best_epoch": best_epoch,
            "trend": trend,
            "breakdown": final_breakdown,
            "elapsed": f"{elapsed:.0f}s",
        })
        cat_str = " | ".join(f"{c.split('_')[0]}:{v:.1f}%" for c, v in final_breakdown.items())
        print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.5f} | NRMSE: {first_nrmse:.1f}% → {final_nrmse:.1f}% ({trend}) best={best_nrmse:.1f}% @epoch{best_epoch}")
        print(f"    By category: {cat_str}")
        print(f"    {elapsed:.0f}s")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {args.sweep} sweep")
    print(f"{'='*80}")
    header = f"{'Value':>10} | {'Final Loss':>11} | {'Best NRMSE':>11} | {'Harmonic':>10} | {'LF Random':>10} | {'HF Random':>10} | Best@ | Trend | Time"
    print(header)
    print("-" * len(header))
    for r in results:
        bd = r.get("breakdown", {})
        h = next((v for k, v in bd.items() if "Harmonic" in k), float("nan"))
        lf = next((v for k, v in bd.items() if "LF" in k), float("nan"))
        hf_val = next((v for k, v in bd.items() if "HF" in k), float("nan"))
        print(f"{r['value']:>10} | {r['final_loss']:>11} | {r['best_nrmse']:>10}% | {h:>9.1f}% | {lf:>9.1f}% | {hf_val:>9.1f}% | {r['best_epoch']:>5} | {r['trend']:>5} | {r['elapsed']:>4}")

    best = min(results, key=lambda r: float(r['best_nrmse']))
    print(f"\nBest: {param_name}={best['value']} → NRMSE={best['best_nrmse']}%")

    # Save
    out_dir = args.output_dir if args.output_dir else str(REPO_ROOT / "training_outputs" / "sweeps")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = Path(out_dir) / f"sweep_{args.sweep}_{ts}.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Saved to: {csv_path}")


if __name__ == "__main__":
    main()
