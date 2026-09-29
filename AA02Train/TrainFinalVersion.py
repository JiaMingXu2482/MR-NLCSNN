import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import random
import copy

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from mr_nlcsnn.model import NLCSNN, rk4_step

try:
    from train_config import PROCESSED_GROUPS
except ImportError:
    PROCESSED_GROUPS = r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Processed_Groups"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check the PyTorch CUDA install before training.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda")


device = get_device()


def _trainable_module(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        mod = _trainable_module(model)
        for name, p in mod.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        mod = _trainable_module(model)
        for name, p in mod.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_shadow(self, model):
        mod = _trainable_module(model)
        self.backup.clear()
        for name, p in mod.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        mod = _trainable_module(model)
        for name, p in mod.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[name])
        self.backup.clear()


class DatasetStatsManager:
    def __init__(self):
        self.cfg = {
            "x_ref": 280.0,
            "x_scale": 60.0,
            "v_scale": 1000.0,
            "a_scale": 50000.0,
            "i_scale": 1.6,
            "di_scale": 1000.0,
            "t_scale": 40.0,
            "f_scale": 8000.0,
        }
        self.case_p2p = {}

    def pre_scan(self, all_files):
        print(f"[*] Pre-scanning {len(all_files)} files for normalization...")
        x_min, x_max, di_abs, f_abs = [], [], [], []
        for p in all_files:
            df = pd.read_csv(p)
            x_min.append(df["rod_length"].min())
            x_max.append(df["rod_length"].max())
            di_abs.append(df["current_dot"].abs().max())
            f_abs.append(df["force"].abs().max())
            self.case_p2p[str(p)] = (df["force"].max() - df["force"].min()) + 1e-6

        self.cfg.update({
            "x_ref": (max(x_max) + min(x_min)) / 2.0,
            "x_scale": (max(x_max) - min(x_min)) / 2.0 + 1e-6,
            "di_scale": max(di_abs) + 1e-6,
            "f_scale": max(f_abs) + 1e-6,
        })
        print(
            "[OK] Scalers fixed | "
            f"x_ref={self.cfg['x_ref']:.3f}, x_scale={self.cfg['x_scale']:.3f}, "
            f"F_Max={self.cfg['f_scale']:.1f} N, dI_Max={self.cfg['di_scale']:.1f} A/s"
        )


class ParallelDataManager:
    """Flattened VRAM preloading with offset index for RWS (Randomized Window Sampling).

    Stores all training sequences in a single flattened GPU tensor with a secondary
    offset-index array, enabling O(1) random-window sampling across the entire dataset
    without per-file Python slicing overhead. (Plan.md §IV)
    """
    # TASK 2: Train on Harmonic, Step, Steady Random. Validate on LF Random + HF Random.
    train_cats = os.environ.get("CDC_TRAIN_CATS", "1_Steady_Harmonic,2_Steady_Random,5_Step_Current_Triangle").split(",")
    val_cats = ["4_Random_LF_Current", "3_Random_HF_Current"]

    def __init__(self, root_dir, seed=20260511):
        self.root = Path(root_dir)
        self.stats = DatasetStatsManager()
        self.rng = random.Random(seed)

        # ── File discovery: training ──
        train_files = []
        for cat in self.train_cats:
            train_files.extend(sorted((self.root / cat).glob("*.csv")))
        if not train_files:
            raise RuntimeError(f"No training CSV files found under: {self.root}")

        # ── File discovery: validation (cross-family, NO stratified sampling from training) ──
        val_files = []
        for cat in self.val_cats:
            cat_files = sorted((self.root / cat).glob("*.csv"))
            val_files.extend(cat_files)

        print(
            f"[*] TASK 2 data split | train={len(train_files)} (Harmonic+Random+Step) | "
            f"val={len(val_files)} (LF+HF Random, cross-family only)",
            flush=True,
        )

        # ── Normalization stats from ALL files ──
        all_files = train_files + val_files
        self.stats.pre_scan(all_files)

        # ── Flattened train tensors ──
        self._u_all, self._f_all, self._offsets = self._flatten_to_gpu(train_files)

        # ── Validation as individual tensors (full-sequence eval) ──
        self.val_tensors = self._preload_list(val_files)

        # ── Category sampling weights ──
        self._cat_weights = self._build_category_weights(train_files)

        # ── Backward compat attributes ──
        self.train_files = train_files
        self.train_tensors = self._build_compat_list(train_files, self._offsets)
        self.local_val_tensors = []  # No stratified val — cross-family only
        self.hf_val_tensors = self.val_tensors  # All validation is cross-family

    def _stratified_split(self, files, ratio):
        by_parent = {}
        for p in files:
            by_parent.setdefault(p.parent.name, []).append(p)
        train, val = [], []
        for _, group in sorted(by_parent.items()):
            group = list(group)
            self.rng.shuffle(group)
            n = len(group)
            n_val = int(round(n * ratio))
            n_val = min(max(n_val, 0), max(0, n - 1))
            if n_val == 0 and n >= 3:
                n_val = 1
            val.extend(group[:n_val])
            train.extend(group[n_val:])
        self.rng.shuffle(train)
        self.rng.shuffle(val)
        return train, val

    def _load_csv(self, p):
        df = pd.read_csv(p)
        cfg = self.stats.cfg
        u = np.stack([
            (df["rod_length"].values - cfg["x_ref"]) / cfg["x_scale"],
            df["velocity"].values / cfg["v_scale"],
            df["accel"].values / cfg["a_scale"],
            df["current"].values / cfg["i_scale"],
            df["current_dot"].values / cfg["di_scale"],
            df["temp"].values / cfg["t_scale"],
        ], axis=1).astype(np.float32)
        f = (df["force"].values / cfg["f_scale"]).astype(np.float32)
        return u, f

    def _flatten_to_gpu(self, files):
        """Concatenate all files into single flattened GPU tensors with offset index."""
        u_chunks, f_chunks, offsets = [], [], []
        cursor = 0
        for p in files:
            u, f = self._load_csv(p)
            n = len(f)
            u_chunks.append(torch.from_numpy(u))
            f_chunks.append(torch.from_numpy(f))
            p2p_norm = self.stats.case_p2p[str(p)] / self.stats.cfg["f_scale"]
            offsets.append((cursor, n, float(p2p_norm ** 2), p.parent.name))
            cursor += n
        u_all = torch.cat(u_chunks, dim=0).to(device, non_blocking=True)
        f_all = torch.cat(f_chunks, dim=0).to(device, non_blocking=True)
        total_ts = u_all.shape[0]
        print(f"[*] Flattened VRAM: {total_ts / 1e6:.2f}M timesteps | {u_all.element_size() * u_all.numel() / 1e9:.2f} GB u | {len(offsets)} files")
        return u_all, f_all, offsets

    def _preload_list(self, files):
        data_list = []
        for p in files:
            u, f = self._load_csv(p)
            data_list.append({
                "u": torch.from_numpy(u).to(device, non_blocking=True),
                "f": torch.from_numpy(f).to(device, non_blocking=True),
                "p2p_sq": float((self.stats.case_p2p[str(p)] / self.stats.cfg["f_scale"]) ** 2),
                "name": p.stem,
                "path": p,
                "category": p.parent.name,
            })
        return data_list

    def _build_compat_list(self, files, offsets):
        """Build list-of-dicts for validation code that expects the old format."""
        out = []
        for p, (start, length, p2p_sq, cat) in zip(files, offsets):
            out.append({
                "u": self._u_all[start:start + length],
                "f": self._f_all[start:start + length],
                "p2p_sq": p2p_sq,
                "name": p.stem,
                "path": p,
                "category": cat,
            })
        return out

    def _build_category_weights(self, train_files=None):
        mult = {
            "1_Steady_Harmonic": float(os.environ.get("CDC_W_MULT_HARMONIC", "1.0")),
            "2_Steady_Random": float(os.environ.get("CDC_W_MULT_RANDOM", "1.35")),
            "5_Step_Current_Triangle": float(os.environ.get("CDC_W_MULT_STEP", "1.25")),
        }
        default = float(os.environ.get("CDC_W_MULT_DEFAULT", "1.0"))
        w = []
        for _, _, _, cat in self._offsets:
            w.append(max(mult.get(cat, default), 1e-6))
        if any(abs(x - w[0]) > 1e-9 for x in w):
            print(
                "[*] Category-weighted training sampling | "
                f"harmonic={mult['1_Steady_Harmonic']}, random={mult['2_Steady_Random']}, "
                f"step={mult['5_Step_Current_Triangle']}, default={default}",
                flush=True,
            )
        return w

    def sample_prefix_batch(self, batch_size, seq_len):
        """TASK 5: Sample from t=0 (prefix) — always starts from physical initial state.

        Each batch draws random files, takes first `seq_len` samples.
        Short files are padded, long files are truncated.
        Returns (u_seq, f_seq, p2p_sq).
        """
        idxs = random.choices(range(len(self._offsets)), weights=self._cat_weights, k=batch_size)
        u_batch, f_batch, p2p_list = [], [], []
        for i in idxs:
            _start, length, p2p_sq, _cat = self._offsets[i]
            take = min(length, seq_len)
            u = self._u_all[_start:_start + take]
            f = self._f_all[_start:_start + take]
            if take < seq_len:
                pad = seq_len - take
                u = torch.cat([u, u[-1:].expand(pad, -1)], dim=0)
                f = torch.cat([f, f[-1:].expand(pad)], dim=0)
            u_batch.append(u)
            f_batch.append(f)
            p2p_list.append(p2p_sq)
        return (torch.stack(u_batch, dim=1),
                torch.stack(f_batch, dim=1),
                torch.tensor(p2p_list, device=device))

    def sample_rws_batch(self, batch_size, seq_len):
        """Randomized Window Sampling — pick random windows from random files.

        Each batch draws independent random (file, start_index) pairs.
        Hidden state always starts at zero (no TBPTT across batches).
        Returns (u_seq, f_seq, p2p_sq_tensor).
        """
        idxs = random.choices(range(len(self._offsets)), weights=self._cat_weights, k=batch_size)
        u_batch, f_batch, p2p_list = [], [], []
        for i in idxs:
            start, length, p2p_sq, _cat = self._offsets[i]
            if length > seq_len:
                t0 = random.randint(0, length - seq_len)
            else:
                t0 = 0
            slc = slice(start + t0, start + t0 + seq_len)
            u = self._u_all[slc]
            f = self._f_all[slc]
            if u.shape[0] < seq_len:
                pad = seq_len - u.shape[0]
                u = torch.cat([u, u[-1:].expand(pad, -1)], dim=0)
                f = torch.cat([f, f[-1:].expand(pad)], dim=0)
            u_batch.append(u)
            f_batch.append(f)
            p2p_list.append(p2p_sq)
        u_seq = torch.stack(u_batch, dim=1)  # (seq_len, batch, 6)
        f_seq = torch.stack(f_batch, dim=1)  # (seq_len, batch)
        p2p_sq = torch.tensor(p2p_list, device=device)
        return u_seq, f_seq, p2p_sq


def get_curriculum_rollout_steps(epoch):
    """TASK 5: Return (rollout_steps, stage_label) based on epoch number.

    Stage A (0-10): rollout=1 — 100% teacher forcing
    Stage B (11-20): rollout=2
    Stage C (21-35): rollout=4→8
    Stage D (36+): rollout=16→32→64→full
    """
    if epoch <= 10:
        return 1, "A"
    elif epoch <= 20:
        return 2, "B"
    elif epoch <= 28:
        return 4, "C"
    elif epoch <= 35:
        return 8, "C"
    elif epoch <= 50:
        return 16, "D"
    elif epoch <= 70:
        return 32, "D"
    elif epoch <= 100:
        return 64, "D"
    else:
        return -1, "D-full"  # -1 = full rollout (no truncation)


def curriculum_loss(model, criterion, u_seq, f_seq, p2p_sq, rollout_steps, latent_weight=1e-4):
    """TASK 5: Curriculum rollout loss with periodic hidden-state detaching.

    After every `rollout_steps` timesteps, h is detached so gradients don't flow
    beyond the allowed curriculum horizon. h=0 at t=0 always.
    """
    batch_size = u_seq.shape[1]
    h_dim = model._orig_mod.h_dim if hasattr(model, "_orig_mod") else model.h_dim
    h = torch.zeros(batch_size, h_dim, device=device)
    force_loss = 0.0
    l_latent = 0.0
    n_steps = 0

    for t in range(u_seq.shape[0]):
        u_t = u_seq[t]
        pred = model.predict_force(u_t, h).squeeze(-1)
        h = rk4_step(model, u_t, h)
        target = f_seq[t]
        sign_w = torch.where(target < 0, 1.5, 1.0)
        step_mse = criterion(pred, target)
        force_loss = force_loss + (step_mse * sign_w / p2p_sq).mean()
        l_latent = l_latent + torch.mean(torch.norm(h, dim=-1) ** 2)
        n_steps += 1

        # Detach at curriculum boundary
        if rollout_steps > 0 and n_steps >= rollout_steps:
            h = h.detach()
            n_steps = 0

    seq_len = u_seq.shape[0]
    return force_loss / seq_len, l_latent / seq_len


def rws_loss(model, criterion, u_seq, f_seq, p2p_sq):
    """RWS loss with gain normalization, directional bias, and latent constraint (TASK 4).

    Loss = mean(MSE(pred, target) * sign_w / P2P²) + latent_weight * L_latent
    L_latent = mean(||h||²) over rollout — prevents hidden state explosion.
    """
    batch_size = u_seq.shape[1]
    h_dim = model._orig_mod.h_dim if hasattr(model, "_orig_mod") else model.h_dim
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


def evaluate_case(model, item, exclude=50, max_samples=None):
    h_dim = model._orig_mod.h_dim if hasattr(model, "_orig_mod") else model.h_dim
    h = torch.zeros(1, h_dim, device=device)
    preds = []
    n_steps = len(item["u"]) if max_samples is None else min(len(item["u"]), max_samples)
    with torch.no_grad():
        for t in range(n_steps):
            u_t = item["u"][t:t + 1]
            preds.append(model.predict_force(u_t, h).item())
            h = rk4_step(model, u_t, h)
    p_n = np.asarray(preds, dtype=np.float32)
    y_n = item["f"][:n_steps].detach().cpu().numpy()
    start = min(exclude, max(0, len(y_n) // 4))
    rmse = np.sqrt(np.mean((p_n[start:] - y_n[start:]) ** 2))
    return (rmse / (y_n[start:].max() - y_n[start:].min() + 1e-6)) * 100.0


def select_representative_cases(manager):
    """TASK 2: Validation cases from cross-family only (LF Random + HF Random)."""
    pool = manager.val_tensors  # Cat 4 (LF) + Cat 3 (HF) only
    selected = []
    for parent in ["4_Random_LF_Current", "3_Random_HF_Current"]:
        match = next((x for x in pool if x["path"].parent.name == parent), None)
        if match is not None:
            selected.append(match)
    return selected


def predict_representative_cases(model, cases, max_samples=None):
    """TASK 2: Full-sequence rollout with latent diagnostics.

    Returns list of dicts with keys:
    nrmse, pred_norm, target_norm, latent_norm_mean, latent_norm_max, drift_pct
    """
    if not cases:
        return []

    h_dim = model._orig_mod.h_dim if hasattr(model, "_orig_mod") else model.h_dim
    n_steps = max(len(item["u"]) for item in cases)
    if max_samples is not None:
        n_steps = min(n_steps, max_samples)

    h = torch.zeros(len(cases), h_dim, device=device)
    u_rows, f_rows, lengths = [], [], []
    for item in cases:
        n = min(len(item["u"]), n_steps)
        lengths.append(n)
        u = item["u"][:n]
        f = item["f"][:n]
        if n < n_steps:
            pad = n_steps - n
            u = torch.cat([u, u[-1:].expand(pad, -1)], dim=0)
            f = torch.cat([f, f[-1:].expand(pad)], dim=0)
        u_rows.append(u)
        f_rows.append(f)

    u_seq = torch.stack(u_rows, dim=1)
    f_seq = torch.stack(f_rows, dim=1)

    preds = []
    h_norms = []  # Track ||h|| over time
    with torch.no_grad():
        for t in range(n_steps):
            u_t = u_seq[t]
            preds.append(model.predict_force(u_t, h).squeeze(-1))
            h = rk4_step(model, u_t, h)
            h_norms.append(torch.norm(h, dim=-1).mean().item())
    pred_seq = torch.stack(preds, dim=0)

    rows = []
    for idx, item in enumerate(cases):
        n = lengths[idx]
        p_n = pred_seq[:n, idx].detach().cpu().numpy()
        y_n = f_seq[:n, idx].detach().cpu().numpy()
        start = min(50, max(0, len(y_n) // 4))

        # Full NRMSE
        rmse_full = np.sqrt(np.mean((p_n[start:] - y_n[start:]) ** 2))
        nrmse = (rmse_full / (y_n[start:].max() - y_n[start:].min() + 1e-6)) * 100.0

        # Drift: NRMSE in first 1s (0-1000ms) vs last 1s
        split_1s = min(1000, n - 1)
        if split_1s > start:
            err_first = np.mean((p_n[start:split_1s] - y_n[start:split_1s]) ** 2)
            if n - split_1s > start:
                last_start = max(n - 1000, split_1s)
                err_last = np.mean((p_n[last_start:] - y_n[last_start:]) ** 2)
                drift = (np.sqrt(err_last) - np.sqrt(err_first)) / (np.sqrt(err_first) + 1e-6) * 100.0
            else:
                drift = 0.0
        else:
            drift = 0.0

        rows.append({
            "name": item["name"],
            "category": item["path"].parent.name,
            "path": item["path"],
            "nrmse": nrmse,
            "pred_norm": p_n,
            "target_norm": y_n,
            "latent_norm_mean": float(np.mean(h_norms)),
            "latent_norm_max": float(np.max(h_norms)),
            "drift_pct": drift,
        })
    return rows


def evaluate_representative_cases(model, cases, max_samples=5000):
    rows = []
    for row in predict_representative_cases(model, cases, max_samples=max_samples):
        rows.append({k: row[k] for k in ("name", "category", "nrmse")})
    return rows


def evaluate_validation_cases(model, cases, max_samples=2500):
    rows = []
    for item in cases:
        rows.append({
            "name": item["name"],
            "category": item["path"].parent.name,
            "nrmse": evaluate_case(model, item, exclude=50, max_samples=max_samples),
        })
    return rows


def save_representative_val_plot(rep_rows, cfg, out_png, epoch, max_samples=None):
    if not rep_rows:
        return

    fig, axes = plt.subplots(len(rep_rows), 1, figsize=(13, 4 * len(rep_rows)), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, row in zip(axes, rep_rows):
        df = pd.read_csv(row["path"])
        if max_samples is not None:
            df = df.iloc[:max_samples].copy()

        start = min(50, max(0, len(df) // 4))
        f_pred = np.asarray(row["pred_norm"]) * cfg["f_scale"]
        y_m = df["force"].values[start:]
        y_p = f_pred[start:]
        t_axis = df["time"].values[start:]
        rmse = np.sqrt(np.mean((y_m - y_p) ** 2))
        nrmse = (rmse / (y_m.max() - y_m.min() + 1e-6)) * 100

        ax.plot(t_axis, y_m, "k-", alpha=0.5, label="Measured")
        ax.plot(t_axis, y_p, "r--", linewidth=1.1, label="Predicted")
        ax.set_ylabel("Force (N)")
        ax.set_title(f"{row['category']} / {row['name']} | NRMSE={nrmse:.2f}%")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Representative validation cases | Epoch {epoch:04d}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def should_save_monitor_plot(epoch, rep_mean, best_score, last_plot_epoch):
    if epoch == 1:
        return True
    if epoch <= 20 and epoch % 5 == 0:
        return True
    if epoch - last_plot_epoch >= 25:
        return True
    if np.isfinite(rep_mean) and rep_mean + 0.25 < best_score:
        return True
    return False


def should_run_validation(epoch, last_eval_epoch, eval_interval):
    """Evaluation frequency. Uses CDC_EVAL_INTERVAL env var if set, otherwise adaptive (5 then 20)."""
    if eval_interval is not None and eval_interval > 0:
        return epoch - last_eval_epoch >= eval_interval
    interval = 5 if epoch <= 50 else 20
    return epoch - last_eval_epoch >= interval


def append_metrics(path, row):
    candidates = [path, path.with_name(f"{path.stem}_fallback.csv")]
    last_exc = None
    for candidate in candidates:
        try:
            _append_metrics_unlocked(candidate, row)
            return candidate
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.2)
    print(f"[!] Metrics write skipped: {last_exc}", flush=True)
    return None


def _append_metrics_unlocked(path, row):
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new_file:
            f.write(
                "epoch,train_loss,rep_mean_nrmse_pct,rep1_nrmse_pct,rep2_nrmse_pct,rep3_nrmse_pct,rep4_nrmse_pct,rep5_nrmse_pct,"
                "epoch_seconds,torch_peak_gb,nvidia_used_gb,gpu_util_pct,lr\n"
            )
        f.write(
            f"{row['epoch']},{row['train_loss']:.8f},{row['rep_mean']:.6f},"
            f"{row['rep1']:.6f},{row['rep2']:.6f},{row['rep3']:.6f},{row['rep4']:.6f},{row['rep5']:.6f},"
            f"{row['seconds']:.3f},{row['torch_peak']:.3f},{row['nvidia_used']:.3f},"
            f"{row['gpu_util']:.3f},{row['lr']:.8g}\n"
        )


def check_vram_redline(limit_gb=7.0):
    """VRAM guard — warn at limit, raise if exceeded (Plan.md §II.1)."""
    used = torch.cuda.memory_reserved() / (1024 ** 3)
    if used > limit_gb:
        raise RuntimeError(
            f"VRAM usage {used:.2f} GB exceeds red line {limit_gb} GB. "
            f"Reduce batch_size, seq_len, or hidden."
        )
    if used > limit_gb * 0.9:
        print(f"[!] VRAM warning: {used:.2f} GB used, limit={limit_gb} GB")


def query_nvidia_memory_gb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
        )
        return float(out.strip().splitlines()[0]) / 1024.0
    except Exception:
        return float("nan")


def query_nvidia_gpu_util_pct():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return float("nan")


def main():
    seed = int(os.environ.get("CDC_SEED", "20260511"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    epochs = int(os.environ.get("CDC_EPOCHS", "400"))
    batch_size = int(os.environ.get("CDC_BATCH_SIZE", "64"))
    seq_len = int(os.environ.get("CDC_SEQ_LEN", "512"))
    steps_per_epoch = int(os.environ.get("CDC_STEPS_PER_EPOCH", "48"))
    hidden = int(os.environ.get("CDC_HIDDEN", "128"))
    state_dim = int(os.environ.get("CDC_STATE_DIM", "8"))
    monitor_samples = int(os.environ.get("CDC_MONITOR_SAMPLES", "5000"))
    target_nrmse = float(os.environ.get("CDC_TARGET_NRMSE", "3.0"))
    eval_interval_override = os.environ.get("CDC_EVAL_INTERVAL", "")
    eval_interval = int(eval_interval_override) if eval_interval_override else None
    scheduler_tmax = int(os.environ.get("CDC_SCHEDULER_TMAX", str(max(epochs, 400))))
    learning_rate = float(os.environ.get("CDC_LR", "8e-4"))
    nfl_l1_weight = float(os.environ.get("CDC_NFL_L1_WEIGHT", "1e-4"))
    latent_weight = float(os.environ.get("CDC_LATENT_WEIGHT", "1e-3"))
    vram_limit_gb = float(os.environ.get("CDC_VRAM_LIMIT_GB", "7.0"))
    use_ema = os.environ.get("CDC_USE_EMA", "1") == "1"
    ema_decay = float(os.environ.get("CDC_EMA_DECAY", "0.999"))
    save_best_as_ema = os.environ.get("CDC_SAVE_BEST_AS_EMA", "1") == "1"
    loss_type = os.environ.get("CDC_LOSS", "mse")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = REPO_ROOT / "training_outputs"
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"session_adaptive_{epochs}_epochs_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    val_plots_dir = out_dir / "val_plots"
    metrics_csv = out_dir / "val_metrics.csv"

    data_root = Path(PROCESSED_GROUPS)
    fallback_root = Path(r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\TrainData")
    if not data_root.exists() and fallback_root.exists():
        print(f"[!] Configured data root missing: {data_root}")
        print(f"[*] Fallback to: {fallback_root}")
        data_root = fallback_root

    print(f"[*] Data root: {data_root}")
    manager = ParallelDataManager(str(data_root), seed=seed)
    rep_cases = select_representative_cases(manager)
    model = NLCSNN(hidden=hidden, h_dim=state_dim).to(device)

    resume_checkpoint = os.environ.get("CDC_RESUME_CHECKPOINT", "").strip()
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["state_dict"].items()
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[*] Resumed model weights from: {resume_checkpoint}", flush=True)
        if missing or unexpected:
            print(f"[!] Non-strict resume | missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    if os.environ.get("CDC_COMPILE", "1") == "1":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[*] torch.compile activated.")
        except Exception as exc:
            print(f"[!] torch.compile skipped: {exc}")

    ema = ModelEMA(model, decay=ema_decay) if use_ema else None
    if use_ema:
        print(f"[*] EMA enabled | decay={ema_decay} | save_best_as_ema={save_best_as_ema}", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_tmax, eta_min=2e-5)
    scaler = torch.amp.GradScaler("cuda")
    if loss_type == "smooth_l1":
        criterion = nn.SmoothL1Loss(beta=0.06, reduction="none")
    else:
        criterion = nn.MSELoss(reduction="none")

    print(
        f"\n[*] Split: Train={len(manager.train_tensors)}, "
        f"Val={len(manager.val_tensors)} (LF+HF Random, cross-family)"
    )
    print(
        f"[*] Training on {torch.cuda.get_device_name(0)} | hidden={hidden}, state={state_dim}, "
        f"batch={batch_size}, seq={seq_len}, steps/epoch={steps_per_epoch}, epochs={epochs}"
    )
    print(
        f"[*] Loss={loss_type} | gain_norm=MSE/P2P^2 | dir_bias=1.5x compression | "
        f"nfl_l1={nfl_l1_weight} | l_latent={latent_weight} | RWS sampling | vram_guard={vram_limit_gb}GB"
    )
    print(f"[*] Representative validation cases ({monitor_samples / 1000:.1f} s each):")
    for item in rep_cases:
        print(f"    - {item['path'].parent.name}/{item['name']}.csv")
    print(f"[*] Session: {out_dir}\n")

    check_vram_redline(vram_limit_gb)

    best_score = float("inf")
    best_path = out_dir / "best_model_3pct.pth"
    target_path = out_dir / "best_model_3pct_target.pth"
    last_path = out_dir / "nlcsnn_last.pth"
    train_start = time.perf_counter()
    last_plot_epoch = 0
    last_eval_epoch = 0
    rep_rows = []
    rep_vals = [np.nan, np.nan, np.nan, np.nan, np.nan]
    rep_mean = np.nan

    show_tqdm = os.environ.get("CDC_TQDM", "0") == "1"
    epoch_iter = tqdm(range(1, epochs + 1), desc="TOTAL PROGRESS", ascii=True) if show_tqdm else range(1, epochs + 1)

    for epoch in epoch_iter:
        model.train()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        losses = []

        # TASK 5: curriculum rollout schedule
        rollout_steps, stage = get_curriculum_rollout_steps(epoch)
        if epoch == 1 or (rollout_steps > 0 and epoch - 1 > 0 and get_curriculum_rollout_steps(epoch - 1)[0] != rollout_steps):
            label = f"D({rollout_steps})" if rollout_steps > 0 else "D(full)"
            print(f"\n[*] TASK 5 Stage {stage}: rollout_steps={label}", flush=True)

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch:03d}", ascii=True, leave=False, disable=not show_tqdm)
        for _ in pbar:
            u_seq, f_seq, p2p_sq = manager.sample_prefix_batch(batch_size, seq_len)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                force_loss, l_latent = curriculum_loss(
                    model, criterion, u_seq, f_seq, p2p_sq,
                    rollout_steps=rollout_steps, latent_weight=latent_weight)
                nfl_l1 = _trainable_module(model).nfl_l1_loss()
                loss = force_loss + latent_weight * l_latent + nfl_l1_weight * nfl_l1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            losses.append((float(force_loss.detach().cpu()), float(l_latent.detach().cpu())))
            pbar.set_postfix(loss=f"{losses[-1][0]:.5f}")

        scheduler.step()
        epoch_seconds = time.perf_counter() - t0
        torch_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        nvidia_used = query_nvidia_memory_gb()
        gpu_util = query_nvidia_gpu_util_pct()
        force_loss_mean = float(np.mean([x[0] for x in losses]))
        latent_loss_mean = float(np.mean([x[1] for x in losses]))
        train_loss = force_loss_mean  # for backward compat in metrics

        run_validation = should_run_validation(epoch, last_eval_epoch, eval_interval)
        if run_validation:
            model.eval()
            if ema is not None:
                ema.apply_shadow(model)
            try:
                rep_rows = predict_representative_cases(model, rep_cases, max_samples=monitor_samples)
            finally:
                if ema is not None:
                    ema.restore(model)
            rep_vals = [r["nrmse"] for r in rep_rows] + [np.nan] * (5 - len(rep_rows))
            rep_mean = float(np.nanmean(rep_vals))
            last_eval_epoch = epoch
        score = rep_mean
        elapsed = time.perf_counter() - train_start
        avg_epoch = elapsed / epoch
        eta_seconds = avg_epoch * (epochs - epoch)
        rep_text = " | ".join(
            f"{r['category'].split('_')[0]}:{r['nrmse']:.2f}%" for r in rep_rows
        )
        # TASK 2: latent diagnostics
        if rep_rows:
            h_mean = rep_rows[0].get("latent_norm_mean", float("nan"))
            h_max = rep_rows[0].get("latent_norm_max", float("nan"))
            drift_vals = [r.get("drift_pct", 0) for r in rep_rows]
            drift_text = " | ".join(f"{r['category'].split('_')[0]}:{r.get('drift_pct',0):+.0f}%" for r in rep_rows)
        else:
            h_mean = h_max = float("nan")
            drift_text = ""
        print(
            f" >> Epoch {epoch:03d}/{epochs} ({epoch / epochs * 100:.1f}%) | "
            f"Loss={force_loss_mean:.5f} (L_lat={latent_loss_mean:.5f}) | Stage={stage} roll={rollout_steps} | RepMean={rep_mean:.2f}% | {rep_text} | "
            f"||h||_mean={h_mean:.3f} max={h_max:.3f} | drift=[{drift_text}] | "
            f"val={'yes' if run_validation else 'no'} | "
            f"epoch={epoch_seconds:.1f}s | ETA={eta_seconds / 60:.1f}min | "
            f"torch_peak={torch_peak:.2f}GB | nvidia={nvidia_used:.2f}GB | gpu={gpu_util:.0f}%",
            flush=True,
        )
        save_monitor = run_validation and should_save_monitor_plot(epoch, rep_mean, best_score, last_plot_epoch)
        if run_validation and score < best_score:
            best_score = score
            if ema is not None and save_best_as_ema:
                ema.apply_shadow(model)
                try:
                    sd = copy.deepcopy(_trainable_module(model).state_dict())
                finally:
                    ema.restore(model)
                torch.save({
                    "state_dict": sd,
                    "norm_cfg": manager.stats.cfg,
                    "model_cfg": {"hidden": hidden, "h_dim": state_dim},
                    "ema_saved": True,
                }, best_path)
            else:
                torch.save({
                    "state_dict": model.state_dict(),
                    "norm_cfg": manager.stats.cfg,
                    "model_cfg": {"hidden": hidden, "h_dim": state_dim},
                    "ema_saved": False,
                }, best_path)

        append_metrics(metrics_csv, {
            "epoch": epoch,
            "train_loss": train_loss,
            "rep_mean": rep_mean,
            "rep1": rep_vals[0],
            "rep2": rep_vals[1],
            "rep3": rep_vals[2],
            "rep4": rep_vals[3],
            "rep5": rep_vals[4],
            "seconds": epoch_seconds,
            "torch_peak": torch_peak,
            "nvidia_used": nvidia_used,
            "gpu_util": gpu_util,
            "lr": scheduler.get_last_lr()[0],
        })

        if save_monitor:
            model.eval()
            if ema is not None:
                ema.apply_shadow(model)
            try:
                save_representative_val_plot(
                    predict_representative_cases(model, rep_cases, max_samples=monitor_samples),
                    manager.stats.cfg,
                    val_plots_dir / f"epoch_{epoch:04d}_monitor_cases.png",
                    epoch,
                    max_samples=monitor_samples,
                )
            finally:
                if ema is not None:
                    ema.restore(model)
            last_plot_epoch = epoch

        if run_validation and rep_rows and all(r["nrmse"] < target_nrmse for r in rep_rows):
            all_val_cases = manager.val_tensors
            model.eval()
            if ema is not None:
                ema.apply_shadow(model)
            try:
                all_rows = evaluate_validation_cases(model, all_val_cases, max_samples=monitor_samples)
            finally:
                if ema is not None:
                    ema.restore(model)
            all_good = all(row["nrmse"] < target_nrmse for row in all_rows)
            all_csv = out_dir / f"epoch_{epoch:04d}_all_validation_metrics.csv"
            pd.DataFrame(all_rows).to_csv(all_csv, index=False)
            print(
                f"[*] Target check at epoch {epoch}: representative cases are below {target_nrmse:.2f}%. "
                f"All validation max NRMSE={max(row['nrmse'] for row in all_rows):.2f}% | saved {all_csv}",
                flush=True,
            )
            if all_good:
                if ema is not None and save_best_as_ema:
                    ema.apply_shadow(model)
                    try:
                        sd = copy.deepcopy(_trainable_module(model).state_dict())
                    finally:
                        ema.restore(model)
                    torch.save({
                        "state_dict": sd,
                        "norm_cfg": manager.stats.cfg,
                        "model_cfg": {"hidden": hidden, "h_dim": state_dim},
                        "target_nrmse": target_nrmse,
                        "epoch": epoch,
                        "ema_saved": True,
                    }, target_path)
                else:
                    torch.save({
                        "state_dict": model.state_dict(),
                        "norm_cfg": manager.stats.cfg,
                        "model_cfg": {"hidden": hidden, "h_dim": state_dim},
                        "target_nrmse": target_nrmse,
                        "epoch": epoch,
                        "ema_saved": False,
                    }, target_path)
                print(f"[*] Target reached on all validation cases. Saved: {target_path}", flush=True)
                break

    torch.save({
        "state_dict": model.state_dict(),
        "norm_cfg": manager.stats.cfg,
        "model_cfg": {"hidden": hidden, "h_dim": state_dim},
    }, last_path)
    print("\n[*] Training complete.")
    print(f"[*] Last checkpoint: {last_path}")
    print(f"[*] Best checkpoint: {best_path} | best validation score={best_score:.2f}%")


if __name__ == "__main__":
    main()
