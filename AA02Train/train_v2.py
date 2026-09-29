"""NLCSNN v2 training — clean rewrite.

Replaces the env-var jungle of train_legacy_repro.py with a self-contained
script implementing the approved v2 plan:

  - stratified LF holdout (most LF in training, ~8 files held out incl. the
    1 Hz and 3 Hz-slow extreme corners) — explicit file lists, no env modes;
  - per-file TBPTT (each file h=0, carried across chunks, detached at chunk
    boundaries) — the proven legacy training semantics;
  - p2p^2-normalised MSE + compression direction bias + L1 NFL sparsity;
  - CosineAnnealingLR with linear warmup, stepped every epoch;
  - RMS-normalised NRMSE + MAE(N) evaluation on the holdout, full-length.

Run:
  & C:\\Users\\user\\.conda\\envs\\myenvPINN_test\\python.exe \
      "d:/OneDrive - mail.scut.edu.cn/Python/MR-NLCSNN/AA02Train/train_v2.py"
"""
import copy
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from mr_nlcsnn.model_v2 import NLCSNN_v2, rk4_step

# --------------------------------------------------------------------- config
DEFAULT_DATA_ROOT = r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Parsed_Card_force_wide"
DATA_ROOT = os.environ.get("CDC_PROCESSED_GROUPS", DEFAULT_DATA_ROOT).strip()

TRAIN_CATS = ["1_Steady_Harmonic", "2_Steady_Random", "5_Step_Current_Triangle"]
LF_CAT = "4_Random_LF_Current"

# Stratified holdout: covers 2/5/10/15/18 mm x 1-7 Hz, incl. the two worst
# historical extreme corners (RMS10mm_1Hz, RMS18mm_3Hzslow). Rest of LF -> train.
# Corrupt files excluded from all use. RLF_RMS10mm_2Hz_r2 has a rod_length spike
# to ~1272 mm (vs physical ~196-280); it is a corrupt duplicate of the clean
# RLF_RMS10mm_2Hz_r1, so dropping it loses no information.
EXCLUDE_STEMS = ["RLF_RMS10mm_2Hz_r2"]

HOLDOUT_STEMS = [
    "RLF_RMS2mm_7Hz",
    "RLF_RMS5mm_2Hz",
    "RLF_RMS5mm_6Hzslow",
    "RLF_RMS10mm_1Hz",
    "RLF_RMS10mm_4Hz",
    "RLF_RMS15mm_1Hz",
    "RLF_RMS15mm_3Hz",
    "RLF_RMS18mm_3Hzslow",
]

EPOCHS = int(os.environ.get("CDC_EPOCHS", "200"))
LR = float(os.environ.get("CDC_LR", "8e-4"))
WARMUP_EPOCHS = int(os.environ.get("CDC_WARMUP_EPOCHS", "5"))
ETA_MIN_RATIO = float(os.environ.get("CDC_ETA_MIN_RATIO", "0.01"))
HIDDEN = int(os.environ.get("CDC_HIDDEN", "192"))
H_DIM = int(os.environ.get("CDC_STATE_DIM", "8"))
FILE_BATCH = int(os.environ.get("CDC_FILE_BATCH_SIZE", "32"))
CHUNK_LEN = int(os.environ.get("CDC_CHUNK_LEN", "1000"))
WARMUP_SAMPLES = int(os.environ.get("CDC_WARMUP", "50"))      # loss/eval skip-in
NEG_FORCE_WEIGHT = float(os.environ.get("CDC_NEG_FORCE_WEIGHT", "1.5"))
L1_WEIGHT = float(os.environ.get("CDC_NFL_L1_WEIGHT", "1e-5"))
EVAL_EVERY = int(os.environ.get("CDC_EVAL_EVERY", "10"))
EVAL_MAX_SAMPLES = int(os.environ.get("CDC_EVAL_MAX_SAMPLES", "5000"))  # 5 s window (target def); 0=full
PERIODIC_EVERY = int(os.environ.get("CDC_PERIODIC_EVERY", "50"))
MAX_BALANCED_LAMBDA = float(os.environ.get("CDC_MAX_LAMBDA", "0.3"))  # score=mean+lambda*max
SEED = int(os.environ.get("CDC_SEED", "20260613"))

# Debug/smoke knobs: 0 = use all files. Set small to verify the pipeline fast.
MAX_TRAIN_FILES = int(os.environ.get("CDC_MAX_TRAIN_FILES", "0"))
MAX_HOLDOUT_FILES = int(os.environ.get("CDC_MAX_HOLDOUT_FILES", "0"))


def get_device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------- data layer
class DataManagerV2:
    """Explicit-split data manager: prescan train files for norm, preload to GPU."""

    FIXED = {"v_scale": 1000.0, "a_scale": 50000.0, "i_scale": 1.6,
             "t_ref": 40.0, "t_scale": 40.0}

    def __init__(self, root, device):
        self.root = Path(root)
        self.device = device

        exclude_set = set(EXCLUDE_STEMS)
        train_files, holdout_files = [], []
        for cat in TRAIN_CATS:
            train_files += [p for p in sorted((self.root / cat).glob("*.csv"))
                            if p.stem not in exclude_set]
        lf_all = [p for p in sorted((self.root / LF_CAT).glob("*.csv"))
                  if p.stem not in exclude_set]
        holdout_set = set(HOLDOUT_STEMS)
        seen_holdout = set()
        for p in lf_all:
            if p.stem in holdout_set:
                holdout_files.append(p)
                seen_holdout.add(p.stem)
            else:
                train_files.append(p)
        missing = holdout_set - seen_holdout
        if missing:
            raise RuntimeError(f"Holdout files not found under {self.root/LF_CAT}: {sorted(missing)}")
        if not train_files:
            raise RuntimeError(f"No training files under {self.root}")

        if MAX_TRAIN_FILES > 0:
            train_files = train_files[:MAX_TRAIN_FILES]
        if MAX_HOLDOUT_FILES > 0:
            holdout_files = holdout_files[:MAX_HOLDOUT_FILES]

        self.train_files = train_files
        self.holdout_files = holdout_files
        self.cfg = self._prescan(train_files)   # norm from TRAIN only (no leakage)
        self.train_items = self._preload(train_files)
        self.holdout_items = self._preload(holdout_files)

        print(f"[*] Data: train={len(train_files)} files "
              f"({len(lf_all)-len(holdout_files)} LF in train) | holdout={len(holdout_files)} LF")
        print(f"[*] Norm (train-only): x_ref={self.cfg['x_ref']:.3f} x_scale={self.cfg['x_scale']:.3f} "
              f"di_scale={self.cfg['di_scale']:.3f} f_scale={self.cfg['f_scale']:.3f}")

    def _prescan(self, files):
        cfg = dict(self.FIXED)
        x_min, x_max = [], []
        di_abs, f_abs = 0.0, 0.0
        for p in files:
            df = pd.read_csv(p, usecols=["rod_length", "current_dot", "force"])
            x_min.append(float(df["rod_length"].min()))
            x_max.append(float(df["rod_length"].max()))
            di_abs = max(di_abs, float(df["current_dot"].abs().max()))
            f_abs = max(f_abs, float(df["force"].abs().max()))
        cfg["x_ref"] = (max(x_max) + min(x_min)) / 2.0
        cfg["x_scale"] = (max(x_max) - min(x_min)) / 2.0 + 1e-6
        cfg["di_scale"] = di_abs + 1e-6
        cfg["f_scale"] = f_abs + 1e-6
        # Guard against corrupt rod_length spikes poisoning the scaler. Physical
        # stroke keeps rod_length well under ~400 mm; a larger x_scale means a bad
        # file slipped in. Halt loudly with the culprit instead of training garbage.
        if cfg["x_scale"] > 150.0:
            culprits = [str(p) for p in files
                        if pd.read_csv(p, usecols=["rod_length"])["rod_length"].max() > 400.0]
            raise RuntimeError(
                f"x_scale={cfg['x_scale']:.1f} too large (rod_length corruption). "
                f"Suspect files (rod_length>400mm): {culprits}. "
                f"Add their stems to EXCLUDE_STEMS.")
        return cfg

    def _load_norm(self, path):
        df = pd.read_csv(path)
        c = self.cfg
        u = np.stack([
            (df["rod_length"].values - c["x_ref"]) / c["x_scale"],
            df["velocity"].values / c["v_scale"],
            df["accel"].values / c["a_scale"],
            df["current"].values / c["i_scale"],
            df["current_dot"].values / c["di_scale"],
            (df["temp"].values - c["t_ref"]) / c["t_scale"],
        ], axis=1).astype(np.float32)
        f = (df["force"].values / c["f_scale"]).astype(np.float32)
        return u, f

    def _preload(self, files):
        items = []
        for p in files:
            u, f = self._load_norm(p)
            p2p = float(f.max() - f.min()) + 1e-6
            items.append({
                "u": torch.from_numpy(u).to(self.device),
                "f": torch.from_numpy(f).to(self.device),
                "length": int(len(f)),
                "p2p": p2p,
                "name": p.stem,
                "path": p,
            })
        return items


# ------------------------------------------------------------------- training
def make_file_batches(items, batch_size):
    items = sorted(items, key=lambda it: it["length"])   # length buckets -> less padding
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    for b in batches:
        random.shuffle(b)
    random.shuffle(batches)
    return batches


def train_file_batch(model, optimizer, criterion, batch, device):
    """Per-file TBPTT over one length-bucketed batch. Each file keeps its own h."""
    bs = len(batch)
    max_len = max(it["length"] for it in batch)
    p2p = torch.tensor([it["p2p"] for it in batch], device=device)

    u_all = torch.zeros(max_len, bs, 6, device=device)
    f_all = torch.zeros(max_len, bs, device=device)
    valid_all = torch.zeros(max_len, bs, dtype=torch.bool, device=device)
    for j, it in enumerate(batch):
        n = it["length"]
        u_all[:n, j] = it["u"]
        f_all[:n, j] = it["f"]
        valid_all[:n, j] = True

    h = torch.zeros(bs, model.h_dim, device=device)
    loss_sum, n_chunks = 0.0, 0
    for start in range(0, max_len, CHUNK_LEN):
        this_len = min(CHUNK_LEN, max_len - start)
        u_c = u_all[start:start + this_len]
        f_c = f_all[start:start + this_len]
        valid = valid_all[start:start + this_len]
        if not valid.any():
            continue

        optimizer.zero_grad(set_to_none=True)
        h = h.detach()
        preds = []
        for t in range(this_len):
            u_t = u_c[t]
            preds.append(model.predict_force(u_t, h).squeeze(-1))
            h_next = rk4_step(model, u_t, h)
            h = torch.where(valid[t].unsqueeze(-1), h_next, h)
        p_out = torch.stack(preds, dim=0)               # (this_len, bs)

        # warmup-once: skip first WARMUP_SAMPLES of each file (only the chunk
        # containing the file start is affected)
        warm = torch.ones_like(f_c)
        warm_rows = max(0, min(this_len, WARMUP_SAMPLES - start))
        if warm_rows > 0:
            warm[:warm_rows] = 0.0
        point_w = torch.where(f_c < 0, NEG_FORCE_WEIGHT, 1.0)
        loss_mask = valid.float() * warm
        point_loss = criterion(p_out, f_c) * point_w * loss_mask
        denom = valid.float().sum(dim=0).clamp_min(1.0)
        per_file = point_loss.sum(dim=0) / denom        # (bs,)
        active = valid.any(dim=0)
        chunk_loss = (per_file / (p2p ** 2))[active].mean()
        chunk_loss = chunk_loss + L1_WEIGHT * model.nfl_l1_loss()

        chunk_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += float(chunk_loss.detach().cpu())
        n_chunks += 1

    return loss_sum / max(n_chunks, 1)


# ----------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, items, cfg, device):
    """Full-length open-loop rollout from h=0. RMS-normalised NRMSE + MAE(N)."""
    model.eval()
    rows = []
    fscale = cfg["f_scale"]
    for it in items:
        n = it["length"] if EVAL_MAX_SAMPLES <= 0 else min(it["length"], EVAL_MAX_SAMPLES)
        u = it["u"]
        h = torch.zeros(1, model.h_dim, device=device)
        preds = []
        for t in range(n):
            preds.append(model.predict_force(u[t:t + 1], h).squeeze())
            h = rk4_step(model, u[t:t + 1], h)
        p = torch.stack(preds).cpu().numpy().astype(np.float32) * fscale
        y = it["f"][:n].cpu().numpy().astype(np.float32) * fscale
        s = min(WARMUP_SAMPLES, max(0, len(y) // 4))
        err = p[s:] - y[s:]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        rms_true = float(np.sqrt(np.mean(y[s:] ** 2))) + 1e-6
        rows.append({
            "name": it["name"],
            "nrmse_pct": rmse / rms_true * 100.0,     # RMS-normalised
            "mae_N": float(np.mean(np.abs(err))),
            "rmse_N": rmse,
        })
    model.train()
    return rows


def make_lr_lambda(epochs, warmup_epochs, eta_min_ratio):
    def fn(epoch):  # 0-indexed
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        prog = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return fn


def save_checkpoint(path, model, cfg, score, mean_nrmse, max_nrmse, epoch):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": copy.deepcopy(model.state_dict()),
        "norm_cfg": cfg,
        "model_cfg": {
            "architecture": "nlcsnn_v2",
            "hidden": HIDDEN,
            "h_dim": H_DIM,
            "slow_gain": bool(model.slow_gain),
            "extra_dynamic": bool(model.extra_dynamic),
            "prune_h_coupling": bool(model.prune_h_coupling),
            "nfl_dim": int(model.nfl_dim),
        },
        "holdout_stems": HOLDOUT_STEMS,
        "epoch": epoch,
        "score": score,
        "mean_nrmse_pct": mean_nrmse,
        "max_nrmse_pct": max_nrmse,
    }, path)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = get_device()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "training_outputs" / f"v2_{EPOCHS}epochs_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"

    print(f"[*] Device={device} | data_root={DATA_ROOT}")
    mgr = DataManagerV2(DATA_ROOT, device)
    model = NLCSNN_v2(h_dim=H_DIM, hidden=HIDDEN).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(EPOCHS, WARMUP_EPOCHS, ETA_MIN_RATIO))
    criterion = nn.MSELoss(reduction="none")

    print(f"[*] Session: {out_dir}")
    print(f"[*] epochs={EPOCHS} lr={LR} (cosine, warmup={WARMUP_EPOCHS}) hidden={HIDDEN} "
          f"h_dim={H_DIM} file_batch={FILE_BATCH} chunk={CHUNK_LEN} l1={L1_WEIGHT}")
    print(f"[*] Holdout ({len(HOLDOUT_STEMS)}): {', '.join(HOLDOUT_STEMS)}")

    best_score = float("inf")
    best_mean = float("inf")
    best_path = out_dir / "nlcsnn_v2_best.pth"
    best_mean_path = out_dir / "nlcsnn_v2_best_mean.pth"
    last_path = out_dir / "nlcsnn_v2_last.pth"

    with open(metrics_path, "w", encoding="utf-8") as fh:
        fh.write("epoch,train_loss,mean_nrmse_pct,max_nrmse_pct,score,lr,seconds\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.perf_counter()
        batches = make_file_batches(mgr.train_items, FILE_BATCH)
        losses = [train_file_batch(model, optimizer, criterion, b, device) for b in batches]
        scheduler.step()
        train_loss = float(np.mean(losses))

        do_eval = (epoch % EVAL_EVERY == 0) or (epoch == EPOCHS) or (epoch == 1)
        mean_nrmse = max_nrmse = score = float("nan")
        if do_eval:
            rows = evaluate(model, mgr.holdout_items, mgr.cfg, device)
            vals = np.array([r["nrmse_pct"] for r in rows])
            mean_nrmse, max_nrmse = float(vals.mean()), float(vals.max())
            score = mean_nrmse + MAX_BALANCED_LAMBDA * max_nrmse
            pd.DataFrame(rows).to_csv(out_dir / f"epoch_{epoch:04d}_holdout.csv", index=False)
            if score < best_score:
                best_score = score
                save_checkpoint(best_path, model, mgr.cfg, score, mean_nrmse, max_nrmse, epoch)
            if mean_nrmse < best_mean:
                best_mean = mean_nrmse
                save_checkpoint(best_mean_path, model, mgr.cfg, score, mean_nrmse, max_nrmse, epoch)

        if PERIODIC_EVERY > 0 and epoch % PERIODIC_EVERY == 0:
            save_checkpoint(out_dir / "periodic" / f"epoch_{epoch:04d}.pth",
                            model, mgr.cfg, score, mean_nrmse, max_nrmse, epoch)

        secs = time.perf_counter() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(f"{epoch},{train_loss:.8f},{mean_nrmse:.6f},{max_nrmse:.6f},"
                     f"{score:.6f},{lr_now:.8g},{secs:.2f}\n")
        worst = ""
        if do_eval:
            worst = " | ".join(f"{r['name'].replace('RLF_RMS','')}:{r['nrmse_pct']:.2f}%"
                               for r in sorted(rows, key=lambda r: -r["nrmse_pct"])[:3])
        print(f"E[{epoch:03d}/{EPOCHS}] loss={train_loss:.5f} | "
              f"mean={mean_nrmse:.2f}% max={max_nrmse:.2f}% | worst3: {worst} | "
              f"lr={lr_now:.2g} | {secs:.0f}s", flush=True)

    save_checkpoint(last_path, model, mgr.cfg, score, mean_nrmse, max_nrmse, EPOCHS)
    print(f"\n[*] Done. best(score)={best_score:.3f} -> {best_path.name} | "
          f"best_mean={best_mean:.2f}% -> {best_mean_path.name}")


if __name__ == "__main__":
    main()
