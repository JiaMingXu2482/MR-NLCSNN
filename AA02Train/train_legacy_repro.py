import copy
import os
import random
import re
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
from tqdm import tqdm

from mr_nlcsnn.legacy_model import LegacyNLCSNN, rk4_step

try:
    from train_config import PROCESSED_GROUPS
except ImportError:
    PROCESSED_GROUPS = r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Processed_Groups"


def get_device():
    if os.environ.get("CDC_LEGACY_DEVICE", "").strip():
        return torch.device(os.environ["CDC_LEGACY_DEVICE"].strip())
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")


class DatasetStatsManager:
    def __init__(self):
        self.cfg = {
            "x_ref": 250.0,
            "x_scale": 50.0,
            "v_scale": 1000.0,
            "a_scale": 50000.0,
            "i_scale": 1.6,
            "di_scale": 1000.0,
            "t_ref": 40.0,
            "t_scale": 40.0,
            "f_scale": 8000.0,
        }
        self.case_p2p = {}

    def pre_scan(self, files):
        print(f"[*] Legacy pre-scan on {len(files)} files")
        x_min, x_max = [], []
        di_abs_max = 0.0
        f_abs_max = 0.0

        for p in files:
            df = pd.read_csv(p)
            x_min.append(df["rod_length"].min())
            x_max.append(df["rod_length"].max())
            di_abs_max = max(di_abs_max, float(df["current_dot"].abs().max()))
            f_abs_max = max(f_abs_max, float(df["force"].abs().max()))
            self.case_p2p[str(p)] = float(df["force"].max() - df["force"].min()) + 1e-6

        self.cfg["x_ref"] = (max(x_max) + min(x_min)) / 2.0
        self.cfg["x_scale"] = (max(x_max) - min(x_min)) / 2.0 + 1e-6
        self.cfg["di_scale"] = di_abs_max + 1e-6
        self.cfg["f_scale"] = f_abs_max + 1e-6

        print(
            "[*] Legacy scalers | "
            f"x_ref={self.cfg['x_ref']:.3f}, x_scale={self.cfg['x_scale']:.3f}, "
            f"di_scale={self.cfg['di_scale']:.3f}, f_scale={self.cfg['f_scale']:.3f}"
        )


class LegacyCDCDataManager:
    def __init__(self, root_dir, norm_cfg_override=None):
        self.root = Path(root_dir)
        self.stats = DatasetStatsManager()
        self.train_files = []
        self.val_files = []
        self.split_label = ""
        self._categorize()
        self._apply_debug_limits()
        if norm_cfg_override is not None:
            self.stats.cfg.update(norm_cfg_override)
            self._scan_case_p2p(self.train_files)
            print("[*] Normalization cfg restored from resume checkpoint; case P2P scanned on train files")
            print(
                "[*] Legacy scalers | "
                f"x_ref={self.stats.cfg['x_ref']:.3f}, x_scale={self.stats.cfg['x_scale']:.3f}, "
                f"di_scale={self.stats.cfg['di_scale']:.3f}, f_scale={self.stats.cfg['f_scale']:.3f}"
            )
        else:
            use_val_for_norm = os.environ.get("CDC_NORM_USES_VAL", "0").strip() == "1"
            scan_files = self.train_files + self.val_files if use_val_for_norm else self.train_files
            print(f"[*] Normalization pre-scan uses {'train+val' if use_val_for_norm else 'train only'} files")
            self.stats.pre_scan(scan_files)

    def _categorize(self):
        train_cats_env = os.environ.get("CDC_TRAIN_CATS", "").strip()
        val_cats_env = os.environ.get("CDC_VAL_CATS", "").strip()

        if train_cats_env and val_cats_env:
            # ── explicit category split (bypasses val_mode / CDC_INCLUDE_* logic) ──
            train_folders = [c.strip() for c in train_cats_env.split(",") if c.strip()]
            val_folders = [c.strip() for c in val_cats_env.split(",") if c.strip()]
            for cat in train_folders:
                cat_path = self.root / cat
                if cat_path.is_dir():
                    self.train_files.extend(sorted(cat_path.glob("*.csv")))
            for cat in val_folders:
                cat_path = self.root / cat
                if cat_path.is_dir():
                    self.val_files.extend(sorted(cat_path.glob("*.csv")))
            self.split_label = f"train=[{','.join(train_folders)}] val=[{','.join(val_folders)}]"
        else:
            # ── original val_mode-based logic ──
            val_mode = os.environ.get("CDC_VAL_MODE", "hf").strip().lower()
            train_folders = [
                "1_Steady_Harmonic",
                "2_Steady_Random",
                "4_Random_LF_Current",
            ]
            if os.environ.get("CDC_INCLUDE_HF", "0").strip() == "1":
                train_folders.append("3_Random_HF_Current")
            if os.environ.get("CDC_INCLUDE_STEP", "0").strip() == "1":
                train_folders.append("5_Step_Current_Triangle")

            if val_mode == "lf_holdout":
                lf_folder = self.root / "4_Random_LF_Current"
                lf_files = sorted(lf_folder.glob("*.csv"))
                self.val_files = select_lf_holdout_files(lf_files)
                val_set = {str(p) for p in self.val_files}
                for cat in train_folders:
                    cat_files = sorted((self.root / cat).glob("*.csv"))
                    if cat == "4_Random_LF_Current":
                        cat_files = [p for p in cat_files if str(p) not in val_set]
                    self.train_files.extend(cat_files)
                self.split_label = "LF held-out random current"
            else:
                val_folder = os.environ.get("CDC_VAL_CATEGORY", "3_Random_HF_Current").strip()
                for cat in train_folders:
                    self.train_files.extend(sorted((self.root / cat).glob("*.csv")))
                self.val_files = sorted((self.root / val_folder).glob("*.csv"))
                self.split_label = val_folder

        # ── Exclude corrupt files (rod_length spike poisons normalization) ──
        # RLF_RMS10mm_2Hz_r2 has a ~1272 mm rod_length spike; it is a corrupt
        # duplicate of the clean RLF_RMS10mm_2Hz_r1, so dropping it loses nothing.
        _EXCLUDE_STEMS = {"RLF_RMS10mm_2Hz_r2"}
        self.train_files = [p for p in self.train_files if p.stem not in _EXCLUDE_STEMS]
        self.val_files = [p for p in self.val_files if p.stem not in _EXCLUDE_STEMS]

        # ── Exclude quasi-static files (v_peak < CDC_MIN_VPEAK_MS) from train AND val ──
        # Sub-0.05 m/s is below the realistic damper working range (operating-envelope spec).
        min_vpeak = float(os.environ.get("CDC_MIN_VPEAK_MS", "0"))
        if min_vpeak > 0:
            def _vpeak_ok(p):
                try:
                    vv = pd.read_csv(p, usecols=["velocity"])["velocity"].values
                    return (np.percentile(np.abs(vv), 99) / 1000.0) >= min_vpeak
                except Exception:
                    return True
            nt0, nv0 = len(self.train_files), len(self.val_files)
            self.train_files = [p for p in self.train_files if _vpeak_ok(p)]
            self.val_files = [p for p in self.val_files if _vpeak_ok(p)]
            print(f"[*] v_peak>={min_vpeak} filter: train {nt0}->{len(self.train_files)}, val {nv0}->{len(self.val_files)}")

        # ── CDC_VAL_FILES filter (applies in both modes) ──
        val_files_env = os.environ.get("CDC_VAL_FILES", "").strip()
        if val_files_env:
            wanted = {name.strip().removesuffix(".csv") for name in val_files_env.split(",") if name.strip()}
            filtered = [p for p in self.val_files if p.stem in wanted]
            missing = wanted - {p.stem for p in filtered}
            if missing:
                raise RuntimeError(
                    f"CDC_VAL_FILES not found in val categories: {sorted(missing)}"
                )
            self.val_files = filtered

        if not self.train_files:
            raise RuntimeError(f"No training files found under {self.root}")
        if not self.val_files:
            raise RuntimeError(f"No validation files found under {self.root}")

    def _scan_case_p2p(self, files):
        self.stats.case_p2p = {}
        for p in files:
            df = pd.read_csv(p, usecols=["force"])
            self.stats.case_p2p[str(p)] = float(df["force"].max() - df["force"].min()) + 1e-6

    def _apply_debug_limits(self):
        max_train = int(os.environ.get("CDC_LEGACY_MAX_TRAIN_FILES", "0"))
        max_val = int(os.environ.get("CDC_LEGACY_MAX_VAL_FILES", "0"))
        if max_train > 0:
            self.train_files = self.train_files[:max_train]
        if max_val > 0:
            self.val_files = self.val_files[:max_val]
        print(
            f"[*] Legacy split | train={len(self.train_files)}, "
            f"val={len(self.val_files)} ({self.split_label})"
        )

    def load_and_norm(self, file_path, device):
        df = pd.read_csv(file_path)
        cfg = self.stats.cfg
        u = np.stack(
            [
                (df["rod_length"].values - cfg["x_ref"]) / cfg["x_scale"],
                df["velocity"].values / cfg["v_scale"],
                df["accel"].values / cfg["a_scale"],
                df["current"].values / cfg["i_scale"],
                df["current_dot"].values / cfg["di_scale"],
                (df["temp"].values - cfg["t_ref"]) / cfg["t_scale"],
            ],
            axis=1,
        ).astype(np.float32)
        f = (df["force"].values / cfg["f_scale"]).astype(np.float32)
        imp_weight = float(df["importance_weight"].iloc[0]) if "importance_weight" in df.columns else 1.0
        p2p_raw = self.stats.case_p2p.get(str(file_path))
        if p2p_raw is None:
            p2p_raw = float(df["force"].max() - df["force"].min()) + 1e-6
        cat_name = Path(file_path).parent.name
        meta = {
            "p2p_norm": p2p_raw / cfg["f_scale"],
            "imp_weight": imp_weight,
            "case_weight": compute_case_weight(file_path.stem, category=cat_name),
            "name": file_path.stem,
            "path": file_path,
        }
        return torch.from_numpy(u).to(device), torch.from_numpy(f).to(device), meta

    def preload_files(self, files, device, label):
        items = []
        total_samples = 0
        for p in files:
            u, f, meta = self.load_and_norm(p, device)
            item = {
                "u": u,
                "f": f,
                "length": int(len(f)),
                "meta": meta,
                "path": p,
                "name": p.stem,
            }
            items.append(item)
            total_samples += item["length"]
        tensor_gb = total_samples * (6 + 1) * 4 / (1024 ** 3)
        print(
            f"[*] Preloaded {label}: {len(items)} files | "
            f"{total_samples:,} samples | approx {tensor_gb:.3f} GB tensors"
        )
        weights = [item["meta"].get("case_weight", 1.0) for item in items]
        if weights and any(abs(w - 1.0) > 1e-9 for w in weights):
            print(
                f"[*] Case weights active for {label}: "
                f"min={min(weights):.3f}, max={max(weights):.3f}, mean={float(np.mean(weights)):.3f}"
            )
        return items


def parse_case_amp_freq(name):
    m = re.match(r"^(?:AMP|a?RMS)(?P<amp>\d+\.?\d*)(?:mm)?_(?P<freq>\d+\.?\d*)Hz", name)
    if not m:
        return None, None
    return float(m.group("amp")), float(m.group("freq"))


def select_lf_holdout_files(lf_files):
    explicit = os.environ.get("CDC_LF_VAL_FILES", "").strip()
    if explicit:
        wanted = {name.strip().removesuffix(".csv") for name in explicit.split(",") if name.strip()}
        selected = [p for p in lf_files if p.stem in wanted]
        missing = sorted(wanted - {p.stem for p in selected})
        if missing:
            raise RuntimeError(f"CDC_LF_VAL_FILES not found in LF folder: {missing}")
        return selected

    ratio = float(os.environ.get("CDC_LF_VAL_RATIO", "0.25"))
    seed = int(os.environ.get("CDC_LF_VAL_SEED", "20260516"))
    rng = random.Random(seed)
    groups = {}
    for p in lf_files:
        amp, _ = parse_case_amp_freq(p.stem)
        groups.setdefault(amp if amp is not None else -1.0, []).append(p)

    selected = []
    for _, files in sorted(groups.items(), key=lambda kv: kv[0]):
        files = sorted(files)
        n_val = max(1, int(round(len(files) * ratio)))
        n_val = min(n_val, max(1, len(files) - 1))
        selected.extend(sorted(rng.sample(files, n_val)))
    return sorted(selected)


def compute_case_weight(name, category=None):
    amp, freq = parse_case_amp_freq(name)
    weight = 1.0
    # ── per-category weight override: CDC_CAT_WEIGHT_OVERRIDE=2_Steady_Random:2.5,… ──
    cat_override = os.environ.get("CDC_CAT_WEIGHT_OVERRIDE", "").strip()
    if cat_override and category:
        for token in cat_override.split(","):
            token = token.strip()
            if ":" not in token:
                continue
            cat_key, _, val_str = token.partition(":")
            if cat_key.strip() == category:
                weight *= float(val_str.strip())
    regex = os.environ.get("CDC_CASE_WEIGHT_REGEX", "").strip()
    regex_weight = float(os.environ.get("CDC_CASE_WEIGHT_REGEX_WEIGHT", "1.0"))
    low_amp_threshold = float(os.environ.get("CDC_WEIGHT_LOW_AMP_THRESHOLD", "5"))
    low_amp_weight = float(os.environ.get("CDC_WEIGHT_LOW_AMP", "1.0"))
    high_freq_threshold = float(os.environ.get("CDC_WEIGHT_HIGH_FREQ_THRESHOLD", "4"))
    high_freq_weight = float(os.environ.get("CDC_WEIGHT_HIGH_FREQ", "1.0"))
    target_10mm_weight = float(os.environ.get("CDC_WEIGHT_10MM", "1.0"))
    high_amp_threshold = float(os.environ.get("CDC_WEIGHT_HIGH_AMP_THRESHOLD", "15"))
    high_amp_weight = float(os.environ.get("CDC_WEIGHT_HIGH_AMP", "1.0"))
    high_amp_low_freq_threshold = float(os.environ.get("CDC_WEIGHT_HIGH_AMP_LOW_FREQ_THRESHOLD", "3"))
    high_amp_low_freq_weight = float(os.environ.get("CDC_WEIGHT_HIGH_AMP_LOW_FREQ", "1.0"))
    cap = float(os.environ.get("CDC_CASE_WEIGHT_CAP", "2.0"))

    if amp is not None and amp <= low_amp_threshold:
        weight *= low_amp_weight
    if amp is not None and abs(amp - 10.0) < 1e-6:
        weight *= target_10mm_weight
    if amp is not None and amp >= high_amp_threshold:
        weight *= high_amp_weight
        if freq is not None and freq <= high_amp_low_freq_threshold:
            weight *= high_amp_low_freq_weight
    if freq is not None and freq >= high_freq_threshold:
        weight *= high_freq_weight
    if regex:
        try:
            if re.search(regex, name):
                weight *= regex_weight
        except re.error as exc:
            raise RuntimeError(f"Invalid CDC_CASE_WEIGHT_REGEX: {regex}") from exc
    return min(weight, cap)


def select_eval_files(files, max_files=None):
    files = list(files)
    if max_files is None or max_files <= 0 or max_files >= len(files):
        return files
    if max_files == 1:
        return files[:1]
    idx = np.linspace(0, len(files) - 1, num=max_files, dtype=int)
    return [files[int(i)] for i in idx]


@torch.no_grad()
def evaluate_files(model, manager, files, device, warmup_ms=50, max_files=None, max_samples=None, start_sample=0):
    """Roll out validation files from h=0 and compute force metrics.

    The implementation intentionally keeps the same legacy semantics: each physical
    validation file starts from a zero latent state and then rolls forward
    recursively. Predictions and hidden-state norms stay on GPU during the loop and
    are copied back to CPU once per file, avoiding a CPU/GPU synchronization on
    every timestep.
    """
    rows = []
    selected = select_eval_files(files, max_files=max_files)
    cfg = manager.stats.cfg
    model.eval()

    for p in selected:
        u, f_norm, _ = manager.load_and_norm(p, device)
        if start_sample > 0:
            if start_sample >= len(u) - 1:
                continue
            u = u[start_sample:]
            f_norm = f_norm[start_sample:]
        if max_samples is not None and max_samples > 0:
            eval_len = min(int(max_samples), len(u))
            u = u[:eval_len]
            f_norm = f_norm[:eval_len]
        if len(u) == 0:
            continue

        h = torch.zeros(1, model.h_dim, device=device)
        preds = []
        h_norms = []
        for t in range(len(u)):
            u_t = u[t : t + 1]
            preds.append(model.predict_force(u_t, h).detach().squeeze())
            h = rk4_step(model, u_t, h)
            h_norms.append(torch.norm(h, dim=-1).detach().squeeze())

        p_force = torch.stack(preds).detach().cpu().numpy().astype(np.float32) * cfg["f_scale"]
        t_force = f_norm.detach().cpu().numpy().astype(np.float32) * cfg["f_scale"]
        h_norm_arr = torch.stack(h_norms).detach().cpu().numpy().astype(np.float32)

        start = min(int(warmup_ms), max(0, len(t_force) // 4))
        err = p_force[start:] - t_force[start:]
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        # RMS-normalised NRMSE (paper-consistent): robust to force zero-crossings.
        rms_true = float(np.sqrt(np.mean(t_force[start:] ** 2)))
        nrmse = rmse / (rms_true + 1e-6) * 100.0
        rows.append(
            {
                "case": p.stem,
                "path": str(p),
                "start_sample": int(start_sample),
                "eval_samples": int(len(u)),
                "nrmse_pct": nrmse,
                "mae_N": mae,
                "rmse_N": rmse,
                "h_norm_mean": float(np.mean(h_norm_arr)),
                "h_norm_max": float(np.max(h_norm_arr)),
                "pred_force": p_force,
                "target_force": t_force,
            }
        )
    return rows

def iter_file_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def make_file_batches(items, batch_size, length_buckets=True):
    items = list(items)
    if length_buckets:
        items.sort(key=lambda item: item["length"])
        batches = list(iter_file_batches(items, batch_size))
        for batch in batches:
            random.shuffle(batch)
        random.shuffle(batches)
        return batches
    random.shuffle(items)
    return list(iter_file_batches(items, batch_size))


def train_file_batch(
    model,
    optimizer,
    criterion,
    batch_items,
    chunk_len,
    warmup,
    strict_mask,
    hidden_norm_weight,
    dh_norm_weight,
    p2p_loss_power,
    neg_force_weight,
    high_force_abs_threshold_norm,
    high_force_weight,
    near_zero_v_threshold_norm,
    near_zero_v_weight,
    high_force_near_zero_weight,
    gain_loss_weight,
    std_gain_loss_weight,
    warmup_once,
    train_new_nfl_columns_only,
    device,
    train_window_len=0,
    random_window=True,
    train_window_start_sample=0,
    slope_weight=0.0,
    curv_weight=0.0,
):
    """Train several physical files in parallel while preserving per-file hidden state.

    Each file starts with its own h=0. Hidden state is carried across chunks for that
    file and detached at chunk boundaries, matching the CPU-era TBPTT semantics.
    """
    if train_window_len and train_window_len > 0:
        windowed_items = []
        for item in batch_items:
            n = item["length"]
            if n <= train_window_len:
                start = 0
                end = n
            else:
                max_start = n - train_window_len
                if random_window:
                    start = random.randint(0, max_start)
                else:
                    start = min(max(0, train_window_start_sample), max_start)
                end = start + train_window_len
            windowed = dict(item)
            windowed["u"] = item["u"][start:end]
            windowed["f"] = item["f"][start:end]
            windowed["length"] = int(end - start)
            windowed_items.append(windowed)
        batch_items = windowed_items

    batch_size = len(batch_items)
    max_len = max(item["length"] for item in batch_items)
    h = torch.zeros(batch_size, model.h_dim, device=device)
    p2p = torch.tensor([item["meta"]["p2p_norm"] for item in batch_items], device=device)
    imp = torch.tensor([item["meta"]["imp_weight"] for item in batch_items], device=device)
    case_w = torch.tensor([item["meta"].get("case_weight", 1.0) for item in batch_items], device=device)

    # Build one padded tensor for this physical-file batch. This preserves full-file
    # TBPTT semantics but avoids repeatedly allocating and filling padded chunk
    # tensors inside every chunk. Chunking below is only for memory/BPTT truncation;
    # it does not reset h.
    u_all = torch.zeros(max_len, batch_size, 6, device=device)
    f_all = torch.zeros(max_len, batch_size, device=device)
    valid_all = torch.zeros(max_len, batch_size, dtype=torch.bool, device=device)
    for j, item in enumerate(batch_items):
        n = item["length"]
        u_all[:n, j] = item["u"]
        f_all[:n, j] = item["f"]
        valid_all[:n, j] = True

    batch_loss_sum = 0.0
    chunk_count = 0

    for start in range(0, max_len, chunk_len):
        this_len = min(chunk_len, max_len - start)
        u_chunk = u_all[start : start + this_len]
        f_chunk = f_all[start : start + this_len]
        valid = valid_all[start : start + this_len]

        if not valid.any():
            continue

        optimizer.zero_grad(set_to_none=True)
        h = h.detach()
        preds = []
        h_norm_loss = 0.0
        dh_norm_loss = 0.0
        reg_count = 0
        for t in range(this_len):
            u_t = u_chunk[t]
            pred_t = model.predict_force(u_t, h).squeeze(-1)
            h_next = rk4_step(model, u_t, h)
            valid_t = valid[t].unsqueeze(-1)
            if hidden_norm_weight > 0:
                h_norm_loss = h_norm_loss + ((h_next ** 2) * valid_t.float()).sum() / valid_t.float().sum().clamp_min(1.0)
            if dh_norm_weight > 0:
                dh = h_next - h
                dh_norm_loss = dh_norm_loss + ((dh ** 2) * valid_t.float()).sum() / valid_t.float().sum().clamp_min(1.0)
            if hidden_norm_weight > 0 or dh_norm_weight > 0:
                reg_count += 1
            h = torch.where(valid[t].unsqueeze(-1), h_next, h)
            preds.append(pred_t)

        p_out = torch.stack(preds, dim=0)
        point_weights = torch.ones_like(f_chunk)
        if neg_force_weight != 1.0:
            point_weights = point_weights * torch.where(f_chunk < 0, neg_force_weight, 1.0)
        high_force_mask = torch.abs(f_chunk) >= high_force_abs_threshold_norm if high_force_weight != 1.0 or high_force_near_zero_weight != 1.0 else None
        near_zero_v_mask = (
            torch.abs(u_chunk[:, :, 1]) <= near_zero_v_threshold_norm
            if near_zero_v_weight != 1.0 or high_force_near_zero_weight != 1.0
            else None
        )
        if high_force_weight != 1.0:
            point_weights = point_weights * torch.where(high_force_mask, high_force_weight, 1.0)
        if near_zero_v_weight != 1.0:
            point_weights = point_weights * torch.where(near_zero_v_mask, near_zero_v_weight, 1.0)
        if high_force_near_zero_weight != 1.0:
            combo_mask = high_force_mask & near_zero_v_mask
            point_weights = point_weights * torch.where(combo_mask, high_force_near_zero_weight, 1.0)
        warm_mask = torch.ones_like(f_chunk)
        if warmup > 0:
            if warmup_once:
                warm_rows = max(0, min(this_len, warmup - start))
                if warm_rows > 0:
                    warm_mask[:warm_rows] = 0.0
            elif this_len > warmup:
                warm_mask[:warmup] = 0.0
        loss_mask = valid.float() * warm_mask
        point_loss = criterion(p_out, f_chunk) * point_weights * loss_mask

        if strict_mask:
            denom = loss_mask.sum(dim=0).clamp_min(1.0)
        else:
            # CPU-era behavior: warmup points are zeroed but still contribute to mean denominator.
            denom = valid.float().sum(dim=0).clamp_min(1.0)
        per_file_loss = point_loss.sum(dim=0) / denom
        active_files = valid.any(dim=0)
        chunk_losses = (per_file_loss / (p2p**p2p_loss_power)) * imp * case_w
        chunk_loss = chunk_losses[active_files].mean()
        active_loss_files = loss_mask.sum(dim=0) > 1
        if std_gain_loss_weight > 0 and active_loss_files.any():
            amp_denom = loss_mask.sum(dim=0).clamp_min(1.0)
            pred_mean = (p_out * loss_mask).sum(dim=0) / amp_denom
            true_mean = (f_chunk * loss_mask).sum(dim=0) / amp_denom
            pred_var = (((p_out - pred_mean.unsqueeze(0)) ** 2) * loss_mask).sum(dim=0) / amp_denom
            true_var = (((f_chunk - true_mean.unsqueeze(0)) ** 2) * loss_mask).sum(dim=0) / amp_denom
            pred_std = torch.sqrt(pred_var.clamp_min(1e-12))
            true_std = torch.sqrt(true_var.clamp_min(1e-12))
            std_loss = ((pred_std - true_std) ** 2) / (true_std.square() + 1e-6)
            std_loss = std_loss * imp * case_w
            chunk_loss = chunk_loss + std_gain_loss_weight * std_loss[active_loss_files].mean()
        if gain_loss_weight > 0 and active_loss_files.any():
            pos_inf = torch.full_like(p_out, float("inf"))
            neg_inf = torch.full_like(p_out, -float("inf"))
            pred_min = torch.where(loss_mask > 0, p_out, pos_inf).min(dim=0).values
            pred_max = torch.where(loss_mask > 0, p_out, neg_inf).max(dim=0).values
            true_min = torch.where(loss_mask > 0, f_chunk, pos_inf).min(dim=0).values
            true_max = torch.where(loss_mask > 0, f_chunk, neg_inf).max(dim=0).values
            pred_amp = pred_max - pred_min
            true_amp = true_max - true_min
            amp_loss = ((pred_amp - true_amp) ** 2) / (true_amp.square() + 1e-6)
            amp_loss = amp_loss * imp * case_w
            chunk_loss = chunk_loss + gain_loss_weight * amp_loss[active_loss_files].mean()
        if reg_count > 0:
            if hidden_norm_weight > 0:
                chunk_loss = chunk_loss + hidden_norm_weight * (h_norm_loss / reg_count)
            if dh_norm_weight > 0:
                chunk_loss = chunk_loss + dh_norm_weight * (dh_norm_loss / reg_count)

        # Output-smoothness regularization: penalize predicted-force time derivatives
        # to suppress high-frequency prediction jitter (esp. quasi-static cases).
        # slope = 1st difference, curvature = 2nd difference, masked to valid points.
        if (slope_weight > 0 or curv_weight > 0) and this_len > 2 and active_files.any():
            if slope_weight > 0:
                d1 = p_out[1:] - p_out[:-1]
                m1 = loss_mask[1:] * loss_mask[:-1]
                sl = (d1 * d1 * m1).sum(dim=0) / m1.sum(dim=0).clamp_min(1.0) / (p2p ** p2p_loss_power)
                chunk_loss = chunk_loss + slope_weight * (sl * imp * case_w)[active_files].mean()
            if curv_weight > 0:
                d2 = p_out[2:] - 2.0 * p_out[1:-1] + p_out[:-2]
                m2 = loss_mask[2:] * loss_mask[1:-1] * loss_mask[:-2]
                cu = (d2 * d2 * m2).sum(dim=0) / m2.sum(dim=0).clamp_min(1.0) / (p2p ** p2p_loss_power)
                chunk_loss = chunk_loss + curv_weight * (cu * imp * case_w)[active_files].mean()

        chunk_loss.backward()
        if train_new_nfl_columns_only:
            apply_new_nfl_column_gradient_mask(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_loss_sum += float(chunk_loss.detach().cpu())
        chunk_count += 1

    return batch_loss_sum / max(chunk_count, 1)


def apply_new_nfl_column_gradient_mask(model):
    start = getattr(model, "_new_nfl_start", None)
    if start is None:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        return
    keep_names = {"nn_x.0.weight", "nn_y.0.weight", "nn_y_direct.weight"}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if name in keep_names and param.grad.ndim == 2 and param.grad.shape[1] > start:
            param.grad[:, :start] = 0
        else:
            param.grad.zero_()


def save_monitor_plot(row, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    target = row["target_force"]
    pred = row["pred_force"]
    t = np.arange(len(target)) * 0.001
    axes[0].plot(t, target, "k-", alpha=0.4, label="Measured")
    axes[0].plot(t, pred, "r--", linewidth=1.1, label="Predicted")
    axes[0].set_ylabel("Force (N)")
    axes[0].set_title(f"{row['case']} | NRMSE={row['nrmse_pct']:.2f}%")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(t, np.abs(target - pred), color="purple", linewidth=1.0)
    axes[1].set_ylabel("Abs error (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def append_metrics(path, row):
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new_file:
            f.write(
                "epoch,train_loss,monitor_nrmse_pct,all_val_mean_nrmse_pct,"
                "all_val_max_nrmse_pct,monitor_h_mean,monitor_h_max,lr,epoch_seconds\n"
            )
        f.write(
            f"{row['epoch']},{row['train_loss']:.8f},{row['monitor_nrmse']:.6f},"
            f"{row['all_mean']:.6f},{row['all_max']:.6f},{row['h_mean']:.6f},"
            f"{row['h_max']:.6f},{row['lr']:.8g},{row['seconds']:.3f}\n"
        )


def save_checkpoint(path, model, manager, hidden, h_dim, epoch, score, optimizer=None, scheduler=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": copy.deepcopy(model.state_dict()),
        "norm_cfg": manager.stats.cfg,
        "model_cfg": {
            "architecture": "legacy_cpu_repro",
            "hidden": hidden,
            "h_dim": h_dim,
            "extra_dynamic_nfl": bool(getattr(model, "extra_dynamic_nfl", False)),
            "direct_nfl_output": bool(getattr(model, "direct_nfl_output", False)),
            "slow_gain_nfl": bool(getattr(model, "slow_gain_nfl", False)),
            "hysteresis_gain_nfl": bool(getattr(model, "hysteresis_gain_nfl", False)),
            "nfl_slim": bool(getattr(model, "nfl_slim", False)),
            "physics_nfl": bool(getattr(model, "physics_nfl", False)),
            "physics_temp": bool(getattr(model, "physics_temp", False)),
            "physics_path": bool(getattr(model, "physics_path", False)),
            "nfl_dim": int(getattr(model, "nfl_dim", 0)),
        },
        "epoch": epoch,
        "score_nrmse_pct": score,
        "split_cfg": {
            "split_label": manager.split_label,
            "train_files": [str(p) for p in manager.train_files],
            "val_files": [str(p) for p in manager.val_files],
            "norm_uses_val": os.environ.get("CDC_NORM_USES_VAL", "0").strip() == "1",
            "norm_from_resume_checkpoint": os.environ.get("CDC_USE_RESUME_NORM", "0").strip() == "1",
            "include_step": os.environ.get("CDC_INCLUDE_STEP", "0").strip() == "1",
            "include_hf": os.environ.get("CDC_INCLUDE_HF", "0").strip() == "1",
            "train_cats": os.environ.get("CDC_TRAIN_CATS", "").strip(),
            "val_cats": os.environ.get("CDC_VAL_CATS", "").strip(),
            "val_files": os.environ.get("CDC_VAL_FILES", "").strip(),
            "train_time_val_max_seconds": float(os.environ.get("CDC_VAL_MAX_SECONDS", "0")),
            "train_time_val_start_seconds": float(os.environ.get("CDC_VAL_START_SECONDS", "0")),
            "train_time_val_every_seconds": float(os.environ.get("CDC_VAL_EVERY_SECONDS", "0")),
            "train_time_val_all_max_files": int(os.environ.get("CDC_VAL_ALL_MAX_FILES", "0")),
            "train_time_monitor_every": int(os.environ.get("CDC_MONITOR_EVERY", "1")),
            "train_time_val_all_every": int(os.environ.get("CDC_VAL_ALL_EVERY", "10")),
            "train_time_periodic_every": int(os.environ.get("CDC_PERIODIC_EVERY", "0")),
            "train_window_seconds": float(os.environ.get("CDC_TRAIN_WINDOW_SECONDS", "0")),
            "train_window_random": os.environ.get("CDC_TRAIN_WINDOW_RANDOM", "1").strip() == "1",
            "train_window_start_seconds": float(os.environ.get("CDC_TRAIN_WINDOW_START_SECONDS", "0")),
            "case_weight_regex": os.environ.get("CDC_CASE_WEIGHT_REGEX", "").strip(),
            "case_weight_regex_weight": float(os.environ.get("CDC_CASE_WEIGHT_REGEX_WEIGHT", "1.0")),
            "p2p_loss_power": float(os.environ.get("CDC_P2P_LOSS_POWER", "2.0")),
            "warmup_once": os.environ.get("CDC_WARMUP_ONCE", "1").strip() == "1",
            "train_new_nfl_columns_only": os.environ.get("CDC_TRAIN_NEW_NFL_COLUMNS_ONLY", "0").strip() == "1",
            "gain_loss_weight": float(os.environ.get("CDC_GAIN_LOSS_WEIGHT", "0")),
            "std_gain_loss_weight": float(os.environ.get("CDC_STD_GAIN_LOSS_WEIGHT", "0")),
            "neg_force_weight": float(os.environ.get("CDC_NEG_FORCE_WEIGHT", "1.5")),
            "high_force_abs_threshold_N": float(os.environ.get("CDC_HIGH_FORCE_ABS_THRESHOLD_N", "0")),
            "high_force_weight": float(os.environ.get("CDC_HIGH_FORCE_WEIGHT", "1.0")),
            "near_zero_v_threshold_mm_s": float(os.environ.get("CDC_NEAR_ZERO_V_THRESHOLD_MM_S", "0")),
            "near_zero_v_weight": float(os.environ.get("CDC_NEAR_ZERO_V_WEIGHT", "1.0")),
            "high_force_near_zero_weight": float(os.environ.get("CDC_HIGH_FORCE_NEAR_ZERO_WEIGHT", "1.0")),
        },
    }
    if optimizer is not None:
        payload["optimizer_state"] = copy.deepcopy(optimizer.state_dict())
    if scheduler is not None:
        payload["scheduler_state"] = copy.deepcopy(scheduler.state_dict())
    torch.save(payload, path)


def env_flag(*names, default=False):
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value == "1"
    return default


def adapt_legacy_state_dict(model, state_dict):
    current = model.state_dict()
    adapted = {}
    expanded = []
    for key, tensor in state_dict.items():
        clean_key = key.replace("_orig_mod.", "")
        if clean_key not in current:
            adapted[clean_key] = tensor
            continue
        target = current[clean_key]
        if tuple(tensor.shape) == tuple(target.shape):
            adapted[clean_key] = tensor
            continue
        can_expand_first_layer = (
            clean_key in {"nn_x.0.weight", "nn_y.0.weight", "nn_y_direct.weight"}
            and tensor.ndim == 2
            and target.ndim == 2
            and tensor.shape[0] == target.shape[0]
            and tensor.shape[1] < target.shape[1]
        )
        if can_expand_first_layer:
            merged = target.clone()
            merged[:, : tensor.shape[1]] = tensor
            merged[:, tensor.shape[1] :] = 0.0
            adapted[clean_key] = merged
            expanded.append((clean_key, int(tensor.shape[1]), int(target.shape[1])))
            continue
        raise RuntimeError(
            f"Cannot resume parameter {clean_key}: checkpoint shape={tuple(tensor.shape)}, "
            f"model shape={tuple(target.shape)}"
        )
    return adapted, expanded


def load_resume_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError(f"Resume checkpoint does not contain state_dict: {checkpoint_path}")
    state_dict, expanded = adapt_legacy_state_dict(model, checkpoint["state_dict"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    architecture_changed = bool(expanded or missing or unexpected)
    print(f"[*] Resumed legacy weights from: {checkpoint_path}")
    for key, old_dim, new_dim in expanded:
        print(f"[*] Expanded {key}: NFL inputs {old_dim} -> {new_dim}; new columns initialized to zero.")
    if expanded:
        model._new_nfl_start = min(old_dim for _, old_dim, _ in expanded)
    if missing or unexpected:
        print(f"[!] Resume non-strict load | missing={len(missing)} unexpected={len(unexpected)}")
    if "score_nrmse_pct" in checkpoint:
        print(f"[*] Resume checkpoint score: {checkpoint['score_nrmse_pct']:.2f}%")
    if "epoch" in checkpoint:
        print(f"[*] Resume checkpoint epoch: {checkpoint['epoch']}")
    if architecture_changed:
        print("[*] Architecture changed, using fresh optimizer/scheduler state.")
    if optimizer is not None and "optimizer_state" in checkpoint and not architecture_changed:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("[*] Resumed optimizer state.")
    if scheduler is not None and "scheduler_state" in checkpoint and not architecture_changed:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        print("[*] Resumed scheduler state.")

    lr_override = os.environ.get("CDC_RESUME_LR_OVERRIDE", "").strip()
    if lr_override and optimizer is not None:
        lr = float(lr_override)
        for group in optimizer.param_groups:
            group["lr"] = lr
        print(f"[*] Resume LR override: {lr:.8g}")


def load_resume_norm_cfg(checkpoint_path):
    if not checkpoint_path or os.environ.get("CDC_USE_RESUME_NORM", "0").strip() != "1":
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "norm_cfg" not in checkpoint:
        raise RuntimeError(f"Resume checkpoint does not contain norm_cfg: {checkpoint_path}")
    return checkpoint["norm_cfg"]


def main():
    seed = int(os.environ.get("CDC_SEED", "20260427"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    epochs = int(os.environ.get("CDC_EPOCHS", "1000"))
    hidden = int(os.environ.get("CDC_HIDDEN", "160"))
    h_dim = int(os.environ.get("CDC_STATE_DIM", "6"))
    lr = float(os.environ.get("CDC_LR", "1e-3"))
    chunk_len = int(os.environ.get("CDC_CHUNK_LEN", "1000"))
    warmup = int(os.environ.get("CDC_WARMUP", "50"))
    train_window_seconds = float(os.environ.get("CDC_TRAIN_WINDOW_SECONDS", "0"))
    train_window_len = int(round(train_window_seconds * 1000.0)) if train_window_seconds > 0 else 0
    train_window_random = os.environ.get("CDC_TRAIN_WINDOW_RANDOM", "1").strip() == "1"
    train_window_start_seconds = float(os.environ.get("CDC_TRAIN_WINDOW_START_SECONDS", "0"))
    train_window_start_sample = int(round(train_window_start_seconds * 1000.0)) if train_window_start_seconds > 0 else 0
    val_warmup_ms = int(os.environ.get("CDC_VAL_WARMUP_MS", "50"))
    monitor_every = int(os.environ.get("CDC_MONITOR_EVERY", "20"))
    val_all_every = int(os.environ.get("CDC_VAL_ALL_EVERY", "20"))
    val_max_seconds = float(os.environ.get("CDC_VAL_MAX_SECONDS", "5"))
    val_max_samples = int(round(val_max_seconds * 1000.0)) if val_max_seconds > 0 else 0
    val_start_seconds = float(os.environ.get("CDC_VAL_START_SECONDS", "0"))
    val_start_sample = int(round(val_start_seconds * 1000.0)) if val_start_seconds > 0 else 0
    val_every_seconds = float(os.environ.get("CDC_VAL_EVERY_SECONDS", "0"))
    val_all_max_files = int(os.environ.get("CDC_VAL_ALL_MAX_FILES", "0"))
    file_batch_size = int(os.environ.get("CDC_FILE_BATCH_SIZE", "48"))
    length_buckets = os.environ.get("CDC_LENGTH_BUCKETS", "1") == "1"
    validate_all_on_epoch1 = os.environ.get("CDC_VALIDATE_ALL_ON_EPOCH1", "1") == "1"
    hidden_norm_weight = float(os.environ.get("CDC_HIDDEN_NORM_WEIGHT", "0"))
    dh_norm_weight = float(os.environ.get("CDC_DH_NORM_WEIGHT", "0"))
    p2p_loss_power = float(os.environ.get("CDC_P2P_LOSS_POWER", "2.0"))
    warmup_once = os.environ.get("CDC_WARMUP_ONCE", "1").strip() == "1"
    train_new_nfl_columns_only = os.environ.get("CDC_TRAIN_NEW_NFL_COLUMNS_ONLY", "0").strip() == "1"
    gain_loss_weight = float(os.environ.get("CDC_GAIN_LOSS_WEIGHT", "0"))
    std_gain_loss_weight = float(os.environ.get("CDC_STD_GAIN_LOSS_WEIGHT", "0"))
    slope_weight = float(os.environ.get("CDC_SLOPE_WEIGHT", "0"))
    curv_weight = float(os.environ.get("CDC_CURV_WEIGHT", "0"))
    neg_force_weight = float(os.environ.get("CDC_NEG_FORCE_WEIGHT", "1.5"))
    high_force_abs_threshold_N = float(os.environ.get("CDC_HIGH_FORCE_ABS_THRESHOLD_N", "0"))
    high_force_weight = float(os.environ.get("CDC_HIGH_FORCE_WEIGHT", "1.0"))
    near_zero_v_threshold_mm_s = float(os.environ.get("CDC_NEAR_ZERO_V_THRESHOLD_MM_S", "0"))
    near_zero_v_weight = float(os.environ.get("CDC_NEAR_ZERO_V_WEIGHT", "1.0"))
    high_force_near_zero_weight = float(os.environ.get("CDC_HIGH_FORCE_NEAR_ZERO_WEIGHT", "1.0"))
    extra_dynamic_nfl = env_flag("CDC_EXTRA_DYNAMIC_NFL", "CDC_LEGACY_EXTRA_DYNAMIC_NFL")
    direct_nfl_output = env_flag("CDC_DIRECT_NFL_OUTPUT", "CDC_LEGACY_DIRECT_NFL_OUTPUT")
    slow_gain_nfl = env_flag("CDC_SLOW_GAIN_NFL", "CDC_LEGACY_SLOW_GAIN_NFL")
    hysteresis_gain_nfl = env_flag("CDC_HYSTERESIS_GAIN_NFL", "CDC_LEGACY_HYSTERESIS_GAIN_NFL")
    nfl_slim = os.environ.get("CDC_NFL_SLIM", "0").strip() == "1"
    physics_nfl = os.environ.get("CDC_PHYSICS_NFL", "0").strip() == "1"
    physics_temp = os.environ.get("CDC_PHYSICS_TEMP", "0").strip() == "1"
    physics_path = os.environ.get("CDC_PHYSICS_PATH", "0").strip() == "1"
    strict_mask = os.environ.get("CDC_LEGACY_STRICT_MASK", "0") == "1"
    plot_every = int(os.environ.get("CDC_PLOT_EVERY", "0"))
    periodic_every = int(os.environ.get("CDC_PERIODIC_EVERY", "100"))
    show_tqdm = os.environ.get("CDC_TQDM", "0") == "1"

    out_root = Path(os.environ.get("CDC_OUTPUT_ROOT", REPO_ROOT / "training_outputs"))
    out_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"legacy_repro_{epochs}_epochs_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "legacy_metrics.csv"
    plots_dir = out_dir / "plots"

    data_root = Path(os.environ.get("CDC_PROCESSED_GROUPS", PROCESSED_GROUPS))
    resume_checkpoint = os.environ.get("CDC_RESUME_CHECKPOINT", "").strip()
    manager = LegacyCDCDataManager(data_root, norm_cfg_override=load_resume_norm_cfg(resume_checkpoint))
    high_force_abs_threshold_norm = high_force_abs_threshold_N / manager.stats.cfg["f_scale"] if high_force_abs_threshold_N > 0 else float("inf")
    near_zero_v_threshold_norm = near_zero_v_threshold_mm_s / manager.stats.cfg["v_scale"] if near_zero_v_threshold_mm_s > 0 else 0.0
    train_items = manager.preload_files(manager.train_files, device, "train")

    model = LegacyNLCSNN(
        u_dim=6,
        h_dim=h_dim,
        hidden=hidden,
        extra_dynamic_nfl=extra_dynamic_nfl,
        direct_nfl_output=direct_nfl_output,
        slow_gain_nfl=slow_gain_nfl,
        hysteresis_gain_nfl=hysteresis_gain_nfl,
        nfl_slim=nfl_slim,
        physics_nfl=physics_nfl,
        physics_temp=physics_temp,
        physics_path=physics_path,
        phys_scales=(
            manager.stats.cfg["v_scale"], manager.stats.cfg["a_scale"],
            manager.stats.cfg["x_scale"], manager.stats.cfg["i_scale"],
            manager.stats.cfg["f_scale"],
        ),
    ).to(device)
    if physics_path and model.physics is not None:
        _pi_path = REPO_ROOT / "mr_nlcsnn" / "phys_init.npy"
        if _pi_path.exists() and os.environ.get("CDC_PHYS_USE_FIT_INIT", "1").strip() == "1":
            _pi = np.load(_pi_path)
            with torch.no_grad():
                _p = model.physics
                for _attr, _val in zip(("c", "f1", "f2", "b1", "b2", "d1", "d2"), _pi[:7]):
                    getattr(_p, _attr).fill_(float(_val))
                _p.c0.copy_(torch.tensor(_pi[7:12], dtype=torch.float32, device=device))
                _p.a1.copy_(torch.tensor(_pi[12:17], dtype=torch.float32, device=device))
                _p.a2.copy_(torch.tensor(_pi[17:19], dtype=torch.float32, device=device))
                _p.k.copy_(torch.tensor(_pi[19:22], dtype=torch.float32, device=device))
            print("[*] PhysicsForce initialized from fitted phys_init.npy (data-adapted, ~81% baseline)")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    cosine_finetune = os.environ.get("CDC_COSINE_FINETUNE", "0").strip() == "1"
    scheduler_patience = int(os.environ.get("CDC_SCHEDULER_PATIENCE", "8"))
    if cosine_finetune:
        cosine_eta_min = float(os.environ.get("CDC_COSINE_ETA_MIN", "1e-5"))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cosine_eta_min)
        print(f"[*] Cosine fine-tune scheduler: T_max={epochs}, eta_min={cosine_eta_min:.3g}, base_lr={lr:.3g}")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=scheduler_patience, factor=0.5)
    criterion = nn.MSELoss(reduction="none")
    if resume_checkpoint:
        resume_weights_only = os.environ.get("CDC_RESUME_WEIGHTS_ONLY", "0").strip() == "1"
        load_resume_checkpoint(
            model,
            None if resume_weights_only else optimizer,
            None if resume_weights_only else scheduler,
            resume_checkpoint,
            device,
        )
        if resume_weights_only:
            print("[*] Resume weights-only: optimizer/scheduler start fresh (cosine base_lr preserved).")

    print(
        f"[*] Legacy repro session: {out_dir}\n"
        f"[*] Device={device} | model NFL={model.nfl_dim}, hidden={hidden}, h_dim={h_dim}, "
        f"extra_dynamic_nfl={extra_dynamic_nfl}, direct_nfl_output={direct_nfl_output}, "
        f"slow_gain_nfl={slow_gain_nfl}, hysteresis_gain_nfl={hysteresis_gain_nfl}\n"
        f"[*] epochs={epochs}, chunk_len={chunk_len}, warmup={warmup}, warmup_once={warmup_once}, "
        f"train_new_nfl_columns_only={train_new_nfl_columns_only}, "
        f"train_window_seconds={train_window_seconds:g}, train_window_random={train_window_random}, "
        f"val_warmup_ms={val_warmup_ms}, lr={lr}, scheduler_patience={scheduler_patience}, strict_mask={strict_mask}, "
        f"file_batch_size={file_batch_size}, length_buckets={length_buckets}, "
        f"h_norm_w={hidden_norm_weight:g}, dh_norm_w={dh_norm_weight:g}, "
        f"p2p_loss_power={p2p_loss_power:g}, "
        f"gain_loss_w={gain_loss_weight:g}, std_gain_loss_w={std_gain_loss_weight:g}\n"
        f"[*] point weights: neg_force={neg_force_weight:g}, "
        f"high_force>{high_force_abs_threshold_N:g}N x{high_force_weight:g}, "
        f"near_zero_v<{near_zero_v_threshold_mm_s:g}mm/s x{near_zero_v_weight:g}, "
        f"combo x{high_force_near_zero_weight:g}\n"
        f"[*] include_step={os.environ.get('CDC_INCLUDE_STEP', '0').strip() == '1'}, "
        f"include_hf={os.environ.get('CDC_INCLUDE_HF', '0').strip() == '1'}, "
        f"use_resume_norm={os.environ.get('CDC_USE_RESUME_NORM', '0').strip() == '1'}\n"
        f"[*] train_cats={os.environ.get('CDC_TRAIN_CATS', '(default)').strip()}, "
        f"val_cats={os.environ.get('CDC_VAL_CATS', '(default)').strip()}, "
        f"val_files={os.environ.get('CDC_VAL_FILES', '(all)').strip()}\n"
        f"[*] validation monitor: every={monitor_every}, start_seconds={val_start_seconds:g}, "
        f"max_seconds={val_max_seconds:g}, "
        f"time_every_seconds={val_every_seconds:g}, "
        f"all_val_every={val_all_every}, "
        f"all_val_max_files={val_all_max_files if val_all_max_files > 0 else 'all'}, "
        f"validate_all_on_epoch1={validate_all_on_epoch1}, periodic_every={periodic_every}"
    )

    best_monitor = float("inf")
    best_all = float("inf")
    best_monitor_path = out_dir / "nlcsnn_legacy_best_monitor.pth"
    best_all_path = out_dir / "nlcsnn_legacy_best_all_val.pth"
    last_path = out_dir / "nlcsnn_legacy_last.pth"
    periodic_dir = out_dir / "periodic_checkpoints"
    last_timed_val = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total_epoch_loss = 0.0
        batch_count = 0

        batches = make_file_batches(train_items, max(1, file_batch_size), length_buckets=length_buckets)
        iterator = tqdm(batches, desc=f"Legacy Epoch {epoch:03d}", ascii=True, leave=False, disable=not show_tqdm)
        for batch_items in iterator:
            batch_loss = train_file_batch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                batch_items=batch_items,
                chunk_len=chunk_len,
                warmup=warmup,
                strict_mask=strict_mask,
                hidden_norm_weight=hidden_norm_weight,
                dh_norm_weight=dh_norm_weight,
                p2p_loss_power=p2p_loss_power,
                neg_force_weight=neg_force_weight,
                high_force_abs_threshold_norm=high_force_abs_threshold_norm,
                high_force_weight=high_force_weight,
                near_zero_v_threshold_norm=near_zero_v_threshold_norm,
                near_zero_v_weight=near_zero_v_weight,
                high_force_near_zero_weight=high_force_near_zero_weight,
                gain_loss_weight=gain_loss_weight,
                std_gain_loss_weight=std_gain_loss_weight,
                warmup_once=warmup_once,
                train_new_nfl_columns_only=train_new_nfl_columns_only,
                device=device,
                train_window_len=train_window_len,
                random_window=train_window_random,
                train_window_start_sample=train_window_start_sample,
                slope_weight=slope_weight,
                curv_weight=curv_weight,
            )
            total_epoch_loss += batch_loss
            batch_count += 1
            if show_tqdm:
                iterator.set_postfix(loss=f"{batch_loss:.5f}")

        train_loss = total_epoch_loss / max(batch_count, 1)

        now_after_train = time.perf_counter()
        timed_val_due = val_every_seconds > 0 and (now_after_train - last_timed_val) >= val_every_seconds
        do_monitor = epoch == epochs or timed_val_due or (monitor_every > 0 and epoch % monitor_every == 0)
        do_all_val = (
            (validate_all_on_epoch1 and epoch == 1)
            or epoch == epochs
            or timed_val_due
            or (val_all_every > 0 and epoch % val_all_every == 0)
        )

        monitor = None
        monitor_nrmse = float("nan")
        monitor_h_mean = float("nan")
        monitor_h_max = float("nan")
        all_mean = float("nan")
        all_max = float("nan")

        # Prefer all-validation when it is due. This avoids evaluating the first
        # validation file twice and makes LR scheduling follow the real 5-file
        # representative objective instead of a single monitor case.
        if do_all_val:
            all_rows = evaluate_files(
                model,
                manager,
                manager.val_files,
                device,
                warmup_ms=val_warmup_ms,
                max_files=val_all_max_files,
                max_samples=val_max_samples,
                start_sample=val_start_sample,
            )
            if not all_rows:
                raise RuntimeError("Validation produced no rows. Check CDC_VAL_FILES, CDC_VAL_MAX_SECONDS, and CDC_VAL_START_SECONDS.")
            all_mean = float(np.mean([r["nrmse_pct"] for r in all_rows]))
            all_max = float(np.max([r["nrmse_pct"] for r in all_rows]))
            pd.DataFrame([{k: v for k, v in r.items() if k not in ("pred_force", "target_force")} for r in all_rows]).to_csv(
                out_dir / f"epoch_{epoch:04d}_all_val.csv",
                index=False,
            )

            monitor_case = manager.val_files[0].stem
            monitor = next((r for r in all_rows if r["case"] == monitor_case), all_rows[0])
            monitor_nrmse = monitor["nrmse_pct"]
            monitor_h_mean = monitor["h_norm_mean"]
            monitor_h_max = monitor["h_norm_max"]
        elif do_monitor:
            monitor_rows = evaluate_files(
                model,
                manager,
                manager.val_files[:1],
                device,
                warmup_ms=val_warmup_ms,
                max_samples=val_max_samples,
                start_sample=val_start_sample,
            )
            if not monitor_rows:
                raise RuntimeError("Monitor validation produced no rows. Check validation settings.")
            monitor = monitor_rows[0]
            monitor_nrmse = monitor["nrmse_pct"]
            monitor_h_mean = monitor["h_norm_mean"]
            monitor_h_max = monitor["h_norm_max"]

        if cosine_finetune:
            scheduler.step()
        else:
            scheduler_metric = all_mean if np.isfinite(all_mean) else monitor_nrmse
            if np.isfinite(scheduler_metric):
                scheduler.step(scheduler_metric)

        if timed_val_due:
            last_timed_val = time.perf_counter()

        if np.isfinite(all_mean) and all_mean < best_all:
            best_all = all_mean
            save_checkpoint(best_all_path, model, manager, hidden, h_dim, epoch, all_mean, optimizer=optimizer, scheduler=scheduler)

        if np.isfinite(monitor_nrmse) and monitor_nrmse < best_monitor:
            best_monitor = monitor_nrmse
            save_checkpoint(best_monitor_path, model, manager, hidden, h_dim, epoch, monitor_nrmse, optimizer=optimizer, scheduler=scheduler)

        if monitor is not None and plot_every > 0 and (epoch == 1 or epoch % plot_every == 0):
            save_monitor_plot(monitor, plots_dir / f"epoch_{epoch:04d}_{monitor['case']}.png")

        if periodic_every > 0 and epoch % periodic_every == 0:
            periodic_score = all_mean if np.isfinite(all_mean) else monitor_nrmse
            save_checkpoint(
                periodic_dir / f"nlcsnn_legacy_epoch_{epoch:04d}.pth",
                model,
                manager,
                hidden,
                h_dim,
                epoch,
                periodic_score,
                optimizer=optimizer,
                scheduler=scheduler,
            )

        epoch_seconds = time.perf_counter() - t0
        append_metrics(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "monitor_nrmse": monitor_nrmse,
                "all_mean": all_mean,
                "all_max": all_max,
                "h_mean": monitor_h_mean,
                "h_max": monitor_h_max,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": epoch_seconds,
            },
        )

        all_text = f" | all_val mean={all_mean:.2f}% max={all_max:.2f}%" if do_all_val else ""
        monitor_text = (
            f"monitor {monitor['case']}={monitor_nrmse:.2f}%"
            if do_monitor
            else "monitor skipped"
        )
        h_text = (
            f"||h|| mean={monitor_h_mean:.3f} max={monitor_h_max:.3f}"
            if do_monitor
            else "||h|| mean=nan max=nan"
        )
        print(
            f"E[{epoch:03d}/{epochs}] loss={train_loss:.6f} | "
            f"{monitor_text}{all_text} | "
            f"{h_text} | "
            f"lr={optimizer.param_groups[0]['lr']:.3g} | {epoch_seconds:.1f}s",
            flush=True,
        )

    last_score = all_mean if np.isfinite(all_mean) else monitor_nrmse
    save_checkpoint(last_path, model, manager, hidden, h_dim, epochs, last_score, optimizer=optimizer, scheduler=scheduler)
    print("[*] Legacy repro training complete.")
    print(f"[*] Last checkpoint: {last_path}")
    print(f"[*] Best monitor checkpoint: {best_monitor_path} | {best_monitor:.2f}%")
    print(f"[*] Best all-val checkpoint: {best_all_path} | {best_all:.2f}%")


if __name__ == "__main__":
    main()
