import argparse
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mr_nlcsnn.legacy_model import LegacyNLCSNN, rk4_step

try:
    from train_config import PROCESSED_GROUPS
except ImportError:
    PROCESSED_GROUPS = r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Processed_Groups"


def get_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_case_amp_freq(name):
    m = re.match(r"^(?:AMP|a?RMS)(?P<amp>\d+\.?\d*)(?:mm)?_(?P<freq>\d+\.?\d*)Hz", name)
    if not m:
        return None, None
    return float(m.group("amp")), float(m.group("freq"))


def select_lf_holdout_files(lf_files, ratio=0.25, seed=20260516, explicit=""):
    if explicit.strip():
        wanted = {name.strip().removesuffix(".csv") for name in explicit.split(",") if name.strip()}
        selected = [p for p in lf_files if p.stem in wanted]
        missing = sorted(wanted - {p.stem for p in selected})
        if missing:
            raise RuntimeError(f"LF validation files not found: {missing}")
        return selected

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


class LegacyCDCValidator:
    def __init__(self, model_path, device):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise RuntimeError("Expected checkpoint with keys: state_dict, norm_cfg")
        if "norm_cfg" not in checkpoint:
            raise RuntimeError("Checkpoint has no norm_cfg. Use the training normalization config.")

        self.cfg = checkpoint["norm_cfg"]
        model_cfg = checkpoint.get("model_cfg", {})
        hidden = int(model_cfg.get("hidden", 160))
        h_dim = int(model_cfg.get("h_dim", 6))
        extra_dynamic_nfl = bool(model_cfg.get("extra_dynamic_nfl", False))
        direct_nfl_output = bool(model_cfg.get("direct_nfl_output", False))
        slow_gain_nfl = bool(model_cfg.get("slow_gain_nfl", False))
        hysteresis_gain_nfl = bool(model_cfg.get("hysteresis_gain_nfl", False))
        nfl_slim = bool(model_cfg.get("nfl_slim", False))
        physics_nfl = bool(model_cfg.get("physics_nfl", False))
        physics_temp = bool(model_cfg.get("physics_temp", False))
        physics_path = bool(model_cfg.get("physics_path", False))
        self.model = LegacyNLCSNN(
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
                self.cfg["v_scale"], self.cfg["a_scale"],
                self.cfg["x_scale"], self.cfg["i_scale"], self.cfg["f_scale"],
            ),
        ).to(device)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["state_dict"].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[*] Loaded legacy model: {model_path}")
        print(
            f"[*] Model cfg: hidden={hidden}, h_dim={h_dim}, "
            f"nfl_dim={self.model.nfl_dim}, extra_dynamic_nfl={extra_dynamic_nfl}, "
            f"direct_nfl_output={direct_nfl_output}, slow_gain_nfl={slow_gain_nfl}, "
            f"hysteresis_gain_nfl={hysteresis_gain_nfl}"
        )

    def load_and_norm(self, path):
        df = pd.read_csv(path)
        cfg = self.cfg
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
        return torch.from_numpy(u).to(self.device), df

    @torch.no_grad()
    def validate_case(
        self,
        path,
        warmup_ms=200,
        start_sample=0,
        max_samples=0,
        score_start_sample=None,
        score_max_samples=0,
        baseline_cal_samples=0,
    ):
        u_seq, df = self.load_and_norm(path)
        source_start = 0
        if start_sample > 0 or max_samples > 0:
            start_idx = min(max(0, int(start_sample)), max(0, len(u_seq) - 1))
            end_idx = len(u_seq) if max_samples <= 0 else min(len(u_seq), start_idx + int(max_samples))
            u_seq = u_seq[start_idx:end_idx]
            df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            source_start = start_idx
        h = torch.zeros(1, self.model.h_dim, device=self.device)
        preds = []
        h_norms = []
        for t in range(len(u_seq)):
            u_t = u_seq[t : t + 1]
            preds.append(self.model.predict_force(u_t, h).item())
            h = rk4_step(self.model, u_t, h)
            h_norms.append(float(torch.norm(h, dim=-1).item()))

        p_force = np.asarray(preds, dtype=np.float32) * self.cfg["f_scale"]
        t_force = df["force"].values.astype(np.float32)
        time = df["time"].values if "time" in df.columns else np.arange(len(t_force)) * 0.001
        warmup_start = min(int(warmup_ms), max(0, len(t_force) // 4))
        if score_start_sample is None:
            eval_start = warmup_start
        else:
            eval_start = max(warmup_start, int(score_start_sample) - source_start)
            eval_start = min(max(0, eval_start), max(0, len(t_force) - 1))
        eval_end = len(t_force) if score_max_samples <= 0 else min(len(t_force), eval_start + int(score_max_samples))
        if eval_end <= eval_start:
            raise RuntimeError(f"Empty evaluation window for {path}: start={eval_start}, end={eval_end}")
        # Deployable baseline (zero-point) calibration: the per-file force offset is
        # constant, so estimate it from the first baseline_cal_samples of the eval
        # window using the force sensor, subtract it, then score AFTER that window.
        # This is the standard sensor zeroing step a semi-active controller performs.
        if baseline_cal_samples > 0:
            cal_end = min(eval_start + int(baseline_cal_samples), eval_end)
            if cal_end > eval_start:
                offset = float(np.mean(p_force[eval_start:cal_end] - t_force[eval_start:cal_end]))
                p_force = p_force - offset
                eval_start = cal_end
        if eval_end <= eval_start:
            raise RuntimeError(f"Empty scoring window after calibration for {path}")
        p_eval = p_force[eval_start:eval_end]
        t_eval = t_force[eval_start:eval_end]
        h_eval = h_norms[eval_start:eval_end]
        err = p_eval - t_eval
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        # RMS-normalised NRMSE (paper-consistent): robust to force zero-crossings.
        rms_true = float(np.sqrt(np.mean(t_eval ** 2)))
        nrmse = rmse / (rms_true + 1e-6) * 100.0

        return {
            "Case": path.stem,
            "NRMSE (%)": nrmse,
            "MAE (N)": mae,
            "RMSE (N)": rmse,
            "h_norm_mean": float(np.mean(h_eval)),
            "h_norm_max": float(np.max(h_eval)),
            "time": time,
            "measured": t_force,
            "predicted": p_force,
            "warmup_s": time[eval_start] if len(time) else eval_start * 0.001,
        }

    def run(
        self,
        val_files,
        output_dir,
        warmup_ms=200,
        save_plots=True,
        start_sample=0,
        max_samples=0,
        score_start_sample=None,
        score_max_samples=0,
        baseline_cal_samples=0,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = []

        print(f"[*] Starting legacy validation on {len(val_files)} files")
        for path in val_files:
            row = self.validate_case(
                path,
                warmup_ms=warmup_ms,
                start_sample=start_sample,
                max_samples=max_samples,
                score_start_sample=score_start_sample,
                score_max_samples=score_max_samples,
                baseline_cal_samples=baseline_cal_samples,
            )
            summary.append({k: row[k] for k in ("Case", "NRMSE (%)", "MAE (N)", "RMSE (N)", "h_norm_mean", "h_norm_max")})
            print(
                f" -> {row['Case']}: NRMSE={row['NRMSE (%)']:.2f}% | "
                f"MAE={row['MAE (N)']:.2f} N | ||h|| max={row['h_norm_max']:.3f}"
            )
            if save_plots:
                self._plot_case(row, output_dir / f"Result_{row['Case']}.png")

        summary_df = pd.DataFrame(summary)
        out_csv = output_dir / "Validation_Summary.csv"
        summary_df.to_csv(out_csv, index=False)
        print("\n" + "=" * 48)
        print(f"Average NRMSE: {summary_df['NRMSE (%)'].mean():.2f}%")
        print(f"Max NRMSE: {summary_df['NRMSE (%)'].max():.2f}%")
        print(f"Results saved to: {output_dir.resolve()}")
        print("=" * 48)
        return summary_df

    def _plot_case(self, row, out_path):
        time = row["time"]
        measured = row["measured"]
        predicted = row["predicted"]
        abs_err = np.abs(measured - predicted)

        fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
        axes[0].plot(time, measured, color="black", alpha=0.35, label="Measured")
        axes[0].plot(time, predicted, color="red", linestyle="--", linewidth=1.1, label="NLCSNN Pred")
        axes[0].axvline(x=row["warmup_s"], color="orange", linestyle=":", label="Warmup End")
        axes[0].set_ylabel("Force (N)")
        axes[0].set_title(f"{row['Case']} | NRMSE={row['NRMSE (%)']:.2f}%")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(time, abs_err, color="purple", linewidth=1.0)
        axes[1].set_ylabel("Absolute Error (N)")
        axes[1].set_xlabel("Time (s)")
        axes[1].grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def find_default_checkpoint():
    candidates = sorted(
        (REPO_ROOT / "training_outputs").rglob("nlcsnn_legacy_best_all_val.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            (REPO_ROOT / "training_outputs").rglob("nlcsnn_legacy_best_monitor.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    return candidates[0] if candidates else None


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a legacy CPU-era NLCSNN checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to legacy checkpoint .pth")
    parser.add_argument("--data-root", type=str, default=str(PROCESSED_GROUPS), help="Processed_Groups root")
    parser.add_argument("--val-category", type=str, default="3_Random_HF_Current", help="Validation category folder")
    parser.add_argument("--split-mode", type=str, default="category", choices=["category", "lf_holdout"], help="Validation file selection mode")
    parser.add_argument("--lf-val-ratio", type=float, default=0.25, help="Per-amplitude LF holdout ratio")
    parser.add_argument("--lf-val-seed", type=int, default=20260516, help="Deterministic LF holdout seed")
    parser.add_argument("--lf-val-files", type=str, default="", help="Comma-separated LF validation stems or .csv names")
    parser.add_argument("--output-dir", type=str, default="", help="Report output directory")
    parser.add_argument("--warmup-ms", type=int, default=200, help="Initial samples to exclude")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Validation window start time in seconds")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Validation window duration in seconds; 0 means full file")
    parser.add_argument("--score-start-seconds", type=float, default=-1.0, help="Score window start in original-file seconds; negative means use warmup")
    parser.add_argument("--score-max-seconds", type=float, default=0.0, help="Score window duration in seconds; 0 means to the end")
    parser.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto")
    parser.add_argument("--no-plots", action="store_true", help="Skip Result_*.png generation")
    parser.add_argument("--max-files", type=int, default=0, help="Limit validation files for smoke tests")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint) if args.checkpoint else find_default_checkpoint()
    if checkpoint is None or not checkpoint.exists():
        raise RuntimeError("No checkpoint found. Pass --checkpoint or run AA02Train/train_legacy_repro.py first.")

    data_root = Path(args.data_root)
    if args.split_mode == "lf_holdout":
        val_dir = data_root / "4_Random_LF_Current"
        val_files = select_lf_holdout_files(
            sorted(val_dir.glob("*.csv")),
            ratio=args.lf_val_ratio,
            seed=args.lf_val_seed,
            explicit=args.lf_val_files,
        )
    else:
        val_dir = data_root / args.val_category
        val_files = sorted(val_dir.glob("*.csv"))
    # Optional v_peak filter: drop quasi-static cases (v_peak < threshold m/s) where
    # friction dominates and the draw-wire jitter floor makes them not performance-relevant.
    min_vpeak = float(os.environ.get("CDC_MIN_VPEAK_MS", "0"))
    if min_vpeak > 0:
        def _vpeak_ok(p):
            try:
                vv = pd.read_csv(p, usecols=["velocity"])["velocity"].values
                return (np.percentile(np.abs(vv), 99) / 1000.0) >= min_vpeak
            except Exception:
                return True
        n0 = len(val_files)
        val_files = [p for p in val_files if _vpeak_ok(p)]
        print(f"[*] v_peak>={min_vpeak} m/s filter: val {n0}->{len(val_files)}")
    if args.max_files > 0:
        val_files = val_files[: args.max_files]
    if not val_files:
        raise RuntimeError(f"No validation files found: {val_dir}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "AA03Validate" / "reports" / f"Legacy_Validation_{ts}"

    validator = LegacyCDCValidator(checkpoint, get_device(args.device.strip() or None))
    start_sample = int(round(args.start_seconds * 1000.0)) if args.start_seconds > 0 else 0
    max_samples = int(round(args.max_seconds * 1000.0)) if args.max_seconds > 0 else 0
    score_start_sample = int(round(args.score_start_seconds * 1000.0)) if args.score_start_seconds >= 0 else None
    score_max_samples = int(round(args.score_max_seconds * 1000.0)) if args.score_max_seconds > 0 else 0
    baseline_cal_samples = int(round(float(os.environ.get("CDC_BASELINE_CAL_MS", "0"))))
    if baseline_cal_samples > 0:
        print(f"[*] Deployable baseline calibration: {baseline_cal_samples} ms zeroing window")
    validator.run(
        val_files,
        output_dir,
        warmup_ms=args.warmup_ms,
        save_plots=not args.no_plots,
        start_sample=start_sample,
        max_samples=max_samples,
        score_start_sample=score_start_sample,
        score_max_samples=score_max_samples,
        baseline_cal_samples=baseline_cal_samples,
    )


if __name__ == "__main__":
    main()
