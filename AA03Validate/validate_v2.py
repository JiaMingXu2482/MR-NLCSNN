"""Validate an NLCSNN_v2 checkpoint.

Reconstructs the model from the checkpoint's model_cfg, rolls every requested
file open-loop from h=0, and reports RMS-normalised NRMSE + MAE(N) + hidden-state
norm. Also dumps the interpretable linear-path coefficient table.

Examples:
  # full LF set (all 37 files)
  python AA03Validate/validate_v2.py --checkpoint <path> --category 4_Random_LF_Current
  # only the held-out cases recorded in the checkpoint
  python AA03Validate/validate_v2.py --checkpoint <path> --holdout-only
"""
import argparse
import os
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

from mr_nlcsnn.model_v2 import NLCSNN_v2, rk4_step

DEFAULT_DATA_ROOT = r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Parsed_Card_force_wide"


def get_device(name=""):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValidatorV2:
    def __init__(self, ckpt_path, device):
        self.device = device
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "norm_cfg" not in ckpt or "state_dict" not in ckpt:
            raise RuntimeError("Checkpoint missing norm_cfg/state_dict")
        mc = ckpt.get("model_cfg", {})
        if mc.get("architecture") != "nlcsnn_v2":
            raise RuntimeError(f"Not an nlcsnn_v2 checkpoint: architecture={mc.get('architecture')}")
        self.cfg = ckpt["norm_cfg"]
        self.holdout_stems = ckpt.get("holdout_stems", [])
        self.model = NLCSNN_v2(
            u_dim=6,
            h_dim=int(mc.get("h_dim", 8)),
            hidden=int(mc.get("hidden", 192)),
            slow_gain=bool(mc.get("slow_gain", True)),
            extra_dynamic=bool(mc.get("extra_dynamic", True)),
            prune_h_coupling=bool(mc.get("prune_h_coupling", True)),
        ).to(device)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["state_dict"].items()}
        self.model.load_state_dict(sd)
        self.model.eval()
        print(f"[*] Loaded nlcsnn_v2: {ckpt_path}")
        print(f"[*] epoch={ckpt.get('epoch')} mean={ckpt.get('mean_nrmse_pct')} "
              f"max={ckpt.get('max_nrmse_pct')} nfl_dim={self.model.nfl_dim}")

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
        return torch.from_numpy(u).to(self.device), df

    @torch.no_grad()
    def validate_case(self, path, warmup_ms=50, max_samples=0):
        u, df = self._load_norm(path)
        if max_samples > 0 and len(u) > max_samples:
            u = u[:max_samples]
            df = df.iloc[:max_samples].reset_index(drop=True)
        h = torch.zeros(1, self.model.h_dim, device=self.device)
        preds, hns = [], []
        for t in range(len(u)):
            preds.append(self.model.predict_force(u[t:t + 1], h).item())
            h = rk4_step(self.model, u[t:t + 1], h)
            hns.append(float(torch.norm(h, dim=-1).item()))
        p = np.asarray(preds, dtype=np.float32) * self.cfg["f_scale"]
        y = df["force"].values.astype(np.float32)
        time = df["time"].values if "time" in df.columns else np.arange(len(y)) * 0.001
        s = min(int(warmup_ms), max(0, len(y) // 4))
        err = p[s:] - y[s:]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        rms_true = float(np.sqrt(np.mean(y[s:] ** 2))) + 1e-6
        return {
            "Case": path.stem,
            "NRMSE (%)": rmse / rms_true * 100.0,      # RMS-normalised
            "MAE (N)": float(np.mean(np.abs(err))),
            "RMSE (N)": rmse,
            "h_norm_mean": float(np.mean(hns[s:])),
            "h_norm_max": float(np.max(hns[s:])),
            "time": time, "measured": y, "predicted": p, "warmup_s": float(time[s]) if len(time) else 0.0,
        }

    def run(self, files, out_dir, warmup_ms=50, save_plots=True, max_samples=0):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = []
        print(f"[*] Validating {len(files)} files (RMS-normalised NRMSE)")
        for p in files:
            r = self.validate_case(p, warmup_ms=warmup_ms, max_samples=max_samples)
            summary.append({k: r[k] for k in
                            ("Case", "NRMSE (%)", "MAE (N)", "RMSE (N)", "h_norm_mean", "h_norm_max")})
            print(f" -> {r['Case']}: NRMSE={r['NRMSE (%)']:.2f}% | MAE={r['MAE (N)']:.1f} N | "
                  f"||h||max={r['h_norm_max']:.2f}")
            if save_plots:
                self._plot(r, out_dir / f"Result_{r['Case']}.png")
        sdf = pd.DataFrame(summary)
        sdf.to_csv(out_dir / "Validation_Summary.csv", index=False)
        self._dump_coeffs(out_dir / "nfl_linear_coeffs.csv")
        print("=" * 48)
        print(f"Mean NRMSE: {sdf['NRMSE (%)'].mean():.2f}% | Max NRMSE: {sdf['NRMSE (%)'].max():.2f}%")
        print(f"Saved: {out_dir.resolve()}")
        print("=" * 48)
        return sdf

    def _dump_coeffs(self, path):
        rows = [{"feature": n, "y_coeff": w} for n, w in self.model.nfl_linear_weights()]
        pd.DataFrame(rows).to_csv(path, index=False)
        print("[*] Top force-path NFL coefficients:")
        for n, w in self.model.nfl_linear_weights()[:12]:
            print(f"      {w:+.4f}  {n}")

    def _plot(self, r, out_path):
        t, meas, pred = r["time"], r["measured"], r["predicted"]
        fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
        ax[0].plot(t, meas, color="black", alpha=0.35, label="Measured")
        ax[0].plot(t, pred, color="red", linestyle="--", linewidth=1.1, label="NLCSNN_v2")
        ax[0].axvline(x=r["warmup_s"], color="orange", linestyle=":", label="Warmup end")
        ax[0].set_ylabel("Force (N)")
        ax[0].set_title(f"{r['Case']} | NRMSE(RMS)={r['NRMSE (%)']:.2f}% | MAE={r['MAE (N)']:.1f} N")
        ax[0].legend(loc="best"); ax[0].grid(True, alpha=0.25)
        ax[1].plot(t, np.abs(meas - pred), color="purple", linewidth=1.0)
        ax[1].set_ylabel("Abs error (N)"); ax[1].set_xlabel("Time (s)"); ax[1].grid(True, alpha=0.25)
        fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def find_default_checkpoint():
    cands = sorted((REPO_ROOT / "training_outputs").rglob("nlcsnn_v2_best.pth"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def parse_args():
    ap = argparse.ArgumentParser(description="Validate an NLCSNN_v2 checkpoint.")
    ap.add_argument("--checkpoint", default="", help="Path to nlcsnn_v2 .pth")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--category", default="4_Random_LF_Current")
    ap.add_argument("--holdout-only", action="store_true", help="Only the checkpoint's holdout stems")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--warmup-ms", type=int, default=50)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="Cap each rollout to N s (0=full file)")
    ap.add_argument("--device", default="")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--max-files", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint) if args.checkpoint else find_default_checkpoint()
    if ckpt is None or not ckpt.exists():
        raise RuntimeError("No checkpoint found. Pass --checkpoint or train first.")

    validator = ValidatorV2(ckpt, get_device(args.device.strip()))
    val_dir = Path(args.data_root) / args.category
    files = sorted(val_dir.glob("*.csv"))
    if args.holdout_only:
        keep = set(validator.holdout_stems)
        files = [p for p in files if p.stem in keep]
    if args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        raise RuntimeError(f"No validation files: {val_dir}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        tag = "holdout" if args.holdout_only else args.category
        out_dir = REPO_ROOT / "AA03Validate" / "reports" / f"v2_{ckpt.parent.name}_{tag}"
    max_samples = int(round(args.max_seconds * 1000.0)) if args.max_seconds > 0 else 0
    validator.run(files, out_dir, warmup_ms=args.warmup_ms,
                  save_plots=not args.no_plots, max_samples=max_samples)


if __name__ == "__main__":
    main()
