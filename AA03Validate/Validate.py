import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from mr_nlcsnn.model import NLCSNN, rk4_step

try:
    from train_config import PROCESSED_GROUPS
except ImportError:
    PROCESSED_GROUPS = r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Processed_Groups"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CDCValidator:
    def __init__(self, pth_path):
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
        self.cfg = checkpoint['norm_cfg']

        model_cfg = checkpoint.get('model_cfg', {})
        self.model = NLCSNN(
            hidden=int(model_cfg.get('hidden', 128)),
            h_dim=int(model_cfg.get('h_dim', 8)),
        ).to(device)

        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('_orig_mod.', '')
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        print(f"[*] Model and Normalization Config loaded from: {pth_path}")

    def validate_case(self, csv_path, exclude_ms=50, save_dir: Optional[Path] = None):
        df = pd.read_csv(csv_path)
        cfg = self.cfg

        u = np.stack([
            (df['rod_length'].values - cfg['x_ref']) / cfg['x_scale'],
            df['velocity'].values / cfg['v_scale'],
            df['accel'].values / cfg['a_scale'],
            df['current'].values / cfg['i_scale'],
            df['current_dot'].values / cfg['di_scale'],
            df['temp'].values / cfg['t_scale']
        ], axis=1)
        u_tensor = torch.FloatTensor(u).to(device)
        f_measured = df['force'].values

        h = torch.zeros(1, self.model.h_dim).to(device)
        preds = []
        print(f"[*] Simulating: {csv_path.name}...")

        with torch.no_grad():
            for t in range(len(u_tensor)):
                u_t = u_tensor[t:t+1]
                p_t = self.model.predict_force(u_t, h).item()
                preds.append(p_t * cfg['f_scale'])
                h = rk4_step(self.model, u_t, h)

        f_pred = np.array(preds)

        start_idx = int(exclude_ms)
        f_meas_s = f_measured[start_idx:]
        f_pred_s = f_pred[start_idx:]
        time_s = df['time'].values[start_idx:]
        abs_err = np.abs(f_meas_s - f_pred_s)

        rmse = np.sqrt(np.mean((f_meas_s - f_pred_s)**2))
        nrmse = (rmse / (f_meas_s.max() - f_meas_s.min() + 1e-6)) * 100
        mae = np.mean(abs_err)

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            safe = csv_path.stem.replace(" ", "_")
            out_csv = save_dir / f"{safe}_timeseries.csv"
            pd.DataFrame({
                "time_s": time_s,
                "force_measured_N": f_meas_s,
                "force_predicted_N": f_pred_s,
                "abs_error_N": abs_err,
            }).to_csv(out_csv, index=False)
            fig_path = save_dir / f"{safe}_plot.png"
            self._plot_stacked(
                case_name=csv_path.stem,
                time=time_s,
                measured=f_meas_s,
                predicted=f_pred_s,
                error=abs_err,
                nrmse=nrmse,
                mae=mae,
                save_path=fig_path,
            )
            print(f"    saved: {out_csv.name}, {fig_path.name}")
        else:
            self._plot_stacked(
                case_name=csv_path.stem,
                time=time_s,
                measured=f_meas_s,
                predicted=f_pred_s,
                error=abs_err,
                nrmse=nrmse,
                mae=mae,
                save_path=None,
            )

        return {"nrmse": nrmse, "mae": mae, "rmse": rmse}

    def _plot_stacked(self, case_name, time, measured, predicted, error, nrmse, mae, save_path: Optional[Path]):
        """Time-Force comparison only (Plan.md §II.3)."""
        fig, ax = plt.subplots(1, 1, figsize=(13, 5))
        fig.suptitle(f"CDC NLCSNN Validation: {case_name}\nSteady-State Analysis (T > 50ms)", fontsize=14)

        ax.plot(time, measured, color='black', alpha=0.4, label='Measured')
        ax.plot(time, predicted, color='red', linestyle='--', linewidth=1.2, label='Predicted')
        ax.set_ylabel("Damping Force (N)")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Force Trajectory (NRMSE: {nrmse:.2f}%, MAE: {mae:.2f} N)")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.2)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path is not None:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    train_out = REPO_ROOT / "AA02Train" / "outputs"
    _candidates = list(train_out.rglob("nlcsnn_ultra.pth"))
    MODEL_PATH = str(max(_candidates, key=lambda p: p.stat().st_mtime)) if _candidates else ""

    DATA_DIR = Path(PROCESSED_GROUPS)

    VAL_CASES = [
        "4_Random_LF_Current/RMS16.67mm_1.66Hz.csv",
        "3_Random_HF_Current/aRMS10mm_4Hz.csv",
        "2_Steady_Random/RMS10mm_4.97Hz_0.2A.csv"
    ]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = REPO_ROOT / "AA03Validate" / "results" / f"run_{ts}"

    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        print("Error: No checkpoint under AA02Train/outputs/. Train first (AA02Train/TrainFinalVersion.py).")
    else:
        validator = CDCValidator(MODEL_PATH)
        summary_rows = []

        for case_rel_path in VAL_CASES:
            full_path = DATA_DIR / case_rel_path
            if full_path.exists():
                m = validator.validate_case(full_path, exclude_ms=50, save_dir=RESULTS_DIR)
                summary_rows.append({
                    "case": case_rel_path,
                    "nrmse_pct": m["nrmse"],
                    "mae_N": m["mae"],
                    "rmse_N": m["rmse"],
                })
            else:
                print(f"Warning: Data file not found: {full_path}")

        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "summary.csv", index=False)
            print(f"\n[*] All results in: {RESULTS_DIR}")
