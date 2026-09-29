import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


SAMPLE_CASES = [
    "RMS2mm_7Hz",
    "RMS5mm_7Hz",
    "RMS5mm_6Hzslow",
    "RMS15mm_3Hzslow",
    "RMS18mm_3Hzslow",
    "RMS10mm_4Hz",
    "AMP10mm_Vel0.13",
    "AMP30mm_Vel0.13",
    "AMP10mm_4.97Hz_0.8A",
]

CANDIDATES = {
    "old_cpu": {"disp": 30.0, "force": 30.0, "di_dt": 40.0},
    "balanced": {"disp": 40.0, "force": 30.0, "di_dt": 40.0},
    "new_bandwidth_zp": {"disp": 50.0, "force": 25.0, "di_dt": 40.0},
    "force_wide": {"disp": 50.0, "force": 40.0, "di_dt": 60.0},
}

FS = 1000.0
FILTER_ORDER = 4
FC_CURRENT = 40.0
FC_TEMP = 2.0

CATEGORY_DIRS = {
    1: "1_Steady_Harmonic",
    2: "2_Steady_Random",
    3: "3_Random_HF_Current",
    4: "4_Random_LF_Current",
    5: "5_Step_Current_Triangle",
}


def sos_lowpass_zero_phase(data, cutoff, fs=FS, order=FILTER_ORDER):
    values = np.asarray(data, dtype=np.float64)
    if len(values) == 0 or cutoff >= fs / 2.0:
        return values.copy()
    sos = butter(order, cutoff / (0.5 * fs), btype="low", output="sos")
    # sosfiltfilt needs enough samples for padding. These files are long, but
    # this fallback keeps the diagnostic script robust for accidental short files.
    try:
        return sosfiltfilt(sos, values)
    except ValueError:
        padlen = max(0, min(len(values) - 1, 3 * (2 * len(sos) + 1)))
        return sosfiltfilt(sos, values, padlen=padlen)


def read_raw_csv(path):
    skip = 15
    with open(path, "r", encoding="gbk", errors="ignore") as f:
        for i, line in enumerate(f):
            if "#EndHeader" in line:
                skip = i + 1
                break
    return pd.read_csv(path, skiprows=skip, encoding="gbk")


def classify_case(folder_name):
    if "Vel" in folder_name:
        return 5
    has_current_suffix = re.search(r"_(\d+\.?\d*)A$", folder_name)
    if has_current_suffix:
        return 1 if "AMP" in folder_name else 2
    return 3 if folder_name.startswith("aRMS") else 4


def find_sample_paths(raw_root, sample_cases):
    raw_root = Path(raw_root)
    by_name = {}
    for csv_path in raw_root.rglob("auto_1.csv"):
        if csv_path.parent.name in sample_cases:
            by_name[csv_path.parent.name] = csv_path
    missing = sorted(set(sample_cases) - set(by_name))
    if missing:
        raise FileNotFoundError(f"Missing sample cases under {raw_root}: {missing}")
    return {name: by_name[name] for name in sample_cases}


def find_reference_paths(reference_root, sample_cases):
    reference_root = Path(reference_root)
    by_name = {}
    for csv_path in reference_root.rglob("*.csv"):
        if csv_path.stem in sample_cases:
            by_name[csv_path.stem] = csv_path
    missing = sorted(set(sample_cases) - set(by_name))
    if missing:
        raise FileNotFoundError(f"Missing reference processed cases under {reference_root}: {missing}")
    return {name: by_name[name] for name in sample_cases}


def best_lag_ms(a, b, max_lag=120):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n < 20:
        return np.nan, np.nan
    a = a[:n]
    b = b[:n]
    a = a - np.mean(a)
    b = b - np.mean(b)
    best = (0, -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa, bb = a[-lag:], b[: n + lag]
        elif lag > 0:
            aa, bb = a[: n - lag], b[lag:]
        else:
            aa, bb = a, b
        if len(aa) < 20:
            continue
        denom = (np.std(aa) * np.std(bb)) + 1e-12
        corr = float(np.mean((aa - aa.mean()) * (bb - bb.mean())) / denom)
        if corr > best[1]:
            best = (lag, corr)
    return int(best[0]), float(best[1])


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if len(x) else np.nan


def process_case(raw_path, candidate):
    raw = read_raw_csv(raw_path)
    folder = raw_path.parent.name
    category = classify_case(folder)

    d_raw = ((raw.iloc[:, 3] - 4.0) / 16.0 * 1500.0).to_numpy(dtype=np.float64)
    f_raw = ((raw.iloc[:, 5] / 10.0) * 1000.0 * 9.80665).to_numpy(dtype=np.float64)
    i_raw = (raw.iloc[:, 6] / 1000.0).to_numpy(dtype=np.float64)
    t_raw = raw.iloc[:, 8].to_numpy(dtype=np.float64)

    d_f = sos_lowpass_zero_phase(d_raw, candidate["disp"])
    f_f = sos_lowpass_zero_phase(f_raw, candidate["force"])
    i_f = sos_lowpass_zero_phase(i_raw, FC_CURRENT)
    t_f = sos_lowpass_zero_phase(t_raw, FC_TEMP)

    if category in (1, 2, 5):
        active_idx = np.where(np.abs(d_f - np.mean(d_f[:200])) > 0.15)[0]
    else:
        active_idx = np.where(i_f > 0.02)[0]
    if len(active_idx) == 0:
        raise RuntimeError(f"No active segment found: {folder}")

    s = int(active_idx[0])
    e = int(active_idx[-1])
    baseline = slice(0, s) if s > 10 else slice(0, min(200, len(d_f)))
    mts_ref = float(np.mean(d_f[baseline]))
    force_offset = float(np.mean(f_f[baseline]))

    rod_length = 250.0 - (d_f[s:e] - mts_ref)
    force = f_f[s:e] - force_offset
    current = i_f[s:e]
    temp = t_f[s:e]

    velocity_raw = np.gradient(rod_length, 0.001)
    velocity = sos_lowpass_zero_phase(velocity_raw, candidate["disp"])
    accel_raw = np.gradient(velocity, 0.001)
    accel = sos_lowpass_zero_phase(accel_raw, candidate["disp"])
    current_dot_raw = np.gradient(current, 0.001)
    current_dot = sos_lowpass_zero_phase(current_dot_raw, candidate["di_dt"])

    df = pd.DataFrame(
        {
            "time": np.arange(len(rod_length), dtype=np.float64) * 0.001,
            "rod_length": rod_length,
            "velocity": velocity,
            "accel": accel,
            "force": force,
            "current": current,
            "current_dot": current_dot,
            "temp": temp,
        }
    )
    raw_active = {
        "disp": d_raw[s:e],
        "disp_filtered": d_f[s:e],
        "force": f_raw[s:e],
        "force_filtered": f_f[s:e],
        "current": i_raw[s:e],
        "current_filtered": i_f[s:e],
        "start": s,
        "end": e,
        "category": CATEGORY_DIRS[category],
    }
    return df, raw_active


def plot_channels(case_name, df, out_path):
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    cols = [
        ("rod_length", "Rod length (mm)", "tab:green"),
        ("velocity", "Velocity (mm/s)", "tab:orange"),
        ("accel", "Acceleration (mm/s^2)", "tab:purple"),
        ("force", "Force (N)", "tab:red"),
        ("current", "Current (A)", "tab:blue"),
        ("current_dot", "Current dot (A/s)", "tab:cyan"),
    ]
    t = df["time"].to_numpy()
    for ax, (col, title, color) in zip(axes.flat, cols):
        ax.plot(t, df[col], color=color, linewidth=0.9)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(case_name, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_raw_overlay(case_name, raw_active, out_path):
    n = len(raw_active["disp"])
    t = np.arange(n, dtype=np.float64) * 0.001
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(t, raw_active["disp"], color="gray", alpha=0.35, label="raw")
    axes[0].plot(t, raw_active["disp_filtered"], color="tab:green", label="filtered")
    axes[0].set_ylabel("Disp raw (mm)")
    axes[1].plot(t, raw_active["force"], color="gray", alpha=0.35, label="raw")
    axes[1].plot(t, raw_active["force_filtered"], color="tab:red", label="filtered")
    axes[1].set_ylabel("Force raw (N)")
    axes[2].plot(t, raw_active["current"], color="gray", alpha=0.35, label="raw")
    axes[2].plot(t, raw_active["current_filtered"], color="tab:blue", label="filtered")
    axes[2].set_ylabel("Current raw (A)")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{case_name} raw vs zero-phase filtered", fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def metrics_for_case(case_name, candidate_name, df, raw_active, reference_path):
    ref = pd.read_csv(reference_path)
    rows = {
        "candidate": candidate_name,
        "case": case_name,
        "category": raw_active["category"],
        "active_start_raw_idx": raw_active["start"],
        "active_end_raw_idx": raw_active["end"],
        "active_samples": int(len(df)),
        "active_duration_s": float(len(df) * 0.001),
        "force_p2p_N": float(df["force"].max() - df["force"].min()),
        "rod_p2p_mm": float(df["rod_length"].max() - df["rod_length"].min()),
        "velocity_rms": rms(df["velocity"]),
        "velocity_abs_max": float(np.max(np.abs(df["velocity"]))),
        "accel_rms": rms(df["accel"]),
        "accel_abs_max": float(np.max(np.abs(df["accel"]))),
        "current_dot_rms": rms(df["current_dot"]),
        "current_dot_abs_max": float(np.max(np.abs(df["current_dot"]))),
        "disp_residual_rms_raw_minus_filtered": rms(raw_active["disp"] - raw_active["disp_filtered"]),
        "force_residual_rms_raw_minus_filtered": rms(raw_active["force"] - raw_active["force_filtered"]),
        "current_residual_rms_raw_minus_filtered": rms(raw_active["current"] - raw_active["current_filtered"]),
        "reference_samples": int(len(ref)),
        "sample_count_delta_vs_reference": int(len(df) - len(ref)),
    }
    for signal, ref_col, new_col in [
        ("rod", "rod_length", "rod_length"),
        ("force", "force", "force"),
        ("current", "current", "current"),
    ]:
        lag, corr = best_lag_ms(ref[ref_col].to_numpy(), df[new_col].to_numpy())
        rows[f"{signal}_lag_ms_vs_processed_new_raw"] = lag
        rows[f"{signal}_corr_vs_processed_new_raw"] = corr
    return rows


def run_grid(raw_root, reference_root, output_dir, sample_cases):
    raw_paths = find_sample_paths(raw_root, sample_cases)
    reference_paths = find_reference_paths(reference_root, sample_cases)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for candidate_name, candidate in CANDIDATES.items():
        candidate_dir = output_dir / candidate_name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for case_name in tqdm(sample_cases, desc=f"filter {candidate_name}", ascii=True):
            df, raw_active = process_case(raw_paths[case_name], candidate)
            safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_name)
            plot_channels(case_name, df, candidate_dir / f"{safe_case}_channels.png")
            plot_raw_overlay(case_name, raw_active, candidate_dir / f"{safe_case}_raw_overlay.png")
            row = metrics_for_case(case_name, candidate_name, df, raw_active, reference_paths[case_name])
            rows.append(row)
            all_metrics.append(row)
        pd.DataFrame(rows).to_csv(candidate_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(output_dir / "all_candidates_metrics.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for candidate_name, group in metrics.groupby("candidate"):
        lag_cols = [
            "rod_lag_ms_vs_processed_new_raw",
            "force_lag_ms_vs_processed_new_raw",
            "current_lag_ms_vs_processed_new_raw",
        ]
        abs_lag = group[lag_cols].abs().to_numpy().reshape(-1)
        summary_rows.append(
            {
                "candidate": candidate_name,
                "mean_abs_lag_ms": float(np.nanmean(abs_lag)),
                "max_abs_lag_ms": float(np.nanmax(abs_lag)),
                "mean_force_p2p_N": float(group["force_p2p_N"].mean()),
                "mean_rod_p2p_mm": float(group["rod_p2p_mm"].mean()),
                "mean_accel_rms": float(group["accel_rms"].mean()),
                "mean_current_dot_rms": float(group["current_dot_rms"].mean()),
                "mean_force_residual_rms": float(group["force_residual_rms_raw_minus_filtered"].mean()),
                "mean_disp_residual_rms": float(group["disp_residual_rms_raw_minus_filtered"].mean()),
                "mean_current_residual_rms": float(group["current_residual_rms_raw_minus_filtered"].mean()),
                "total_abs_sample_count_delta": int(group["sample_count_delta_vs_reference"].abs().sum()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["mean_abs_lag_ms", "mean_accel_rms"])
    summary.to_csv(output_dir / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    return metrics, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run zero-phase cutoff diagnostics on selected raw Testdata cases.")
    parser.add_argument(
        "--raw-dir",
        default=r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Testdata",
        help="Raw Testdata directory containing sample case folders.",
    )
    parser.add_argument(
        "--reference-dir",
        default=r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Processed_Groups_new_raw",
        help="Current processed data root for lag/reference comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "filter_cutoff_reports" / "Testdata_zero_phase_grid"),
        help="Output directory for diagnostic plots and CSV files.",
    )
    parser.add_argument(
        "--cases",
        default=",".join(SAMPLE_CASES),
        help="Comma-separated case folder names to process.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sample_cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    metrics, summary = run_grid(args.raw_dir, args.reference_dir, args.output_dir, sample_cases)
    print(f"Saved cutoff diagnostics to: {Path(args.output_dir).resolve()}")
    print("\nCandidate summary:")
    print(summary.to_string(index=False))
    print("\nPer-case metrics rows:", len(metrics))


if __name__ == "__main__":
    main()
