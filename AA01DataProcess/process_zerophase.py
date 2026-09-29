import argparse
import os
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

FS = 1000.0
FILTER_ORDER = 4
FC_CURRENT = 40.0
FC_TEMP = 2.0

CANDIDATES = {
    "old_cpu": {"disp": 30.0, "force": 30.0, "di_dt": 40.0},
    "balanced": {"disp": 40.0, "force": 30.0, "di_dt": 40.0},
    "new_bandwidth_zp": {"disp": 50.0, "force": 25.0, "di_dt": 40.0},
    "force_wide": {"disp": 50.0, "force": 40.0, "di_dt": 60.0},
}

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
        return 5, None
    current_match = re.search(r"_(\d+\.?\d*)A$", folder_name)
    if current_match:
        current_a = float(current_match.group(1))
        return (1 if "AMP" in folder_name else 2), current_a
    return (3 if folder_name.startswith("aRMS") else 4), None


def process_file(csv_path, candidate):
    raw = read_raw_csv(csv_path)
    folder = csv_path.parent.name
    category, current_a = classify_case(folder)

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
        return None, {"folder": folder, "category": CATEGORY_DIRS[category], "error": "no_active_segment"}

    s = int(active_idx[0])
    e = int(active_idx[-1])
    baseline = slice(0, s) if s > 10 else slice(0, min(200, len(d_f)))
    mts_ref = float(np.mean(d_f[baseline]))
    force_offset = float(np.mean(f_f[baseline]))

    rod_length = 250.0 - (d_f[s:e] - mts_ref)
    force = f_f[s:e] - force_offset
    current = i_f[s:e]
    temp = t_f[s:e]

    velocity = sos_lowpass_zero_phase(np.gradient(rod_length, 0.001), candidate["disp"])
    accel = sos_lowpass_zero_phase(np.gradient(velocity, 0.001), candidate["disp"])
    current_dot = sos_lowpass_zero_phase(np.gradient(current, 0.001), candidate["di_dt"])

    df = pd.DataFrame(
        {
            "time": np.arange(len(rod_length), dtype=np.float64) * 0.001,
            "rod_length": np.round(rod_length, 3),
            "velocity": np.round(velocity, 3),
            "accel": np.round(accel, 2),
            "force": np.round(force, 2),
            "current": np.round(current, 3),
            "current_dot": np.round(current_dot, 3),
            "temp": np.round(temp, 2),
            "importance_weight": 1.0 / len(rod_length),
        }
    )
    meta = {
        "folder": folder,
        "category": CATEGORY_DIRS[category],
        "current_a": current_a,
        "active_start_raw_idx": s,
        "active_end_raw_idx": e,
        "samples": len(df),
        "duration_s": len(df) * 0.001,
        "force_min": float(df["force"].min()),
        "force_max": float(df["force"].max()),
        "force_p2p": float(df["force"].max() - df["force"].min()),
        "rod_min": float(df["rod_length"].min()),
        "rod_max": float(df["rod_length"].max()),
        "rod_p2p": float(df["rod_length"].max() - df["rod_length"].min()),
        "current_min": float(df["current"].min()),
        "current_max": float(df["current"].max()),
        "current_dot_abs_max": float(df["current_dot"].abs().max()),
        "velocity_abs_max": float(df["velocity"].abs().max()),
        "accel_abs_max": float(df["accel"].abs().max()),
    }
    return df, meta


def export_all(raw_dir, output_dir, candidate_name, clean_output=False):
    if candidate_name not in CANDIDATES:
        raise ValueError(f"Unknown candidate {candidate_name}. Choices: {sorted(CANDIDATES)}")
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    for folder in CATEGORY_DIRS.values():
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.rglob("auto_1.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No auto_1.csv files found under {raw_dir}")

    manifest_rows = []
    failures = []
    candidate = CANDIDATES[candidate_name]
    for csv_path in tqdm(raw_files, desc=f"zerophase {candidate_name}", ascii=True):
        df, meta = process_file(csv_path, candidate)
        if df is None:
            failures.append({"source_path": str(csv_path), **meta})
            continue
        out_path = output_dir / meta["category"] / f"{meta['folder']}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        manifest_rows.append(
            {
                "source_path": str(csv_path),
                "output_path": str(out_path),
                "filter_candidate": candidate_name,
                "fc_disp": candidate["disp"],
                "fc_force": candidate["force"],
                "fc_curr": FC_CURRENT,
                "fc_di_dt": candidate["di_dt"],
                "fc_temp": FC_TEMP,
                "filter_order": FILTER_ORDER,
                **meta,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False, encoding="utf-8-sig")
    counts = manifest.groupby("category")["folder"].count().sort_index()
    print("\n--- Zero-phase Export Summary ---")
    print(counts)
    print(f"Exported files: {len(manifest)} / {len(raw_files)}")
    print(f"Manifest: {output_dir / 'manifest.csv'}")
    if failures:
        print(f"Failures: {output_dir / 'failures.csv'}")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Export full Testdata using zero-phase cutoff candidate.")
    parser.add_argument(
        "--raw-dir",
        default=r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Testdata",
        help="Raw Testdata directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Processed_Groups_new_zerophase_old_cpu",
        help="Processed output root.",
    )
    parser.add_argument("--candidate", default="old_cpu", choices=sorted(CANDIDATES))
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    export_all(args.raw_dir, args.output_dir, args.candidate, clean_output=args.clean_output)


if __name__ == "__main__":
    main()
