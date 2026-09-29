import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

FS = 1000.0
DT = 1.0 / FS
FILTER_ORDER = 4
FC_CURRENT = 40.0
FC_TEMP = 2.0

DEFAULT_ARCHIVE_DIR = Path(r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Raw_Card_Categorized_All")
DEFAULT_OUTPUT_DIR = Path(r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Parsed_Card_force_wide")

CANDIDATES = {
    "old_cpu": {"disp": 30.0, "force": 30.0, "di_dt": 40.0},
    "balanced": {"disp": 40.0, "force": 30.0, "di_dt": 40.0},
    "new_bandwidth_zp": {"disp": 50.0, "force": 25.0, "di_dt": 40.0},
    "force_wide": {"disp": 50.0, "force": 40.0, "di_dt": 60.0},
}

CATEGORY_MAP = {
    "1_SH": "1_Steady_Harmonic",
    "2_SR": "2_Steady_Random",
    "3_RHF": "3_Random_HF_Current",
    "4_RLF": "4_Random_LF_Current",
    "5_ST": "5_Step_Current_Triangle",
}

OUTPUT_COLUMNS = [
    "time",
    "rod_length",
    "velocity",
    "accel",
    "force",
    "current",
    "current_dot",
    "temp",
]


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def quality_string(flags):
    return ";".join(flags) if flags else "ok"


def detect_segment(category, condition_key, d_f, i_f):
    is_step = category == "5_ST" and "STEP" in condition_key.upper()
    if is_step:
        active_idx = np.where(i_f > 0.05)[0]
        detection = "current_gt_0p05_with_500pt_padding"
    elif category in {"1_SH", "2_SR", "5_ST"}:
        ref = np.mean(d_f[: min(200, len(d_f))])
        active_idx = np.where(np.abs(d_f - ref) > 0.15)[0]
        detection = "disp_gt_0p15"
    else:
        active_idx = np.where(i_f > 0.02)[0]
        detection = "current_gt_0p02"

    if len(active_idx) == 0:
        return None, detection

    active_first = int(active_idx[0])
    active_last = int(active_idx[-1])
    if is_step:
        crop_start = max(0, active_first - 500)
        crop_end = min(len(d_f), active_last + 501)
    else:
        crop_start = active_first
        crop_end = active_last + 1
    return {
        "active_start_raw_idx": active_first,
        "active_end_raw_idx": active_last,
        "crop_start_raw_idx": crop_start,
        "crop_end_raw_idx": crop_end,
        "is_step": is_step,
    }, detection


def find_static_zero(d_f, f_f, thr=0.15, min_len=200):
    """Displacement-static zero reference (recording start or end).

    Random-displacement experiments start MTS before acquisition, so the
    current-triggered pre-window is NOT quiescent (displacement already moving).
    The correct zero comes from a displacement-still window: at the start
    (acquisition began before MTS moved) or the end (MTS stopped before
    acquisition stopped). Returns
    (mts_ref, force_offset, zero_source, win_start, win_stop) or None.
    """
    n = len(d_f)
    ref_s = np.mean(d_f[:50])
    mv_s = np.where(np.abs(d_f - ref_s) > thr)[0]
    start_len = int(mv_s[0]) if len(mv_s) else n          # quiet samples at start
    ref_e = np.mean(d_f[-50:])
    mv_e = np.where(np.abs(d_f - ref_e) > thr)[0]
    end_len = int(n - 1 - mv_e[-1]) if len(mv_e) else n   # quiet samples at end
    if start_len >= min_len and start_len >= end_len:
        w = slice(0, start_len); src = "static-window-start"
    elif end_len >= min_len:
        w = slice(n - end_len, n); src = "static-window-end"
    else:
        return None
    return float(np.mean(d_f[w])), float(np.mean(f_f[w])), src, int(w.start), int(w.stop)


def physics_estimate_zero(d_f, f_f, seg, dt=DT):
    """Zero estimate for files with no static window (recording captured only the
    active region). Displacement zero = mean-active (validated ~1 mm). Force
    preload = mean force where the damper is near the neutral position with
    near-zero velocity (no damping contribution) ~ static preload to ~30 N.
    """
    d = np.asarray(d_f[seg], float); f = np.asarray(f_f[seg], float)
    neutral = float(np.mean(d))
    vel = np.gradient(d, dt)
    dstd = float(np.std(d)) + 1e-9
    vthr = np.percentile(np.abs(vel), 10)
    mask = (np.abs(d - neutral) < 0.1 * dstd) & (np.abs(vel) < vthr)
    preload = float(np.mean(f[mask])) if int(mask.sum()) > 20 else float(np.mean(f))
    return neutral, preload, "physics-estimate"


def process_file(row, candidate):
    input_path = Path(row["output_path"])
    category = row["category"]
    condition_key = row["condition_key"]
    flags = []

    if category not in CATEGORY_MAP:
        return None, {"error": f"unknown_category:{category}"}
    if not input_path.exists():
        return None, {"error": "missing_archive_csv"}

    raw = read_raw_csv(input_path)
    if raw.shape[1] <= 8:
        return None, {"error": f"not_enough_columns:{raw.shape[1]}"}

    d_raw = ((raw.iloc[:, 3] - 4.0) / 16.0 * 1500.0).to_numpy(dtype=np.float64)
    f_raw = ((raw.iloc[:, 5] / 10.0) * 1000.0 * 9.80665).to_numpy(dtype=np.float64)
    i_raw = (raw.iloc[:, 6] / 1000.0).to_numpy(dtype=np.float64)
    t_raw = raw.iloc[:, 8].to_numpy(dtype=np.float64)

    current_sign_corrected = parse_bool(row.get("is_reversed_current_required", False))
    if current_sign_corrected:
        i_raw = -i_raw
        flags.append("reversed-current-corrected")

    d_f = sos_lowpass_zero_phase(d_raw, candidate["disp"])
    f_f = sos_lowpass_zero_phase(f_raw, candidate["force"])
    i_f = sos_lowpass_zero_phase(i_raw, FC_CURRENT)
    t_f = sos_lowpass_zero_phase(t_raw, FC_TEMP)

    segment, active_detection = detect_segment(category, condition_key, d_f, i_f)
    if segment is None:
        return None, {"error": "no_active_segment", "active_detection": active_detection}

    active_start = segment["active_start_raw_idx"]
    crop_start = segment["crop_start_raw_idx"]
    crop_end = segment["crop_end_raw_idx"]
    if crop_end - crop_start < 3:
        return None, {"error": "active_segment_too_short", "active_detection": active_detection, **segment}

    # Zeroing: prefer a displacement-static window (start or end); else estimate
    # the preload physically from near-neutral, near-zero-velocity points.
    static = find_static_zero(d_f, f_f)
    if static is not None:
        mts_ref, force_offset, zero_source, base_start, base_stop = static
    else:
        mts_ref, force_offset, zero_source = physics_estimate_zero(
            d_f, f_f, slice(crop_start, crop_end))
        base_start, base_stop = -1, -1
        flags.append("estimated-zero")
    if parse_bool(row.get("is_manual_zero_required", False)):
        zero_source = "auto-baseline-override"
        flags.append("manual-zero-required")

    if segment["is_step"] and segment["active_end_raw_idx"] >= len(i_f) - 2:
        flags.append("truncated-current")
    if parse_bool(row.get("is_legacy_1p4A_step", False)):
        flags.append("legacy-1.4A-step")

    rod_length = 250.0 - (d_f[crop_start:crop_end] - mts_ref)
    force = f_f[crop_start:crop_end] - force_offset
    current = i_f[crop_start:crop_end]
    temp = t_f[crop_start:crop_end]

    velocity = sos_lowpass_zero_phase(np.gradient(rod_length, DT), candidate["disp"])
    accel = sos_lowpass_zero_phase(np.gradient(velocity, DT), candidate["disp"])
    current_dot = sos_lowpass_zero_phase(np.gradient(current, DT), candidate["di_dt"])

    df = pd.DataFrame(
        {
            "time": np.arange(len(rod_length), dtype=np.float64) * DT,
            "rod_length": np.round(rod_length, 3),
            "velocity": np.round(velocity, 3),
            "accel": np.round(accel, 2),
            "force": np.round(force, 2),
            "current": np.round(current, 3),
            "current_dot": np.round(current_dot, 3),
            "temp": np.round(temp, 2),
        }
    )

    meta = {
        "archive_source_path": str(input_path),
        "original_source_path": row.get("source_path", ""),
        "source_root": row.get("source_root", ""),
        "source_folder": row.get("source_folder", ""),
        "category": CATEGORY_MAP[category],
        "archive_category": category,
        "short_name": row["short_name"],
        "condition_key": condition_key,
        "active_detection": active_detection,
        "active_start_raw_idx": segment["active_start_raw_idx"],
        "active_end_raw_idx": segment["active_end_raw_idx"],
        "crop_start_raw_idx": segment["crop_start_raw_idx"],
        "crop_end_raw_idx": segment["crop_end_raw_idx"],
        "baseline_start_raw_idx": base_start,
        "baseline_end_raw_idx": base_stop,
        "mts_ref": mts_ref,
        "force_offset": force_offset,
        "zero_source": zero_source,
        "manual_zero_required": parse_bool(row.get("is_manual_zero_required", False)),
        "current_sign_corrected": current_sign_corrected,
        "legacy_1p4A_step": parse_bool(row.get("is_legacy_1p4A_step", False)),
        "quality_flags": quality_string(flags),
        "samples": len(df),
        "duration_s": len(df) * DT,
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


def load_archive_manifest(archive_dir):
    manifest_path = archive_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Archive manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required = {"output_path", "source_path", "category", "short_name", "condition_key"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Archive manifest missing required columns: {missing}")
    return manifest


def write_readme(output_dir, candidate_name, source_archive):
    text = f"""# Parsed_Card_force_wide

This directory contains processed physical-quantity CSV files parsed from the raw card archive.

- Source archive: `{source_archive}`
- Filter candidate: `{candidate_name}`
- Filtering: Butterworth SOS + zero-phase `sosfiltfilt`
- Sampling rate: 1000 Hz
- This is not raw acquisition-card data.

Each case CSV contains:

- `time`
- `rod_length`
- `velocity`
- `accel`
- `force`
- `current`
- `current_dot`
- `temp`

Use `manifest.csv` for source provenance, active segment indices, zeroing information, filter settings, and quality flags.
Use `failures.csv` if any source files failed parsing.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def dry_run(manifest, output_dir):
    output_pairs = []
    for _, row in manifest.iterrows():
        category = str(row["category"])
        if category not in CATEGORY_MAP:
            output_pairs.append(("Unknown", str(row["short_name"])))
        else:
            output_pairs.append((CATEGORY_MAP[category], str(row["short_name"])))
    collisions = len(output_pairs) - len(set(output_pairs))
    print(f"Input manifest rows: {len(manifest)}")
    print(f"Output directory: {output_dir}")
    print(f"Output filename collisions: {collisions}")
    print("By archive category:")
    print(manifest["category"].value_counts().sort_index())
    print("Dry-run only; no files written.")
    if collisions:
        raise RuntimeError("Output filename collisions detected.")


def export_all(archive_dir, output_dir, candidate_name, clean_output=False, dry_run_only=False):
    if candidate_name not in CANDIDATES:
        raise ValueError(f"Unknown candidate {candidate_name}. Choices: {sorted(CANDIDATES)}")

    archive_dir = Path(archive_dir)
    output_dir = Path(output_dir)
    manifest = load_archive_manifest(archive_dir)
    if dry_run_only:
        dry_run(manifest, output_dir)
        return pd.DataFrame()

    if output_dir.exists():
        if not clean_output:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --clean-output to rebuild.")
        shutil.rmtree(output_dir)
    for category_dir in CATEGORY_MAP.values():
        (output_dir / category_dir).mkdir(parents=True, exist_ok=True)

    candidate = CANDIDATES[candidate_name]
    manifest_rows = []
    failures = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc=f"parse {candidate_name}", ascii=True):
        row_dict = {k: row[k] for k in manifest.columns}
        try:
            df, meta = process_file(row_dict, candidate)
        except Exception as exc:
            df, meta = None, {"error": f"exception:{type(exc).__name__}:{exc}"}

        if df is None:
            failures.append(
                {
                    "archive_source_path": str(row_dict.get("output_path", "")),
                    "original_source_path": str(row_dict.get("source_path", "")),
                    "short_name": str(row_dict.get("short_name", "")),
                    "condition_key": str(row_dict.get("condition_key", "")),
                    "category": str(row_dict.get("category", "")),
                    **meta,
                }
            )
            continue

        out_path = output_dir / meta["category"] / str(row["short_name"])
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        manifest_rows.append(
            {
                "output_path": str(out_path),
                "filter_candidate": candidate_name,
                "fc_disp": candidate["disp"],
                "fc_force": candidate["force"],
                "fc_curr": FC_CURRENT,
                "fc_di_dt": candidate["di_dt"],
                "fc_temp": FC_TEMP,
                "filter_order": FILTER_ORDER,
                "fs": FS,
                **meta,
            }
        )

    parsed_manifest = pd.DataFrame(manifest_rows)
    parsed_manifest.to_csv(output_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["archive_source_path", "original_source_path", "short_name", "condition_key", "category", "error"]).to_csv(
            output_dir / "failures.csv", index=False, encoding="utf-8-sig"
        )
    write_readme(output_dir, candidate_name, archive_dir)

    print("\n--- Parsed Card Export Summary ---")
    if not parsed_manifest.empty:
        print(parsed_manifest.groupby("category")["short_name"].count().sort_index())
    print(f"Exported files: {len(parsed_manifest)} / {len(manifest)}")
    print(f"Manifest: {output_dir / 'manifest.csv'}")
    print(f"Failures: {output_dir / 'failures.csv'}")
    return parsed_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Parse categorized raw card CSV archive into physical CDC model inputs.")
    parser.add_argument("--archive-dir", default=str(DEFAULT_ARCHIVE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--candidate", default="force_wide", choices=sorted(CANDIDATES))
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    export_all(
        archive_dir=args.archive_dir,
        output_dir=args.output_dir,
        candidate_name=args.candidate,
        clean_output=args.clean_output,
        dry_run_only=args.dry_run,
    )


if __name__ == "__main__":
    main()
