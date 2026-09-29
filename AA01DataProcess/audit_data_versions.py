import argparse
import os
from pathlib import Path

import pandas as pd


DEFAULT_OLD_RAW = r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Testdata - 副本"
DEFAULT_NEW_RAW = r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Testdata"


def count_rows(path):
    try:
        with open(path, "r", encoding="gbk", errors="ignore") as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def classify_case(folder_name):
    if "Vel" in folder_name:
        return "5_Step_Current_Triangle"
    if folder_name.startswith("AMP") and folder_name.endswith("A"):
        return "1_Steady_Harmonic"
    if folder_name.startswith("RMS") and folder_name.endswith("A"):
        return "2_Steady_Random"
    if folder_name.startswith("aRMS"):
        return "3_Random_HF_Current"
    if folder_name.startswith("RMS"):
        return "4_Random_LF_Current"
    return "Unknown"


def inventory_raw(root, label):
    root = Path(root)
    rows = []
    if not root.exists():
        raise FileNotFoundError(f"{label} raw directory not found: {root}")
    for csv_path in sorted(root.rglob("auto_1.csv")):
        case = csv_path.parent.name
        stat = csv_path.stat()
        rows.append(
            {
                "version": label,
                "case": case,
                "category_guess": classify_case(case),
                "path": str(csv_path),
                "size_bytes": stat.st_size,
                "last_write_time": pd.Timestamp.fromtimestamp(stat.st_mtime),
                "raw_line_count": count_rows(csv_path),
            }
        )
    return pd.DataFrame(rows)


def compare_versions(old_df, new_df):
    old_cases = set(old_df["case"])
    new_cases = set(new_df["case"])
    rows = []
    for case in sorted(old_cases | new_cases):
        old_row = old_df[old_df["case"] == case]
        new_row = new_df[new_df["case"] == case]
        status = "common"
        if case in old_cases and case not in new_cases:
            status = "old_only"
        elif case not in old_cases and case in new_cases:
            status = "new_only"
        rows.append(
            {
                "case": case,
                "status": status,
                "category_guess": (
                    new_row["category_guess"].iloc[0]
                    if not new_row.empty
                    else old_row["category_guess"].iloc[0]
                ),
                "old_size_bytes": None if old_row.empty else int(old_row["size_bytes"].iloc[0]),
                "new_size_bytes": None if new_row.empty else int(new_row["size_bytes"].iloc[0]),
                "old_line_count": None if old_row.empty else int(old_row["raw_line_count"].iloc[0]),
                "new_line_count": None if new_row.empty else int(new_row["raw_line_count"].iloc[0]),
                "old_last_write_time": None if old_row.empty else old_row["last_write_time"].iloc[0],
                "new_last_write_time": None if new_row.empty else new_row["last_write_time"].iloc[0],
            }
        )
    out = pd.DataFrame(rows)
    out["line_count_delta"] = out["new_line_count"].fillna(0) - out["old_line_count"].fillna(0)
    out["size_delta_bytes"] = out["new_size_bytes"].fillna(0) - out["old_size_bytes"].fillna(0)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Audit old/new CDC raw Testdata versions.")
    parser.add_argument("--old-raw", default=os.environ.get("CDC_OLD_RAW_TESTDATA", DEFAULT_OLD_RAW))
    parser.add_argument("--new-raw", default=os.environ.get("CDC_NEW_RAW_TESTDATA", DEFAULT_NEW_RAW))
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "CDC_DATA_AUDIT_DIR",
            r"D:\OneDrive - mail.scut.edu.cn\Python\MR-NLCSNN\training_outputs\data_audit",
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    old_df = inventory_raw(args.old_raw, "old")
    new_df = inventory_raw(args.new_raw, "new")
    comparison = compare_versions(old_df, new_df)

    old_df.to_csv(output_dir / "raw_inventory_old.csv", index=False, encoding="utf-8-sig")
    new_df.to_csv(output_dir / "raw_inventory_new.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "raw_version_comparison.csv", index=False, encoding="utf-8-sig")

    print("--- Raw Version Audit ---")
    print(f"Old raw: {args.old_raw}")
    print(f"New raw: {args.new_raw}")
    print(f"Old files: {len(old_df)}")
    print(f"New files: {len(new_df)}")
    print("\nBy version/category:")
    both = pd.concat([old_df, new_df], ignore_index=True)
    print(both.groupby(["version", "category_guess"])["case"].count())
    print("\nCase status:")
    print(comparison.groupby("status")["case"].count())
    print(f"\nAudit files saved to: {output_dir}")


if __name__ == "__main__":
    main()
