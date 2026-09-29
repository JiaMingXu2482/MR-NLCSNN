import argparse
import csv
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


DEFAULT_BASE = Path(r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest")
DEFAULT_OUTPUT_NAME = "Raw_Card_Categorized_All"

CATEGORY_DIRS = {
    "SH": "1_SH",
    "SR": "2_SR",
    "RHF": "3_RHF",
    "RLF": "4_RLF",
    "ST": "5_ST",
}

SOURCE_PRIORITY = {
    "Testdata": 10,
    "TestdataLegacyCopy": 20,
    "Data430": 30,
    "Random430": 40,
    "standard430": 50,
    "step430": 60,
}


def pnum(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace(".", "p")


def unpnum(value):
    return str(value).replace("p", ".")


def format_f_value(value):
    text = unpnum(value)
    if text.endswith("slow"):
        return f"{text[:-4]}Hzslow"
    if text.endswith("11") and "p" not in value and len(value) > 2:
        return f"{text[:-2]}Hz11"
    return f"{text}Hz"


def display_name_from_condition(condition_key):
    parts = condition_key.split("_")
    cat = parts[0]
    if cat == "SH":
        amp = unpnum(parts[1][1:])
        freq = format_f_value(parts[2][1:])
        curr = unpnum(parts[3][1:])
        return f"SH_{amp}mm_{freq}_{curr}A"
    if cat == "SR":
        rms = unpnum(parts[1][3:])
        freq = format_f_value(parts[2][1:])
        curr = unpnum(parts[3][1:])
        return f"SR_RMS{rms}mm_{freq}_{curr}A"
    if cat in {"RHF", "RLF"}:
        rms = unpnum(parts[1][3:])
        freq = format_f_value(parts[2][1:])
        return f"{cat}_RMS{rms}mm_{freq}"
    if cat == "ST":
        amp = unpnum(parts[1][1:])
        vel = unpnum(parts[2][1:])
        base = f"ST_{amp}mm_Vel{vel}"
        if len(parts) >= 6 and parts[3].startswith("STEP") and parts[5].startswith("I"):
            step_a = parts[3][4:]
            step_b = parts[4]
            curr = unpnum(parts[5][1:])
            base += f"_step{step_a}_{step_b}_{curr}A"
        return base
    return condition_key


def normalize_source_root(name):
    if name == "Testdata":
        return "Testdata"
    if name.startswith("Testdata -"):
        return "TestdataLegacyCopy"
    return name


def discover_source_roots(base_dir):
    wanted = {"Testdata", "Data430", "Random430", "standard430", "step430"}
    roots = []
    for path in sorted(base_dir.iterdir(), key=lambda p: p.name):
        if not path.is_dir():
            continue
        if path.name in wanted or path.name.startswith("Testdata -"):
            roots.append(path)
    return roots


def classify_and_setting(folder_name):
    step_match = re.fullmatch(
        r"AMP(?P<amp>\d+(?:\.\d+)?)mm_Vel(?P<vel>\d+(?:\.\d+)?)_step(?P<step_a>\d+)_"
        r"(?P<step_b>\d+)_(?P<current>\d+(?:\.\d+)?)A(?:_(?P<repeat>\d+))?",
        folder_name,
    )
    if step_match:
        d = step_match.groupdict()
        setting = (
            f"A{pnum(d['amp'])}_V{pnum(d['vel'])}_"
            f"STEP{d['step_a']}_{d['step_b']}_I{pnum(d['current'])}"
        )
        return "ST", setting, d.get("repeat") or ""

    standard_match = re.fullmatch(r"AMP(?P<amp>\d+(?:\.\d+)?)mm_Vel(?P<vel>\d+(?:\.\d+)?)", folder_name)
    if standard_match:
        d = standard_match.groupdict()
        return "ST", f"A{pnum(d['amp'])}_V{pnum(d['vel'])}", ""

    harmonic_match = re.fullmatch(
        r"AMP(?P<amp>\d+(?:\.\d+)?)mm_(?P<freq>\d+(?:\.\d+)?)Hz_(?P<current>\d+(?:\.\d+)?)A",
        folder_name,
    )
    if harmonic_match:
        d = harmonic_match.groupdict()
        return "SH", f"A{pnum(d['amp'])}_F{pnum(d['freq'])}_I{pnum(d['current'])}", ""

    steady_random_match = re.fullmatch(
        r"RMS(?P<rms>\d+(?:\.\d+)?)(?:mm)?_(?P<freq>\d+(?:\.\d+)?)Hz_(?P<current>\d+(?:\.\d+)?)A",
        folder_name,
    )
    if steady_random_match:
        d = steady_random_match.groupdict()
        return "SR", f"RMS{pnum(d['rms'])}_F{pnum(d['freq'])}_I{pnum(d['current'])}", ""

    random_hf_match = re.fullmatch(
        r"aRMS(?P<rms>\d+(?:\.\d+)?)mm_(?P<freq>\d+(?:\.\d+)?)Hz(?P<suffix>slow|11)?",
        folder_name,
    )
    if random_hf_match:
        d = random_hf_match.groupdict()
        suffix = d.get("suffix") or ""
        return "RHF", f"RMS{pnum(d['rms'])}_F{pnum(d['freq'])}{suffix}", ""

    random_lf_match = re.fullmatch(
        r"RMS(?P<rms>\d+(?:\.\d+)?)mm_(?P<freq>\d+(?:\.\d+)?)Hz(?P<suffix>slow|11)?",
        folder_name,
    )
    if random_lf_match:
        d = random_lf_match.groupdict()
        suffix = d.get("suffix") or ""
        return "RLF", f"RMS{pnum(d['rms'])}_F{pnum(d['freq'])}{suffix}", ""

    return "Unknown", re.sub(r"[^A-Za-z0-9_]+", "_", folder_name).strip("_"), ""


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_short_names(rows):
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition_key"]].append(row)
    for items in by_condition.values():
        use_repeat_suffix = len(items) > 1
        for idx, row in enumerate(items, 1):
            row["repeat_id"] = idx
            suffix = f"_r{idx}" if use_repeat_suffix else ""
            row["short_name"] = f"{display_name_from_condition(row['condition_key'])}{suffix}.csv"


def dedupe_exact_duplicates(rows):
    seen = {}
    kept = []
    removed = []
    for row in rows:
        key = (row["condition_key"], row["source_sha256"])
        if key in seen:
            row["duplicate_of"] = seen[key]
            removed.append(row)
            continue
        seen[key] = row
        kept.append(row)
    assign_short_names(kept)
    return kept, removed


def collect_files(base_dir, dedupe=True):
    rows = []
    for root in discover_source_roots(base_dir):
        source_root = normalize_source_root(root.name)
        for csv_path in sorted(root.rglob("auto_1.csv"), key=lambda p: str(p).lower()):
            folder = csv_path.parent.name
            cat, setting, folder_repeat = classify_and_setting(folder)
            condition_key = f"{cat}_{setting}"
            rows.append(
                {
                    "source_path": csv_path,
                    "source_root": source_root,
                    "source_folder": folder,
                    "source_sha256": sha256_file(csv_path),
                    "category_code": cat,
                    "category": CATEGORY_DIRS.get(cat, "Unknown"),
                    "setting": setting,
                    "condition_key": condition_key,
                    "folder_repeat": folder_repeat,
                    "source_order": SOURCE_PRIORITY.get(source_root, 999),
                }
            )
    rows.sort(
        key=lambda r: (
            r["condition_key"],
            r["source_order"],
            r["source_root"],
            r["source_folder"],
            str(r["source_path"]).lower(),
        )
    )
    if not dedupe:
        assign_short_names(rows)
        return rows, []
    return dedupe_exact_duplicates(rows)


def build_manifest_row(row, output_path, include_hash):
    src = row["source_path"]
    stat = src.stat()
    folder = row["source_folder"]
    source_root = row["source_root"]
    is_step430 = source_root == "step430"
    is_legacy_1p4a = is_step430 and "1.4A" in folder
    is_reversed_current_required = is_step430 and not is_legacy_1p4a
    is_manual_zero_required = folder == "AMP30mm_Vel0.13_step5_0_1.6A"
    return {
        "source_path": str(src),
        "output_path": str(output_path),
        "source_root": source_root,
        "source_folder": folder,
        "category": row["category"],
        "short_name": row["short_name"],
        "condition_key": row["condition_key"],
        "repeat_id": row["repeat_id"],
        "size_bytes": stat.st_size,
        "last_write_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "sha256": row.get("source_sha256") or (sha256_file(src) if include_hash else ""),
        "is_legacy_copy": source_root == "TestdataLegacyCopy",
        "is_step430": is_step430,
        "is_reversed_current_required": is_reversed_current_required,
        "is_manual_zero_required": is_manual_zero_required,
        "is_legacy_1p4A_step": is_legacy_1p4a,
    }


def write_readme(output_dir):
    text = """# Raw_Card_Categorized_All

This directory is a categorized archive of raw acquisition-card `auto_1.csv` files. Files are copied byte-for-byte from the source experiment folders after removing exact duplicate copies within the same normalized condition. They are not filtered, zeroed, converted to physical quantities, or otherwise processed.

## Categories

- `1_SH`: steady harmonic displacement with fixed current
- `2_SR`: steady random displacement with fixed current
- `3_RHF`: random high-frequency current plus random displacement
- `4_RLF`: random low-frequency current plus random displacement
- `5_ST`: standard velocity or step-current cases

## File Name Format

`{CAT}_{readable_setting}.csv` when only one file remains for a condition.
`{CAT}_{readable_setting}_r{N}.csv` when multiple distinct files remain for the same condition.

Examples:

- `SH_10mm_1.66Hz_0.8A_r1.csv`
- `SR_RMS10mm_4.97Hz_0.2A.csv`
- `RLF_RMS15mm_3Hzslow.csv`
- `ST_30mm_Vel0.13.csv`
- `ST_50mm_Vel0.13_step5_0_1.6A_r3.csv`

## Abbreviations

- `SH`: steady harmonic displacement with fixed current
- `SR`: steady random displacement with fixed current
- `RHF`: random high-frequency current plus random displacement
- `RLF`: random low-frequency current plus random displacement
- `ST`: standard velocity or step-current case
- `10mm`: displacement amplitude 10 mm
- `RMS15mm`: displacement RMS 15 mm
- `1.66Hz`: frequency 1.66 Hz
- `3Hzslow`: 3 Hz slow condition
- `0.8A`: current 0.8 A
- `1.6A`: current or peak current 1.6 A
- `Vel0.13`: velocity setting 0.13
- `step5_0`: step5_0 step-current program
- `r1`, `r2`, ...: repeat number used only when multiple distinct files remain within the same normalized condition key after exact-duplicate removal

The source root is intentionally not included in file names. Use `manifest.csv` to trace each archived file back to its original folder and full source path.

Exact duplicates are detected by `(condition_key, sha256)`. When multiple source files have the same normalized condition and identical content, only the first source in stable source order is copied, then the remaining files are renumbered. Omitted duplicate source files are recorded in `duplicates_removed.csv`.

## Manifest

`manifest.csv` records:

- original source path and output path
- source root and original source folder
- category and short file name
- normalized condition key and repeat number
- file size, last write time, and SHA256
- special flags for legacy copies, `step430`, reversed-current correction requirements, manual-zero cases, and legacy `1.4A` step files

## Important Notes

- `step430` files collected on 2026-06-11 have reversed current wiring. The raw files here are not changed; the manifest flag `is_reversed_current_required` tells later processing to multiply current by `-1`.
- `AMP30mm_Vel0.13_step5_0_1.6A` requires manual zero points during physical processing. The raw file here is not changed.
- This archive is for raw data organization only. Use the CDC processing skill before converting these files into physical model-ready datasets.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def summarize(rows):
    category_counts = Counter(row["category"] for row in rows)
    unknown_count = sum(1 for row in rows if row["category_code"] == "Unknown")
    filename_counts = Counter((row["category"], row["short_name"]) for row in rows)
    collisions = [name for name, count in filename_counts.items() if count > 1]
    return category_counts, unknown_count, collisions


def write_duplicates_removed(output_dir, removed_rows, output_paths):
    duplicate_path = output_dir / "duplicates_removed.csv"
    fieldnames = [
        "skipped_source_path",
        "kept_source_path",
        "kept_output_path",
        "source_root",
        "source_folder",
        "condition_key",
        "sha256",
        "reason",
    ]
    with duplicate_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in removed_rows:
            kept = row["duplicate_of"]
            writer.writerow(
                {
                    "skipped_source_path": str(row["source_path"]),
                    "kept_source_path": str(kept["source_path"]),
                    "kept_output_path": str(output_paths[id(kept)]),
                    "source_root": row["source_root"],
                    "source_folder": row["source_folder"],
                    "condition_key": row["condition_key"],
                    "sha256": row["source_sha256"],
                    "reason": "same condition_key and sha256",
                }
            )


def write_multi_condition_index(output_dir, rows, output_paths):
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition_key"]].append(row)
    multi_rows = []
    for condition_key, items in sorted(by_condition.items()):
        if len(items) <= 1:
            continue
        for row in items:
            multi_rows.append(
                {
                    "condition_key": condition_key,
                    "retained_count": len(items),
                    "source_path": str(row["source_path"]),
                    "output_path": str(output_paths[id(row)]),
                    "source_root": row["source_root"],
                    "source_folder": row["source_folder"],
                    "sha256": row["source_sha256"],
                }
            )
    index_path = output_dir / "multi_condition_index.csv"
    fieldnames = [
        "condition_key",
        "retained_count",
        "source_path",
        "output_path",
        "source_root",
        "source_folder",
        "sha256",
    ]
    with index_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(multi_rows)


def export(rows, removed_rows, discovered_count, output_dir, dry_run=False, clean_output=False):
    category_counts, unknown_count, collisions = summarize(rows)
    print(f"Discovered auto_1.csv files: {discovered_count}")
    print(f"Exact duplicates removed: {len(removed_rows)}")
    print(f"Archived auto_1.csv files: {len(rows)}")
    print("By category:")
    for category in sorted(category_counts):
        print(f"  {category}: {category_counts[category]}")
    print(f"Unknown: {unknown_count}")
    print(f"Filename collisions: {len(collisions)}")
    if collisions:
        for category, short_name in collisions[:20]:
            print(f"  {category}\\{short_name}")
    if dry_run:
        print("Dry-run only; no files written.")
        return
    if unknown_count:
        raise RuntimeError("Refusing to export while Unknown classifications exist.")
    if collisions:
        raise RuntimeError("Refusing to export because output filename collisions exist.")
    if output_dir.exists():
        if not clean_output:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --clean-output to rebuild.")
        shutil.rmtree(output_dir)
    for category in CATEGORY_DIRS.values():
        (output_dir / category).mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    output_paths = {}
    for row in rows:
        output_path = output_dir / row["category"] / row["short_name"]
        shutil.copy2(row["source_path"], output_path)
        output_paths[id(row)] = output_path
        manifest_rows.append(build_manifest_row(row, output_path, include_hash=True))
    write_readme(output_dir)
    write_duplicates_removed(output_dir, removed_rows, output_paths)
    write_multi_condition_index(output_dir, rows, output_paths)
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Exported: {len(manifest_rows)} files")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Archive raw CDC acquisition-card auto_1.csv files by condition class.")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true", help="Copy exact duplicate raw files instead of deduping by condition_key and SHA256.")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / DEFAULT_OUTPUT_NAME
    rows, removed_rows = collect_files(base_dir, dedupe=not args.keep_duplicates)
    export(rows, removed_rows, len(rows) + len(removed_rows), output_dir, dry_run=args.dry_run, clean_output=args.clean_output)


if __name__ == "__main__":
    main()
