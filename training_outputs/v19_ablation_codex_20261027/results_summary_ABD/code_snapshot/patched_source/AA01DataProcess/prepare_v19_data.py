"""Raw-column alignment BEFORE filtering/segmentation/zeroing; never overwrite.

Positive shift: aligned F[k]=raw F[k+shift], channel presumed to lag x.
No circular roll/interpolation/padding/per-file fitted delay. No training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_raw_card_archive import read_raw_csv, process_file, CANDIDATES, CATEGORY_MAP


def align_raw(raw, force_shift, current_shift):
    shifts = {5: int(force_shift), 6: int(current_shift)}
    begin = max(0, -min(shifts.values()))
    end = min(len(raw), len(raw)-max(shifts.values()))
    if end-begin < 1000:
        raise ValueError('fewer than 1000 common samples after alignment')
    aligned = raw.iloc[begin:end].copy().reset_index(drop=True)
    for column, shift in shifts.items():
        aligned.iloc[:, column] = raw.iloc[begin+shift:end+shift, column].to_numpy()
    return aligned, dict(common_start=begin, common_stop_exclusive=end,
                         source_samples=len(raw), force_shift_samples=force_shift,
                         current_shift_samples=current_shift, dt_s=0.001)


def assign_splits(rows, spec):
    """Group repeats and identical raw hashes transitively, then deduplicate."""
    names = {Path(r['short_name']).stem for r in rows}
    missing = (set(spec['validation_stems']) | set(spec['final_test_stems']))-names
    if missing:
        raise ValueError('missing declared split files: '+str(sorted(missing)))
    parent = list(range(len(rows)))
    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k
    seen = {}
    for k, row in enumerate(rows):
        stem = Path(row['short_name']).stem
        for key in ('name:'+re.sub(r'_r\d+$', '', stem), 'sha:'+row['sha256']):
            if key in seen:
                parent[root(k)] = root(seen[key])
            else:
                seen[key] = k
    labels = {}
    for k, row in enumerate(rows):
        stem = Path(row['short_name']).stem
        label = ('final_test' if stem in spec['final_test_stems'] else
                 'validation' if stem in spec['validation_stems'] else 'train')
        labels.setdefault(root(k), set()).add(label)
    used_hash = set()
    for k, row in enumerate(rows):
        group = labels[root(k)]
        if {'validation', 'final_test'} <= group:
            raise ValueError('validation/test share a condition or raw hash')
        label = 'final_test' if 'final_test' in group else 'validation' if 'validation' in group else 'train'
        if Path(row['short_name']).stem in spec['exclude_stems']:
            label = 'excluded'
        elif row['sha256'] in used_hash:
            label = 'duplicate'
        else:
            used_hash.add(row['sha256'])
        row['split'] = label
        row['condition_group'] = re.sub(r'_r\d+$', '', Path(row['short_name']).stem)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--force-shift', type=int, required=True)
    parser.add_argument('--current-shift', type=int, required=True)
    parser.add_argument('--split', type=Path, default=Path(__file__).resolve().parents[1]/'AA02Train/v19_split.json')
    args = parser.parse_args()
    archive, output = args.archive.resolve(), args.output.resolve()
    if output.exists() or output == archive or archive in output.parents:
        raise ValueError('output must be a new directory outside the immutable archive')
    manifest = pd.read_csv(archive/'manifest.csv', encoding='utf-8-sig').to_dict('records')
    spec = json.loads(args.split.read_text(encoding='utf-8'))
    for row in manifest:
        if row['category'] not in CATEGORY_MAP or Path(row['short_name']).name != row['short_name']:
            raise ValueError('invalid archive category or basename')
        source = (archive/row['category']/row['short_name']).resolve()
        if archive not in source.parents:
            raise ValueError('source escapes archive')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != row.get('sha256'):
            raise ValueError('source hash mismatch: '+str(source))
        row.update(output_path=str(source), sha256=digest)
    assign_splits(manifest, spec)
    output.mkdir(parents=True)
    (output/'split_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    records = []
    for row in manifest:
        if row['split'] in ('excluded', 'duplicate'):
            records.append(dict(short_name=row['short_name'], split=row['split'],
                raw_sha256=row['sha256'], condition_group=row['condition_group'],
                relative_path=None, reason='Excluded by locked split; immutable raw file retained.'))
            continue
        raw = read_raw_csv(row['output_path'])
        aligned, mapping = align_raw(raw, args.force_shift, args.current_shift)
        df, meta = process_file(row, CANDIDATES['force_wide'], raw_override=aligned)
        _, before = process_file(row, CANDIDATES['force_wide'], raw_override=raw)
        if df is None:
            raise ValueError(f"{row['short_name']}: {meta}")
        start, stop = meta['crop_start_raw_idx'], meta['crop_end_raw_idx']
        source_idx = np.arange(start, stop)+mapping['common_start']
        df['source_x_index'] = source_idx
        df['source_F_index'] = source_idx+args.force_shift
        df['source_I_index'] = source_idx+args.current_shift
        # The legacy parser sees aligned data: its indices are NOT original indices.
        meta = {k.replace('_raw_idx', '_aligned_idx'): v for k, v in meta.items()}
        meta.update(mapping, raw_sha256=row['sha256'], split=row['split'],
                    condition_group=row['condition_group'],
                    zero_before=before.get('force_offset'), zero_after=meta['force_offset'],
                    zero_change_N=(meta['force_offset']-before['force_offset']) if 'force_offset' in before else None,
                    baseline_before=before, source_x_start=int(source_idx[0]),
                    source_x_stop_exclusive=int(source_idx[-1]+1),
                    raw_time_note='source indices are data rows after #EndHeader, not CSV line numbers',
                    edge_guard_samples=1000)
        destination = output/CATEGORY_MAP[row['category']]/row['short_name']
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            raise ValueError('duplicate output filename: '+str(destination))
        df.to_csv(destination, index=False)
        meta['relative_path'] = destination.relative_to(output).as_posix()
        meta['processed_sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
        records.append(meta)
    (output/'alignment_manifest.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    # Written last; absence means partial export, not usable training data.
    (output/'COMPLETE.json').write_text(json.dumps(dict(files=len(records),
        manifest_sha256=hashlib.sha256((output/'alignment_manifest.json').read_bytes()).hexdigest(),
        split_sha256=hashlib.sha256(args.split.read_bytes()).hexdigest())), encoding='utf-8')


if __name__ == '__main__':
    main()
