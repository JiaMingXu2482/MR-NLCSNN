"""Read-only metrics provenance checker. No training or checkpoint loading."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def summarize(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    evaluated = [r for r in rows if math.isfinite(float(r['pooled_nrmse_pct']))]
    if not evaluated:
        raise ValueError('no finite evaluated rows')
    best = min(evaluated, key=lambda r: float(r['pooled_nrmse_pct']))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                row_count=len(rows), evaluated_rows=len(evaluated), best_row=best)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v18', type=Path, required=True)
    p.add_argument('--v17', type=Path, required=True)
    a = p.parse_args()
    old, new = summarize(a.v17), summarize(a.v18)
    v17, v18 = (float(x['best_row']['pooled_nrmse_pct']) for x in (old, new))
    print(json.dumps(dict(v17=old, v18=new, ratio=v18/v17 if v17 else None,
                         degradation_pp=v18-v17,
                         warning='Historical inverse evaluation may use unmasked I. Not deployment accuracy; verify split/norm/checkpoint provenance.'),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
