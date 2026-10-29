"""Supplement saved P0 statistics only; no CSV reads or reference changes."""
import collections,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
P=HERE/'p0_reference_audit_20261028'
OUT=HERE/'claude_execution_20261029'
OUT.mkdir(exist_ok=True)
rows=json.loads((P/'per_file.json').read_text())
bands=json.loads((P/'low_speed_pairs.json').read_text())
summary=json.loads((P/'summary.json').read_text())
ends=[r for r in rows if r['end']['matched'] and r['end']['last_motion_sign']]
groups=sorted(set(r['group'] for r in ends))
by_group={g:[r for r in ends if r['group']==g] for g in groups}
def diff(gs):
    signed={s:[] for s in [-1,1]}
    for g in gs:
        for s in signed:
            vals=[r['end']['delta_F_N'] for r in by_group[g] if r['end']['last_motion_sign']==s]
            if vals: signed[s].append(np.mean(vals))
    if not all(signed.values()): return None
    return float(np.mean(signed[1])-np.mean(signed[-1]))
rng=np.random.default_rng(20261028)
boot=[diff(rng.choice(groups,len(groups),replace=True)) for _ in range(2000)]
boot=[b for b in boot if b is not None]
matched_bins=collections.defaultdict(list)
for b in bands: matched_bins[(b['x_bin_mm'],b['I_bin_A'],b['T_bin_C'])].append(b)
repeats=[]
for key,bb in matched_bins.items():
    if len(set(b['name'] for b in bb))>1:
        repeats.append(dict(bin=key,files=[b['name'] for b in bb],
                            midpoint_range_N=float(np.ptp([b['midpoint_N'] for b in bb]))))
order={}
for cat in ['SH','ST','SR','RLF','RHF']:
    rr=sorted([r for r in rows if r['category']==cat],key=lambda r:r['acquisition_time'])
    delta=np.diff([r['F0_N'] for r in rr])
    order[cat]=dict(files=len(rr),first=rr[0]['acquisition_time'],last=rr[-1]['acquisition_time'],
                    adjacent_abs_delta_median_N=float(np.median(abs(delta))),
                    adjacent_abs_delta_max_N=float(max(abs(delta))),
                    note='Changing test conditions; not a pure sensor drift experiment')
result=dict(end_direction_cluster_bootstrap=dict(groups=len(groups),difference_N=diff(groups),
    ci95=np.quantile(boot,[.025,.975]).tolist(),valid=len(boot),
    correction='Resample whole groups jointly; preserves groups with both movement directions'),
    repeated_low_speed_bins=repeats,low_speed_pair_files=len(set(b['name'] for b in bands)),
    low_speed_speed_imbalance_median_mm_s=float(np.median([abs(b['positive_v_mm_s']+b['negative_v_mm_s']) for b in bands])),
    acquisition_order=order,reference_gate='inconclusive; no relabeling or P2 training',
    limitation='Finite-speed pairs are not v->0 extrapolations. Midpoints include viscous asymmetry, hysteresis and measurement offsets; cannot identify gas spring separately from this audit.')
(OUT/'p0_supplement.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(result,indent=2,ensure_ascii=False))
