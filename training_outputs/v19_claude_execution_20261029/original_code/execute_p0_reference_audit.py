"""P0: raw TRAIN-only reference audit; no label changes and no training."""
import csv,json,re,hashlib,sys,collections
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
DATA=HERE/'Parsed_v19_F42_I42_initial_static_20261028'
OUT=HERE/'p0_reference_audit_20261028'
sys.path.insert(0,r'C:\Users\user\AppData\Local\Temp\v19_bd_runtime_20261027\source\AA01DataProcess')
from process_raw_card_archive import read_raw_csv,sos_lowpass_zero_phase
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def stats(x):
    x=np.asarray(x); return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),min=float(x.min()),max=float(x.max()),std=float(x.std())) if len(x) else dict(n=0)
def regression(rows,adjusted):
    rows=[r for r in rows if r['motion_mean_raw_F_N'] is not None]
    if len(rows)<5: return dict(n=len(rows),status='insufficient')
    X=np.column_stack([np.ones(len(rows)),[r['F0_N'] for r in rows]])
    cols=['intercept','F0_N']
    if adjusted:
        keys=['motion_temperature_C','motion_x_abs_mm','motion_current_A','motion_v_rms_mm_s','motion_v_mean_mm_s']
        Z=np.array([[r[k] for k in keys] for r in rows]); Z=(Z-Z.mean(0))/np.maximum(Z.std(0),1e-8)
        cats=sorted(set(r['category'] for r in rows))
        X=np.column_stack([X,Z]+[np.array([r['category']==c for r in rows],float) for c in cats[1:]])
        cols+=keys+['category_'+c for c in cats[1:]]
    # Center/scale F0 for condition diagnostics and convert coefficient back.
    fstd=X[:,1].std(); X[:,1]=(X[:,1]-X[:,1].mean())/fstd
    y=np.array([r['motion_mean_raw_F_N'] for r in rows]); co=np.linalg.lstsq(X,y,rcond=None)[0]
    groups=sorted(set(r['group'] for r in rows)); ids={g:[j for j,r in enumerate(rows) if r['group']==g] for g in groups}
    rng=np.random.default_rng(20261028); slopes=[]
    for _ in range(2000):
        ix=np.concatenate([ids[g] for g in rng.choice(groups,len(groups),replace=True)])
        if np.linalg.matrix_rank(X[ix])<X.shape[1]: continue
        slopes.append(float(np.linalg.lstsq(X[ix],y[ix],rcond=None)[0][1]/fstd))
    ci=np.quantile(slopes,[.025,.975]).tolist() if len(slopes)>=100 else None
    return dict(n=len(rows),groups=len(groups),columns=cols,rank=int(np.linalg.matrix_rank(X)),parameters=X.shape[1],
        design_condition=float(np.linalg.cond(X)),slope=float(co[1]/fstd),ci95=ci,bootstrap_valid=len(slopes),
        motion_mean_residual_rmse_N=float(np.sqrt(np.mean((y-X@co)**2))))
def main():
    OUT.mkdir(exist_ok=False)
    plan=HERE/'p0_reference_plan_20261028.md'; (OUT/plan.name).write_bytes(plan.read_bytes())
    manifest=DATA/'alignment_manifest.json'; assert sha(manifest)=='e3a8c984f6415beeabfab05b8178ec32b2172e2542570d41110dffeb1b1c1ced'
    records=[r for r in json.loads(manifest.read_text()) if r['split']=='train']; assert len(records)==248
    rows=[]; bands=[]
    for no,m in enumerate(records):
        p=Path(m['original_source_path']); assert sha(p)==m['raw_sha256']
        raw=read_raw_csv(p); name=Path(m['relative_path']).stem; cat=name.split('_')[0]
        header=p.open(encoding='gbk',errors='ignore'); start_time=None
        with header:
            for line in header:
                if '#EndHeader' in line: break
                q=re.search(r'20\d\d/\d\d/\d\d\s+\d\d:\d\d:\d\d(?:\.\d+)?',line)
                if q: start_time=q.group(0)
        aa=m['common_start']; bb=m['common_stop_exclusive']; shift=m['force_shift_samples']; ishift=m['current_shift_samples']
        x=sos_lowpass_zero_phase((raw.iloc[aa:bb,3].to_numpy()-4)/16*1500,50)
        f=sos_lowpass_zero_phase(raw.iloc[aa+shift:bb+shift,5].to_numpy()/10*1000*9.80665,40)
        i=np.abs(sos_lowpass_zero_phase(raw.iloc[aa+ishift:bb+ishift,6].to_numpy()/1000,40))
        temp=sos_lowpass_zero_phase(raw.iloc[aa:bb,8].to_numpy(),2); v=-np.gradient(x,.001)
        z0=m['baseline_start_aligned_idx']; z1=m['baseline_end_aligned_idx']; f0=float(f[z0:z1].mean())
        assert abs(f0-m['force_offset'])<1e-8
        n=m['crop_end_aligned_idx']-m['crop_start_aligned_idx']; guard=1000 if n>2000 else min(100,n//4)
        lo=m['crop_start_aligned_idx']+guard; hi=m['crop_end_aligned_idx']-guard
        xx=x[lo:hi]; ff=f[lo:hi]; ts=np.arange(lo,hi,dtype=float); cycles=None
        if cat=='SH':
            mid=(np.percentile(xx,5)+np.percentile(xx,95))/2
            hits=np.flatnonzero((xx[:-1]<mid)&(xx[1:]>=mid)); freq=float(re.search(r'_([\d.]+)Hz',name).group(1))
            good=[]
            for q in hits:
                t=lo+q+(mid-xx[q])/(xx[q+1]-xx[q])
                if not good or t-good[-1]>.6*1000/freq: good.append(t)
            cycles=len(good)-1
            if cycles>=2:
                lo2,hi2=good[0],good[-1]
                ts=np.r_[lo2,np.arange(np.floor(lo2)+1,np.ceil(hi2)),hi2]
            else: ts=np.array([])
        def avg(values):
            return float(np.trapz(np.interp(ts,np.arange(len(values)),values),ts)/(ts[-1]-ts[0])) if len(ts)>1 else None
        r=dict(name=name,category=cat,group=m.get('condition_group',re.sub(r'_r\d+$','',name)),raw_sha256=m['raw_sha256'],raw_path=str(p),
            acquisition_time=start_time,F0_N=f0,x0_abs_mm=float(x[z0:z1].mean()),I0_A=float(i[z0:z1].mean()),T0_C=float(temp[z0:z1].mean()),
            initial_force_std_N=float(f[z0:z1].std()),motion_mean_raw_F_N=avg(f),motion_x_abs_mm=avg(x),motion_current_A=avg(i),
            motion_temperature_C=avg(temp),motion_v_mean_mm_s=avg(v),motion_v_rms_mm_s=float(np.sqrt(avg(v*v))) if len(ts)>1 else None,
            complete_SH_cycles=cycles)
        # End-window test is independent of the processed crop and initial F0.
        ea=len(x)-550; eb=len(x)-50; ex=x[ea:eb]; ev=v[ea:eb]
        still=bool(np.ptp(ex)<=.15 and np.sqrt(np.mean(ev*ev))<=2)
        dx=float(ex.mean()-r['x0_abs_mm']); di=float(i[ea:eb].mean()-r['I0_A']); dtemp=float(temp[ea:eb].mean()-r['T0_C'])
        moving=np.flatnonzero(np.abs(v[:ea])>=5); direction=int(np.sign(v[moving[-1]])) if len(moving) else 0
        r['end']=dict(still=still,x_p2p_mm=float(np.ptp(ex)),v_rms_mm_s=float(np.sqrt(np.mean(ev*ev))),
            delta_x_mm=dx,delta_I_A=di,delta_T_C=dtemp,delta_F_N=float(f[ea:eb].mean()-f0),last_motion_sign=direction,
            matched=bool(still and abs(dx)<=.15 and abs(di)<=.05 and abs(dtemp)<=.5))
        rows.append(r)
        if cat in ('SH','ST'):
            ix=np.arange(lo,hi); ix=ix[(np.abs(v[ix])>=1)&(np.abs(v[ix])<=10)]
            buckets=collections.defaultdict(lambda:[[],[]])
            for q in ix: buckets[(int(np.floor(x[q])),int(np.floor(i[q]/.1)),int(np.floor(temp[q])))][int(v[q]>0)].append(int(q))
            for key,(neg,pos) in buckets.items():
                if len(neg)<20 or len(pos)<20: continue
                bands.append(dict(name=name,category=cat,group=r['group'],x_bin_mm=key[0],I_bin_A=key[1]*.1,T_bin_C=key[2],
                    samples_negative=len(neg),samples_positive=len(pos),negative_F_N=float(f[neg].mean()),positive_F_N=float(f[pos].mean()),
                    gap_N=float(f[pos].mean()-f[neg].mean()),midpoint_N=float((f[pos].mean()+f[neg].mean())/2),
                    negative_v_mm_s=float(v[neg].mean()),positive_v_mm_s=float(v[pos].mean())))
        if (no+1)%25==0: print(f'P0 raw train {no+1}/248',flush=True)
    write('per_file.json',rows); write('low_speed_pairs.json',bands)
    regress={c:regression([r for r in rows if r['category']==c],False) for c in ['SH','ST','SR','RLF','RHF']}
    randoms=[r for r in rows if r['category'] in ('SR','RLF','RHF')]
    regress['random_adjusted']=regression(randoms,True)
    ends=[r for r in rows if r['end']['matched'] and r['end']['last_motion_sign']]
    groups={g:[r for r in ends if r['group']==g] for g in set(r['group'] for r in ends)}
    signed={sg:[np.mean([r['end']['delta_F_N'] for r in rr if r['end']['last_motion_sign']==sg]) for rr in groups.values() if any(r['end']['last_motion_sign']==sg for r in rr)] for sg in [-1,1]}
    direction_test=dict(eligible_files=len(ends),eligible_groups=len(groups),negative_direction=stats(signed[-1]),positive_direction=stats(signed[1]))
    if min(len(signed[-1]),len(signed[1]))>=3:
        rng=np.random.default_rng(20261028)
        diffs=[np.mean(rng.choice(signed[1],len(signed[1]),replace=True))-np.mean(rng.choice(signed[-1],len(signed[-1]),replace=True)) for _ in range(2000)]
        direction_test.update(mean_difference_N=float(np.mean(signed[1])-np.mean(signed[-1])),ci95=np.quantile(diffs,[.025,.975]).tolist())
    adj=regress['random_adjusted']; ci=adj['ci95']; stable=adj['groups']>=20 and adj['rank']==adj['parameters'] and adj['design_condition']<1e4
    slope_support='near_zero' if stable and ci and ci[0]>=-.25 and ci[1]<=.25 else ('near_one' if stable and ci and ci[0]>=.75 and ci[1]<=1.25 else 'inconclusive')
    direction_pass=bool(len(groups)>=10 and 'ci95' in direction_test and abs(direction_test['mean_difference_N'])>10 and np.prod(direction_test['ci95'])>0)
    summary=dict(regression=regress,end_direction=direction_test,initial_offset_spread_by_category={c:stats([r['F0_N'] for r in rows if r['category']==c]) for c in ['SH','ST','SR','RLF','RHF']},
        low_speed_pairs=len(bands),paired_files=len(set(r['name'] for r in bands)),low_speed_gap_abs_N=stats([abs(r['gap_N']) for r in bands]),
        acquisition_timestamps=sum(r['acquisition_time'] is not None for r in rows),slope_gate=slope_support,end_direction_gate=direction_pass,
        reference_change_authorized_by_evidence=False,reason='R1 additionally requires identifiable repeatable low-speed midpoint; not automatically inferred from a correlation.',
        train_files=248,validation_or_final_csv_read=False,training_started=False,data_modified=False,manifest_sha256=sha(manifest),plan_sha256=sha(plan),script_sha256=sha(__file__))
    write('summary.json',summary)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for c in ['SR','RLF','RHF']:
        rr=[r for r in randoms if r['category']==c]; axs[0].scatter([r['F0_N'] for r in rr],[r['motion_mean_raw_F_N'] for r in rr],label=c)
        rr=sorted([r for r in rr if r['acquisition_time']],key=lambda r:r['acquisition_time'])
        axs[1].plot(pd.to_datetime([r['acquisition_time'] for r in rr]),[r['F0_N'] for r in rr],'.-',label=c)
    axs[0].set(xlabel='Initial raw static F0 (N)',ylabel='Motion raw mean force (N)',title='TRAIN only; without subtracting F0')
    axs[1].set(ylabel='Initial F0 (N)',title='Acquisition timestamps (conditions confounded)'); axs[1].tick_params(axis='x',rotation=25)
    for ax in axs: ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/'p0_reference_evidence.png',dpi=160); plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()
