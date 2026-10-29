"""Read only non-test ST CSVs and saved predictions; no fitting or inference.

10% of current STEP change -> 90% of force STEP change, signed endpoint
interpolation; not 10/90% of absolute amplitudes. Current endpoint is an
observed plateau proxy because command current is unavailable.
"""
from pathlib import Path
import csv, hashlib, json, re
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = Path(r'D:\v20_data\Parsed_v19_F42_I42')
PRED = Path(r'D:\OneDrive - mail.scut.edu.cn\Thesis\MR_NLCSNN\v20_training\v20\analysis\results\st_delay_preds.npz')
REJECTED=[]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def edges(i):
    hit = np.flatnonzero(np.abs(np.diff(i)) > .04)
    if not len(hit): return []
    groups = np.split(hit, np.flatnonzero(np.diff(hit) > 40)+1)
    return [(int(g[0]), int(g[-1])+1) for g in groups]

def crossing(sig, level, direction, lo, hi):
    """First crossing in either signed direction, linearly interpolated."""
    q = direction*(sig-level)
    for k in range(max(1,lo), min(hi,len(sig))):
        if q[k-1] < 0 <= q[k]:
            return k-1+float(-q[k-1]/(q[k]-q[k-1]))
    return None

def stats(rows):
    x = np.array([r['delay_ms'] for r in rows if r['delay_ms'] is not None])
    good = [r for r in rows if not r['quality_flags'] and r['delay_ms'] is not None]
    return dict(n_events=len(rows), n_crossed=len(x), clean_n=len(good),
                median_ms=float(np.median(x)) if len(x) else None,
                mean_ms=float(np.mean(x)) if len(x) else None,
                p10_ms=float(np.percentile(x,10)) if len(x) else None,
                p90_ms=float(np.percentile(x,90)) if len(x) else None,
                negative_n=int(np.sum(x<0)),
                flags=dict(Counter(f for r in rows for f in r['quality_flags'])))

def calculate(name, split, t, cur, vel, force, label, windows, valid=None):
    out=[]
    es=edges(cur)
    post0,post1=windows
    for j,(a,b) in enumerate(es):
        def reject(reason):
            REJECTED.append(dict(file=name,curve=label,event=j,post_window_ms=list(windows),reason=reason))
        if a-40<0 or b+post1>=len(t): reject('insufficient_endpoint_window');continue
        if valid is not None and not (valid[a-30:b+post1].all()): reject('outside_prediction_valid_guard');continue
        if j+1<len(es) and es[j+1][0]<=b+post1: reject('next_current_edge_in_endpoint_window');continue
        pre=slice(a-30,a-10);post=slice(b+post0,b+post1)
        i0,i1=float(np.median(cur[pre])),float(np.median(cur[post]))
        if abs(i1-i0)<.3: reject('current_step_below_0.3A');continue
        f0,f1=float(np.median(force[pre])),float(np.median(force[post]))
        if abs(f1-f0)<30: reject('force_step_below_30N');continue
        lo=a-40; hi=min(b+120,es[j+1][0] if j+1<len(es) else len(t))
        qi=i0+.1*(i1-i0);qf=f0+.9*(f1-f0)
        ki=crossing(cur,qi,np.sign(i1-i0),lo,hi)
        kf=crossing(force,qf,np.sign(f1-f0),lo,hi)
        flags=[]
        v0,v1=float(np.median(vel[pre])),float(np.median(vel[post]))
        vv=vel[a-30:b+post1]
        if np.min(vv)<-10 and np.max(vv)>10: flags.append('velocity_reversal')
        if abs(v1-v0) > max(10,.1*max(abs(v0),abs(v1))): flags.append('velocity_change')
        if min(abs(v0),abs(v1))<10: flags.append('low_speed')
        # Endpoint quality is checked, not forced by normalising each segment.
        for what,s,delta,floor in [('I',cur,i1-i0,.02),('F',force,f1-f0,10)]:
            for side,sl in [('pre',pre),('post',post)]:
                p=s[sl];half=len(p)//2
                drift=abs(float(np.median(p[half:])-np.median(p[:half])))
                spread=float(np.percentile(p,90)-np.percentile(p,10))
                if drift>max(floor,.1*abs(delta)) or spread>max(2*floor,.2*abs(delta)):
                    flags.append(what+'_'+side+'_not_flat')
        if ki is None: flags.append('I10_not_crossed')
        if kf is None: flags.append('F90_not_crossed')
        delay=None if ki is None or kf is None else float((kf-ki)*1000*np.median(np.diff(t)))
        if delay is not None and delay<0: flags.append('force_precedes_I10')
        def ts(k): return None if k is None else float(np.interp(k,np.arange(len(t)),t))
        out.append(dict(file=name,split=split,curve=label,event=j,edge_s=float(t[a]),
            direction='up' if i1>i0 else 'down',I0_A=i0,I1_A=i1,I10_A=qi,F0_N=f0,F1_N=f1,
            F90_N=qf,t_I10_s=ts(ki),t_F90_s=ts(kf),delay_ms=delay,
            v0_mm_s=v0,v1_mm_s=v1,post_window_ms=list(windows),quality_flags=flags))
    return out

def run():
    manifest=json.loads((DATA/'alignment_manifest.json').read_text(encoding='utf-8'))
    meta=json.loads((DATA/'COMPLETE.json').read_text(encoding='utf-8'))
    assert sha(DATA/'alignment_manifest.json')==meta['manifest_sha256']
    selected=[r for r in manifest if r['split'] in ('train','validation') and Path(r['relative_path']).name.startswith('ST_')]
    z=np.load(PRED,allow_pickle=False)
    rows=[];sensitivity=[];inputs=[];curves={}
    for r in selected:
        p=DATA/r['relative_path'];assert sha(p)==r['processed_sha256']
        tab=np.genfromtxt(p,delimiter=',',names=True,encoding='utf-8-sig')
        name=p.stem;t=tab['time'];cur=np.abs(tab['current']);v=tab['velocity'];f=tab['force']
        assert np.allclose(np.diff(t),.001,atol=1e-7)
        inputs.append(dict(path=str(p),sha256=sha(p),split=r['split'],current_shift=r['current_shift_samples'],force_shift=r['force_shift_samples']))
        curves[name]=(t,cur,v,{'measured':f})
        rows+=calculate(name,r['split'],t,cur,v,f,'measured',(30,80))
        for win in [(20,50),(50,100)]:
            sensitivity+=calculate(name,r['split'],t,cur,v,f,'measured',win)
        for label in ['A','E']:
            pp=HERE/'predictions'/(name+'__'+label+'.f32')
            if pp.exists():
                prediction=np.fromfile(pp,dtype='<f4');assert len(prediction)==len(t)
                rows+=calculate(name,r['split'],t,cur,v,prediction,label+'_all_ST',(30,80))
        if name+'|truth' in z:
            assert np.max(np.abs(f-z[name+'|truth']))<.002
            assert np.max(np.abs(cur-np.abs(z[name+'|I'])))<1e-5
            for label in ['truth','A','E']:
                lab='measured_valid' if label=='truth' else label
                rows+=calculate(name,'validation',t,cur,v,z[name+'|'+label],lab,(30,80),z[name+'|valid'])
                curves[name][3][lab]=z[name+'|'+label]
                for win in [(20,50),(50,100)]:
                    sensitivity+=calculate(name,'validation',t,cur,v,z[name+'|'+label],lab,win,z[name+'|valid'])
    groups=defaultdict(list)
    for r in rows:
        condition=re.sub(r'_r\d+$','',r['file'])
        groups[(condition,r['curve'],r['direction'])].append(r)
    grouped=[dict(condition=k[0],curve=k[1],direction=k[2],**stats(v)) for k,v in sorted(groups.items())]
    overall={lab:{direction:stats([r for r in rows if r['curve']==lab and r['direction']==direction]) for direction in ['up','down']} for lab in ['measured','measured_valid','A','E']}
    # Paired comparison uses the identical file/event IDs with all three crossings.
    idx={(r['file'],r['event'],r['curve']):r for r in rows if r['curve'] in ('measured_valid','A','E')}
    pairs=[]
    for file,j,lab in idx:
        if lab!='measured_valid': continue
        trio=[idx.get((file,j,l)) for l in ['measured_valid','A','E']]
        if all(r is not None and r['delay_ms'] is not None for r in trio):
            pairs.append(dict(file=file,event=j,direction=trio[0]['direction'],
                measured_ms=trio[0]['delay_ms'],A_ms=trio[1]['delay_ms'],E_ms=trio[2]['delay_ms'],
                any_flags=sorted(set(f for r in trio for f in r['quality_flags']))))
    paired={d:{lab:stats([{'delay_ms':r[lab+'_ms'],'quality_flags':r['any_flags']} for r in pairs if r['direction']==d]) for lab in ['measured','A','E']} for d in ['up','down']}
    payload=dict(definition='I0+0.10*(I1-I0) -> F0+0.90*(F1-F0)',
        command_current_recorded=False, current_used='abs(measured current), no clipping',
        plateau_proxy='median; before [edge-30,edge-10) ms, after [last steep edge+30,+80) ms',
        threshold_crossing='first directional crossing, linear interpolation, no signal smoothing',
        endpoint_sensitivity_windows_ms=[[20,50],[30,80],[50,100]],
        limitations='Apparent joint motion/current response, not isolated actuator delay when velocity changes; first crossing is not settling time.',
        source_manifest_sha256=sha(DATA/'alignment_manifest.json'),prediction_sha256=sha(PRED),
        files=len(inputs),overall=overall,paired=paired,by_condition=grouped,events=rows,sensitivity_events=sensitivity,paired_events=pairs,inputs=inputs,
        rejected_candidates=REJECTED)
    (HERE/'delay_10_90.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,table in [('events',rows),('by_condition',grouped),('paired_events',pairs)]:
        with (HERE/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as fh:
            wr=csv.DictWriter(fh,fieldnames=list(table[0]));wr.writeheader()
            for r in table:wr.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    plots(curves,rows)
    print(json.dumps(dict(files=len(inputs),overall=overall,paired=paired),ensure_ascii=False,indent=2))

def plots(curves,rows):
    selected=[r for r in rows if r['curve']=='measured_valid']
    examples=[next(r for r in selected if r['direction']==d and r['delay_ms'] is not None) for d in ['up','down']]
    fig,axs=plt.subplots(3,2,figsize=(12,8),sharex='col')
    for col,r in enumerate(examples):
        t,i,v,fs=curves[r['file']];t0=r['edge_s'];mask=(t>t0-.05)&(t<t0+.15);x=1000*(t-t0)
        axs[0,col].plot(x[mask],i[mask],color='purple');axs[0,col].axhline(r['I10_A'],color='gray',ls='--')
        axs[1,col].plot(x[mask],v[mask],color='gray');axs[1,col].axhline(0,color='gray',lw=.5)
        for lab,color in [('measured_valid','black'),('A','#d95f02'),('E','#1b9e77')]:
            rr=next(a for a in rows if a['file']==r['file'] and a['event']==r['event'] and a['curve']==lab)
            axs[2,col].plot(x[mask],fs[lab][mask],label=lab,color=color)
            axs[2,col].axhline(rr['F90_N'],color=color,ls=':',lw=.8)
            if rr['t_F90_s'] is not None:
                axs[2,col].scatter((rr['t_F90_s']-t0)*1000,rr['F90_N'],color=color,s=28)
        for ax in axs[:,col]:
            ax.axvline((r['t_I10_s']-t0)*1000,color='purple',ls='--');ax.grid(alpha=.2)
        axs[0,col].set_title(r['direction']+' step | '+r['file'].split('_')[-1])
        axs[2,col].set_xlabel('Time relative to detected edge (ms)');axs[2,col].legend(fontsize=8)
    for ax,lab in zip(axs[:,0],['Current (A)','Velocity (mm/s)','Force (N)']): ax.set_ylabel(lab)
    fig.suptitle('I10 -> F90: velocity reverses at the current step\nFirst threshold crossing; plateau estimate is not an independent command')
    fig.tight_layout();fig.savefig(HERE/'threshold_examples.png',dpi=180);plt.close(fig)

if __name__=='__main__': run()
