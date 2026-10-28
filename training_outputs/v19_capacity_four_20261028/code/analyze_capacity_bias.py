"""Read-only residual/reference audit from frozen predictions and three raw files."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; OUT=HERE/'diagnosis_capacity_bias_20261028'
DATA=HERE/'Parsed_v19_F42_I42_initial_static_20261028'
TARGETS=['RLF_RMS10mm_2Hz_r1','RLF_RMS10mm_4Hz','SR_RMS10mm_1.66Hz_0A']
ARMS=['D','A','A_GRU','A_MLP']
sys.path.insert(0,r'C:\Users\user\AppData\Local\Temp\v19_bd_runtime_20261027\source\AA01DataProcess')
from process_raw_card_archive import read_raw_csv,sos_lowpass_zero_phase
def write(p,obj): p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    manifest={Path(r['relative_path']).stem:r for r in json.loads((DATA/'alignment_manifest.json').read_text()) if r['split'] in ('train','validation')}
    changes={r['name']:r for r in json.loads((DATA/'zero_reference_audit.json').read_text())['changes']}
    windows={r['name']:r for r in json.loads((HERE/'training_zero_audit_20261028/initial_windows.json').read_text())}
    rows=[]; rawrows=[]; fig,axes=plt.subplots(3,2,figsize=(13,10))
    for name,axs in zip(TARGETS,axes):
        for arm in ARMS:
            d=json.loads((OUT/f'{arm}_{name}.json').read_text()); g=d['guard']; sl=slice(g,-g)
            y=np.array(d['measured_force_N'])[sl]; p=np.array(d['predicted_force_N'])[sl]; e=p-y
            v=np.array(d['values'])[sl,1]; bias=e.mean(); rmse=np.sqrt(np.mean(e**2)); gain,intercept=np.polyfit(y,p,1)
            best=(rmse,0)
            for lag in range(-50,51):
                yy=y[max(0,lag):len(y)+min(0,lag)]; pp=p[max(0,-lag):len(p)-max(0,lag)]
                r=np.sqrt(np.mean((pp-yy)**2))
                if r<best[0]: best=(r,lag)
            row=dict(name=name,arm=arm,bias_N=float(bias),rmse_N=float(rmse),bias_sse_pct=float(100*bias*bias/(rmse*rmse)),
                demeaned_diagnostic_rmse_N=float(np.std(e)),prediction_vs_truth_gain=float(gain),intercept_N=float(intercept),
                optimal_lag_ms=best[1],lag_rmse_N=float(best[0]),quarter_bias_N=[float(x.mean()) for x in np.array_split(e,4)],
                near_zero_rmse_N=float(np.sqrt(np.mean(e[np.abs(v)<10]**2))),
                branch_means_N={k:float(np.array(d[k])[sl].mean()) for k in ('viscous_N','friction_N','b_N')})
            rows.append(row)
            t=(np.arange(len(e))+g)*.001
            # 200ms block means expose persistent drift without curve clutter.
            n=len(e)//200
            axs[0].plot(t[:n*200].reshape(n,200).mean(1),e[:n*200].reshape(n,200).mean(1),label=f'{arm}: mean {bias:+.1f} N')
        axs[0].axhline(0,color='black',lw=.8); axs[0].set(title=name,ylabel='Prediction - measurement (N)',xlabel='Processed time (s)'); axs[0].legend(fontsize=8)
        m=manifest[name]; path=Path(m['original_source_path']); assert m['split']=='validation'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==m['raw_sha256']
        raw=read_raw_csv(path); a=m['common_start']; b=m['common_stop_exclusive']; shift=m['force_shift_samples']
        x=sos_lowpass_zero_phase((raw.iloc[a:b,3].to_numpy()-4)/16*1500,50)
        force=sos_lowpass_zero_phase(raw.iloc[a+shift:b+shift,5].to_numpy()/10*1000*9.80665,40)
        z0=m['baseline_start_aligned_idx']; z1=m['baseline_end_aligned_idx']; onset=windows[name]['initial_still_duration_ms']
        stop=max(z1,onset-100); f0=m['force_offset']; q=force[z0:z1]
        r=dict(name=name,reference=changes[name],initial_still_ms=onset,crop_start_ms=m['crop_start_aligned_idx'],
            new_baseline_reproduction_error_N=float(q.mean()-f0),baseline_first_half_mean_N=float(q[:len(q)//2].mean()),
            baseline_last_half_mean_N=float(q[len(q)//2:].mean()),
            later_still_mean_minus_initial_N=float(force[z1:stop].mean()-f0) if stop>z1 else None,
            baseline_50ms_block_mean_min_N=float(min(z.mean() for z in np.array_split(q,max(1,len(q)//50)))),
            baseline_50ms_block_mean_max_N=float(max(z.mean() for z in np.array_split(q,max(1,len(q)//50)))),
            mts_reference_mm=m['mts_ref'],force_offset_N=f0)
        start=m['crop_start_aligned_idx']; end=m['crop_end_aligned_idx']; measured=np.array(d['measured_force_N'])
        r['raw_to_label_max_abs_error_N']=float(np.max(np.abs(force[start:end]-f0-measured)))
        rawrows.append(r)
        t=np.arange(min(len(force),onset+300))*.001; axs[1].plot(t,force[:len(t)]-f0,lw=1)
        axs[1].axvspan(z0*.001,z1*.001,color='orange',alpha=.3,label='Baseline used'); axs[1].axvline(onset*.001,color='red',ls='--',label='Motion onset')
        axs[1].set(title='Raw aligned force minus initial baseline',xlabel='Aligned raw time (s)',ylabel='Force (N)'); axs[1].legend(fontsize=8)
        for ax in axs: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/'residual_and_initial_reference.png',dpi=170); plt.close(fig)
    metrics=json.loads((OUT/'random_file_metrics.json').read_text()); associations=[]
    for arm in ARMS:
        for cat in ('SR','RLF','RHF'):
            rr=[r for r in metrics if r['arm']==arm and r['split']=='train' and r['name'].startswith(cat+'_')]
            bias=np.array([r['mean_error_N'] for r in rr]); offset=np.array([manifest[r['name']]['force_offset'] for r in rr]); xref=np.array([manifest[r['name']]['mts_ref'] for r in rr])
            associations.append(dict(arm=arm,category=cat,n=len(rr),mean_bias_min_N=float(bias.min()),mean_bias_max_N=float(bias.max()),
                corr_bias_force_offset=float(np.corrcoef(bias,offset)[0,1]),slope_bias_force_offset=float(np.polyfit(offset,bias,1)[0]),
                corr_bias_x_reference=float(np.corrcoef(bias,xref)[0,1])))
    probes=[]
    for arm in ARMS:
        for name in TARGETS:
            cat=name.split('_')[0]
            rr=[r for r in metrics if r['arm']==arm and r['split']=='train' and r['name'].startswith(cat+'_')]
            alpha,beta=np.polyfit([manifest[r['name']]['force_offset'] for r in rr],[r['mean_error_N'] for r in rr],1)
            predicted_bias=float(alpha*manifest[name]['force_offset']+beta)
            orig=next(r for r in rows if r['arm']==arm and r['name']==name)
            rmse=float(np.sqrt(orig['demeaned_diagnostic_rmse_N']**2+(orig['bias_N']-predicted_bias)**2))
            probes.append(dict(arm=arm,name=name,training_files=len(rr),train_only_slope=float(alpha),train_only_intercept=float(beta),
                predicted_bias_from_training_N=predicted_bias,actual_validation_bias_N=orig['bias_N'],
                original_rmse_N=orig['rmse_N'],diagnostic_rmse_after_train_only_bias_prediction_N=rmse,
                warning='Diagnostic association only. No checkpoint/data edits; not a deployed correction or a proof of sensor error.'))
    write(OUT/'training_only_reference_probe.json',probes)
    write(OUT/'residual_summary.json',rows); write(OUT/'raw_reference_audit.json',rawrows); write(OUT/'training_associations.json',associations)
    print(json.dumps(dict(residuals=rows,raw=rawrows,training_associations=associations),indent=2))
if __name__=='__main__': main()
