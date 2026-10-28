"""Inspect raw initial windows for existing train/validation files only."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
DATA=Path(r'C:\Users\user\AppData\Local\Temp\v19_bd_runtime_20261027\pack\Parsed_v19_F42_I42')
sys.path.insert(0,r'C:\Users\user\AppData\Local\Temp\v19_bd_runtime_20261027\source\AA01DataProcess')
from process_raw_card_archive import read_raw_csv,sos_lowpass_zero_phase
OUT=HERE/'training_zero_audit_20261028'
def main():
    rows=[]; examples=[]
    for m in json.loads((DATA/'alignment_manifest.json').read_text()):
        if m['split'] not in ('train','validation'): continue
        path=Path(m['original_source_path']); assert hashlib.sha256(path.read_bytes()).hexdigest()==m['raw_sha256']
        raw=read_raw_csv(path); a=m['common_start']; b=m['common_stop_exclusive']; shift=m['force_shift_samples']
        x=sos_lowpass_zero_phase((raw.iloc[a:b,3].to_numpy()-4)/16*1500,50)
        f=sos_lowpass_zero_phase(raw.iloc[a+shift:b+shift,5].to_numpy()/10*1000*9.80665,40)
        ref=x[:50].mean(); where=np.flatnonzero(np.abs(x-ref)>.15); n=int(where[0]) if len(where) else len(x)
        v=np.gradient(x,.001)
        r=dict(name=Path(m['relative_path']).stem,split=m['split'],zero_source=m['zero_source'],initial_still_duration_ms=n,
            first_500ms_x_p2p_mm=float(np.ptp(x[:500])),first_500ms_velocity_rms_mm_s=float(np.sqrt(np.mean(v[:500]**2))),
            first_500ms_force_mean_N=float(f[:500].mean()),first_500ms_force_std_N=float(f[:500].std()),
            old_offset_N=m['force_offset'],mean_500ms_minus_old_N=float(f[:500].mean()-m['force_offset']),
            path=str(path))
        rows.append(r)
        if m['zero_source']=='physics-estimate': examples.append((r,x[:1000],f[:1000]))
    (OUT/'initial_windows.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    fig,axs=plt.subplots(len(examples),2,figsize=(12,2.3*len(examples)),squeeze=False)
    for ax,(r,x,f) in zip(axs,examples):
        t=np.arange(len(x))*.001; ax[0].plot(t,x-x[:50].mean()); ax[0].set(title=r['name']+' ('+r['split']+')',ylabel='Relative x (mm)')
        ax[1].plot(t,f); ax[1].axhline(r['old_offset_N'],ls='--',color='orange'); ax[1].set(title=f'Initial still window: {r["initial_still_duration_ms"]}ms',ylabel='Raw aligned F (N)')
        for aa in ax: aa.set_xlabel('Time from raw start (s)'); aa.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/'physics_estimate_initial_windows.png',dpi=130); plt.close(fig)
    for split in ('train','validation'):
        rr=[r for r in rows if r['split']==split]
        print(split,'count',len(rr),'initial<200ms',sum(r['initial_still_duration_ms']<200 for r in rr),flush=True)
        print(json.dumps([r for r in rr if r['zero_source']=='physics-estimate'],ensure_ascii=False),flush=True)
if __name__=='__main__': main()
