"""Create a separate initial-static-reference dataset; preserve all old files."""
import csv,json,hashlib,sys,copy
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
RUNTIME=Path(r'C:\Users\user\AppData\Local\Temp\v19_bd_runtime_20261027')
SOURCE=RUNTIME/'pack/Parsed_v19_F42_I42'
OUT=HERE/'Parsed_v19_F42_I42_initial_static_20261028'
sys.path.insert(0,str(RUNTIME/'source/AA01DataProcess'))
from process_raw_card_archive import read_raw_csv,sos_lowpass_zero_phase
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d): p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def main():
    if OUT.exists(): raise FileExistsError(OUT)
    manifest=SOURCE/'alignment_manifest.json'
    assert sha(manifest)=='0766763e7fb77880ffa04f2d93bcf629bfeb817e51c571188eee1ea83262e2ff'
    rows=json.loads(manifest.read_text()); windows={r['name']:r for r in json.loads((HERE/'training_zero_audit_20261028/initial_windows.json').read_text())}
    OUT.mkdir(); changes=[]
    for m in rows:
        if m['split'] not in ('train','validation'): continue
        name=Path(m['relative_path']).stem; w=windows[name]; n=w['initial_still_duration_ms']
        m['source_processed_sha256']=m['processed_sha256']; m['previous_reference']=dict(force_offset=m['force_offset'],mts_ref=m['mts_ref'],zero_source=m['zero_source'])
        if n<200 and m['split']=='train':
            m['original_split']='train'; m['split']='quarantined_train'; m['zero_reference_verified']=False
            changes.append(dict(name=name,action='excluded_training',reason='initial displacement-still segment <200ms',original_split='train'))
            continue
        original=SOURCE/m['relative_path']; assert sha(original)==m['source_processed_sha256']
        dest=OUT/m['relative_path']; dest.parent.mkdir(exist_ok=True,parents=True)
        if n<200:
            # Preserve this validation file for visibly labelled secondary reporting.
            dest.write_bytes(original.read_bytes()); m['zero_reference_verified']=False
            changes.append(dict(name=name,action='validation_secondary_only',reason='no reliable initial static reference'))
        else:
            path=Path(m['original_source_path']); assert sha(path)==m['raw_sha256']
            raw=read_raw_csv(path); a=m['common_start']; b=m['common_stop_exclusive']; shift=m['force_shift_samples']
            x=sos_lowpass_zero_phase((raw.iloc[a:b,3].to_numpy()-4)/16*1500,50)
            force=sos_lowpass_zero_phase(raw.iloc[a+shift:b+shift,5].to_numpy()/10*1000*9.80665,40)
            # Fixed policy for every category: skip first50ms, stop100ms before
            # detected displacement onset, use at most500ms. No residual fitting.
            z0=50; z1=min(550,n-100); assert z1-z0>=50
            new_x=float(x[z0:z1].mean()); new_f=float(force[z0:z1].mean())
            df=pd.read_csv(original)
            df['force']=np.round(df['force']+m['force_offset']-new_f,2)
            df['rod_length']=np.round(df['rod_length']+new_x-m['mts_ref'],3)
            df.to_csv(dest,index=False,float_format='%.9g')
            # Derivatives are unaffected by constant reference changes.
            reread=pd.read_csv(dest); prev=pd.read_csv(original)
            for col in ['time','velocity','accel','current','current_dot','temp']: assert np.allclose(reread[col],prev[col],rtol=0,atol=1e-8),col
            assert np.isfinite(reread.to_numpy()).all() and len(reread)==len(prev)
            changes.append(dict(name=name,action='initial_static_corrected',split=m['split'],baseline_start=z0,baseline_stop=z1,
                old_force_offset_N=m['force_offset'],new_force_offset_N=new_f,label_delta_N=m['force_offset']-new_f,
                old_x_reference_mm=m['mts_ref'],new_x_reference_mm=new_x,
                baseline_force_std_N=float(force[z0:z1].std()),baseline_x_p2p_mm=float(np.ptp(x[z0:z1]))))
            m.update(force_offset=new_f,mts_ref=new_x,zero_source='initial-static-fixed-margin-v1',
                baseline_start_aligned_idx=z0,baseline_end_aligned_idx=z1,zero_reference_verified=True)
        m['processed_sha256']=sha(dest)
    write(OUT/'alignment_manifest.json',rows)
    import collections
    counts=collections.Counter(m['split'] for m in rows)
    assert counts['train']==248 and counts['validation']==16
    write(OUT/'COMPLETE.json',dict(manifest_sha256=sha(OUT/'alignment_manifest.json'),train_files=248,validation_files=16,final_test_read=False))
    write(OUT/'zero_reference_audit.json',dict(target='Dynamic force after subtracting per-file initial static preload',
        source_manifest_sha256=sha(manifest),script_sha256=sha(Path(__file__)),
        rule='x within0.15mm of first50-sample mean for >=200 samples; baseline[50,min(550,onset-100)) at1kHz; arithmetic mean; x and F share window',
        limitations='Displacement stillness does not establish unloaded sensor zero. Target is initial-preload-referenced force; no per-file residual fitting.',
        train_files=248,validation_verified=15,validation_unverified=1,final_test_files_present=False,changes=changes))
    cfg=dict(session_root=str(HERE/'sessions/v19_capacity_four_balanced_300_20261028'),data_path=str(OUT),
        data_manifest_sha256=sha(OUT/'alignment_manifest.json'),data_review_approved=True,epochs=300,train_files=248,validation_files=16,
        verified_validation_files=15,secondary_unverified_validation=['RLF_RMS15mm_3Hz'],seed=7,
        sampling={'SH':.25,'ST':.15,'SR':.2,'RLF':.2,'RHF':.2},
        optimization='AdamW lr3e-4 wd1e-4 clip1; warmup5epochs then cosine to3e-6 at300; identical for all four cold starts',
        loss='unchanged file-p2p normalized MSE, negative-force1.5, ST3',
        validation_policy='After all four arms train300epochs; verified15 primary, all16 and unverified1 secondary',
        history_di='A arms recompute causal (I[t-1]-I[t-2])/dt, normalized by1600A/s; D omits a/dI',
        original_data_unchanged=True,final_test_access=False)
    write(HERE/'capacity_four_config_20261028.json',cfg)
    print(json.dumps(dict(counts=dict(counts),manifest=cfg['data_manifest_sha256'],output=str(OUT)),ensure_ascii=False))
if __name__=='__main__': main()
