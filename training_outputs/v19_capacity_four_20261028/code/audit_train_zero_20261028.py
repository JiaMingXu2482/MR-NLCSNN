"""Read-only audit of 259 train files and their exact manifest-listed raw files.
No validation/final-test CSV, no training, no corrected labels produced.
"""
import json,sys,hashlib,collections,math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
RUNTIME=Path(r'C:\Users\user\AppData\Local\Temp\v19_bd_runtime_20261027')
DATA=RUNTIME/'pack/Parsed_v19_F42_I42'
OUT=HERE/'training_zero_audit_20261028'
sys.path.insert(0,str(RUNTIME/'source/AA01DataProcess'))
from process_raw_card_archive import read_raw_csv,sos_lowpass_zero_phase

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir()
    manifest=DATA/'alignment_manifest.json'
    assert sha(manifest)=='0766763e7fb77880ffa04f2d93bcf629bfeb817e51c571188eee1ea83262e2ff'
    records=[r for r in json.loads(manifest.read_text()) if r['split']=='train']; assert len(records)==259
    old=json.loads((HERE/'diagnosis_random_20261027/file_diagnostics.json').read_text())
    diag={r['name']:r for r in old if r['split']=='train'}
    result=[]
    for idx,m in enumerate(records):
        name=Path(m['relative_path']).stem; cat=name.split('_')[0]
        r=dict(name=name,category=cat,zero_source=m['zero_source'],offset_N=m['force_offset'],
            x_reference_mm=m['mts_ref'],manual_zero_required=m['manual_zero_required'],
            mean_prediction_error_N=diag[name]['mean_error_N'],raw_path=m['original_source_path'])
        p=DATA/m['relative_path']; assert sha(p)==m['processed_sha256']
        parsed=pd.read_csv(p); r.update(label_mean_N=float(parsed.force.mean()),label_median_N=float(parsed.force.median()))
        path=Path(m['original_source_path'])
        if not path.exists():
            r['raw_status']='missing'; result.append(r); continue
        if sha(path)!=m['raw_sha256']:
            r['raw_status']='hash_mismatch_not_used'; result.append(r); continue
        raw=read_raw_csv(path); a=m['common_start']; b=m['common_stop_exclusive']; shift=m['force_shift_samples']
        x=sos_lowpass_zero_phase((raw.iloc[a:b,3].to_numpy()-4)/16*1500,50)
        f=sos_lowpass_zero_phase(raw.iloc[a+shift:b+shift,5].to_numpy()/10*1000*9.80665,40)
        start=m['crop_start_aligned_idx']; end=m['crop_end_aligned_idx']; ca=m['baseline_start_aligned_idx']; cb=m['baseline_end_aligned_idx']
        delta_f=float(np.max(np.abs(f[start:end]-m['force_offset']-parsed.force.to_numpy())))
        delta_x=float(np.max(np.abs(250-(x[start:end]-m['mts_ref'])-parsed.rod_length.to_numpy())))
        assert delta_f<.012 and delta_x<.002,(name,delta_f,delta_x)
        r.update(raw_status='exact_hash_and_reproduction',reproduced_F_max_abs_error_N=delta_f,reproduced_x_max_abs_error_mm=delta_x)
        if ca>=0:
            n=cb-ca; q=f[ca:cb]; v=np.gradient(x,.001)[ca:cb]; seg=min(500,n//3)
            # Uniform diagnostic margin: trim 200ms at both ends. Never applied to labels.
            r.update(baseline_samples=n,baseline_force_std_N=float(q.std()),baseline_x_p2p_mm=float(np.ptp(x[ca:cb])),
                baseline_velocity_rms_mm_s=float(np.sqrt(np.mean(v*v))),
                first_last_mean_delta_N=float(q[-seg:].mean()-q[:seg].mean()),
                old_minus_trimmed_zero_N=float(m['force_offset']-q[200:-200].mean()) if n>=600 else None)
        else:
            xx=x[start:end]; ff=f[start:end]; neutral=xx.mean(); vv=np.gradient(xx,.001)
            mask=(np.abs(xx-neutral)<.1*(xx.std()+1e-9)) & (np.abs(vv)<np.percentile(np.abs(vv),10))
            r.update(physics_candidate_samples=int(mask.sum()),physics_used_active_mean=int(mask.sum())<=20,
                physics_estimated_offset_N=float(ff[mask].mean()) if mask.sum()>20 else float(ff.mean()))
        result.append(r)
        if (idx+1)%25==0: print(f'Audited {idx+1}/259 train files',flush=True)
    (OUT/'per_file.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    summary=[]
    for cat in ('SH','ST','SR','RLF','RHF'):
        rr=[r for r in result if r['category']==cat]; offsets=[r['offset_N'] for r in rr]
        trim=[r['old_minus_trimmed_zero_N'] for r in rr if r.get('old_minus_trimmed_zero_N') is not None]
        summary.append(dict(category=cat,files=len(rr),zero_sources=dict(collections.Counter(r['zero_source'] for r in rr)),
            raw_status=dict(collections.Counter(r['raw_status'] for r in rr)),offset_min_N=min(offsets),offset_max_N=max(offsets),
            physics_fallback_active_mean=sum(r.get('physics_used_active_mean',False) for r in rr),
            trimmed_delta_abs_median_N=float(np.median(np.abs(trim))) if trim else None,
            trimmed_delta_abs_max_N=float(np.max(np.abs(trim))) if trim else None,
            trimmed_delta_abs_gt10N=sum(abs(v)>10 for v in trim)))
    record=dict(groups=summary,files=259,validation_or_final_csv_read=False,training_started=False,
        data_modified=False,manifest_sha256=sha(manifest),script_sha256=sha(Path(__file__)),
        diagnostic_trim='200 samples at each baseline edge, only windows>=600; not a proposed automatic correction',
        caveat='Different offsets or nonzero dynamic means alone do not prove a bias error; initial physical preload and history may differ.')
    (OUT/'summary.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    fig,axs=plt.subplots(1,2,figsize=(13,4.5)); colors={'SH':'#aaa','ST':'#999020','SR':'#d77d31','RLF':'#1177aa','RHF':'#7c4ea0'}
    for cat in colors:
        rr=[r for r in result if r['category']==cat]
        axs[0].scatter([r['offset_N'] for r in rr],[r['mean_prediction_error_N'] for r in rr],s=17,label=cat,color=colors[cat],alpha=.7)
        q=[r for r in rr if r.get('old_minus_trimmed_zero_N') is not None]
        axs[1].scatter([r['old_minus_trimmed_zero_N'] for r in q],[r['mean_prediction_error_N'] for r in q],s=17,label=cat,color=colors[cat],alpha=.7)
    axs[0].set(xlabel='Per-file subtracted force offset (N)',ylabel='Frozen D mean prediction error (N)',title='Training set: association, not proof of causality')
    axs[1].set(xlabel='Old zero minus 200ms-edge-trimmed zero (N)',ylabel='Frozen D mean prediction error (N)',title='Uniform zero-window sensitivity (diagnostic only)')
    for ax in axs: ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/'training_zero_audit.png',dpi=160); plt.close(fig)
    text=['# 训练集力零点审查（只读）','','没有启动四组训练；未改数据；未读取验证或最终测试CSV。','',
        '结论：减力偏置的公式统一，但基准的获取方法不统一；不能把“同一代码处理”视为物理零点已统一。',
        '', '| 类别 | 文件数 | 零点来源 | 原始文件复现 | offset范围 N | 200ms裁边敏感性中位/最大绝对值 N |','|---|---:|---|---|---|---|']
    for r in summary:
        text.append(f'| {r["category"]} | {r["files"]} | {r["zero_sources"]} | {r["raw_status"]} | {r["offset_min_N"]:.2f} ~ {r["offset_max_N"]:.2f} | {r["trimmed_delta_abs_median_N"]} / {r["trimmed_delta_abs_max_N"]} |')
    text += ['', '## 已确认的实现问题','',
        '- process_raw_card_archive.py:112–135 静止窗口仅按位移阈值0.15mm和时长筛选，没有联合检查力/速度是否稳态。',
        '- 同文件138–153 physics-estimate使用运动段近中位、低速点估计；点数≤20时使用整段平均力。低速不等于无力，不能把该估计当作已知仪器零偏。',
        '- 206–208 auto-baseline-override仅修改标记，并未读取人工零点数值。',
        '- 215–216 每文件位移和力都重新建立基准；若扣除可变预载，其信息未显式作为模型输入。',
        '', '## 限制与处理建议','',
        '200ms裁边仅检查窗口敏感性，不是可直接应用的校正，也不能从预测残差反推每个文件真零点。动态力均值不必为0。',
        '需要确认采集时是否逐文件清零，以及目标是总轴向力还是扣除试验前静态预载的动态力。确认之前不对原始或验证标签做按文件减均值。',
        '如需修正，保留旧集并输出新版本，所有模型使用同一预先固定规则；无可靠静态基准的文件须标记或补充参考，不能从验证误差拟合校正量。',
        '', '四组计划：D(2×16/48)、A(2×16/48)、A_GRU(2×32/48)、A_MLP(2×16/96)，G=24；各300轮=77700次更新，全程同一平衡采样，结束后统一验证。A三组启用严格历史dI，D不使用a/dI。']
    (OUT/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__': main()
