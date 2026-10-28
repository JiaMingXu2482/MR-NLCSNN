"""Package completed four-model results; no model execution or data CSV access."""
import json,hashlib,shutil,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
RUN=HERE/'sessions/v19_capacity_four_balanced_300_20261028'
OUT=HERE/'delivery_capacity_four_300_20261028'
ARMS=['D','A','A_GRU','A_MLP']
COLORS=['#999999','#d08723','#2a9955','#227ac0']
LABELS=['D','A (16 / 48)','A GRU (32 / 48)','A MLP (16 / 96)']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42})
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight'); fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight'); plt.close(fig)
def main():
    assert read(RUN/'COMPLETE.json')['status']=='complete'
    if OUT.exists(): raise FileExistsError(OUT)
    metrics={a:read(RUN/a/'validation_metrics.json') for a in ARMS}
    configs={a:read(RUN/a/'run_config.json') for a in ARMS}
    done={a:read(RUN/a/'TRAIN_COMPLETE.json') for a in ARMS}
    completion={a:read(RUN/a/'COMPLETE.json') for a in ARMS}
    histories={a:[json.loads(l) for l in (RUN/a/'metrics.jsonl').read_text().splitlines()] for a in ARMS}
    for a in ARMS:
        assert [r['epoch'] for r in histories[a]]==list(range(1,301))
        assert [r['updates'] for r in histories[a]]==list(range(248,74401,248))
        assert done[a]['updates']==74400 and done[a]['seen_scored_samples']==632902888
        assert sha(RUN/a/'final_unvalidated.pth')==done[a]['final_sha256']
        assert configs[a]['schedule_sha256']==sha(RUN/'schedule.json')
        for p,h in configs[a]['source_hashes'].items(): assert sha(Path(p))==h,p
    preds={a:{p.stem:read(p) for p in sorted((RUN/a/'validation_predictions').glob('*.json'))} for a in ARMS}
    names=list(preds['D']); assert len(names)==16 and all(set(preds[a])==set(names) for a in ARMS)
    for name in names:
        base=preds['D'][name]; g=base['guard']; y=np.array(base['measured_force_N']); valid=slice(g,len(y)-g)
        for a in ARMS:
            d=preds[a][name]; assert d['guard']==g and np.array_equal(d['measured_force_N'],y)
            e=np.array(d['predicted_force_N'])[valid]-y[valid]
            score=100*np.sqrt(np.sum(e*e)/np.sum(y[valid]**2))
            expected=next(r['nrmse_pct'] for r in metrics[a]['rows'] if r['name']==name)
            assert abs(score-expected)<.002,(a,name,score,expected)
    OUT.mkdir(); (OUT/'sessions').mkdir(); (OUT/'per_file').mkdir()
    for a in ARMS: shutil.copytree(RUN/a,OUT/'sessions'/a)
    for name in ['COMPLETE.json','ERROR.json','experiment_config.json','schedule.json','processes.json']+[a+'_console.log' for a in ARMS]:
        if (RUN/name).exists(): shutil.copy2(RUN/name,OUT/'sessions'/name)
    shutil.copytree(RUN/'code_snapshot',OUT/'code_snapshot')
    for name in ['CAPACITY_FOUR_RUN_20261028.md','capacity_four_preflight_20261028.json','capacity_four_config_20261028.json']:
        shutil.copy2(HERE/name,OUT/name)
    data=Path(configs['D']['experiment']['data_path']); (OUT/'data_provenance').mkdir()
    for name in ['alignment_manifest.json','zero_reference_audit.json','COMPLETE.json']: shutil.copy2(data/name,OUT/'data_provenance'/name)
    # Preserve external upstream source used by the run, with a map to original paths.
    paths=list(configs['D']['source_hashes']); source_map={}
    for i,p in enumerate(paths):
        target=OUT/'code_snapshot'/f'provenance_{i:02d}_{Path(p).name}'; shutil.copy2(p,target); source_map[str(target.relative_to(OUT))]=p
    summary={a:dict(primary=metrics[a]['verified_primary'],secondary_all16=dict(pooled_nrmse_pct=metrics[a]['pooled_nrmse_pct'],category_nrmse_pct=metrics[a]['category_nrmse_pct']),
        unverified_file=metrics[a]['unverified_secondary'],random_macro=completion[a]['random_macro_nrmse_pct'],parameters=completion[a]['parameter_count'],training=done[a]) for a in ARMS}
    (OUT/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    groups=['SH','SH6.62','ST','SR','RLF','RHF','Random mean','Pooled15']
    def value(a,g):
        p=summary[a]['primary']
        return p['hf_nrmse_pct'] if g=='SH6.62' else summary[a]['random_macro'] if g=='Random mean' else p['pooled_nrmse_pct'] if g=='Pooled15' else p['category_nrmse_pct'][g]
    fig,ax=plt.subplots(figsize=(14,5.2)); x=np.arange(len(groups))
    for i,a in enumerate(ARMS):
        bars=ax.bar(x+(i-1.5)*.2,[value(a,g) for g in groups],width=.19,color=COLORS[i],label=LABELS[i]); ax.bar_label(bars,fmt='%.2f',fontsize=7,padding=2)
    ax.set_xticks(x,groups); ax.set(ylabel='Validation NRMSE (%)',ylim=(0,38),title='Four cold-start models | 300 epochs | verified-reference validation (15 files)'); ax.legend(ncol=4,fontsize=9); fig.tight_layout(); save(fig,'01_category_comparison')
    def force(ax,name,zoom=False):
        d=preds['D'][name]; y=np.array(d['measured_force_N']); t=np.arange(len(y))*.001; ix=np.arange(d['guard'],len(y)-d['guard'])
        if zoom: ix=ix[np.abs(t[ix]-(t[ix[0]]+t[ix[-1]])/2)<=1]
        ax.plot(t[ix],y[ix],color='#111',lw=1.4,label='Measured')
        for a,c,l in zip(ARMS,COLORS,LABELS): ax.plot(t[ix],np.array(preds[a][name]['predicted_force_N'])[ix],color=c,lw=.9,alpha=.8,label=l)
        ax.set(xlabel='Time (s)',ylabel='Force (N)')
    random_names=[n for n in names if n.startswith(('SR_','RLF_','RHF_'))]
    fig,axs=plt.subplots(5,2,figsize=(16,17))
    for row,name in zip(axs,random_names):
        force(row[0],name); force(row[1],name,True)
        flag='' if preds['D'][name]['zero_reference_verified'] else ' [UNVERIFIED ZERO; SECONDARY]'
        row[0].set_title(name+flag,loc='left',fontsize=10)
        vals=[next(r['nrmse_pct'] for r in metrics[a]['rows'] if r['name']==name) for a in ARMS]
        row[1].set_title('Center 2 s | full-score NRMSE: '+' / '.join(f'{v:.2f}%' for v in vals),fontsize=9)
    h,l=axs[0,0].get_legend_handles_labels(); fig.legend(h,l,ncol=5,loc='upper center',bbox_to_anchor=(.5,.977),fontsize=9)
    fig.suptitle('SR / RLF / RHF | same files, measured motion/current history',y=.998,fontsize=16)
    fig.tight_layout(rect=(0,.02,1,.955)); fig.text(.5,.006,'No inference rerun. One RLF file lacks a static reference and is excluded from primary metrics.',ha='center'); save(fig,'02_random_predictions')
    for name in names:
        fig,ax=plt.subplots(3,1,figsize=(13,9.5)); force(ax[0],name); force(ax[1],name,True)
        flag='' if preds['D'][name]['zero_reference_verified'] else ' | UNVERIFIED ZERO: SECONDARY ONLY'
        ax[0].set_title(name+flag,fontsize=11); ax[0].legend(ncol=5,fontsize=8); ax[1].set_title('Fixed center window (up to 2 s)',fontsize=10)
        d=preds['D'][name]; g=d['guard']; y=np.array(d['measured_force_N']); ix=np.arange(g,len(y)-g); t=ix*.001
        for a,c in zip(ARMS,COLORS): ax[2].plot(t,(np.array(preds[a][name]['predicted_force_N'])-y)[ix],lw=.8,color=c)
        ax[2].axhline(0,lw=.5,color='black'); ax[2].set(xlabel='Time (s)',ylabel='Error (N)'); fig.tight_layout(); save(fig,'per_file/'+name)
    fig,ax=plt.subplots(figsize=(10,4.5)); x=np.arange(4)
    for i,a in enumerate(ARMS): ax.bar(x+(i-1.5)*.2,[b['rmse_N'] for b in summary[a]['primary']['speed_bins']],width=.19,label=LABELS[i],color=COLORS[i])
    ax.set_xticks(x,['[0,10)','[10,50)','[50,100)','[100,infinity)']); ax.set(xlabel='Absolute velocity (mm/s)',ylabel='RMSE (N)',title='15 verified-reference files | speed-bin error'); ax.legend(ncol=2); fig.tight_layout(); save(fig,'03_speed_bins')
    fig,ax=plt.subplots(figsize=(10,4.5))
    for a,c,l in zip(ARMS,COLORS,LABELS): ax.plot([r['epoch'] for r in histories[a]],[r['training_loss'] for r in histories[a]],label=l,color=c,lw=1)
    ax.set(yscale='log',xlabel='Epoch',ylabel='Training loss (log scale)',title='Training loss only | no intermediate validation'); ax.legend(); fig.tight_layout(); save(fig,'04_training_loss')
    text=['# 四模型300轮实验结果','',
      '四组均完成300轮冷启动训练，随后统一验证。当前单seed结果：A_MLP的主要总体NRMSE最低；普通A的SR/RLF/RHF等权平均最低。随机工况误差仍较大。没有启动后续训练。',
      '', '## 主要指标：15个初始静态基准核验文件','',
      '| 模型 | 参数 | SH | ST | SR | RLF | RHF | 三类等权平均 | 总体 | SH6.62Hz |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in ARMS: text.append('| '+a+' | '+str(summary[a]['parameters'])+' | '+' | '.join(f'{value(a,g):.6f}' for g in ['SH','ST','SR','RLF','RHF','Random mean','Pooled15','SH6.62'])+' |')
    text += ['', '## 解读与限制','',
      '- 同宽度D→A，SR误差下降，但a与历史dI同时加入，参数量也增加；不能分别证明a、dI的贡献，不能认定输入信息比容量普遍重要。',
      '- A_GRU比A的RLF/RHF略低，但SR更高；A_MLP总体与高频最低，随机三类平均略高于A。扩大网络没有在所有指标上胜出。',
      '- 本次只有seed7，没有重复试验，不作统计显著性或“噪声量级”判断。参数量不是硬件耗时，未做S32速度或部署验收。',
      '- 四组标签、采样、预算一致；旧D使用旧零点和训练集，不能直接把旧分数作为本次同口径基线。',
      '- 静态基准“核验”是按固定起始位移静止规则取得；仍含测量噪声，不等于取得真实无载零点。',
      '- 给定实测运动与电流历史的预测结果，不证明闭环稳定性。最终测试集未访问。',
      '', '## 次要指标：全部16文件与缺静态基准文件','',
      '| 模型 | 全16总体NRMSE % | RLF_RMS15mm_3Hz NRMSE %（未核验零点） |','|---|---:|---:|']
    for a in ARMS: text.append(f'| {a} | {metrics[a]["pooled_nrmse_pct"]:.6f} | {metrics[a]["unverified_secondary"][0]["nrmse_pct"]:.6f} |')
    text += ['', '## 逐文件NRMSE (%)','', '| 文件 | 基准核验 | D | A | A_GRU | A_MLP |','|---|---|---:|---:|---:|---:|']
    for name in names: text.append('| '+name+' | '+('是' if preds['D'][name]['zero_reference_verified'] else '否：次要单列')+' | '+' | '.join(f'{next(r["nrmse_pct"] for r in metrics[a]["rows"] if r["name"]==name):.6f}' for a in ARMS)+' |')
    text += ['', '## 速度分段：主要15文件RMSE (N)','', '| 绝对速度 mm/s | D | A | A_GRU | A_MLP |','|---|---:|---:|---:|---:|']
    for j,label in enumerate(['[0,10)','[10,50)','[50,100)','[100,∞)']): text.append('| '+label+' | '+' | '.join(f'{summary[a]["primary"]["speed_bins"][j]["rmse_N"]:.6f}' for a in ARMS)+' |')
    text += ['', '## 训练预算与恢复','',
      '四组均74400次优化更新、累计632902888个有效样本，300行epoch日志连续且更新编号核对通过；schedule哈希一致。AdamW、warmup5轮后cosine，lr3e-4→3e-6，完整文件批次，ST权重3，负力权重1.5。',
      '', '| 模型 | 训练块累计耗时 秒 | 第300轮loss | 最终tau ms |','|---|---:|---:|---:|']
    for a in ARMS: text.append(f'| {a} | {done[a]["training_seconds"]:.3f} | {histories[a][-1]["training_loss"]:.8f} | {histories[a][-1]["tau_ms"]:.6f} |')
    text += ['',
      '上述耗时是在同一GPU上并发训练的累计块耗时，恢复前后资源竞争不同，不可据此比较独占GPU速度；也不是推理耗时。',
      'A_GRU曾因写入progress.json遭WinError5（拒绝访问）退出；日志显示从step17360恢复后完成300轮。仅凭错误不能确定哪个进程锁住文件。旧ERROR.json留作历史证据；最后COMPLETE.json及完整指标确认任务已完成。progress.json仍显示waiting_for_all_arms是原脚本未刷新终态，不代表训练仍在运行。',
      '', '## 交付文件','',
      '- 01_category_comparison、02_random_predictions、03_speed_bins、04_training_loss：PNG和PDF。per_file/含16个工况全段、中心2秒及残差图。',
      '- sessions/各组final.pth为第300轮最终权重，未挑选中间最佳轮次。保留最新恢复权重、完整日志、预测JSON、配置和完成标志。',
      '- comparison.json包含主要/次要指标；sessions/*/validation_metrics.json包含逐文件及速度区间数据。',
      '- data_provenance/包含数据manifest、零点修正和剔除记录；不复制CSV。code_snapshot/包含训练用源码。',
      '- audit.json记录源码及数据哈希、日志连续性和预测分数复算；SHA256.json记录交付文件哈希。',
      '', 'NRMSE=100*sqrt(Σ误差²/Σ实测力²)，分母没有减均值。三类平均是SR/RLF/RHF类别NRMSE算术平均。图片固定中心窗口不参与重新选模。未推GitHub。']
    (OUT/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    audit=dict(validation_predictions_reproduced_tolerance_pp=.002,epoch_log_continuity_passed=True,shared_schedule_sha256=sha(RUN/'schedule.json'),
        data_manifest_sha256=sha(data/'alignment_manifest.json'),source_hashes_verified=configs['D']['source_hashes'],source_map=source_map,
        checkpoint_hashes={a:sha(RUN/a/'final.pth') for a in ARMS},final_test_csv_read=False,no_training_or_inference_performed_by_report=True)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8'); shutil.copy2(__file__,OUT/Path(__file__).name)
    hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256.json'}
    (OUT/'SHA256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(OUT),files=len(hashes)),ensure_ascii=False))
if __name__=='__main__': main()
