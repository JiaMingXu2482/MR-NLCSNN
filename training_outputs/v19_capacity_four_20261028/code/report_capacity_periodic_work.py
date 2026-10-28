import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; OUT=HERE/'periodic_work_audit_20261028'
rows=json.loads((OUT/'all_results.json').read_text())
legacy=json.loads((HERE/'periodic_work_legacy_center_20261028/all_results.json').read_text())
arms=['D','A','A_GRU','A_MLP']; colors=['#777777','#e69f00','#009e73','#0072b2']
old_work_1s={.5:{2:.14119912,8:-.13581440,12:-.98069747},5:{2:9.30222262,8:-119.53929099,12:-291.91993258}}
fig,axs=plt.subplots(1,2,figsize=(11,4.4))
for ax,amp in zip(axs,[.5,5]):
    fs=[2,8,12]; positions=np.arange(3); width=.15
    ax.bar(positions-2*width,[old_work_1s[amp][f]/f for f in fs],width,label='v17 (archived)',color='#cc3311')
    for j,(arm,color) in enumerate(zip(arms,colors)):
        values=[next(r['mean_cycle_work_J'] for r in legacy if r['arm']==arm and r['amplitude_mm']==amp and r['frequency_Hz']==f) for f in fs]
        ax.bar(positions+(j-1)*width,values,width,label=arm,color=color)
    ax.axhline(0,color='black',lw=.8); ax.set_xticks(positions,fs); ax.set(xlabel='Frequency (Hz)',ylabel='Net mechanical work (J/cycle)',title=f'Amplitude {amp} mm; I=0 A; T=30 C')
    ax.grid(axis='y',alpha=.2)
handles,labels=axs[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='upper center',ncol=5)
fig.suptitle('Same numerical position center: 279.1375 mm; positive work = dissipative',y=.90,fontsize=10)
fig.tight_layout(rect=(0,0,1,.88)); fig.savefig(OUT/'v17_vs_four_cycle_work.png',dpi=180); fig.savefig(OUT/'v17_vs_four_cycle_work.pdf'); plt.close(fig)
summary={arm:dict(cases=426,negative_cases=sum(r['negative'] for r in rows+legacy if r['arm']==arm),
    minimum_individual_cycle_J=min(min(r['cycle_work_J']) for r in rows+legacy if r['arm']==arm),
    max_cycle_work_relative_spread=max(r['work_cycle_spread_J']/abs(r['mean_cycle_work_J']) for r in rows+legacy if r['arm']==arm)) for arm in arms}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# 冻结四模型周期功审计（2026-10-28）','',
'结论：本次1704组恒定电流正弦输入检查，四模型每个被检查的完整周期均为正功，没有复现v17的持续负周期功。该结论只针对测试覆盖范围，不是全域无源性证明，也不是CarSim闭环稳定性结论。','',
'## 协议','',
'- 使用本次300轮最终D/A/A_GRU/A_MLP权重，无保护、无力裁剪、无额外滤波、力倍率1、不训练。',
'- 符号采用此前审计口径：W=∫F*v dt，v为所输入位移的导数，速度从mm/s转m/s；正值表示净耗能。',
'- 1ms离散推理，x、v、a由同一个正弦解析生成；A使用模型自身历史电流协议。',
'- 每组先运行至少15周期且不少于5秒，再积分最后5个完整周期；非整数采样周期用边界插值，不截取残缺周期。',
'- 主扫描中心250mm（当前初始位置参考），频率0.5/1/1.66/2/3.3/4.97/5/6.62/8/8.5/10/12/15/20Hz，振幅0.1/0.5/1/5/10mm，电流0/0.8/1.6A，温度30/40℃，每模型420组。',
'- 同旧数值中心279.1375mm复核：振幅0.5/5mm，频率2/8/12Hz，0A、30℃，每模型6组，见相邻periodic_work_legacy_center_20261028目录。',
'- 同一记录末5周期功最大相对极差约0.415%；非整周期采样存在离散相位差，未宣称每个样点严格重复。',
'- 只用合成输入，未读取最终测试数据。','',
'## 同旧数值中心的关键点（J/周期）','',
'|振幅mm|频率Hz|v17存档|D|A|A_GRU|A_MLP|','|---:|---:|---:|---:|---:|---:|---:|']
for amp in [.5,5]:
    for f in [2,8,12]:
        vals=[next(r['mean_cycle_work_J'] for r in legacy if r['arm']==arm and r['amplitude_mm']==amp and r['frequency_Hz']==f) for arm in arms]
        lines.append('|'+ '|'.join([str(amp),str(f),f'{old_work_1s[amp][f]/f:+.6f}']+[f'{v:+.6f}' for v in vals])+'|')
lines+=['','旧值来自10.12/forward4_runs/raw_audit_20261021_190430/audit.txt和confirmation.txt；periodic_a1_f8.mat确认旧中心279.1375mm。旧记录标注final1sWork，为最后1秒累计功，上表除以频率转为每周期功，不能把旧的−0.1358J误写成8Hz单周期功。此次没有重新运行v17。相同数值中心用于复查位置敏感性，不代表新旧预处理的安装参考物理条件完全相同。','',
'## 如何解释','',
'1. 短时Fv<0仍可存在；判据是完整周期净功，不要求瞬时功处处为正。',
'2. 主扫描中59个记录的b分支末周期功为负，但总功仍为正。b分支没有被强行删除或夹紧；不能据b单独负功判定整个模型净发能。',
'3. 对真正闭合的位移周期，常量力偏置B的贡献是B∮dx=0。因此上一轮发现的文件级预测偏置与周期净发能是不同问题；本次功检查通过不代表预测偏置已解决。',
'4. 仍需单独检验变化电流、其他温度/位移中心、随机激励、实际CarSim反馈及接口符号/延迟。有限正弦网格不能证明任意输入无源，正周期功也不证明力预测准确。',
'5. 合成扫描部分工况可能超出实际训练联合分布，只作为数值压力检查，不作为对应工况的精度证据。','',
'## 文件','',
'all_results.json包含每组末5周期功、总功/正项/b项分解、瞬时负功比例和周期重复差异；summary.json为汇总；provenance.json包含权重哈希与脚本哈希。两个独立audit脚本和report脚本位于上级目录。']
(OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
