from pathlib import Path
import json,re,csv,hashlib
from collections import defaultdict,Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_delay import stats,crossing
P=Path(__file__).resolve().parent
r=json.loads((P/'delay_10_90.json').read_text(encoding='utf-8'))
idx={(a['file'],a['event'],a['curve']):a for a in r['events']}
clean=[a for a in r['events'] if a['curve']=='measured' and not a['quality_flags'] and a['delay_ms'] is not None]
paired=[]
for a in clean:
    other=[idx.get((a['file'],a['event'],l)) for l in ['A_all_ST','E_all_ST']]
    if all(x is not None and x['delay_ms'] is not None for x in other):paired.append([a]+other)
def med(xs):return None if not xs else float(np.median(xs))
def fmt(x):return '—' if x is None else f'{x:.1f}'
groups=defaultdict(list)
for a in clean:groups[(re.sub(r'_r\d+$','',a['file']),a['direction'])].append(a)
pg=defaultdict(list)
for a in paired:pg[(re.sub(r'_r\d+$','',a[0]['file']),a[0]['direction'])].append(a)
tables=[]
for k,vals in sorted(groups.items()):
    ps=pg[k]
    tables.append(dict(condition=k[0],direction=k[1],n_measured=len(vals),measured_median_ms=med([v['delay_ms'] for v in vals]),
        n_paired=len(ps),paired_measured_ms=med([v[0]['delay_ms'] for v in ps]),
        A_ms=med([v[1]['delay_ms'] for v in ps]),E_ms=med([v[2]['delay_ms'] for v in ps]),
        A_flagged=sum(bool(v[1]['quality_flags']) for v in ps),E_flagged=sum(bool(v[2]['quality_flags']) for v in ps)))
summary={}
for d in ['up','down']:
    ps=[a for a in paired if a[0]['direction']==d]
    strict=[a for a in ps if all(not x['quality_flags'] for x in a)]
    summary[d]=dict(measured=stats([a for a in clean if a['direction']==d]),
        paired_n=len(ps),paired_medians_ms={label:med([a[j]['delay_ms'] for a in ps]) for j,label in enumerate(['measured','A','E'])},
        all_three_flat_n=len(strict),all_three_flat_medians_ms={label:med([a[j]['delay_ms'] for a in strict]) for j,label in enumerate(['measured','A','E'])},
        paired_model_minus_measurement_ms={label:med([a[j]['delay_ms']-a[0]['delay_ms'] for a in ps]) for j,label in [(1,'A'),(2,'E')]})
tests={}
xx=np.arange(300,dtype=float);i=np.clip((xx-50)/100,0,1);f=np.clip((xx-100)/100,0,1)
tests['rising_known_130ms']=abs(crossing(f,.9,1,0,300)-crossing(i,.1,1,0,300)-130)<1e-9
tests['falling_known_130ms']=abs(crossing(1-f,.1,-1,0,300)-crossing(1-i,.9,-1,0,300)-130)<1e-9
tests['force_offset_invariance']=abs(crossing(f+43,43.9,1,0,300)-crossing(f,.9,1,0,300))<1e-9
tests['no_final_test_input']=all(x['split'] in ['train','validation'] for x in r['inputs'])
assert all(tests.values())
result=dict(summary=summary,conditions=tables,checks=tests,
    interpretation='Measured-eligible common events; model endpoint flags retained; strict all-three-flat subset also reported.')
(P/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
with (P/'clean_condition_comparison.csv').open('w',encoding='utf-8-sig',newline='') as fh:
    writer=csv.DictWriter(fh,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)

md=['# v20 电流 10% → 力 90% 时延重新统计','',
'## 定义与范围','',
'`delay = t(F0 + 0.9*(F1-F0)) - t(I0 + 0.1*(I1-I0))`。下降阶跃同样按完成变化量计算，即终点阈值为 F0 向 F1 变化的 90%，不是绝对力幅值的 90%。',
'',
'读取 33 个 ST 文件（30 训练 + 3 验证），没有读取最终测试 CSV。其他 SH/SR/RLF/RHF 不在此次阶跃统计中；没有可靠电流平台和力稳态时，不能直接套用阶跃 10%→90% 定义。',
'',
'文件不含独立的目标电流命令。使用实测电流幅值的阶跃后平台中位数代理目标值，未把名义 1.6A 强加给各个阶梯。时间轴中 F、I 同时前移 42 点，因此该共同平移不改变二者的相对时延。',
'',
'主窗口：陡变开始前 [−30,−10) ms 估计 I0/F0；最后陡变点后 [30,80) ms 估计 I1/F1。用首个定向越阈点，线性插值，信号不平滑，不是进入并保持误差带的稳定时间。1kHz 数据不应把插值小数解读成亚毫秒测量精度。',
'',
'电流差分 >0.04A/点检测陡变，相邻候选间隔≤40ms合并；阶跃幅度至少0.3A、力变化至少30N。候选剔除原因保存在 delay_10_90.json。',
'',
'## 质量筛选','',
'从全体 320 个可计算实测事件中保留122个：上升55、下降67。排除速度换向（窗口中同时出现 v<−10 与 v>10 mm/s）、前后速度变化超过 max(10mm/s,10%幅值)、低速、平台不平、力早于电流阈值。平台不平判据：半窗中位数漂移 > max(10%阶跃幅度, I:0.02A/F:10N)，或 P90−P10 > max(20%阶跃幅度,I:0.04A/F:20N)。所有事件及标记均保留，没有将负时延截断成0。',
'',
'筛选是分析用的可审查规则，不等于已证明速度、位置及摩擦影响完全消失。平台值仍是窗口估计，不是另做静态试验得到的真稳态值。',
'',
'## 总体（筛选后实测事件）','',
'| 方向 | 事件数 | 中位数 ms | 均值 ms | P10–P90 ms |',
'|---|---:|---:|---:|---:|']
for d in ['up','down']:
    x=summary[d]['measured'];md.append(f"| {'上升' if d=='up' else '下降'} | {x['n_crossed']} | {fmt(x['median_ms'])} | {fmt(x['mean_ms'])} | {fmt(x['p10_ms'])}–{fmt(x['p90_ms'])} |")
md+=['','## 各工况及 A/E 同事件对照','',
'A/E 均为 v20 seed7 best（A epoch126，E epoch127），使用 checkpoint 自带归一化，以实测历史推理，没有重新训练。此处入选事件都来自训练文件；3个验证文件的匹配阶跃全部发生速度换向，不能作为隔离电流响应的验证证据。不要将此表称为独立测试成绩。',
'',
'模型各用自身的 F0/F1 算相对响应时间，故并不同时检验稳态幅值误差。按实测质量筛选后逐事件配对；A/E 自身平台不平等标记在 CSV 列中保留。另在 summary.json 提供三条曲线同时通过全部判据的更严格子集。',
'',
'| 工况 | 方向 | 实测n | 实测中位数 | 配对n | 配对实测 | A | E |',
'|---|---|---:|---:|---:|---:|---:|---:|']
for row in tables:
    md.append('| '+row['condition']+' | '+('↑' if row['direction']=='up' else '↓')+' | '+str(row['n_measured'])+' | '+fmt(row['measured_median_ms'])+' | '+str(row['n_paired'])+' | '+fmt(row['paired_measured_ms'])+' | '+fmt(row['A_ms'])+' | '+fmt(row['E_ms'])+' |')
md+=['','所有时延单位 ms。单事件行不能代表稳定的工况统计。',
'','## 原先三个验证文件的问题','',
'相同有效区间里三曲线可配对45个事件（上升21、下降24），全部有速度换向。实测上/下中位数22.98/26.74ms；A为21.88/27.17ms；E为21.48/25.57ms。这些只能称电流与运动共同作用的表观10%→90%时延。旧脚本“50ms内速度只变约4mm/s”不适用于这些阶跃：在事件处速度由约+120变−120mm/s，或相反。',
'',
'![换向混杂示例](threshold_examples.png)',
'',
'## 稳态窗口敏感性','',
'后窗口 [20,50)、[30,80)、[50,100) ms，筛选后实测上升中位数分别22.88、22.81、22.46ms；下降25.48、26.02、27.13ms。各窗口合格样本不同，不能当作严格配对敏感性。逐事件数据均已保存。',
'',
'## 文件','',
'- `clean_condition_comparison.csv`：筛选后的逐工况及 A/E 同事件表。',
'- `events.csv`：全部主窗口事件、阈值时刻、前后平台和质量标记。',
'- `delay_10_90.json`：全部事件、窗口敏感性、剔除原因和输入哈希。',
'- `summary.json`：汇总、严格共同子集和算法检查。',
'- `predictions/manifest.json`：A/E权重、源码、预测文件哈希；原始训练文件与权重未修改。',
'- `analyze_delay.py` / `predict_st.py` / `report_delay.py`：复现脚本。',
'',
'没有将时延设为训练目标，没有修改正模型、采集对齐或旧结果。']
(P/'README.md').write_text('\n'.join(md)+'\n',encoding='utf-8')

# Plot sufficiently populated conditions only, all values from identical events.
keys=[k for k in sorted(pg) if len(pg[k])>=10]
fig,ax=plt.subplots(figsize=(12,5))
x=np.arange(len(keys));width=.24
for j,(lab,col) in enumerate([('Measured','black'),('A','#d95f02'),('E','#1b9e77')]):
    means=[med([a[j]['delay_ms'] for a in pg[k]]) for k in keys]
    ax.bar(x+(j-1)*width,means,width,label=lab,color=col,alpha=.8)
ax.set_xticks(x,[k[0].replace('ST_','').replace('_step5_0_1.6A','')+'\n'+k[1]+f' (n={len(pg[k])})' for k in keys],fontsize=8)
ax.set_ylabel('Median I10-to-F90 response time (ms)');ax.legend()
ax.set_title('Same measured-eligible events | v20 A/E seed 7 | training files\nRelative endpoint thresholds; model endpoint flags retained in tables')
ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(P/'condition_comparison.png',dpi=180)
print(json.dumps(summary,ensure_ascii=False,indent=2))
