"""Within-stroke/signed-speed response time versus current step amplitude.
Uses existing event extraction, not new thresholds selected for a trend.
"""
from pathlib import Path
from collections import defaultdict
import json,csv,re,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
OUT=P/'delta_i_trend';OUT.mkdir(exist_ok=True)
source=P/'delay_10_90.json';data=json.loads(source.read_text(encoding='utf-8'))
allrows=data['events'];idx={(a['file'],a['event'],a['curve']):a for a in allrows}
selected=[];excluded=[]
levels=np.array([.32,.64,.96,1.28,1.60])
for a in allrows:
    if a['curve']!='measured' or not re.match(r'ST_50mm_Vel0\.(13|26)_step5_0_1\.6A_r[123]$',a['file']):continue
    reasons=list(a['quality_flags'])
    v=(a['v0_mm_s']+a['v1_mm_s'])/2
    center=120 if 'Vel0.13_' in a['file'] else 240
    if abs(abs(v)-center)>.15*center:reasons.append('not_in_actual_speed_band')
    zero=a['I0_A'] if a['direction']=='up' else a['I1_A']
    if abs(zero)>.05:reasons.append('nonzero_common_endpoint')
    delta=abs(a['I1_A']-a['I0_A']);level=float(levels[np.argmin(abs(levels-delta))])
    if abs(level-delta)>.04:reasons.append('outside_current_level_tolerance')
    if reasons:
        excluded.append(dict(file=a['file'],event=a['event'],reasons=reasons));continue
    row=dict(file=a['file'],repeat=int(a['file'][-1]),event=a['event'],direction=a['direction'],
             signed_speed_bin_mm_s=int(np.sign(v)*center),actual_v_mm_s=v,
             nominal_deltaI_A=level,actual_deltaI_A=delta,I0_A=a['I0_A'],I1_A=a['I1_A'],
             delay_ms=a['delay_ms'],split=a['split'])
    for lab in ['A','E']:
        other=idx.get((a['file'],a['event'],lab+'_all_ST'))
        row[lab+'_delay_ms']=other['delay_ms'] if other else None
        row[lab+'_quality_flags']=other['quality_flags'] if other else ['no_event']
    selected.append(row)

groups=defaultdict(list)
for a in selected:groups[(a['signed_speed_bin_mm_s'],a['direction'],a['nominal_deltaI_A'])].append(a)
summary=[]
for k,rows in sorted(groups.items()):
    ds=[a['delay_ms'] for a in rows]
    summary.append(dict(signed_speed_mm_s=k[0],direction=k[1],deltaI_level_A=k[2],n=len(rows),
                        median_ms=float(np.median(ds)),min_ms=min(ds),max_ms=max(ds),
                        actual_v_min_mm_s=min(a['actual_v_mm_s'] for a in rows),
                        actual_v_max_mm_s=max(a['actual_v_mm_s'] for a in rows),
                        actual_deltaI_mean_A=float(np.mean([a['actual_deltaI_A'] for a in rows])),
                        A_median_ms=float(np.median([a['A_delay_ms'] for a in rows if a['A_delay_ms'] is not None])) if any(a['A_delay_ms'] is not None for a in rows) else None,
                        E_median_ms=float(np.median([a['E_delay_ms'] for a in rows if a['E_delay_ms'] is not None])) if any(a['E_delay_ms'] is not None for a in rows) else None))
trends=[]
for v in [120,-120,240,-240]:
    for direction in ['up','down']:
        g=sorted([a for a in summary if a['signed_speed_mm_s']==v and a['direction']==direction],key=lambda a:a['deltaI_level_A'])
        if len(g)<2:continue
        # Match adjacent levels within a repeated file, avoiding differences in file composition.
        pairs=[]
        for left,right in zip(g[:-1],g[1:]):
            for file in sorted(set(a['file'] for a in selected)):
                l=[a for a in selected if a['file']==file and a['signed_speed_bin_mm_s']==v and a['direction']==direction and a['nominal_deltaI_A']==left['deltaI_level_A']]
                rr=[a for a in selected if a['file']==file and a['signed_speed_bin_mm_s']==v and a['direction']==direction and a['nominal_deltaI_A']==right['deltaI_level_A']]
                if l and rr:pairs.append(dict(file=file,low_deltaI=left['deltaI_level_A'],high_deltaI=right['deltaI_level_A'],delay_increase_ms=rr[0]['delay_ms']-l[0]['delay_ms']))
        trends.append(dict(speed=v,direction=direction,n_levels=len(g),strictly_increasing_medians=all(b['median_ms']>a['median_ms'] for a,b in zip(g,g[1:])),
                           within_file_adjacent_pairs=pairs,positive_pairs=sum(a['delay_increase_ms']>0 for a in pairs),total_pairs=len(pairs)))
result=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            protocol='Fixed 50mm stroke; actual signed speed within 15% of +/-120 or +/-240 mm/s; up: I0<=0.05A; down: I1<=0.05A; original event quality filters unchanged.',
            current_levels='Group to nearest 0.32A ladder with <=0.04A tolerance; use actual levels in event table.',
            events=selected,excluded=excluded,summary=summary,trends=trends)
(OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
for name,rows in [('events',selected),('summary',summary)]:
    with (OUT/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for row in rows:w.writerow({k:json.dumps(val) if isinstance(val,list) else val for k,val in row.items()})

plt.rcParams.update({'font.size':10})
fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
for ax,v in zip(axs.flat,[120,-120,240,-240]):
    for direction,color,marker in [('up','#1368aa','o'),('down','#da781b','s')]:
        gg=sorted([a for a in summary if a['signed_speed_mm_s']==v and a['direction']==direction],key=lambda a:a['deltaI_level_A'])
        x=np.array([a['deltaI_level_A'] for a in gg]);y=np.array([a['median_ms'] for a in gg]);low=np.array([a['min_ms'] for a in gg]);high=np.array([a['max_ms'] for a in gg])
        ax.errorbar(x,y,yerr=[y-low,high-y],fmt=marker+'-',color=color,label='0 -> I (up)' if direction=='up' else 'I -> 0 (down)',capsize=4,lw=1.7)
        for a in selected:
            if a['signed_speed_bin_mm_s']==v and a['direction']==direction:
                ax.scatter(a['actual_deltaI_A'],a['delay_ms'],s=14,color=color,alpha=.4,zorder=3)
        for a in gg:
            ax.annotate('n='+str(a['n']),(a['deltaI_level_A'],a['median_ms']),xytext=(0,-16 if direction=='up' else 9),textcoords='offset points',ha='center',fontsize=8,color=color)
    vv=[a['actual_v_mm_s'] for a in selected if a['signed_speed_bin_mm_s']==v]
    ax.set_title(f'v ~ {v:+d} mm/s (observed {min(vv):+.0f} to {max(vv):+.0f})')
    ax.grid(alpha=.25);ax.set_xticks([.64,.96,1.28,1.6]);ax.set_xlim(.54,1.70);ax.set_ylim(10,40)
    ax.legend(fontsize=8,loc='upper left')
for ax in axs[:,0]:ax.set_ylabel('I10 -> F90 response time (ms)')
for ax in axs[-1,:]:ax.set_xlabel('|delta I| (A; nominal ladder level)')
fig.suptitle('Measured response vs current step at matched signed speed | 50 mm stroke\nMarkers: median; whiskers: observed min-max; faint dots: individual events. Lines guide the eye.',fontsize=12)
fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'deltaI_delay_measured.png',dpi=180);fig.savefig(OUT/'deltaI_delay_measured.pdf');plt.close(fig)

# Same matched events, models displayed separately from measured trend.
fig,axs=plt.subplots(2,4,figsize=(16,7),sharex=True,sharey=True)
for col,v in enumerate([120,-120,240,-240]):
    for row,direction in enumerate(['up','down']):
        ax=axs[row,col];gg=sorted([a for a in summary if a['signed_speed_mm_s']==v and a['direction']==direction],key=lambda a:a['deltaI_level_A'])
        for lab,key,color in [('Measured','median_ms','black'),('A','A_median_ms','#d95f02'),('E','E_median_ms','#1b9e77')]:
            ax.plot([a['deltaI_level_A'] for a in gg],[a[key] for a in gg],'o-',label=lab,color=color)
        ax.set_title(f'v ~ {v:+d} mm/s | {direction}');ax.grid(alpha=.2);ax.set_xticks([.64,.96,1.28,1.6]);ax.set_xlim(.54,1.70)
        if col==0:ax.set_ylabel('Response time (ms)')
        if row==1:ax.set_xlabel('|delta I| (A)')
axs[0,0].legend(fontsize=9)
fig.suptitle('Fixed 50 mm stroke, same measured-eligible events | v20 seed 7 A / E\nTraining-file descriptive comparison; model endpoint flags retained in events.csv')
fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(OUT/'deltaI_delay_models.png',dpi=180);plt.close(fig)

text=['# 固定速度下 ΔI 与 10%→90% 时延的关系','',
'固定50mm行程；按实测有符号速度约 ±120、±240mm/s 分组（对应文件名 Vel0.13/0.26）。正负速度单独处理，未假定正号就是压缩或回弹。保留前次同一套平台/运动质量筛选，无为获得单调趋势重新调整阈值。',
'',
'上升只比较 I0≈0 → I1；下降只比较 I0 → I1≈0。下降时 ΔI 与初始电流同时改变，不能据此识别“独立于初始电流的 ΔI 效应”。横轴是电流阶梯名义幅值，CSV保留实际幅值。实际幅值约比名义低0.01–0.02A。',
'',
'| 实测速度组 mm/s | 方向 | |ΔI| A | n | 时延中位数 ms | 最小–最大 ms |',
'|---:|---|---:|---:|---:|---:|']
# Avoid pipe symbols inside a table cell.
text[-2]='| 实测速度组 mm/s | 方向 | ΔI幅值 A | n | 时延中位数 ms | 最小–最大 ms |'
for v in [120,-120,240,-240]:
    for d in ['up','down']:
        for a in sorted([a for a in summary if a['signed_speed_mm_s']==v and a['direction']==d],key=lambda a:a['deltaI_level_A']):
            text.append(f"| {v:+d} | {d} | {a['deltaI_level_A']:.2f} | {a['n']} | {a['median_ms']:.2f} | {a['min_ms']:.2f}–{a['max_ms']:.2f} |")
text+=['','## 同文件内的幅值比较','',
'为避免不同幅值恰好使用了不同重复文件，在同一个文件、同一速度方向、同一电流上/下方向内计算相邻可用幅值的时延差。这里只报告计数，不将同文件事件当作独立样本做显著性检验。','']
for a in trends:
    text.append(f"- v≈{a['speed']:+d} mm/s，{a['direction']}：{a['n_levels']}个幅值；各档中位数严格递增={a['strictly_increasing_medians']}；同文件相邻幅值比较 {a['positive_pairs']}/{a['total_pairs']} 次为正增量。")
text+=['','## 解释边界','',
'速度匹配是近似匹配，并非每点速度严格相同。除控制行程、速度方向及电流方向外，事件的位置、历史和阶梯顺序仍有差异。最多3个重复文件；一些档只有1个事件。0.32A档在固定50mm且通过原质量筛选的事件中没有可用数据，空缺不插补。',
'',
'结果用于描述本批数据的条件趋势，不能据此声称全局单调规律或唯一的 ΔI→时延函数，也不能将10%→90%响应时间直接当作一阶时间常数。三个验证ST文件受换向混杂，图中使用的是训练文件的实测数据；未读取最终测试。',
'',
'模型曲线使用之前已保存的 A/E 推理结果，无重新训练；质量标记在 events.csv 中保留。','',
'![实测](deltaI_delay_measured.png)','', '![模型对照](deltaI_delay_models.png)']
(OUT/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps(dict(events=len(selected),summary=summary,trends=trends),ensure_ascii=False,indent=2))
