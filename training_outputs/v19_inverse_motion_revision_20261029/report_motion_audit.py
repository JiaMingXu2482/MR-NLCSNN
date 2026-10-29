"""Report saved synthetic results; no model or dataset loading."""
import json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
R=HERE/'results'
d=json.loads((R/'summary.json').read_text()); w=json.loads((R/'windows.json').read_text())
oracle=json.loads((R/'oracle_refinement.json').read_text()); ow=json.loads((R/'oracle_windows.json').read_text())
names=w['case_names']; mm=d['methods']
def table(methods):
    text='| Method / case | Force MAE N | I MAE A | TV / true TV | Saturation |\n|---|---:|---:|---:|---:|\n'
    for m in methods:
        for n in names:
            v=mm[m][n]
            text+=f"| {m} / {n} | {v['force_error_N']['mae']:.4f} | {v['current_error_A']['mae']:.4f} | {v['TV_ratio']:.3f} | {100*v['saturation_fraction']:.1f}% |\n"
    return text
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
for j,(name,ax) in enumerate(zip(names,axes.flat)):
    t=[1+.01*k for k in range(100)]
    ax.plot(t,[r[j] for r in w['known_current_A']],color='black',lw=2,label='Known command')
    for key,color,label in [('old_frozen_window','#999999','Old frozen (teacher history)'),
        ('own_history_full_ca9_lambda0','#0072b2','Full CA grid (own history)'),
        ('own_history_full_ca9_lambda10_rate0p2','#d95f02','Full CA + penalty/rate (own history)')]:
        ax.plot(t,[r[j] for r in w['methods'][key]['current_A']],color=color,lw=.9,label=label)
    ax.set(title=name,xlabel='Time (s)',ylabel='Command current (A)'); ax.grid(alpha=.2)
axes[0,0].legend(fontsize=7)
fig.suptitle('Command oscillation audit; every curve and failed case retained')
fig.tight_layout(); fig.savefig(R/'current_comparison.png',dpi=170); plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
methods=['old_frozen_window','true_v_frozen_csb','ca_v_frozen_csb','full_true_motion_9','full_ca_motion_9']
labels=['Frozen','True v / frozen coefficients','CA v / frozen coefficients','True motion / full / 9','CA motion / full / 9']
for k,(m,label) in enumerate(zip(methods,labels)):
    for ax,metric in zip(axes,['force_error_N','current_error_A']):
        ax.plot(range(4),[mm[m][n][metric]['mae'] for n in names],marker='o',label=label)
for ax,title,ylabel in zip(axes,['Force error','Current error'],['MAE (N)','MAE (A)']):
    ax.set_xticks(range(4),['Low multi','Mixed multi','8 Hz','12 Hz']); ax.set(title=title,ylabel=ylabel); ax.grid(alpha=.2)
axes[0].legend(fontsize=7); fig.suptitle('Same-history ablation; oracle inputs are diagnosis only')
fig.tight_layout(); fig.savefig(R/'ablation_comparison.png',dpi=170); plt.close(fig)
initial_jump=[abs(v-.8) for v in w['known_current_A'][0]]
rate='own_history_full_ca9_lambda10_rate0p2'
first_effect={}
for j,n in enumerate(names):
    e=[w['methods'][rate]['actual_mean_N'][k][j]-w['target_mean_N'][k][j] for k in range(100)]
    first_effect[n]=dict(initial_command_change_A=initial_jump[j],first_window_force_error_N=e[0],remaining99_MAE_N=sum(abs(v) for v in e[1:])/99)
(R/'rate_initial_transition.json').write_text(json.dumps(first_effect,indent=2),encoding='utf-8')
oracle_rows='| Case | Force max residual N | I MAE A | TV ratio | Multi-root windows |\n|---|---:|---:|---:|---:|\n'
for n,v in oracle['summary'].items():
    oracle_rows+=f"| {n} | {v['force_error_N']['max_abs']:.3g} | {v['current_error_A']['mae']:.4f} | {v['TV_ratio']:.3f} | {v['multiple_brackets']}/100 |\n"
readme=f'''# 10 ms 逆解：运动、上下文、求根与电流平滑分解

2026-10-29。沿用提交36793e6的A权重及全部四组合成工况。仅新增实验代码与结果，未训练、未改任何权重/标签、未读数据集CSV、未替换CarSim/Simulink。

## 结论

用户的核心判断成立：旧10 ms冻结逆解的10–43 N误差不能作为自洽测试通过。完整正模型在已知电流处的目标100%可达，冻结代理的“不可达”是代理范围失真。

实测结果不支持“只改v+a*t就解决大部分问题”：真实未来速度只把四组MAE从10.67/43.12/7.47/10.75 N降至7.72/34.33/5.65/8.53 N。保留GRU滚动同样重要。

真实未来运动+完整模型+9点插值达到0.528/1.105/0.283/0.237 N，但真实未来运动在线不可得。进一步离线自适应网格找到全部400窗的根，最大力残差小于6e-6 N；这是自洽参照通过，不是可部署逆解通过。

**常加速度预测+完整滚动仍有明显误差。** 独立窗口同历史下1.33/16.50/2.82/6.55 N；携带自己实际电流与状态后4.83/41.26/3.91/11.02 N。只看前者会高估连续运行表现。

正则和限速必须独立验收。lambda=10 N/A、增量0.2 A/10ms使电流更平滑，但连续混合多正弦力MAE升至85.11 N，不可接受。这是实验平滑参数，不是新增执行器物理限幅。

## 口径和自洽参照

- 用户表里的4.8%/8.9%/11.8%/11.7%对应RMSE/目标力RMS，不是MAE/RMS；本报告以NRMSE明确命名。
- 目标由同一A正模型、同一已知历史、真实未来运动、已知保持电流生成。所有逆解器均未获得真实待求电流作为初值或候选；它只用于目标生成和评分。
- FP64/CPU单线程用于降低求根数值噪声，原FP32权重数值不变；与原FP32目标最大差为 {d['max_target_difference_from_original_FP32_N']:.6g} N。内部旧方案复算与先前10.67/43.12/7.47/10.75 N一致。
- 7项实现检查全部通过：非单调求根、真实电流处可达、完整序列与逐步提交一致、查询不改原状态/权重、平坦映射保留电流、限速生效。模型仍在候选副本上每1ms更新历史dI/GRU/CurrentLag。
- 四组各100窗，共400窗；没有弃掉高误差工况。完整结果、换向窗统计、灵敏度和饱和均在summary.json/windows.json。

## 第一阶段：相同已知历史，隔离近似误差

{table(['old_frozen_window','true_v_frozen_csb','ca_v_frozen_csb','full_true_motion_9','full_ca_motion_9'])}

`true_v_frozen_csb`只提供真实未来速度，c/s/b仍固定。`ca_v_frozen_csb`用v0+a0*t且仍固定系数。`full_ca_motion_9`完整推进GRU与电流历史，输入x0+v0*t+0.5*a0*t²、v0+a0*t、a0（及当前温度），无需未来测量。9点插值不是精确求根；`*_refined`结果也全部保留。

窗口运动在t0..t0+9ms取样，电流EMA对应第1..10次更新，避免错一拍。独立窗口方法使用教师历史，仅用于误差来源隔离。

## 第二阶段：连续携带自己的实际电流与状态

所有方法从相同1秒预热结束状态出发，每个10ms窗只用起点测得的x/v/a；实际正模型在外生真实运动下推进10次，成为下一窗自己的历史。这里仍是外生运动模型回放，不是车辆闭环。

{table(['own_history_full_ca9_lambda0','own_history_full_ca9_lambda2','own_history_full_ca9_lambda10','own_history_full_ca9_lambda10_rate0p2'])}

目标是绝对力误差(N)+lambda(N/A)*电流增量(A)。lambda=0/2/10在看到结果前固定，全数报告。0.2 A限速可能拦截首窗真实命令跳变；其四组初始跳变为 {', '.join(f'{x:.3f}' for x in initial_jump)} A，首窗与余下99窗分别在rate_initial_transition.json，不能把首窗影响隐藏。此试验不以四组合成结果选择最终lambda。

## 第三阶段：离线oracle补查网格漏根和多根

这一补查由第一阶段失败触发，预先登记在ORACLE_SUPPLEMENT.md，与原PLAN.md分开记录。9点网格有43窗没发现括区间；只对这些窗口逐级33/129/513点加密，剩余6/2/0窗。24次二分只在发现的局部括区间内进行，不假设全局单调。

{oracle_rows}

力残差在0.001 N数值等价带内时，选择已发现根中最接近上一实际电流的根，防止小数值差异把解切换到远根。这不是正则化力误差，也不保证找到所有根或全局最近根。8Hz的TV比1.554仍略高于1.5，因此不能宣称所有指标通过。

**相同窗均值不一定唯一确定电流。** 已找到同历史下的两个电流：8Hz第6窗，真实1.168625 A与另一解0.049520 A，均值力都约−118.790735 N，后者残差1.74e−8 N。两个解可有不同逐点波形和结束内部状态。此例既证明力误差能接近0，也说明电流恢复与分支连续性需要单独评价；不能把全滚动映射也当作严格单调。

## 验收与下一步

同意用户提出TV<=真实1.5倍、饱和<10%和力误差预算作为工程筛选，但它不替代oracle接近0的数值验收；7 N、10 N两档均报告，同时检查此前电流MAE<=0.1 A。

本次没有可宣布全面通过的在线方案。下一步重点是因果运动预测（尤其高频/换向时的加速度演化）和多根连续选择，而不是继续放宽力误差或靠更强限速掩盖不准确的力映射。新增网络/训练前应先有这两个环节的独立证据。

NI未实测。本次运行耗时是CPU/PyTorch FP64离线批处理，不能据此宣称9候选方案满足NI 10ms周期；计算量估算也不能替代目标机测量。

## 复现

权重沿用 `training_outputs/v19_capacity_four_20261028/training_delivery/sessions/A/final.pth`，SHA256 `{d['source_checkpoint_sha256']}`。只需要PyTorch；报告绘图另需matplotlib。冻结模型和旧基线依赖放在frozen_dependencies/，无训练/CSV加载入口。

```sh
python run_motion_audit.py /path/to/A/final.pth --output replay_results
python refine_oracle.py /path/to/A/final.pth --results replay_results
```

拒绝覆盖已有运行结果。results/executed_source/保留实际产生首轮及oracle指标时的源码；根目录脚本仅增添便携路径/命令行参数，模型与实验逻辑一致。哈希见SHA256.json与各summary的source字段。
'''
(HERE/'README.md').write_text(readme,encoding='utf-8')
manifest={str(p.relative_to(HERE)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and 'replay_results' not in p.parts and p.name not in ('SHA256.json','replay_stdout.txt')}
(HERE/'SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Report saved:',HERE/'README.md')

if __name__=='__main__': pass
