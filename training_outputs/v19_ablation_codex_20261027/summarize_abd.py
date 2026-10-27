"""Build a separate A/B/D report from completed sessions only."""
import csv, hashlib, json, shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
RUNTIME=Path(r'C:\Users\user\AppData\Local\Temp\v19_bd_runtime_20261027')
ARMS=('A','B','D')
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(p,rows):
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    out=HERE/'results_summary_ABD'
    if out.exists(): raise FileExistsError(out)
    details={}; summaries=[]; rows=[]; bins=[]
    for arm in ARMS:
        folder=HERE/'sessions'/f'v19_{arm}_seed7_30epochs'
        config=read(folder/'run_config.json'); complete=read(folder/'COMPLETE.json')
        metrics=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()]
        assert [m['epoch'] for m in metrics]==list(range(1,31))
        best=min(metrics,key=lambda m:m['pooled_nrmse_pct'])
        assert best['epoch']==complete['best_epoch']
        assert best['pooled_nrmse_pct']==complete['best_pooled_nrmse_pct']
        assert sha(folder/'best.pth')==complete['best_checkpoint_sha256']
        assert len(best['rows'])==16 and len(best['speed_bins'])==4
        for path,digest in config['source_hashes'].items(): assert sha(Path(path))==digest,path
        timing=read(folder/'pc_cpu_timing.json')
        details[arm]=dict(config=config,best=best,complete=complete,timing=timing)
        summaries.append(dict(arm=arm,epochs=30,best_epoch=best['epoch'],NFL=config['nfl_dim'],
            parameters=config['parameter_count'],stage1_trainable_parameters=config['trainable_parameter_count'],
            pooled_nrmse_pct=best['pooled_nrmse_pct'],hf_pooled_nrmse_pct_at_best=best['hf_nrmse_pct'],
            learned_tau_ms_at_best=best['tau_ms'],
            pc_cpu_one_corner_mean_ms=timing['one_corner']['mean_ms'],
            pc_cpu_one_corner_p95_ms=timing['one_corner']['p95_ms'],
            pc_cpu_four_corners_mean_ms=timing['four_corners_sequential']['mean_ms'],
            pc_cpu_four_corners_p95_ms=timing['four_corners_sequential']['p95_ms'],
            sum_epoch_seconds=sum(m['seconds'] for m in metrics)))
        rows.extend(dict(arm=arm,**r) for r in best['rows'])
        bins.extend(dict(arm=arm,**r) for r in best['speed_bins'])
    # Only metadata/path existence checks; never read final-test CSV content.
    audit=read(RUNTIME/'preparation_audit.json')
    assert all(not (RUNTIME/'pack'/p).exists() for p in audit['sealed_final_test_members'])
    out.mkdir()
    write_csv(out/'summary.csv',summaries); write_csv(out/'per_file_errors.csv',rows); write_csv(out/'speed_bin_errors.csv',bins)
    (out/'summary.json').write_text(json.dumps(details,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    text=['# v19 A / B / D 本地输入消融结果','',
        'A 为用户追加授权后的本地运行。B、D 复用已完成结果，未重训、未改动旧会话。',
        '三臂均为 seed 7 冷启动、Stage 1、30轮、259训练文件和16验证文件；最终测试10个CSV未解压或打开。',
        'A: x/v/a/I/dI/T，B: x/v/I/dI/T，D: x/v/I/T；分别为35/27/24维NFL。',
        '三臂 observer 的 dI 均为0。B与D的区别包括死通道去留；本文不作差异原因推断。',
        '所有训练使用本机 RTX 4060 Ti、FP32，无 AMP/TF32。相同 AdamW、学习率和余弦调度、文件顺序种子、归一化及loss设置详见各会话run_config。',
        '各臂选择验证集 pooled NRMSE 最小的 checkpoint，高频、逐文件及速度区间指标均取该 checkpoint。',
        'NRMSE = 100 × sqrt(Σ(F预测−F实测)² / ΣF实测²)；pooled先汇总分子、分母，不是逐文件百分比平均。训练loss另按逐文件p2p归一化。','',
        '|臂|NFL|总参数量|Stage 1可训练参数|最佳轮|验证pooled NRMSE (%)|6.62Hz验证pooled (%)|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        text.append(f'|{r["arm"]}|{r["NFL"]}|{r["parameters"]}|{r["stage1_trainable_parameters"]}|{r["best_epoch"]}|{r["pooled_nrmse_pct"]:.6f}|{r["hf_pooled_nrmse_pct_at_best"]:.6f}|')
    text+=['','## 逐文件验证误差','', '|文件|A NRMSE (%)|B NRMSE (%)|D NRMSE (%)|','|---|---:|---:|---:|']
    by={a:{r['name']:r for r in details[a]['best']['rows']} for a in ARMS}
    assert set(by['A'])==set(by['B'])==set(by['D'])
    for name in by['A']:
        text.append('|'+name+'|'+'|'.join(f'{by[a][name]["nrmse_pct"]:.6f}' for a in ARMS)+'|')
    text+=['','RMSE (N)、MAE (N)、有效样本数及SSE/SST详见 per_file_errors.csv。','',
        '## 分速度区间','', '速度为 |v|，单位mm/s；统计去掉首尾guard后的验证样本。','',
        '|臂|速度区间|样本数|RMSE (N)|MAE (N)|NRMSE (%)|','|---|---|---:|---:|---:|---:|']
    for r in bins:
        upper=r['max_abs_v_mm_s'] if r['max_abs_v_mm_s'] is not None else '+∞'
        text.append(f'|{r["arm"]}|[{r["min_abs_v_mm_s"]},{upper})|{r["samples"]}|{r["rmse_N"]:.6f}|{r["mae_N"]:.6f}|{r["nrmse_pct"]:.6f}|')
    text+=['','## 本地 PC CPU 推理计时','',
        'Python/PyTorch、单线程、FP32；预热100次后计时1000次。仅测正向 prepare + force_at + commit，不含逆头。',
        '这是 PC 值，不是 S32 或 NI 实测；未作编译部署实时性认证。','',
        '|臂|单角均值 (ms)|单角p95 (ms)|四角串行均值 (ms)|四角串行p95 (ms)|','|---|---:|---:|---:|---:|']
    for r in summaries:
        text.append(f'|{r["arm"]}|{r["pc_cpu_one_corner_mean_ms"]:.6f}|{r["pc_cpu_one_corner_p95_ms"]:.6f}|{r["pc_cpu_four_corners_mean_ms"]:.6f}|{r["pc_cpu_four_corners_p95_ms"]:.6f}|')
    text+=['','## 交付与复核','',
        '- sessions 下为三个独立会话，包含 run_config、30轮metrics、best/latest checkpoint和计时记录。',
        '- 这些是Stage 1正向训练结果，逆头未训练。',
        '- code_snapshot 为实际代码、补丁、G0、划分和检查记录副本；未推送Git分支。',
        '- 基础提交：cac96e8f85491e731a0030c268f8c9f814b06953。',
        '- manifest SHA256：0766763e7fb77880ffa04f2d93bcf629bfeb817e51c571188eee1ea83262e2ff，301项。',
        '- runner修复后原16项测试和B/D新增4项测试已通过；A追加5项集成检查通过，模型参数与原版相同seed初始化逐项一致。',
        '- run_config绝对路径保留本次运行溯源。代码副本对应这些路径，但重定位时需调整runner的数据/代码根路径。','']
    (out/'README.md').write_text('\n'.join(text),encoding='utf-8')
    shutil.copytree(HERE/'results_summary/code_snapshot',out/'code_snapshot')
    for name in ('train_a_local.py','check_a_local.py','a_runner_diff.txt','prepare_a_local.py'):
        shutil.copyfile(HERE/name,out/'code_snapshot/local'/name)
    shutil.copyfile(HERE/'a_preflight_checks.json',out/'code_snapshot/a_preflight_checks.json')
    print(json.dumps(summaries,indent=2))
    print(out/'README.md')

if __name__=='__main__': main()
