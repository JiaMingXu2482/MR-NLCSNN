"""Four parallel cold-start Stage-1 experiments; validate only after all finish.

Requires an explicit frozen experiment config and --execute. No final-test path.
"""
import argparse,collections,json,math,os,random,subprocess,sys,time,traceback
from pathlib import Path
import train_a_local as base
import train_bd_local as bd
import torch
from model_v19_capacity import CapacityModel

HERE=Path(__file__).resolve().parent
ARMS={'D':(16,48),'A':(16,48),'A_GRU':(32,48),'A_MLP':(16,96)}
PROPORTIONS={'SH':.25,'ST':.15,'SR':.20,'RLF':.20,'RHF':.20}

def setup():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')

def build(arm):
    h,m=ARMS[arm]
    return bd.NLCSNN_v19_BD('D') if arm=='D' else CapacityModel(gru_hidden=h,mlp_hidden=m,history_di=True)

def sample(it,norm,arm):
    return bd.sample(it,norm,'D','cuda') if arm=='D' else base.sample(it,norm,'A','cuda')

def load(cfg,partition):
    if partition not in ('train','validation'): raise ValueError('Forbidden partition')
    data=Path(cfg['data_path']); assert base.digest(data/'alignment_manifest.json')==cfg['data_manifest_sha256']
    items=base.upstream.read_dataset(data,partition)
    assert len(items)==cfg[partition+'_files']
    return items

def schedule(names,total):
    counts={c:int(total*p) for c,p in PROPORTIONS.items()}
    for c in sorted(counts,key=lambda c:-(total*PROPORTIONS[c]-counts[c]))[:total-sum(counts.values())]: counts[c]+=1
    labels=[c for c,n in counts.items() for _ in range(n)]; random.Random(7).shuffle(labels)
    pools={c:[n for n in names if n.split('_')[0]==c] for c in counts}
    assert all(pools.values())
    rng={c:random.Random(700+i) for i,c in enumerate(counts)}; cycles={}; at={}; result=[]
    for c in labels:
        if c not in cycles or at[c]==len(cycles[c]): cycles[c]=list(pools[c]); rng[c].shuffle(cycles[c]); at[c]=0
        result.append(cycles[c][at[c]]); at[c]+=1
    assert len(result)==total and set(result)==set(names)
    return result

def hashes(cfgpath):
    paths=[cfgpath,Path(__file__),HERE/'model_v19_capacity.py',HERE/'model_v19_bd.py',HERE/'train_a_local.py',HERE/'train_bd_local.py',
        base.SOURCE/'AA02Train/train_v19.py',base.SOURCE/'mr_nlcsnn/model_v19.py',base.SOURCE/'mr_nlcsnn/v19_protocol.py']
    return {str(p):base.digest(p) for p in paths}

def train(cfgpath,cfg,arm,resume):
    setup(); random.seed(7); torch.manual_seed(7); torch.cuda.manual_seed_all(7)
    root=Path(cfg['session_root']); folder=root/arm; total=cfg['epochs']*cfg['train_files']; h=hashes(cfgpath)
    if (folder/'TRAIN_COMPLETE.json').exists():
        assert resume; print(arm+' already trained'); return
    items=load(cfg,'train'); by_name={it['name']:it for it in items}; norm=base.upstream.normalization(items)
    order=json.loads((root/'schedule.json').read_text()); assert len(order)==total and set(order)==set(by_name)
    schedule_sha=base.digest(root/'schedule.json'); model=build(arm).cuda(); params=base.set_stage(model,1)
    opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-4); history=[]; seen=0; start=0
    def checkpoint(step):
        return dict(architecture=model.architecture,state_dict=model.state_dict(),model_cfg=model.config,
            arm=arm,stage=1,norm_cfg=norm,optimizer_state_dict=opt.state_dict(),step=step,history=history,
            seen_scored_samples=seen,torch_rng_state=torch.get_rng_state(),cuda_rng_states=torch.cuda.get_rng_state_all(),
            source_hashes=h,schedule_sha256=schedule_sha,total_updates=total,data_manifest_sha256=cfg['data_manifest_sha256'])
    if folder.exists():
        if not resume: raise FileExistsError(folder)
        s=torch.load(folder/'latest.pth',map_location='cuda',weights_only=True)
        assert s['source_hashes']==h and s['norm_cfg']==norm and s['schedule_sha256']==schedule_sha and s['total_updates']==total
        model.load_state_dict(s['state_dict']); opt.load_state_dict(s['optimizer_state_dict'])
        start=s['step']; history=s['history']; seen=s['seen_scored_samples']
        torch.set_rng_state(s['torch_rng_state'].cpu()); torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng_states']])
        (folder/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf-8')
    else:
        folder.mkdir()
        base.write_json(folder/'run_config.json',dict(experiment=cfg,arm=arm,model_cfg=model.config,
            parameter_count=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in params),
            source_hashes=h,schedule_sha256=schedule_sha,cold_start=True,norm_cfg=norm,
            input_protocol='D: x,v,I,T; A arms: x,v,a,I,T plus strictly past dI=(I[t-1]-I[t-2])/dt normalized by1600 A/s',
            validation_policy='No validation during any arm training; barrier after all four TRAIN_COMPLETE',
            hardware=torch.cuda.get_device_name(0),precision='FP32; deterministic; no AMP/TF32',cpu_threads=1))
        base.save_checkpoint(folder/'latest.pth',checkpoint(0))
    model.train(); losses=[]; gradmax=0.; began=time.perf_counter()
    print(f'START {arm} step={start}/{total} parameters={sum(p.numel() for p in model.parameters())}',flush=True)
    for step in range(start+1,total+1):
        epoch=(step-1)//cfg['train_files']+1
        factor=min(epoch/5,1) if epoch<=5 else .01+.99*(1+math.cos(math.pi*(epoch-5)/(cfg['epochs']-5)))/2
        for group in opt.param_groups: group['lr']=3e-4*factor
        it=by_name[order[step-1]]; u,f,valid=sample(it,norm,arm); opt.zero_grad(set_to_none=True)
        loss=base.forward_loss(model(u),f,valid,(f.max()-f.min()).reshape(1),f.new_tensor([3 if it['name'].startswith('ST_') else 1]))
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss: '+it['name'])
        loss.backward(); gn=float(torch.nn.utils.clip_grad_norm_(params,1))
        if not math.isfinite(gn): raise ValueError('Nonfinite gradient')
        opt.step(); losses.append(float(loss.detach())); gradmax=max(gradmax,gn); seen+=int(valid.sum())
        if step%100==0 or step==start+1:
            base.write_json(folder/'progress.json',dict(status='training',arm=arm,epoch=epoch,updates=step,total_updates=total,validation_performed=False))
        if step%cfg['train_files']==0:
            record=dict(epoch=epoch,updates=step,training_loss=sum(losses)/len(losses),max_gradient_norm=gradmax,
                seconds=time.perf_counter()-began,lr=opt.param_groups[0]['lr'],seen_scored_samples=seen,tau_ms=float(model.lag.tau_ms().detach()))
            history.append(record); base.save_checkpoint(folder/'latest.pth',checkpoint(step))
            with (folder/'metrics.jsonl').open('a',encoding='utf-8') as fstream: fstream.write(json.dumps(record)+'\n')
            if epoch%10==0: print(f'{arm} E{epoch}/{cfg["epochs"]} loss={record["training_loss"]:.7f}',flush=True)
            losses=[]; gradmax=0.; began=time.perf_counter()
    final=checkpoint(total); final.pop('optimizer_state_dict')
    base.save_checkpoint(folder/'final_unvalidated.pth',final)
    base.write_json(folder/'TRAIN_COMPLETE.json',dict(status='trained_not_validated',updates=total,epochs=cfg['epochs'],
        final_sha256=base.digest(folder/'final_unvalidated.pth'),seen_scored_samples=seen,training_seconds=sum(r['seconds'] for r in history)))
    base.write_json(folder/'progress.json',dict(status='waiting_for_all_arms',updates=total,validation_performed=False))

def evaluate_all(cfgpath,cfg):
    setup(); root=Path(cfg['session_root'])
    assert all((root/a/'TRAIN_COMPLETE.json').exists() for a in ARMS)
    items=load(cfg,'validation')
    for arm in ARMS:
        folder=root/arm
        if (folder/'COMPLETE.json').exists(): continue
        s=torch.load(folder/'final_unvalidated.pth',map_location='cpu',weights_only=True)
        assert s['source_hashes']==hashes(cfgpath)
        model=build(arm).cuda().eval(); model.load_state_dict(s['state_dict'])
        metrics=evaluate_once(model,items,s['norm_cfg'],arm,cfg)
        groups=metrics['verified_primary']['category_nrmse_pct']
        base.write_json(folder/'validation_metrics.json',metrics)
        base.write_json(folder/'COMPLETE.json',dict(status='complete',category_nrmse_pct=groups,pooled_nrmse_pct=metrics['pooled_nrmse_pct'],
            hf_nrmse_pct=metrics['hf_nrmse_pct'],random_macro_nrmse_pct=sum(groups[c] for c in ('SR','RLF','RHF'))/3,
            primary_metric_scope='15 verified initial-static-reference validation files',
            verified_primary_pooled_nrmse_pct=metrics['verified_primary']['pooled_nrmse_pct'],
            secondary_all16_pooled_nrmse_pct=metrics['pooled_nrmse_pct'],
            final_test_read=False,parameter_count=sum(p.numel() for p in model.parameters())))
        s['metrics']=metrics; base.save_checkpoint(folder/'final.pth',s)
        del model; torch.cuda.empty_cache()
    base.write_json(root/'COMPLETE.json',dict(status='complete',arms=list(ARMS),final_test_read=False))

@torch.no_grad()
def evaluate_once(model,items,norm,arm,cfg):
    rows=[]; folder=Path(cfg['session_root'])/arm; pred_dir=folder/'validation_predictions'; pred_dir.mkdir(exist_ok=True)
    for it in items:
        u,y,valid=sample(it,norm,arm); pred=model(u)
        err=((pred-y)*norm['f_scale'])[valid].double(); truth=(y*norm['f_scale'])[valid].double()
        speed=(u[...,1].abs()*1000)[valid].double()
        row=dict(name=it['name'],zero_reference_verified=it['name'] not in cfg['secondary_unverified_validation'],
            **base.summarize(float(err.square().sum()),float(truth.square().sum()),float(err.abs().sum()),int(valid.sum())))
        bins=[]
        for lo,hi in base.BINS:
            mask=speed>=lo
            if hi is not None: mask &= speed<hi
            e,t=err[mask],truth[mask]
            bins.append(base.summarize(float(e.square().sum()),float(t.square().sum()),float(e.abs().sum()),int(mask.sum())))
        row['speed_bins']=bins; rows.append(row)
        base.write_json(pred_dir/(it['name']+'.json'),dict(name=it['name'],dt_s=.001,guard=base.upstream.edge_guard(len(it['values'])),
            zero_reference_verified=row['zero_reference_verified'],measured_force_N=it['values'][:,6].tolist(),
            predicted_force_N=(pred[0]*norm['f_scale']).cpu().tolist(),velocity_mm_s=it['values'][:,1].tolist()))
    def aggregate(chosen):
        def pool(rr): return 100*math.sqrt(sum(r['SSE'] for r in rr)/sum(r['SST'] for r in rr))
        bins=[]
        for j,(lo,hi) in enumerate(base.BINS):
            bb=[r['speed_bins'][j] for r in chosen]
            bins.append(dict(min_abs_v_mm_s=lo,max_abs_v_mm_s=hi,**base.summarize(sum(b['SSE'] for b in bb),sum(b['SST'] for b in bb),
                sum(b['mae_N']*b['samples'] for b in bb if b['samples']),sum(b['samples'] for b in bb))))
        return dict(files=len(chosen),pooled_nrmse_pct=pool(chosen),hf_nrmse_pct=pool([r for r in chosen if r['name'].startswith('SH_') and '6.62Hz' in r['name']]),
            category_nrmse_pct={c:pool([r for r in chosen if r['name'].startswith(c+'_')]) for c in PROPORTIONS},speed_bins=bins)
    verified=[r for r in rows if r['zero_reference_verified']]; assert len(verified)==cfg['verified_validation_files']
    return dict(**aggregate(rows),rows=rows,verified_primary=aggregate(verified),
        unverified_secondary=[r for r in rows if not r['zero_reference_verified']],history_protocol='strict-past current; known measured motion history',final_test_read=False)

def manage(cfgpath,cfg,resume):
    root=Path(cfg['session_root'])
    if root.exists():
        if not resume: raise FileExistsError(root)
        assert json.loads((root/'experiment_config.json').read_text())==cfg
        if (root/'COMPLETE.json').exists(): return
    else:
        items=load(cfg,'train'); order=schedule([it['name'] for it in items],cfg['epochs']*len(items))
        root.mkdir(parents=True); base.write_json(root/'experiment_config.json',cfg); base.write_json(root/'schedule.json',order)
    children=[]
    try:
        for arm in ARMS:
            if (root/arm/'TRAIN_COMPLETE.json').exists(): continue
            logfile=(root/(arm+'_console.log')).open('ab')
            args=[sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(cfgpath),'--execute','--worker',arm]
            if resume: args.append('--resume')
            p=subprocess.Popen(args,stdout=logfile,stderr=subprocess.STDOUT,cwd=str(HERE),creationflags=subprocess.CREATE_NO_WINDOW)
            logfile.close(); children.append((arm,p)); print(f'LAUNCHED {arm} pid={p.pid}',flush=True)
        base.write_json(root/'processes.json',dict(manager_pid=os.getpid(),workers={a:p.pid for a,p in children}))
        while any(p.poll() is None for _,p in children): time.sleep(15)
        failed={a:p.returncode for a,p in children if p.returncode}
        if failed: raise RuntimeError('Worker failures; inspect logs: '+str(failed))
        evaluate_all(cfgpath,cfg)
    except BaseException:
        base.write_json(root/'ERROR.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True); parser.add_argument('--execute',action='store_true'); parser.add_argument('--resume',action='store_true'); parser.add_argument('--worker',choices=list(ARMS))
    args=parser.parse_args()
    if not args.execute: raise SystemExit('--execute required')
    cfgpath=args.config.resolve(); cfg=json.loads(cfgpath.read_text()); assert cfg['epochs']==300 and cfg['data_review_approved'] is True
    if args.worker: train(cfgpath,cfg,args.worker,args.resume)
    else: manage(cfgpath,cfg,args.resume)
if __name__=='__main__': main()
