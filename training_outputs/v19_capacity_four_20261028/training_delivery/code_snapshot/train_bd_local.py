"""Local B/D Stage-1-only runner. No final-test loading, no A, no warm-start.
Uses Muse-patched source for data/sample/loss; separate files and sessions only.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,csv,hashlib,json,math,platform,random,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
RUNTIME=Path(r'C:\Users\user\AppData\Local\Temp\v19_bd_runtime_20261027')
SOURCE=RUNTIME/'source'
DATA=RUNTIME/'pack/Parsed_v19_F42_I42'
G0=RUNTIME/'pack/training_outputs/v19_g0/g0_report_F42_I42.json'
EXPECTED='0766763e7fb77880ffa04f2d93bcf629bfeb817e51c571188eee1ea83262e2ff'
sys.path.insert(0,str(SOURCE))
import torch
from AA02Train import train_v19 as upstream
from mr_nlcsnn.v19_protocol import set_stage,forward_loss
from model_v19_bd import NLCSNN_v19_BD,INPUTS,FEATURES

BINS=((0,10),(10,50),(50,100),(100,None))

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_json(path,obj):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temporary.replace(path)

def save_checkpoint(path,obj):
    temporary=path.with_suffix('.tmp'); torch.save(obj,temporary); temporary.replace(path)

def compact(u,arm): return u[...,[0,1,3,4,5] if arm=='B' else [0,1,3,5]]

def sample(item,norm,arm,device):
    u,f,valid=upstream.sample(item,norm)
    return compact(u,arm).to(device),f.to(device),valid.to(device)

def summarize(sse,sst,absolute,n):
    return dict(samples=n,rmse_N=math.sqrt(sse/n) if n else None,
        mae_N=absolute/n if n else None,
        nrmse_pct=100*math.sqrt(sse/sst) if n and sst>1e-12 else None,
        SSE=sse,SST=sst)

@torch.no_grad()
def evaluate(model,items,norm,device):
    model.eval(); rows=[]; bucket=[[0.0,0.0,0.0,0] for _ in BINS]
    for it in items:
        u,f,valid=sample(it,norm,model.arm,device)
        pred=model(u)
        err=((pred-f)*norm['f_scale'])[valid].double()
        truth=(f*norm['f_scale'])[valid].double()
        speed=(u[...,1].abs()*1000)[valid].double()
        row=summarize(float(err.square().sum()),float(truth.square().sum()),
                      float(err.abs().sum()),int(err.numel()))
        row.update(name=it['name'],edge_guard=upstream.edge_guard(len(it['values'])))
        rows.append(row)
        for j,(lo,hi) in enumerate(BINS):
            mask=speed>=lo
            if hi is not None: mask &= speed<hi
            e,y=err[mask],truth[mask]
            bucket[j][0]+=float(e.square().sum()); bucket[j][1]+=float(y.square().sum())
            bucket[j][2]+=float(e.abs().sum()); bucket[j][3]+=int(mask.sum())
    def pooled(chosen):
        return 100*math.sqrt(sum(r['SSE'] for r in chosen)/max(sum(r['SST'] for r in chosen),1e-12))
    return dict(pooled_nrmse_pct=pooled(rows),
        hf_nrmse_pct=pooled([r for r in rows if r['name'].startswith('SH_') and '6.62Hz' in r['name']]),
        rows=rows,speed_bins=[dict(min_abs_v_mm_s=lo,max_abs_v_mm_s=hi,
                **summarize(*values)) for (lo,hi),values in zip(BINS,bucket)],
        history_protocol='strict-past measured applied current; teacher-forced Stage 1')

def provenance():
    paths=[Path(__file__),HERE/'model_v19_bd.py',SOURCE/'AA02Train/train_v19.py',
        SOURCE/'mr_nlcsnn/model_v19.py',SOURCE/'mr_nlcsnn/v19_protocol.py',
        DATA/'alignment_manifest.json',DATA/'COMPLETE.json',G0,
        RUNTIME/'pack/AA02Train/v19_split.json',
        RUNTIME/'pack/training_outputs/v19_g0/runner_fixes_diff.txt']
    return {str(p):digest(p) for p in paths}

def load_data():
    if digest(DATA/'alignment_manifest.json')!=EXPECTED: raise ValueError('Manifest changed')
    g0=json.loads(G0.read_text())
    if g0['passed'] is not True or g0['data_manifest_sha256']!=EXPECTED: raise ValueError('G0 mismatch')
    # Metadata only: final-test files must be physically absent from this runtime.
    manifest=json.loads((DATA/'alignment_manifest.json').read_text())
    if len(manifest)!=301: raise ValueError('Expected 301 manifest records')
    for r in manifest:
        if r['split']=='final_test' and (DATA/r['relative_path']).exists():
            raise ValueError('Final-test isolation violated')
    train=upstream.read_dataset(DATA,'train'); val=upstream.read_dataset(DATA,'validation')
    if (len(train),len(val))!=(259,16): raise ValueError('Unexpected train/validation counts')
    return train,val,upstream.normalization(train)

def cpu_benchmark(model,validation,norm):
    model=model.cpu().eval(); torch.set_num_threads(1)
    u,_,_=sample(validation[0],norm,model.arm,'cpu')
    guard=upstream.edge_guard(u.shape[1]); sequence=u[:,guard:-guard]
    def measure(corners,iterations=1000):
        states=[model.initial_state(1) for _ in range(corners)]
        samples=[]
        with torch.no_grad():
            for k in range(iterations+100):
                motion=sequence[:,k%sequence.shape[1]]
                start=time.perf_counter_ns()
                for corner in range(corners):
                    ctx,h=model.prepare(motion,states[corner])
                    model.force_at(ctx,motion[:,2:3])
                    states[corner]=model.commit(h,states[corner],motion[:,2])
                elapsed=(time.perf_counter_ns()-start)/1e6
                if k>=100: samples.append(elapsed)
        ordered=sorted(samples)
        return dict(iterations=iterations,mean_ms=sum(samples)/iterations,
            p50_ms=ordered[int(.50*(iterations-1))],p95_ms=ordered[int(.95*(iterations-1))],
            p99_ms=ordered[int(.99*(iterations-1))],max_ms=max(samples))
    return dict(platform='LOCAL PC CPU; NOT S32/NI measurement',cpu=platform.processor(),
        torch_version=str(torch.__version__),threads=1,precision='float32',
        measured_operation='prepare + forward force_at + commit; inverse excluded',
        warmup_iterations=100,one_corner=measure(1),four_corners_sequential=measure(4))

def run_arm(arm,train_items,validation,norm,device,resume=False):
    folder=HERE/'sessions'/('v19_'+arm+'_seed7_30epochs')
    hashes=provenance()
    random.seed(7); torch.manual_seed(7)
    if device.type=='cuda': torch.cuda.manual_seed_all(7)
    model=NLCSNN_v19_BD(arm).to(device)
    parameters=set_stage(model,1)
    optimizer=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=1e-4)
    train=list(train_items); start_epoch=0; best_score=float('inf'); best_epoch=0
    if resume:
        latest=torch.load(folder/'latest.pth',map_location=device,weights_only=True)
        if latest['source_hashes']!=hashes or latest['norm_cfg']!=norm or latest['arm']!=arm:
            raise ValueError('Strict resume provenance/arm/normalization mismatch')
        if latest['total_epochs']!=30 or latest['device']!=str(device): raise ValueError('Resume schedule/device mismatch')
        model.load_state_dict(latest['state_dict']); optimizer.load_state_dict(latest['optimizer_state_dict'])
        random.setstate(latest['random_state']); torch.set_rng_state(latest['torch_rng_state'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([x.cpu() for x in latest['cuda_rng_states']])
        by_name={it['name']:it for it in train}
        if len(by_name)!=len(latest['train_order']) or set(by_name)!=set(latest['train_order']): raise ValueError('Train order mismatch')
        train=[by_name[n] for n in latest['train_order']]
        start_epoch=latest['epoch']; best_score=latest['best_score']; best_epoch=latest['best_epoch']
    else:
        if folder.exists(): raise FileExistsError('Refuse overwrite; explicit --resume required')
        folder.mkdir(parents=True)
        write_json(folder/'run_config.json',dict(arm=arm,source_commit='cac96e8f85491e731a0030c268f8c9f814b06953',
            source_hashes=hashes,data_manifest_sha256=EXPECTED,train_files=259,validation_files=16,
            final_test_read=False,model_cfg=model.config,input_columns=INPUTS[arm],nfl_features=FEATURES[arm],
            nfl_dim=model.nfl_dim,parameter_count=sum(p.numel() for p in model.parameters()),
            trainable_parameter_count=sum(p.numel() for p in parameters),stage=1,seed=7,total_epochs=30,
            cold_start=True,lr=3e-4,weight_decay=1e-4,clip=1,warmup_epochs=5,
            schedule='min(epoch/5,1) through epoch5; cosine to 0.01 of LR by epoch30',
            batch='one complete file',current='abs(I)/1.6 clipped [0,1]; raw magnitude tolerance2.10A',
            edge_guard='1000 if n>2000 else min(100,n//4)',
            loss='per-file p2p normalized squared error, negative force weight1.5, ST weight3',
            norm_cfg=norm,device=str(device),training_hardware=torch.cuda.get_device_name(0) if device.type=='cuda' else platform.processor(),
            torch_version=str(torch.__version__),python_version=sys.version,precision='float32; no AMP; no TF32',
            deterministic_algorithms=True,cpu_threads=torch.get_num_threads(),
            note='B observer dI is zero. B/D comparison is dead-channel removal, not dI information.'))
    if (folder/'COMPLETE.json').exists():
        print(f'{arm} already complete',flush=True); return
    print(f'START arm={arm} device={device} params={sum(p.numel() for p in model.parameters())} epoch={start_epoch+1}',flush=True)
    for epoch in range(start_epoch+1,31):
        began=time.perf_counter(); model.train(); random.shuffle(train)
        factor=min(epoch/5,1) if epoch<=5 else .01+.99*(1+math.cos(math.pi*(epoch-5)/25))/2
        for group in optimizer.param_groups: group['lr']=3e-4*factor
        losses=[]; grads=[]
        for index,it in enumerate(train,1):
            u,f,valid=sample(it,norm,arm,device)
            optimizer.zero_grad(set_to_none=True)
            loss=forward_loss(model(u),f,valid,(f.max()-f.min()).reshape(1),
                             f.new_tensor([3 if it['name'].startswith('ST_') else 1]))
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss '+it['name'])
            loss.backward(); grads.append(float(torch.nn.utils.clip_grad_norm_(parameters,1)))
            optimizer.step(); losses.append(float(loss.detach()))
            if index==1 or index%50==0 or index==len(train):
                write_json(folder/'progress.json',dict(arm=arm,epoch=epoch,file_index=index,files=259,
                    file=it['name'],elapsed_seconds=time.perf_counter()-began,best_epoch=best_epoch,
                    best_pooled_nrmse_pct=best_score if math.isfinite(best_score) else None))
                print(f'{arm} E{epoch:02d} file={index}/259 elapsed={time.perf_counter()-began:.1f}s',flush=True)
        metrics=evaluate(model,validation,norm,device)
        metrics.update(epoch=epoch,train_loss=sum(losses)/len(losses),loss_forward=sum(losses)/len(losses),
            loss_inverse=0.0,lr=optimizer.param_groups[0]['lr'],gradient_norm_max=max(grads),
            seconds=time.perf_counter()-began,tau_ms=float(model.lag.tau_ms().detach()),arm=arm)
        score=metrics['pooled_nrmse_pct']
        if not math.isfinite(score): raise ValueError('Nonfinite validation score')
        if score<best_score:
            best_score=score; best_epoch=epoch
            save_checkpoint(folder/'best.pth',dict(architecture=model.architecture,state_dict=model.state_dict(),
                model_cfg=model.config,norm_cfg=norm,metrics=metrics,epoch=epoch,arm=arm,stage=1,
                source_hashes=hashes,data_manifest_sha256=EXPECTED))
            write_json(folder/'best_validation_metrics.json',metrics)
        with (folder/'metrics.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(metrics,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
        save_checkpoint(folder/'latest.pth',dict(architecture=model.architecture,state_dict=model.state_dict(),
            model_cfg=model.config,norm_cfg=norm,arm=arm,epoch=epoch,total_epochs=30,stage=1,device=str(device),
            optimizer_state_dict=optimizer.state_dict(),random_state=random.getstate(),torch_rng_state=torch.get_rng_state(),
            cuda_rng_states=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            train_order=[it['name'] for it in train],best_score=best_score,best_epoch=best_epoch,
            source_hashes=hashes,data_manifest_sha256=EXPECTED,metrics=metrics))
        print(f'{arm} E{epoch:02d} pooled={score:.6f}% HF={metrics["hf_nrmse_pct"]:.6f}% best={best_score:.6f}%@{best_epoch} seconds={metrics["seconds"]:.1f}',flush=True)
    chosen=torch.load(folder/'best.pth',map_location='cpu',weights_only=True)
    selected=NLCSNN_v19_BD(**chosen['model_cfg']); selected.load_state_dict(chosen['state_dict'])
    timing=cpu_benchmark(selected,validation,norm)
    write_json(folder/'pc_cpu_timing.json',timing)
    write_json(folder/'COMPLETE.json',dict(arm=arm,epochs=30,best_epoch=best_epoch,
        best_pooled_nrmse_pct=best_score,hf_nrmse_pct_at_best=chosen['metrics']['hf_nrmse_pct'],
        parameter_count=sum(p.numel() for p in selected.parameters()),best_checkpoint_sha256=digest(folder/'best.pth'),
        final_test_read=False,pc_cpu_timing=timing))
    print(f'COMPLETE {arm}: best={best_score:.6f}% E{best_epoch}',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms',nargs='+',choices=('B','D'),default=['B','D'])
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    train,validation,norm=load_data()
    print('DATA verified train=259 validation=16 manifest='+EXPECTED,flush=True)
    print('NORMALIZATION '+json.dumps(norm),flush=True)
    for arm in args.arms: run_arm(arm,train,validation,norm,device,args.resume)

if __name__=='__main__': main()
