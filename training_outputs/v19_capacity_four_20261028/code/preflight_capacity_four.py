"""Train-only startup checks; no optimizer steps or validation evaluation."""
import json,collections,unittest
from pathlib import Path
import torch
import train_capacity_four as run
import test_capacity_model

def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_capacity_model))
    assert result.wasSuccessful()
    path=run.HERE/'capacity_four_config_20261028.json'; cfg=json.loads(path.read_text()); run.setup()
    items=run.load(cfg,'train'); norm=run.base.upstream.normalization(items)
    order=run.schedule([it['name'] for it in items],cfg['epochs']*len(items))
    assert len(order)==74400 and len(set(order))==248
    assert collections.Counter(n.split('_')[0] for n in order)==dict(SH=18600,ST=11160,SR=14880,RLF=14880,RHF=14880)
    assert order==run.schedule([it['name'] for it in items],74400)
    longest=max(items,key=lambda it:len(it['values'])); records=[]
    # Memory is measured per model; concurrent processes also each need a CUDA context.
    for arm in run.ARMS:
        model=run.build(arm).cuda(); params=run.base.set_stage(model,1)
        u,f,valid=run.sample(longest,norm,arm); torch.cuda.reset_peak_memory_stats()
        loss=run.base.forward_loss(model(u),f,valid,(f.max()-f.min()).reshape(1),f.new_tensor([1]))
        loss.backward(); assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
        records.append(dict(arm=arm,total_parameters=sum(p.numel() for p in model.parameters()),stage1_parameters=sum(p.numel() for p in params),
            longest_train_file=longest['name'],sequence_samples=len(longest['values']),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        del model,params,u,f,valid,loss; torch.cuda.empty_cache()
    data=Path(cfg['data_path']); manifest=json.loads((data/'alignment_manifest.json').read_text())
    assert all(not (data/m['relative_path']).exists() for m in manifest if m['split']=='final_test')
    record=dict(passed=True,model_checks=result.testsRun,train_files=248,normalization=norm,models=records,
        schedule_counts=dict(collections.Counter(n.split('_')[0] for n in order)),total_updates_per_arm=74400,
        source_hashes=run.hashes(path),data_manifest_sha256=cfg['data_manifest_sha256'],validation_evaluated=False,final_test_read=False,
        cuda_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,optimizer_steps=0)
    run.base.write_json(run.HERE/'capacity_four_preflight_20261028.json',record)
    print(json.dumps(record,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
