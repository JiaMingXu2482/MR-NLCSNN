"""Frozen random-file diagnostic; no training, no final-test access."""
import json
from pathlib import Path
import torch
import train_capacity_four as run

HERE=Path(__file__).resolve().parent
OUT=HERE/'diagnosis_capacity_bias_20261028'
TARGETS=['RLF_RMS10mm_2Hz_r1','RLF_RMS10mm_4Hz','SR_RMS10mm_1.66Hz_0A']

@torch.no_grad()
def main():
    run.setup(); OUT.mkdir(exist_ok=True)
    cfg=json.loads((HERE/'capacity_four_config_20261028.json').read_text())
    root=Path(cfg['session_root']); rows=[]; hashes={}
    items=[(s,it) for s in ('train','validation') for it in run.load(cfg,s) if it['name'].split('_')[0] in ('SR','RLF','RHF')]
    for arm in run.ARMS:
        cp=root/arm/'final.pth'; hashes[arm]=run.base.digest(cp)
        ck=torch.load(cp,map_location='cpu',weights_only=True)
        model=run.build(arm).cuda().eval(); model.load_state_dict(ck['state_dict']); norm=ck['norm_cfg']
        for split,it in items:
            u,f,valid=run.sample(it,norm,arm); ctx,_=model.contexts(u)
            current=u[...,2 if arm=='D' else 3]
            pred=model.force_at(ctx,current)*norm['f_scale']; truth=f*norm['f_scale']
            e=(pred-truth)[valid].double(); y=truth[valid].double()
            values=it['values']; v=values[:,1].cuda()[valid[0]]
            row=dict(arm=arm,name=it['name'],split=split,mean_error_N=float(e.mean()),
                     **run.base.summarize(float(e.square().sum()),float(y.square().sum()),float(e.abs().sum()),e.numel()),
                     mean_x_mm=float(values[:,0].mean()),temperature_mean=float(values[:,5].mean()),
                     current_mean=float(values[:,3].abs().mean()),signed_speed_bias={})
            for label,m in [('negative',v<-10),('near_zero',v.abs()<10),('positive',v>10)]:
                row['signed_speed_bias'][label]=float(e[m].mean()) if m.any() else None
            rows.append(row)
            if it['name'] in TARGETS:
                q=1-model.lag.step(current,ctx.previous_effective_current)
                visc=ctx.c*ctx.v*model.G_c(q)*norm['f_scale']
                fric=(ctx.s*torch.tanh(ctx.v[...,None]/model.sigmas)).sum(-1)*model.G_s(q)*norm['f_scale']
                # Sensitivity only: an artificial constant prefix, not a proposed correction.
                prefix=u[:,:1].expand(-1,5000,-1)
                _,state=model.contexts(prefix)
                warmed,_=model.contexts(u,state)
                pre=model.force_at(warmed,current)*norm['f_scale']
                row['artificial_prefix_rms_prediction_change_N']=float(((pre-pred)[valid].square().mean()).sqrt())
                out=dict(name=it['name'],arm=arm,guard=run.base.upstream.edge_guard(len(values)),
                         measured_force_N=truth[0].cpu().tolist(),predicted_force_N=pred[0].cpu().tolist(),
                         viscous_N=visc[0].cpu().tolist(),friction_N=fric[0].cpu().tolist(),b_N=(ctx.b[0]*norm['f_scale']).cpu().tolist(),
                         values=values.tolist())
                run.base.write_json(OUT/f'{arm}_{it["name"]}.json',out)
        print(f'{arm}: frozen diagnostics complete',flush=True)
        del model
    run.base.write_json(OUT/'random_file_metrics.json',rows)
    run.base.write_json(OUT/'provenance.json',dict(checkpoints=hashes,data_manifest_sha256=cfg['data_manifest_sha256'],
                                                script_sha256=run.base.digest(__file__),training_updates=0,final_test_read=False))

if __name__=='__main__': main()
