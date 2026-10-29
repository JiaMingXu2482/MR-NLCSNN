"""Frozen A checkpoint tests and synthetic changing-current rollout audit.

Reads no dataset CSV. No training, recurrent-state correction or force guard.
All methods start at the SAME committed history. Desired force is the mean
of a known-current forward rollout; future motion is used only for auditing.
"""
import json,math,time,platform
from dataclasses import replace
from pathlib import Path
import torch
import train_capacity_four as run
from shared_window_inverse import window_force_N,solve_window_current

HERE=Path(__file__).resolve().parent
OUT=HERE/'claude_execution_20261029'

def numbers(e):
    e=e.detach().double().flatten()
    return dict(n=e.numel(),mae=float(e.abs().mean()),rms=float(e.square().mean().sqrt()),
                max_abs=float(e.abs().max()),bias=float(e.mean()))

@torch.no_grad()
def main():
    torch.set_num_threads(1); torch.manual_seed(7)
    cfg=json.loads((HERE/'capacity_four_config_20261028.json').read_text())
    cp=Path(cfg['session_root'])/'A/final.pth'
    ck=torch.load(cp,map_location='cpu',weights_only=True)
    model=run.build('A').eval(); model.load_state_dict(ck['state_dict'])
    norm=ck['norm_cfg']; scale=norm['f_scale']; OUT.mkdir(exist_ok=True)
    before={k:v.clone() for k,v in model.state_dict().items()}
    state=model.initial_state(12,initial_current=.5)
    state_before={k:v.clone() for k,v in vars(state).items()}
    motion=torch.zeros(12,6); motion[:,1]=torch.linspace(-.2,.2,12)
    ctx,_=model.prepare(motion,state)
    current=torch.linspace(.1,1.5,12)[:,None]
    tests={}
    f1=window_force_N(model,ctx,current,force_scale=scale,steps=1)
    tests['one_step_matches_shared_forward']=bool(torch.allclose(f1,model.force_at(ctx,current/1.6)*scale,atol=.001,rtol=1e-5))
    previous=ctx.previous_effective_current.clone(); fs=[]
    for _ in range(10):
        cc=replace(ctx,previous_effective_current=previous)
        fs.append(model.force_at(cc,current/1.6)*scale)
        previous=model.lag.step(current/1.6,previous)
    fw=window_force_N(model,ctx,current,force_scale=scale)
    tests['closed_form_matches_ten_EMA_steps']=bool(torch.allclose(fw,torch.stack(fs).mean(0),atol=.002,rtol=1e-5))
    sol,d=solve_window_current(model,ctx,fw,torch.full_like(current,.8),force_scale=scale)
    mask=~d['weak_authority']
    tests['reachable_current_recovery']=bool((sol[mask]-current[mask]).abs().max()<1e-3)
    tests['reachable_force_residual_under_1N']=bool(d['residual_N'][mask].abs().max()<1)
    vals=torch.stack([window_force_N(model,ctx,torch.full_like(current,i),force_scale=scale) for i in torch.linspace(0,1.6,17)])
    delta=vals[1:]-vals[:-1]
    tests['monotone_both_velocity_signs']=bool(torch.where(ctx.v>0,delta<=.001,delta>=-.001).all())
    zero=replace(ctx,v=torch.zeros_like(ctx.v))
    solz,dz=solve_window_current(model,zero,zero.b*scale,torch.full_like(current,.7),force_scale=scale)
    tests['zero_velocity_holds_previous_applied_current']=bool(dz['weak_authority'].all() and torch.equal(solz,torch.full_like(current,.7)))
    _,du=solve_window_current(model,ctx,d['force_high_N']+100,torch.full_like(current,.8),force_scale=scale)
    tests['unreachable_flag_and_residual']=bool((~du['reachable']).all() and (du['residual_N']<-99).all())
    tests['state_not_mutated']=all(torch.equal(v,state_before[k]) for k,v in vars(state).items())
    tests['weights_unchanged']=all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert all(tests.values()),tests
    # Synthetic cases: two multisine motions plus 8/12 Hz sine, held varying I.
    names=['multisine_low','multisine_mixed','sine_8Hz','sine_12Hz']
    freq=torch.tensor([1.66,4.97,8.,12.]); amp=torch.tensor([10.,5.,.5,.5])
    def inputs(t):
        phase=2*math.pi*freq*t
        extra=torch.tensor([2.,1.,0.,0.]); w2=2*math.pi*2.3*freq
        x=250+amp*phase.sin()+extra*(w2*t+.3).sin()
        v=amp*2*math.pi*freq*phase.cos()+extra*w2*(w2*t+.3).cos()
        a=-amp*(2*math.pi*freq)**2*phase.sin()-extra*w2*w2*(w2*t+.3).sin()
        return torch.stack(((x-norm['x_ref'])/norm['x_scale'],v/1000,a/50000,
                            torch.zeros_like(x),torch.zeros_like(x),torch.full_like(x,-.25)),-1)
    def rollout(st,at,i):
        ff=[]
        for j in range(10):
            u=inputs(at+j*.001); c,h=model.prepare(u,st)
            ff.append(model.force_at(c,i[:,None]/1.6)[:,0]*scale)
            st=model.commit(h,st,i/1.6)
        return torch.stack(ff).mean(0),st
    st=model.initial_state(4,initial_current=.5)
    # Identical committed warmup for all inverse queries.
    for k in range(100): _,st=rollout(st,k*.01,torch.full((4,),.8))
    records=[]
    for k in range(100):
        at=1+k*.01
        truth_I=.8+.6*torch.sin(torch.tensor([.2,.7,1.1,1.8])+k*.23)
        target,end=rollout(st,at,truth_I)
        c,_=model.prepare(inputs(at),st)
        previous_A=st.previous_current[:,None]*1.6
        methods={}
        for method,steps in [('single_step',1),('window10',10)]:
            ii,diag=solve_window_current(model,c,target[:,None],previous_A,force_scale=scale,steps=steps)
            actual,_=rollout(st,at,ii[:,0])
            methods[method]=dict(current_A=ii[:,0].tolist(),actual_mean_N=actual.tolist(),
                surrogate_residual_N=diag['residual_N'][:,0].tolist(),
                reachable=diag['reachable'][:,0].tolist(),weak=diag['weak_authority'][:,0].tolist(),
                slope_N_per_A=diag['local_slope_N_per_A'][:,0].tolist(),at_bound=diag['at_bound'][:,0].tolist())
        head=model.inverse_predict_context(c,target[:,None]/scale)[:,0]*1.6
        headF,_=rollout(st,at,head)
        methods['untrained_inv_head_reference']=dict(current_A=head.tolist(),actual_mean_N=headF.tolist())
        records.append(dict(time_s=at,truth_current_A=truth_I.tolist(),target_mean_N=target.tolist(),
                            speed_mm_s=(inputs(at)[:,1]*1000).tolist(),methods=methods))
        st=end
    results={}
    for method in records[0]['methods']:
        rr={}
        for j,name in enumerate(names):
            error=torch.tensor([r['methods'][method]['actual_mean_N'][j]-r['target_mean_N'][j] for r in records])
            ei=torch.tensor([r['methods'][method]['current_A'][j]-r['truth_current_A'][j] for r in records])
            rr[name]=dict(actual_rollout_mean_error_N=numbers(error),current_error_A=numbers(ei),
                current_total_variation_A=sum(abs(records[k]['methods'][method]['current_A'][j]-records[k-1]['methods'][method]['current_A'][j]) for k in range(1,len(records))))
            if method!='untrained_inv_head_reference':
                rr[name].update(surrogate_residual_N=numbers(torch.tensor([r['methods'][method]['surrogate_residual_N'][j] for r in records])),
                    reachable_fraction=sum(r['methods'][method]['reachable'][j] for r in records)/len(records),
                    saturation_fraction=sum(r['methods'][method]['at_bound'][j] for r in records)/len(records))
        results[method]=rr
    # Whole-call PC timings; no claim about NI deadline.
    s=model.initial_state(1); u=inputs(1.)[:1]; c,h=model.prepare(u,s)
    tar=window_force_N(model,c,torch.full_like(c.v,.8),force_scale=scale)
    timings={}
    for name,call in [('forward_prepare_force_commit',lambda: (lambda ch: model.commit(ch[1],s,torch.tensor([.5])) if model.force_at(ch[0],torch.tensor([[.5]])).isfinite().all() else None)(model.prepare(u,s))),
                      ('inverse10_24bisections',lambda:solve_window_current(model,c,tar,torch.full_like(tar,.8),force_scale=scale))]:
        for _ in range(10): call()
        tt=[]
        for _ in range(100):
            start=time.perf_counter_ns(); call(); tt.append((time.perf_counter_ns()-start)/1000)
        timings[name]=dict(median_us=float(torch.tensor(tt).median()),p95_us=float(torch.quantile(torch.tensor(tt),.95)),max_us=max(tt))
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    output=dict(tests=tests,synthetic_results=results,PC_CPU_timings=timings,
                torch_version=torch.__version__,cpu=platform.processor(),threads=1,
                checkpoint_sha256=run.base.digest(cp),model_parameter_count=sum(p.numel() for p in model.parameters()),
                inverse_added_parameters=0,checkpoint=str(cp),force_scale=scale,tau_ms=float(model.lag.tau_ms()),
                steps=10,dt_s=.001,iterations=24,span_tol_N=1.,training_started=False,
                dataset_CSV_read=False,inv_head_trained=False,
                scope='Synthetic model-consistency audit only; no measured inverse-accuracy or NI timing claim',
                sources={str(p.name):run.base.digest(p) for p in [Path(__file__),HERE/'shared_window_inverse.py']})
    (OUT/'inverse_audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    (OUT/'synthetic_rollout.json').write_text(json.dumps(records),encoding='utf-8')
    print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__': main()
