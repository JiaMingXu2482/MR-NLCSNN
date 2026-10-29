"""Post-audit adaptive-grid oracle. No training, no truth-current solver seed."""
import json,math,sys,hashlib
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'claude_execution_20261029/code'))
import portable_frozen_loader as loader
from motion_window_inverse import full_rollout_query,grid_inverse
from run_motion_audit import NAMES,stats

@torch.no_grad()
def main():
    torch.set_num_threads(1)
    cp=HERE.parent/'sessions/v19_capacity_four_balanced_300_20261028/A/final.pth'
    ck=torch.load(cp,map_location='cpu',weights_only=True)
    model=loader.build('A').eval(); model.load_state_dict(ck['state_dict']); norm=ck['norm_cfg']; scale=norm['f_scale']
    freq=torch.tensor([1.66,4.97,8.,12.]); amp=torch.tensor([10.,5.,.5,.5])
    def motion(t):
        phase=2*math.pi*freq*t; extra=torch.tensor([2.,1.,0.,0.]); w2=2*math.pi*2.3*freq
        x=250+amp*phase.sin()+extra*(w2*t+.3).sin()
        v=amp*2*math.pi*freq*phase.cos()+extra*w2*(w2*t+.3).cos()
        a=-amp*(2*math.pi*freq)**2*phase.sin()-extra*w2*w2*(w2*t+.3).sin()
        return torch.stack(((x-norm['x_ref'])/norm['x_scale'],v/1000,a/50000,torch.zeros_like(x),torch.zeros_like(x),torch.full_like(x,-.25)),-1)
    st=model.initial_state(4,initial_current=.5); states=[]; sequences=[]; currents=[]
    for k in range(200):
        seq=torch.stack([motion(k*.01+j*.001) for j in range(10)],1)
        ii=torch.full((4,),.8) if k<100 else .8+.6*torch.sin(torch.tensor([.2,.7,1.1,1.8])+(k-100)*.23)
        if k>=100:
            states.append(type(st)(**{key:v.clone() for key,v in vars(st).items()})); sequences.append(seq); currents.append(ii)
        for j in range(10):
            _,h=model.prepare(seq[:,j],st); st=model.commit(h,st,ii/1.6)
    state=type(st)(**{key:torch.cat([vars(s)[key] for s in states],dim=1 if key=='h' else 0).double() for key in vars(st)})
    seq=torch.cat(sequences).double(); truth=torch.cat(currents).double(); model.double()
    full=full_rollout_query(model,state,seq,scale); target=full(truth[:,None])[:,0]; prev=state.previous_current*1.6
    saved=json.loads((HERE/'results/windows.json').read_text())
    # Match the original audited inputs/history; never use saved truth in solver.
    delta=float((target-torch.tensor(saved['target_mean_N'],dtype=torch.float64).flatten()).abs().max())
    assert delta<.01,delta
    chosen,dd=grid_inverse(full,target,prev,refine=24,force_tie_N=.001)
    solved=~dd['no_grid_bracket']; levels=[dict(nodes=9,unbracketed=int((~solved).sum()))]
    multiple=(dd['bracket_count']>1).clone(); examples=[]
    def subset_state(ix):
        return type(state)(**{key:(v[:,ix,:] if key=='h' else v[ix]) for key,v in vars(state).items()})
    for n in [33,129,513]:
        ix=torch.nonzero(~solved).flatten()
        if not len(ix): break
        q=full_rollout_query(model,subset_state(ix),seq[ix],scale)
        ii,diag=grid_inverse(q,target[ix],prev[ix],node_count=n,refine=24,force_tie_N=.001)
        chosen[ix]=ii; solved[ix]=~diag['no_grid_bracket']; multiple[ix]=diag['bracket_count']>1
        levels.append(dict(nodes=n,evaluated_windows=len(ix),unbracketed=int((~solved).sum())))
    actual=full(chosen[:,None])[:,0]
    summary={}
    for j,name in enumerate(NAMES):
        ix=torch.arange(j,400,4); ei=chosen[ix]-truth[ix]; ef=actual[ix]-target[ix]
        tv=float(chosen[ix].diff().abs().sum()); tvtrue=float(truth[ix].diff().abs().sum())
        summary[name]=dict(force_error_N=stats(ef),current_error_A=stats(ei),TV_ratio=tv/tvtrue,
            current_TV_A=tv,truth_TV_A=tvtrue,unbracketed=int((~solved[ix]).sum()),multiple_brackets=int(multiple[ix].sum()),
            saturation_fraction=float(((chosen[ix]<1e-5)|(chosen[ix]>1.6-1e-5)).double().mean()),max_force_residual_below_point01N=bool(ef.abs().max()<.01))
    for ix in torch.argsort((chosen-truth).abs(),descending=True)[:8].tolist():
        examples.append(dict(case=NAMES[ix%4],window=ix//4,time_s=1+.01*(ix//4),known_I_A=float(truth[ix]),chosen_I_A=float(chosen[ix]),
            previous_I_A=float(prev[ix]),target_mean_N=float(target[ix]),chosen_mean_N=float(actual[ix]),residual_N=float(actual[ix]-target[ix])))
    result=dict(method='offline adaptive oracle; true future motion, private full GRU rollouts',summary=summary,
        grid_levels=levels,multiple_solution_examples=examples,force_tie_N=.001,known_current_used_as_solver_seed=False,
        original_target_match_max_N=delta,training=False,data_read=False,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'motion_window_inverse.py',HERE/'ORACLE_SUPPLEMENT.md']})
    (HERE/'results/oracle_refinement.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (HERE/'results/oracle_windows.json').write_text(json.dumps(dict(current_A=chosen.reshape(100,4).tolist(),actual_mean_N=actual.reshape(100,4).tolist())),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()
