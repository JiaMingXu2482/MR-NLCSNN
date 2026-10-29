"""No-training synthetic inverse ablation. No dataset CSV or live model edits."""
import argparse,json,math,sys,hashlib,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
# Frozen inference-only dependency from previous delivery (no training imports).
sys.path.insert(0,str(HERE/'frozen_dependencies'))
import portable_frozen_loader as loader
from shared_window_inverse import solve_window_current
from motion_window_inverse import frozen_query,full_rollout_query,motion_forecast,grid_inverse

NAMES=['multisine_low','multisine_mixed','sine_8Hz','sine_12Hz']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clone_state(st): return type(st)(**{k:v.clone() for k,v in vars(st).items()})
def convert_state(st,dtype): return type(st)(**{k:v.to(dtype) for k,v in vars(st).items()})
def stats(e):
    e=e.double().flatten()
    return dict(mae=float(e.abs().mean()),rmse=float(e.square().mean().sqrt()),max_abs=float(e.abs().max()),bias=float(e.mean()))

@torch.no_grad()
def main():
    torch.set_num_threads(1); torch.manual_seed(7)
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--output',type=Path,default=HERE/'replay_results')
    args=parser.parse_args()
    out=args.output; out.mkdir(exist_ok=False)
    cp=args.checkpoint
    ck=torch.load(cp,map_location='cpu',weights_only=True)
    assert sha(cp)=='3a407a77e8f0b9167282fdbfc77923dac23fcd0430fa498432e36e31f97423c7'
    model=loader.build('A').eval(); model.load_state_dict(ck['state_dict']); norm=ck['norm_cfg']; scale=norm['f_scale']
    freq=torch.tensor([1.66,4.97,8.,12.]); amp=torch.tensor([10.,5.,.5,.5])
    def motion(t):
        phase=2*math.pi*freq*t; extra=torch.tensor([2.,1.,0.,0.]); w2=2*math.pi*2.3*freq
        x=250+amp*phase.sin()+extra*(w2*t+.3).sin()
        v=amp*2*math.pi*freq*phase.cos()+extra*w2*(w2*t+.3).cos()
        a=-amp*(2*math.pi*freq)**2*phase.sin()-extra*w2*w2*(w2*t+.3).sin()
        return torch.stack(((x-norm['x_ref'])/norm['x_scale'],v/1000,a/50000,
                            torch.zeros_like(x),torch.zeros_like(x),torch.full_like(x,-.25)),-1)
    def sequential(st,seq,current):
        forces=[]
        for j in range(seq.shape[1]):
            c,h=model.prepare(seq[:,j,:],st)
            forces.append(model.force_at(c,current[:,None]/1.6)[:,0]*scale)
            st=model.commit(h,st,current/1.6)
        return torch.stack(forces,1).mean(1),st
    st=model.initial_state(4,initial_current=.5)
    for k in range(100):
        seq=torch.stack([motion(k*.01+j*.001) for j in range(10)],1)
        _,st=sequential(st,seq,torch.full((4,),.8))
    warm=clone_state(st); states=[]; sequences=[]; currents=[]; old_targets=[]
    for k in range(100):
        seq=torch.stack([motion(1+k*.01+j*.001) for j in range(10)],1)
        ii=.8+.6*torch.sin(torch.tensor([.2,.7,1.1,1.8])+k*.23)
        states.append(clone_state(st)); sequences.append(seq); currents.append(ii)
        ff,st=sequential(st,seq,ii); old_targets.append(ff)
    state=type(st)(**{key:torch.cat([vars(s)[key] for s in states],dim=1 if key=='h' else 0).double() for key in vars(st)})
    seq=torch.cat(sequences).double(); truth=torch.cat(currents).double()
    original_targets=torch.cat(old_targets).double()
    model.double(); before={k:v.clone() for k,v in model.state_dict().items()}; saved_state=clone_state(state)
    qtrue=full_rollout_query(model,state,seq,scale); target=qtrue(truth[:,None])[:,0]
    c,_=model.prepare(seq[:,0,:],state); prev=state.previous_current*1.6
    ca=motion_forecast(seq[:,0,:],10,.001,norm['x_scale'],'ca')
    held=motion_forecast(seq[:,0,:],10,.001,norm['x_scale'],'hold')
    queries={'frozen9_refined':frozen_query(model,c,held[:,:,1],scale),
             'true_v_frozen_csb':frozen_query(model,c,seq[:,:,1],scale),
             'ca_v_frozen_csb':frozen_query(model,c,ca[:,:,1],scale),
             'full_true_motion_9':qtrue,'full_true_motion_refined':qtrue,
             'full_hold_motion_9':full_rollout_query(model,state,held,scale),
             'full_ca_motion_9':full_rollout_query(model,state,ca,scale),
             'full_ca_motion_refined':full_rollout_query(model,state,ca,scale)}
    slope=(qtrue((truth+.001)[:,None])[:,0]-qtrue((truth-.001)[:,None])[:,0])/.002
    reversal=(seq[:,:,1].min(1).values<0)&(seq[:,:,1].max(1).values>0)
    all_results={}; per_window={}
    def summarize(name,ii,actual,diag):
        rr={}
        for j,case in enumerate(NAMES):
            ix=torch.arange(j,400,4); y=target[ix]; err=actual[ix]-y; eI=ii[ix]-truth[ix]
            tv=float(ii[ix].diff().abs().sum()); tvtrue=float(truth[ix].diff().abs().sum())
            sat=float(((ii[ix]<1e-5)|(ii[ix]>1.6-1e-5)).double().mean())
            force=stats(err); current=stats(eI); rms=float(y.square().mean().sqrt())
            rr[case]=dict(force_error_N=force,target_RMS_N=rms,NRMSE_pct=100*force['rmse']/rms,current_error_A=current,
                current_TV_A=tv,truth_TV_A=tvtrue,TV_ratio=tv/tvtrue,saturation_fraction=sat,
                force_MAE_le_7N=force['mae']<=7,force_MAE_le_10N=force['mae']<=10,
                current_MAE_le_point1A=current['mae']<=.1,TV_le_1point5=tv/tvtrue<=1.5,saturation_lt_10pct=sat<.1,
                zero_force_numerical_check=force['max_abs']<.01,
                reversal_windows=int(reversal[ix].sum()),
                reversal_force_error_N=stats(err[reversal[ix]]) if reversal[ix].any() else None,
                abs_true_sensitivity_median_N_per_A=float(slope[ix].abs().median()))
            for key in ['no_grid_bracket','sampled_nonmonotone','at_bound','rate_limited']:
                if key in diag: rr[case][key+'_fraction']=float(diag[key][ix].double().mean())
            if 'bracket_count' in diag: rr[case]['multiple_grid_crossings_fraction']=float((diag['bracket_count'][ix]>1).double().mean())
            if 'query_residual_N' in diag: rr[case]['query_residual_N']=stats(diag['query_residual_N'][ix])
        all_results[name]=rr
        per_window[name]=dict(current_A=ii.reshape(100,4).tolist(),actual_mean_N=actual.reshape(100,4).tolist(),
             diagnostics={k:v.reshape(100,4).tolist() for k,v in diag.items() if v.numel()==400})
        print(name,json.dumps({k:dict(MAE_N=round(v['force_error_N']['mae'],4),I_MAE=round(v['current_error_A']['mae'],4),TV_ratio=round(v['TV_ratio'],3),sat=v['saturation_fraction']) for k,v in rr.items()}),flush=True)
    ii,dd=solve_window_current(model,c,target[:,None],prev[:,None],force_scale=scale)
    summarize('old_frozen_window',ii[:,0],qtrue(ii)[:,0],{'query_residual_N':dd['residual_N'][:,0],
             'no_grid_bracket':~dd['reachable'][:,0],'at_bound':dd['at_bound'][:,0]})
    timings={}
    for name,query in queries.items():
        refine=0 if name.endswith('_9') else 24
        start=time.perf_counter(); ii,dd=grid_inverse(query,target,prev,refine=refine)
        timings[name]=time.perf_counter()-start
        summarize(name,ii,qtrue(ii[:,None])[:,0],dd)
    # Implementation tests: mapping can have opposite slope/multiple roots.
    tg=torch.tensor([.0,.3],dtype=torch.float64); pr=torch.tensor([.9,.4],dtype=torch.float64)
    roots,rd=grid_inverse(lambda i:(i-.4)*(i-1.2),tg,pr,refine=24)
    tests=dict(nonmonotone_root_search=bool(rd['query_residual_N'].abs().max()<.01),
         known_current_reachable_all_windows=bool(torch.equal(target,qtrue(truth[:,None])[:,0])),
         one_candidate_full_vs_sequential=bool((target-original_targets).abs().max()<.01),
         query_state_not_mutated=all(torch.equal(v,vars(saved_state)[k]) for k,v in vars(state).items()),
         weights_unchanged=all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
    assert all(tests.values()),tests
    # Deployable CA/full-rollout grid methods, carrying OWN history (not teacher).
    for lam,rate in [(0.,None),(2.,None),(10.,None),(10.,.2)]:
        name=f'own_history_full_ca9_lambda{lam:g}'+('_rate0p2' if rate else '')
        live=convert_state(warm,torch.float64); chosen=[]; achieved=[]; diaglist=[]
        for k in range(100):
            measured=sequences[k].double(); predicted=motion_forecast(measured[:,0,:],10,.001,norm['x_scale'],'ca')
            q=full_rollout_query(model,live,predicted,scale)
            ii,dd=grid_inverse(q,target[k*4:(k+1)*4],live.previous_current*1.6,lam_N_per_A=lam,rate_A=rate)
            actual,live=sequential(live,measured,ii)
            chosen.append(ii); achieved.append(actual); diaglist.append(dd)
        dd={key:torch.cat([d[key] for d in diaglist]) for key in diaglist[0]}
        summarize(name,torch.cat(chosen),torch.cat(achieved),dd)
    # Direct regularization checks are separate from inverse consistency tests.
    flat=grid_inverse(lambda i:torch.zeros_like(i),torch.zeros(2,dtype=torch.float64),pr,lam_N_per_A=10)[0]
    tests['flat_force_regularization_preserves_current']=bool(torch.equal(flat,pr))
    rate_sol=grid_inverse(lambda i:100*i,torch.tensor([160.,0.],dtype=torch.float64),pr,lam_N_per_A=10,rate_A=.2)[0]
    tests['rate_limit_enforced']=bool(((rate_sol-pr).abs()<=.2000000001).all())
    assert all(tests.values())
    result=dict(tests=tests,methods=all_results,precision='CPU FP64, single thread; weights copied from FP32 checkpoint',
         source_checkpoint_sha256=sha(cp),plan_sha256=sha(HERE/'PLAN.md'),
         sources={p.name:sha(p) for p in [Path(__file__),HERE/'motion_window_inverse.py']},
         max_target_difference_from_original_FP32_N=float((target-original_targets).abs().max()),
         truth_all_windows_reachable=True,window_count=400,samples_per_window=10,dt_s=.001,
         total_400_window_solver_seconds=timings,training=False,dataset_read=False,weights_changed=False,
         note='Oracle gets future motion only for diagnosing inverse consistency; CA forecast is causal. No NI/CarSim validation.')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'windows.json').write_text(json.dumps(dict(case_names=NAMES,target_mean_N=target.reshape(100,4).tolist(),
        known_current_A=truth.reshape(100,4).tolist(),true_local_slope_N_per_A=slope.reshape(100,4).tolist(),
        methods=per_window)),encoding='utf-8')
    print('COMPLETE',flush=True)

if __name__=='__main__': main()
