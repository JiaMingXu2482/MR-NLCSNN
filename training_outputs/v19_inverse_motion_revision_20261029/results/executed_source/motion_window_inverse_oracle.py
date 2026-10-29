"""Stateless queries on private candidate states. Physical units: force N, I A.

No assumption of monotonic force when velocity changes sign or GRU rolls.
Nine-point range and bracket diagnostics are sampled evidence, not guarantees.
"""
import torch


def repeat_state(state, candidates):
    L,B,H=state.h.shape
    fields={k:(v[:,:,None,:].expand(L,B,candidates,H).reshape(L,B*candidates,H)
               if k=='h' else v[:,None].expand(B,candidates).reshape(-1))
            for k,v in vars(state).items()}
    return type(state)(**fields)


def motion_forecast(motion0, steps, dt, x_scale, mode='ca'):
    out=motion0[:,None,:].expand(-1,steps,-1).clone()
    if mode=='hold': return out
    if mode!='ca': raise ValueError(mode)
    t=torch.arange(steps,device=motion0.device,dtype=motion0.dtype)*dt
    v=motion0[:,1]*1000; a=motion0[:,2]*50000
    out[:,:,0]+=(v[:,None]*t+.5*a[:,None]*t*t)/x_scale
    out[:,:,1]+=(a[:,None]*t)/1000
    return out


def frozen_query(model,ctx,velocity_norm,force_scale):
    """Freeze c/s/b, allow each future velocity; exact held-current EMA."""
    steps=velocity_norm.shape[1]
    t=torch.arange(1,steps+1,device=ctx.v.device,dtype=ctx.v.dtype)*model.lag.dt_s
    decay=torch.exp(-1000*t/model.lag.tau_ms())
    v=velocity_norm*model.v_out_scale
    friction=(ctx.s[:,0,None,:]*torch.tanh(v[:,:,None]/model.sigmas)).sum(-1)
    def query(current_A):
        effective=current_A[:,:,None]/1.6+(ctx.previous_effective_current[:,0,None,None]-current_A[:,:,None]/1.6)*decay
        q=1-effective
        return ((ctx.c[:,0,None,None]*v[:,None,:])*model.G_c(q)
                +friction[:,None,:]*model.G_s(q)+ctx.b[:,0,None,None]).mean(-1)*force_scale
    return query


def full_rollout_query(model,state,motion_sequence,force_scale):
    """Candidate I updates its own GRU/history/lag. Caller state is untouched."""
    B,T,F=motion_sequence.shape
    def query(current_A):
        K=current_A.shape[1]
        u=motion_sequence[:,None,:,:].expand(B,K,T,F).clone()
        u[:,:,:,3]=current_A[:,:,None]/1.6
        u=u.reshape(B*K,T,F)
        f=model(u,repeat_state(state,K))
        return f.reshape(B,K,T).mean(-1)*force_scale
    return query


@torch.no_grad()
def grid_inverse(query,target_N,previous_A,*,refine=0,lam_N_per_A=0.,rate_A=None,node_count=9,force_tie_N=1e-7):
    """9-node piecewise-linear search; optional all-bracket root refinement.

    PWL |F-target|+lambda|I-prev| minima occur at nodes, target crossings, or
    previous I. For refine>0 all found brackets are solved using private queries;
    output is chosen using actual query errors, with nearest-I numerical ties.
    No unsampled extremum/global-infeasibility claim. Flat targets hold previous.
    """
    if lam_N_per_A<0 or (rate_A is not None and rate_A<=0) or node_count<2 or force_tie_N<0: raise ValueError('Invalid penalty/rate/grid')
    B=target_N.numel()
    lo=torch.zeros_like(target_N); hi=torch.full_like(target_N,1.6)
    if rate_A is not None:
        lo=(previous_A-rate_A).clamp(0,1.6); hi=(previous_A+rate_A).clamp(0,1.6)
    nodes=lo[:,None]+(hi-lo)[:,None]*torch.linspace(0,1,node_count,device=target_N.device,dtype=target_N.dtype)
    vals=query(nodes); residual=vals-target_N[:,None]
    left,right=nodes[:,:-1].clone(),nodes[:,1:].clone()
    fl,fr=residual[:,:-1].clone(),residual[:,1:].clone()
    brackets=(fl*fr<=0)&((vals[:,1:]-vals[:,:-1]).abs()>1e-12)
    denom=fr-fl
    roots=(left-fl*(right-left)/torch.where(denom.abs()>1e-12,denom,torch.ones_like(denom))).clamp(min=left,max=right)
    for _ in range(refine):
        mid=(left+right)/2
        fm=query(mid)-target_N[:,None]
        same=(fm*fl)>0
        left=torch.where(same,mid,left); fl=torch.where(same,fm,fl)
        right=torch.where(same,right,mid); fr=torch.where(same,fr,fm)
    if refine: roots=(left+right)/2
    prior=previous_A.clamp(min=lo,max=hi)
    cand=torch.cat((nodes,roots,prior[:,None]),1)
    valid=torch.cat((torch.ones_like(nodes,dtype=torch.bool),brackets,torch.ones(B,1,dtype=torch.bool,device=nodes.device)),1)
    if refine:
        candidate_force=query(cand)
    else:
        # Prior value from the same 9-node interpolant (not another GRU pass).
        seg=((prior-lo)/(hi-lo).clamp_min(1e-12)*(node_count-1)).floor().long().clamp(0,node_count-2)
        yl=vals.gather(1,seg[:,None])[:,0]; yr=vals.gather(1,(seg+1)[:,None])[:,0]
        xl=nodes.gather(1,seg[:,None])[:,0]; xr=nodes.gather(1,(seg+1)[:,None])[:,0]
        prior_f=yl+(yr-yl)*(prior-xl)/(xr-xl).clamp_min(1e-12)
        candidate_force=torch.cat((vals,target_N[:,None].expand(-1,node_count-1),prior_f[:,None]),1)
    dist=(cand-previous_A[:,None]).abs()
    loss=(candidate_force-target_N[:,None]).abs()+lam_N_per_A*dist
    loss=torch.where(valid,loss,torch.full_like(loss,float('inf')))
    eligible=loss<=loss.min(1,keepdim=True).values+force_tie_N
    select=torch.where(eligible,dist,torch.full_like(dist,float('inf'))).argmin(1)
    chosen=cand.gather(1,select[:,None])[:,0]
    actual=query(chosen[:,None])[:,0]
    dv=vals[:,1:]-vals[:,:-1]
    return chosen,dict(query_force_N=actual,query_residual_N=actual-target_N,
        no_grid_bracket=~brackets.any(1),bracket_count=brackets.sum(1),
        sampled_nonmonotone=(dv>1e-8).any(1)&(dv<-1e-8).any(1),
        sampled_span_N=vals.max(1).values-vals.min(1).values,
        at_bound=(chosen<=1e-5)|(chosen>=1.6-1e-5),
        rate_limited=((chosen-lo).abs()<1e-5)|((chosen-hi).abs()<1e-5) if rate_A is not None else torch.zeros_like(chosen,dtype=torch.bool),
        nodes_A=nodes,node_force_N=vals)
