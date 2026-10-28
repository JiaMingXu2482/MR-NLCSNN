"""Isolated A capacity ablation; optional strictly past causal current derivative.

No changes to upstream v19 or old checkpoints. The derivative is recomputed
from committed current history, never from unknown current at the query time.
"""
from dataclasses import dataclass
import train_a_local as base
import torch
from torch.nn import functional as F
from mr_nlcsnn.model_v19 import NLCSNN_v19,Context,nfl_v19

@dataclass(frozen=True)
class HistoryState:
    h: torch.Tensor
    previous_current: torch.Tensor
    previous_effective_current: torch.Tensor
    previous_previous_current: torch.Tensor

    def detach(self):
        return HistoryState(*(x.detach() for x in (self.h,self.previous_current,
            self.previous_effective_current,self.previous_previous_current)))

class CapacityModel(NLCSNN_v19):
    architecture='v19_A_capacity_strict_past_di_v1'
    def __init__(self,history_di=True,di_scale_A_per_s=1600.,**kwargs):
        super().__init__(**kwargs)
        if di_scale_A_per_s<=0: raise ValueError('Positive derivative scale required')
        self.history_di=history_di; self.di_scale_A_per_s=float(di_scale_A_per_s)
        self.config.update(history_di=history_di,di_scale_A_per_s=self.di_scale_A_per_s)
        self.arm='A'; self.nfl_dim=35

    def initial_state(self,batch,*,initial_current=0.):
        s=super().initial_state(batch,initial_current=initial_current)
        return HistoryState(s.h,s.previous_current,s.previous_effective_current,s.previous_current.clone())

    def observer(self,u,state):
        applied=u[...,3].clamp(0,1)
        previous=torch.cat((state.previous_current[:,None],applied[:,:-1]),1)
        previous_previous=torch.cat((state.previous_previous_current[:,None],previous[:,:-1]),1)
        # physical A/s divided by a fixed scale; for dt=1ms this equals delta(I/1.6A).
        di=(previous-previous_previous)*1.6/(self.lag.dt_s*self.di_scale_A_per_s)
        if not self.history_di: di=torch.zeros_like(di)
        observer=torch.cat((u[...,:3],previous[...,None],di[...,None],u[...,5:6]),-1)
        return observer,applied,previous

    def contexts(self,u,state=None):
        if u.ndim!=3 or u.shape[-1]!=6 or not u.shape[1]: raise ValueError('Expected B,T,6')
        state=self.initial_state(u.shape[0]) if state is None else state
        observer,applied,previous=self.observer(u,state)
        effective=self.lag(applied,state.previous_effective_current)
        prev_eff=torch.cat((state.previous_effective_current[:,None],effective[:,:-1]),1)
        nfl=nfl_v19(observer); h,last_h=self.gru(nfl,state.h)
        v=u[...,1]*self.v_out_scale; z=torch.cat((h,nfl),-1)
        feat=torch.cat((z,v.abs()[...,None]),-1)
        fast=torch.stack((1-previous,1-prev_eff),-1)
        c=F.softplus(self.c_head(feat).squeeze(-1)+self.c_fast(fast).squeeze(-1))+2e-3+1e-6
        s=F.softplus(self.s_head(feat)+self.s_fast(fast))+5e-3/len(self.sigmas)+1e-6
        b=self.b_head(z).squeeze(-1)
        inv_feat=torch.cat((z,u[...,1:2],prev_eff[...,None]),-1)
        ctx=Context(c,s,b,v,inv_feat,prev_eff)
        return ctx,HistoryState(last_h,applied[:,-1],effective[:,-1],previous[:,-1])

    def commit(self,prepared_h,state,applied_current):
        applied=applied_current.clamp(0,1)
        return HistoryState(prepared_h,applied,self.lag.step(applied,state.previous_effective_current),state.previous_current)
