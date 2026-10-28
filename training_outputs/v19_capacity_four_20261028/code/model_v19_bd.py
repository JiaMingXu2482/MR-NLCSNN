"""Local B/D ablations only. Import the user-supplied v19 source on PYTHONPATH.

No baseline source edits, weight transplantation, A training or data access.
B input: x,v,I,dI,T (27 NFL); D: x,v,I,T (24 NFL).
Like baseline, observer dI is zero. B/D is NOT a test of dI information.
"""
import torch
from torch import nn
from torch.nn import functional as F
from mr_nlcsnn.model_v19 import NLCSNN_v19, State, Context, CurrentLag, MonotonicG, head


INPUTS = {'B': ('x','v','I','dI','T'), 'D': ('x','v','I','T')}
FEATURES = {
    'B': ('x','v','I','dI','T','v_kin','xv','vI','vT',
          'tanh2vp','tanhvp','I_tanhvp','vn_abs_v','vn3','I_vn_abs_v',
          'dI_v','abs_dI_I','sign_v','I_sign_v','x2','x3',
          'slow','x_slow','abs_x_slow','I_x_slow','I_abs_x_slow','sign_v_abs_x'),
    'D': ('x','v','I','T','v_kin','xv','vI','vT',
          'tanh2vp','tanhvp','I_tanhvp','vn_abs_v','vn3','I_vn_abs_v',
          'sign_v','I_sign_v','x2','x3',
          'slow','x_slow','abs_x_slow','I_x_slow','I_abs_x_slow','sign_v_abs_x'),
}


def nfl_bd(u, arm, sigma=0.1):
    if arm not in INPUTS or u.shape[-1] != len(INPUTS[arm]):
        raise ValueError('B requires 5 input columns; D requires 4')
    x, v, i = u[...,0:1],u[...,1:2],u[...,2:3]
    t = u[...,-1:]
    av, vp, vn = v.abs(),v.relu(),-(-v).relu()
    sign, slow = torch.tanh(v/sigma),torch.exp(-av/sigma)
    pieces=[u,v,x*v,v*i,v*t,torch.tanh(2*vp),torch.tanh(vp/sigma),
            i*torch.tanh(vp/sigma),vn*av,vn**3,i*vn*av]
    if arm=='B':
        di=u[...,3:4]
        pieces += [di*v,di.abs()*i]
    pieces += [sign,i*sign,x**2,x**3,slow,x*slow,x.abs()*slow,
               i*x*slow,i*x.abs()*slow,sign*x.abs()]
    result=torch.cat(pieces,-1)
    if result.shape[-1] != len(FEATURES[arm]):
        raise RuntimeError('NFL feature schema mismatch')
    return result


class NLCSNN_v19_BD(NLCSNN_v19):
    """Same construction order/defaults as v19, with physically removed features."""
    def __init__(self, arm, gru_hidden=16, gru_layers=2, mlp_hidden=48, g_hidden=24,
                 init_tau_ms=11.0, dt_s=0.001, v_out_scale=30.0,
                 sigmas=(0.5,1.0,2.0,4.0,8.0)):
        nn.Module.__init__(self)
        if arm not in INPUTS:
            raise ValueError('Only B and D are authorized')
        if (gru_hidden,gru_layers,mlp_hidden,g_hidden,init_tau_ms,dt_s,tuple(sigmas)) != (
                16,2,48,24,11.0,0.001,(0.5,1.0,2.0,4.0,8.0)):
            raise ValueError('Ablation architecture is locked to user specification')
        self.arm=arm
        self.architecture='nlcsnn_v19_strict_past_ablation_'+arm
        self.nfl_dim=len(FEATURES[arm])
        self.config=dict(arm=arm,gru_hidden=gru_hidden,gru_layers=gru_layers,
                         mlp_hidden=mlp_hidden,g_hidden=g_hidden,init_tau_ms=init_tau_ms,
                         dt_s=dt_s,v_out_scale=v_out_scale,sigmas=list(sigmas))
        self.gru_hidden,self.gru_layers=gru_hidden,gru_layers
        self.v_out_scale=v_out_scale
        self.register_buffer('sigmas',torch.tensor(sigmas))
        self.gru=nn.GRU(self.nfl_dim,gru_hidden,gru_layers,batch_first=True)
        self.c_head=head(gru_hidden+self.nfl_dim+1,mlp_hidden,bias=-4.0)
        self.s_head=head(gru_hidden+self.nfl_dim+1,mlp_hidden,len(sigmas),bias=-4.0)
        self.b_head=head(gru_hidden+self.nfl_dim,mlp_hidden,bias=0.0)
        self.c_fast=nn.Sequential(nn.Linear(2,32),nn.ReLU(),nn.Linear(32,1))
        self.s_fast=nn.Sequential(nn.Linear(2,32),nn.ReLU(),nn.Linear(32,len(sigmas)))
        for net in (self.c_fast,self.s_fast):
            nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
        self.lag=CurrentLag(init_tau_ms,dt_s)
        self.G_c,self.G_s=MonotonicG(g_hidden),MonotonicG(g_hidden)
        self.inv_head=head(gru_hidden+self.nfl_dim+3,mlp_hidden,sigmoid=True)

    def contexts(self,u,state=None):
        if u.ndim != 3 or u.shape[-1] != len(INPUTS[self.arm]) or u.shape[1]==0:
            raise ValueError('invalid compact input shape')
        state=self.initial_state(u.shape[0]) if state is None else state
        applied=u[...,2].clamp(0,1)
        effective=self.lag(applied,state.previous_effective_current)
        previous=torch.cat((state.previous_current[:,None],applied[:,:-1]),1)
        prev_eff=torch.cat((state.previous_effective_current[:,None],effective[:,:-1]),1)
        inputs=[u[...,:2],previous[...,None]]
        if self.arm=='B':
            inputs.append(torch.zeros_like(u[...,3:4]))
        inputs.append(u[...,-1:])
        nfl=nfl_bd(torch.cat(inputs,-1),self.arm)
        h,last_h=self.gru(nfl,state.h)
        v=u[...,1]*self.v_out_scale
        z=torch.cat((h,nfl),-1)
        feat=torch.cat((z,v.abs()[...,None]),-1)
        fast=torch.stack((1-previous,1-prev_eff),-1)
        c=F.softplus(self.c_head(feat).squeeze(-1)+self.c_fast(fast).squeeze(-1))+2e-3+1e-6
        s=F.softplus(self.s_head(feat)+self.s_fast(fast))+5e-3/len(self.sigmas)+1e-6
        b=self.b_head(z).squeeze(-1)
        inv_feat=torch.cat((z,u[...,1:2],prev_eff[...,None]),-1)
        return Context(c,s,b,v,inv_feat,prev_eff),State(last_h,applied[:,-1],effective[:,-1])

    def forward(self,u,state=None,return_state=False):
        ctx,end=self.contexts(u,state)
        force=self.force_at(ctx,u[...,2])
        return (force,end) if return_state else force
