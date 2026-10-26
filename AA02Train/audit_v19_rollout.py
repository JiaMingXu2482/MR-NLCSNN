"""Read-only inverse rollout on recorded motion; NOT CarSim closed-loop proof.

Observer runs at 1 kHz; inverse current is selected/held every 10 ms. After the
first 1 s measured-history burn-in, ONLY its own applied commands update state.
Targets are current-sample measured force; no future motion/force is supplied.
No passivity projection, slew limiter, learning or checkpoint modifications.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from AA02Train.train_v19 import load_v19, read_dataset, sample, digest


@torch.no_grad()
def rollout(model, u, target, *, mode='head', burn=1000, hold=10):
    if hold < 1 or mode not in ('head','bounded'):
        raise ValueError('invalid hold or inverse mode')
    state = model.initial_state(u.shape[0])
    currents, forces = [], []
    current = state.previous_current
    for k in range(u.shape[1]):
        ctx, h = model.prepare(u[:,k],state)
        if k < burn:
            current = u[:,k,3]
        elif (k-burn) % hold == 0:
            if mode == 'head':
                current = model.inverse_predict_context(ctx,target[:,k:k+1]).squeeze(1)
            else:
                current,_ = model.inverse_current(ctx,target[:,k:k+1]); current=current.squeeze(1)
        forces.append(model.force_at(ctx,current[:,None]))
        currents.append(current[:,None])
        state = model.commit(h,state,current)
    return torch.cat(currents,1),torch.cat(forces,1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--partition',choices=('validation','final_test'),default='validation')
    p.add_argument('--mode',choices=('head','bounded'),default='head')
    a=p.parse_args()
    model,ckpt=load_v19(a.checkpoint); model.eval()
    if digest(a.data/'alignment_manifest.json') != ckpt['data_manifest_sha256']:
        raise ValueError('dataset mismatch')
    rows=[]
    for it in read_dataset(a.data,a.partition):
        u,f,valid=sample(it,ckpt['norm_cfg'])
        current,pred=rollout(model,u,f,mode=a.mode)
        force_error=(pred-f)*ckpt['norm_cfg']['f_scale']
        rows.append(dict(name=it['name'],force_mae_N=float(force_error[valid].abs().mean()),
            force_rmse_N=float(force_error[valid].square().mean().sqrt()),
            current_mae_A=float(((current-u[...,3]).abs()*1.6)[valid].mean()),
            saturation_fraction=float(((current<=0.001)|(current>=0.999))[valid].float().mean())))
    print(json.dumps(dict(mode=a.mode,observer_dt_ms=1,inverse_hold_ms=10,
        caveat='recorded-motion rollout; current MAE includes nonidentifiable samples; not a vehicle closed-loop test',rows=rows),indent=2))


if __name__=='__main__':
    main()
