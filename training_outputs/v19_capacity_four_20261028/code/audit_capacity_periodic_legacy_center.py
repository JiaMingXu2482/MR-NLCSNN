"""Frozen 1 kHz sine-input periodic-work audit. No protective force projection."""
import math,json
from pathlib import Path
import torch
import train_capacity_four as run
HERE=Path(__file__).resolve().parent
OUT=HERE/'periodic_work_legacy_center_20261028'
FREQS=[2,8,12]
AMPS=[.5,5]
CURRENTS=[0.]
TEMPS=[30.]
DT=.001

def interp(y,t):
    at=t/DT; lo=at.floor().long().clamp(0,len(y)-2); q=at-lo
    return y[lo]*(1-q)+y[lo+1]*q

def integral(y,start,end):
    # Exact physical cycle boundaries with interpolated endpoints; no partial cycles.
    first=math.floor(start/DT)+1; last=math.ceil(end/DT)
    times=torch.cat((torch.tensor([start],dtype=torch.float64),torch.arange(first,last,dtype=torch.float64)*DT,torch.tensor([end],dtype=torch.float64)))
    return float(torch.trapezoid(interp(y,times),times))

@torch.no_grad()
def main():
    run.setup(); OUT.mkdir(exist_ok=False); results=[]; provenance={}
    cfg=json.loads((HERE/'capacity_four_config_20261028.json').read_text())
    for arm in run.ARMS:
        cp=Path(cfg['session_root'])/arm/'final.pth'; ck=torch.load(cp,map_location='cpu',weights_only=True)
        model=run.build(arm).cuda().eval(); model.load_state_dict(ck['state_dict']); norm=ck['norm_cfg']
        provenance[arm]=dict(checkpoint=str(cp),sha256=run.base.digest(cp),norm_cfg=norm)
        for freq in FREQS:
            warm=max(15,math.ceil(5*freq)); cycles=warm+5; duration=cycles/freq
            t=torch.arange(math.ceil(duration/DT)+2,device='cuda',dtype=torch.float64)*DT
            omega=2*math.pi*freq
            for amp in AMPS:
                cases=[(i,temp) for temp in TEMPS for i in CURRENTS]
                x=279.1375+amp*torch.sin(omega*t); v=amp*omega*torch.cos(omega*t); a=-amp*omega**2*torch.sin(omega*t)
                u=torch.stack([torch.stack(((x-norm['x_ref'])/norm['x_scale'],v/1000,a/50000,
                    torch.full_like(t,i/1.6),torch.zeros_like(t),torch.full_like(t,(temp-40)/40)),-1) for i,temp in cases]).float()
                if arm=='D': u=u[...,[0,1,3,5]]
                ctx,_=model.contexts(u); current=u[...,2 if arm=='D' else 3]
                q=1-model.lag.step(current,ctx.previous_effective_current)
                f=model.force_at(ctx,current)*norm['f_scale']
                dissip=(ctx.c*ctx.v*model.G_c(q)+(ctx.s*torch.tanh(ctx.v[...,None]/model.sigmas)).sum(-1)*model.G_s(q))*norm['f_scale']
                bb=ctx.b*norm['f_scale']
                force=f.double().cpu(); velocity=v.cpu(); power=(force*velocity/1000)
                pd=dissip.double().cpu()*velocity/1000; pb=bb.double().cpu()*velocity/1000
                for j,(i,temp) in enumerate(cases):
                    works=[integral(power[j],k/freq,(k+1)/freq) for k in range(warm,cycles)]
                    k=cycles-1; wd=integral(pd[j],k/freq,(k+1)/freq); wb=integral(pb[j],k/freq,(k+1)/freq)
                    phase=torch.linspace(0,1,2001,dtype=torch.float64)/freq
                    last=interp(force[j],phase+k/freq); prev=interp(force[j],phase+(k-1)/freq)
                    repeat=float((last-prev).square().mean().sqrt()); rms=float(last.square().mean().sqrt())
                    abswork=integral(power[j].abs(),k/freq,(k+1)/freq)
                    tol=max(1e-7,1e-4*abswork)
                    row=dict(arm=arm,frequency_Hz=freq,amplitude_mm=amp,current_A=i,temperature_C=temp,
                        position_center_mm=279.1375,dt_s=DT,warmup_cycles=warm,cycle_work_J=works,
                        mean_cycle_work_J=sum(works)/5,mean_power_W=sum(works)/5*freq,last_passive_work_J=wd,last_b_work_J=wb,
                        waveform_repeat_rms_N=repeat,waveform_repeat_relative=repeat/max(rms,1e-6),
                        work_cycle_spread_J=max(works)-min(works),negative_tolerance_J=tol,
                        negative=all(w < -tol for w in works),instant_negative_power_fraction=float((last*(amp*omega*torch.cos(omega*phase))<0).double().mean()))
                    results.append(row)
                    if freq in (2,8,12) and amp in (.5,5) and i==0 and temp==30:
                        run.base.write_json(OUT/f'cycle_{arm}_{amp}mm_{freq}Hz.json',dict(row=row,phase= (phase*freq).tolist(),force_N=last.tolist(),
                            displacement_mm=(amp*torch.sin(omega*phase)).tolist(),velocity_mm_s=(amp*omega*torch.cos(omega*phase)).tolist()))
            print(f'{arm} {freq}Hz done; negatives so far={sum(r["negative"] for r in results if r["arm"]==arm)}',flush=True)
        run.base.write_json(OUT/f'{arm}_results.json',[r for r in results if r['arm']==arm]); del model
    run.base.write_json(OUT/'all_results.json',results)
    run.base.write_json(OUT/'provenance.json',dict(models=provenance,script_sha256=run.base.digest(__file__),parameter_updates=0,
        force_projection=False,force_gain=1,final_test_access=False,inputs='Analytic sine x,v,a; fixed current; 1kHz; position center279.1375mm'))
    print('COMPLETE',flush=True)
if __name__=='__main__': main()

