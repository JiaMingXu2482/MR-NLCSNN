"""Explicit opt-in v19 runner. No training on import or without --execute-training.

Full-file batches of one: no invented current prehistory at random chunk starts.
Requires reviewed G0 JSON, immutable prepared data, disjoint split manifest.
No CarSim/deployment export and no automatic Stage 3 or final-test evaluation.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mr_nlcsnn.model_v19 import NLCSNN_v19
from mr_nlcsnn.v19_protocol import (set_stage, forward_loss, inverse_loss,
    reference_labels, detached_context, joint_gate)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_dataset(root, partition):
    manifest = root/'alignment_manifest.json'
    complete = json.loads((root/'COMPLETE.json').read_text(encoding='utf-8'))
    if digest(manifest) != complete['manifest_sha256']:
        raise ValueError('manifest differs from completed export')
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    used = set()
    for row in rows:
        if row['split'] in ('train', 'validation', 'final_test'):
            if row['raw_sha256'] in used:
                raise ValueError('duplicate raw hash in active splits')
            used.add(row['raw_sha256'])
    result = []
    for row in rows:
        if row['split'] != partition:
            continue  # NEVER open final-test CSVs during training/validation.
        path = (root/row['relative_path']).resolve()
        if root.resolve() not in path.parents or digest(path) != row['processed_sha256']:
            raise ValueError('invalid processed path/hash: '+str(path))
        with path.open(encoding='utf-8-sig', newline='') as stream:
            table = list(csv.DictReader(stream))
        keys = ('rod_length','velocity','accel','current','current_dot','temp','force')
        values = torch.tensor([[float(r[k]) for k in keys] for r in table])
        timestamps = [float(r['time']) for r in table]
        if any(abs(b-a-0.001) > 1e-6 for a,b in zip(timestamps,timestamps[1:])):
            raise ValueError('data are not uniformly sampled at 1 kHz')
        # Muse fix 2026-10-27 (blocking, recorded in v19_g0/runner_fixes_diff.txt):
        # (a) Parsed current is SIGNED (sensor polarity varies by file: 82/194
        #     train+validation SH files have mean current -nominal, |mean| matches
        #     nominal within 0.05 A for all 194). Physical damper current is a
        #     magnitude (0-1.6 A, unipolar), and the legacy v14 model used
        #     |I| throughout (_current_pair). Range-check the MAGNITUDE here;
        #     sample() converts to magnitude below. The old [0,1.6] check
        #     rejected every negative-polarity file, so training could not start.
        # (b) Edge guard (see edge_guard): files shorter than 2000 samples
        #     (68 train+validation files, 3.07% of train samples; all short
        #     bursts of ~2 cycles) cannot meet a 1000-sample guard on both ends.
        #     Rejecting them would also gut final_test (6/10 files short,
        #     including all 1.6 A files). Short files get a capped 100-sample
        #     guard instead; long files keep the 1000-sample guard exactly.
        if len(values) < 200 or not torch.isfinite(values).all():
            raise ValueError('short/nonfinite file: '+str(path))
        # Tolerance raised 2026-10-27 from 1.63 to 2.10 A after measurement:
        # 56 train+validation files exceed 1.63 A, max 2.011 A, with spikes
        # clustered at current step edges (zero-phase SOS filter overshoot on
        # 0->1.6 A steps; e.g. RHF_RMS10mm_2Hz: 123/7262 samples >1.63, max
        # 1.84 A). sample() clamps magnitude to 1.6 A, so overshoot is treated
        # as full current, not as a higher physical level. 2.10 A still rejects
        # gross channel/scale errors while passing all measured files.
        if values[:,3].abs().max() > 2.10:
            raise ValueError('current magnitude outside [0,2.1] A tolerance: '+str(path))
        result.append(dict(name=path.stem, values=values))
    if not result:
        raise ValueError('empty '+partition)
    return result


def normalization(items):
    xmin = min(float(it['values'][:,0].min()) for it in items)
    xmax = max(float(it['values'][:,0].max()) for it in items)
    return dict(x_ref=(xmin+xmax)/2, x_scale=max((xmax-xmin)/2, 1e-6),
                f_scale=max(max(float(it['values'][:,6].abs().max()) for it in items),1e-6),
                v_scale=1000.0, a_scale=50000.0, i_scale=1.6, t_ref=40.0, t_scale=40.0)


def edge_guard(n):
    """Edge guard in samples. Long files keep the planned 1000-sample (1 s)
    guard on both ends. Short files (<2000 samples, ~2-cycle bursts) get at
    most 100 samples, so they still contribute valid samples instead of an
    empty mask. See the read_dataset note above for the evidence."""
    return 1000 if n > 2000 else min(100, n // 4)


def sample(item, norm):
    v = item['values']
    u = torch.stack(((v[:,0]-norm['x_ref'])/norm['x_scale'], v[:,1]/1000,
                     v[:,2]/50000, (v[:,3].abs()/1.6).clamp(0,1),
                     torch.zeros_like(v[:,4]), (v[:,5]-40)/40), -1)[None]
    f = (v[:,6]/norm['f_scale'])[None]
    valid = torch.ones_like(f, dtype=torch.bool)
    guard = edge_guard(v.shape[0])
    if guard:
        valid[:,:guard] = False; valid[:,-guard:] = False
    return u, f, valid


def fit_sensitivity(reference, items, norm):
    """Train-only p90 at preregistered stride 10, |v|>=10mm/s, 1s edge guard."""
    samples = []
    for it in items:
        u, _, valid = sample(it, norm)
        with torch.no_grad():
            ctx, _ = reference.contexts(u)
        current = u[...,3].detach().requires_grad_(True)
        derivative, = torch.autograd.grad(reference.force_at(ctx, current).sum(), current)
        keep = valid & (u[...,1].abs()*1000 >= 10)
        vals = derivative.detach().abs()*norm['f_scale']/1.6
        samples.append(vals[:,::10][keep[:,::10]])
    vals = torch.cat(samples)
    if not vals.numel():
        raise ValueError('no training sensitivity samples')
    scale = float(torch.quantile(vals, 0.9))
    if not math.isfinite(scale) or scale <= 1e-6:
        raise ValueError('forward model has no usable current sensitivity')
    return scale


def evaluate(model, items, norm, reference=None, sensitivity_scale=None):
    """Teacher-forced-history diagnostics, explicitly NOT closed-loop accuracy."""
    model.eval()
    rows = []
    for it in items:
        u, f, valid = sample(it, norm)
        with torch.no_grad():
            ctx, _ = model.contexts(u)
            pred = model.force_at(ctx, u[...,3])
            err = (pred-f)[valid]*norm['f_scale']; y = f[valid]*norm['f_scale']
            sse, sst = float(err.square().sum()), float(y.square().sum())
        row = dict(name=it['name'], samples=int(valid.sum()), SSE=sse, SST=sst,
                   nrmse_pct=100*math.sqrt(sse/max(sst,1e-12)))
        if reference is not None:
            labels = reference_labels(reference, u, f, valid, f_scale=norm['f_scale'],
                                      sensitivity_scale=sensitivity_scale)
            with torch.no_grad():
                inv = model.inverse_predict_context(ctx, f)
                # Audit only: the cycle value is never added to an optimization loss.
                cycle = (model.force_at(ctx, inv)-f).abs()*norm['f_scale']
                eligible = labels['eligible']
                row.update(eligible_count=int(eligible.sum()),
                    reachable_count=int(labels['reachable'].sum()),
                    speed_ok_count=int(labels['speed_ok'].sum()),
                    inv_abs_sum_A=float(((inv-u[...,3]).abs()*1.6)[eligible].sum()),
                    inv_all_abs_sum_A=float(((inv-u[...,3]).abs()*1.6)[valid].sum()),
                    teacher_abs_sum_A=float(((inv-labels['teacher']).abs()*1.6)[eligible].sum()),
                    cycle_abs_sum_N=float(cycle[valid].sum()),
                    saturation_count=int((((inv<=0.001)|(inv>=0.999)) & valid).sum()),
                    eligible_weight_sum=float((labels['weight']*eligible).sum()),
                    coverage_grid_counts=labels['coverage_grid_counts'])
        rows.append(row)
    def pooled(subset):
        return 100*math.sqrt(sum(r['SSE'] for r in subset)/max(sum(r['SST'] for r in subset),1e-12)) if subset else float('nan')
    result = dict(pooled_nrmse_pct=pooled(rows),
                  hf_nrmse_pct=pooled([r for r in rows if r['name'].startswith('SH_') and '6.62Hz' in r['name']]),
                  rows=rows, history_protocol='strict-past applied current, teacher-forced; not rollout')
    if reference is not None:
        n, e = sum(r['samples'] for r in rows), sum(r['eligible_count'] for r in rows)
        result.update(eligible_fraction=e/n,
            reachable_fraction=sum(r['reachable_count'] for r in rows)/n,
            inv_mae_A=sum(r['inv_abs_sum_A'] for r in rows)/e if e else None,
            teacher_mae_A=sum(r['teacher_abs_sum_A'] for r in rows)/e if e else None,
            inv_all_mae_A=sum(r['inv_all_abs_sum_A'] for r in rows)/n,
            cycle_mae_N=sum(r['cycle_abs_sum_N'] for r in rows)/n,
            saturation_fraction=sum(r['saturation_count'] for r in rows)/n,
            speed_ok_fraction=sum(r['speed_ok_count'] for r in rows)/n,
            coverage_grid={k:sum(r['coverage_grid_counts'][k] for r in rows)/n
                           for k in rows[0]['coverage_grid_counts']})
    return result


def warm_transfer(model, checkpoint):
    """Explicit partial v14->v19 transplant, not checkpoint compatibility.

    Only same-shaped GRU/slow-head tensors. New fast/G/lag/inverse semantics are
    reinitialized; record the exact map. Training-only normalization is shared
    with the cold arm. No silent resizing of 64-wide v18 into 16-wide v19.
    """
    source, own = checkpoint['state_dict'], model.state_dict()
    copied = {}
    for key in own:
        if key.startswith('gru.'):
            old = key
        elif key.split('.')[0] in ('c_head','s_head','b_head'):
            part = key.split('.')
            part[1] = {'0':'0','2':'3','4':'5'}[part[1]]
            old = '.'.join(part)
        else:
            continue
        if old in source and own[key].shape == source[old].shape:
            own[key] = source[old]; copied[key] = old
    if not any(k.startswith('gru.') for k in copied):
        raise ValueError('warm checkpoint has no compatible GRU; cannot perform declared ablation')
    model.load_state_dict(own)
    return copied


def load_v19(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != NLCSNN_v19.architecture:
        raise ValueError('not a v19 checkpoint; use explicit Stage-1 --warm-start for legacy weights')
    model = NLCSNN_v19(**checkpoint['model_cfg'])
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--stage', type=int, choices=(1,2,3), default=1)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--warm-start', type=Path)
    p.add_argument('--inverse-mode', choices=('true','teacher'), default='true')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--tau-ms', type=float, default=11.0)
    p.add_argument('--g0-report', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--resume', type=Path, default=None,
                   help='Resume a v19 session directory (its latest.pth). '
                        '--epochs is the TOTAL epoch count for the session, '
                        'not additional epochs. --output/--checkpoint/--warm-start '
                        'must not be passed with --resume.')
    p.add_argument('--execute-training', action='store_true')
    p.add_argument('--evaluate-final-test', action='store_true')
    a = p.parse_args()
    if a.execute_training and a.evaluate_final_test:
        p.error('training and final-test evaluation must be separate invocations')
    if not a.execute_training and not a.evaluate_final_test:
        print('NO TRAINING. Review docs/TRAIN_PLAN_v19_joint_fwd_inv.md first. Explicit --execute-training required.')
        return
    if a.evaluate_final_test:
        if not a.checkpoint:
            p.error('--checkpoint required')
        model, ckpt = load_v19(a.checkpoint)
        if digest(a.data/'alignment_manifest.json') != ckpt['data_manifest_sha256']:
            raise ValueError('final-test dataset differs from locked training manifest')
        ref = NLCSNN_v19(**ckpt['model_cfg']); ref.load_state_dict(ckpt['reference_state_dict'])
        print(json.dumps(evaluate(model, read_dataset(a.data,'final_test'), ckpt['norm_cfg'],
                                 ref, ckpt['sensitivity_scale_N_per_A']), indent=2))
        return
    # --- Strict v19 resume (local Muse addition, 2026-10-27; NOT part of Codex cac96e8).
    # --epochs is the TOTAL epoch count for the session. LR factor is recomputed
    # from (epoch, total) at each epoch, so resuming with an UNCHANGED total is
    # schedule-identical to an uninterrupted run; resuming with a LARGER total
    # stretches the cosine schedule from the resume point onward (documented in
    # the resume record) and is therefore NOT bit-identical to the original-total
    # run. Resume never loads a legacy v14 checkpoint as a v19 checkpoint:
    # it only loads this runner's own latest.pth in the same session directory.
    if a.resume is not None:
        if a.output is not None or a.checkpoint is not None or a.warm_start is not None:
            p.error('--resume takes the session directory as output; do not pass --output/--checkpoint/--warm-start')
        if not a.g0_report or not 1 <= a.epochs <= 500:
            p.error('reviewed --g0-report and 1..500 total epochs required with --resume')
        a.output = a.resume
    elif not a.g0_report or not a.output or not 1 <= a.epochs <= 500:
        p.error('reviewed --g0-report, new --output, and 1..500 epochs required')
    if a.stage == 3 and a.epochs > 30:
        p.error('Stage 3 is limited to 30 epochs')
    g0 = json.loads(a.g0_report.read_text(encoding='utf-8'))
    manifest_hash = digest(a.data/'alignment_manifest.json')
    if g0.get('passed') is not True or g0.get('data_manifest_sha256') != manifest_hash:
        raise ValueError('G0 review missing or for a different data manifest')
    runner_sha256 = digest(Path(__file__))
    if a.resume is None:
        if a.output.exists():
            raise ValueError('output exists; never overwrite a training session')
    elif not a.output.is_dir() or not (a.output/'latest.pth').exists() or not (a.output/'run_config.json').exists():
        raise ValueError('resume directory must contain latest.pth and run_config.json')
    random.seed(a.seed); torch.manual_seed(a.seed)
    train = read_dataset(a.data,'train'); validation = read_dataset(a.data,'validation')
    norm = normalization(train)
    transfer = {}; start_epoch = 0; best_score = float('inf'); resume_record = None
    if a.resume is None:
        if a.stage == 1:
            model = NLCSNN_v19(init_tau_ms=a.tau_ms)
            if a.checkpoint:
                p.error('Stage 1 uses --warm-start, not --checkpoint')
            if a.warm_start:
                transfer = warm_transfer(model, torch.load(a.warm_start, map_location='cpu', weights_only=True))
            ref, scale = None, None
        else:
            if not a.checkpoint or a.warm_start:
                p.error('Stage 2/3 require --checkpoint and no --warm-start')
            model, ckpt = load_v19(a.checkpoint)
            if ckpt['stage'] != a.stage-1 or ckpt['data_manifest_sha256'] != manifest_hash or ckpt['norm_cfg'] != norm:
                raise ValueError('stage/data/normalization mismatch')
            ref = NLCSNN_v19(**ckpt['model_cfg'])
            ref.load_state_dict(ckpt['reference_state_dict']); ref.eval()
            for par in ref.parameters(): par.requires_grad_(False)
            scale = ckpt['sensitivity_scale_N_per_A']
            if a.stage == 3 and a.inverse_mode != ckpt['inverse_mode']:
                raise ValueError('Stage 3 must keep selected Stage-2 objective')
    else:
        orig_config = json.loads((a.output/'run_config.json').read_text(encoding='utf-8'))
        if orig_config.get('stage') != a.stage or orig_config.get('inverse_mode') != a.inverse_mode:
            raise ValueError('resume stage/inverse_mode must match the original session')
        if orig_config.get('seed') != a.seed:
            raise ValueError('resume seed must match the original session seed')
        if str(a.data) not in (orig_config.get('data'),) and Path(orig_config.get('data','')).resolve() != a.data.resolve():
            raise ValueError('resume data directory must match the original session')
        latest = torch.load(a.output/'latest.pth', map_location='cpu', weights_only=True)
        if latest.get('stage3_rejected'):
            raise ValueError('session was terminated by the Stage-3 gate; it cannot be resumed')
        if latest.get('architecture') != NLCSNN_v19.architecture or latest.get('stage') != a.stage:
            raise ValueError('latest.pth is not a v19 checkpoint for this stage')
        if latest.get('runner_sha256') != runner_sha256:
            raise ValueError('runner file changed since the session started; refusing silent code change mid-training')
        if latest.get('data_manifest_sha256') != manifest_hash or latest.get('norm_cfg') != norm:
            raise ValueError('resume data/normalization mismatch')
        if latest.get('inverse_mode') != a.inverse_mode:
            raise ValueError('resume inverse_mode must match the session')
        model = NLCSNN_v19(**latest['model_cfg'])
        model.load_state_dict(latest['state_dict'])
        if a.stage == 1:
            ref, scale = None, None
        else:
            ref = NLCSNN_v19(**latest['model_cfg'])
            ref.load_state_dict(latest['reference_state_dict']); ref.eval()
            for par in ref.parameters(): par.requires_grad_(False)
            scale = latest['sensitivity_scale_N_per_A']
        start_epoch = int(latest['epoch'])
        if start_epoch >= a.epochs:
            raise ValueError(f'session already at epoch {start_epoch}; --epochs (total) must exceed it')
        resume_record = dict(resumed_from_epoch=start_epoch, total_epochs=a.epochs,
                             runner_sha256=runner_sha256, original_run_config=orig_config.get('warm_transfer'),
                             note='LR cosine recomputed with the resumed total; identical to uninterrupted run only if total unchanged')
    parameters = set_stage(model, a.stage)
    lr = {1:3e-4, 2:1e-4, 3:3e-5}[a.stage]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    baseline = evaluate(ref, validation, norm) if a.stage == 3 else None
    rollback = copy.deepcopy(model.state_dict())
    if a.resume is None:
        a.output.mkdir(parents=True)
        config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        config.update(warm_transfer=transfer, warm_sha256=digest(a.warm_start) if a.warm_start else None,
                      parameter_count=sum(q.numel() for q in model.parameters()), batch='one full file',
                      augmentation='none; current labels and observer history remain clean',
                      runner_sha256=runner_sha256)
        (a.output/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    else:
        # Restore optimizer/RNG/best/rollback AFTER optimizer creation; best.pth
        # is the only best-state store, latest.pth carries the rest.
        # Muse fix 2026-10-27 (resume-test failure): the epoch loop calls
        # random.shuffle(train) at each epoch start, and random.shuffle is a
        # Fisher-Yates in-place permutation whose result depends on the
        # STARTING order as well as the RNG state. Restoring the RNG state
        # alone is NOT enough: a resumed run starts from read_dataset order
        # while an uninterrupted run starts epoch N+1 from epoch N's shuffled
        # order, so the file order (and hence gradients) diverge. Measured
        # 2026-10-27: resumed epoch 2 pooled 43.0339% / train_loss 0.074023
        # vs uninterrupted epoch 2 pooled 42.8504% / train_loss 0.076448.
        # latest.pth therefore also carries the train list order at the end
        # of the completed epoch; restore it here so the next shuffle starts
        # from exactly the same order as an uninterrupted run. Sessions whose
        # latest.pth predates this capture are refused (strict, not silent).
        optimizer.load_state_dict(latest['optimizer_state_dict'])
        random.setstate(latest['random_state']); torch.set_rng_state(latest['torch_rng_state'])
        if 'train_order' not in latest:
            raise ValueError('latest.pth predates train-order capture; cannot resume strictly (restart the session)')
        _by_name = {it['name']: it for it in train}
        if set(_by_name) != set(latest['train_order']) or len(_by_name) != len(latest['train_order']):
            raise ValueError('train-order names do not match the current dataset; cannot resume strictly')
        train = [_by_name[n] for n in latest['train_order']]
        if a.stage == 3:
            if 'rollback_state_dict' not in latest:
                raise ValueError('Stage-3 session predates rollback capture; cannot resume safely')
            rollback = copy.deepcopy(latest['rollback_state_dict'])
        best_path = a.output/'best.pth'
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location='cpu', weights_only=True)
            bm = best_ckpt.get('metrics') or {}
            bs = bm.get('pooled_nrmse_pct') if a.stage == 1 else bm.get('inv_mae_A' if a.inverse_mode == 'true' else 'teacher_mae_A')
            if bs is not None:
                best_score = float(bs)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        (a.output/f'resume_{start_epoch+1}_to_{a.epochs}_{stamp}.json').write_text(
            json.dumps(resume_record, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'RESUMED session={a.output} from epoch {start_epoch} to total {a.epochs} '
              f'(best_score={best_score})', flush=True)
    for epoch in range(start_epoch+1,a.epochs+1):
        start = time.perf_counter(); model.train(); random.shuffle(train)
        factor = min(epoch/5,1.0) if epoch<=5 else 0.01+0.99*(1+math.cos(math.pi*(epoch-5)/max(a.epochs-5,1)))/2
        for group in optimizer.param_groups: group['lr'] = lr*factor
        losses=[]; forward_losses=[]; inverse_losses=[]; gradient_norms=[]; component_norms={}
        for it in train:
            u,f,valid = sample(it,norm)
            optimizer.zero_grad(set_to_none=True)
            ctx,_ = model.contexts(u)
            lf = forward_loss(model.force_at(ctx,u[...,3]),f,valid,
                (f.max()-f.min()).reshape(1), f.new_tensor([3 if it['name'].startswith('ST_') else 1]))
            li = f.new_zeros(())
            if a.stage >= 2:
                labels=reference_labels(ref,u,f,valid,f_scale=norm['f_scale'],sensitivity_scale=scale)
                li,_=inverse_loss(model.inverse_predict_context(ctx,f),u,labels,a.inverse_mode)
            loss = lf if a.stage==1 else li if a.stage==2 else lf+0.02*li
            if not torch.isfinite(loss): raise ValueError('nonfinite loss')
            if a.stage==3 and not losses:
                for name, term in (('forward',lf),('weighted_inverse',0.02*li)):
                    gradients=torch.autograd.grad(term,parameters,retain_graph=True,allow_unused=True)
                    component_norms[name]=math.sqrt(sum(float(g.detach().square().sum()) for g in gradients if g is not None))
            loss.backward()
            gradient_norms.append(float(torch.nn.utils.clip_grad_norm_(parameters,1.0)))
            optimizer.step()
            losses.append(float(loss.detach()))
            forward_losses.append(float(lf.detach())); inverse_losses.append(float(li.detach()))
        # Every epoch for short ablations and immediate Stage-3 rollback; validation only.
        metrics=evaluate(model,validation,norm,ref,scale)
        metrics.update(epoch=epoch,train_loss=sum(losses)/len(losses),seconds=time.perf_counter()-start)
        metrics.update(loss_forward=sum(forward_losses)/len(forward_losses),
            loss_inverse=sum(inverse_losses)/len(inverse_losses),
            gradient_norm_max=max(gradient_norms), first_file_component_gradient_norms=component_norms)
        failed = a.stage==3 and not joint_gate(metrics,baseline)
        metrics['rollback']=failed
        with (a.output/'metrics.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(metrics)+'\n')
        if failed:
            model.load_state_dict(rollback)
            # Terminal marker: a rejected Stage-3 session must NOT be resumed
            # (resuming would re-attempt the rejected epoch from a stale state).
            torch.save(dict(architecture=model.architecture, state_dict=copy.deepcopy(model.state_dict()),
                model_cfg=model.config, norm_cfg=norm, stage=a.stage, inverse_mode=a.inverse_mode,
                epoch=epoch, metrics=metrics, data_manifest_sha256=manifest_hash,
                reference_state_dict=ref.state_dict() if ref is not None else None,
                sensitivity_scale_N_per_A=scale, optimizer_state_dict=optimizer.state_dict(),
                random_state=random.getstate(), torch_rng_state=torch.get_rng_state(),
                train_order=[it['name'] for it in train],
                runner_sha256=runner_sha256, rollback_state_dict=copy.deepcopy(rollback),
                stage3_rejected=True), a.output/'latest.pth')
            print('Stage 3 rejected; retain the input Stage-2 checkpoint.'); break
        score = metrics['pooled_nrmse_pct'] if a.stage==1 else metrics['inv_mae_A' if a.inverse_mode=='true' else 'teacher_mae_A']
        if score is None:
            raise ValueError('no identifiable validation samples; do not publish inverse checkpoint')
        if score < best_score:
            best_score=score
            snapshot=copy.deepcopy(model.state_dict())
            # Stage-1 scale is computed after the run, for the selected model only.
            torch.save(dict(architecture=model.architecture,state_dict=snapshot,model_cfg=model.config,
                norm_cfg=norm,stage=a.stage,inverse_mode=a.inverse_mode,epoch=epoch,metrics=metrics,
                data_manifest_sha256=manifest_hash,reference_state_dict=snapshot if a.stage==1 else ref.state_dict(),
                sensitivity_scale_N_per_A=scale),a.output/'best.pth')
        # Strict-resume checkpoint: saved AFTER the epoch's metrics/best update,
        # so latest.pth always represents a completed epoch, never a half epoch.
        torch.save(dict(architecture=model.architecture, state_dict=copy.deepcopy(model.state_dict()),
            model_cfg=model.config, norm_cfg=norm, stage=a.stage, inverse_mode=a.inverse_mode,
            epoch=epoch, metrics=metrics, data_manifest_sha256=manifest_hash,
            reference_state_dict=copy.deepcopy(model.state_dict()) if a.stage == 1 else ref.state_dict(),
            sensitivity_scale_N_per_A=scale, optimizer_state_dict=optimizer.state_dict(),
            random_state=random.getstate(), torch_rng_state=torch.get_rng_state(),
            train_order=[it['name'] for it in train],
            runner_sha256=runner_sha256, rollback_state_dict=copy.deepcopy(rollback),
            stage3_rejected=False), a.output/'latest.pth')
        print(f'epoch={epoch} validation={metrics["pooled_nrmse_pct"]:.4f}%')
    if a.stage==1:
        selected, ckpt=load_v19(a.output/'best.pth')
        ckpt['sensitivity_scale_N_per_A']=fit_sensitivity(selected,train,norm)
        torch.save(ckpt,a.output/'best.pth')


if __name__ == '__main__':
    main()
