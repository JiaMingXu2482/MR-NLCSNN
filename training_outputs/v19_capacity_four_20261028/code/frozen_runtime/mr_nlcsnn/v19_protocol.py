"""v19 loss/gate primitives; importing this module never starts training.

All current losses use normalized physical current I/1.6 A. Force is F/Fscale.
Sensitivity/masks/teachers MUST use a frozen Stage-1 reference, including in
Stage 3, so the live model cannot lower its loss by changing its own mask.
"""
from dataclasses import fields
import math
import torch
from .model_v19 import Context


def detached_context(ctx):
    return Context(**{f.name: getattr(ctx, f.name).detach() for f in fields(ctx)})


def set_stage(model, stage):
    if stage not in (1, 2, 3):
        raise ValueError('stage must be 1, 2 or 3')
    for name, p in model.named_parameters():
        inverse = name.startswith('inv_head.')
        p.requires_grad_(inverse if stage == 2 else (not inverse if stage == 1 else True))
    return [p for p in model.parameters() if p.requires_grad]


def reference_labels(reference, u, force, valid, *, f_scale, sensitivity_scale):
    """Fixed reference context, teacher and dF/dI (N/A), no parameter gradients.

    sensitivity_scale is a PRECOMPUTED TRAIN-only p90 |dF/dI| (N/A), at true
    currents with |v|>=10 mm/s after warmup. Store it in every checkpoint.
    Autograd differentiates the entire shared map, including CurrentLag/G.
    A finite difference of G alone is not the force sensitivity.
    """
    if not (math.isfinite(f_scale) and f_scale > 0 and
            math.isfinite(sensitivity_scale) and sensitivity_scale > 0):
        raise ValueError('force and sensitivity scales must be finite positive')
    with torch.no_grad():
        ctx, _ = reference.contexts(u)
        teacher, info = reference.inverse_current(ctx, force)
    probe = u[..., 3].detach().clamp(0, 1).requires_grad_(True)
    with torch.enable_grad():
        predicted = reference.force_at(detached_context(ctx), probe)
        derivative, = torch.autograd.grad(predicted.sum(), probe)
    sensitivity = (derivative.detach().abs()*f_scale/1.6)
    weight = (sensitivity/sensitivity_scale).clamp(0, 1)
    speed_ok = u[..., 1].abs()*1000 >= 10
    reach = info['reachable'] & ~info['flat']
    eligible = valid & speed_ok & reach & (weight >= 0.2)
    coverage_grid = {f'v{v}_w{w}': int((valid & reach &
        (u[..., 1].abs()*1000 >= v) & (weight >= w)).sum())
        for v in (5, 10, 20) for w in (0.1, 0.2, 0.3)}
    return dict(teacher=teacher.detach(), weight=weight.detach(),
                eligible=eligible.detach(), reachable=(valid & reach),
                speed_ok=(valid & speed_ok), sensitivity_N_per_A=sensitivity,
                flat=info['flat'], valid=valid, coverage_grid_counts=coverage_grid)


def inverse_loss(prediction, u, labels, mode):
    """Separate true-current and teacher experiments; no implicit mixture.

    Self-normalized sensitivity weights per file, then equal file weighting.
    No ST or force-sign weighting on inverse task. Empty eligible files excluded;
    an entirely empty batch produces differentiable zero plus coverage=0.
    """
    if mode not in ('true', 'teacher'):
        raise ValueError('mode must be true or teacher')
    target = u[..., 3].detach().clamp(0, 1) if mode == 'true' else labels['teacher']
    weight = labels['weight']*labels['eligible']
    denominator = weight.sum(1)
    per_file = (weight*(prediction-target).square()).sum(1)/denominator.clamp_min(1e-12)
    present = denominator > 0
    loss = (per_file*present).sum()/present.sum().clamp_min(1)
    n = labels['valid'].sum().clamp_min(1)
    stats = {key+'_fraction': float(labels[key].sum()/n)
             for key in ('eligible', 'reachable', 'speed_ok')}
    stats['files_eligible'] = int(present.sum())
    return loss, stats


def forward_loss(prediction, force, valid, p2p, st_weight):
    weight = torch.where(force < 0, 1.5, 1.0)*valid
    per_file = (weight*(prediction-force).square()).sum(1)/valid.sum(1).clamp_min(1)
    return (per_file/p2p.clamp_min(1e-6).square()*st_weight).mean()


def joint_gate(candidate, baseline, max_pp=1.0):
    """Validation-only absolute percentage-point guard, not relative percent."""
    keys = ('pooled_nrmse_pct', 'hf_nrmse_pct')
    return all(math.isfinite(candidate[k]) and math.isfinite(baseline[k]) and
               candidate[k] <= baseline[k] + max_pp for k in keys)
