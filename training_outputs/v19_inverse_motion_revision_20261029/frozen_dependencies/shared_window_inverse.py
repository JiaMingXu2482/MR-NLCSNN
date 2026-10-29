"""Read-only inverse of a frozen force-context surrogate, with physical-unit API.

No new model parameters, no GRU/lag commits, no mutation of context or weights.
The mean over steps 1..N uses exact EMA evolution for a held command. Motion,
c/s/b are frozen: this is NOT an exact inverse of future recurrent rollout.
Current range defaults to the forward model's calibrated 0..1.6 A range.
"""
import math
import torch


def window_force_N(model, ctx, current_A, *, force_scale, steps=10):
    if steps < 1 or not math.isfinite(force_scale) or force_scale <= 0:
        raise ValueError('Positive steps and force_scale required')
    if current_A.shape != ctx.v.shape:
        raise ValueError('Current shape must match context velocity')
    # Exact EMA; no ten sequential calls and no advancement of caller state.
    n = torch.arange(1, steps + 1, device=current_A.device, dtype=current_A.dtype)
    decay = torch.exp(-1000 * model.lag.dt_s * n / model.lag.tau_ms())
    command = current_A / 1.6
    effective = command[..., None] + (ctx.previous_effective_current-command)[..., None]*decay
    q = 1-effective
    viscous = (ctx.c*ctx.v)[..., None]*model.G_c(q)
    friction = (ctx.s*torch.tanh(ctx.v[..., None]/model.sigmas)).sum(-1)
    return (viscous + friction[..., None]*model.G_s(q) + ctx.b[..., None]).mean(-1)*force_scale


@torch.no_grad()
def solve_window_current(model, ctx, target_N, previous_applied_A, *,
                         force_scale, steps=10, iterations=24,
                         span_tol_N=1., lower_A=0., upper_A=1.6):
    """Bounded bisection, clipping explicitly reported, weak authority holds I.

    span_tol_N is a numerical/control-authority policy, not an identifiability
    theorem. Local slope is also returned for caller-specific eligibility.
    Force residual is to ORIGINAL target, including unreachable/flat samples.
    """
    if target_N.shape != ctx.v.shape or previous_applied_A.shape != target_N.shape:
        raise ValueError('Shape mismatch')
    if not (0 <= lower_A < upper_A <= 1.6) or iterations < 1 or span_tol_N < 0:
        raise ValueError('Invalid bounds, iterations or authority tolerance')
    if not torch.isfinite(target_N).all() or not torch.isfinite(previous_applied_A).all():
        raise ValueError('Finite targets and previous currents required')
    def f(i): return window_force_N(model, ctx, i, force_scale=force_scale, steps=steps)
    lo, hi = torch.full_like(target_N, lower_A), torch.full_like(target_N, upper_A)
    f0, f1 = f(lo), f(hi)
    if not torch.isfinite(f0).all() or not torch.isfinite(f1).all():
        raise ValueError('Nonfinite context force')
    flo, fhi = torch.minimum(f0, f1), torch.maximum(f0, f1)
    target_clip = target_N.clamp(min=flo, max=fhi)
    weak = (fhi-flo) <= span_tol_N
    increasing = f1 >= f0
    for _ in range(iterations):
        mid = (lo+hi)/2
        fm = f(mid)
        move_low = torch.where(increasing, fm < target_clip, fm > target_clip)
        lo, hi = torch.where(move_low, mid, lo), torch.where(move_low, hi, mid)
    chosen = torch.where(weak, previous_applied_A.clamp(lower_A, upper_A), (lo+hi)/2)
    achieved = f(chosen)
    left, right = (chosen-.001).clamp(lower_A, upper_A), (chosen+.001).clamp(lower_A, upper_A)
    slope = (f(right)-f(left))/(right-left)
    return chosen, dict(force_low_N=flo, force_high_N=fhi, force_span_N=fhi-flo,
                       weak_authority=weak, reachable=(target_N>=flo)&(target_N<=fhi),
                       target_clipped_N=target_clip, achieved_N=achieved,
                       residual_N=achieved-target_N, local_slope_N_per_A=slope,
                       at_bound=(chosen<=lower_A+1e-5)|(chosen>=upper_A-1e-5))
