"""Untrained v19 candidate: one causal state and one force map for both directions.

u = [normalized rod, velocity, acceleration, applied current, dI, temperature].
Current is a NONNEGATIVE magnitude normalized by 1.6 A. dI is unused.
At t the observer sees motion/temperature at t and current through t-1 only.
An inverse query never commits state. Commit exactly once using APPLIED current.
CDC polarity is inherited explicitly: I=0 is hard, I=1.6 A is soft.
Free b is retained: this architecture is NOT a passivity guarantee.
No legacy checkpoint is loaded implicitly; this is a new architecture revision.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def nfl_v19(u, sigma=0.1):
    """35-feature smoothed NFL, evaluated only on a strict-past-current input.

    Feature ordering follows model_v12.static_nfl_v12 at 4c9681ff.
    Caller replaces current by previous applied current and sets dI=0 FIRST.
    """
    x, v, a, i, di, t = u.split(1, dim=-1)
    av, vp, vn = v.abs(), v.relu(), -(-v).relu()
    sign = torch.tanh(v / sigma)
    slow = torch.exp(-av / sigma)
    return torch.cat([
        u, v, x*v, v*i, v*t,
        torch.tanh(2*vp), torch.tanh(vp/sigma), i*torch.tanh(vp/sigma),
        vn*av, vn**3, a*vn, i*vn*av,
        di*v, di*a, di.abs()*i,
        sign, i*sign, x**2, x**3, a*i,
        a.abs(), a*v, a.abs()*sign, i*a.abs(),
        slow, x*slow, x.abs()*slow, i*x*slow, i*x.abs()*slow, sign*x.abs(),
    ], dim=-1)


class PositiveLinear(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.full((n_out, n_in), -1.0))
        self.bias = nn.Parameter(torch.zeros(n_out))
        nn.init.normal_(self.weight_raw, -1.0, 0.1)

    def forward(self, x):
        return F.linear(x, F.softplus(self.weight_raw), self.bias)


class MonotonicG(nn.Module):
    """G(q), q in [0,1], increasing, G(0)=0 and G(1)=1.

    q is effective valve hardness, NOT physical current in amperes.
    No detached denominator: gradients match the actual normalized function.
    """
    def __init__(self, hidden=24, skip=0.1):
        super().__init__()
        self.net = nn.Sequential(PositiveLinear(1, hidden), nn.Tanh(),
                                 PositiveLinear(hidden, hidden), nn.Tanh(),
                                 PositiveLinear(hidden, 1))
        self.skip = skip

    def forward(self, q):
        zero = q.new_zeros(1, 1)
        f0 = self.net(zero)
        denominator = self.net(zero + 1) - f0 + self.skip
        return ((self.net(q.unsqueeze(-1)) - f0 + self.skip*q.unsqueeze(-1))
                / denominator.clamp_min(1e-8)).squeeze(-1)


class CurrentLag(nn.Module):
    """Exact causal EMA with an explicit carried initial state and dt.

    Parallel affine-prefix scan, O(log L) tensor operations, no truncated or
    reversed conv1d kernel. Returns normalized EFFECTIVE PHYSICAL current.
    """
    def __init__(self, init_tau_ms=11.0, dt_s=0.001):
        super().__init__()
        if not 1 < init_tau_ms < 100 or dt_s <= 0:
            raise ValueError('tau must be in (1,100) ms; dt must be positive')
        self.dt_s = float(dt_s)
        p = (init_tau_ms - 1) / 99
        self.raw_tau = nn.Parameter(torch.tensor(math.log(p/(1-p))))

    def tau_ms(self):
        return 1 + 99*torch.sigmoid(self.raw_tau)

    def alpha(self):
        return -torch.expm1(-1000*self.dt_s/self.tau_ms())

    def step(self, current, previous):
        return previous + self.alpha()*(current - previous)

    def forward(self, current, initial):
        if current.ndim != 2 or current.shape[1] == 0:
            raise ValueError('current must have shape (batch, nonempty time)')
        decay = 1 - self.alpha()
        b = self.alpha()*current
        # Each element initially denotes affine transform y -> decay*y + b.
        a = torch.ones_like(current)*decay
        stride = 1
        while stride < current.shape[1]:
            b = torch.cat((b[:, :stride], b[:, stride:] + a[:, stride:]*b[:, :-stride]), 1)
            a = torch.cat((a[:, :stride], a[:, stride:]*a[:, :-stride]), 1)
            stride *= 2
        return b + a*initial[:, None]


def head(n_in, hidden, n_out=1, bias=None, sigmoid=False):
    layers = [nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden//2),
              nn.ReLU(), nn.Linear(hidden//2, n_out)]
    if bias is not None:
        nn.init.zeros_(layers[-1].weight)
        nn.init.constant_(layers[-1].bias, bias)
    if sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class State:
    h: torch.Tensor
    previous_current: torch.Tensor
    previous_effective_current: torch.Tensor

    def detach(self):
        return State(self.h.detach(), self.previous_current.detach(),
                     self.previous_effective_current.detach())


@dataclass(frozen=True)
class Context:
    c: torch.Tensor
    s: torch.Tensor
    b: torch.Tensor
    v: torch.Tensor
    inverse_features: torch.Tensor
    previous_effective_current: torch.Tensor


class NLCSNN_v19(nn.Module):
    architecture = 'nlcsnn_v19_strict_past_v1'

    def __init__(self, gru_hidden=16, gru_layers=2, mlp_hidden=48, g_hidden=24,
                 init_tau_ms=11.0, dt_s=0.001, v_out_scale=30.0,
                 sigmas=(0.5, 1.0, 2.0, 4.0, 8.0)):
        super().__init__()
        if min(sigmas) <= 0 or v_out_scale <= 0:
            raise ValueError('velocity scales must be positive')
        self.config = dict(gru_hidden=gru_hidden, gru_layers=gru_layers,
                           mlp_hidden=mlp_hidden, g_hidden=g_hidden,
                           init_tau_ms=init_tau_ms, dt_s=dt_s,
                           v_out_scale=v_out_scale, sigmas=list(sigmas))
        self.gru_hidden, self.gru_layers = gru_hidden, gru_layers
        self.v_out_scale = v_out_scale
        self.register_buffer('sigmas', torch.tensor(sigmas))
        self.gru = nn.GRU(35, gru_hidden, gru_layers, batch_first=True)
        self.c_head = head(gru_hidden + 36, mlp_hidden, bias=-4.0)
        self.s_head = head(gru_hidden + 36, mlp_hidden, len(sigmas), bias=-4.0)
        self.b_head = head(gru_hidden + 35, mlp_hidden, bias=0.0)
        # Retained small side heads; ONLY known previous state enters them.
        self.c_fast = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
        self.s_fast = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, len(sigmas)))
        for net in (self.c_fast, self.s_fast):
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.lag = CurrentLag(init_tau_ms, dt_s)
        self.G_c, self.G_s = MonotonicG(g_hidden), MonotonicG(g_hidden)
        # h + NFL + v_norm + previous_effective_current + F_target
        self.inv_head = head(gru_hidden + 38, mlp_hidden, sigmoid=True)

    def initial_state(self, batch, *, initial_current=0.0):
        """Initial current is known prehistory, never taken from unknown I[t]."""
        p = next(self.parameters())
        if not 0 <= initial_current <= 1:
            raise ValueError('initial_current must be normalized to [0,1]')
        i = p.new_full((batch,), initial_current)
        return State(p.new_zeros(self.gru_layers, batch, self.gru_hidden), i, i.clone())

    def contexts(self, u, state=None):
        """Teacher-forced sequence contexts; each context depends on I[:t] only.

        In rollout use prepare/force_at/commit instead of true future history.
        """
        if u.ndim != 3 or u.shape[-1] != 6 or u.shape[1] == 0:
            raise ValueError('u must have shape (batch, nonempty time, 6)')
        state = self.initial_state(u.shape[0]) if state is None else state
        applied = u[..., 3].clamp(0, 1)
        effective = self.lag(applied, state.previous_effective_current)
        previous = torch.cat((state.previous_current[:, None], applied[:, :-1]), 1)
        prev_eff = torch.cat((state.previous_effective_current[:, None], effective[:, :-1]), 1)
        observer_u = torch.cat((u[..., :3], previous[..., None],
                                torch.zeros_like(u[..., 4:5]), u[..., 5:6]), -1)
        nfl = nfl_v19(observer_u)
        h, last_h = self.gru(nfl, state.h)
        v = u[..., 1]*self.v_out_scale
        z = torch.cat((h, nfl), -1)
        feat = torch.cat((z, v.abs()[..., None]), -1)
        fast = torch.stack((1-previous, 1-prev_eff), -1)
        c = F.softplus(self.c_head(feat).squeeze(-1) + self.c_fast(fast).squeeze(-1)) + 2e-3 + 1e-6
        s = F.softplus(self.s_head(feat) + self.s_fast(fast)) + 5e-3/len(self.sigmas) + 1e-6
        b = self.b_head(z).squeeze(-1)
        inv_feat = torch.cat((z, u[..., 1:2], prev_eff[..., None]), -1)
        ctx = Context(c, s, b, v, inv_feat, prev_eff)
        end = State(last_h, applied[:, -1], effective[:, -1])
        return ctx, end

    def force_at(self, ctx, current):
        """Single shared force expression, current in normalized physical units."""
        effective = self.lag.step(current.clamp(0, 1), ctx.previous_effective_current)
        q = 1-effective
        viscous = ctx.c*ctx.v*self.G_c(q)
        friction = (ctx.s*torch.tanh(ctx.v[..., None]/self.sigmas)).sum(-1)*self.G_s(q)
        return viscous + friction + ctx.b

    def forward(self, u, state=None, return_state=False):
        ctx, end = self.contexts(u, state)
        force = self.force_at(ctx, u[..., 3])
        return (force, end) if return_state else force

    def prepare(self, motion, state):
        """motion (B,6); I/dI at this sample ignored. No state mutation."""
        ctx, end = self.contexts(motion[:, None, :], state)
        return ctx, end.h

    def commit(self, prepared_h, state, applied_current):
        applied = applied_current.clamp(0, 1)
        return State(prepared_h, applied,
                     self.lag.step(applied, state.previous_effective_current))

    def inverse_predict_context(self, ctx, target):
        return self.inv_head(torch.cat((ctx.inverse_features, target[..., None]), -1)).squeeze(-1)

    def inverse_predict(self, u, target, state=None):
        ctx, _ = self.contexts(u, state)
        return self.inverse_predict_context(ctx, target)

    @torch.no_grad()
    def inverse_current(self, ctx, target, iterations=32, flat_tol=1e-7):
        """Bounded monotone solve of EXACT force_at; no GRU/lag state advancement.

        Returns chosen current and diagnostics. At zero sensitivity select the
        previous effective current deterministically; not an identifiable label.
        """
        if target.shape != ctx.v.shape or iterations < 1:
            raise ValueError('target shape must match context; iterations >= 1')
        lo, hi = torch.zeros_like(target), torch.ones_like(target)
        f0, f1 = self.force_at(ctx, lo), self.force_at(ctx, hi)
        lower, upper = torch.minimum(f0, f1), torch.maximum(f0, f1)
        reachable = (target >= lower) & (target <= upper)
        clipped = torch.maximum(lower, torch.minimum(upper, target))
        flat = (upper-lower) <= flat_tol
        increasing = f1 >= f0
        for _ in range(iterations):
            mid = (lo+hi)/2
            fm = self.force_at(ctx, mid)
            move_low = torch.where(increasing, fm < clipped, fm > clipped)
            lo, hi = torch.where(move_low, mid, lo), torch.where(move_low, hi, mid)
        chosen = torch.where(flat, ctx.previous_effective_current, (lo+hi)/2)
        achieved = self.force_at(ctx, chosen)
        return chosen, dict(reachable=reachable, flat=flat, force_low=lower,
                            force_high=upper, target_clipped=clipped,
                            force_achieved=achieved, residual=achieved-target)
