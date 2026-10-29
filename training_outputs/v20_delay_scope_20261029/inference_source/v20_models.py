"""v20 forward models. All models predict normalised damping force at 1 kHz
from the 5 inputs [x, v, a, I, T]. There is NO CurrentLag, NO TauNet and NO
separate tau parameter in any model: delay is left to the GRU hidden state,
as the user decided for v20. (A separate tau target was ruled out on
2026-10-29: delay is distributed over driver ramp, valve dynamics, GRU
state and coefficients, so a single separately-trained tau is not
identifiable from force loss alone.)

Models (letters are v20 labels, NOT the v19 input-ablation A/B/D):
  A raw_gru:      raw [x,v,a,I,T] -> GRU(16x2) -> MLP([h, raw_now])
  B nfl_gru:      NFL(31)         -> GRU(16x2) -> MLP([h, NFL_now])
  C nfl_mlp:      NFL(31)         -> MLP, no GRU (tests whether memory is needed)
  D parallel:     raw -> GRU(16x2) temporal branch, current NFL branch,
                  -> MLP([h, NFL_now]). This is the GPT chat's preferred
                  parallel fusion, and it is also what its "GRU-first then
                  fuse NFL" scheme reduces to once NFL is computed from the
                  current physical inputs (never from the hidden state, as
                  the GPT chat itself warned). B and C in that chat collapse
                  to this one implementation, so it is trained once, not twice.
  E nfl_small_gru: NFL(31) -> GRU(8x2) -> MLP([h, NFL_now]) (compression)
  G raw_small_gru: raw [x,v,a,I,T] -> GRU(8x2) -> MLP([h, raw_now]).
                  Same as A but with the small GRU. Completes the 2x2
                  factorial with A (raw,16), B (NFL,16), E (NFL,8), G (raw,8):
                    A vs B  -> effect of NFL at hidden 16
                    G vs E  -> effect of NFL at hidden 8
                    A vs G  -> effect of shrinking GRU on raw input
                    B vs E  -> effect of shrinking GRU on NFL input
  F csb_nolag:    OPTIONAL control, not part of the GPT comparison. The
                  v19-style c/s/b + monotonic-G force formula, but with NO
                  CurrentLag: q = 1 - I_now, c/s/b heads from [h, NFL].
                  Run it separately to answer whether deleting c/s/b helps;
                  it is not in the parallel launcher's default list.

MLP heads are one hidden layer of 16 ReLU units, as in the GPT chat's
starting sizes (8-16), except F, which keeps the v19 head sizes (48->24)
so that F is a fair v19-formula control, not a shrunk version of it.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def nfl_v20(u, sigma=0.1):
    """31-feature NFL from [x, v, a, I, T]. Same forms as v19's NFL, minus
    the dI column and the three dI-derived features (di*v, di*a, |di|*i).
    v is the normalised velocity (mm/s / 1000), so sigma=0.1 matches v19.
    Feature order (31):
      x, v, a, i, t, v, x*v, v*i, v*t,
      tanh(2*vp), tanh(vp/sigma), i*tanh(vp/sigma),
      vn*|v|, vn^3, a*vn, i*vn*|v|,
      sign, i*sign, x^2, x^3, a*i,
      |a|, a*v, |a|*sign, i*|a|,
      slow, x*slow, |x|*slow, i*x*slow, i*|x|*slow, sign*|x|
    where vp=relu(v), vn=-relu(-v), sign=tanh(v/sigma), slow=exp(-|v|/sigma).
    """
    x, v, a, i, t = u.split(1, dim=-1)
    av, vp, vn = v.abs(), v.relu(), -(-v).relu()
    sign = torch.tanh(v / sigma)
    slow = torch.exp(-av / sigma)
    return torch.cat(
        [
            u,
            v,
            x * v,
            v * i,
            v * t,
            torch.tanh(2 * vp),
            torch.tanh(vp / sigma),
            i * torch.tanh(vp / sigma),
            vn * av,
            vn**3,
            a * vn,
            i * vn * av,
            sign,
            i * sign,
            x**2,
            x**3,
            a * i,
            a.abs(),
            a * v,
            a.abs() * sign,
            i * a.abs(),
            slow,
            x * slow,
            x.abs() * slow,
            i * x * slow,
            i * x.abs() * slow,
            sign * x.abs(),
        ],
        dim=-1,
    )


def _mlp(in_dim, hidden=16, out_dim=1):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class _PositiveLinear(nn.Module):
    """Linear layer with softplus weights, so its output is monotone in its
    input. Used only by the F (c/s/b) control, copied from v19's MonotonicG."""

    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.full((out_f, in_f), -1.0))
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        return F.linear(x, F.softplus(self.weight_raw), self.bias)


class _MonotonicG(nn.Module):
    """G(q), q in [0,1], increasing, G(0)=0, G(1)=1. Same form as v19."""

    def __init__(self, hidden=24, skip=0.1):
        super().__init__()
        self.net = nn.Sequential(
            _PositiveLinear(1, hidden),
            nn.Tanh(),
            _PositiveLinear(hidden, hidden),
            nn.Tanh(),
            _PositiveLinear(hidden, 1),
        )
        self.skip = skip

    def forward(self, q):
        zero = q.new_zeros(1, 1)
        f0 = self.net(zero)
        denominator = self.net(zero + 1) - f0 + self.skip
        return ((self.net(q.unsqueeze(-1)) - f0 + self.skip * q.unsqueeze(-1)) / denominator.clamp_min(1e-8)).squeeze(-1)


ARCHES = {
    "A": "raw_gru",
    "B": "nfl_gru",
    "C": "nfl_mlp",
    "D": "parallel",
    "E": "nfl_small_gru",
    "F": "csb_nolag",
    "G": "raw_small_gru",
}


class V20Model(nn.Module):
    def __init__(self, arch="nfl_gru", gru_hidden=16, gru_layers=2, mlp_hidden=16):
        super().__init__()
        if arch not in ("raw_gru", "nfl_gru", "nfl_mlp", "parallel", "nfl_small_gru",
                         "raw_small_gru", "csb_nolag"):
            raise ValueError("unknown arch: " + str(arch))
        self.arch = arch
        if arch in ("nfl_small_gru", "raw_small_gru"):
            gru_hidden = 8
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.config = dict(arch=arch, gru_hidden=gru_hidden, gru_layers=gru_layers, mlp_hidden=mlp_hidden)
        if arch == "nfl_mlp":
            self.mlp = _mlp(31, mlp_hidden, 1)
            return
        gru_in = 5 if arch in ("raw_gru", "parallel", "raw_small_gru") else 31
        self.gru = nn.GRU(gru_in, gru_hidden, gru_layers, batch_first=True)
        if arch == "csb_nolag":
            # v19 head sizes, kept deliberately: F is a formula control.
            def head(n_in, n_out, bias=None):
                layers = [nn.Linear(n_in, 48), nn.ReLU(), nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, n_out)]
                if bias is not None:
                    nn.init.zeros_(layers[-1].weight)
                    nn.init.constant_(layers[-1].bias, bias)
                return nn.Sequential(*layers)

            self.c_head = head(gru_hidden + 32, 1, bias=-4.0)
            self.s_head = head(gru_hidden + 32, 5, bias=-4.0)
            self.b_head = head(gru_hidden + 31, 1, bias=0.0)
            self.sigmas = (0.5, 1.0, 2.0, 4.0, 8.0)
            self.register_buffer("sigmas_t", torch.tensor(self.sigmas))
            self.G_c = _MonotonicG(24)
            self.G_s = _MonotonicG(24)
            return
        mlp_in = gru_hidden + 5 if arch in ("raw_gru", "raw_small_gru") else gru_hidden + 31
        self.mlp = _mlp(mlp_in, mlp_hidden, 1)

    def forward(self, u):
        """u: (batch, time, 5). Returns normalised force (batch, time)."""
        nfl = nfl_v20(u)
        if self.arch == "nfl_mlp":
            return self.mlp(nfl).squeeze(-1)
        gru_in = u if self.arch in ("raw_gru", "parallel", "raw_small_gru") else nfl
        h, _ = self.gru(gru_in)
        if self.arch == "csb_nolag":
            # v in the force formula is the v19 scaled velocity.
            v_s = u[..., 1] * 30.0
            feat = torch.cat((h, nfl, v_s.abs()[..., None]), -1)
            c = F.softplus(self.c_head(feat).squeeze(-1)) + 2e-3
            s = F.softplus(self.s_head(feat)) + 5e-3 / 5
            b = self.b_head(torch.cat((h, nfl), -1)).squeeze(-1)
            # NO lag: q uses the current timestep's current, not a lagged one.
            q = 1 - u[..., 3]
            viscous = c * v_s * self.G_c(q)
            friction = (s * torch.tanh(v_s[..., None] / self.sigmas_t)).sum(-1) * self.G_s(q)
            return viscous + friction + b
        mlp_in = (torch.cat((h, u), -1) if self.arch in ("raw_gru", "raw_small_gru")
                  else torch.cat((h, nfl), -1))
        return self.mlp(mlp_in).squeeze(-1)
