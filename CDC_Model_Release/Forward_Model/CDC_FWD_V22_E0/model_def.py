"""V22-AE Hybrid: five models on the V20 A/E GRU+MLP backbones.

    F_t = F_base,t + v_hat_t * dF_t * q_t   (+ b_a * a_hat_t for EGTA)

Hard contract (task book section 0 and 2):
  * observer input is [x, v, a, I_prev, T] with I_prev[0] = 0 and
    I_prev[t] = clip(|I[t-1]|/1.6, 0, 1).  v20_data.sample() already stores
    clip(|I|/1.6,0,1) in channel 3, so I_prev is that channel shifted by one.
  * the CURRENT-STEP candidate current may only reach G -> q -> current gain.
    It must not reach the GRU, the NFL features, F_base or dF.  Therefore the
    fusion input for the A backbone is [h, obs5] (observer vector, I_prev),
    NOT [h, u5]; see test_v22.py::test_candidate_isolated.
  * q[0] = g[0] per file, reset between files.
  * tau is a single global positive scalar in [1, 50] ms, init 11 ms,
    alpha = 1 - exp(-dt/tau) with tau in SECONDS (the V20 MTAU bug is fixed
    here and covered by test_v22.py::test_tau_time_constant).

Models (params must match the task book; all <= 3500):
  AGT  3220   A backbone + G + tau
  E0   2073   E backbone, force head only, no G / tau  (forward baseline only)
  EG   2187   E backbone + G, q = g
  EGT  2188   E backbone + G + tau
  EGTA 2189   EGT + scalar b_a * a_hat
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TAU_MIN_MS, TAU_MAX_MS, TAU_INIT_MS = 1.0, 50.0, 11.0
MODELS = ("AGT", "E0", "EG", "EGT", "EGTA")

# per-model structure.  E01 is E0 with a 3-layer GRU (depth ablation only:
# same backbone, same heads, no G / no tau).
SPECS = {
    "AGT":  dict(backbone="A", g=True,  tau=True,  acc=False, layers=2),
    "E0":   dict(backbone="E", g=False, tau=False, acc=False, layers=2),
    "E01":  dict(backbone="E", g=False, tau=False, acc=False, layers=3),
    "EG":   dict(backbone="E", g=True,  tau=False, acc=False, layers=2),
    "EGT":  dict(backbone="E", g=True,  tau=True,  acc=False, layers=2),
    "EGTA": dict(backbone="E", g=True,  tau=True,  acc=True,  layers=2),
}
ALLOWED = tuple(SPECS)


# --------------------------------------------------------------------------
def observer_input(u):
    """u = [x, v, a, I, T] at t  ->  [x, v, a, I[t-1], T] (first sample 0)."""
    prev = torch.cat((torch.zeros_like(u[:, :1, 3:4]), u[:, :-1, 3:4]), 1)
    return torch.cat((u[..., :3], prev, u[..., 4:5]), -1)


def nfl_v22(u_obs, sigma=0.1):
    """31 features from [x, v, a, i, t]; i must already be I[t-1].

    Same feature order as V20's nfl_v20 / v20_improvement_plan section 2.2.
    """
    x, v, a, i, t = u_obs.split(1, dim=-1)
    av, vp, vn = v.abs(), v.relu(), -(-v).relu()
    sign = torch.tanh(v / sigma)
    slow = torch.exp(-av / sigma)
    return torch.cat(
        [u_obs, v, x * v, v * i, v * t,
         torch.tanh(2 * vp), torch.tanh(vp / sigma), i * torch.tanh(vp / sigma),
         vn * av, vn ** 3, a * vn, i * vn * av,
         sign, i * sign, x ** 2, x ** 3, a * i,
         a.abs(), a * v, a.abs() * sign, i * a.abs(),
         slow, x * slow, x.abs() * slow, i * x * slow, i * x.abs() * slow,
         sign * x.abs()],
        dim=-1,
    )


class PositiveLinear(nn.Module):
    """Weights are softplus(raw) > 0, so the composite map is monotone."""

    def __init__(self, ni, no):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.full((no, ni), -1.0))
        self.bias = nn.Parameter(torch.zeros(no))

    def forward(self, x):
        return F.linear(x, F.softplus(self.weight_raw), self.bias)


class MonotonicG(nn.Module):
    """G_pos: 1 -> 8 -> 8 -> 1 with tanh, normalised so G_pos(0)=0, G_pos(1)=1."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(PositiveLinear(1, 8), nn.Tanh(),
                                 PositiveLinear(8, 8), nn.Tanh(), PositiveLinear(8, 1))

    def forward(self, magnitude):
        x = magnitude.unsqueeze(-1)
        f0 = self.net(torch.zeros_like(x))
        den = self.net(torch.ones_like(x)) - f0 + 0.1
        return ((self.net(x) - f0 + 0.1 * x) / den.clamp_min(1e-8)).squeeze(-1)


def lag_scan(g, tau_s, dt=0.001, chunk=256):
    """Exact causal EMA, q[0] = g[0].

        alpha = 1 - exp(-dt/tau)     (tau in SECONDS)
        q[k] = q[k-1] + alpha*(g[k] - q[k-1])

    Chunked exact scan: inside a chunk the recursion is a causal convolution
    with kernel alpha*(1-alpha)^j (reversed for F.conv1d), and the carried
    state decays by (1-alpha)^(j+1).  Result is independent of chunk size.
    """
    alpha = -torch.expm1(-dt / tau_s)
    out = [g[:, :1]]
    state = g[:, :1]
    for start in range(1, g.shape[1], chunk):
        x = g[:, start:start + chunk]
        n = x.shape[1]
        j = torch.arange(n, device=g.device, dtype=g.dtype)
        kernel = (alpha * (1 - alpha) ** j).flip(0).reshape(1, 1, n)
        y = F.conv1d(F.pad(x.unsqueeze(1), (n - 1, 0)), kernel).squeeze(1)
        y = y + state * (1 - alpha) ** (j + 1)
        out.append(y)
        state = y[:, -1:]
    return torch.cat(out, 1)


# --------------------------------------------------------------------------
class V22Model(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        if model_id not in SPECS:
            raise ValueError("unknown V22 model: " + str(model_id))
        self.model_id = model_id
        spec = SPECS[model_id]
        self.backbone = spec["backbone"]
        self.use_g = spec["g"]
        self.use_tau = spec["tau"]
        self.use_acc = spec["acc"]
        self.gru_layers = spec["layers"]

        if self.backbone == "A":
            self.gru = nn.GRU(5, 16, self.gru_layers, batch_first=True)
            self.trunk = nn.Sequential(nn.Linear(21, 16), nn.ReLU())
            fused = 21
        else:
            self.gru = nn.GRU(31, 8, self.gru_layers, batch_first=True)
            self.trunk = nn.Sequential(nn.Linear(39, 16), nn.ReLU())
            fused = 39
        self.fused = fused

        if self.use_g:
            self.head_base = nn.Linear(16, 1)
            self.head_gain = nn.Linear(16, 1)
            nn.init.zeros_(self.head_base.weight)
            nn.init.zeros_(self.head_base.bias)
            nn.init.zeros_(self.head_gain.weight)
            nn.init.constant_(self.head_gain.bias, -1.0)     # softplus(-1) ~ 0.31
            self.G = MonotonicG()
        else:
            self.head_force = nn.Linear(16, 1)
            nn.init.zeros_(self.head_force.weight)
            nn.init.zeros_(self.head_force.bias)

        if self.use_tau:
            p = (TAU_INIT_MS - TAU_MIN_MS) / (TAU_MAX_MS - TAU_MIN_MS)
            self.tau_raw = nn.Parameter(torch.tensor(math.log(p / (1 - p))))
        if self.use_acc:
            self.b_a = nn.Parameter(torch.zeros(()))

    # -- state / parameters -------------------------------------------------
    def tau_s(self):
        return (TAU_MIN_MS + (TAU_MAX_MS - TAU_MIN_MS) * torch.sigmoid(self.tau_raw)) / 1000.0

    def tau_ms(self):
        return self.tau_s() * 1000.0

    # -- pieces -------------------------------------------------------------
    def current_map(self, i_candidate):
        """i_candidate already in clip(|I|/1.6,0,1).  g = 1 - G_pos(i), monotone
        non-increasing in current magnitude (F-v slope falls with current)."""
        return 1.0 - self.G(i_candidate.clamp(0, 1))

    def observe(self, u):
        """Returns (z, g) for the whole sequence.  z feeds F_base / dF only."""
        obs = observer_input(u)
        gru_in = obs if self.backbone == "A" else nfl_v22(obs)
        h, _ = self.gru(gru_in)
        feat = torch.cat((h, obs), -1) if self.backbone == "A" \
            else torch.cat((h, nfl_v22(obs)), -1)
        return self.trunk(feat), obs

    def head_parts(self, z):
        base = self.head_base(z).squeeze(-1)
        dgain = F.softplus(self.head_gain(z)).squeeze(-1)
        return base, dgain

    # -- forward ------------------------------------------------------------
    def forward(self, u, parts=False):
        z, obs = self.observe(u)
        v_hat, a_hat = obs[..., 1], obs[..., 2]
        if not self.use_g:                       # E0: plain force regression
            force = self.head_force(z).squeeze(-1)
            if parts:
                return dict(force=force, base=force, dF=None, q=None, g=None, z=z)
            return force

        base, dgain = self.head_parts(z)
        g = self.current_map(u[..., 3])           # candidate current at t
        q = lag_scan(g, self.tau_s()) if self.use_tau else g
        force = base + v_hat * dgain * q
        if self.use_acc:
            force = force + self.b_a * a_hat
        if parts:
            return dict(force=force, base=base, dF=dgain, q=q, g=g, z=z,
                        acc=(self.b_a * a_hat) if self.use_acc else None)
        return force

    # -- pure trial for the 100 Hz inverse solve ---------------------------
    def trial_force(self, z, v_hat, a_hat, q_prev, i_candidate, dt=0.01):
        """No state mutation: uses q_prev, returns the candidate force and the
        candidate q without committing it."""
        base, dgain = self.head_parts(z)
        g = self.current_map(i_candidate)
        if self.use_tau:
            alpha = -torch.expm1(-dt / self.tau_s())
            q = q_prev + alpha * (g - q_prev)
        else:
            q = g
        force = base + v_hat * dgain * q
        if self.use_acc:
            force = force + self.b_a * a_hat
        return force, q


def count_params(model):
    groups = {}
    for n, p in model.named_parameters():
        key = n.split(".")[0]
        groups[key] = groups.get(key, 0) + p.numel()
    return dict(total=sum(p.numel() for p in model.parameters()), groups=groups)


EXPECTED = {"AGT": 3220, "E0": 2073, "EG": 2187, "EGT": 2188, "EGTA": 2189}
