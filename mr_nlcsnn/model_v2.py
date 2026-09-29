"""NLCSNN v2 — rich physics NFL + parallel L1-readable linear path.

Combines the proven CPU-era rich nonlinear feature library (legacy_model.py)
with the paper-style parallel linear path + L1 sparsity (model.py), and fixes
the diagnosed low-frequency failure of the 200-epoch run:

  - prune the dead h*sgn coupling block found near-zero by the first-layer
    column-norm readout (raw h is KEPT — it is the always-on linear state
    feedback that keeps the open-loop rollout stable at low speed);
  - enable the slow-speed gate terms (slow_gate = exp(-80|v|) and its x / i
    products) so the model has an explicit quasi-static current/displacement
    pathway for 1-2 Hz cases;
  - expose the linear-path coefficients (nn_x_linear / nn_y_linear, bias-free)
    as the interpretable, L1-sparsified NFL coefficients.

Output semantics follow legacy (no tanh bound, no explicit dissipation decay):
the legacy stack kept hidden norms stable (~1.8) without them, and tanh
bounding saturates high-amplitude force.
"""
import torch
import torch.nn as nn


class NLCSNN_v2(nn.Module):
    def __init__(
        self,
        u_dim=6,
        h_dim=8,
        hidden=160,
        slow_gain=True,
        extra_dynamic=True,
        prune_h_coupling=True,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.slow_gain = bool(slow_gain)
        self.extra_dynamic = bool(extra_dynamic)
        self.prune_h_coupling = bool(prune_h_coupling)

        # Build feature-name list (also fixes NFL dimension) from a dummy pass.
        self._feature_names = self._build_feature_names(h_dim)
        dummy_u = torch.zeros(1, u_dim)
        dummy_h = torch.zeros(1, h_dim)
        with torch.no_grad():
            total_in = self._get_nfl(dummy_u, dummy_h).shape[1]
        assert total_in == len(self._feature_names), (
            f"NFL dim {total_in} != feature-name count {len(self._feature_names)}"
        )
        self.nfl_dim = total_in

        # --- Nonlinear residual path (legacy deep MLP) ---
        self.nn_x = nn.Sequential(
            nn.Linear(total_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, h_dim, bias=False),
        )
        self.nn_y = nn.Sequential(
            nn.Linear(total_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # --- Linear output readout (interpretable, L1-sparse NFL coefficients) ---
        # Only on the force output. A linear STATE path (nn_x_linear) was tried and
        # removed: it destabilised the latent ODE (open-loop divergence), and the
        # decay/tanh added to counter it then capped the model (underfit, train loss
        # stuck ~0.08). Legacy's deep-MLP state path is both stable and expressive,
        # so the state derivative uses nn_x alone (no linear state path, no decay).
        self.nn_y_linear = nn.Linear(total_in, 1, bias=False)

        print(f"[*] NLCSNN_v2: NFL={total_in} -> hidden={hidden} -> h_dim={h_dim} "
              f"| slow_gain={self.slow_gain} extra_dynamic={self.extra_dynamic} "
              f"prune_h_coupling={self.prune_h_coupling}")

    # ------------------------------------------------------------------ NFL
    def _build_feature_names(self, h_dim):
        names = ["x", "v", "a", "i", "di", "t"]                       # u (6)
        names += ["v_kin", "x*v", "v*i", "v*t"]                       # kin (4)
        names += ["tanh2v+", "tanh50v+", "i*tanh5v+"]                 # rebound (3)
        names += ["vneg*|v|", "vneg^3", "a*vneg", "i*vneg*|v|"]       # compression (4)
        names += ["di*v", "di*a", "|di|*i"]                           # curr_dyn (3)
        names += ["sgn_v", "i*sgn_v", "x^2", "x^3", "a*i"]            # aux (5)
        if self.prune_h_coupling:
            # Keep raw h: it is the always-on linear state feedback (the A*h term
            # that keeps the ODE stable). Only h*sgn is dropped — it is gated and
            # was near-dead in the coefficient readout. Dropping raw h caused
            # open-loop divergence on low-speed cases (no restoring feedback when
            # |v|~0, di~0 make the gated h-terms vanish).
            h_blocks = ["h", "h*|v|", "h*i", "h*di"]
        else:
            h_blocks = ["h", "h*|v|", "h*sgn", "h*i", "h*di"]
        for blk in h_blocks:
            names += [f"{blk}[{k}]" for k in range(h_dim)]
        if self.extra_dynamic:
            names += ["|a|", "a*v", "|a|*sgn", "i*|a|"]               # extra scalar (4)
            names += [f"h*|a|[{k}]" for k in range(h_dim)]            # extra h (h_dim)
        if self.slow_gain:
            names += ["slow_gate", "x*gate", "|x|*gate",
                      "i*x*gate", "i*|x|*gate", "sgn_v*|x|"]          # slow_gain (6)
        return names

    def _get_nfl(self, u, h):
        x, v, a, i, di, t = (
            u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4], u[:, 4:5], u[:, 5:6],
        )
        v_abs = torch.abs(v)
        v_pos = torch.relu(v)
        v_neg = -torch.relu(-v)
        sgn_v = torch.tanh(50.0 * v)

        u_raw = u
        kin = torch.cat([v, x * v, v * i, v * t], dim=-1)
        rebound = torch.cat([
            torch.tanh(2.0 * v_pos),
            torch.tanh(50.0 * v_pos),
            i * torch.tanh(5.0 * v_pos),
        ], dim=-1)
        compression = torch.cat([
            v_neg * v_abs,
            v_neg ** 3,
            a * v_neg,
            i * v_neg * v_abs,
        ], dim=-1)
        curr_dyn = torch.cat([di * v, di * a, torch.abs(di) * i], dim=-1)
        aux = torch.cat([sgn_v, i * sgn_v, x ** 2, x ** 3, a * i], dim=-1)

        if self.prune_h_coupling:
            h_int = torch.cat([h, h * v_abs, h * i, h * di], dim=-1)
        else:
            h_int = torch.cat([h, h * v_abs, h * sgn_v, h * i, h * di], dim=-1)

        parts = [u_raw, kin, rebound, compression, curr_dyn, aux, h_int]

        if self.extra_dynamic:
            a_abs = torch.abs(a)
            hf_dyn = torch.cat([a_abs, a * v, a_abs * sgn_v, i * a_abs], dim=-1)
            parts.append(hf_dyn)
            parts.append(h * a_abs)

        if self.slow_gain:
            x_abs = torch.abs(x)
            # Low-speed gate: ~1 when |v|~0, decays fast as speed rises.
            slow_gate = torch.exp(-torch.clamp(80.0 * v_abs, max=30.0))
            slow = torch.cat([
                slow_gate,
                x * slow_gate,
                x_abs * slow_gate,
                i * x * slow_gate,
                i * x_abs * slow_gate,
                sgn_v * x_abs,
            ], dim=-1)
            parts.append(slow)

        return torch.cat(parts, dim=-1)

    # ---------------------------------------------------------------- dynamics
    def state_derivative(self, u, h):
        # Legacy-style: deep MLP only — stable AND expressive (no linear state
        # path, no dissipation decay).
        return self.nn_x(self._get_nfl(u, h))

    def predict_force(self, u, h):
        nfl = self._get_nfl(u, h)
        return self.nn_y(nfl) + self.nn_y_linear(nfl)

    def forward_output(self, u, h):
        return self.predict_force(u, h)

    # ------------------------------------------------------- interpretability
    def nfl_l1_loss(self):
        """L1 on the linear output-readout weights (the interpretable NFL coefficients)."""
        return self.nn_y_linear.weight.abs().sum()

    def nfl_feature_names(self):
        return list(self._feature_names)

    def nfl_linear_weights(self):
        """Force-path linear coefficients, sorted by |weight| descending.

        Returns list of (feature_name, weight).
        """
        w = self.nn_y_linear.weight.detach().squeeze(0)
        order = w.abs().argsort(descending=True)
        return [(self._feature_names[k], float(w[k])) for k in order]


def rk4_step(model, u, h, dt=0.001):
    k1 = model.state_derivative(u, h)
    k2 = model.state_derivative(u, h + 0.5 * dt * k1)
    k3 = model.state_derivative(u, h + 0.5 * dt * k2)
    k4 = model.state_derivative(u, h + dt * k3)
    return h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
