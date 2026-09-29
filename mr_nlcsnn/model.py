import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )

    def forward(self, x):
        return x + self.ff(self.norm(x))


class NLCSNN(nn.Module):
    def __init__(self, u_dim=6, h_dim=8, hidden=128):
        super().__init__()
        self.h_dim = h_dim
        self.dissipation_alpha = nn.Parameter(torch.tensor([-3.0]))
        self.output_scale = 1.4
        self.output_uses_state = True

        dummy_u = torch.zeros(1, u_dim)
        dummy_h = torch.zeros(1, h_dim)
        with torch.no_grad():
            self.nfl_dim = self._get_nfl(dummy_u, dummy_h).shape[1]

        print(f"[*] Architecture: {self.nfl_dim} NFL -> {hidden} Hidden -> {h_dim} States")
        print(f"[*] Parallel linear path: NFL -> state_deriv + output (L1-sparse)")

        # --- Latent state partition (TASK 4) ---
        # h = [z_elastic(2), z_dissipative(3), z_hysteretic(3)]
        #   [0:2]    elastic    — displacement-correlated, slow decay
        #   [2:5]    dissipative — velocity-correlated, damping behavior
        #   [5:8]    hysteretic — long memory, bounded evolution
        self.partition = {
            "elastic":     (0, 2),
            "dissipative": (2, 5),
            "hysteretic":  (5, 8),
        }

        # --- Nonlinear path ---
        self.input_proj = nn.Sequential(
            nn.Linear(self.nfl_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.trunk = nn.Sequential(
            ResidualMLPBlock(hidden, dropout=0.05),
            ResidualMLPBlock(hidden, dropout=0.05),
        )
        self.nn_x_nonlinear = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, h_dim, bias=False),
        )
        self.nn_y_nonlinear = nn.Sequential(
            nn.Linear(hidden + h_dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        # --- Parallel linear path (NFL coefficients learned here) ---
        self.nn_x_linear = nn.Linear(self.nfl_dim, h_dim, bias=False)
        self.nn_y_linear = nn.Linear(self.nfl_dim, 1, bias=False)

    def _get_nfl(self, u, h):
        """Physics-motivated NFL (TASK 1 + post-TASK-5 fix).

        Measurement (11): x, v, |v|, x*v, v^2, sign(v), I, I*v, I*sign(v), I^2, tanh(v)
        State coupling (h_dim): h — hidden state itself, enables linear path to read ODE state.
        Total: 11 + h_dim features.
        """
        x, v, _a, i, _di, _t = u[:,0:1], u[:,1:2], u[:,2:3], u[:,3:4], u[:,4:5], u[:,5:6]
        v_abs = torch.abs(v)
        sign_v = torch.sign(v)

        measurements = torch.cat([
            # Kinematic (6)
            x,                              # displacement
            v,                              # velocity
            v_abs,                          # velocity magnitude (amplitude-dependent damping)
            x * v,                          # velocity-displacement coupling
            v**2,                           # quadratic damping (orifice flow)
            sign_v,                         # velocity direction (compression vs rebound)

            # Current coupling (4) — I as control variable, NOT dI/dt
            i,                              # current (electromagnetic control input)
            i * v,                          # current-velocity coupling (semi-active damping)
            i * sign_v,                     # current-dependent yield force asymmetry
            i**2,                           # electromagnetic force nonlinearity

            # Hysteresis (1)
            torch.tanh(v),                  # saturation / smooth direction transition
        ], dim=-1)

        return torch.cat([measurements, h], dim=-1)

    def state_derivative(self, u, h):
        nfl = self._get_nfl(u, h)

        # Nonlinear path (residual dynamics not captured by NFL)
        feats = self.trunk(self.input_proj(nfl))
        dh_nonlinear = self.nn_x_nonlinear(feats)

        # Linear path (NFL terms directly mapped to state derivative)
        dh_linear = self.nn_x_linear(nfl)

        # Dissipation decay (keeps hidden state bounded)
        v, di = u[:, 1:2], u[:, 4:5]
        decay = torch.exp(self.dissipation_alpha) * h * (torch.abs(v) + torch.abs(di) + 0.1)

        return dh_nonlinear + dh_linear - decay

    def predict_force(self, u, h):
        nfl = self._get_nfl(u, h)

        # Nonlinear path
        h_for_output = h if self.output_uses_state else torch.zeros_like(h)
        feats = self.trunk(self.input_proj(nfl))
        f_nonlinear = self.nn_y_nonlinear(torch.cat([feats, h_for_output], dim=-1))

        # Linear path (NFL terms directly mapped to force output)
        f_linear = self.nn_y_linear(nfl)

        raw_force = f_nonlinear + f_linear
        return self.output_scale * torch.tanh(raw_force / self.output_scale)

    def nfl_l1_loss(self):
        """L1 penalty on linear-path weights for NFL sparsity (paper future-work suggestion)."""
        return self.nn_x_linear.weight.abs().sum() + self.nn_y_linear.weight.abs().sum()

    def nfl_linear_weights(self):
        """Return the linear-path coefficients for interpretability (sorted by abs value)."""
        w = self.nn_y_linear.weight.detach().squeeze(0)
        sorted_idx = w.abs().argsort(descending=True)
        return [(i.item(), w[i].item()) for i in sorted_idx]


def rk4_step(model, u, h, dt=0.001):
    k1 = model.state_derivative(u, h)
    k2 = model.state_derivative(u, h + 0.5 * dt * k1)
    k3 = model.state_derivative(u, h + 0.5 * dt * k2)
    k4 = model.state_derivative(u, h + dt * k3)
    return h + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
