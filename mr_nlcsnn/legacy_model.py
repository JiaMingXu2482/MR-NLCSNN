import torch
import torch.nn as nn


class PhysicsForce(nn.Module):
    """CDC patent physical model (CN121409644B) as a learnable parametric force.

    F = { v>=0:  (c + c0(i))*v + 2*a1(i)*tanh(b1*v + d1*sign(x)) + f1
          v<0:   c*v        + 2*a2(i)*tanh(b2*v + d2*sign(x)) + k(i)*a + f2 }

    Coefficients are learnable nn.Parameters initialized to patent values. Inputs are the
    normalized state u; physical units are recovered via scales (u[:,1] is already m/s,
    a via a_scale/1000, |i| via i_scale). Output is divided by f_scale to return a
    normalized force, matching nn_y's output space.
    """

    def __init__(self, v_scale=1000.0, a_scale=50000.0, x_scale=62.0, i_scale=1.6, f_scale=7662.0):
        super().__init__()
        self.register_buffer("v_fac", torch.tensor(float(v_scale) / 1000.0))
        self.register_buffer("a_fac", torch.tensor(float(a_scale) / 1000.0))
        self.register_buffer("i_fac", torch.tensor(float(i_scale)))
        self.register_buffer("f_scale", torch.tensor(float(f_scale)))
        self.c = nn.Parameter(torch.tensor(900.0))
        self.f1 = nn.Parameter(torch.tensor(320.0))
        self.f2 = nn.Parameter(torch.tensor(-106.0))
        self.b1 = nn.Parameter(torch.tensor(2.0))
        self.b2 = nn.Parameter(torch.tensor(-10.8))
        self.d1 = nn.Parameter(torch.tensor(0.055))
        self.d2 = nn.Parameter(torch.tensor(0.3))
        # current polynomials, highest power first
        self.c0 = nn.Parameter(torch.tensor([17060.0, -57820.0, 63150.0, -26970.0, 9896.0]))
        self.a1 = nn.Parameter(torch.tensor([-14430.0, 47510.0, -52090.0, 22580.0, -5273.0]))
        self.a2 = nn.Parameter(torch.tensor([513.7, -907.8]))
        self.k = nn.Parameter(torch.tensor([-172.4, 208.0, 126.5]))

    @staticmethod
    def _poly(coeffs, x):
        r = torch.zeros_like(x)
        for c in coeffs:
            r = r * x + c
        return r

    def forward(self, u):
        v = u[:, 1:2] * self.v_fac
        a = u[:, 2:3] * self.a_fac
        xs = torch.sign(u[:, 0:1])
        i = torch.abs(u[:, 3:4]) * self.i_fac
        c0 = self._poly(self.c0, i)
        a1 = self._poly(self.a1, i)
        a2 = self.a2[0] * i + self.a2[1]
        k = self.k[0] * i * i + self.k[1] * i + self.k[2]
        reb = (self.c + c0) * v + 2.0 * a1 * torch.tanh(self.b1 * v + self.d1 * xs) + self.f1
        com = self.c * v + 2.0 * a2 * torch.tanh(self.b2 * v + self.d2 * xs) + k * a + self.f2
        return torch.where(v >= 0, reb, com) / self.f_scale


class LegacyNLCSNN(nn.Module):
    """CPU-era NLCSNN architecture kept for reproducible baseline training."""

    def __init__(
        self,
        u_dim=6,
        h_dim=6,
        hidden=160,
        extra_dynamic_nfl=False,
        direct_nfl_output=False,
        slow_gain_nfl=False,
        hysteresis_gain_nfl=False,
        nfl_slim=False,
        physics_nfl=False,
        physics_temp=False,
        physics_path=False,
        phys_scales=None,
        input_clamp=5.0,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.extra_dynamic_nfl = bool(extra_dynamic_nfl)
        self.direct_nfl_output = bool(direct_nfl_output)
        self.slow_gain_nfl = bool(slow_gain_nfl)
        self.hysteresis_gain_nfl = bool(hysteresis_gain_nfl)
        self.nfl_slim = bool(nfl_slim)
        self.physics_nfl = bool(physics_nfl)
        self.physics_temp = bool(physics_temp)
        self.physics_path = bool(physics_path)
        # Inference robustness: clamp normalized inputs to ±input_clamp before building
        # the NFL. Normal data tops out at |u|≈1.75, so ±5 never touches valid samples,
        # but it bounds draw-wire sensor glitches (e.g. a 721mm displacement spike → x³
        # explosion → open-loop state divergence). 0/None disables.
        self.input_clamp = float(input_clamp) if input_clamp else 0.0

        dummy_u = torch.zeros(1, u_dim)
        dummy_h = torch.zeros(1, h_dim)
        with torch.no_grad():
            total_in = self._get_nfl(dummy_u, dummy_h).shape[1]

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
        if self.direct_nfl_output:
            self.nn_y_direct = nn.Linear(total_in, 1, bias=False)
            nn.init.zeros_(self.nn_y_direct.weight)
        else:
            self.nn_y_direct = None

        self.nfl_dim = total_in
        if self.physics_path:
            scales = phys_scales if phys_scales is not None else (1000.0, 50000.0, 62.0, 1.6, 7662.0)
            self.physics = PhysicsForce(*scales)
            # Zero-init the residual force head so the initial prediction equals the
            # physics baseline; the residual then learns only what physics misses
            # (standard physics-guided initialization, avoids random residual wrecking it).
            nn.init.zeros_(self.nn_y[-1].weight)
            nn.init.zeros_(self.nn_y[-1].bias)
        else:
            self.physics = None

    def _get_nfl(self, u, h):
        if self.input_clamp:
            u = torch.clamp(u, -self.input_clamp, self.input_clamp)
        x, v, a, i, di, t = (
            u[:, 0:1],
            u[:, 1:2],
            u[:, 2:3],
            u[:, 3:4],
            u[:, 4:5],
            u[:, 5:6],
        )

        v_abs = torch.abs(v)
        v_pos = torch.relu(v)
        v_neg = -torch.relu(-v)
        sgn_v = torch.tanh(50.0 * v)

        if self.physics_nfl:
            # Fluid-dynamics first-principles NFL (derived from orifice/valve flow, not
            # parametric Bouc-Wen). Laminar Hagen-Poiseuille (∝v), turbulent Bernoulli
            # (∝|v|v), valve-area current modulation A_o(i), fluid inertia (∝a), gas
            # chamber polytropic stiffness, seal friction; state h learns the dynamics.
            laminar = torch.cat([v, v_pos, v_neg], dim=-1)
            turbulent = torch.cat([v_abs * v, v_pos * v, v_neg * v_abs], dim=-1)
            current_mod = torch.cat([v * i, v * i * i, v_abs * v * i, v_abs * v * i * i], dim=-1)
            inertia = torch.cat([a, a * v], dim=-1)
            gas = torch.cat([x, x * x, x * x * x], dim=-1)
            friction = torch.cat([sgn_v, i * sgn_v], dim=-1)
            # Relief/blow-off valve saturation: continuous orifice flow (laminar/turbulent)
            # can't produce velocity-independent force plateaus. tanh(k·v) is a smooth
            # valve-opening switch; rebound & compression valves at two slopes, plus a
            # current-modulated (solenoid threshold) valve. This was the gap that forced
            # the state h to "fake" the plateau (h over-loaded/drifting).
            valve = torch.cat(
                [
                    torch.tanh(5.0 * v_pos), torch.tanh(20.0 * v_pos),
                    torch.tanh(5.0 * v_neg), torch.tanh(20.0 * v_neg),
                    i * torch.tanh(10.0 * v_pos), i * torch.tanh(10.0 * v_neg),
                ],
                dim=-1,
            )
            state = torch.cat([h, h * v, h * v_abs, h * i, h * a], dim=-1)
            parts = [laminar, turbulent, current_mod, inertia, gas, friction]
            if self.physics_temp:
                # Temperature: gas-chamber preload & oil viscosity are temp-dependent
                # (per-condition baseline term, addresses the offset error component).
                parts.append(torch.cat([t, t * sgn_v, t * v], dim=-1))
            parts += [valve, state]
            return torch.cat(parts, dim=-1)

        kin = torch.cat([v, x * v, v * i, v * t], dim=-1)
        rebound = torch.cat(
            [
                torch.tanh(2.0 * v_pos),
                torch.tanh(50.0 * v_pos),
                i * torch.tanh(5.0 * v_pos),
            ],
            dim=-1,
        )
        compression = torch.cat(
            [
                v_neg * v_abs,
                v_neg**3,
                a * v_neg,
                i * v_neg * v_abs,
            ],
            dim=-1,
        )
        curr_dyn = torch.cat([di * v, di * a, torch.abs(di) * i], dim=-1)
        aux = torch.cat([sgn_v, i * sgn_v, x**2, x**3, a * i], dim=-1)
        if self.nfl_slim:
            # Data-driven prune (E40 importance analysis): drop compression(4) +
            # curr_dyn(3) + h*|v|(8) + h*di(8) = 23 low-importance dims (<0.4% each).
            h_int = torch.cat([h, h * sgn_v, h * i], dim=-1)
            parts = [u, kin, rebound, h_int, aux]
        else:
            h_int = torch.cat([h, h * v_abs, h * sgn_v, h * i, h * di], dim=-1)
            parts = [u, kin, rebound, compression, curr_dyn, h_int, aux]

        if self.extra_dynamic_nfl:
            a_abs = torch.abs(a)
            # High-frequency force errors are mostly gain errors. These terms expose
            # acceleration magnitude and acceleration-state coupling without changing
            # the CPU-era state-space training semantics.
            hf_dyn = torch.cat(
                [
                    a_abs,
                    a * v,
                    a_abs * sgn_v,
                    i * a_abs,
                    h * a_abs,
                ],
                dim=-1,
            )
            parts.append(hf_dyn)

        if self.slow_gain_nfl:
            x_abs = torch.abs(x)
            slow_gate = torch.exp(-torch.clamp(80.0 * v_abs, max=30.0))
            slow_gain = torch.cat(
                [
                    slow_gate,
                    x * slow_gate,
                    x_abs * slow_gate,
                    i * x * slow_gate,
                    i * x_abs * slow_gate,
                    sgn_v * x_abs,
                    h * x_abs * slow_gate,
                ],
                dim=-1,
            )
            parts.append(slow_gain)

        if self.hysteresis_gain_nfl:
            x_abs = torch.abs(x)
            i_abs = torch.abs(i)
            x2 = x * x
            x_abs2 = x_abs * x_abs
            hyst_gain = torch.cat(
                [
                    x * sgn_v,
                    x_abs * v,
                    x_abs * v_abs,
                    i * x * sgn_v,
                    i * x_abs * sgn_v,
                    i_abs * x_abs * sgn_v,
                    i * x_abs * v,
                    i_abs * x_abs * v_abs,
                    x2 * sgn_v,
                    x2 * i * sgn_v,
                    x_abs2 * v,
                    h * x_abs,
                    h * x_abs * i,
                    h * x_abs * sgn_v,
                ],
                dim=-1,
            )
            parts.append(hyst_gain)

        return torch.cat(parts, dim=-1)

    def state_derivative(self, u, h):
        return self.nn_x(self._get_nfl(u, h))

    def predict_force(self, u, h):
        nfl = self._get_nfl(u, h)
        y = self.nn_y(nfl)
        if self.nn_y_direct is not None:
            y = y + self.nn_y_direct(nfl)
        if self.physics is not None:
            y = y + self.physics(u)
        return y

    def forward_output(self, u, h):
        return self.predict_force(u, h)


def rk4_step(model, u, h, dt=0.001):
    k1 = model.state_derivative(u, h)
    k2 = model.state_derivative(u, h + 0.5 * dt * k1)
    k3 = model.state_derivative(u, h + 0.5 * dt * k2)
    k4 = model.state_derivative(u, h + dt * k3)
    return h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
