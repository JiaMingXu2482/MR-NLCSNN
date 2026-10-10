function [I_cmd, F_ach] = v17_inverse_bisection(net, F_des, x, v, a, T, st, n_it)
%V17_INVERSE_BISECTION  Inverse v17 damper model: desired force -> current.
%   %#codegen compatible (no file I/O, no persistent state).
%
%   [I_cmd, F_ach] = v17_inverse_bisection(net, F_des, x, v, a, T, st, n_it)
%
%   The PMPC controller outputs a desired damping force, but the physical
%   CDC damper takes current. This solves for the current I in [0, 1.6] A
%   whose v17-predicted force equals F_des.
%
%   Algorithm (faithful to model_v14.inverse_current, plus semi-active
%   clipping):
%     1. Mask the current channel (u(4) = 0): the current being solved for
%        must not leak into nfl/GRU features.
%     2. Advance the GRU one step from st.h with the masked input to get
%        h_m (st itself is NOT modified).
%     3. Coarse-scan Ie over [0,1] (9 pts), clip F_des to the achievable
%        [min F, max F] (semi-active passivity: an "active" force request
%        saturates at the closest feasible value).
%     4. Bisection on Ie in [0,1] (default 40 iters, matches training code).
%        c/s fast paths use [Ie, Ie] (no lag in inverse, as in training).
%     5. I_cmd = (1 - Ie_sol) * 1.6, clamped to [0, 1.6].
%
%   Inputs (physical units, scalars):
%     net   struct from v17_damper_init
%     F_des desired damper force [N] (from PMPC / QP allocation)
%     x     damper displacement [mm]
%     v     damper velocity     [mm/s]
%     a     damper acceleration [mm/s^2]
%     T     damper temperature  [degC]
%     st    state struct from v17_new_state (uses st.h at START of step)
%     n_it  bisection iterations (default 40; [] for default)
%
%   Outputs:
%     I_cmd  current command [A], feed into v17_damper_step
%     F_ach  force the model predicts at I_cmd [N] (diagnostics)
%
%   Per-corner chain per 100 Hz controller step:
%     [I_cmd, ~] = v17_inverse_bisection(net, F_des, x, v, a, T, st)
%     [F, st]    = v17_damper_step(net, x, v, a, I_cmd, 0, T, 0.01, st)
%
%   Note: st.h used here is the state at the START of the step (before this
%   step's forward pass), exactly like h_prev in run_v14.py.

%#codegen

if nargin < 8 || isempty(n_it)
    n_it = 40;
end
n = net.norm;

% ---- masked kinematics: current channel = 0 ----
u = [(x - n.x_ref) / n.x_scale; ...
      v / n.v_scale; ...
      a / n.a_scale; ...
      0; ...
      0; ...
      (T - n.t_ref) / n.t_scale];
nfl = build_nfl_v12(net, u);
h_m = gru_step_masked(net, nfl, st.h);

% ---- slow heads + friction shape at masked state (fixed during bisection) ----
h2 = h_m(2, :)';
v_out = u(2) * net.cfg.v_out_scale;
feat = [h2; nfl; abs(v_out)];
c_slow = mlp3(net.c_head_W1, net.c_head_b1, net.c_head_W2, ...
              net.c_head_b2, net.c_head_W3, net.c_head_b3, feat);
s_slow = mlp3(net.s_head_W1, net.s_head_b1, net.s_head_W2, ...
              net.s_head_b2, net.s_head_W3, net.s_head_b3, feat);
b      = mlp3(net.b_head_W1, net.b_head_b1, net.b_head_W2, ...
              net.b_head_b2, net.b_head_W3, net.b_head_b3, [h2; nfl]);
phi = tanh(v_out ./ net.sigmas);

% ---- 1. coarse scan + clip to achievable range ----
Nc = 9;
Ie_grid = linspace(0, 1, Nc);
F_grid = zeros(1, Nc);
for j = 1:Nc
    F_grid(j) = force_of_Ie(net, Ie_grid(j), c_slow, s_slow, b, phi, v_out);
end
Flo = min(F_grid);
Fhi = max(F_grid);
Fd = min(max(F_des, Flo), Fhi);

% ---- 2. bisection on Ie in [0,1] ----
f0 = F_grid(1);
f1 = F_grid(Nc);
increasing = f1 > f0;
lo = 0;
hi = 1;
for k = 1:n_it
    mid = 0.5 * (lo + hi);
    Fm = force_of_Ie(net, mid, c_slow, s_slow, b, phi, v_out);
    if (increasing && Fm < Fd) || (~increasing && Fm > Fd)
        lo = mid;
    else
        hi = mid;
    end
end
Ie_sol = 0.5 * (lo + hi);
I_cmd = min(max((1 - Ie_sol) * n.i_scale, 0), n.i_scale);
F_ach = force_of_Ie(net, Ie_sol, c_slow, s_slow, b, phi, v_out);
end


function F = force_of_Ie(net, Ie, c_slow, s_slow, b, phi, v_out)
% Forward force [N] at given Ie (scalar in [0,1]).
% Fast paths use [Ie, Ie] (no lag in inverse), exactly as in training.
i_feat = [Ie; Ie];
c_fast = mlp2(net.c_fast_W1, net.c_fast_b1, net.c_fast_W2, net.c_fast_b2, i_feat);
s_fast = mlp2(net.s_fast_W1, net.s_fast_b1, net.s_fast_W2, net.s_fast_b2, i_feat);
c  = softplus_stable(c_slow + c_fast) + 1e-6 + 2e-3;
sj = softplus_stable(s_slow + s_fast) + 1e-6 + 5e-3/5;
Gc = monoG(net.G_c_W1, net.G_c_b1, net.G_c_W2, net.G_c_b2, ...
           net.G_c_W3, net.G_c_b3, net.G_c_f0, net.G_c_gref, net.k_skip, Ie);
Gs = monoG(net.G_s_W1, net.G_s_b1, net.G_s_W2, net.G_s_b2, ...
           net.G_s_W3, net.G_s_b3, net.G_s_f0, net.G_s_gref, net.k_skip, Ie);
Fn = c * v_out * Gc + (sj' * phi) * Gs + b;
F = Fn * net.norm.f_scale;
end


function nfl = build_nfl_v12(net, u)
x = u(1); v = u(2); a = u(3); ci = u(4); di = u(5); t = u(6);
sv = 0.1;
v_abs = abs(v);
v_pos = max(v, 0);
v_neg = min(v, 0);
sgn_v = tanh(v / sv);
kin = [v; x*v; v*ci; v*t];
rebound = [tanh(2.0*v_pos); tanh(v_pos/sv); ci*tanh(v_pos/sv)];
compression = [v_neg*v_abs; v_neg^3; a*v_neg; ci*v_neg*v_abs];
curr_dyn = [di*v; di*a; abs(di)*ci];
aux = [sgn_v; ci*sgn_v; x^2; x^3; a*ci];
hf_dyn = [abs(a); a*v; abs(a)*sgn_v; ci*abs(a)];
x_abs = abs(x);
slow_gate = exp(-v_abs / sv);
slow = [slow_gate; x*slow_gate; x_abs*slow_gate; ...
        ci*x*slow_gate; ci*x_abs*slow_gate; sgn_v*x_abs];
nfl = [u; kin; rebound; compression; curr_dyn; aux; hf_dyn; slow];
end


function h_new = gru_step_masked(net, nfl, h)
hh0 = h(1, :)';
r0 = sigmoid(net.W_ir0 * nfl + net.b_ir0 + net.W_hr0 * hh0 + net.b_hr0);
z0 = sigmoid(net.W_iz0 * nfl + net.b_iz0 + net.W_hz0 * hh0 + net.b_hz0);
n0 = tanh(net.W_in0 * nfl + net.b_in0 + r0 .* (net.W_hn0 * hh0 + net.b_hn0));
h0_new = (1 - z0) .* n0 + z0 .* hh0;
hh1 = h(2, :)';
r1 = sigmoid(net.W_ir1 * h0_new + net.b_ir1 + net.W_hr1 * hh1 + net.b_hr1);
z1 = sigmoid(net.W_iz1 * h0_new + net.b_iz1 + net.W_hz1 * hh1 + net.b_hz1);
n1 = tanh(net.W_in1 * h0_new + net.b_in1 + r1 .* (net.W_hn1 * hh1 + net.b_hn1));
h1_new = (1 - z1) .* n1 + z1 .* hh1;
h_new = [h0_new'; h1_new'];
end


function y = mlp3(W1, b1, W2, b2, W3, b3, x)
z1 = max(W1 * x + b1, 0);
z2 = max(W2 * z1 + b2, 0);
y  = W3 * z2 + b3;
end


function y = mlp2(W1, b1, W2, b2, x)
z1 = max(W1 * x + b1, 0);
y  = W2 * z1 + b2;
end


function G = monoG(W1, b1, W2, b2, W3, b3, f0, gref, k_skip, Ie)
z1 = tanh(W1 * Ie + b1);
z2 = tanh(W2 * z1 + b2);
f  = W3 * z2 + b3;
G = max(((f - f0) + k_skip * Ie) / gref, 0);
end


function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end


function y = softplus_stable(x)
y = max(x, 0) + log1p(exp(-abs(x)));
end
