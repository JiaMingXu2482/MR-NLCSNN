function [F, st_new] = v17_damper_step(net, x, v, a, I, dI, T, dt, st)
%V17_DAMPER_STEP  One macro time step of the v17 NLCSNN damper model.
%   %#codegen compatible (no file I/O, no persistent state).
%
%   [F, st_new] = v17_damper_step(net, x, v, a, I, dI, T, dt, st)
%
%   Inputs (physical units, scalars):
%     net  struct from v17_damper_init
%     x    damper displacement            [mm]
%     v    damper velocity                [mm/s]
%     a    damper acceleration            [mm/s^2]
%     I    damper current command         [A]
%     dI   current derivative             [A/s] (forced 0: training zeroed
%          this channel, see train_v14.py collate)
%     T    damper temperature             [degC]
%     dt   macro step length              [s]  (e.g. 0.01 for 100 Hz)
%     st   state struct from v17_new_state (h: 2x16, ie_buf: KMAXx1)
%
%   Outputs:
%     F      damper force                  [N]
%     st_new updated state struct
%
%   Inside, the macro step is substepped at 1 ms (the training sample
%   step): n_sub = round(dt/0.001), inputs held constant (zero-order hold).
%   Matches the validated recipe: GRU+lag @1 kHz, bisection inverse @100 Hz.
%
%   Simulink wiring (one corner): MATLAB Function block calling this,
%   with a Unit Delay feeding st back to st_new. Use one block instance
%   per corner so each keeps its own state.
%
%   Per 1 ms substep (matches training rollout):
%     nfl = static_nfl_v12(u);  h = GRU(nfl, h);  F = heads(h, nfl, Ie)
%   i.e. force uses the state AFTER the GRU update of this substep,
%   exactly like the training sequence rollout.

%#codegen

n = net.norm;
h = st.h;
ie_buf = st.ie_buf;
K = net.lag_kmax;
w = net.lag_w;

% ---- normalize physical inputs -> u (6x1) ----
% (dI channel forced to 0: training zeroed it, collate line 203)
u = [(x - n.x_ref) / n.x_scale; ...
      v / n.v_scale; ...
      a / n.a_scale; ...
      I / n.i_scale; ...
      0; ...
      (T - n.t_ref) / n.t_scale];

% ---- current pair ----
Ie = 1 - min(abs(u(4)), 1);   % Ie in [0,1], unfiltered

% ---- 1 ms substeps ----
n_sub = max(1, round(dt / 0.001));
Fn = 0;
for k = 1:n_sub
    nfl = build_nfl_v12(net, u);
    h = gru_step(net, nfl, h);              % h: 2x16
    ie_buf = [ie_buf(2:end); Ie];          % shift in newest
    Ie_filt = w' * ie_buf;                 % reversed-kernel lag (training-exact)
    Fn = predict_force_norm(net, u, nfl, h, Ie, Ie_filt);
end

F = Fn * n.f_scale;
st_new = struct('h', h, 'ie_buf', ie_buf);
end


function nfl = build_nfl_v12(net, u)
% static_nfl_v12, sigma_v = 0.1 (v_norm units). u: 6x1 -> nfl: 35x1.
x = u(1); v = u(2); a = u(3); ci = u(4); di = u(5); t = u(6);
sv = 0.1;
v_abs = abs(v);
v_pos = max(v, 0);
v_neg = min(v, 0);
sgn_v = tanh(v / sv);

kin = [v; x*v; v*ci; v*t];                                            % 4
rebound = [tanh(2.0*v_pos); tanh(v_pos/sv); ci*tanh(v_pos/sv)];       % 3
compression = [v_neg*v_abs; v_neg^3; a*v_neg; ci*v_neg*v_abs];        % 4
curr_dyn = [di*v; di*a; abs(di)*ci];                                 % 3
aux = [sgn_v; ci*sgn_v; x^2; x^3; a*ci];                             % 5
hf_dyn = [abs(a); a*v; abs(a)*sgn_v; ci*abs(a)];                      % 4
x_abs = abs(x);
slow_gate = exp(-v_abs / sv);
slow = [slow_gate; x*slow_gate; x_abs*slow_gate; ...                  % 6
        ci*x*slow_gate; ci*x_abs*slow_gate; sgn_v*x_abs];

nfl = [u; kin; rebound; compression; curr_dyn; aux; hf_dyn; slow];   % 35
end


function h_new = gru_step(net, nfl, h)
% One GRU step, 2 layers x 16. nfl: 35x1, h: 2x16 -> h_new: 2x16.
% Matches torch.nn.GRU (batch_first), bias=True, dropout=0.
% (Layers written out explicitly: dynamic field names are not %#codegen
% compatible.)
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


function Fn = predict_force_norm(net, u, nfl, h, Ie, Ie_filt)
% Normalized force from heads. Matches model_v14.forward_components.
h2 = h(2, :)';                       % last layer, 16x1
v_out = u(2) * net.cfg.v_out_scale;  % scalar

feat = [h2; nfl; abs(v_out)];        % 52x1
c_slow = mlp3(net.c_head_W1, net.c_head_b1, net.c_head_W2, ...
              net.c_head_b2, net.c_head_W3, net.c_head_b3, feat);
s_slow = mlp3(net.s_head_W1, net.s_head_b1, net.s_head_W2, ...
              net.s_head_b2, net.s_head_W3, net.s_head_b3, feat);
b      = mlp3(net.b_head_W1, net.b_head_b1, net.b_head_W2, ...
              net.b_head_b2, net.b_head_W3, net.b_head_b3, [h2; nfl]);

i_feat = [Ie; Ie_filt];              % 2x1
c_fast = mlp2(net.c_fast_W1, net.c_fast_b1, ...
              net.c_fast_W2, net.c_fast_b2, i_feat);
s_fast = mlp2(net.s_fast_W1, net.s_fast_b1, ...
              net.s_fast_W2, net.s_fast_b2, i_feat);

c  = softplus_stable(c_slow + c_fast) + 1e-6 + 2e-3;
sj = softplus_stable(s_slow + s_fast) + 1e-6 + 5e-3/5;
phi = tanh(v_out ./ net.sigmas);     % 5x1

Gc = monoG(net.G_c_W1, net.G_c_b1, net.G_c_W2, net.G_c_b2, ...
           net.G_c_W3, net.G_c_b3, net.G_c_f0, net.G_c_gref, net.k_skip, Ie);
Gs = monoG(net.G_s_W1, net.G_s_b1, net.G_s_W2, net.G_s_b2, ...
           net.G_s_W3, net.G_s_b3, net.G_s_f0, net.G_s_gref, net.k_skip, Ie);

Fn = c * v_out * Gc + (sj' * phi) * Gs + b;
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
% MonotonicG at scalar Ie in [0,1]: G(0)=0, G(1)=1, strictly increasing.
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
