function I_cmd = v17_inverse_head(net, F_des, x, v, a, T, st)
%V17_INVERSE_HEAD  Trained MLP inverse: desired force -> current (fast).
%   %#codegen compatible (no file I/O, no persistent state).
%
%   I_cmd = v17_inverse_head(net, F_des, x, v, a, T, st)
%
%   One forward pass through the jointly-trained inverse head:
%     feat = [h(16); nfl(35); v_norm(1); F_des/f_scale(1)]  (53-dim)
%     I_norm = sigmoid(MLP(feat)), 53->48->24->1
%     I_cmd = I_norm * 1.6, clamped to [0, 1.6]
%
%   Much cheaper than v17_inverse_bisection (single MLP vs 40-iteration
%   bisection), suitable for tight real-time budgets.
%
%   CAVEAT (read before using): the head was trained with the TRUE current
%   history in its GRU/nfl features (see train_v14.py train_epoch), while
%   deployment masks the current channel. Holdout inv_mae = 0.0101 A was
%   measured with that same unmasked protocol, so it is optimistic.
%   v17_inverse_bisection is the deployment-faithful primary inverse;
%   use this head when you need speed and can tolerate the bias, or as a
%   warm start for the bisection.
%
%   Inputs: same physical units as v17_inverse_bisection (F_des [N]).
%   st.h is used as the GRU state at START of step (masked GRU advance,
%   st itself not modified).

%#codegen

n = net.norm;

u = [(x - n.x_ref) / n.x_scale; ...
      v / n.v_scale; ...
      a / n.a_scale; ...
      0; ...
      0; ...
      (T - n.t_ref) / n.t_scale];
nfl = build_nfl_v12(net, u);
h_m = gru_step_masked(net, nfl, st.h);
h2 = h_m(2, :)';

feat = [h2; nfl; u(2); F_des / n.f_scale];   % 53x1
z1 = max(net.inv_head_W1 * feat + net.inv_head_b1, 0);
z2 = max(net.inv_head_W2 * z1 + net.inv_head_b2, 0);
I_norm = 1 / (1 + exp(-(net.inv_head_W3 * z2 + net.inv_head_b3)));
I_cmd = min(max(I_norm, 0), 1) * n.i_scale;
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


function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end
