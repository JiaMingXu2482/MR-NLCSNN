function F = v17_fwd_mf_impl(t, Temp_C, cmpD, cmpRD, I_cmd, cid)
%V17_FWD_MF  v17 RAW forward damper model, ONE corner, for MATLAB Function block.
%   %#codegen compatible. Build MEX per block for speed.
%
%   F = v17_fwd_mf(t, Temp_C, cmpD, cmpRD, I_cmd, cid)
%
%   TEST ONLY: raw v17 network, no wrapper (no TAU_X, no passive
%   projection, no output filters). For inverse-model round-trip
%   validation: F_des must be the raw network force, otherwise the
%   inversion target is inconsistent with the map the inverse inverts.
%
%   Drop a MATLAB Function block per corner (4 blocks). Wire 6 scalar
%   inputs / 1 scalar output. For speed: right-click -> C/C++ Code ->
%   Build This Subsystem. The cid selects the state slot so corners
%   never cross-contaminate even in interpreted mode.
%
%   Inputs (all scalars):
%     t       simulation time [s]
%     Temp_C  oil temperature [degC] (e.g. 30)
%     cmpD    damper displacement [mm], CarSim sign (compression +)
%     cmpRD   damper velocity [mm/s], CarSim sign
%     I_cmd   current command [A], clamped to [0,1.6] inside
%     cid     corner id: 1=L1, 2=L2, 3=R1, 4=R2 (constant per block)
%
%   Output:
%     F       damper force [N], CarSim sign (compression +)
%
%   Reset between runs: clear v17_fwd_mf_impl   (resets all 4 corners)

%#codegen

persistent net st_h st_iebuf t_prevs v_prevs a_filts

NC = 4;
TAU_ACCEL = 0.02;

if isempty(net)
    net = v17_net_init_cg();
    st_h = zeros(2, 16, NC);
    st_iebuf = zeros(net.lag_kmax, NC);
    t_prevs = NaN(1, NC);
    v_prevs = zeros(1, NC);
    a_filts = zeros(1, NC);
end

cid = max(1, min(NC, round(cid)));
I_cmd = min(max(I_cmd, 0), 1.6);

t_prev = t_prevs(cid);
first_step = isnan(t_prev);
if first_step
    dt = 0.001;
else
    dt = t - t_prev;
    if ~(dt > 0) || ~isfinite(dt)
        dt = 0.001;
    end
end
t_prevs(cid) = t;

n = net.norm;
x_rig = n.x_ref - cmpD;
v_rig = -cmpRD;
if first_step
    a_rig = 0;
else
    a_raw = (v_rig - v_prevs(cid)) / dt;
    if ~isfinite(a_raw), a_raw = 0; end
    alpha = 1 - exp(-dt / TAU_ACCEL);
    a_rig = (1 - alpha) * a_filts(cid) + alpha * a_raw;
    if ~isfinite(a_rig), a_rig = 0; end
end
v_prevs(cid) = v_rig;
a_filts(cid) = a_rig;

% First-step warmup: pre-roll 200 steps with the initial inputs so the
% GRU hidden state and ie_buf settle. Training skipped the first 1000
% samples because the zero state is out-of-distribution; without warmup
% the force spikes at t=0.
if first_step
    Ie_init = 1 - min(abs(I_cmd) / 1.6, 1);
    u_warm = [(x_rig - n.x_ref) / n.x_scale; ...
              v_rig / n.v_scale; ...
              a_rig / n.a_scale; ...
              I_cmd / n.i_scale; ...
              0; ...
              (Temp_C - n.t_ref) / n.t_scale];
    h_w = zeros(2, 16);
    buf_w = ones(net.lag_kmax, 1) * Ie_init;
    for kw = 1:200
        nfl_w = build_nfl_v12(net, u_warm);
        h_w = gru_step_masked(net, nfl_w, h_w);
        buf_w = [buf_w(2:end); Ie_init];
    end
    st_h(:, :, cid) = h_w;
    st_iebuf(:, cid) = buf_w;
end

% ---- forward step (matches v17_damper_step, 1 ms substeps) ----
u = [(x_rig - n.x_ref) / n.x_scale; ...
     v_rig / n.v_scale; ...
     a_rig / n.a_scale; ...
     I_cmd / n.i_scale; ...
     0; ...
     (Temp_C - n.t_ref) / n.t_scale];
Ie = 1 - min(abs(u(4)), 1);

h = st_h(:, :, cid);
ie_buf = st_iebuf(:, cid);
K = net.lag_kmax;
w = net.lag_w;

n_sub = max(1, round(dt / 0.001));
Fn = 0;
for k = 1:n_sub
    nfl = build_nfl_v12(net, u);
    h = gru_step_masked(net, nfl, h);
    ie_buf = [ie_buf(2:end); Ie];
    Ie_filt = w' * ie_buf;
    Fn = predict_force_norm(net, u, nfl, h, Ie, Ie_filt);
end

st_h(:, :, cid) = h;
st_iebuf(:, cid) = ie_buf;

F_model = Fn * n.f_scale;
F = -F_model;   % rig -> CarSim sign
end


function net = v17_net_init_cg()
%#codegen
S = coder.load('v17_weights.mat');
net = struct();
net.W_ir0=S.W_ir0; net.W_hr0=S.W_hr0; net.b_ir0=S.b_ir0; net.b_hr0=S.b_hr0;
net.W_iz0=S.W_iz0; net.W_hz0=S.W_hz0; net.b_iz0=S.b_iz0; net.b_hz0=S.b_hz0;
net.W_in0=S.W_in0; net.W_hn0=S.W_hn0; net.b_in0=S.b_in0; net.b_hn0=S.b_hn0;
net.W_ir1=S.W_ir1; net.W_hr1=S.W_hr1; net.b_ir1=S.b_ir1; net.b_hr1=S.b_hr1;
net.W_iz1=S.W_iz1; net.W_hz1=S.W_hz1; net.b_iz1=S.b_iz1; net.b_hz1=S.b_hz1;
net.W_in1=S.W_in1; net.W_hn1=S.W_hn1; net.b_in1=S.b_in1; net.b_hn1=S.b_hn1;
net.c_head_W1=S.c_head_W1; net.c_head_b1=S.c_head_b1;
net.c_head_W2=S.c_head_W2; net.c_head_b2=S.c_head_b2;
net.c_head_W3=S.c_head_W3; net.c_head_b3=S.c_head_b3;
net.s_head_W1=S.s_head_W1; net.s_head_b1=S.s_head_b1;
net.s_head_W2=S.s_head_W2; net.s_head_b2=S.s_head_b2;
net.s_head_W3=S.s_head_W3; net.s_head_b3=S.s_head_b3;
net.b_head_W1=S.b_head_W1; net.b_head_b1=S.b_head_b1;
net.b_head_W2=S.b_head_W2; net.b_head_b2=S.b_head_b2;
net.b_head_W3=S.b_head_W3; net.b_head_b3=S.b_head_b3;
net.c_fast_W1=S.c_fast_W1; net.c_fast_b1=S.c_fast_b1;
net.c_fast_W2=S.c_fast_W2; net.c_fast_b2=S.c_fast_b2;
net.s_fast_W1=S.s_fast_W1; net.s_fast_b1=S.s_fast_b1;
net.s_fast_W2=S.s_fast_W2; net.s_fast_b2=S.s_fast_b2;
% NOTE: compute G f0/gref from LOCAL copies; do not pass `net` to a
% function before all its fields are assigned (codegen forbids adding
% fields after the struct has been read/used).
k_skip = S.cfg.g_k_skip;
g_i_ref = S.cfg.g_i_ref;
Gc_W1=softplus_stable(S.G_c_Wr1); Gc_b1=S.G_c_b1;
Gc_W2=softplus_stable(S.G_c_Wr2); Gc_b2=S.G_c_b2;
Gc_W3=softplus_stable(S.G_c_Wr3); Gc_b3=S.G_c_b3;
Gs_W1=softplus_stable(S.G_s_Wr1); Gs_b1=S.G_s_b1;
Gs_W2=softplus_stable(S.G_s_Wr2); Gs_b2=S.G_s_b2;
Gs_W3=softplus_stable(S.G_s_Wr3); Gs_b3=S.G_s_b3;
Gc_f0 = monoG_raw_w(Gc_W1,Gc_b1,Gc_W2,Gc_b2,Gc_W3,Gc_b3,0);
Gc_gref = (monoG_raw_w(Gc_W1,Gc_b1,Gc_W2,Gc_b2,Gc_W3,Gc_b3,g_i_ref)-Gc_f0) ...
          + k_skip*g_i_ref;
Gs_f0 = monoG_raw_w(Gs_W1,Gs_b1,Gs_W2,Gs_b2,Gs_W3,Gs_b3,0);
Gs_gref = (monoG_raw_w(Gs_W1,Gs_b1,Gs_W2,Gs_b2,Gs_W3,Gs_b3,g_i_ref)-Gs_f0) ...
          + k_skip*g_i_ref;
net.G_c_W1=Gc_W1; net.G_c_b1=Gc_b1;
net.G_c_W2=Gc_W2; net.G_c_b2=Gc_b2;
net.G_c_W3=Gc_W3; net.G_c_b3=Gc_b3;
net.G_s_W1=Gs_W1; net.G_s_b1=Gs_b1;
net.G_s_W2=Gs_W2; net.G_s_b2=Gs_b2;
net.G_s_W3=Gs_W3; net.G_s_b3=Gs_b3;
net.k_skip=k_skip;
net.G_c_f0=Gc_f0; net.G_c_gref=Gc_gref;
net.G_s_f0=Gs_f0; net.G_s_gref=Gs_gref;
K = double(S.cfg.lag_kmax);
a = S.cfg.a_lag;
net.lag_w = (a * (1 - a) .^ ((0:K-1)'));
net.lag_kmax = K;
net.sigmas = S.cfg.sigmas;
net.norm   = S.norm;
net.cfg    = S.cfg;
end


function f = monoG_raw_w(W1,b1,W2,b2,W3,b3,Ie)
%#codegen
z1 = tanh(W1*Ie + b1);
z2 = tanh(W2*z1 + b2);
f  = W3*z2 + b3;
end


function Fn = predict_force_norm(net, u, nfl, h, Ie, Ie_filt)
%#codegen
h2 = h(2, :)';
v_out = u(2) * net.cfg.v_out_scale;
feat = [h2; nfl; abs(v_out)];
c_slow = mlp3(net.c_head_W1, net.c_head_b1, net.c_head_W2, ...
              net.c_head_b2, net.c_head_W3, net.c_head_b3, feat);
s_slow = mlp3(net.s_head_W1, net.s_head_b1, net.s_head_W2, ...
              net.s_head_b2, net.s_head_W3, net.s_head_b3, feat);
b      = mlp3(net.b_head_W1, net.b_head_b1, net.b_head_W2, ...
              net.b_head_b2, net.b_head_W3, net.b_head_b3, [h2; nfl]);
i_feat = [Ie; Ie_filt];
c_fast = mlp2(net.c_fast_W1, net.c_fast_b1, net.c_fast_W2, net.c_fast_b2, i_feat);
s_fast = mlp2(net.s_fast_W1, net.s_fast_b1, net.s_fast_W2, net.s_fast_b2, i_feat);
c  = softplus_stable(c_slow + c_fast) + 1e-6 + 2e-3;
sj = softplus_stable(s_slow + s_fast) + 1e-6 + 5e-3/5;
phi = tanh(v_out ./ net.sigmas);
Gc = monoG(net.G_c_W1, net.G_c_b1, net.G_c_W2, net.G_c_b2, ...
           net.G_c_W3, net.G_c_b3, net.G_c_f0, net.G_c_gref, net.k_skip, Ie);
Gs = monoG(net.G_s_W1, net.G_s_b1, net.G_s_W2, net.G_s_b2, ...
           net.G_s_W3, net.G_s_b3, net.G_s_f0, net.G_s_gref, net.k_skip, Ie);
Fn = c * v_out * Gc + (sj' * phi) * Gs + b;
end


function nfl = build_nfl_v12(net, u)
%#codegen
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
%#codegen
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
%#codegen
z1 = max(W1 * x + b1, 0);
z2 = max(W2 * z1 + b2, 0);
y  = W3 * z2 + b3;
end


function y = mlp2(W1, b1, W2, b2, x)
%#codegen
z1 = max(W1 * x + b1, 0);
y  = W2 * z1 + b2;
end


function G = monoG(W1, b1, W2, b2, W3, b3, f0, gref, k_skip, Ie)
%#codegen
z1 = tanh(W1 * Ie + b1);
z2 = tanh(W2 * z1 + b2);
f  = W3 * z2 + b3;
G = max(((f - f0) + k_skip * Ie) / gref, 0);
end


function y = sigmoid(x)
%#codegen
y = 1 ./ (1 + exp(-x));
end


function y = softplus_stable(x)
%#codegen
y = max(x, 0) + log1p(exp(-abs(x)));
end
