function [I_cmd, Flo, Fhi, F_clipped, F_ach] = ...
    v17_inv_mf_impl_fast(Fd_N, t, Temp_C, cmpD, cmpRD, cid)
%V17_INV_MF  v17 inverse damper model, ONE corner, for MATLAB Function block.
%   %#codegen compatible. Build MEX per block for speed.
%
%   [I_cmd, Flo, Fhi, F_clipped, F_ach] = v17_inv_mf(Fd_N, t, Temp_C, cmpD, cmpRD, cid)
%
%   Drop a MATLAB Function block per corner (4 blocks). In the block editor
%   paste this function name; wire 6 scalar inputs / 5 scalar outputs.
%   For speed: right-click each block -> C/C++ Code -> Build This Subsystem,
%   or run slbuild on the model. Each generated MEX keeps its own
%   persistent state; the cid selects the state slot so corners never
%   cross-contaminate even in interpreted mode.
%
%   Inputs (all scalars):
%     Fd_N    desired damper force [N], CarSim sign (compression +).
%             MUST be the RAW v17 network force (use v17_fwd_raw_sl_1c to
%             generate it in open-loop tests). Do NOT feed the wrapper
%             (TAU_X / passive-projection) force here.
%     t       simulation time [s] (for dt and slew limit)
%     Temp_C  oil temperature [degC] (e.g. 30)
%     cmpD    damper displacement [mm], CarSim sign (compression +)
%     cmpRD   damper velocity [mm/s], CarSim sign
%     cid     corner id: 1=L1, 2=L2, 3=R1, 4=R2 (constant per block)
%
%   Outputs (all scalars, [N] in CarSim sign so they overlay Fd on a Scope):
%     I_cmd      current command [A], [0,1.6], slew-limited 160 A/s
%     Flo, Fhi   achievable force envelope [N]
%     F_clipped  Fd after envelope clipping [N]
%     F_ach      model-predicted force at I_cmd [N]
%
%   Solver: exact single-step v17 map (full nfl with ci, GRU re-step,
%   true ie_buf lag) evaluated on a 9-point Ie grid; best root picked,
%   tie-broken toward previous Ie (continuity) within TOL_F=15 N.
%   Low speed (|v|<5 mm/s): hold previous current.
%
%   Reset between runs: clear v17_inv_mf_impl_fast   (resets all 4 corners)

%#codegen

persistent net st_h st_iebuf t_prevs I_prevs Ie_prevs v_prevs a_filts

MAX_I = 1.6;
SLEW  = 160;      % A/s
N_GRID = 9;       % exact-map grid over Ie in [0,1]
TOL_F  = 15;      % tie-break tolerance [N]
NC = 4;
TAU_ACCEL = 0.02; % a from v diff + 1st-order lag (matches fwd model)

if isempty(net)
    net = v17_net_init_cg();
    st_h = zeros(2, 16, NC);          % GRU hidden, per corner
    st_iebuf = zeros(net.lag_kmax, NC); % CurrentLag buffer, per corner
    t_prevs  = NaN(1, NC);
    I_prevs  = zeros(1, NC);
    Ie_prevs = ones(1, NC);   % Ie=1 <-> I=0
    v_prevs  = zeros(1, NC);
    a_filts  = zeros(1, NC);
end

cid = max(1, min(NC, round(cid)));

t_prev  = t_prevs(cid);
I_prev  = I_prevs(cid);
Ie_prev = Ie_prevs(cid);
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
max_dI = SLEW * dt;



n = net.norm;

% CarSim -> rig sign convention (matches v17_fwd_raw_sl_1c)
x_rig = n.x_ref - cmpD;
v_rig = -cmpRD;
if first_step
    a_rig = 0;
else
    a_raw = (v_rig - v_prevs(cid)) / dt;
    if ~isfinite(a_raw), a_raw = 0; end
    alpha_a = 1 - exp(-dt / TAU_ACCEL);
    a_rig = (1 - alpha_a) * a_filts(cid) + alpha_a * a_raw;
    if ~isfinite(a_rig), a_rig = 0; end
end
v_prevs(cid) = v_rig;
a_filts(cid) = a_rig;
F_des_rig = -Fd_N;

% First-step warmup: pre-roll 200 steps with initial inputs to settle
% GRU hidden state and ie_buf (zero state is out-of-distribution).
if first_step
    u_warm = [(x_rig - n.x_ref) / n.x_scale; ...
              v_rig / n.v_scale; ...
              a_rig / n.a_scale; ...
              I_prev / n.i_scale; ...
              0; ...
              (Temp_C - n.t_ref) / n.t_scale];
    h_w = zeros(2, 16);
    buf_w = ones(net.lag_kmax, 1) * Ie_prev;
    for kw = 1:200
        nfl_w = build_nfl_v12(net, u_warm);
        h_w = gru_step_masked(net, nfl_w, h_w);
        buf_w = [buf_w(2:end); Ie_prev];
    end
    st_h(:, :, cid) = h_w;
    st_iebuf(:, cid) = buf_w;
end

% ---- hybrid solver: inv_head warm start + 5-point local exact refine ----
%   v17_inverse_head gives a cheap learned guess (1 MLP fwd); we refine
%   with 5 exact evaluations around it. Falls back to full 9-pt grid if
%   the local best is outside TOL_F (head was wrong).
st_h_c = st_h(:, :, cid);
st_iebuf_c = st_iebuf(:, cid);
st = struct('h', st_h_c, 'ie_buf', st_iebuf_c);

I_pred = v17_inverse_head(net, F_des_rig, x_rig, v_rig, a_rig, Temp_C, st);
Ie_pred = 1 - min(max(I_pred, 0), MAX_I) / MAX_I;
dIe = 0.12;
Ies_local = Ie_pred + [-2*dIe, -dIe, 0, dIe, 2*dIe];
Ies_local = min(max(Ies_local, 0), 1);
N_LOCAL = 5;
Fs_local = zeros(1, N_LOCAL);
for gi = 1:N_LOCAL
    Fs_local(gi) = force_of_Ie_exact(net, Ies_local(gi), x_rig, v_rig, a_rig, Temp_C, st);
end
errs_local = abs(Fs_local - F_des_rig);
[best_err_local, best_gi_local] = min(errs_local);

use_fallback = best_err_local > TOL_F;
% Fixed-size arrays for codegen: always 1x9, use first n_pts entries.
Ies = zeros(1, N_GRID);
Fs = zeros(1, N_GRID);
if use_fallback
    % Head was off: full 9-pt grid
    Ies = linspace(0, 1, N_GRID);
    for gi = 1:N_GRID
        Fs(gi) = force_of_Ie_exact(net, Ies(gi), x_rig, v_rig, a_rig, Temp_C, st);
    end
    n_pts = N_GRID;
else
    Ies(1:N_LOCAL) = Ies_local;
    Fs(1:N_LOCAL) = Fs_local;
    n_pts = N_LOCAL;
end
Flo_rig = min(Fs(1:n_pts));
Fhi_rig = max(Fs(1:n_pts));
Fd_clip = min(max(F_des_rig, Flo_rig), Fhi_rig);
errs = abs(Fs(1:n_pts) - F_des_rig);
best_err = min(errs);
best_gi = 1;
best_score = inf;
for gi = 1:n_pts
    if errs(gi) <= best_err + TOL_F
        score = abs(Ies(gi) - Ie_prev);
        if score < best_score
            best_score = score;
            best_gi = gi;
        end
    end
end
Ie_sol = Ies(best_gi);
I_raw = (1 - Ie_sol) * MAX_I;

if abs(v_rig) < 5
    I_raw = I_prev;   % low-speed hold
end

I_cmd = min(max(I_raw, I_prev - max_dI), I_prev + max_dI);
I_cmd = min(max(I_cmd, 0), MAX_I);
I_prevs(cid) = I_cmd;
Ie_prevs(cid) = 1 - I_cmd / MAX_I;

% F_ach at the final (held + slewed) current, exact map
Ie_cmd = 1 - I_cmd / MAX_I;
F_ach_rig = force_of_Ie_exact(net, Ie_cmd, x_rig, v_rig, a_rig, Temp_C, st);

% Advance the true recurrent state with I_cmd (1 ms substep, matches
% v17_damper_step with dt=0.001).
u_cmd = [(x_rig - n.x_ref) / n.x_scale; ...
         v_rig / n.v_scale; ...
         a_rig / n.a_scale; ...
         I_cmd / n.i_scale; ...
         0; ...
         (Temp_C - n.t_ref) / n.t_scale];
Ie_c = 1 - min(abs(u_cmd(4)), 1);
nfl_c = build_nfl_v12(net, u_cmd);
h_new = gru_step_masked(net, nfl_c, st.h);
ie_buf_new = [st.ie_buf(2:end); Ie_c];
st_h(:, :, cid) = h_new;
st_iebuf(:, cid) = ie_buf_new;

% rig -> CarSim sign; keep Flo <= Fhi
Flo = min(-Flo_rig, -Fhi_rig);
Fhi = max(-Flo_rig, -Fhi_rig);
F_clipped = -Fd_clip;
F_ach = -F_ach_rig;
end


function net = v17_net_init_cg()
%V17_NET_INIT_CG  Codegen-compatible v17 weight loader.
%   Uses coder.load so weights are baked in at compile time.
%#codegen
S = coder.load('v17_weights.mat');

net = struct();

% ---- GRU 2x16 ----
net.W_ir0=S.W_ir0; net.W_hr0=S.W_hr0; net.b_ir0=S.b_ir0; net.b_hr0=S.b_hr0;
net.W_iz0=S.W_iz0; net.W_hz0=S.W_hz0; net.b_iz0=S.b_iz0; net.b_hz0=S.b_hz0;
net.W_in0=S.W_in0; net.W_hn0=S.W_hn0; net.b_in0=S.b_in0; net.b_hn0=S.b_hn0;
net.W_ir1=S.W_ir1; net.W_hr1=S.W_hr1; net.b_ir1=S.b_ir1; net.b_hr1=S.b_hr1;
net.W_iz1=S.W_iz1; net.W_hz1=S.W_hz1; net.b_iz1=S.b_iz1; net.b_hz1=S.b_hz1;
net.W_in1=S.W_in1; net.W_hn1=S.W_hn1; net.b_in1=S.b_in1; net.b_hn1=S.b_hn1;

% ---- MLP heads (c/s/b/inv: 48-24-1 with ReLU; fast: 2-layer) ----
net.c_head_W1=S.c_head_W1; net.c_head_b1=S.c_head_b1;
net.c_head_W2=S.c_head_W2; net.c_head_b2=S.c_head_b2;
net.c_head_W3=S.c_head_W3; net.c_head_b3=S.c_head_b3;
net.s_head_W1=S.s_head_W1; net.s_head_b1=S.s_head_b1;
net.s_head_W2=S.s_head_W2; net.s_head_b2=S.s_head_b2;
net.s_head_W3=S.s_head_W3; net.s_head_b3=S.s_head_b3;
net.b_head_W1=S.b_head_W1; net.b_head_b1=S.b_head_b1;
net.b_head_W2=S.b_head_W2; net.b_head_b2=S.b_head_b2;
net.b_head_W3=S.b_head_W3; net.b_head_b3=S.b_head_b3;
net.inv_head_W1=S.inv_head_W1; net.inv_head_b1=S.inv_head_b1;
net.inv_head_W2=S.inv_head_W2; net.inv_head_b2=S.inv_head_b2;
net.inv_head_W3=S.inv_head_W3; net.inv_head_b3=S.inv_head_b3;
net.c_fast_W1=S.c_fast_W1; net.c_fast_b1=S.c_fast_b1;
net.c_fast_W2=S.c_fast_W2; net.c_fast_b2=S.c_fast_b2;
net.s_fast_W1=S.s_fast_W1; net.s_fast_b1=S.s_fast_b1;
net.s_fast_W2=S.s_fast_W2; net.s_fast_b2=S.s_fast_b2;

% ---- MonotonicG: PositiveLinear weights = softplus(w_raw) ----
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

% ---- CurrentLag reversed kernel ----
K = double(S.cfg.lag_kmax);
a = S.cfg.a_lag;
net.lag_w = (a * (1 - a) .^ ((0:K-1)'));
net.lag_kmax = K;

% ---- misc ----
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


function F = force_of_Ie_exact(net, Ie, x_rig, v_rig, a_rig, Temp_C, st)
%#codegen
I = (1 - Ie) * 1.6;
n = net.norm;
u = [(x_rig - n.x_ref) / n.x_scale; ...
     v_rig / n.v_scale; ...
     a_rig / n.a_scale; ...
     I / n.i_scale; ...
     0; ...
     (Temp_C - n.t_ref) / n.t_scale];
Ie_c = 1 - min(abs(u(4)), 1);
nfl = build_nfl_v12(net, u);
h = gru_step_masked(net, nfl, st.h);
buf = [st.ie_buf(2:end); Ie_c];
Ie_filt = net.lag_w' * buf;
h2 = h(2, :)';
v_out = u(2) * net.cfg.v_out_scale;
feat = [h2; nfl; abs(v_out)];
c_slow = mlp3(net.c_head_W1, net.c_head_b1, net.c_head_W2, ...
              net.c_head_b2, net.c_head_W3, net.c_head_b3, feat);
s_slow = mlp3(net.s_head_W1, net.s_head_b1, net.s_head_W2, ...
              net.s_head_b2, net.s_head_W3, net.s_head_b3, feat);
b      = mlp3(net.b_head_W1, net.b_head_b1, net.b_head_W2, ...
              net.b_head_b2, net.b_head_W3, net.b_head_b3, [h2; nfl]);
i_feat = [Ie_c; Ie_filt];
c_fast = mlp2(net.c_fast_W1, net.c_fast_b1, net.c_fast_W2, net.c_fast_b2, i_feat);
s_fast = mlp2(net.s_fast_W1, net.s_fast_b1, net.s_fast_W2, net.s_fast_b2, i_feat);
c  = softplus_stable(c_slow + c_fast) + 1e-6 + 2e-3;
sj = softplus_stable(s_slow + s_fast) + 1e-6 + 5e-3/5;
phi = tanh(v_out ./ net.sigmas);
Gc = monoG(net.G_c_W1, net.G_c_b1, net.G_c_W2, net.G_c_b2, ...
           net.G_c_W3, net.G_c_b3, net.G_c_f0, net.G_c_gref, net.k_skip, Ie_c);
Gs = monoG(net.G_s_W1, net.G_s_b1, net.G_s_W2, net.G_s_b2, ...
           net.G_s_W3, net.G_s_b3, net.G_s_f0, net.G_s_gref, net.k_skip, Ie_c);
Fn = c * v_out * Gc + (sj' * phi) * Gs + b;
F = Fn * n.f_scale;
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
