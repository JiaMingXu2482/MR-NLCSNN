function y = v17_inv_carsim_sl_1c(u, cid)
%V17_INV_CARSIM_SL_1C  v17 inverse damper model, ONE corner, for Simulink.
%
%   y = v17_inv_carsim_sl_1c(u, cid)
%
%   Runs in the MATLAB interpreter (no code generation).
%   Use with: Mux (6 inputs) -> Interpreted MATLAB Function (output dim 5)
%             -> Demux [1,1,1,1,1] or direct indexing.
%   Instantiate ONE block per corner (4 parallel branches).
%
%   IMPORTANT: Interpreted MATLAB Function blocks that call the SAME
%   function share persistent variables. The corner ID (cid) selects
%   which of the 4 internally-isolated states this block uses, so the
%   four corners never contaminate each other.
%   In Simulink, set each block's "MATLAB function" field to:
%       L1: v17_inv_carsim_sl_1c(u, 1)
%       L2: v17_inv_carsim_sl_1c(u, 2)
%       R1: v17_inv_carsim_sl_1c(u, 3)
%       R2: v17_inv_carsim_sl_1c(u, 4)
%
%   u = [t, Temp_C, cmpD, cmpRD, cmpAD, Fd_N]   (6x1, from Mux)
%       t       CarSim/Simulink time [s] (for dt)
%       Temp_C  oil temperature [degC] (Constant, e.g. 30)
%       cmpD    damper compression [mm], CarSim sign (compression +)
%       cmpRD   compression rate [mm/s], CarSim sign
%       cmpAD   compression accel [g], CarSim sign (read but unused:
%               a_rig is derived from v diff + TAU_ACCEL filter)
%       Fd_N    desired damper force [N], CarSim sign (compressive +).
%               NOTE: must be the RAW v17 network force (no output
%               wrapper/projection), otherwise the inversion target is
%               inconsistent with the map below.
%   cid     corner ID: 1=L1, 2=L2, 3=R1, 4=R2 (CarSim order)
%
%   y = [I_cmd; Flo; Fhi; F_clipped; F_ach]   (5x1)
%       I_cmd      current command [A], [0,1.6], slew-limited 160 A/s
%       Flo/Fhi    achievable force envelope [N], CarSim sign
%       F_clipped  desired force after envelope clipping [N], CarSim sign
%       F_ach      model-predicted force at I_cmd [N], CarSim sign
%       (force diagnostics are CarSim sign so they overlay Fd on a Scope)
%
%   Solver (2026-10-14 rewrite): the v17 map F(Ie) is only *mostly*
%   monotonic in Ie -- current also enters through non-monotonic fast-MLP
%   and ci->nfl->GRU paths (~15 N wiggles), so plain bisection can lock
%   onto a wrong root and the state feedback then latches it at a rail.
%   We now evaluate the EXACT single-step map (full nfl with ci, GRU
%   re-step, true ie_buf lag) on a 9-point Ie grid and pick the best
%   root, tie-broken toward the previous Ie (continuity) within TOL_F.
%
%   Reset between runs:  clear v17_inv_carsim_sl_1c   (resets all 4 corners)

persistent net sts t_prevs I_prevs Ie_prevs v_prevs a_filts

MAX_I = 1.6;
SLEW  = 160;
N_GRID = 9;     % exact-map grid over Ie in [0,1] (0.2 A resolution;
              % 9 pts validated 2026-10-15: resolves the ~15 N wiggle, 2x faster)
TOL_F  = 15;    % tie-break tolerance [N]: covers the ~15 N map wiggle
NC    = 4;
TAU_ACCEL = 0.02;   % 与正模型一致：a 由 v 差分+一阶滞后得到

if isempty(net)
    net = v17_damper_init();
    sts = cell(1, NC);
    for k = 1:NC
        sts{k} = v17_new_state(net);
    end
    t_prevs = NaN(1, NC);
    I_prevs = zeros(1, NC);
    Ie_prevs = ones(1, NC);   % Ie=1 <-> I=0
    v_prevs = zeros(1, NC);
    a_filts = zeros(1, NC);
end

t      = u(1);
Temp_C = u(2);
cmpD   = u(3);
cmpRD  = u(4);
% cmpAD = u(5);  % read but unused
Fd_N   = u(6);

t_prev = t_prevs(cid);
I_prev = I_prevs(cid);
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

x_rig = net.norm.x_ref - cmpD;
v_rig = -cmpRD;
% a_rig 由 v 差分+一阶滞后得到（与正模型一致），不直接使用 cmpAD。
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

% ---- exact-map grid solver ----
st = sts{cid};
Ies = linspace(0, 1, N_GRID);
Fs = zeros(1, N_GRID);
for gi = 1:N_GRID
    Fs(gi) = force_of_Ie_exact(net, Ies(gi), x_rig, v_rig, a_rig, Temp_C, st);
end
Flo_rig = min(Fs);
Fhi_rig = max(Fs);
Fd_clip = min(max(F_des_rig, Flo_rig), Fhi_rig);
errs = abs(Fs - F_des_rig);
best_err = min(errs);
best_gi = 1;
best_score = inf;
for gi = 1:N_GRID
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
    I_raw = I_prev;
end

I_cmd = min(max(I_raw, I_prev - max_dI), I_prev + max_dI);
I_cmd = min(max(I_cmd, 0), MAX_I);
I_prevs(cid) = I_cmd;
Ie_prevs(cid) = 1 - I_cmd / MAX_I;

% F_ach 对应最终 I_cmd（经低速保持+限流后），用精确映射计算。
Ie_cmd = 1 - I_cmd / MAX_I;
F_ach_rig = force_of_Ie_exact(net, Ie_cmd, x_rig, v_rig, a_rig, Temp_C, st);

% 状态按真实动力学用 I_cmd 推进（与正模型 v17_damper_step 一致）。
[~, st_new] = v17_damper_step(net, x_rig, v_rig, a_rig, ...
                              I_cmd, 0, Temp_C, dt, st);
sts{cid} = st_new;

% 极性翻转后重排包络，保证 Flo_car <= Fhi_car。
Flo_car = min(-Flo_rig, -Fhi_rig);
Fhi_car = max(-Flo_rig, -Fhi_rig);
y = [I_cmd; Flo_car; Fhi_car; -Fd_clip; -F_ach_rig];
end


function F = force_of_Ie_exact(net, Ie, x_rig, v_rig, a_rig, Temp_C, st)
%FORCE_OF_IE_EXACT  Exact single-step v17 forward map at candidate Ie.
%   Mirrors v17_damper_step for a 1 ms step: full nfl (with ci = I/i_scale),
%   GRU re-step from st.h, ie_buf shifted with the candidate Ie, and the
%   true lagged-current channel i_feat = [Ie_c; Ie_filt].
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
