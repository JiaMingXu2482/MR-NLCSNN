function F = v17_carsim_sl_1c(u, cid)
%V17_CARSIM_SL_1C  v17 forward damper model, ONE corner, for Simulink.
%   （头注释略，与之前一致）

persistent net sts t_prevs v_prevs a_filts f_filts ...
           notch_x1 notch_x2 notch_y1 notch_y2 v_filts

NC = 4;

C_MIN = 0.8;
V_EPS = 5;
P_EPS = 50;
TAU_ACCEL = 0.02;
TAU_F = 0.003;
TAU_V = 0.03;   % 新增 2026-10-14：速度输入一阶滞后，压 CarSim cmpRD 闭环抖动

% 新增：陷波滤波器参数
F_NOTCH = 10;   % 中心频率 Hz，如实测共振不在10Hz就改这里
Q_NOTCH = 5;    % 品质因数，带宽=f0/Q；Q越大陷得越窄

if isempty(net)
    net = v17_damper_init();
    sts = cell(1, NC);
    for k = 1:NC
        sts{k} = v17_new_state(net);
    end
    t_prevs = NaN(1, NC);
    v_prevs = zeros(1, NC);
    a_filts = zeros(1, NC);
    f_filts = zeros(1, NC);
    notch_x1 = zeros(1, NC);
    notch_x2 = zeros(1, NC);
    notch_y1 = zeros(1, NC);
    notch_y2 = zeros(1, NC);
    v_filts = zeros(1, NC);
end

t      = u(1);
Temp_C = u(2);
cmpD   = u(3);
cmpRD  = u(4);
I_cmd  = u(6);
I_cmd  = min(max(I_cmd, 0), 1.6);

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

x_rig = net.norm.x_ref - cmpD;
v_rig = -cmpRD;

% 新增 2026-10-14：速度输入低通
% CarSim cmpRD 在闭环 Replace 下带高频抖动；GRU 对 v 敏感，且 a_rig
% 由 v_rig 差分得到，v 抖会被放大两次。在输入端加一阶滞后，
% TAU_V=5ms（约 30Hz 截止）：远高于车身/车轮模态，只压抖动频带。
% a_rig 下面的差分用的是滤波后的 v_rig。
if first_step
    v_filts(cid) = v_rig;
else
    alpha_v = 1 - exp(-dt / TAU_V);
    v_rig = (1 - alpha_v) * v_filts(cid) + alpha_v * v_rig;
    if ~isfinite(v_rig), v_rig = v_filts(cid); end
    v_filts(cid) = v_rig;
end

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
t_prevs(cid) = t;

st = sts{cid};
[F_model, st] = v17_damper_step(net, x_rig, v_rig, a_rig, ...
                                I_cmd, 0, Temp_C, dt, st);
sts{cid} = st;
F = -F_model;

if isfinite(F) && isfinite(cmpRD)
    Ffloor = C_MIN * cmpRD;
    Pexcess = (F - Ffloor) * cmpRD;
    Ppositive = 0.5 * (Pexcess + sqrt(Pexcess*Pexcess + P_EPS*P_EPS));
    F = Ffloor + Ppositive * cmpRD / (cmpRD*cmpRD + V_EPS*V_EPS);
else
    F = 0;
end

if first_step
    f_filts(cid) = F;
else
    alpha_f = 1 - exp(-dt / TAU_F);
    F = (1 - alpha_f) * f_filts(cid) + alpha_f * F;
    f_filts(cid) = F;
end

% ---- 新增：biquad 陷波 ----
if first_step
    notch_x1(cid) = F; notch_x2(cid) = F;
    notch_y1(cid) = F; notch_y2(cid) = F;
else
    fs = 1/dt;
    w0 = 2*pi*F_NOTCH/fs;
    alpha_n = sin(w0)/(2*Q_NOTCH);
    cw = cos(w0);
    a0n = 1 + alpha_n;
    b0n = 1/a0n; b1n = -2*cw/a0n; b2n = 1/a0n;
    a1n = -2*cw/a0n; a2n = (1-alpha_n)/a0n;
    x0 = F;
    y0 = b0n*x0 + b1n*notch_x1(cid) + b2n*notch_x2(cid) ...
         - a1n*notch_y1(cid) - a2n*notch_y2(cid);
    if ~isfinite(y0), y0 = F; end
    notch_x2(cid) = notch_x1(cid); notch_x1(cid) = x0;
    notch_y2(cid) = notch_y1(cid); notch_y1(cid) = y0;
    F = y0;
end
end
