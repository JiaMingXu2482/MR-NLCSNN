function F = v17_carsim_sl_1c(u, cid)
%V17_CARSIM_SL_1C  v17 forward damper model, ONE corner, for Simulink.
%   （头注释略，与之前一致）

persistent net sts t_prevs v_prevs a_filts x_filts

NC = 4;

C_MIN = 0.8;
V_EPS = 5;
P_EPS = 50;
TAU_ACCEL = 0.02;
TAU_X = 0.01;    % 新增 2026-10-14：x 输入轻滤波，约 15Hz 截止，去车轮跳动纹波

if isempty(net)
    net = v17_damper_init();
    sts = cell(1, NC);
    for k = 1:NC
        sts{k} = v17_new_state(net);
    end
    t_prevs = NaN(1, NC);
    v_prevs = zeros(1, NC);
    a_filts = zeros(1, NC);
    x_filts = zeros(1, NC);
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

% 新增 2026-10-14：x 输入轻滤波
% cmpD 的 10Hz 车轮跳动纹波直接进 GRU 是残余抖动的另一通路。
% TAU_X=10ms（约 15Hz 截止）：只去纹波，主运动（<2Hz）几乎无影响。
if first_step
    x_filts(cid) = x_rig;
else
    alpha_x = 1 - exp(-dt / TAU_X);
    x_rig = (1 - alpha_x) * x_filts(cid) + alpha_x * x_rig;
    if ~isfinite(x_rig), x_rig = x_filts(cid); end
    x_filts(cid) = x_rig;
end

v_rig = -cmpRD;

% 2026-10-14 回退：取消 TAU_V=0.03 速度输入滤波（正逆往返一致性测试用），
% 逆模型内部没有这个滤波，两边预处理必须一致。
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
end
