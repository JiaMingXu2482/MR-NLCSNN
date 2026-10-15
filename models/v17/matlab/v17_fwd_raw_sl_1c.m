function F = v17_fwd_raw_sl_1c(u, cid)
%V17_FWD_RAW_SL_1C  v17 RAW forward damper model, ONE corner, for Simulink.
%   TEST ONLY: raw v17 network, no wrapper (no TAU_X, no passive projection,
%   no output filters). For inverse-model round-trip validation: F_des must
%   be the raw network force, otherwise the inversion target is inconsistent
%   with the map the inverse inverts.
%   Interface identical to v17_carsim_sl_1c: F [N], CarSim sign.
%   Reset between runs: clear v17_fwd_raw_sl_1c (resets all 4 corners).

persistent net sts t_prevs v_prevs a_filts

NC = 4;

TAU_ACCEL = 0.02;

if isempty(net)
    net = v17_damper_init();
    sts = cell(1, NC);
    for k = 1:NC
        sts{k} = v17_new_state(net);
    end
    t_prevs = NaN(1, NC);
    v_prevs = zeros(1, NC);
    a_filts = zeros(1, NC);
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

% a_rig 由 v 差分+一阶滞后得到（与逆模型一致）。
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
end
