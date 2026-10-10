function [F0, F16, x_mm, v_mms] = v17_sine_check(net, t, Amp_mm, Freq_Hz, x0_mm, Temp_C, reset)
%V17_SINE_CHECK  Simulink MATLAB Function: sine excitation -> v17 damper forces.
%
%   [F0, F16, x_mm, v_mms] = v17_sine_check(net, t, Amp_mm, Freq_Hz, x0_mm, Temp_C, reset)
%
%   net FIRST: load once in the MATLAB workspace with
%       net = v17_damper_init();
%   then add `net` as a Parameter in the MATLAB Function block's
%   Ports and Data Manager (value: net). File I/O must stay out of the
%   compiled block, so net cannot be loaded inside.
%
%   Remaining block inputs: t [s] (Clock), Amp_mm, Freq_Hz, x0_mm,
%   Temp_C, reset (>0.5 re-zeros both states).
%   Outputs: F0 [N] (0 A), F16 [N] (1.6 A), x_mm, v_mms (for Scope).
%
%   Excitation is analytic: x = x0 + Amp*sin(2*pi*Freq*t), so v and a are
%   exact (no noisy Simulink Derivative blocks needed).
%
%   Two independent v17 twin instances (0 A and 1.6 A) keep separate
%   GRU hidden states and CurrentLag buffers (persistent), so each force
%   trace is the true constant-current response.
%
%   NOTE: this damper has INVERSE current characteristic (confirmed by rig
%   data: ST file rms|F| = 658 N at 0 A vs 175 N at 1.6 A). Expect F0 > F16.
%   The model uses Ie = 1 - I/1.6 with G increasing in Ie, consistent.
%
%   Solver: fixed-step recommended. dt is measured from t each call;
%   1 kHz (dt=0.001) is ideal, 100 Hz (dt=0.01, 10 substeps inside) works.
%   After reset, allow ~1 s warmup before trusting the force (GRU transient).

%#codegen
persistent st0 st16 t_prev

if isempty(st0)
    st0 = v17_new_state(net);     % 0 A branch state
    st16 = v17_new_state(net);    % 1.6 A branch state
    t_prev = t;
end

if reset > 0.5
    st0 = v17_new_state(net);
    st16 = v17_new_state(net);
    t_prev = t;
end

dt = t - t_prev;
t_prev = t;

% ---- analytic sine excitation ----
w = 2 * pi * Freq_Hz;
x_mm = x0_mm + Amp_mm * sin(w * t);
v_mms = Amp_mm * w * cos(w * t);          % mm/s
a_mms2 = -Amp_mm * w * w * sin(w * t);    % mm/s^2

% ---- two constant-current branches ----
[F0, st0] = v17_damper_step(net, x_mm, v_mms, a_mms2, 0.0, 0, Temp_C, dt, st0);
[F16, st16] = v17_damper_step(net, x_mm, v_mms, a_mms2, 1.6, 0, Temp_C, dt, st16);
end
