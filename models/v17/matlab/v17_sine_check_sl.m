function y = v17_sine_check_sl(u)
%V17_SINE_CHECK_SL  Sine check for Simulink "Interpreted MATLAB Function" block.
%
%   y = v17_sine_check_sl(u)
%
%   This runs in the MATLAB interpreter (NO code generation), so file I/O,
%   persistent state and any data types are all fine. Use it with:
%     Mux (6 inputs) -> Interpreted MATLAB Function -> Demux (4 outputs)
%
%   u = [t, Amp_mm, Freq_Hz, x0_mm, Temp_C, reset]   (6x1, from Mux)
%       t       simulation time [s] (Clock)
%       Amp_mm  sine amplitude [mm]
%       Freq_Hz sine frequency [Hz]
%       x0_mm   rod center position [mm], e.g. 280
%       Temp_C  damper temperature [degC]
%       reset   >0.5 re-zeros both states
%   y = [F0; F16; x_mm; v_mms]                      (4x1, to Demux)
%       F0      force at constant 0 A   [N]
%       F16     force at constant 1.6 A [N]
%
%   NOTE: this damper has INVERSE current characteristic (rig data:
%   rms|F| = 658 N at 0 A vs 175 N at 1.6 A). Expect F0 > F16.
%   After reset, allow ~1 s warmup (GRU transient).

persistent net st0 st16 t_prev

t      = u(1);
Amp_mm = u(2);
Freq_Hz = u(3);
x0_mm  = u(4);
Temp_C = u(5);
reset  = u(6);

if isempty(net)
    net = v17_damper_init();      % file I/O OK here (interpreted)
    st0 = v17_new_state(net);     % 0 A branch
    st16 = v17_new_state(net);    % 1.6 A branch
    t_prev = t;
end

if reset > 0.5
    st0 = v17_new_state(net);
    st16 = v17_new_state(net);
    t_prev = t;
end

dt = t - t_prev;
t_prev = t;

w = 2 * pi * Freq_Hz;
x_mm = x0_mm + Amp_mm * sin(w * t);
v_mms = Amp_mm * w * cos(w * t);
a_mms2 = -Amp_mm * w * w * sin(w * t);

[F0, st0] = v17_damper_step(net, x_mm, v_mms, a_mms2, 0.0, 0, Temp_C, dt, st0);
[F16, st16] = v17_damper_step(net, x_mm, v_mms, a_mms2, 1.6, 0, Temp_C, dt, st16);

y = [F0; F16; x_mm; v_mms];
end
