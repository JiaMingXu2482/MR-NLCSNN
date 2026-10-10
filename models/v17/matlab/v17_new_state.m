function st = v17_new_state(net)
%V17_NEW_STATE  Fresh per-corner recurrent state for the v17 twin.
%
%   st = v17_new_state(net)
%
%   Returns struct with:
%     h      GRU hidden state, 2x16 (zeros = training rollout start)
%     ie_buf CurrentLag Ie history buffer, KMAXx1 (zeros)
%
%   One instance per damper corner; feed st back through v17_damper_step.
%   After reset, allow ~1 s warmup before trusting the force output
%   (training skipped the first 1000 samples at 1 kHz).

st = struct('h', zeros(2, 16), 'ie_buf', zeros(net.lag_kmax, 1));
end
