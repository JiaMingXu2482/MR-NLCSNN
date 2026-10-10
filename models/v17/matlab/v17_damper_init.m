function net = v17_damper_init(matFile)
%V17_DAMPER_INIT  Load trained v17 NLCSNN damper weights into a struct.
%
%   net = v17_damper_init() loads 'v17_weights.mat' sitting next to
%   this file. Pass an explicit path to override:
%       net = v17_damper_init('C:\path\to\v17_weights.mat')
%
%   Call once (e.g. in the model InitFcn), then pass `net` as a parameter
%   into v17_damper_step / v17_inverse_bisection / v17_inverse_head.
%   Keeps file I/O out of the per-step functions so they stay
%   %#codegen compatible.
%
%   Model: v17 best checkpoint (E330, holdout pooled NRMSE 16.80%)
%     - 6-dim normalized input u = [x, v, a, i, di, t]
%     - 35-dim static NFL (static_nfl_v12, sigma_v=0.1)
%     - GRU(35->16, 2 layers), 1 ms sample step
%     - F = c*v*G_c(Ie) + (sum_j s_j*tanh(v/sigma_j))*G_s(Ie) + b(h,nfl)
%       sigma = [0.5 1.0 2.0 4.0 8.0], v = v_norm*30
%     - CurrentLag on Ie = 1-|I|/1.6, learned a=0.032075 (tau=30.67 ms),
%       exact training replication: reversed conv kernel, KMAX=185
%     - MonotonicG(24 hidden, i_ref=1.0, k_skip=0.1), G(0)=0, G(1)=1
%     - Force output denormalized by f_scale [N]
%     - Inverse head: [h(16); nfl(35); v_norm; F_norm] -> sigmoid MLP
%       53->48->24->1 -> I_norm in [0,1]

if nargin < 1 || isempty(matFile)
    thisDir = fileparts(mfilename('fullpath'));
    matFile = fullfile(thisDir, 'v17_weights.mat');
end

S = load(matFile);
net = struct();

% ---- GRU weights (already (out,in) orientation, W*x works directly) ----
f = {'W_ir0','W_iz0','W_in0','W_hr0','W_hz0','W_hn0','b_ir0','b_iz0','b_in0', ...
     'b_hr0','b_hz0','b_hn0', ...
     'W_ir1','W_iz1','W_in1','W_hr1','W_hz1','W_hn1','b_ir1','b_iz1','b_in1', ...
     'b_hr1','b_hz1','b_hn1'};
for k = 1:numel(f)
    net.(f{k}) = S.(f{k});
end

% ---- MLP heads ----
for pfx = {'c_head','s_head','b_head','inv_head'}
    p = pfx{1};
    net.([p '_W1']) = S.([p '_W1']); net.([p '_b1']) = S.([p '_b1']);
    net.([p '_W2']) = S.([p '_W2']); net.([p '_b2']) = S.([p '_b2']);
    net.([p '_W3']) = S.([p '_W3']); net.([p '_b3']) = S.([p '_b3']);
end
for pfx = {'c_fast','s_fast'}
    p = pfx{1};
    net.([p '_W1']) = S.([p '_W1']); net.([p '_b1']) = S.([p '_b1']);
    net.([p '_W2']) = S.([p '_W2']); net.([p '_b2']) = S.([p '_b2']);
end

% ---- MonotonicG: PositiveLinear weights = softplus(w_raw) ----
net.k_skip = S.cfg.g_k_skip;
for pfx = {'G_c','G_s'}
    p = pfx{1};
    net.([p '_W1']) = softplus_stable(S.([p '_Wr1']));
    net.([p '_b1']) = S.([p '_b1']);
    net.([p '_W2']) = softplus_stable(S.([p '_Wr2']));
    net.([p '_b2']) = S.([p '_b2']);
    net.([p '_W3']) = softplus_stable(S.([p '_Wr3']));
    net.([p '_b3']) = S.([p '_b3']);
    % precompute f(0) and g_raw(i_ref) for the normalization
    f0   = monoG_raw(net, p, 0);
    gref = (monoG_raw(net, p, S.cfg.g_i_ref) - f0) + net.k_skip * S.cfg.g_i_ref;
    net.([p '_f0'])   = f0;
    net.([p '_gref']) = gref;
end

% ---- CurrentLag reversed kernel: w(k) = a*(1-a)^(k-1), k=1 oldest ----
a = S.cfg.a_lag;
K = double(S.cfg.lag_kmax);   % .mat里存的是int64, 转double才能进Simulink代码生成
net.lag_w = (a * (1 - a) .^ ((0:K-1)')) ;   % Kx1, oldest first
net.lag_kmax = K;

% ---- misc ----
net.sigmas = S.cfg.sigmas;          % 5x1
net.norm   = S.norm;
net.cfg    = S.cfg;
net.meta   = S.meta;
end


function y = softplus_stable(x)
% Numerically stable softplus, matches torch.nn.functional.softplus.
y = max(x, 0) + log1p(exp(-abs(x)));
end


function f = monoG_raw(net, p, I)
% Un-normalized MLP part of MonotonicG at scalar I.
z1 = tanh(net.([p '_W1']) * I + net.([p '_b1']));
z2 = tanh(net.([p '_W2']) * z1 + net.([p '_b2']));
f  = net.([p '_W3']) * z2 + net.([p '_b3']);
end
