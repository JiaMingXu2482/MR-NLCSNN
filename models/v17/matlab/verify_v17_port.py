#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17 MATLAB 移植验证脚本
=======================
本机无 MATLAB/Octave, 用 numpy 逐行镜像翻译 .m 文件
(v17_damper_step / v17_inverse_bisection / v17_inverse_head),
与 PyTorch 参考实现 (model_v14.py 同代码路径) 对比, 验证移植正确性,
并生成 golden_vectors.mat 供用户在 MATLAB 里跑 test_v17_port.m。

镜像翻译规则: 与 .m 文件逐行对应, 全部 float64, 列向量。
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path.home() / "workspace" / "mr-nlcsnn-v3"
PKG = Path.home() / "workspace" / "your_files" / "v17_pmpc_package"
import sys
sys.path.insert(0, str(REPO))
from mr_nlcsnn.model_v14 import NLCSNN_v14
from mr_nlcsnn.model_v12 import static_nfl_v12

CKPT = REPO / "training_outputs" / "v17_20261010_045838" / "nlcsnn_v17_best.pth"
CSV = REPO / "Parsed_Card_force_wide" / "1_Steady_Harmonic" / "SH_10mm_1.66Hz_1A_r1.csv"

torch.set_grad_enabled(False)


# ============================================================ numpy 镜像
class M:
    def __init__(self, mat_path):
        from scipy.io import loadmat
        S = loadmat(str(mat_path), squeeze_me=False)
        g = lambda k: np.asarray(S[k], dtype=np.float64)
        self.P = {k: g(k) for k in [
            'W_ir0','W_iz0','W_in0','W_hr0','W_hz0','W_hn0','b_ir0','b_iz0','b_in0','b_hr0','b_hz0','b_hn0',
            'W_ir1','W_iz1','W_in1','W_hr1','W_hz1','W_hn1','b_ir1','b_iz1','b_in1','b_hr1','b_hz1','b_hn1',
            'c_head_W1','c_head_b1','c_head_W2','c_head_b2','c_head_W3','c_head_b3',
            's_head_W1','s_head_b1','s_head_W2','s_head_b2','s_head_W3','s_head_b3',
            'b_head_W1','b_head_b1','b_head_W2','b_head_b2','b_head_W3','b_head_b3',
            'inv_head_W1','inv_head_b1','inv_head_W2','inv_head_b2','inv_head_W3','inv_head_b3',
            'c_fast_W1','c_fast_b1','c_fast_W2','c_fast_b2',
            's_fast_W1','s_fast_b1','s_fast_W2','s_fast_b2',
            'G_c_Wr1','G_c_b1','G_c_Wr2','G_c_b2','G_c_Wr3','G_c_b3',
            'G_s_Wr1','G_s_b1','G_s_Wr2','G_s_b2','G_s_Wr3','G_s_b3']}
        n = S['norm']
        self.norm = {k: float(np.asarray(n[k][0, 0]).ravel()[0]) for k in
                     ['x_ref','x_scale','v_scale','a_scale','i_scale','di_scale',
                      't_ref','t_scale','f_scale']}
        # MonotonicG: 与 v17_damper_init.m 一致, W = softplus(W_raw),
        # f0/gref 在 init 时预计算
        self.k_skip = 0.1
        for pfx in ['G_c', 'G_s']:
            for li in ['1', '2', '3']:
                self.P[f'{pfx}_W{li}'] = self.softplus(self.P.pop(f'{pfx}_Wr{li}'))
            f0 = float(np.asarray(self._monoG_raw(pfx, 0.0)).ravel()[0])
            gref = (float(np.asarray(self._monoG_raw(pfx, 1.0)).ravel()[0]) - f0) + self.k_skip * 1.0
            self.P[f'{pfx}_f0'] = np.array([[f0]])
            self.P[f'{pfx}_gref'] = np.array([[gref]])
        # lag_w: 与 v17_damper_init.m 一致, w(k) = a*(1-a)^(k-1), oldest first
        cfg = S['cfg']
        self.K = int(np.asarray(cfg['lag_kmax'][0, 0]).ravel()[0])
        a_lag = float(np.asarray(cfg['a_lag'][0, 0]).ravel()[0])
        self.P['lag_w'] = (a_lag * (1 - a_lag) ** np.arange(self.K)).reshape(-1, 1)
        self.P['sigmas'] = np.asarray(cfg['sigmas'][0, 0], dtype=np.float64).reshape(-1, 1)
        self.v_out_scale = 30.0

    def _monoG_raw(self, pfx, Ie):
        P = self.P
        z1 = np.tanh(P[f'{pfx}_W1'] * Ie + P[f'{pfx}_b1'])
        z2 = np.tanh(P[f'{pfx}_W2'] @ z1 + P[f'{pfx}_b2'])
        return P[f'{pfx}_W3'] @ z2 + P[f'{pfx}_b3']

    @staticmethod
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
    @staticmethod
    def softplus(x): return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
    @staticmethod
    def relu(x): return np.maximum(x, 0)

    def build_nfl(self, u):
        x, v, a, ci, di, t = u.ravel()
        sv = 0.1
        v_abs = abs(v); v_pos = max(v, 0.0); v_neg = min(v, 0.0)
        sgn_v = math.tanh(v / sv)
        kin = np.array([v, x*v, v*ci, v*t])
        rebound = np.array([math.tanh(2.0*v_pos), math.tanh(v_pos/sv), ci*math.tanh(v_pos/sv)])
        compression = np.array([v_neg*v_abs, v_neg**3, a*v_neg, ci*v_neg*v_abs])
        curr_dyn = np.array([di*v, di*a, abs(di)*ci])
        aux = np.array([sgn_v, ci*sgn_v, x**2, x**3, a*ci])
        hf_dyn = np.array([abs(a), a*v, abs(a)*sgn_v, ci*abs(a)])
        x_abs = abs(x); slow_gate = math.exp(-v_abs / sv)
        slow = np.array([slow_gate, x*slow_gate, x_abs*slow_gate,
                         ci*x*slow_gate, ci*x_abs*slow_gate, sgn_v*x_abs])
        return np.concatenate([u.ravel(), kin, rebound, compression,
                               curr_dyn, aux, hf_dyn, slow]).reshape(-1, 1)

    def gru_step(self, nfl, h):
        P = self.P
        def layer(x, hh, s):
            r = self.sigmoid(P[f'W_ir{s}'] @ x + P[f'b_ir{s}'] + P[f'W_hr{s}'] @ hh + P[f'b_hr{s}'])
            z = self.sigmoid(P[f'W_iz{s}'] @ x + P[f'b_iz{s}'] + P[f'W_hz{s}'] @ hh + P[f'b_hz{s}'])
            nn = np.tanh(P[f'W_in{s}'] @ x + P[f'b_in{s}'] + r * (P[f'W_hn{s}'] @ hh + P[f'b_hn{s}']))
            return (1 - z) * nn + z * hh
        h0 = layer(nfl, h[0:1, :].T, 0)
        h1 = layer(h0, h[1:2, :].T, 1)
        return np.vstack([h0.T, h1.T])

    def mlp3(self, pfx, x):
        P = self.P
        z1 = self.relu(P[f'{pfx}_W1'] @ x + P[f'{pfx}_b1'])
        z2 = self.relu(P[f'{pfx}_W2'] @ z1 + P[f'{pfx}_b2'])
        return P[f'{pfx}_W3'] @ z2 + P[f'{pfx}_b3']

    def mlp2(self, pfx, x):
        P = self.P
        z1 = self.relu(P[f'{pfx}_W1'] @ x + P[f'{pfx}_b1'])
        return P[f'{pfx}_W2'] @ z1 + P[f'{pfx}_b2']

    def monoG(self, pfx, Ie):
        P = self.P
        z1 = np.tanh(P[f'{pfx}_W1'] * Ie + P[f'{pfx}_b1'])
        z2 = np.tanh(P[f'{pfx}_W2'] @ z1 + P[f'{pfx}_b2'])
        f = (P[f'{pfx}_W3'] @ z2 + P[f'{pfx}_b3']).item()
        return max(((f - P[f'{pfx}_f0'].item()) + self.k_skip * Ie) / P[f'{pfx}_gref'].item(), 0.0)

    def predict_force_norm(self, u, nfl, h, Ie, Ie_filt):
        P = self.P
        h2 = h[1:2, :].T
        v_out = float(u[1, 0]) * self.v_out_scale
        feat = np.vstack([h2, nfl, [[abs(v_out)]]])
        c_slow = self.mlp3('c_head', feat).item()
        s_slow = self.mlp3('s_head', feat)
        b = self.mlp3('b_head', np.vstack([h2, nfl])).item()
        i_feat = np.array([[Ie], [Ie_filt]])
        c_fast = self.mlp2('c_fast', i_feat).item()
        s_fast = self.mlp2('s_fast', i_feat)
        c = self.softplus(c_slow + c_fast) + 1e-6 + 2e-3
        sj = self.softplus(s_slow + s_fast) + 1e-6 + 5e-3 / 5
        phi = np.tanh(v_out / P['sigmas'])
        Gc = self.monoG('G_c', Ie); Gs = self.monoG('G_s', Ie)
        return float(c * v_out * Gc + (sj.T @ phi).item() * Gs + b)

    def norm_u(self, x, v, a, I, T):
        n = self.norm
        return np.array([[(x - n['x_ref']) / n['x_scale'], v / n['v_scale'],
                          a / n['a_scale'], I / n['i_scale'], 0.0,
                          (T - n['t_ref']) / n['t_scale']]]).T

    def fwd_step_1ms(self, x, v, a, I, T, h, ie_buf):
        u = self.norm_u(x, v, a, I, T)
        Ie = 1 - min(abs(u[3, 0]), 1.0)
        nfl = self.build_nfl(u)
        h = self.gru_step(nfl, h)
        ie_buf = np.vstack([ie_buf[1:], [[Ie]]])
        Ie_filt = float((self.P['lag_w'].T @ ie_buf).ravel()[0])
        Fn = self.predict_force_norm(u, nfl, h, Ie, Ie_filt)
        return Fn * self.norm['f_scale'], h, ie_buf

    def force_of_Ie(self, Ie, c_slow, s_slow, b, phi, v_out):
        i_feat = np.array([[Ie], [Ie]])
        c_fast = self.mlp2('c_fast', i_feat).item()
        s_fast = self.mlp2('s_fast', i_feat)
        c = self.softplus(c_slow + c_fast) + 1e-6 + 2e-3
        sj = self.softplus(s_slow + s_fast) + 1e-6 + 5e-3 / 5
        Gc = self.monoG('G_c', Ie); Gs = self.monoG('G_s', Ie)
        return float(c * v_out * Gc + (sj.T @ phi).item() * Gs + b) * self.norm['f_scale']

    def masked_heads(self, x, v, a, T, h):
        u = self.norm_u(x, v, a, 0.0, T)
        nfl = self.build_nfl(u)
        h_m = self.gru_step(nfl, h)
        h2 = h_m[1:2, :].T
        v_out = float(u[1, 0]) * self.v_out_scale
        feat = np.vstack([h2, nfl, [[abs(v_out)]]])
        c_slow = self.mlp3('c_head', feat).item()
        s_slow = self.mlp3('s_head', feat)
        b = self.mlp3('b_head', np.vstack([h2, nfl])).item()
        phi = np.tanh(v_out / self.P['sigmas'])
        return c_slow, s_slow, b, phi, v_out

    def inverse_bisection(self, F_des, x, v, a, T, h, n_it=40):
        c_slow, s_slow, b, phi, v_out = self.masked_heads(x, v, a, T, h)
        Fg = np.array([self.force_of_Ie(Ie, c_slow, s_slow, b, phi, v_out)
                       for Ie in np.linspace(0, 1, 9)])
        Fd = min(max(F_des, Fg.min()), Fg.max())
        increasing = Fg[-1] > Fg[0]
        lo, hi = 0.0, 1.0
        for _ in range(n_it):
            mid = 0.5 * (lo + hi)
            Fm = self.force_of_Ie(mid, c_slow, s_slow, b, phi, v_out)
            if (increasing and Fm < Fd) or ((not increasing) and Fm > Fd):
                lo = mid
            else:
                hi = mid
        Ie_sol = 0.5 * (lo + hi)
        I_cmd = min(max((1 - Ie_sol) * 1.6, 0.0), 1.6)
        F_ach = self.force_of_Ie(Ie_sol, c_slow, s_slow, b, phi, v_out)
        return I_cmd, F_ach

    def inverse_head(self, F_des, x, v, a, T, h):
        n = self.norm
        u = self.norm_u(x, v, a, 0.0, T)
        nfl = self.build_nfl(u)
        h_m = self.gru_step(nfl, h)
        h2 = h_m[1:2, :].T
        feat = np.vstack([h2, nfl, u[1:2, :], [[F_des / n['f_scale']]]])
        z1 = self.relu(self.P['inv_head_W1'] @ feat + self.P['inv_head_b1'])
        z2 = self.relu(self.P['inv_head_W2'] @ z1 + self.P['inv_head_b2'])
        I_norm = 1.0 / (1.0 + math.exp(-float(np.asarray(self.P['inv_head_W3'] @ z2 + self.P['inv_head_b3']).ravel()[0])))
        return min(max(I_norm, 0.0), 1.0) * 1.6


# ============================================================ torch 参考
def torch_forward_seq(model, nc, df, n):
    xs = df["rod_length"].values[:n]; vs = df["velocity"].values[:n]
    aa = df["accel"].values[:n]; II = df["current"].values[:n]; TT = df["temp"].values[:n]
    sd = model.state_dict()
    a_lag = float(torch.sigmoid(sd["i_lag.log_a"]))
    K = int(6 * (-1.0 / math.log(1.0 - a_lag + 1e-12))) + 1
    w = a_lag * (1 - a_lag) ** np.arange(K)
    h = None; buf = np.zeros(K); Fs = np.zeros(n)
    for t in range(n):
        u = torch.tensor([[(xs[t]-nc["x_ref"])/nc["x_scale"], vs[t]/nc["v_scale"],
                            aa[t]/nc["a_scale"], II[t]/nc["i_scale"], 0.0,
                            (TT[t]-nc["t_ref"])/nc["t_scale"]]], dtype=torch.float64)
        nfl = static_nfl_v12(u, sigma_v=model.nfl_sigma_v)
        _, h = model.gru(nfl.unsqueeze(0), h)
        Ie = 1.0 - min(abs(float(u[0, 3])), 1.0)
        buf = np.concatenate([buf[1:], [Ie]])
        Ie_filt = float(w @ buf)
        v_out = u[:, 1:2] * model.v_out_scale
        feat = torch.cat([h[-1], nfl, v_out.abs()], dim=-1)
        c_slow = model.c_head(feat).squeeze(-1)
        s_slow = model.s_head(feat)
        b = model.b_head(torch.cat([h[-1], nfl], dim=-1)).squeeze(-1)
        i_feat = torch.tensor([[Ie, Ie_filt]], dtype=torch.float64)
        c = F.softplus(c_slow + model.c_fast(i_feat).squeeze(-1)) + 1e-6 + 2e-3
        s_j = F.softplus(s_slow + model.s_fast(i_feat)) + 1e-6 + 5e-3/5
        phi = torch.tanh(v_out.unsqueeze(-1) / model.sigma)
        Gc = model.G_c(torch.tensor([[Ie]], dtype=torch.float64)).squeeze(-1)
        Gs = model.G_s(torch.tensor([[Ie]], dtype=torch.float64)).squeeze(-1)
        Fn = c * v_out.squeeze(-1) * Gc + (s_j * phi).sum(-1) * Gs + b
        Fs[t] = float(Fn.item()) * nc["f_scale"]
    return Fs


def main():
    ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    nc, mc = ck["norm_cfg"], ck["model_cfg"]
    model = NLCSNN_v14(gru_hidden=mc["gru_hidden"], gru_layers=mc["gru_layers"],
                       mlp_hidden=mc["mlp_hidden"], g_hidden=mc["g_hidden"],
                       sigmas=tuple(mc["sigmas"]))
    model.load_state_dict(ck["state_dict"]); model.eval(); model.double()

    m = M(PKG / "v17_weights.mat")
    df = pd.read_csv(str(CSV))
    N = min(3000, len(df) - 10)
    xs = df["rod_length"].values[:N]; vs = df["velocity"].values[:N]
    aa = df["accel"].values[:N]; II = df["current"].values[:N]; TT = df["temp"].values[:N]

    # ---- 1. 前向 1ms 步进 ----
    F_ref = torch_forward_seq(model, nc, df, N)
    h = np.zeros((2, 16)); buf = np.zeros((m.K, 1)); F_mir = np.zeros(N)
    for t in range(N):
        F_mir[t], h, buf = m.fwd_step_1ms(xs[t], vs[t], aa[t], II[t], TT[t], h, buf)
    e1 = np.max(np.abs(F_mir - F_ref))
    print(f"[1] forward 1ms x{N}: max|err| = {e1:.3e} N  {'PASS' if e1 < 1e-6 else 'FAIL'}")

    # ---- 2. 宏步 dt=0.01 (10 子步 ZOH): torch 参考同样用 ZOH 输入 ----
    df_zoh = df.iloc[np.repeat(np.arange(N // 10) * 10, 10)].reset_index(drop=True)
    F_ref_zoh = torch_forward_seq(model, nc, df_zoh, len(df_zoh))
    F_ref_macro = F_ref_zoh[9::10]
    h = np.zeros((2, 16)); buf = np.zeros((m.K, 1)); Fm2 = []
    for blk in range(N // 10):
        t = blk * 10
        for _ in range(10):
            Fb, h, buf = m.fwd_step_1ms(xs[t], vs[t], aa[t], II[t], TT[t], h, buf)
        Fm2.append(Fb)
    e2 = np.max(np.abs(np.array(Fm2) - F_ref_macro))
    print(f"[2] forward macro dt=0.01: max|err| = {e2:.3e} N  {'PASS' if e2 < 1e-6 else 'FAIL'}")

    # ---- 3/4. 逆解: 先 torch 滚出各 idx 的 h_prev ----
    rng = np.random.default_rng(7)
    idx = sorted(rng.choice(np.arange(1200, N - 1), size=5, replace=False).tolist())
    h_torch_buf = {}
    hh = None
    for t in range(N):
        u = torch.tensor([[(xs[t]-nc["x_ref"])/nc["x_scale"], vs[t]/nc["v_scale"],
                            aa[t]/nc["a_scale"], II[t]/nc["i_scale"], 0.0,
                            (TT[t]-nc["t_ref"])/nc["t_scale"]]], dtype=torch.float64)
        nfl = static_nfl_v12(u.double(), sigma_v=model.nfl_sigma_v)
        _, hh = model.gru(nfl.unsqueeze(0), hh)
        if t in idx:
            h_torch_buf[t] = hh.clone()
    max_e3 = 0.0; max_e4 = 0.0; cases = []
    for t in idx:
        hh_t = h_torch_buf[t]
        x, v, a_, Tt = xs[t], vs[t], aa[t], TT[t]
        F_tgt = float(F_ref[t])
        u_m = torch.tensor([[(x-nc["x_ref"])/nc["x_scale"], v/nc["v_scale"],
                              a_/nc["a_scale"], 0.0, 0.0,
                              (Tt-nc["t_ref"])/nc["t_scale"]]], dtype=torch.float64)
        nfl_m = static_nfl_v12(u_m, sigma_v=model.nfl_sigma_v)
        _, h_m = model.gru(nfl_m.unsqueeze(0), hh_t)
        v_out = u_m[:, 1:2] * model.v_out_scale
        feat = torch.cat([h_m[-1], nfl_m, v_out.abs()], dim=-1)
        c_slow = model.c_head(feat).squeeze(-1); s_slow = model.s_head(feat)
        b = model.b_head(torch.cat([h_m[-1], nfl_m], dim=-1)).squeeze(-1)
        phi = torch.tanh(v_out.unsqueeze(-1) / model.sigma)
        def f_of_Ie(Ie):
            pair = torch.stack([Ie, Ie], dim=-1).unsqueeze(1)
            c = F.softplus(c_slow + model.c_fast(pair).squeeze(-1).squeeze(-1)) + 1e-6 + 2e-3
            s_j = F.softplus(s_slow + model.s_fast(pair).squeeze(1)) + 1e-6 + 5e-3/5
            Fn = c*v_out.squeeze(-1)*model.G_c(Ie.unsqueeze(-1)).squeeze(-1) \
                 + (s_j*phi).sum(-1)*model.G_s(Ie.unsqueeze(-1)).squeeze(-1) + b
            return Fn * nc["f_scale"]
        Fg = torch.stack([f_of_Ie(torch.tensor([gg], dtype=torch.float64)) for gg in np.linspace(0, 1, 9)])
        Fd = min(max(F_tgt, float(Fg.min())), float(Fg.max()))
        lo, hi = torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)
        inc = Fg[-1] > Fg[0]
        for _ in range(40):
            mid = (lo+hi)/2; Fm = f_of_Ie(mid)
            go = torch.where(inc, Fm < Fd, Fm > Fd)
            lo = torch.where(go, mid, lo); hi = torch.where(go, hi, mid)
        I_ref = float(((1-(lo+hi)/2).clamp(0, 1)*1.6).item())
        h_np = hh_t.squeeze(1).numpy()
        I_mir, _ = m.inverse_bisection(F_tgt, x, v, a_, Tt, h_np)
        e = abs(I_mir - I_ref); max_e3 = max(max_e3, e)
        # 逆头
        feat_h = torch.cat([h_m[-1], nfl_m, u_m[:, 1:2],
                            torch.tensor([[F_tgt / nc["f_scale"]]], dtype=torch.float64)], dim=-1)
        I_href = float((model.inv_head(feat_h).clamp(0, 1) * 1.6).item())
        I_hmir = m.inverse_head(F_tgt, x, v, a_, Tt, h_np)
        e4 = abs(I_hmir - I_href); max_e4 = max(max_e4, e4)
        cases.append((x, v, a_, Tt, h_np, F_tgt, I_ref, I_href))
    print(f"[3] inverse bisection x5: max|err| = {max_e3:.3e} A  {'PASS' if max_e3 < 1e-6 else 'FAIL'}")
    print(f"[4] inverse head x5:     max|err| = {max_e4:.3e} A  {'PASS' if max_e4 < 1e-9 else 'FAIL'}")

    from scipy.io import savemat
    g = {'fwd_x': xs, 'fwd_v': vs, 'fwd_a': aa, 'fwd_I': II, 'fwd_T': TT,
         'fwd_F_ref': F_ref, 'macro_F_ref': F_ref_macro,
         'inv_x': np.array([c[0] for c in cases]), 'inv_v': np.array([c[1] for c in cases]),
         'inv_a': np.array([c[2] for c in cases]), 'inv_T': np.array([c[3] for c in cases]),
         'inv_h': np.stack([c[4] for c in cases]),
         'inv_Fdes': np.array([c[5] for c in cases]),
         'inv_I_ref': np.array([c[6] for c in cases]),
         'head_I_ref': np.array([c[7] for c in cases])}
    savemat(str(PKG / "golden_vectors.mat"), g, oned_as="column")
    print("[*] wrote golden_vectors.mat")
    ok = (e1 < 1e-6) and (e2 < 1e-6) and (max_e3 < 1e-6) and (max_e4 < 1e-9)
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
