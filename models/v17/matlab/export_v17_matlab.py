#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17 -> MATLAB 导出脚本
======================
把 v17 最优 checkpoint (E330, pooled NRMSE 16.80%) 的权重导出为
v17_weights.mat, 供纯 MATLAB 函数 (v17_damper_init/step, v17_inverse_*)
直接加载。权重按 torch Linear 的 (out,in) 方向原样存为 double,
MATLAB 里直接 W*x 即可, 无需转置。

架构 (见 mr_nlcsnn/model_v14.py, 训练脚本 AA02Train/train_v14.py):
  u (归一化 6 维) = [(x-x_ref)/x_scale, v/v_scale, a/a_scale,
                     I/i_scale, dI(训练恒置0), (T-t_ref)/t_scale]
  nfl = static_nfl_v12(u, sigma_v=0.1)            # 35 维
  GRU(35 -> 16, 2 层)                            # 隐状态 2x16
  v_out = v_norm * 30
  feat  = [h_last(16); nfl(35); |v_out|(1)]      # 52 维
  c_slow = MLP(feat) 48->24->1 ; s_slow = MLP(feat) 48->24->5
  b      = MLP([h;nfl]) 48->24->1                # 无约束偏置
  Ie = 1 - clamp(|I_norm|,0,1) ; Ie_filt = CurrentLag(Ie)  # 反向核, 见 README
  c_fast = MLP2([Ie;Ie_filt]) ; s_fast = MLP2([Ie;Ie_filt]) -> 5
  c = softplus(c_slow+c_fast)+1e-6+2e-3
  s_j = softplus(s_slow+s_fast)+1e-6+5e-3/5
  G_c/G_s = MonotonicG(24 隐单元, i_ref=1.0, k_skip=0.1) @ Ie
  F_norm = c*v_out*G_c + sum_j s_j*tanh(v_out/sigma_j)*G_s + b
  F = F_norm * f_scale  [N]
  逆头: [h(16); nfl(35); v_norm(1); F_norm(1)] -> sigmoid MLP 48->24->1
        -> I_norm in [0,1] (训练时特征含真电流, 部署见 README 说明)

运行:
  cd ~/workspace/mr-nlcsnn-v3
  ./venv/bin/python /path/to/export_v17_matlab.py
输出: <包目录>/v17_weights.mat
"""
import math
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

REPO = Path(__file__).resolve().parent.parent if (Path(__file__).parent / "mr_nlcsnn").exists() \
    else Path.home() / "workspace" / "mr-nlcsnn-v3"
CKPT = REPO / "training_outputs" / "v17_20261010_045838" / "nlcsnn_v17_best.pth"
OUT = Path.home() / "workspace" / "your_files" / "v17_pmpc_package" / "v17_weights.mat"


def t(x):
    return np.asarray(x.detach().cpu(), dtype=np.float64)


def main():
    ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    nc, mc = ck["norm_cfg"], ck["model_cfg"]
    print(f"[*] checkpoint total_epoch={ck['total_epoch']}, "
          f"pooled={ck['pooled_nrmse_pct']:.4f}%, inv_mae={ck['inv_mae_A']:.6f}A")

    def gru_layer(pfx):
        Wih = t(sd[f"gru.weight_ih_l{pfx}"])   # (48, in): [ir; iz; in]
        Whh = t(sd[f"gru.weight_hh_l{pfx}"])   # (48, 16): [hr; hz; hn]
        bih = t(sd[f"gru.bias_ih_l{pfx}"])
        bhh = t(sd[f"gru.bias_hh_l{pfx}"])
        H = 16
        d = {}
        for i, nm in enumerate(["r", "z", "n"]):
            d[f"W_i{nm}{pfx}"] = Wih[i * H:(i + 1) * H, :]
            d[f"W_h{nm}{pfx}"] = Whh[i * H:(i + 1) * H, :]
            d[f"b_i{nm}{pfx}"] = bih[i * H:(i + 1) * H]
            d[f"b_h{nm}{pfx}"] = bhh[i * H:(i + 1) * H]
        return d

    d = {}
    d.update(gru_layer(0))
    d.update(gru_layer(1))

    def head(pfx, layers):
        # layers: [(Wkey_suffix, ...)] -> W1,b1,W2,b2,W3,b3
        o = {}
        o[f"{pfx}_W1"] = t(sd[f"{pfx}.0.weight"]); o[f"{pfx}_b1"] = t(sd[f"{pfx}.0.bias"])
        o[f"{pfx}_W2"] = t(sd[f"{pfx}.3.weight"]); o[f"{pfx}_b2"] = t(sd[f"{pfx}.3.bias"])
        o[f"{pfx}_W3"] = t(sd[f"{pfx}.5.weight"]); o[f"{pfx}_b3"] = t(sd[f"{pfx}.5.bias"])
        return o

    d.update(head("c_head", None)); d.update(head("s_head", None))
    d.update(head("b_head", None)); d.update(head("inv_head", None))

    def fast(pfx):
        o = {}
        o[f"{pfx}_W1"] = t(sd[f"{pfx}.0.weight"]); o[f"{pfx}_b1"] = t(sd[f"{pfx}.0.bias"])
        o[f"{pfx}_W2"] = t(sd[f"{pfx}.2.weight"]); o[f"{pfx}_b2"] = t(sd[f"{pfx}.2.bias"])
        return o

    d.update(fast("c_fast")); d.update(fast("s_fast"))

    def monoG(pfx):
        o = {}
        o[f"{pfx}_Wr1"] = t(sd[f"{pfx}.mlp.0.w_raw"]); o[f"{pfx}_b1"] = t(sd[f"{pfx}.mlp.0.bias"])
        o[f"{pfx}_Wr2"] = t(sd[f"{pfx}.mlp.2.w_raw"]); o[f"{pfx}_b2"] = t(sd[f"{pfx}.mlp.2.bias"])
        o[f"{pfx}_Wr3"] = t(sd[f"{pfx}.mlp.4.w_raw"]); o[f"{pfx}_b3"] = t(sd[f"{pfx}.mlp.4.bias"])
        return o

    d.update(monoG("G_c")); d.update(monoG("G_s"))

    log_a = float(sd["i_lag.log_a"])
    a_lag = 1.0 / (1.0 + math.exp(-log_a))
    tau_ms = -1.0 / math.log(1.0 - a_lag + 1e-12)
    kmax = int(6 * tau_ms) + 1

    d["norm"] = {
        "x_ref": float(nc["x_ref"]), "x_scale": float(nc["x_scale"]),
        "v_scale": float(nc["v_scale"]), "a_scale": float(nc["a_scale"]),
        "i_scale": float(nc["i_scale"]), "di_scale": float(nc["di_scale"]),
        "t_ref": float(nc["t_ref"]), "t_scale": float(nc["t_scale"]),
        "f_scale": float(nc["f_scale"]),
    }
    d["cfg"] = {
        "gru_hidden": 16, "gru_layers": 2, "mlp_hidden": 48, "g_hidden": 24,
        "n_basis": 5,
        "sigmas": np.array(t(sd["sigma"]), dtype=np.float64),
        "nfl_sigma_v": 0.1, "v_out_scale": 30.0,
        "c_eps": 1e-6, "c_min": 2e-3, "s_min": 5e-3,
        "a_lag": a_lag, "lag_tau_ms": tau_ms, "lag_kmax": kmax,
        "g_k_skip": 0.1, "g_i_ref": 1.0, "i_max": 1.6,
    }
    d["meta"] = {
        "version": "v17", "total_epoch": int(ck["total_epoch"]),
        "pooled_nrmse_pct": float(ck["pooled_nrmse_pct"]),
        "inv_mae_A": float(ck["inv_mae_A"]),
        "n_params": 20998,
        "note": "v17 best (E330). GRU 2x16 exact from v15; wider deploy-cheap heads.",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(OUT), d, oned_as="column")
    print(f"[*] wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")
    print(f"[*] lag: a={a_lag:.6f}, tau={tau_ms:.2f}ms, KMAX={kmax}")


if __name__ == "__main__":
    main()
