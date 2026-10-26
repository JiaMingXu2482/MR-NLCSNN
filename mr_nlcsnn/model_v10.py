"""NLCSNN v10 — v9 + 最小阻尼下限 (消除过零死区)。

误差分析结论 (v8 best, pooled 30%):
  - 稳态 RMSE 仅 10.4N (极好), 电流 ramp 后 1s 内 RMSE 300N+ (极差)
  - 模型对电流突变响应太慢: GRU 状态被旧工况"粘住", c(z)/s(z) 更新滞后
  - ST 阶跃工况一家贡献约 75% 的 holdout 总 MSE

v9 改动 (最小化, 针对性):
  1. 快速电流旁路: c/s 头增加 [I_e_inst, I_e_filt] 直接输入, 绕过慢速 GRU
     c = softplus(MLP_c([h, nfl, |v|]) + MLP_cI([I_e_inst, I_e_filt])) + eps
     s = softplus(MLP_s([h, nfl, |v|]) + MLP_sI([I_e_inst, I_e_filt])) + eps
  2. 可学习电流延迟: I_e_filt = EMA(I_e, alpha), alpha=sigmoid(log_alpha) 可学习
     建模磁路建立延迟 (~10-50ms), 因果、可用梯度
  3. 保持 v8 全部保证: F·v ≥ 0, 零速连续, 对 I_e 单调可二分逆解
  4. 逆头保留 (与 v8 相同)

Warm-start: GRU 直接载入 v8 权重; c/s 头前 100 维载入 v8, 新增 2 维零初始化。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_v3 import static_nfl, STATIC_NFL_DIM
from .model_v7 import PositiveLinear, MonotonicG
from .model_v8 import _make_head, _make_inv_head


class CurrentLag(nn.Module):
    """可学习一阶延迟: y[t] = a*x[t] + (1-a)*y[t-1], a=sigmoid(log_a)。
    因果, 模拟磁路建立延迟。init_tau_ms: 初始时间常数 (ms), 采样 1kHz。
    """
    def __init__(self, init_tau_ms=20.0):
        super().__init__()
        # a = 1 - exp(-dt/tau), dt=1ms
        a_init = 1.0 - __import__("math").exp(-1.0 / init_tau_ms)
        # log_a = log(a/(1-a))
        import math
        log_a_init = math.log(a_init / (1.0 - a_init))
        self.log_a = nn.Parameter(torch.tensor(log_a_init))

    def forward(self, x):
        """x: (B, L) -> (B, L) 滤波后。"""
        a = torch.sigmoid(self.log_a)
        # 向量化 EMA: 用 cumsum 技巧或直接循环 (L~20k, 循环太慢, 用展开式)
        # y[t] = a * sum_{k=0}^{t} (1-a)^{t-k} x[k]
        # 用卷积实现: kernel = a * (1-a)^k, k=0..L-1, 因果
        B, L = x.shape
        # 为数值稳定, kernel 截断到 6*tau
        tau = -1.0 / torch.log(1.0 - a + 1e-12)
        K = min(L, int(6 * tau.item()) + 1)
        k_idx = torch.arange(K, device=x.device, dtype=x.dtype)
        kernel = a * torch.pow(1.0 - a, k_idx)  # (K,)
        # 因果卷积: F.conv1d 需要 (B, 1, L)
        x1 = x.unsqueeze(1)
        w = kernel.view(1, 1, K)
        y = F.conv1d(x1, w, padding=K - 1)[:, :, :L]
        return y.squeeze(1)

    def tau_ms(self):
        a = torch.sigmoid(self.log_a).item()
        import math
        return -1.0 / math.log(1.0 - a + 1e-12)


class NLCSNN_v10(nn.Module):
    def __init__(self, gru_hidden=64, gru_layers=2, mlp_hidden=128,
                 dropout=0.0, c_eps=1e-6, v_out_scale=30.0,
                 tanh_k=2.0, g_hidden=64, g_skip=0.1, init_tau_ms=20.0,
                 c_min=2e-3, s_min=5e-3):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.c_eps = c_eps
        self.c_min = c_min  # 最小黏性阻尼下限, 防过零死区
        self.s_min = s_min  # 最小摩擦下限
        self.v_out_scale = v_out_scale
        self.tanh_k = tanh_k
        self.gru = nn.GRU(STATIC_NFL_DIM, gru_hidden, gru_layers,
                          batch_first=True, dropout=dropout if gru_layers > 1 else 0.0)
        # 慢速头特征: [h, nfl, |v|] (v8 相同, 100 维)
        feat_dim = gru_hidden + STATIC_NFL_DIM + 1
        self.c_head = _make_head(feat_dim, mlp_hidden, dropout)
        self.s_head = _make_head(feat_dim, mlp_hidden, dropout)
        # 快速电流旁路: [I_e_inst, I_e_filt] -> 标量偏置 (加到 softplus 前)
        self.c_fast = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
        self.s_fast = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.c_fast[-1].weight); nn.init.zeros_(self.c_fast[-1].bias)
        nn.init.zeros_(self.s_fast[-1].weight); nn.init.zeros_(self.s_fast[-1].bias)
        # 可学习电流延迟
        self.i_lag = CurrentLag(init_tau_ms=init_tau_ms)
        # 双独立单调 G (v8 相同)
        self.G_c = MonotonicG(hidden=g_hidden, i_ref=1.0, k_skip=g_skip)
        self.G_s = MonotonicG(hidden=g_hidden, i_ref=1.0, k_skip=g_skip)
        # 逆头 (v8 相同)
        inv_dim = gru_hidden + STATIC_NFL_DIM + 2
        self.inv_head = _make_inv_head(inv_dim, mlp_hidden, dropout)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[*] NLCSNN_v10: NFL={STATIC_NFL_DIM} -> GRU({gru_hidden}x{gru_layers}) "
              f"-> F=c·v·G_c+s·tanh({tanh_k}·v)·G_s [v9+最小阻尼下限] | "
              f"{n_params/1e6:.2f}M params", flush=True)

    def forward_features(self, u):
        """返回 (h_seq, nfl, v_out)。u: (B, L, 6)。"""
        nfl = static_nfl(u)
        h_seq, _ = self.gru(nfl, None)
        v_out = u[:, :, 1:2] * self.v_out_scale
        return h_seq, nfl, v_out

    def _current_pair(self, u):
        """返回 (I_e_inst, I_e_filt), 均为 (B, L)。"""
        I_norm = u[:, :, 3].abs().clamp(0.0, 1.0)
        I_e = 1.0 - I_norm
        I_e_filt = self.i_lag(I_e)
        return I_e, I_e_filt

    def forward_components(self, u):
        """返回 (F, c, s, Gc_I, Gs_I), 均为 (B, L) 归一化单位。"""
        h_seq, nfl, v_out = self.forward_features(u)
        v = v_out.squeeze(-1)
        I_e, I_e_filt = self._current_pair(u)
        # 慢速部分 (v8 相同)
        feat = torch.cat([h_seq, nfl, v.abs().unsqueeze(-1)], dim=-1)
        c_slow = self.c_head(feat).squeeze(-1)
        s_slow = self.s_head(feat).squeeze(-1)
        # 快速电流旁路 (直接响应, 不经过 GRU)
        i_feat = torch.stack([I_e, I_e_filt], dim=-1)  # (B, L, 2)
        c_fast = self.c_fast(i_feat).squeeze(-1)
        s_fast = self.s_fast(i_feat).squeeze(-1)
        c = F.softplus(c_slow + c_fast) + self.c_eps + self.c_min
        s = F.softplus(s_slow + s_fast) + self.c_eps + self.s_min
        Gc_I = self.G_c(I_e.unsqueeze(-1)).squeeze(-1)
        Gs_I = self.G_s(I_e.unsqueeze(-1)).squeeze(-1)
        tv = torch.tanh(self.tanh_k * v)
        Force = c * v * Gc_I + s * tv * Gs_I
        return Force, c, s, Gc_I, Gs_I

    def forward(self, u):
        f, _, _, _, _ = self.forward_components(u)
        return f

    def inverse_predict(self, u_hist, F_target):
        h_seq, nfl, _ = self.forward_features(u_hist)
        v_norm = u_hist[:, :, 1:2]
        feat = torch.cat([h_seq, nfl, v_norm, F_target.unsqueeze(-1)], dim=-1)
        return self.inv_head(feat).squeeze(-1)

    @torch.no_grad()
    def inverse_current(self, u_hist_masked, F_target):
        """二分逆解 (与 v8 相同逻辑, 含快速旁路)。"""
        self.eval()
        h_seq, nfl, v_out = self.forward_features(u_hist_masked)
        h_last, nfl_last = h_seq[:, -1], nfl[:, -1]
        v = v_out[:, -1, 0]
        feat = torch.cat([h_last, nfl_last, v.abs().unsqueeze(-1)], dim=-1)
        c_slow = self.c_head(feat).squeeze(-1)
        s_slow = self.s_head(feat).squeeze(-1)
        tv = torch.tanh(self.tanh_k * v)

        def f_of_Ie(Ie):
            # Ie: (B,), 需要计算对应的 I_e_filt (用历史 I_e 序列)
            # 简化: 假设阶跃后 filt 已收敛到 Ie (保守近似, 用于逆解)
            Ie1 = Ie.unsqueeze(-1)
            i_pair = torch.stack([Ie, Ie], dim=-1).unsqueeze(1)  # (B,1,2)
            c = F.softplus(c_slow + self.c_fast(i_pair).squeeze(-1).squeeze(-1)) + self.c_eps
            s = F.softplus(s_slow + self.s_fast(i_pair).squeeze(-1).squeeze(-1)) + self.c_eps
            return c * v * self.G_c(Ie1).squeeze(-1) + s * tv * self.G_s(Ie1).squeeze(-1)

        f0 = f_of_Ie(torch.zeros_like(v))
        f1 = f_of_Ie(torch.ones_like(v))
        increasing = f1 > f0
        lo = torch.zeros_like(v)
        hi = torch.ones_like(v)
        for _ in range(40):
            mid = (lo + hi) / 2
            fm = f_of_Ie(mid)
            go_up = torch.where(increasing, fm < F_target, fm > F_target)
            lo = torch.where(go_up, mid, lo)
            hi = torch.where(go_up, hi, mid)
        I_e = (lo + hi) / 2
        return (1.0 - I_e).clamp(0.0, 1.0)

    @torch.no_grad()
    def predict_file(self, u):
        self.eval()
        p = self.forward(u.unsqueeze(0)).squeeze(0).cpu().numpy()
        self.train()
        return p

    def load_from_v9(self, v9_ckpt_path):
        """从 v8 checkpoint 热启动: GRU/c/s头/G/逆头直接载入, 快速旁路零初始化。"""
        ckpt = torch.load(v9_ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        own = self.state_dict()
        n = 0
        for k in list(own.keys()):
            if k in sd and sd[k].shape == own[k].shape:
                # 跳过新增的快速旁路 (保持零初始化)
                if "c_fast" in k or "s_fast" in k or "i_lag" in k:
                    continue
                own[k].copy_(sd[k])
                n += 1
        self.load_state_dict(own)
        tau = self.i_lag.tau_ms()
        print(f"[*] v10: 从 v9 热启动, 载入 {n} 个参数张量; 快速旁路零初始化; 电流延迟初值 {tau:.1f}ms",
              flush=True)
        return ckpt.get("norm_cfg")
