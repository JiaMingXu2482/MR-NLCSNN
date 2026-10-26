"""NLCSNN v12 — v11 + 磨平 NFL 输入特征 (根治 s_j 过零 100 倍崩塌)。

问题 (v11 E20/E40 验证图, SH_30mm_0A):
  - v11 输出基 Σ s_j·tanh(v/σ_j) 本身数值验证平滑 (慢扫过零跳变仅 7e-5)
  - 但实测 s_j 在过零 2ms 内从 0.48 崩到 0.0036 (100 倍), 形成 -900N→0N 近垂直台阶
  - 根因: 输入特征层 static_nfl 含陡峭特征
      sgn_v = tanh(50·v)          (v_norm 单位, ±60mm/s 内翻转)
      rebound: tanh(50·v_pos)
      slow_gate = exp(-80·|v|)    (v=0 处尖峰, ±40mm/s 内衰减完)
    网络顺着这些陡峭特征把 s_j 学出了近垂直翻转。

v12 改动 (最小化, 针对性):
  新增 static_nfl_v12: 35 维结构、顺序与 static_nfl 完全一致 (GRU 可直接热启动),
  仅把跟 v 有关的陡峭特征磨平, 磨平尺度与 v11 输出基统一哲学 (温和过渡):
      sgn_v         = tanh(v / σ_nfl)        [原 tanh(50·v)]
      rebound       = tanh(v_pos / σ_nfl)    [原 tanh(50·v_pos)]
      i·tanh(v_pos) = i·tanh(v_pos / σ_nfl)  [原 i·tanh(5·v_pos)]
      slow_gate     = exp(-|v| / σ_nfl)      [原 exp(-clamp(80·|v|))]
      σ_nfl = 0.1 (v_norm 单位 ≈ 100mm/s, 比 v11 最陡输出基 σ=0.5 更温和 6 倍,
              斜率从 50 降到 10; 工作速度 265mm/s 处饱和, 方向信息保留)
  其余 35 维特征 (kin/compression/curr_dyn/aux/hf_dyn/slow 其余) 保持原样。

保持 v11 全部保证:
  - F = c·v·G_c(I_e) + (Σ_j s_j·tanh(v/σ_j))·G_s(I_e), F·v ≥ 0
  - 零速连续、双 G 单调二分逆解、快速电流旁路、可学习电流延迟、最小阻尼下限

Warm-start: 从 v11 best 直接载入全部形状匹配参数 (35 维 NFL 不变, GRU/头/G/逆头/i_lag
  全对齐); NFL 分布变平滑, 训练自适应。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_v3 import STATIC_NFL_DIM
from .model_v7 import PositiveLinear, MonotonicG
from .model_v8 import _make_head, _make_inv_head
from .model_v10 import CurrentLag
from .model_v11 import _make_head_out


def static_nfl_v12(u, sigma_v=0.1):
    """磨平版静态 NFL。u: (..., 6) -> (..., 35), 结构/顺序与 static_nfl 一致。

    sigma_v: v 相关陡峭特征的磨平尺度 (v_norm 单位, 默认 0.1 ≈ 100mm/s,
             比 v11 最陡输出基 (σ=0.5, 15mm/s) 温和约 6 倍)。
    """
    x = u[..., 0:1]; v = u[..., 1:2]; a = u[..., 2:3]
    i = u[..., 3:4]; di = u[..., 4:5]; t = u[..., 5:6]
    v_abs = torch.abs(v)
    v_pos = torch.relu(v)
    v_neg = -torch.relu(-v)
    # 磨平: sgn_v 原为 tanh(50·v)
    sgn_v = torch.tanh(v / sigma_v)

    kin = torch.cat([v, x * v, v * i, v * t], dim=-1)                       # 4
    # 磨平: tanh(50·v_pos) -> tanh(v_pos/σ); i·tanh(5·v_pos) -> i·tanh(v_pos/σ)
    rebound = torch.cat([torch.tanh(2.0 * v_pos),
                         torch.tanh(v_pos / sigma_v),
                         i * torch.tanh(v_pos / sigma_v)], dim=-1)          # 3
    compression = torch.cat([v_neg * v_abs, v_neg ** 3,
                             a * v_neg, i * v_neg * v_abs], dim=-1)        # 4
    curr_dyn = torch.cat([di * v, di * a, torch.abs(di) * i], dim=-1)      # 3
    aux = torch.cat([sgn_v, i * sgn_v, x ** 2, x ** 3, a * i], dim=-1)     # 5
    hf_dyn = torch.cat([torch.abs(a), a * v,
                        torch.abs(a) * sgn_v, i * torch.abs(a)], dim=-1)   # 4
    x_abs = torch.abs(x)
    # 磨平: slow_gate 原为 exp(-clamp(80·|v|))
    slow_gate = torch.exp(-v_abs / sigma_v)
    slow = torch.cat([slow_gate, x * slow_gate, x_abs * slow_gate,
                      i * x * slow_gate, i * x_abs * slow_gate,
                      sgn_v * x_abs], dim=-1)                              # 6
    out = torch.cat([u, kin, rebound, compression, curr_dyn,
                     aux, hf_dyn, slow], dim=-1)                            # 35
    assert out.shape[-1] == STATIC_NFL_DIM, f"NFL 维度 {out.shape[-1]} != 35"
    return out


class NLCSNN_v12(nn.Module):
    def __init__(self, gru_hidden=64, gru_layers=2, mlp_hidden=128,
                 dropout=0.0, c_eps=1e-6, v_out_scale=30.0,
                 sigmas=(0.5, 2.0, 8.0), nfl_sigma_v=0.1,
                 g_hidden=64, g_skip=0.1,
                 init_tau_ms=20.0, c_min=2e-3, s_min=5e-3):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.c_eps = c_eps
        self.c_min = c_min
        self.s_min = s_min
        self.v_out_scale = v_out_scale
        self.sigmas = tuple(float(s) for s in sigmas)
        self.n_basis = len(self.sigmas)
        self.nfl_sigma_v = float(nfl_sigma_v)
        assert self.n_basis >= 1 and min(self.sigmas) > 0 and self.nfl_sigma_v > 0
        self.gru = nn.GRU(STATIC_NFL_DIM, gru_hidden, gru_layers,
                          batch_first=True, dropout=dropout if gru_layers > 1 else 0.0)
        feat_dim = gru_hidden + STATIC_NFL_DIM + 1
        self.c_head = _make_head(feat_dim, mlp_hidden, dropout)
        self.s_head = _make_head_out(feat_dim, mlp_hidden, dropout, self.n_basis)
        self.c_fast = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
        self.s_fast = nn.Sequential(nn.Linear(2, 32), nn.ReLU(),
                                    nn.Linear(32, self.n_basis))
        nn.init.zeros_(self.c_fast[-1].weight); nn.init.zeros_(self.c_fast[-1].bias)
        nn.init.zeros_(self.s_fast[-1].weight); nn.init.zeros_(self.s_fast[-1].bias)
        self.i_lag = CurrentLag(init_tau_ms=init_tau_ms)
        self.G_c = MonotonicG(hidden=g_hidden, i_ref=1.0, k_skip=g_skip)
        self.G_s = MonotonicG(hidden=g_hidden, i_ref=1.0, k_skip=g_skip)
        inv_dim = gru_hidden + STATIC_NFL_DIM + 2
        self.inv_head = _make_inv_head(inv_dim, mlp_hidden, dropout)
        self.register_buffer("sigma", torch.tensor(self.sigmas, dtype=torch.float32))
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[*] NLCSNN_v12: NFL=35(磨平σ_v={self.nfl_sigma_v}) -> GRU({gru_hidden}x{gru_layers}) "
              f"-> F=c·v·G_c+(Σ_j s_j·tanh(v/{self.sigmas}))·G_s [v11+磨平NFL] | "
              f"{n_params/1e6:.2f}M params", flush=True)

    def forward_features(self, u):
        """返回 (h_seq, nfl, v_out)。u: (B, L, 6)。nfl 用磨平版。"""
        nfl = static_nfl_v12(u, sigma_v=self.nfl_sigma_v)
        h_seq, _ = self.gru(nfl, None)
        v_out = u[:, :, 1:2] * self.v_out_scale
        return h_seq, nfl, v_out

    def _current_pair(self, u):
        I_norm = u[:, :, 3].abs().clamp(0.0, 1.0)
        I_e = 1.0 - I_norm
        I_e_filt = self.i_lag(I_e)
        return I_e, I_e_filt

    def friction_shape(self, v):
        return torch.tanh(v.unsqueeze(-1) / self.sigma)

    def forward_components(self, u):
        h_seq, nfl, v_out = self.forward_features(u)
        v = v_out.squeeze(-1)
        I_e, I_e_filt = self._current_pair(u)
        feat = torch.cat([h_seq, nfl, v.abs().unsqueeze(-1)], dim=-1)
        c_slow = self.c_head(feat).squeeze(-1)
        s_slow = self.s_head(feat)
        i_feat = torch.stack([I_e, I_e_filt], dim=-1)
        c_fast = self.c_fast(i_feat).squeeze(-1)
        s_fast = self.s_fast(i_feat)
        c = F.softplus(c_slow + c_fast) + self.c_eps + self.c_min
        s_j = F.softplus(s_slow + s_fast) + self.c_eps + self.s_min / self.n_basis
        Gc_I = self.G_c(I_e.unsqueeze(-1)).squeeze(-1)
        Gs_I = self.G_s(I_e.unsqueeze(-1)).squeeze(-1)
        fric = (s_j * self.friction_shape(v)).sum(-1) * Gs_I
        Force = c * v * Gc_I + fric
        return Force, c, s_j, Gc_I, Gs_I

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
        self.eval()
        h_seq, nfl, v_out = self.forward_features(u_hist_masked)
        h_last, nfl_last = h_seq[:, -1], nfl[:, -1]
        v = v_out[:, -1, 0]
        feat = torch.cat([h_last, nfl_last, v.abs().unsqueeze(-1)], dim=-1)
        c_slow = self.c_head(feat).squeeze(-1)
        s_slow = self.s_head(feat)
        phi = self.friction_shape(v)

        def f_of_Ie(Ie):
            Ie1 = Ie.unsqueeze(-1)
            i_pair = torch.stack([Ie, Ie], dim=-1).unsqueeze(1)
            c = F.softplus(c_slow + self.c_fast(i_pair).squeeze(-1).squeeze(-1)) + self.c_eps
            s_j = F.softplus(s_slow + self.s_fast(i_pair).squeeze(1)) + self.c_eps
            fric = (s_j * phi).sum(-1) * self.G_s(Ie1).squeeze(-1)
            return c * v * self.G_c(Ie1).squeeze(-1) + fric

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

    def load_from_v11(self, v11_ckpt_path):
        """从 v11 best 热启动: 全部形状匹配参数直接载入 (NFL 35 维不变)。"""
        ckpt = torch.load(v11_ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        own = self.state_dict()
        n = 0
        for k in list(own.keys()):
            if k in sd and sd[k].shape == own[k].shape:
                own[k].copy_(sd[k])
                n += 1
        self.load_state_dict(own)
        print(f"[*] v12: 从 v11 热启动, 载入 {n} 个参数张量; NFL 磨平 σ_v={self.nfl_sigma_v}",
              flush=True)
        return ckpt.get("norm_cfg")
