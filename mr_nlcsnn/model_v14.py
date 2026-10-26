"""NLCSNN v13 — v12 + 自由偏置头, 取消 F·v≥0 硬约束。

背景 (v12 E20, 2026-10-03):
  - SH_30mm_0A 真值在 v=0 处约 -700N (v=+0.8mm/s 时 F=-721N, F·v<0),
    过零时真值平滑, 根本没有台阶。
  - v8 起架构强制 F(0)=0、F·v≥0, 与实测数据根本冲突 —— “台阶”是架构逼出来的。
  - 用户决定: 取消强制 F·v≥0, 重构。

v13 改动 (最小化):
  F = c·v·G_c(I_e) + (Σ_j s_j·tanh(v/σ_j))·G_s(I_e) + b(h, nfl)
  - b_head: MLP([h, nfl]) -> 无约束标量 (末层零初始化, 初值 b≈0 即 v12 行为)。
    可学习零速偏置 (气室预载/密封摩擦/传感器零漂), F(0)=b≠0 允许。
  - c, s_j ≥ 0 (softplus) 保留: 耗散主干仍良态, 偏置与阻尼解耦。
  - G_c, G_s 单调保留: 逆解时 b 由历史固定 (不随待求电流变),
    f(I_e) 仍对 I_e 单调, 二分逆解继续可用。
  - F·v≥0 不再保证 (用户取消); 零速连续保持 (b 光滑)。

Warm-start: 从 v12 best 载入全部形状匹配参数; b_head 全新初始化 (b≈0)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_v3 import STATIC_NFL_DIM
from .model_v7 import PositiveLinear, MonotonicG
from .model_v8 import _make_head, _make_inv_head
from .model_v10 import CurrentLag
from .model_v11 import _make_head_out
from .model_v12 import static_nfl_v12


def _make_bias_head(in_dim, mlp_hidden, dropout):
    """无约束偏置头: Linear->ReLU->Dropout->Linear->ReLU->Linear, 末层零初始化 (b≈0)。"""
    head = nn.Sequential(
        nn.Linear(in_dim, mlp_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden, mlp_hidden // 2),
        nn.ReLU(),
        nn.Linear(mlp_hidden // 2, 1),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class NLCSNN_v14(nn.Module):
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
        # 偏置头: [h, nfl] -> 无约束 (初值 0)
        self.b_head = _make_bias_head(gru_hidden + STATIC_NFL_DIM, mlp_hidden, dropout)
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
        print(f"[*] NLCSNN_v14: F=c·v·G_c+(Σ_j s_j·tanh(v/{self.sigmas}))·G_s+b(h,nfl) "
              f"[v12+自由偏置, 取消F·v≥0] | {n_params/1e6:.2f}M params", flush=True)

    def forward_features(self, u):
        """返回 (h_seq, nfl, v_out)。u: (B, L, 6)。"""
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
        b = self.b_head(torch.cat([h_seq, nfl], dim=-1)).squeeze(-1)
        i_feat = torch.stack([I_e, I_e_filt], dim=-1)
        c_fast = self.c_fast(i_feat).squeeze(-1)
        s_fast = self.s_fast(i_feat)
        c = F.softplus(c_slow + c_fast) + self.c_eps + self.c_min
        s_j = F.softplus(s_slow + s_fast) + self.c_eps + self.s_min / self.n_basis
        Gc_I = self.G_c(I_e.unsqueeze(-1)).squeeze(-1)
        Gs_I = self.G_s(I_e.unsqueeze(-1)).squeeze(-1)
        fric = (s_j * self.friction_shape(v)).sum(-1) * Gs_I
        Force = c * v * Gc_I + fric + b
        return Force, c, s_j, Gc_I, Gs_I, b

    def forward(self, u):
        f, _, _, _, _, _ = self.forward_components(u)
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
        b = self.b_head(torch.cat([h_last, nfl_last], dim=-1)).squeeze(-1)
        phi = self.friction_shape(v)

        def f_of_Ie(Ie):
            Ie1 = Ie.unsqueeze(-1)
            i_pair = torch.stack([Ie, Ie], dim=-1).unsqueeze(1)
            # 注意: 与 forward_components 一致, c/s 必须加 c_min / s_min/n_basis 下限,
            # 否则 f_of_Ie 与前向不一致, 二分会收敛到错误电流或卡在边界。
            c = F.softplus(c_slow + self.c_fast(i_pair).squeeze(-1).squeeze(-1)) + self.c_eps + self.c_min
            s_j = F.softplus(s_slow + self.s_fast(i_pair).squeeze(1)) + self.c_eps + self.s_min / self.n_basis
            fric = (s_j * phi).sum(-1) * self.G_s(Ie1).squeeze(-1)
            return c * v * self.G_c(Ie1).squeeze(-1) + fric + b

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

    def load_from_v12(self, v12_ckpt_path):
        """从 v12 best 热启动: 全部形状匹配参数直接载入; b_head 全新 (b≈0)。"""
        ckpt = torch.load(v12_ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        own = self.state_dict()
        n = 0
        for k in list(own.keys()):
            if k in sd and sd[k].shape == own[k].shape:
                own[k].copy_(sd[k])
                n += 1
        self.load_state_dict(own)
        print(f"[*] v13: 从 v12 热启动, 载入 {n} 个参数张量; b_head 全新初始化 (b≈0)",
              flush=True)
        return ckpt.get("norm_cfg")
