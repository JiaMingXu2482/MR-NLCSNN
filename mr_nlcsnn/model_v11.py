"""NLCSNN v11 — v10 + 多尺度平滑 tanh 基函数 (根治过零台阶跳变)。

问题 (v10 E20/E40 验证图, SH 工况):
  - 摩擦项 s·tanh(2·v) 在过零处形成近垂直台阶跳变, 误差尖峰 ±150N (SH_1A) / ±700N (SH_0A)
  - 真实力是平滑过零的; tanh 本身平滑, 但 tanh_k=2.0 使过渡带过窄 (~±30mm/s)

v11 改动 (最小化, 针对性):
  摩擦项由单尺度 tanh 改为多尺度平滑基展开:
      F_fric(v) = Σ_j s_j(·) · tanh(v / σ_j),   j = 1..J
      σ = [0.5, 2.0, 8.0] 固定 (v_out 单位, 对应物理 ~[17, 67, 267] mm/s)
      s_j(·) = softplus(slow_j + fast_j) + eps + s_min/J  ≥ 0
  - 每个基都是奇函数、光滑、单调、过零 (tanh 天然平滑过零, 用户要求)
  - σ 固定不学习: 结构上禁止基函数变陡成阶跃, 最陡过渡带也铺展在 ±50mm/s 上
  - 网络只学习非负组合权重 s_j(·): 仍是光滑奇单调饱和函数的通用逼近子
  - 完整公式: F = c·v·G_c(I_e) + (Σ_j s_j·tanh(v/σ_j))·G_s(I_e)

保持 v10 全部保证:
  - F·v ≥ 0 (每项: c·v²·G_c ≥ 0; tanh(v/σ) 与 v 同号 → 求和项 ·v ≥ 0)
  - 零速连续 (v→0 时每项 →0, 且过零斜率有限温和)
  - 对 I_e 单调 → 二分逆解 (G_c, G_s 独立单调, 磨擦形状 Φ(v)=Σs_j·tanh 与 v 同号)
  - 快速电流旁路 (每基独立) + 可学习电流延迟 + 最小阻尼下限

Warm-start: 从 v10 best 载入 GRU/c头/G/逆头/i_lag/c_fast;
  s 头末层按 1/J 复制 (bias 移 -logJ, 使 Σ_j s_j ≈ v10 的 s, 初始为 v10 摩擦的光滑化版本)。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_v3 import static_nfl, STATIC_NFL_DIM
from .model_v7 import PositiveLinear, MonotonicG
from .model_v8 import _make_head, _make_inv_head
from .model_v10 import CurrentLag


def _make_head_out(in_dim, mlp_hidden, dropout, out_dim):
    """v4 式 MLP 头, 输出 out_dim (v8 的 _make_head 的通用版, 同样保守初值)。"""
    head = nn.Sequential(
        nn.Linear(in_dim, mlp_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden, mlp_hidden // 2),
        nn.ReLU(),
        nn.Linear(mlp_hidden // 2, out_dim),
    )
    nn.init.zeros_(head[-1].weight)
    # 保守初值: softplus(-4)≈0.018 (v7 已验证)
    nn.init.constant_(head[-1].bias, -4.0)
    return head


class NLCSNN_v11(nn.Module):
    def __init__(self, gru_hidden=64, gru_layers=2, mlp_hidden=128,
                 dropout=0.0, c_eps=1e-6, v_out_scale=30.0,
                 sigmas=(0.5, 2.0, 8.0), g_hidden=64, g_skip=0.1,
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
        assert self.n_basis >= 1 and min(self.sigmas) > 0
        self.gru = nn.GRU(STATIC_NFL_DIM, gru_hidden, gru_layers,
                          batch_first=True, dropout=dropout if gru_layers > 1 else 0.0)
        # 慢速头特征: [h, nfl, |v|] (v10 相同)
        feat_dim = gru_hidden + STATIC_NFL_DIM + 1
        self.c_head = _make_head(feat_dim, mlp_hidden, dropout)
        # s 头输出 J 个基权重
        self.s_head = _make_head_out(feat_dim, mlp_hidden, dropout, self.n_basis)
        # 快速电流旁路: [I_e_inst, I_e_filt] -> c 偏置 / s 每基偏置
        self.c_fast = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
        self.s_fast = nn.Sequential(nn.Linear(2, 32), nn.ReLU(),
                                    nn.Linear(32, self.n_basis))
        nn.init.zeros_(self.c_fast[-1].weight); nn.init.zeros_(self.c_fast[-1].bias)
        nn.init.zeros_(self.s_fast[-1].weight); nn.init.zeros_(self.s_fast[-1].bias)
        # 可学习电流延迟
        self.i_lag = CurrentLag(init_tau_ms=init_tau_ms)
        # 双独立单调 G (v10 相同)
        self.G_c = MonotonicG(hidden=g_hidden, i_ref=1.0, k_skip=g_skip)
        self.G_s = MonotonicG(hidden=g_hidden, i_ref=1.0, k_skip=g_skip)
        # 逆头 (v10 相同)
        inv_dim = gru_hidden + STATIC_NFL_DIM + 2
        self.inv_head = _make_inv_head(inv_dim, mlp_hidden, dropout)
        # σ 注册为 buffer (不学习, 固定温和)
        self.register_buffer("sigma", torch.tensor(self.sigmas, dtype=torch.float32))
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[*] NLCSNN_v11: NFL={STATIC_NFL_DIM} -> GRU({gru_hidden}x{gru_layers}) "
              f"-> F=c·v·G_c+(Σ_j s_j·tanh(v/{self.sigmas}))·G_s [v10+多尺度平滑基] | "
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

    def friction_shape(self, v):
        """多尺度基形状 Φ_j(v) = tanh(v/σ_j), (..., J)。奇、光滑、单调、过零。"""
        return torch.tanh(v.unsqueeze(-1) / self.sigma)

    def forward_components(self, u):
        """返回 (F, c, s_basis, Gc_I, Gs_I)。F,c,Gc_I,Gs_I: (B,L); s_basis: (B,L,J)。"""
        h_seq, nfl, v_out = self.forward_features(u)
        v = v_out.squeeze(-1)
        I_e, I_e_filt = self._current_pair(u)
        # 慢速部分
        feat = torch.cat([h_seq, nfl, v.abs().unsqueeze(-1)], dim=-1)
        c_slow = self.c_head(feat).squeeze(-1)
        s_slow = self.s_head(feat)  # (B, L, J)
        # 快速电流旁路
        i_feat = torch.stack([I_e, I_e_filt], dim=-1)  # (B, L, 2)
        c_fast = self.c_fast(i_feat).squeeze(-1)
        s_fast = self.s_fast(i_feat)  # (B, L, J)
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
        """二分逆解 (与 v10 相同逻辑, 摩擦项换成多尺度基)。"""
        self.eval()
        h_seq, nfl, v_out = self.forward_features(u_hist_masked)
        h_last, nfl_last = h_seq[:, -1], nfl[:, -1]
        v = v_out[:, -1, 0]
        feat = torch.cat([h_last, nfl_last, v.abs().unsqueeze(-1)], dim=-1)
        c_slow = self.c_head(feat).squeeze(-1)
        s_slow = self.s_head(feat)  # (B, J)
        phi = self.friction_shape(v)  # (B, J)

        def f_of_Ie(Ie):
            Ie1 = Ie.unsqueeze(-1)
            i_pair = torch.stack([Ie, Ie], dim=-1).unsqueeze(1)  # (B,1,2)
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

    def load_from_v10(self, v10_ckpt_path):
        """从 v10 best 热启动。
        GRU/c头/G/逆头/i_lag/c_fast 直接载入 (形状相同);
        s 头末层 (1->J) 按 1/J 复制: bias 移 -logJ, 使 Σ_j s_j ≈ v10 的 s
        (初始摩擦 = v10 摩擦的光滑化版本); s_fast 末层同样处理。
        """
        ckpt = torch.load(v10_ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        own = self.state_dict()
        n = 0
        J = self.n_basis
        for k in list(own.keys()):
            if k not in sd:
                continue
            if sd[k].shape == own[k].shape:
                own[k].copy_(sd[k])
                n += 1
            elif k == "s_head.5.weight" and sd[k].shape[0] == 1 and own[k].shape[0] == J:
                own[k].copy_(sd[k].repeat(J, 1))
                n += 1
            elif k == "s_head.5.bias" and sd[k].shape[0] == 1 and own[k].shape[0] == J:
                own[k].copy_(sd[k].repeat(J) - math.log(J))
                n += 1
            elif k == "s_fast.2.weight" and sd[k].shape[0] == 1 and own[k].shape[0] == J:
                own[k].copy_(sd[k].repeat(J, 1))
                n += 1
            elif k == "s_fast.2.bias" and sd[k].shape[0] == 1 and own[k].shape[0] == J:
                own[k].copy_(sd[k].repeat(J) - math.log(J))
                n += 1
        self.load_state_dict(own)
        tau = self.i_lag.tau_ms()
        print(f"[*] v11: 从 v10 热启动, 载入 {n} 个参数张量 "
              f"(s 基头 1/{J} 复制); 电流延迟 {tau:.1f}ms; σ={self.sigmas}", flush=True)
        return ckpt.get("norm_cfg")
