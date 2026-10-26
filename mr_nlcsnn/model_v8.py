"""NLCSNN v8 — v4 骨架 + 解耦耗散 + 正逆联合训练。

用户要求: 回到原来的 NLCSNN (v4: GRU + MLP 读出, E30 pooled 33.9%),
做改进: (1) 正逆模型同时训练; (2) 保证耗散性。

结构 (v7 的改进版):
  F = c(z)·v_out·G_c(I_e)  +  s(z)·tanh(k·v_out)·G_s(I_e)

  - z = [h_gru, nfl, |v_out|]: v4 式 MLP 读出特征
  - c(z), s(z) ≥ 0 (softplus): 黏性系数 / 库伦摩擦幅值, v4 式 MLP 头
  - G_c, G_s: 两个独立 MonotonicG (v7 的单调 MLP), 都对 I_e 递增
  - I_e = 1 - |I_norm|: CDC 常闭阀, 电流越大越软

相比 v7 (F = c_base·v + s_mod·tanh(k·v)·G(I)):
  - v7 的电流只调制摩擦项, 黏性基座与电流无关 (物理上不真实:
    MR 效应同时改变黏性和屈服应力)
  - v8 用两个独立 G_c/G_s, 电流同时调制两项, 各自灵敏度不同
  - 仍保证耗散 + 可逆 (见下)

耗散保证 (架构):
  F·v = c·G_c·(v_out·v) + s·G_s·(tanh(k·v_out)·v)
  - v_out = S_v·v_norm 与 v 同号 => v_out·v ≥ 0
  - tanh(k·v_out) 与 v 同号 => tanh(k·v_out)·v ≥ 0
  - c,s,G_c,G_s ≥ 0
  => F·v ≥ 0 恒成立。v=0 时 F=0, 全光滑, 零速连续。

可逆性:
  固定 (c,s,v), f(I_e) = c·v·G_c(I_e) + s·tanh(k·v)·G_s(I_e)
  G_c,G_s 都对 I_e 严格递增, c·v 与 s·tanh(k·v) 同号 (都与 v 同号),
  => f 对 I_e 单调 (v>0 递增, v<0 递减, v=0 恒为 0)。
  二分可唯一求解 (v=0 时物理上电流不影响力, 返回任意值无害)。

逆模型 (论文 PDIA 双 observer 思想):
  独立逆头 I_pred = sigmoid(MLP_inv([h, nfl, v_norm, F_target])) ∈ [0,1],
  与前向联合训练: loss = loss_fwd + λ·loss_inv。
  部署时可用逆头 (快) 或对前向二分 (严格一致)。

尺度钉死: G_c(1.6A)=G_s(1.6A)=1 (与 v7 相同, detach 归一化),
使 c/s 独自承担幅值, 分解唯一。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_v3 import static_nfl, STATIC_NFL_DIM
from .model_v7 import PositiveLinear, MonotonicG


def _make_head(in_dim, mlp_hidden, dropout):
    """v4 式 MLP 头: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Linear。"""
    head = nn.Sequential(
        nn.Linear(in_dim, mlp_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden, mlp_hidden // 2),
        nn.ReLU(),
        nn.Linear(mlp_hidden // 2, 1),
    )
    nn.init.zeros_(head[-1].weight)
    # 保守初值: softplus(-4)≈0.018 (v7 已验证)
    nn.init.constant_(head[-1].bias, -4.0)
    return head


def _make_inv_head(in_dim, mlp_hidden, dropout):
    """逆头: [h, nfl, v_norm, F_target] -> I_norm ∈ [0,1]。"""
    head = nn.Sequential(
        nn.Linear(in_dim, mlp_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden, mlp_hidden // 2),
        nn.ReLU(),
        nn.Linear(mlp_hidden // 2, 1),
        nn.Sigmoid(),
    )
    return head


class NLCSNN_v8(nn.Module):
    def __init__(self, gru_hidden=64, gru_layers=2, mlp_hidden=128,
                 dropout=0.0, c_eps=1e-6, v_out_scale=30.0,
                 tanh_k=2.0, g_hidden=64, g_skip=0.1):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.c_eps = c_eps
        self.v_out_scale = v_out_scale
        self.tanh_k = tanh_k
        self.gru = nn.GRU(STATIC_NFL_DIM, gru_hidden, gru_layers,
                          batch_first=True, dropout=dropout if gru_layers > 1 else 0.0)
        # 前向头特征: [h, nfl, |v_out|] (v4 式, 加 |v| 显式幅值)
        feat_dim = gru_hidden + STATIC_NFL_DIM + 1
        self.c_head = _make_head(feat_dim, mlp_hidden, dropout)
        self.s_head = _make_head(feat_dim, mlp_hidden, dropout)
        # 两个独立单调 G: 黏性项 / 摩擦项各自电流灵敏度
        self.G_c = MonotonicG(hidden=g_hidden, i_ref=1.0, k_skip=g_skip)
        self.G_s = MonotonicG(hidden=g_hidden, i_ref=1.0, k_skip=g_skip)
        # 逆头特征: [h, nfl, v_norm, F_target_norm]
        inv_dim = gru_hidden + STATIC_NFL_DIM + 2
        self.inv_head = _make_inv_head(inv_dim, mlp_hidden, dropout)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[*] NLCSNN_v8: NFL={STATIC_NFL_DIM} -> GRU({gru_hidden}x{gru_layers}) "
              f"-> F=c·v·G_c(I_e)+s·tanh({tanh_k}·v)·G_s(I_e) "
              f"[v4骨架+解耦耗散+正逆联合] | {n_params/1e6:.2f}M params", flush=True)

    def forward_features(self, u):
        """返回 (h_seq, nfl, v_out)。u: (B, L, 6)。"""
        nfl = static_nfl(u)                       # (B, L, 35)
        h_seq, _ = self.gru(nfl, None)             # (B, L, H)
        v_out = u[:, :, 1:2] * self.v_out_scale   # (B, L, 1)
        return h_seq, nfl, v_out

    def forward_components(self, u):
        """返回 (F, c, s, Gc_I, Gs_I), 均为 (B, L) 归一化单位。

        F = c·v_out·G_c(I_e) + s·tanh(k·v_out)·G_s(I_e)
        """
        h_seq, nfl, v_out = self.forward_features(u)
        v = v_out.squeeze(-1)                     # (B, L)
        # 电流取幅值 + CDC 方向 (与 v7 一致)
        I_norm = u[:, :, 3].abs().clamp(0.0, 1.0)  # (B, L)
        I_e = 1.0 - I_norm
        feat = torch.cat([h_seq, nfl, v.abs().unsqueeze(-1)], dim=-1)
        c = F.softplus(self.c_head(feat)).squeeze(-1) + self.c_eps  # ≥0
        s = F.softplus(self.s_head(feat)).squeeze(-1) + self.c_eps  # ≥0
        Gc_I = self.G_c(I_e.unsqueeze(-1)).squeeze(-1)  # ≥0, 随 I_e 递增
        Gs_I = self.G_s(I_e.unsqueeze(-1)).squeeze(-1)  # ≥0, 随 I_e 递增
        tv = torch.tanh(self.tanh_k * v)          # 与 v 同号
        Force = c * v * Gc_I + s * tv * Gs_I
        return Force, c, s, Gc_I, Gs_I

    def forward(self, u):
        """u: (B, L, 6) -> (B, L) 归一化力。恒耗散、零速连续、可逆。"""
        f, _, _, _, _ = self.forward_components(u)
        return f

    def inverse_predict(self, u_hist, F_target):
        """逆头快速预测: u_hist (B,L,6, I可为任意值/掩蔽), F_target (B,L) 归一化。
        返回 (B,L) I_norm ∈ [0,1]。
        """
        h_seq, nfl, _ = self.forward_features(u_hist)
        v_norm = u_hist[:, :, 1:2]                # (B, L, 1) 归一化速度
        feat = torch.cat([h_seq, nfl, v_norm, F_target.unsqueeze(-1)], dim=-1)
        return self.inv_head(feat).squeeze(-1)    # (B, L) sigmoid ∈ [0,1]

    @torch.no_grad()
    def inverse_current(self, u_hist_masked, F_target):
        """二分逆解 (严格一致版): 给定掩蔽历史与目标力, 求 I_t ∈ [0,1]。

        f(I_e) = c·v·G_c(I_e) + s·tanh(k·v)·G_s(I_e) 对 I_e 单调
        (v>0 递增, v<0 递减), 二分唯一解。v≈0 时 f 平坦, 返回值任意 (物理上无影响)。
        u_hist_masked: (B, L, 6), I_t 已掩蔽; F_target: (B,) 末步目标力 (归一化)。
        """
        self.eval()
        h_seq, nfl, v_out = self.forward_features(u_hist_masked)
        h_last, nfl_last = h_seq[:, -1], nfl[:, -1]       # (B, H), (B, 35)
        v = v_out[:, -1, 0]                                # (B,)
        feat = torch.cat([h_last, nfl_last, v.abs().unsqueeze(-1)], dim=-1)
        c = F.softplus(self.c_head(feat)).squeeze(-1) + self.c_eps
        s = F.softplus(self.s_head(feat)).squeeze(-1) + self.c_eps
        tv = torch.tanh(self.tanh_k * v)

        def f_of_Ie(Ie):
            Ie1 = Ie.unsqueeze(-1)
            return c * v * self.G_c(Ie1).squeeze(-1) + s * tv * self.G_s(Ie1).squeeze(-1)

        f0 = f_of_Ie(torch.zeros_like(v))
        f1 = f_of_Ie(torch.ones_like(v))
        increasing = f1 > f0                               # (B,) v>0 时 True
        lo = torch.zeros_like(v)
        hi = torch.ones_like(v)
        for _ in range(40):
            mid = (lo + hi) / 2
            fm = f_of_Ie(mid)
            go_up = torch.where(increasing, fm < F_target, fm > F_target)
            lo = torch.where(go_up, mid, lo)
            hi = torch.where(go_up, hi, mid)
        I_e = (lo + hi) / 2
        return (1.0 - I_e).clamp(0.0, 1.0)                # 物理归一化电流

    @torch.no_grad()
    def predict_file(self, u):
        self.eval()
        p = self.forward(u.unsqueeze(0)).squeeze(0).cpu().numpy()
        self.train()
        return p

    def load_gru_from_v4(self, v4_ckpt_path):
        """从 v4 最优 checkpoint 热启动 GRU (与 v7 相同逻辑)。"""
        ckpt = torch.load(v4_ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        own = self.state_dict()
        n = 0
        for k in list(own.keys()):
            if k.startswith("gru.") and k in sd and sd[k].shape == own[k].shape:
                own[k].copy_(sd[k])
                n += 1
        self.load_state_dict(own)
        print(f"[*] v8: 从 v4 热启动 GRU, 载入 {n} 个参数张量", flush=True)
        return ckpt.get("norm_cfg")
