"""NLCSNN v7 — PDIA 思想的解耦耗散模型 (正/逆统一)。

用户硬要求 (PMPC 闭环稳定性):
  必须在任意力、任意速度下保证 Fd × vel_damper ≥ 0, 且在零速度附近连续。

结构 (受 PDIA 论文 s11071-026-13039-w 启发, 但收紧到用户口径):
  F = c_base(h,nfl,v)·v_out  +  s_mod(h,nfl,v)·tanh(k·v_out)·G(I)

  - c_base ≥ 0 (softplus): 与电流无关的机械阻尼基座 (黏性/密封/气室)
  - s_mod ≥ 0 (softplus): 电流调制幅度
  - tanh(k·v_out): 硬换号项。与 v 同号、零速连续过零。
    这是论文 tanh-Shape 思想的硬化版: 论文只靠"网络自己学"换号,
    v7 从结构上锁死 S·v ≥ 0 (S = s_mod·tanh(k·v_out))。
  - G(I): 单调 MLP (PositiveLinear + tanh + skip), G(0)=0, dG/dI>0,
    并归一化钉死 G(1.6A)=1。逆解唯一, 可二分求电流。
    CDC 常闭阀 (电流越大越软): 实际用 I_e = 1 - I_norm 输入 G,
    G(I_e) 递增 => 物理上随 I 递减。I=0 -> G=1 (最硬); I=1.6A -> G=0 (最软)。

耗散保证 (架构, 非 loss 惩罚):
  F·v = c_base·(v_out·v) + s_mod·G·(tanh(k·v_out)·v)
  - v_out = S_v·v_norm 与 v 同号 (S_v>0)  =>  v_out·v ≥ 0
  - tanh(k·v_out) 与 v_out 同号  =>  tanh(k·v_out)·v ≥ 0
  - c_base,s_mod,G ≥ 0
  => F·v ≥ 0 恒成立, 对任意 v (含 OOD) 成立。
  v=0 时: v_out=0 => B=0, tanh(0)=0 => S=0 => F=0, 且全光滑 => 零速连续。

相比 v6 (F = c·v, 单一阻尼系数):
  - 解耦"机械基座"与"电流调制", 优化目标不再纠缠 (v6 难训的主因)。
  - G(I) 是一维单调函数, 极强的归纳偏置, 几乎不可能学偏。
  - 附带原理化逆模型: F = B + S·G(I) => G(I) = (F-B)·S/(S²+ε),
    I = G⁻¹(·) 二分唯一解。PMPC 力→电流不再需要对着整个网络二分。

相比论文 PDIA:
  - 论文 Bias 无约束 (允许 F·v<0, 理由是气室弹性可做正功);
    v7 把 Bias 收紧为 c_base·v_out, 满足用户"总力恒耗散"硬要求。
    数学上: 连续函数若处处 F·v≥0 则必有 F(0)=0, 故任何静态偏置
    都与用户口径不兼容, v7 与 v6 一样 F(0)≡0。
  - 论文逆解用双 observer; v7 逆解复用同一个前向网络
    (c_base/s_mod 用 I_t 掩蔽历史计算, 保证 F 对 I_t 严格单调)。

尺度钉死 (重要):
  s_mod·G(I) 存在尺度歧义 (s 翻倍、G 减半则前向不变, 但逆解 G⁻¹ 会变)。
  用 G(I) = G_raw(I)/G_raw(1.6A).detach() 强制 G(1.6A)=1,
  使 s_mod 独自承担幅值, 分解唯一, 逆解 well-defined。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_v3 import static_nfl, STATIC_NFL_DIM


class PositiveLinear(nn.Module):
    """权重经 softplus 映射为严格正 => 层输出对输入严格单调递增。

    初始化 w_raw=-1 (softplus≈0.31, 小正权重): 使 f_MLP 初值平缓近线性,
    避免 G_raw(I_ref) 过大导致归一化后 dG/dI 过小、梯度消失。
    """
    def __init__(self, in_f, out_f):
        super().__init__()
        self.w_raw = nn.Parameter(torch.full((out_f, in_f), -1.0) +
                                  torch.randn(out_f, in_f) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        w_pos = F.softplus(self.w_raw)  # > 0, 严格
        return F.linear(x, w_pos, self.bias)


class MonotonicG(nn.Module):
    """电磁核心 G(I): 严格单调递增, G(0)=0, G(I_ref)=1。

    G_raw(I) = [f_MLP(I) - f_MLP(0)] + k_skip·I
      - f_MLP: PositiveLinear + tanh, 权重全正 => df/dI > 0 (链式法则, 见论文 §3.3.2)
      - tanh 二阶导 <0 模拟磁饱和; k_skip>0 给梯度下界 => 严格单调, 逆解唯一
      - f(I)-f(0) 相减消除零漂 => G(0)=0 (零基线, 物理约束 II)
    G(I) = G_raw(I) / G_raw(I_ref).detach()  =>  G(I_ref) ≡ 1, 尺度钉死。
    """
    def __init__(self, hidden=64, i_ref=1.0, k_skip=0.1):
        super().__init__()
        self.i_ref = i_ref
        self.k_skip = k_skip
        self.mlp = nn.Sequential(
            PositiveLinear(1, hidden), nn.Tanh(),
            PositiveLinear(hidden, hidden), nn.Tanh(),
            PositiveLinear(hidden, 1),
        )

    def _g_raw(self, I, f_0):
        return (self.mlp(I) - f_0) + self.k_skip * I

    def forward(self, I):
        """I: (..., 1) 归一化电流. 返回 (..., 1), I≥0 时 ≥0 且单调增。"""
        dev, dt = I.device, I.dtype
        f_0 = self.mlp(torch.zeros(1, 1, device=dev, dtype=dt))
        g_raw = self._g_raw(I, f_0)
        i_ref_t = torch.full((1, 1), self.i_ref, device=dev, dtype=dt)
        g_ref = self._g_raw(i_ref_t, f_0)  # (1,1)
        # clamp_min(0): 消除 f_0 单独计算引入的浮点误差 (约 -1e-5 量级),
        # 理论上 G(I≥0)≥0, 钳制只影响数值噪声, 不改变单调性。
        return (g_raw / (g_ref.detach() + 1e-8)).clamp_min(0.0)

    @torch.no_grad()
    def inverse(self, y_target, iters=40):
        """y_target: (...) ∈ [0,1], 返回 I ∈ [0,1] (归一化)。单调二分, 唯一解。

        y<0 => I=0 (饱和下限); y>1 => I=1 (饱和上限, 即 1.6A)。
        """
        y = y_target.clamp(0.0, 1.0).unsqueeze(-1)  # (..., 1)
        lo = torch.zeros_like(y)
        hi = torch.ones_like(y)
        for _ in range(iters):
            mid = (lo + hi) / 2
            g_mid = self.forward(mid)
            go_up = g_mid < y  # G 单调增: 输出偏小 => 电流偏小
            lo = torch.where(go_up, mid, lo)
            hi = torch.where(go_up, hi, mid)
        return ((lo + hi) / 2).squeeze(-1)

    @torch.no_grad()
    def check_monotonic(self, n=321):
        """数值验证单调性与非负性。返回 (min_dG_dI, min_G)。"""
        I = torch.linspace(0, 1, n).unsqueeze(-1)
        G = self.forward(I).squeeze(-1)
        dG = G[1:] - G[:-1]
        return float(dG.min()), float(G.min())


def _make_head(in_dim, mlp_hidden, dropout):
    head = nn.Sequential(
        nn.Linear(in_dim, mlp_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden, mlp_hidden // 2),
        nn.ReLU(),
        nn.Linear(mlp_hidden // 2, 1),
    )
    nn.init.zeros_(head[-1].weight)
    # 保守初值: softplus(-4)≈0.018, 避免初始输出比真值大一个量级
    nn.init.constant_(head[-1].bias, -4.0)
    return head


class NLCSNN_v7(nn.Module):
    def __init__(self, gru_hidden=64, gru_layers=2, mlp_hidden=128,
                 dropout=0.0, c_eps=1e-6, v_out_scale=30.0,
                 tanh_k=2.0, g_hidden=64, g_skip=0.1, inv_eps=1e-4):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.c_eps = c_eps
        self.v_out_scale = v_out_scale  # 输出乘法速度放大 (v6 已验证, 改善条件数)
        self.tanh_k = tanh_k            # 换号项锐度 (固定, >0 即可保 S·v≥0)
        self.inv_eps = inv_eps          # 鲁棒除法稳定因子 (论文 ε=1e-4)
        self.gru = nn.GRU(STATIC_NFL_DIM, gru_hidden, gru_layers,
                          batch_first=True, dropout=dropout if gru_layers > 1 else 0.0)
        feat_dim = gru_hidden + STATIC_NFL_DIM + 1  # [h, nfl, v_out]
        self.c_base_head = _make_head(feat_dim, mlp_hidden, dropout)
        self.s_mod_head = _make_head(feat_dim, mlp_hidden, dropout)
        # G 输入为归一化电流 I/1.6 ∈ [0,1], i_ref=1.0 即 1.6A 处钉死 G=1
        self.G = MonotonicG(hidden=g_hidden, i_ref=1.0, k_skip=g_skip)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[*] NLCSNN_v7: NFL={STATIC_NFL_DIM} -> GRU({gru_hidden}x{gru_layers}) "
              f"-> F=c_base·v + s_mod·tanh({tanh_k}·v)·G(I) "
              f"[dissipative+invertible] | {n_params/1e6:.2f}M params", flush=True)

    def forward_components(self, u):
        """返回 (F, B, S, G_I, c_base, s_mod), 均为 (B, L) 归一化单位。

        F = B + S·G_I, 其中 B = c_base·v_out (基座), S = s_mod·tanh(k·v_out) (调制)。
        """
        nfl = static_nfl(u)                       # (B, L, 35)
        h_seq, _ = self.gru(nfl, None)             # (B, L, H)
        v_out = u[:, :, 1:2] * self.v_out_scale   # (B, L, 1)
        # 电流取幅值: MR 效应只与 |I| 有关; 负值是传感器接线符号问题 (163/301 文件为负)。
        # GRU/nfl 仍用原始符号 (保 v4 warm-start 分布一致), 只有 G(I) 用 |I| (单调性要求 I≥0)。
        # CDC 常闭阀: 电流越大越软 => 用 I_e = 1 - I_norm (I_max=1.6A 对应 norm=1),
        # G(I_e) 递增 => 物理上 G 随 I 递减。I=0 -> I_e=1 -> G=1 (最硬); I=1.6A -> I_e=0 -> G=0 (最软)。
        I_norm = u[:, :, 3:4].abs().clamp(0.0, 1.0)  # (B, L, 1) 归一化电流幅值 [0,1]
        I_e = 1.0 - I_norm                            # 等效电流: 递增 G => 递减物理响应
        feat = torch.cat([h_seq, nfl, v_out], dim=-1)
        c_base = F.softplus(self.c_base_head(feat)).squeeze(-1) + self.c_eps  # ≥0
        s_mod = F.softplus(self.s_mod_head(feat)).squeeze(-1) + self.c_eps    # ≥0
        S = s_mod * torch.tanh(self.tanh_k * v_out.squeeze(-1))  # 可正负, 但 S·v≥0
        G_I = self.G(I_e).squeeze(-1)             # ≥0, 随 I_e 递增 => 随物理 I 递减
        B = c_base * v_out.squeeze(-1)
        return B + S * G_I, B, S, G_I, c_base, s_mod

    def forward(self, u):
        """u: (B, L, 6) -> (B, L) 归一化力。恒耗散、零速连续、可逆。"""
        f, _, _, _, _, _ = self.forward_components(u)
        return f

    def inverse_latent(self, F_true, B, S):
        """鲁棒除法求目标隐变量 y_target = (F-B)·S/(S²+ε) (论文式 24, 可微)。"""
        return (F_true - B) * S / (S * S + self.inv_eps)

    @torch.no_grad()
    def inverse_current(self, u_hist_masked, F_target):
        """逆解: 给定历史 (I_t 已掩蔽为 0) 与目标力 (归一化), 求 I_t (归一化 [0,1])。

        用掩蔽历史算 B,S (使它们不依赖待求的 I_t, 保证 F 对 I_t 严格单调),
        再经鲁棒除法 + G 二分逆解。输出钳在 [0,1] (0~1.6A)。
        """
        self.eval()
        _, B, S, _, _, _ = self.forward_components(u_hist_masked)
        y_target = self.inverse_latent(F_target, B[:, -1], S[:, -1])
        I_e = self.G.inverse(y_target)  # 二分得 I_e ∈ [0,1] (G 递增空间)
        return 1.0 - I_e                # 换回物理归一化电流 I_norm ∈ [0,1]

    @torch.no_grad()
    def predict_file(self, u):
        self.eval()
        p = self.forward(u.unsqueeze(0)).squeeze(0).cpu().numpy()
        self.train()
        return p

    def load_gru_from_v4(self, v4_ckpt_path):
        """从 v4 最优 checkpoint 热启动 GRU (NFL 相同; 读出头重新训练)。"""
        ckpt = torch.load(v4_ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        own = self.state_dict()
        n = 0
        for k in list(own.keys()):
            if k.startswith("gru.") and k in sd and sd[k].shape == own[k].shape:
                own[k].copy_(sd[k])
                n += 1
        self.load_state_dict(own)
        print(f"[*] v7: 从 v4 热启动 GRU, 载入 {n} 个参数张量", flush=True)
        return ckpt.get("norm_cfg")
