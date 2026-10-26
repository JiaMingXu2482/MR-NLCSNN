"""NLCSNN v3 — physics-NFL + causal TCN (with LayerNorm for stability).

v2 的 RK4 循环在 CPU 上太慢 (561 step.file/s, 单轮 151 分钟), 无法做超参数调优。
v3 保留经过验证的物理 NFL 特征库, 把循环隐状态换成因果时序卷积 (TCN):

  - 时间维度完全并行, 训练加速约 450 倍, 可以真正调超参数;
  - 因果 (只看过去), 可实时部署, 与控制器 1kHz 部署一致;
  - 感受野 1023 点 (~1s @1kHz), 覆盖最慢 1Hz 工况的半个周期以上。
  - LayerNorm 稳定深层残差卷积 (v3.1 修复: 无归一化时训练爆炸)。

输入: 35 维静态 NFL (v2 的 75 维去掉 h 相关项) + 时间维。
输出: 每时刻力 (归一化)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def static_nfl(u):
    """向量化静态 NFL (无 h 项)。u: (..., 6) -> (..., 35)。"""
    x = u[..., 0:1]; v = u[..., 1:2]; a = u[..., 2:3]
    i = u[..., 3:4]; di = u[..., 4:5]; t = u[..., 5:6]
    v_abs = torch.abs(v)
    v_pos = torch.relu(v)
    v_neg = -torch.relu(-v)
    sgn_v = torch.tanh(50.0 * v)

    kin = torch.cat([v, x * v, v * i, v * t], dim=-1)                       # 4
    rebound = torch.cat([torch.tanh(2.0 * v_pos),
                         torch.tanh(50.0 * v_pos),
                         i * torch.tanh(5.0 * v_pos)], dim=-1)              # 3
    compression = torch.cat([v_neg * v_abs, v_neg ** 3,
                             a * v_neg, i * v_neg * v_abs], dim=-1)        # 4
    curr_dyn = torch.cat([di * v, di * a, torch.abs(di) * i], dim=-1)      # 3
    aux = torch.cat([sgn_v, i * sgn_v, x ** 2, x ** 3, a * i], dim=-1)     # 5
    hf_dyn = torch.cat([torch.abs(a), a * v,
                        torch.abs(a) * sgn_v, i * torch.abs(a)], dim=-1)   # 4
    x_abs = torch.abs(x)
    slow_gate = torch.exp(-torch.clamp(80.0 * v_abs, max=30.0))
    slow = torch.cat([slow_gate, x * slow_gate, x_abs * slow_gate,
                      i * x * slow_gate, i * x_abs * slow_gate,
                      sgn_v * x_abs], dim=-1)                              # 6
    return torch.cat([u, kin, rebound, compression, curr_dyn,
                      aux, hf_dyn, slow], dim=-1)                          # 35


STATIC_NFL_DIM = 35


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    """WaveNet 风格门控残差块:
    filter=tanh(conv_f(x)), gate=sigmoid(conv_g(x)), z=filter*gate;
    z -> 1x1 conv -> (+残差). 门控提供乘性交互, 适合学状态相关的迟滞物理。
    (v3.2: 从 ReLU 残差块升级, 改善泛化)"""
    def __init__(self, ch, kernel, dilation, dropout=0.0):
        super().__init__()
        self.cf = CausalConv1d(ch, ch, kernel, dilation)
        self.cg = CausalConv1d(ch, ch, kernel, dilation)
        self.ln = nn.LayerNorm(ch)
        self.proj = nn.Conv1d(ch, ch, 1)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.proj.bias)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: (B, ch, L)
        f = self.cf(x).transpose(1, 2)         # (B, L, ch)
        g = self.cg(x).transpose(1, 2)
        z = torch.tanh(f) * torch.sigmoid(g)   # 门控乘性交互
        z = self.drop(z).transpose(1, 2)       # (B, ch, L)
        z = self.proj(z)
        z = self.ln(z.transpose(1, 2)).transpose(1, 2)
        return x + z                           # 纯残差 (无末端 ReLU, 保信息)


class NLCSNN_v3(nn.Module):
    def __init__(self, channels=64, levels=9, kernel=3, dropout=0.0):
        super().__init__()
        self.receptive_field = 1 + 2 * (kernel - 1) * sum(2 ** i for i in range(levels))
        self.in_proj = CausalConv1d(STATIC_NFL_DIM, channels, 1, 1)
        self.in_ln = nn.LayerNorm(channels)
        self.blocks = nn.Sequential(*[
            TCNBlock(channels, kernel, 2 ** i, dropout) for i in range(levels)
        ])
        self.out_ln = nn.LayerNorm(channels)
        self.out_proj = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        print(f"[*] NLCSNN_v3(gated): staticNFL={STATIC_NFL_DIM} -> ch={channels} "
              f"x{levels} (k={kernel}) -> RF={self.receptive_field}", flush=True)

    def forward(self, u):
        """u: (B, L, 6) 归一化输入 -> (B, L) 归一化力。"""
        nfl = static_nfl(u)                       # (B, L, 35)
        x = nfl.transpose(1, 2)                   # (B, 35, L)
        x = torch.relu(self.in_ln(self.in_proj(x).transpose(1, 2)).transpose(1, 2))
        x = self.blocks(x)
        x = self.out_ln(x.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(x).squeeze(1)        # (B, L)

    @torch.no_grad()
    def predict_file(self, u):
        self.eval()
        p = self.forward(u.unsqueeze(0)).squeeze(0).cpu().numpy()
        self.train()
        return p
