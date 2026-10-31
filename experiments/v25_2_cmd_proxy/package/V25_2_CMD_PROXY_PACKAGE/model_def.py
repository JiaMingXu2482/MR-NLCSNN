"""V25 residual inverse model.

Identical backbone to V24-I0 / E0 (GRU 2x8, skip concat 8+31=39, MLP16, 2073 param);
the ONLY change is that the head is linear (no sigmoid) and predicts a current
increment relative to the previous measured 100 Hz current.

    I_raw[k] = I_base[k] + delta_pred[k]
    I_cmd[k] = clip(I_raw[k], 0, 1.6)
"""
import torch
import torch.nn as nn


class ResidualI0(nn.Module):
    def __init__(self, d_in=31, hidden=8, layers=2, mlp=16):
        super().__init__()
        self.d_in, self.hidden, self.layers = d_in, hidden, layers
        self.gru = nn.GRU(d_in, hidden, layers, batch_first=True)
        self.trunk = nn.Sequential(nn.Linear(hidden + d_in, mlp), nn.ReLU())
        self.head = nn.Linear(mlp, 1)
        # zero head => delta == 0 at init => the model starts exactly at Persistence,
        # which is the reference it must beat.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """x: (B,T,31) -> delta_I in amps (B,T).  Linear, unbounded, can be negative."""
        h, _ = self.gru(x)
        z = self.trunk(torch.cat((h, x), -1))
        return self.head(z).squeeze(-1)

    # ---- single-step API for 100 Hz latency measurement -------------------
    def init_state(self, batch=1, device=None):
        dev = device or next(self.parameters()).device
        return torch.zeros(self.layers, batch, self.hidden, device=dev)

    @torch.no_grad()
    def step(self, x_t, state):
        h, st = self.gru(x_t.view(1, 1, -1), state)
        z = self.trunk(torch.cat((h[:, 0], x_t.view(1, -1)), -1))
        return self.head(z).squeeze(), st


def count_params(model):
    return dict(total=sum(p.numel() for p in model.parameters() if p.requires_grad),
                groups={n.split(".")[0]: sum(p.numel() for p in m.parameters())
                        for n, m in model.named_children()})


if __name__ == "__main__":
    print(count_params(ResidualI0()))
