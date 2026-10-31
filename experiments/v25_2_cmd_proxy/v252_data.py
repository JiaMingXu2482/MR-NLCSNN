"""V25.2 data layer: direct [::10] decimation (NO filter) + command-proxy features.

Timing (one control-step response assumption I_act[k] ~= I_cmd[k-1]):
    decision index k          -> 100 Hz sample k (1 kHz index 10k)
    base_proxy[k]   = clip(|I_meas[k]|,   0, 1.6)   A   (previous ECU command)
    target_proxy[k] = clip(|I_meas[k+1]|, 0, 1.6)   A   (current command, LABEL only)
    delta[k]        = target_proxy[k] - base_proxy[k]
    I_cmd[k]        = clip(base_proxy[k] + delta_hat[k], 0, 1.6)

Current-history features (all of them) take the SAME proxy:
    I_cmd[k-j] ~= clip(|I_meas[k-j+1]|, 0, 1.6) / 1.6
so in the feature builder the current input is `proxy/1.6` in [0, 1].
"""
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:/v25_2_cmd_proxy")
from v23_data import cond_inv, motion_feats                         # noqa: E402

I_SCALE = 1.6
GUARD = 10          # 100 ms at each end (kept from V25 for metric comparability)


def decimate_direct(v):
    """(N,C) 1 kHz -> (ceil(N/10),C) 100 Hz, index 0 phase, NO filtering."""
    return np.asarray(v)[::10]


def build_v252(item, norm):
    """-> x (1,T,31), base_A (1,T), target_A (1,T), mask (1,T)"""
    v = item["values"].numpy().astype(np.float64)
    vd = decimate_direct(v)                                  # <-- no FIR / no IIR
    x_ = (vd[:, 0] - norm["x_ref"]) / norm["x_scale"]
    vel = vd[:, 1] / norm["v_scale"]
    acc = vd[:, 2] / norm["a_scale"]
    cur = vd[:, 3] / norm["i_scale"]                          # signed, raw sample
    tmp = (vd[:, 4] - norm["t_ref"]) / norm["t_scale"]
    fce = vd[:, 5] / norm["f_scale"]

    proxy = np.clip(np.abs(cur), 0.0, 1.0)                    # clip(|I|/1.6,0,1)
    m = motion_feats(x_, vel, acc)                            # 20 motion features
    c = cond_inv(fce, proxy, x_, vel, acc)                    # 11 condition, ip -> proxy
    N = len(proxy)

    base = proxy * I_SCALE                                    # A, at index k
    tgt = np.concatenate([proxy[1:], [np.nan]]) * I_SCALE     # A, from index k+1
    k = np.arange(N)
    mask = (k >= GUARD) & (k <= N - 1 - GUARD)                # also drops the last sample

    tgt = np.nan_to_num(tgt, nan=0.0)
    return (torch.from_numpy(np.concatenate([m, c], 1)).float()[None],
            torch.from_numpy(base).float()[None],
            torch.from_numpy(tgt).float()[None],
            torch.from_numpy(mask)[None])
