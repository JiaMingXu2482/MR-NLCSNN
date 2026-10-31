"""V23-B data layer: reuses the V20/V22 reader, adds the audited 1 kHz -> 100 Hz
path and the two task adapters.

Decimation is NOT naive every-10th-sample subsampling: a 61-tap causal
windowed-sinc low-pass (40 Hz) runs first.  The filter is linear phase, so every
channel is delayed by the same (61-1)/2 = 30 ms group delay; input and label
shift together, so alignment is preserved.
"""
import numpy as np
import torch

from v20_data import digest, edge_guard, normalization, read_dataset   # noqa: F401

FS = 1000.0
DECIM = 10
TAPS = 61
CUTOFF = 40.0
GUARD_100 = 10          # 100 ms dropped at each end of a 100 Hz sequence


def fir_lowpass(taps=TAPS, cutoff=CUTOFF, fs=FS):
    n = np.arange(taps) - (taps - 1) / 2.0
    h = np.sinc(2.0 * cutoff / fs * n) * np.hamming(taps)
    return h / h.sum()


H = fir_lowpass()


def decimate(v):
    """(N, C) 1 kHz -> (ceil(N/10), C) 100 Hz, causal linear-phase anti-alias."""
    lp = np.stack([np.convolve(v[:, c], H, mode="full")[:v.shape[0]]
                   for c in range(v.shape[1])], 1)
    return lp[::DECIM]


def norm_channels(v, norm):
    """raw (N, 6) -> normalised [x, v, a, I_norm_signed, T] + force."""
    x = (v[:, 0] - norm["x_ref"]) / norm["x_scale"]
    vel = v[:, 1] / norm["v_scale"]
    acc = v[:, 2] / norm["a_scale"]
    cur = v[:, 3] / norm["i_scale"]                      # SIGNED, no clip
    tmp = (v[:, 4] - norm["t_ref"]) / norm["t_scale"]
    fce = v[:, 5] / norm["f_scale"]
    return x, vel, acc, cur, tmp, fce


# ---------------------------------------------------------------- adapters
def build_fwd(item, norm):
    """1 kHz forward task.  Returns motion(20), cond(11), y, valid."""
    v = item["values"].numpy().astype(np.float64)
    x, vel, acc, cur, tmp, fce = norm_channels(v, norm)
    iprev = np.concatenate(([0.0], cur[:-1]))            # I[t-1] strictly causal
    m = motion_feats(x, vel, acc)
    c = cond_fwd(iprev, tmp, x, vel, acc)
    n = len(x)
    valid = np.ones(n, dtype=bool)
    g = edge_guard(n)
    if g:
        valid[:g] = False
        valid[-g:] = False
    return (torch.from_numpy(m).float()[None], torch.from_numpy(c).float()[None],
            torch.from_numpy(fce).float()[None], torch.from_numpy(valid)[None])


def build_inv(item, norm):
    """100 Hz inverse task.  Label is SIGNED I/1.6, never clipped.

    f_des[k]  = measured force at control instant k (teacher forced)
    i_past[k] = measured current at control instant k-1  (I[k-1] only)
    """
    v = item["values"].numpy().astype(np.float64)
    vd = decimate(v)
    x, vel, acc, cur, _tmp, fce = norm_channels(vd, norm)
    ipast = np.concatenate(([0.0], cur[:-1]))
    m = motion_feats(x, vel, acc)
    c = cond_inv(fce, ipast, x, vel, acc)
    n = len(x)
    valid = np.ones(n, dtype=bool)
    g = min(GUARD_100, n // 10)
    if g:
        valid[:g] = False
        valid[-g:] = False
    return (torch.from_numpy(m).float()[None], torch.from_numpy(c).float()[None],
            torch.from_numpy(cur).float()[None], torch.from_numpy(valid)[None],
            torch.from_numpy(vd[:, 3]).float())


# ---------------------------------------------------------------- features
def motion_feats(x, v, a):
    """20 motion-only features, original NFL31 order preserved."""
    av = np.abs(v)
    vp = np.maximum(v, 0.0)
    vn = -np.minimum(v, 0.0)
    sign = np.tanh(v / 0.1)
    slow = np.exp(-av / 0.1)
    return np.stack([x, v, a, v, x * v,
                     np.tanh(2 * vp), np.tanh(vp / 0.1),
                     vn * av, vn ** 3, a * vn,
                     sign, x ** 2, x ** 3,
                     np.abs(a), a * v, np.abs(a) * sign,
                     slow, x * slow, np.abs(x) * slow,
                     sign * np.abs(x)], 1)


def cond_fwd(i, t, x, v, a):
    """11 condition features driven by the actual current (I[t-1])."""
    av = np.abs(v)
    vp = np.maximum(v, 0.0)
    vn = -np.minimum(v, 0.0)
    sign = np.tanh(v / 0.1)
    slow = np.exp(-av / 0.1)
    return np.stack([i, t, v * i, v * t,
                     i * np.tanh(vp / 0.1), i * vn * av, i * sign,
                     a * i, i * np.abs(a),
                     i * x * slow, i * np.abs(x) * slow], 1)


def cond_inv(f, ip, x, v, a):
    """11 condition features, slot-for-slot mirror of cond_fwd.

    The drive channel `i` becomes the desired force `f`; the environment slot
    `t` becomes the past current `ip` (I[k-1], the only current INV may see).
    """
    av = np.abs(v)
    vp = np.maximum(v, 0.0)
    vn = -np.minimum(v, 0.0)
    sign = np.tanh(v / 0.1)
    slow = np.exp(-av / 0.1)
    return np.stack([f, ip, v * f, v * ip,
                     f * np.tanh(vp / 0.1), f * vn * av, f * sign,
                     a * f, f * np.abs(a),
                     f * x * slow, f * np.abs(x) * slow], 1)


D_MOTION, D_COND = 20, 11
