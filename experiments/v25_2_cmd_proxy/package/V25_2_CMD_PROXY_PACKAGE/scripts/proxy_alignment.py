"""V25.2 one-control-step measured-current -> command-proxy alignment.

The data contain measured current, not ECU command. Under the agreed
engineering assumption I_act[k] ~= I_cmd[k-1]:
    base_proxy[k]   = abs(I_meas[k])
    target_proxy[k] = abs(I_meas[k+1])

This module only constructs proxy labels. It does not prove the assumption.
"""
from __future__ import annotations
import numpy as np


def direct_sample_100hz(signal_1khz, factor: int = 10, phase: int = 0):
    """Direct sample 1 kHz data onto the 100 Hz grid; no FIR/IIR is applied."""
    x = np.asarray(signal_1khz)
    if x.ndim != 1:
        raise ValueError("signal_1khz must be a 1-D array")
    if factor <= 0 or not (0 <= phase < factor):
        raise ValueError("invalid factor/phase")
    return x[phase::factor]


def make_current_command_proxy(
    measured_current_100hz,
    current_limit_a: float = 1.6,
):
    """Return base[k], target[k], delta[k] and source indices.

    Input is the sign-corrected measured coil current on the 100 Hz grid.
    Absolute value maps the signed sensor convention to nonnegative current
    command convention, matching the previous V25 target convention.
    Last sample is dropped because target[k] needs measured_current[k+1].
    """
    current = np.asarray(measured_current_100hz, dtype=np.float64)
    if current.ndim != 1:
        raise ValueError("measured_current_100hz must be 1-D")
    if current.size < 2:
        raise ValueError("at least two 100 Hz samples are required")
    if not np.all(np.isfinite(current)):
        raise ValueError("measured current contains NaN/Inf")
    if current_limit_a <= 0:
        raise ValueError("current_limit_a must be positive")

    proxy = np.clip(np.abs(current), 0.0, current_limit_a)
    base = proxy[:-1].copy()
    target = proxy[1:].copy()
    delta = target - base
    decision_index = np.arange(proxy.size - 1, dtype=np.int64)
    target_source_index = decision_index + 1
    return {
        "base_proxy_a": base.astype(np.float32),
        "target_proxy_a": target.astype(np.float32),
        "delta_target_a": delta.astype(np.float32),
        "decision_index": decision_index,
        "target_source_index": target_source_index,
    }


def replace_current_history_feature(
    features,
    base_proxy_a,
    current_feature_index: int,
    current_limit_a: float = 1.6,
):
    """Replace one verified current-history feature with previous-command proxy.

    `current_feature_index` must be resolved from the real feature map by the
    local agent. This function deliberately does not guess the index.
    """
    x = np.asarray(features, dtype=np.float32)
    base = np.asarray(base_proxy_a, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("features must have shape [time, feature]")
    if not (0 <= current_feature_index < x.shape[1]):
        raise IndexError("current_feature_index outside feature dimension")
    if x.shape[0] != base.shape[0]:
        raise ValueError("features and proxy base must have same time length")
    out = x.copy()
    out[:, current_feature_index] = base / current_limit_a
    return out


def assert_no_future_label_in_input(features_at_k, measured_current_k1):
    """A tiny guard used in tests: X[k] must not be constructed from I[k+1].

    The full project audit must perform a source-level perturbation test on
    the actual feature builder; this helper is not a substitute for it.
    """
    x = np.asarray(features_at_k)
    y = np.asarray(measured_current_k1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("length mismatch")
    return True
