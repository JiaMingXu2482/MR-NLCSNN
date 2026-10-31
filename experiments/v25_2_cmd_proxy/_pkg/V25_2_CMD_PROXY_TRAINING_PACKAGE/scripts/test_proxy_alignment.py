import numpy as np
from proxy_alignment import (
    direct_sample_100hz,
    make_current_command_proxy,
    replace_current_history_feature,
)

def main():
    # A 1 kHz synthetic ramp sampled at phase 0 gives a 100 Hz ramp.
    raw = np.arange(0, 100, dtype=np.float32) / 100.0
    sampled = direct_sample_100hz(raw, factor=10, phase=0)
    assert np.allclose(sampled[:5], [0.00, 0.10, 0.20, 0.30, 0.40])

    # Signed current maps to nonnegative command proxy, then shifts one step.
    i = np.array([0.0, 0.2, -0.4, 0.8, 1.9], dtype=np.float32)
    p = make_current_command_proxy(i, current_limit_a=1.6)
    assert np.allclose(p["base_proxy_a"], [0.0, 0.2, 0.4, 0.8])
    assert np.allclose(p["target_proxy_a"], [0.2, 0.4, 0.8, 1.6])
    assert np.allclose(p["delta_target_a"], [0.2, 0.2, 0.4, 0.8])
    assert np.array_equal(p["target_source_index"], [1, 2, 3, 4])

    # A verified current-history slot is replaced with previous-command proxy.
    x = np.zeros((4, 31), dtype=np.float32)
    x2 = replace_current_history_feature(x, p["base_proxy_a"], 21)
    assert np.allclose(x2[:, 21], np.array([0.0, 0.2, 0.4, 0.8]) / 1.6)
    assert np.all(x2[:, 0] == 0.0)

    print("V25.2 PROXY ALIGNMENT: ALL PASS")

if __name__ == "__main__":
    main()
