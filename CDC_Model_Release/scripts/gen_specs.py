"""Phase 4 (part 1): feature specs, golden vectors for C-port verification, model reports."""
import hashlib
import json
import os
import sys

import numpy as np
import torch

REL = r"D:/CDC_Model_Release"
FWD_DIR = os.path.join(REL, "Forward_Model", "CDC_FWD_V22_E0")
INV_DIR = os.path.join(REL, "Inverse_Model", "CDC_INV_V25_RI0")
ROOT = r"D:/v20_data/Parsed_v19_F42_I42"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def write(p, t):
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(t)
    return sha(p)


# ------------------------------------------------------------------ FWD spec
NFL = [
    ("x", "obs_x", "motion", "(x - x_ref)/x_scale", "t"),
    ("v", "obs_v", "motion", "v/v_scale", "t"),
    ("a", "obs_a", "motion", "a/a_scale", "t"),
    ("i", "obs_i", "current", "clip(|I|/i_scale, 0, 1) evaluated at t-1", "t-1"),
    ("T", "obs_T", "environment", "(T - t_ref)/t_scale", "t"),
    ("v_dup", "v", "motion", "v  (second copy, kept for order fidelity)", "t"),
    ("x_v", "x*v", "motion", "x*v", "t"),
    ("v_i", "v*i", "cross", "v*i", "t"),
    ("v_T", "v*T", "cross", "v*T", "t"),
    ("tanh_2vp", "tanh(2*vp)", "motion", "vp=max(v,0)", "t"),
    ("tanh_vp", "tanh(vp/0.1)", "motion", "vp=max(v,0)", "t"),
    ("i_tanh_vp", "i*tanh(vp/0.1)", "cross", "vp=max(v,0)", "t"),
    ("vn_av", "vn*|v|", "motion", "vn=-min(v,0)", "t"),
    ("vn3", "vn^3", "motion", "vn=-min(v,0)", "t"),
    ("a_vn", "a*vn", "motion", "vn=-min(v,0)", "t"),
    ("i_vn_av", "i*vn*|v|", "cross", "vn=-min(v,0)", "t"),
    ("sign", "tanh(v/0.1)", "motion", "sign(v) soft", "t"),
    ("i_sign", "i*tanh(v/0.1)", "cross", "sign", "t"),
    ("x2", "x^2", "motion", "-", "t"),
    ("x3", "x^3", "motion", "-", "t"),
    ("a_i", "a*i", "cross", "-", "t"),
    ("abs_a", "|a|", "motion", "-", "t"),
    ("a_v", "a*v", "motion", "-", "t"),
    ("abs_a_sign", "|a|*tanh(v/0.1)", "motion", "-", "t"),
    ("i_abs_a", "i*|a|", "cross", "-", "t"),
    ("slow", "exp(-|v|/0.1)", "motion", "-", "t"),
    ("x_slow", "x*exp(-|v|/0.1)", "motion", "-", "t"),
    ("absx_slow", "|x|*exp(-|v|/0.1)", "motion", "-", "t"),
    ("i_x_slow", "i*x*exp(-|v|/0.1)", "cross", "-", "t"),
    ("i_absx_slow", "i*|x|*exp(-|v|/0.1)", "cross", "-", "t"),
    ("sign_absx", "tanh(v/0.1)*|x|", "motion", "-", "t"),
]
fwd_spec = dict(
    release="CDC_FWD_V22_E0", input_rate_hz=1000,
    raw_input=dict(channels=["x_mm", "v_mm_s", "a_mm_s2", "I_A", "T_C"],
                   note="raw sensor units; the model normalises internally"),
    normalisation=dict(x="(x - x_ref)/x_scale", v="v/v_scale", a="a/a_scale",
                       i="clip(|I|/i_scale, 0, 1)", T="(T - t_ref)/t_scale"),
    current_history_rule="the 5-channel input carries I[t]; inside the model the "
                         "current column is shifted by one sample, so every current "
                         "term uses I[t-1]. I[t-1] at t=0 is 0.",
    feature_order=[dict(index=i, name=n, formula=f, group=g, raw_source=s, time_index=ti)
                   for i, (n, f, g, s, ti) in enumerate(NFL)],
    feature_count=31,
    output=dict(units="N", rule="F = model_output * f_scale",
                normalised_range="roughly -1..1"),
    state=dict(gru_initial="zeros, per file", window="none (whole sequence)",
               update="standard GRU, one step per 1 ms"),
)
print("  FWD feature_spec.json", write(os.path.join(FWD_DIR, "feature_spec.json"),
                                      json.dumps(fwd_spec, indent=2, ensure_ascii=False))[:16])

# ------------------------------------------------------------------ INV spec
motion = ["x", "v", "a", "v_dup", "x*v", "tanh(2*vp)", "tanh(vp/0.1)", "vn*|v|", "vn^3",
          "a*vn", "sign", "x^2", "x^3", "|a|", "a*v", "|a|*sign", "slow", "x*slow",
          "|x|*slow", "sign*|x|"]
cond = [("F_des", "F_meas_decimated[k]/f_scale"),
        ("I_past", "I_meas_decimated[k-1]/i_scale  (SIGNED, not clipped)"),
        ("v*F_des", "v*F_des"), ("v*I_past", "v*I_past"),
        ("F_des*tanh(vp/0.1)", "-"), ("F_des*vn*|v|", "-"),
        ("F_des*sign", "sign=tanh(v/0.1)"), ("a*F_des", "-"),
        ("F_des*|a|", "-"), ("F_des*x*slow", "slow=exp(-|v|/0.1)"),
        ("F_des*|x|*slow", "-")]
inv_spec = dict(
    release="CDC_INV_V25_RI0", input_rate_hz=100,
    preprocessing=dict(
        filter="61-tap causal windowed-sinc low-pass, 40 Hz (coefficients in "
               "fir_61tap_40hz.json)",
        decimation="take every 10th filtered sample starting at index 0",
        group_delay_ms=30.0,
        note="the SAME filtered/decimated x, v, a, I, T, F must be used; the 30 ms "
             "delay is common to inputs and label so it cancels"),
    motion_features=[dict(index=i, name=n, time_index="k") for i, n in enumerate(motion)],
    condition_features=[dict(index=20 + i, name=n, formula=f, time_index="k / k-1")
                        for i, (n, f) in enumerate(cond)],
    feature_count=31,
    residual_base=dict(
        definition="I_base[k] = clip(|I_meas_100Hz[k-1]|, 0, 1.6)  [A]",
        NOTE="this is DIFFERENT from feature slot 1 (I_past), which is the SIGNED, "
             "unclipped I/1.6. Both derive from the same sensor sample at k-1 but are "
             "transformed differently; do not merge them in the C port."),
    output=dict(units="A", semantics="delta current",
                equation="I_cmd[k] = clip(I_base[k] + delta[k], 0.0, 1.6)"),
    state=dict(gru_initial="zeros, per file", base_initial="undefined at k=0; the first "
               "control step of every file is excluded from training and metrics",
               window="none (whole sequence), one step per 10 ms"),
)
print("  INV feature_spec.json", write(os.path.join(INV_DIR, "feature_spec.json"),
                                      json.dumps(inv_spec, indent=2, ensure_ascii=False))[:16])

# ------------------------------------------------------------------ golden vectors
sys.path.insert(0, r"D:/v22_ae_hybrid")
sys.path.insert(0, r"D:/v25_residual")
from v20_data import normalization, read_dataset, sample                   # noqa: E402
from v22_models import V22Model                                            # noqa: E402
from v25_data import build_residual                                        # noqa: E402
from v25_models import ResidualI0                                          # noqa: E402

train = read_dataset(ROOT, "train")
val = read_dataset(ROOT, "validation")
norm = normalization(train)
it = next(x for x in val if x["name"].startswith("SH_"))

mf = V22Model("E0")
mf.load_state_dict(torch.load(os.path.join(FWD_DIR, "best.pth"), map_location="cpu",
                              weights_only=True)["state_dict"])
mf.eval()
with torch.no_grad():
    u, f, valid = sample(it, norm)
    pred = mf(u)[0] * norm["f_scale"]
s0, s1 = 0, 1200   # from file start: GRU state must start from zero
gv_f = dict(model="CDC_FWD_V22_E0", source_file=it["name"],
            index_range=[s0, s1], rate_hz=1000,
            input_raw=u[0, s0:s1].numpy().tolist(),
            input_raw_units=["mm", "mm/s", "mm/s^2", "A", "degC"],
            expected_force_N=pred[s0:s1].numpy().tolist(),
            note="feed input_raw through the documented normalisation + I[t-1] shift; "
                 "the force must match expected_force_N to ~1e-5 relative")
np.savez(os.path.join(FWD_DIR, "golden_vectors.npz"),
         input_raw=u[0, s0:s1].numpy(), expected_force_N=pred[s0:s1].numpy())
write(os.path.join(FWD_DIR, "golden_vectors_meta.json"),
      json.dumps(gv_f, indent=2, ensure_ascii=False))

mi = ResidualI0()
mi.load_state_dict(torch.load(os.path.join(INV_DIR, "best.pth"), map_location="cpu",
                              weights_only=True)["state_dict"])
mi.eval()
with torch.no_grad():
    x, y, base, mask = build_residual(it, norm)
    delta = mi(x)[0]
    cmd = (base[0] + delta).clamp(0, 1.6)
k0, k1 = 0, 150    # from file start: GRU state must start from zero
gv_i = dict(model="CDC_INV_V25_RI0", source_file=it["name"], index_range=[k0, k1],
            rate_hz=100,
            input_features=x[0, k0:k1].numpy().tolist(),
            base_A=base[0, k0:k1].numpy().tolist(),
            expected_delta_A=delta[k0:k1].numpy().tolist(),
            expected_I_cmd_A=cmd[k0:k1].numpy().tolist(),
            note="input_features is the already-normalised 31-vector; base_A is the "
                 "residual base in amps; I_cmd = clip(base+delta, 0, 1.6)")
np.savez(os.path.join(INV_DIR, "golden_vectors.npz"),
         input_features=x[0, k0:k1].numpy(), base_A=base[0, k0:k1].numpy(),
         expected_delta_A=delta[k0:k1].numpy(), expected_I_cmd_A=cmd[k0:k1].numpy())
write(os.path.join(INV_DIR, "golden_vectors_meta.json"),
      json.dumps(gv_i, indent=2, ensure_ascii=False))
print("  golden vectors written for both models")
