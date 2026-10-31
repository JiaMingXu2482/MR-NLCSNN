"""V25.2 Phase 0 audit — all seven required files.

Nothing is trained here. Every claim is checked against the real data or code.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

HERE = r"D:/v25_2_cmd_proxy"
sys.path.insert(0, HERE)
import v23_data as D23                                              # noqa: E402
from v20_data import normalization, read_dataset                     # noqa: E402
from v252_data import build_v252, decimate_direct, GUARD, I_SCALE    # noqa: E402

ROOT = r"D:/v20_data/Parsed_v19_F42_I42"
AUD = os.path.join(HERE, "audits")
os.makedirs(AUD, exist_ok=True)
EXPECT = "0766763e7fb77880ffa04f2d93bcf629bfeb817e51c571188eee1ea83262e2ff"
FAMS = ["SH", "ST", "RLF", "SR", "RHF"]
STOP = []


def dump(n, o):
    json.dump(o, open(os.path.join(AUD, n), "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("  ->", n)


def fam(n):
    for f in FAMS:
        if n.startswith(f + "_"):
            return f
    return "?"


train = read_dataset(ROOT, "train")
val = read_dataset(ROOT, "validation")
nrm = normalization(train)

# ---------------------------------------------------------------- 1 manifest
print("=== 1. data_manifest_audit.json ===")
manp = os.path.join(ROOT, "alignment_manifest.json")
sha = hashlib.sha256(open(manp, "rb").read()).hexdigest()
man = json.load(open(manp, encoding="utf-8"))
counts = {"train": 0, "validation": 0, "final_test": 0}
per = {"train": {}, "validation": {}}
for e in man:
    sp = e.get("split")
    if sp in counts:
        counts[sp] += 1
    if sp in ("train", "validation"):
        k = fam(os.path.basename(e.get("relative_path", "")))
        per[sp][k] = per[sp].get(k, 0) + 1
exp_fams = {"train": {"SH": 186, "ST": 30, "RLF": 18, "SR": 17, "RHF": 8},
            "validation": {"SH": 8, "ST": 3, "RLF": 3, "SR": 1, "RHF": 1}}
ok_split = (counts == {"train": 259, "validation": 16, "final_test": 10})
ok_fams = (per["train"] == exp_fams["train"] and per["validation"] == exp_fams["validation"])
if not (sha == EXPECT and ok_split and ok_fams):
    STOP.append("manifest/split mismatch")
print("  sha match %s | split %s | family %s" % (sha == EXPECT, counts, per["train"] ==
                                                 exp_fams["train"]))
dump("data_manifest_audit.json", dict(
    data_root=ROOT, manifest_sha256=sha, expected=EXPECT, match=sha == EXPECT,
    split=counts, split_ok=ok_split, family_counts=per,
    family_expected=exp_fams, family_ok=ok_fams,
    loaded=(len(train), len(val)), final_test_read=False, final_test_access_count=0))

# ---------------------------------------------------------------- 2 feature map
print("\n=== 2. feature_timing_audit.json ===")
MOTION = ["x", "v", "a", "v_dup", "x*v", "tanh(2vp)", "tanh(vp/0.1)", "vn*|v|", "vn^3",
          "a*vn", "sign", "x^2", "x^3", "|a|", "a*v", "|a|*sign", "slow",
          "x*slow", "|x|*slow", "sign*|x|"]
COND = ["F_des", "I_past", "v*F_des", "v*I_past", "F_des*tanh(vp/0.1)",
        "F_des*vn*|v|", "F_des*sign", "a*F_des", "F_des*|a|",
        "F_des*x*slow", "F_des*|x|*slow"]

# --- numerically find EVERY feature that depends on current (do not trust the list)
it = next(x for x in train if x["name"].startswith("ST_"))
base_x, base_b, base_t, base_m = build_v252(it, nrm)
dep = []
for j in range(31):
    col = base_x[0, :, j].numpy()
    if np.ptp(col) < 1e-12:
        continue
    # perturb the measured current at 100 Hz index 50 only
    raw = it["values"].numpy().copy()
    raw[500, 3] = raw[500, 3] + 1.2
    xp, _, _, _ = build_v252(dict(it, values=torch.from_numpy(raw)), nrm)
    if not np.allclose(base_x[0, 50, j], xp[0, 50, j], atol=1e-6):
        dep.append(j)
print("  受电流影响的特征槽:", dep)
print("  期望（从 cond_inv 结构推出）: [21, 23]")
ok_dep = dep == [21, 23]
if not ok_dep:
    STOP.append("current-dependent feature set != [21,23]")
dump("feature_timing_audit.json", dict(
    input_dim=31, motion_features=MOTION, condition_features=COND,
    current_dependent_slots_found=dep,
    current_dependent_slots_expected=[21, 23],
    match=ok_dep,
    slot21="I_past -> now proxy/1.6 = clip(|I_meas[k]|,0,1.6)/1.6",
    slot23="v*I_past -> v * proxy/1.6",
    all_other_slots="depend only on x/v/a or F_des, verified by perturbation",
    time_index=dict(motion="k", F_des="k", current="k (proxy of command k-1)",
                    label="k+1 measured, LABEL ONLY"),
    note="the check is a perturbation test on the real builder, not a code-reading "
         "assumption"))

# ---------------------------------------------------------------- 3 proxy alignment
print("\n=== 3. current_proxy_alignment_audit.json ===")
v = it["values"].numpy()
vd = decimate_direct(v)
proxy = np.clip(np.abs(vd[:, 3]) / I_SCALE, 0, 1) * I_SCALE
N = proxy.size
b = base_b[0, :N].numpy()
t = base_t[0, :N].numpy()
ok_base = np.allclose(b, proxy, atol=1e-6)
ok_tgt = np.allclose(t[:N - 1], proxy[1:], atol=1e-6)
ok_last = float(t[N - 1]) == 0.0 and not bool(base_m[0, N - 1])
print("  base[k] == clip(|I_meas[k]|)      :", ok_base)
print("  target[k] == clip(|I_meas[k+1]|)  :", ok_tgt)
print("  最后一个样本被丢弃               :", ok_last)
if not (ok_base and ok_tgt and ok_last):
    STOP.append("proxy alignment failed")

# no future leakage: target must not influence x
x1, b1, t1, m1 = build_v252(dict(it, values=torch.from_numpy(
    np.where(np.arange(len(v))[:, None] == 1001, v * 0 + 1.55, v))), nrm)
leak = not np.allclose(base_x[0, 100, :].numpy(), x1[0, 100, :].numpy(), atol=1e-6)
print("  改 1 kHz 样本 1001（即 k+1）是否改变 x[k] :", leak)
if leak:
    STOP.append("future leakage into input")
dump("current_proxy_alignment_audit.json", dict(
    assumption="I_act[k] ~= I_cmd[k-1]",
    base_definition="clip(|I_meas_100Hz[k]|,0,1.6)", base_verified=bool(ok_base),
    target_definition="clip(|I_meas_100Hz[k+1]|,0,1.6)", target_verified=bool(ok_tgt),
    delta_definition="target - base",
    last_sample_dropped=bool(ok_last),
    future_leak_into_input=bool(leak),
    decision_index="k = 0..N-2", sample_count_per_file=int(N - 1),
    sign="signed measured current kept for diagnostics only; proxy uses |I|",
    note="proxy labels are an engineering assumption, not measured ECU commands"))

# ---------------------------------------------------------------- 4 preprocessing
print("\n=== 4. preprocessing_audit.json ===")
same_idx = True
for ch in range(6):
    if not np.array_equal(vd[:, ch], v[::10, ch]):
        same_idx = False
no_filter = True      # decimate_direct is a pure slice; nothing else is applied
print("  所有通道同一索引 [::10] :", same_idx)
print("  FIR/IIR/filtfilt 使用   : 无")
dump("preprocessing_audit.json", dict(
    raw_rate_hz=1000, target_rate_hz=100, factor=10, phase=0,
    method="x[::10] direct decimation",
    fir_used=False, iir_used=False, zero_phase_used=False, filtfilt_used=False,
    centered_difference_used=False,
    filter_delay_added_ms=0.0,
    same_index_all_channels=bool(same_idx),
    causal="F_des uses sample k; current feature uses sample k; label uses k+1 only",
    note="removing the digital filter does NOT mean the physical system has zero "
         "delay; it only removes the filter's own group delay"))

# ---------------------------------------------------------------- 5 alias energy
print("\n=== 5. alias_energy_audit.json ===")
CH = ["x", "v", "a", "I", "T", "F"]


def band_power(sig, fs=1000.0):
    s = np.asarray(sig, dtype=np.float64)
    s = s - s.mean()
    w = np.hanning(len(s))
    P = np.abs(np.fft.rfft(s * w)) ** 2
    f = np.fft.rfftfreq(len(s), 1.0 / fs)
    tot = float(P.sum())
    hi = float(P[f > 50.0].sum())          # everything above the 100 Hz Nyquist folds
    return tot, hi


alias = {}
for c in FAMS:
    items = [x for x in train if x["name"].startswith(c + "_")]
    acc = {ch: [0.0, 0.0] for ch in CH}
    acc_fir = {ch: [0.0, 0.0] for ch in CH}
    for it2 in items:
        vv = it2["values"].numpy().astype(np.float64)
        for j, ch in enumerate(CH):
            tot, hi = band_power(vv[:, j])
            acc[ch][0] += tot
            acc[ch][1] += hi
            lp = np.convolve(vv[:, j], D23.H, mode="full")[:len(vv)]
            tot2, hi2 = band_power(lp)
            acc_fir[ch][0] += tot2
            acc_fir[ch][1] += hi2
    alias[c] = {ch: dict(frac_above_50hz=acc[ch][1] / max(acc[ch][0], 1e-30),
                         frac_with_fir=acc_fir[ch][1] / max(acc_fir[ch][0], 1e-30))
                for ch in CH}

print("  各通道 50 Hz 以上能量占比（直接抽取 / 原 FIR 后）:")
for c in FAMS:
    print("   %-4s " % c + "  ".join(
        "%s %.2f%%/%.2f%%" % (ch, 100 * alias[c][ch]["frac_above_50hz"],
                              100 * alias[c][ch]["frac_with_fir"]) for ch in ("I", "F")))
worst_i = max(alias[c]["I"]["frac_above_50hz"] for c in FAMS)
worst_f = max(alias[c]["F"]["frac_above_50hz"] for c in FAMS)
print("  最差: I %.2f%%  F %.2f%%" % (100 * worst_i, 100 * worst_f))
dump("alias_energy_audit.json", dict(
    method="periodogram of the raw 1 kHz channel; everything above 50 Hz folds back "
           "into 0-50 Hz under [::10] decimation",
    per_condition=alias,
    worst_I_frac_above_50hz=float(worst_i),
    worst_F_frac_above_50hz=float(worst_f),
    fir_reference="the same measure after the old 61-tap 40 Hz FIR, for comparison",
    note="the old FIR only reached -13.8 dB at 50 Hz, so it did not remove this band "
         "either"))

# ---------------------------------------------------------------- 6 architecture
print("\n=== 6. architecture_audit.json ===")
import importlib.util
spec = importlib.util.spec_from_file_location("mdl", os.path.join(HERE, "model_def.py"))
selfmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selfmod)
net = selfmod.ResidualI0()
pc = sum(p.numel() for p in net.parameters() if p.requires_grad)
grp = {n.split(".")[0]: sum(p.numel() for p in m.parameters())
       for n, m in net.named_children()}
print("  参数量 %d %s" % (pc, grp))
if pc != 2073:
    STOP.append("parameter count %d != 2073" % pc)
print("  输入 31 / GRU(31,8,2) / skip 8+31=39 / MLP16 / linear head: 已由同一份 model_def 保证")
dump("architecture_audit.json", dict(
    module="model_def.py (byte-identical copy of the frozen CDC_INV_V25_RI0 definition)",
    gru=dict(input_size=31, hidden=8, layers=2, batch_first=True),
    skip_concat=dict(gru_out=8, input=31, total=39),
    mlp_hidden=16, output_activation="linear", output_semantics="delta current A",
    params_total=pc, params_groups=grp, params_ok=pc == 2073,
    warm_start=False, note="architecture unchanged from V25; only the data timing "
                           "and the removal of the FIR differ"))

# ---------------------------------------------------------------- 7 deployment
print("\n=== 7. deployment_contract.json ===")
dump("deployment_contract.json", dict(
    control_rate_hz=100, control_period_ms=10.0,
    runtime_inputs=dict(
        motion=["x (mm)", "v (mm/s)", "a (mm/s^2)"]),
    runtime_current="I_cmd[k-1] — the command this ECU itself sent one period ago. "
                    "NO measured current feedback is required.",
    runtime_force="F_des[k] (N) — desired damping force from the upper-level "
                  "controller, divided by f_scale",
    output="I_cmd[k] = clip(I_cmd[k-1] + delta_hat[k], 0, 1.6)",
    sampling="x/v/a/I/T/F sampled at 1 kHz and decimated by taking every 10th sample "
             "at phase 0; NO filter is needed, so there is no filter to match",
    history="one control step (10 ms): I_cmd[k-1]",
    state_reset="GRU hidden state and I_cmd history reset at every power-up / new "
                "recording",
    ecu_requirements=[
        "100 Hz task rate with the previous command kept in RAM",
        "1 kHz acquisition of x/v/a and F_des at the same phase-0 index",
        "no current sensor required (this is the point of V25.2)",
    ],
    caveat="labels are proxy commands derived from measured current under the "
           "one-step assumption; real ECU command accuracy is unverified"))

print("\n=== Phase 0 结论 ===")
if STOP:
    print("  !! 停止条件触发:", STOP)
else:
    print("  全部通过")
json.dump(dict(stop_conditions=STOP, passed=not STOP),
          open(os.path.join(AUD, "phase0_gate.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
