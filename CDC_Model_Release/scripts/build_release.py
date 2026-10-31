"""Phase 3: build the frozen release tree. Copies, never modifies, training artifacts.

No retraining, no weight edit, no structure change. Every copied file is hashed.
"""
import hashlib
import json
import os
import shutil

REL = r"D:/CDC_Model_Release"
FWD_SRC = r"D:/v22_ae_hybrid/runs_seed7/E0"
INV_SRC = r"D:/v25_residual/runs/V25_RI0_RESIDUAL_seed7"
FWD_DIR = os.path.join(REL, "Forward_Model", "CDC_FWD_V22_E0")
INV_DIR = os.path.join(REL, "Inverse_Model", "CDC_INV_V25_RI0")
for d in (FWD_DIR, INV_DIR):
    os.makedirs(d, exist_ok=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return sha(path)


print("=== 复制权重与定义文件（原文件不动） ===")
for src, dst in ((os.path.join(FWD_SRC, "best.pth"), os.path.join(FWD_DIR, "best.pth")),
                 (os.path.join(INV_SRC, "best.pth"), os.path.join(INV_DIR, "best.pth")),
                 (r"D:/v22_ae_hybrid/v22_models.py", os.path.join(FWD_DIR, "model_def.py")),
                 (r"D:/v25_residual/v25_models.py", os.path.join(INV_DIR, "model_def.py")),
                 (r"D:/v23_dual_branch/v23_data.py", os.path.join(INV_DIR, "feature_def.py")),
                 (os.path.join(FWD_SRC, "run_config.json"),
                  os.path.join(FWD_DIR, "source_run_config.json")),
                 (os.path.join(INV_SRC, "run_config.json"),
                  os.path.join(INV_DIR, "source_run_config.json")),
                 (os.path.join(INV_SRC, "summary.json"),
                  os.path.join(INV_DIR, "source_summary.json")),
                 (r"D:/v22_ae_hybrid/runs_seed7/E0/summary.json",
                  os.path.join(FWD_DIR, "source_summary.json"))):
    shutil.copy2(src, dst)
    print("  %-58s %s" % (os.path.basename(dst), sha(dst)[:16]))

print("\n=== normalization.json（取自 checkpoint 内嵌 norm） ===")
for d, name in ((FWD_DIR, "CDC_FWD_V22_E0"), (INV_DIR, "CDC_INV_V25_RI0")):
    import torch
    ck = torch.load(os.path.join(d, "best.pth"), map_location="cpu", weights_only=True)
    norm = {k: float(v) for k, v in ck["norm"].items()}
    payload = dict(
        release=name, source="checkpoint['norm'] (the exact values used at train and "
                             "eval time; NOT recomputed)",
        values=norm,
        units=dict(x_ref="mm", x_scale="mm", v_scale="mm/s", a_scale="mm/s^2",
                   i_scale="A", t_ref="degC", t_scale="degC", f_scale="N"),
        usage=dict(
            x="(x - x_ref) / x_scale",
            v="v / v_scale",
            a="a / a_scale",
            current_e0_convention="clip(|I| / i_scale, 0, 1)",
            temperature="(T - t_ref) / t_scale",
            force="F / f_scale  (inverse: force_N = output * f_scale)"))
    print("  %-22s %s" % (name, write(os.path.join(d, "normalization.json"),
                                      json.dumps(payload, indent=2, ensure_ascii=False))[:16]))

print("\n=== model_config.json ===")
FWD_CFG = dict(
    release_name="CDC_FWD_V22_E0", role="forward damping-force predictor",
    version="V22-E0", frozen=True,
    source=dict(project="V22 AE-hybrid matrix", run_dir=FWD_SRC,
                checkpoint="best.pth", checkpoint_epoch=127,
                run_config="source_run_config.json", seed=7, epochs_trained=130),
    architecture=dict(
        module="model_def.py :: V22Model(model_id='E0')",
        gru=dict(input_size=31, hidden_size=8, num_layers=2, batch_first=True),
        trunk=dict(layers=["Linear(39,16)", "ReLU"], in_features=39, out_features=16),
        head=dict(layers=["Linear(16,1)"], activation="linear", out_features=1),
        skip_concat=dict(parts=["gru_output_8", "nfl_features_31"], concat_dim=39),
        uses_monotonic_G=False, uses_tau=False, uses_acc_residual=False),
    parameters=dict(total=2073, gru=1416, trunk=640, head_force=17),
    io=dict(input="u (B,T,5) = [x, v, a, I, T] at 1 kHz, raw units",
            output="normalised force (B,T); multiply by f_scale for newtons",
            history="I is shifted one sample inside the model, so the current "
                    "channel is I[t-1]; no fixed window, zero GRU state per file"),
    metrics=dict(pooled_nrmse_pct=10.594998, convention="100*sqrt(sum e^2 / sum y^2), "
                                                       "uncentred, newtons",
                 per_condition_pct=dict(SH=4.7411, ST=11.5174, RLF=19.6818,
                                        SR=34.9814, RHF=15.2199)),
    files=dict(model_def="model_def.py", normalization="normalization.json",
               feature_spec="feature_spec.json"))
INV_CFG = dict(
    release_name="CDC_INV_V25_RI0", role="residual inverse current model",
    version="V25-RI0", frozen=True,
    source=dict(project="V25 residual inverse", run_dir=INV_SRC,
                checkpoint="best.pth", checkpoint_epoch=130,
                run_config="source_run_config.json", seed=7, epochs_trained=130,
                warm_started_from_v24_i0=False),
    architecture=dict(
        module="model_def.py :: ResidualI0()",
        gru=dict(input_size=31, hidden_size=8, num_layers=2, batch_first=True),
        skip_concat=dict(parts=["gru_output_8", "input_31"], concat_dim=39),
        trunk=dict(layers=["Linear(39,16)", "ReLU"], in_features=39, out_features=16),
        head=dict(layers=["Linear(16,1)"], activation="LINEAR, no sigmoid",
                  out_features=1),
        uses_monotonic_G=False, uses_tau=False),
    parameters=dict(total=2073, gru=1416, trunk=640, head=17),
    io=dict(input="x (B,T,31) = motion[20] + condition[11] at 100 Hz",
            output="delta current in A, unbounded, may be negative",
            equation="I_cmd[k] = clip(I_meas_100Hz[k-1] + delta[k], 0.0, 1.6)",
            residual_base="I_meas_100Hz[k-1] = clip(|I_measured_decimated[k-1]|,0,1.6) A",
            history="one control step (10 ms) of past current; no fixed window; "
                    "zero GRU state and zero base at every file start"),
    metrics=dict(mae_A=0.030007, rmse_A=0.066758, p95_A=0.138436, p99_A=0.316785,
                 pooled_nrmse_pct=9.2948,
                 per_condition_mae_A=dict(SH=0.0069, ST=0.0182, RLF=0.0434,
                                          SR=0.0071, RHF=0.0698),
                 zero_current_pred_mean_A=0.0097, false_turn_on_rate=0.0325,
                 raw_out_of_range_rate=0.0805, clip_activation_rate=0.0805),
    files=dict(model_def="model_def.py", feature_def="feature_def.py",
               normalization="normalization.json", feature_spec="feature_spec.json",
               fir="fir_61tap_40hz.json"))
for d, cfg in ((FWD_DIR, FWD_CFG), (INV_DIR, INV_CFG)):
    print("  %-22s %s" % (cfg["release_name"],
                          write(os.path.join(d, "model_config.json"),
                                json.dumps(cfg, indent=2, ensure_ascii=False))[:16]))

print("\n=== FIR 系数（仅逆模型：100 Hz 输入端需要完全相同的抗混叠滤波） ===")
import sys
sys.path.insert(0, r"D:/v25_residual")
import v23_data as D23
coef = D23.H.tolist()
fir = dict(name="61-tap causal windowed-sinc low-pass", taps=61, cutoff_hz=D23.CUTOFF,
           fs_hz=D23.FS, window="hamming",
           phase="linear causal", group_delay_samples=(len(D23.H) - 1) / 2,
           group_delay_ms=(len(D23.H) - 1) / 2 / D23.FS * 1000,
           dc_gain=float(D23.H.sum()), coefficients=coef,
           application="y[n] = sum_{i=0..60} h[i] * x[n-i]   (causal, zero initial state)",
           decimation="take y[0], y[10], y[20], ... (index 0 aligned)",
           decimation_factor=D23.DECIM,
           note="the SAME filter must be applied to x, v, a, I, T, F before the "
                "inverse model sees them; the 30 ms group delay is common to inputs "
                "and label so it cancels in training")
print("  %-22s %s" % ("fir_61tap_40hz.json",
                      write(os.path.join(INV_DIR, "fir_61tap_40hz.json"),
                            json.dumps(fir, indent=2))[:16]))
print("\nPhase 3 files done.")
