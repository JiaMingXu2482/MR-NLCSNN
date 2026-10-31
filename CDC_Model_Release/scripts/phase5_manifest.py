"""Phase 5b: release_manifest.json (SHA256 of every file) + read-only protection.

Run this LAST: it hashes the finished tree and then locks it.
"""
import hashlib
import json
import os
import stat
import subprocess

REL = r"D:/CDC_Model_Release"
MAN = os.path.join(REL, "release_manifest.json")
SIDECAR = os.path.join(REL, "release_manifest.sha256")
SKIP = {os.path.abspath(MAN), os.path.abspath(SIDECAR)}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, REL).replace("\\", "/")


fwd_cfg = json.load(open(os.path.join(REL, "Forward_Model/CDC_FWD_V22_E0/"
                                      "model_config.json"), encoding="utf-8"))
inv_cfg = json.load(open(os.path.join(REL, "Inverse_Model/CDC_INV_V25_RI0/"
                                      "model_config.json"), encoding="utf-8"))
fwd_p2 = json.load(open(os.path.join(REL, "audits/phase2_fwd_recompute.json"),
                        encoding="utf-8"))
inv_p2 = json.load(open(os.path.join(REL, "audits/phase2_inv_recompute.json"),
                        encoding="utf-8"))
lt = json.load(open(os.path.join(REL, "audits/load_test.json"), encoding="utf-8"))

files = []
for root, dirs, names in os.walk(REL):
    dirs.sort()
    for n in sorted(names):
        p = os.path.join(root, n)
        if os.path.abspath(p) in SKIP:
            continue
        files.append(dict(path=rel(p), size_bytes=os.path.getsize(p), sha256=sha(p)))

manifest = dict(
    release_name="CDC Model Release",
    release_version="1.0.0",
    frozen_at="2026-10-30",
    status="FROZEN",
    policy="No retraining, no weight edit, no structure change, no feature change in "
           "this directory. New experiments must use a new directory and version.",
    models=dict(
        forward=dict(release_name="CDC_FWD_V22_E0", version="V22-E0",
                     checkpoint="Forward_Model/CDC_FWD_V22_E0/best.pth",
                     sha256=lt["fwd"]["sha256"],
                     size_bytes=os.path.getsize(os.path.join(
                         REL, "Forward_Model/CDC_FWD_V22_E0/best.pth")),
                     params=fwd_cfg["parameters"]["total"], seed=7,
                     epochs_trained=130,
                     best_epoch_in_checkpoint=fwd_p2["epoch_in_checkpoint"],
                     validation=dict(pooled_nrmse_pct=fwd_p2["pooled_nrmse"],
                                     per_condition_pct=fwd_p2["per_condition"],
                                     metric_convention="100*sqrt(sum e^2 / sum y^2), "
                                                       "uncentred, newtons"),
                     recompute=dict(cuda_pooled=fwd_p2["pooled_nrmse"],
                                    cpu_pooled=lt["fwd"]["pooled_nrmse"],
                                    note="CPU vs CUDA float32 differ by 2.1e-4 pp; "
                                         "both reproduce the recorded 10.5950%")),
        inverse=dict(release_name="CDC_INV_V25_RI0", version="V25-RI0",
                     checkpoint="Inverse_Model/CDC_INV_V25_RI0/best.pth",
                     sha256=lt["inv"]["sha256"],
                     size_bytes=os.path.getsize(os.path.join(
                         REL, "Inverse_Model/CDC_INV_V25_RI0/best.pth")),
                     params=inv_cfg["parameters"]["total"], seed=7,
                     epochs_trained=130,
                     best_epoch_in_checkpoint=inv_p2["epoch_in_checkpoint"],
                     validation=dict(mae_A=inv_p2["metrics"]["mae"],
                                     rmse_A=inv_p2["metrics"]["rmse"],
                                     p95_A=inv_p2["metrics"]["p95"],
                                     p99_A=inv_p2["metrics"]["p99"],
                                     per_condition_mae_A={
                                         k: v["mae"] for k, v in
                                         inv_p2["per_condition"].items()},
                                     pooled_nrmse_pct=inv_cfg["metrics"]
                                                           ["pooled_nrmse_pct"]),
                     recompute=dict(cuda_mae=inv_p2["metrics"]["mae"],
                                    cpu_mae=lt["inv"]["mae"],
                                    note="CPU vs CUDA float32 differ by 1.3e-5 A"),
                     deployment_requirement="previous-tick MEASURED current must be "
                                            "available every control period",
                     residual_equation="I_cmd = clip(clip(|I_meas[k-1]|,0,1.6) + "
                                         "delta, 0, 1.6)")),
    load_test=dict(verdict="PASS", forward=lt["fwd"]["verdict"],
                   inverse=lt["inv"]["verdict"],
                   record="audits/load_test.json",
                   golden_forward_max_abs_force_error_N=lt["fwd"]
                   ["golden_max_abs_force_error_N"],
                   golden_inverse_max_abs_delta_error_A=lt["inv"]
                   ["golden_max_abs_delta_delta_error_A"]
                   if "golden_max_abs_delta_delta_error_A" in lt["inv"]
                   else lt["inv"]["golden_max_abs_delta_error_A"]),
    interface_spec=dict(file="MODEL_INTERFACE_SPEC.md",
                        known_inconsistencies=[
                            "motion preprocessing differs: forward uses raw 1 kHz, "
                            "inverse uses 61-tap FIR 40 Hz + decimation (30 ms group "
                            "delay)",
                            "current transform differs: forward clip(|I|/1.6,0,1) at "
                            "t-1; inverse feature uses signed I/1.6, residual base uses "
                            "clip(|I|,0,1.6)",
                            "rate differs: 1 kHz vs 100 Hz (cascade needs 10x ZOH)",
                            "inverse has no temperature input"]),
    final_test_read=0,
    file_count=len(files),
    files=files)

json.dump(manifest, open(MAN, "w", encoding="utf-8", newline="\n"), indent=2,
          ensure_ascii=False)
print("release_manifest.json: %d files hashed" % len(files))

print("\n=== 设只读 ===")
locked = 0
for f in files:
    p = os.path.join(REL, f["path"])
    try:
        os.chmod(p, stat.S_IREAD)
        subprocess.run(["attrib", "+R", p.replace("/", "\\")],
                       capture_output=True, check=False)
        locked += 1
    except Exception as e:                                          # noqa: BLE001
        print("  !! %s: %s" % (f["path"], e))
print("  已锁定 %d 个文件" % locked)

# verify write protection on the two checkpoints
print("\n=== 只读校验 ===")
for relp in ("Forward_Model/CDC_FWD_V22_E0/best.pth",
             "Inverse_Model/CDC_INV_V25_RI0/best.pth"):
    p = os.path.join(REL, relp)
    try:
        open(p, "ab").close()
        print("  %-46s 可写 !! 保护失败" % relp)
        writable = True
    except PermissionError:
        print("  %-46s 写入被拒绝 -> 已保护" % relp)
        writable = False

man_sha = sha(MAN)
open(SIDECAR, "w", encoding="utf-8", newline="\n").write(
    man_sha + "  release_manifest.json\n")
try:
    os.chmod(SIDECAR, stat.S_IREAD)
except Exception:                                                   # noqa: BLE001
    pass
print("\nrelease_manifest.json sha256 = %s" % man_sha)
print("-> release_manifest.sha256")
