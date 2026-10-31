"""Add the training logs/artifacts to the frozen release, then re-freeze.

Excludes last.pth and best_by_val_mae.pth on purpose: a frozen release must not
carry an alternative checkpoint that could be mistaken for the released weight.
"""
import hashlib
import json
import os
import shutil
import stat

REL = r"D:/CDC_Model_Release"
PAIRS = [
    ("CDC_FWD_V22_E0", r"D:/v22_ae_hybrid/runs_seed7/E0",
     os.path.join(REL, "Forward_Model/CDC_FWD_V22_E0")),
    ("CDC_INV_V25_RI0", r"D:/v25_residual/runs/V25_RI0_RESIDUAL_seed7",
     os.path.join(REL, "Inverse_Model/CDC_INV_V25_RI0")),
]
SKIP = {"best.pth", "last.pth", "best_by_val_mae.pth",
        "run_config.json", "summary.json"}   # already archived under other names


def unlock_all():
    for root, _, names in os.walk(REL):
        for n in names:
            try:
                os.chmod(os.path.join(root, n), stat.S_IWRITE | stat.S_IREAD)
            except Exception:                                        # noqa: BLE001
                pass


unlock_all()
print("=== 复制训练日志与诊断产物 ===")
for name, src, dst in PAIRS:
    out = os.path.join(dst, "training_artifacts")
    os.makedirs(out, exist_ok=True)
    copied = []
    for n in sorted(os.listdir(src)):
        sp = os.path.join(src, n)
        if n in SKIP:
            continue
        dp = os.path.join(out, n)
        if os.path.isdir(sp):
            if os.path.exists(dp):
                shutil.rmtree(dp, ignore_errors=True)
            shutil.copytree(sp, dp)
        else:
            shutil.copy2(sp, dp)
        copied.append(n)
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(out) for f in fs)
    print("  %-20s %d 项, %.1f KB: %s" % (name, len(copied), total / 1024,
                                          ", ".join(copied)))
    readme = (
        f"# {name} 训练产物（只读归档）\n\n"
        "本目录保存该模型训练时的逐轮日志与诊断产物，**仅供追溯，不是发布权重**。\n\n"
        "- `metrics.jsonl`：**逐轮训练 log**（每行一个 epoch：train_loss、验证指标、耗时）\n"
        "- `per_file_metrics.csv` / `per_condition_metrics.csv`：逐文件 / 分工况指标\n"
        "- `figures/`：训练报告图\n"
        "- 其他 JSON/CSV：各轮专项诊断（阶跃、回线、单调性、零电流等）\n\n"
        "**未收录**：`last.pth`（最后一轮权重）、`best_by_val_mae.pth`（辅助选择权重）——\n"
        "冻结版本只认上一级的 `best.pth`，避免与发布权重混淆。\n"
        "原始 `run_config.json` / `summary.json` 已分别以 `source_run_config.json` /\n"
        "`source_summary.json` 归档在模型目录下。\n")
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(readme)

print("\n=== 更新 model_config.json 的文件清单 ===")
for name, _, dst in PAIRS:
    p = os.path.join(dst, "model_config.json")
    cfg = json.load(open(p, encoding="utf-8"))
    cfg["files"]["training_artifacts"] = "training_artifacts/ (逐轮 metrics.jsonl + " \
                                         "诊断 CSV/JSON + figures，只读追溯用)"
    cfg["files"]["golden_vectors"] = "golden_vectors.npz + golden_vectors_meta.json"
    if name.startswith("INV"):
        cfg["files"]["feature_def"] = "feature_def.py"
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    print("  %-20s updated" % name)
print("\n完成 —— 现在重跑 phase5_manifest.py 重新哈希并上锁。")
