"""Build the V25.2 delivery package: zip with figures + self-contained markdown."""
import base64
import hashlib
import os
import re
import shutil
import zipfile

SRC = r"D:/v25_2_cmd_proxy"
NAME = "V25_2_CMD_PROXY_PACKAGE"
PKG = os.path.join(SRC, "package", NAME)
ZIP = os.path.join(SRC, NAME + ".zip")
REPORT = "V25_2_FINAL_REPORT.md"

if os.path.isdir(os.path.join(SRC, "package")):
    shutil.rmtree(os.path.join(SRC, "package"))
if os.path.exists(ZIP):
    os.remove(ZIP)

DIRS = ["figures", "audits", "configs", "task", "prompts", "logs", "scripts",
        "runs/V25_2_RI0_CMD_PROXY_seed7/figures"]
FILES = ["V25_2_FINAL_REPORT.md", "SHA256SUMS.txt", "model_def.py", "v252_data.py",
         "v20_data.py", "v23_data.py", "phase0_audits.py", "alias_edge_audit.py",
         "smoke_v252.py", "train_v252.py", "verify_v252.py", "plots_v252.py",
         "gen_sha.py"]
RUNFILES = ["best.pth", "last.pth", "best_by_val_mae.pth", "metrics.jsonl",
            "summary.json", "run_config.json", "per_file_metrics.csv",
            "per_condition_metrics.csv", "proxy_command_metrics.json",
            "persistence_comparison.csv"]

os.makedirs(PKG, exist_ok=True)
for d in DIRS:
    s, t = os.path.join(SRC, d), os.path.join(PKG, d)
    if os.path.isdir(s):
        shutil.copytree(s, t, dirs_exist_ok=True)
for f in FILES:
    s = os.path.join(SRC, f)
    if os.path.isfile(s):
        shutil.copy2(s, os.path.join(PKG, f))
rd = os.path.join(PKG, "runs", "V25_2_RI0_CMD_PROXY_seed7")
os.makedirs(rd, exist_ok=True)
for f in RUNFILES:
    s = os.path.join(SRC, "runs", "V25_2_RI0_CMD_PROXY_seed7", f)
    if os.path.isfile(s):
        shutil.copy2(s, os.path.join(rd, f))

# ---------------------------------------------------------------- embedded md
md = open(os.path.join(PKG, REPORT), encoding="utf-8").read()
n = 0
for m in sorted(set(re.findall(r"\(figures/([^)]+\.png)\)", md))):
    p = os.path.join(PKG, "figures", m)
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    md = md.replace("(figures/%s)" % m, "(data:image/png;base64,%s)" % b64)
    n += 1
emb = md.replace("# V25.2-RI0-CMD 最终报告",
                 "# V25.2-RI0-CMD 最终报告（自包含版：图片已内嵌）", 1)
open(os.path.join(PKG, "V25_2_FINAL_REPORT_embedded.md"), "w",
     encoding="utf-8", newline="\n").write(emb)

readme = """# V25.2-RI0-CMD 交付包

基于上一拍电流指令的残差逆模型（无电流反馈、无数字滤波器）。

## 先看哪个

| 需求 | 文件 |
|---|---|
| 完整报告（图片走相对路径） | `V25_2_FINAL_REPORT.md` |
| 单文件转发（图片 base64 内嵌，不依赖目录） | `V25_2_FINAL_REPORT_embedded.md` |
| 十二问速答 | 报告 §7 |
| 结论速览 | 报告 §0 |

## 目录

```
V25_2_CMD_PROXY_PACKAGE/
├── V25_2_FINAL_REPORT.md              最终报告（%d 张图，相对路径）
├── V25_2_FINAL_REPORT_embedded.md     同一份报告，图片已内嵌
├── SHA256SUMS.txt                     全部文件 SHA256 + 字节数
├── figures/                           报告用图（3 张）
├── audits/                            Phase 0 七份审计 + smoke + Phase 4 复算
├── runs/V25_2_RI0_CMD_PROXY_seed7/    权重、指标、每文件/每工况 CSV
├── code/ (根目录)                     数据层 / 审计 / 训练 / 复算 / 出图脚本
├── configs/ task/ prompts/ scripts/   原始训练包内容
└── logs/                              phase0 / alias_edge / train 日志
```

## 关键数字

| 项 | 值 |
|---|---|
| 参数量 | 2073（与 V25 逐项相同） |
| 电流 MAE | 0.0388 A（Persistence 0.0445，−12.8%%） |
| RMSE / P95 / P99 | 0.0912 / 0.1770 / 0.4325 A |
| 指令 vs 实测电流滞后 | **+1.3 ms**（V25 为 +38.7 ms） |
| 数字滤波附加延迟 | **0 ms**（FIR 已移除） |
| CPU 单步 P99 | 0.477 ms |
| final_test 读取 | 0 次 |

## 最重要的限制

所有指标都是**代理标签**结果。`I_act[k] ≈ I_cmd[k-1]` 是工程假设，
数据集中只有实测线圈电流、没有 ECU 指令记录，因此**真实指令精度未被验证**。
在此之前不得宣称指令跟踪精度已达标。
""" % n
open(os.path.join(PKG, "README_PACKAGE.md"), "w", encoding="utf-8",
     newline="\n").write(readme)

# ---------------------------------------------------------------- zip
cnt = 0
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for root, dirs, files in os.walk(PKG):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in sorted(files):
            if f.endswith(".pyc"):
                continue
            full = os.path.join(root, f)
            z.write(full, os.path.join(NAME, os.path.relpath(full, PKG)))
            cnt += 1
print("zip: %s  files=%d  %.2f MB  embedded_imgs=%d"
      % (ZIP, cnt, os.path.getsize(ZIP) / 1e6, n))
