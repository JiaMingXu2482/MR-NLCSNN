import os
import re
import zipfile

Z = r"D:/v25_2_cmd_proxy/V25_2_CMD_PROXY_PACKAGE.zip"
OUT = r"D:/v25_2_cmd_proxy/_zipcheck"
root = "V25_2_CMD_PROXY_PACKAGE"

if os.path.isdir(OUT):
    import shutil
    shutil.rmtree(OUT)
with zipfile.ZipFile(Z) as z:
    names = z.namelist()
    z.extractall(OUT)
print("zip 条目数: %d" % len(names))

base = os.path.join(OUT, root)
md = os.path.join(base, "V25_2_FINAL_REPORT.md")
body = open(md, encoding="utf-8").read()
refs = re.findall(r"\(figures/([^)]+\.png)\)", body)
print("\n相对路径版引用的图 (%d):" % len(refs))
for r in refs:
    print("   %-38s %s" % (r, "存在" if os.path.isfile(os.path.join(base, "figures", r)) else "缺失!"))

emb = open(os.path.join(base, "V25_2_FINAL_REPORT_embedded.md"), encoding="utf-8").read()
print("\n内嵌版 data:image 数量: %d" % emb.count("data:image/png;base64,"))
print("内嵌版残留相对引用: %d" % len(re.findall(r"\(figures/[^)]+\.png\)", emb)))

must = ["runs/V25_2_RI0_CMD_PROXY_seed7/best.pth",
        "runs/V25_2_RI0_CMD_PROXY_seed7/last.pth",
        "runs/V25_2_RI0_CMD_PROXY_seed7/metrics.jsonl",
        "runs/V25_2_RI0_CMD_PROXY_seed7/per_condition_metrics.csv",
        "audits/data_manifest_audit.json",
        "audits/smoke_test.json",
        "audits/phase4_recompute.json",
        "figures/per_condition_vs_persistence.png",
        "SHA256SUMS.txt", "README_PACKAGE.md", "model_def.py"]
miss = [m for m in must if not os.path.isfile(os.path.join(base, m))]
print("\n关键文件缺失: %s" % (miss if miss else "无"))
tot = sum(len(f) for _, _, f in os.walk(base))
print("解包后文件数: %d  大小 %.2f MB"
      % (tot, sum(os.path.getsize(os.path.join(r, f))
                  for r, _, fs in os.walk(base) for f in fs) / 1e6))
