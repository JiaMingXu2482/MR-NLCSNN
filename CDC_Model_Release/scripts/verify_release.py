"""Final verification: manifest integrity, originals untouched, read-only enforced."""
import hashlib
import json
import os
import stat

REL = r"D:/CDC_Model_Release"
MAN = os.path.join(REL, "release_manifest.json")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


man = json.load(open(MAN, encoding="utf-8"))
side = open(os.path.join(REL, "release_manifest.sha256"), encoding="utf-8").read().split()[0]
print("=== 1. 清单自身一致性 ===")
print("  记录的清单 sha256 :", side)
print("  实际计算          :", sha(MAN))
print("  一致 :", sha(MAN) == side)

print("\n=== 2. 逐文件哈希复验 (%d 个) ===" % len(man["files"]))
bad = []
for f in man["files"]:
    p = os.path.join(REL, f["path"])
    if not os.path.exists(p):
        bad.append((f["path"], "missing"))
        continue
    if sha(p) != f["sha256"]:
        bad.append((f["path"], "sha mismatch"))
    if os.path.getsize(p) != f["size_bytes"]:
        bad.append((f["path"], "size mismatch"))
print("  失配文件数 :", len(bad))
for b in bad:
    print("   !!", b)

print("\n=== 3. 原件未被修改 ===")
pairs = [
    (r"D:/v22_ae_hybrid/runs_seed7/E0/best.pth",
     os.path.join(REL, "Forward_Model/CDC_FWD_V22_E0/best.pth")),
    (r"D:/v25_residual/runs/V25_RI0_RESIDUAL_seed7/best.pth",
     os.path.join(REL, "Inverse_Model/CDC_INV_V25_RI0/best.pth")),
    (r"D:/v22_ae_hybrid/v22_models.py",
     os.path.join(REL, "Forward_Model/CDC_FWD_V22_E0/model_def.py")),
    (r"D:/v25_residual/v25_models.py",
     os.path.join(REL, "Inverse_Model/CDC_INV_V25_RI0/model_def.py")),
]
for src, rel_p in pairs:
    a, b = sha(src), sha(rel_p)
    print("  %-52s %s" % (os.path.basename(rel_p), "一致" if a == b else "不一致!!"))

print("\n=== 4. 只读保护 ===")
writable = []
for root, dirs, names in os.walk(REL):
    for n in names:
        p = os.path.join(root, n)
        mode = os.stat(p).st_mode
        if mode & stat.S_IWRITE:
            writable.append(os.path.relpath(p, REL))
print("  可写文件数 :", len(writable))
for w in writable[:10]:
    print("   (可写)", w)

print("\n=== 5. 发布目录结构 ===")
for root, dirs, names in os.walk(REL):
    dirs.sort()
    lvl = root.replace(REL, "").count(os.sep)
    print("  " * lvl + os.path.basename(root) + "/")
    for n in sorted(names):
        print("  " * (lvl + 1) + n)

print("\n=== 6. 关键指标摘录 ===")
for k in ("forward", "inverse"):
    m = man["models"][k]
    print("  %-8s %-20s params=%d seed=%d best_ep=%s sha=%s…"
          % (k, m["release_name"], m["params"], m["seed"],
             m["best_epoch_in_checkpoint"], m["sha256"][:12]))
print("  load_test:", man["load_test"]["verdict"])
print("  final_test_read:", man["final_test_read"])
