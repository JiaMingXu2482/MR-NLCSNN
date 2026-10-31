import hashlib
import os

SKIP = {"_pkg", "__pycache__"}
lines = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in SKIP]
    for f in sorted(files):
        if f.endswith(".pyc") or f == "SHA256SUMS.txt":
            continue
        p = os.path.join(root, f)
        rel = os.path.relpath(p, ".").replace(os.sep, "/")
        lines.append((rel, hashlib.sha256(open(p, "rb").read()).hexdigest(),
                      os.path.getsize(p)))
lines.sort()
with open("SHA256SUMS.txt", "w", encoding="utf-8", newline="\n") as fh:
    for rel, h, s in lines:
        fh.write("%s  %s  %d\n" % (h, rel, s))
print("files: %d  total %.2f MB" % (len(lines), sum(x[2] for x in lines) / 1e6))
