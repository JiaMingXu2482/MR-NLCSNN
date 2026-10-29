import csv
import json
import os

import numpy as np

ROOT = r"D:/v20_data/Parsed_v19_F42_I42"
man = json.load(open(os.path.join(ROOT, "alignment_manifest.json"), encoding="utf-8"))
val = [e for e in man if e.get("split") == "validation"]

print("%-34s %-18s %8s %8s %8s %6s %8s %8s %s" % (
    "file", "cat", "dur_s", "x_mean", "x_ac_rms", "rms/pk", "I_ac_rms", "I_ac_pk", "verdict"))
for e in val:
    p = os.path.join(ROOT, e["relative_path"].replace("/", os.sep))
    with open(p, encoding="utf-8-sig", newline="") as fh:
        t = list(csv.DictReader(fh))
    x = np.array([float(r["rod_length"]) for r in t])
    i = np.array([float(r["current"]) for r in t])
    xa = x - x.mean()
    ia = i - i.mean()
    xrms = float(np.sqrt((xa ** 2).mean()))
    xpk = float(np.abs(xa).max())
    ratio = xrms / xpk if xpk else 0.0
    irms = float(np.sqrt((ia ** 2).mean()))
    ipk = float(np.abs(ia).max())
    cat = e["relative_path"].split("/")[0]
    verdict = "SINUSOID" if ratio > 0.60 else ("RANDOM" if ratio < 0.45 else "mixed")
    print("%-34s %-18s %8.2f %8.2f %8.3f %6.3f %8.3f %8.3f %s" % (
        os.path.basename(e["relative_path"]), cat, len(x) / 1000.0, float(x.mean()),
        xrms, ratio, irms, ipk, verdict))
