"""v20 data loading. Reuses the v19 aligned dataset and guards, with one change:
the input is [x, v, a, I, T] only. current_dot (dI) is never loaded as a model
input, because v20 drops dI for the inverse model. final_test is never loaded
by the training script; see train_v20.py.

Dataset root must contain alignment_manifest.json, COMPLETE.json and the
processed CSVs, e.g. the extracted Parsed_v19_F42_I42 directory from the
v19 Codex pack. Do NOT point this at raw Card data.
"""
import csv
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_dataset(root, partition):
    """Read one partition. partition is 'train' or 'validation' only."""
    if partition not in ("train", "validation"):
        raise ValueError("v20 training only loads train/validation, never final_test")
    root = Path(root)
    manifest = root / "alignment_manifest.json"
    complete = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
    if digest(manifest) != complete["manifest_sha256"]:
        raise ValueError("manifest differs from completed export")
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    used = set()
    for row in rows:
        if row["split"] in ("train", "validation", "final_test"):
            if row["raw_sha256"] in used:
                raise ValueError("duplicate raw hash in active splits")
            used.add(row["raw_sha256"])
    result = []
    for row in rows:
        if row["split"] != partition:
            continue
        path = (root / row["relative_path"]).resolve()
        if root.resolve() not in path.parents or digest(path) != row["processed_sha256"]:
            raise ValueError("invalid processed path/hash: " + str(path))
        with path.open(encoding="utf-8-sig", newline="") as stream:
            table = list(csv.DictReader(stream))
        keys = ("rod_length", "velocity", "accel", "current", "temp", "force")
        values = torch.tensor([[float(r[k]) for k in keys] for r in table])
        timestamps = [float(r["time"]) for r in table]
        if any(abs(b - a - 0.001) > 1e-6 for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("data are not uniformly sampled at 1 kHz")
        if len(values) < 200 or not torch.isfinite(values).all():
            raise ValueError("short/nonfinite file: " + str(path))
        # Same tolerance as the v19 runner fix (2026-10-27): signed current,
        # magnitude up to 2.10 A (filter overshoot at step edges). sample()
        # clamps magnitude to 1.6 A.
        if values[:, 3].abs().max() > 2.10:
            raise ValueError("current magnitude outside [0,2.1] A tolerance: " + str(path))
        result.append(dict(name=path.stem, values=values))
    if not result:
        raise ValueError("empty " + partition)
    return result


def normalization(items):
    xmin = min(float(it["values"][:, 0].min()) for it in items)
    xmax = max(float(it["values"][:, 0].max()) for it in items)
    return dict(
        x_ref=(xmin + xmax) / 2,
        x_scale=max((xmax - xmin) / 2, 1e-6),
        f_scale=max(max(float(it["values"][:, 5].abs().max()) for it in items), 1e-6),
        v_scale=1000.0,
        a_scale=50000.0,
        i_scale=1.6,
        t_ref=40.0,
        t_scale=40.0,
    )


def edge_guard(n):
    """1000 samples (1 s) on both ends for long files, capped at 100 for
    short files (<2000 samples), exactly as in the v19 runner."""
    return 1000 if n > 2000 else min(100, n // 4)


def sample(item, norm):
    """Return (u, f, valid). u is (1, T, 5) = [x, v, a, I, T], normalised.
    No dI column, no CurrentLag, no tau: any current lag must be carried
    by the GRU hidden state from the raw current history itself."""
    v = item["values"]
    u = torch.stack(
        (
            (v[:, 0] - norm["x_ref"]) / norm["x_scale"],
            v[:, 1] / 1000,
            v[:, 2] / 50000,
            (v[:, 3].abs() / 1.6).clamp(0, 1),
            (v[:, 4] - 40) / 40,
        ),
        -1,
    )[None]
    f = (v[:, 5] / norm["f_scale"])[None]
    valid = torch.ones_like(f, dtype=torch.bool)
    guard = edge_guard(v.shape[0])
    if guard:
        valid[:, :guard] = False
        valid[:, -guard:] = False
    return u, f, valid
