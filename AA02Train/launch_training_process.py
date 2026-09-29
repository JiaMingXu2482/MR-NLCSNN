import argparse
import os
import subprocess
from pathlib import Path


def parse_env(items):
    env = os.environ.copy()
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE env item, got: {item}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def normalize_windows_env(env, python_exe):
    if os.name != "nt":
        return env
    path_values = []
    for key in list(env.keys()):
        if key.lower() == "path":
            path_values.append(env.pop(key))

    python_dir = Path(python_exe).resolve().parent
    conda_prefix = python_dir
    preferred = [
        str(conda_prefix),
        str(conda_prefix / "Library" / "bin"),
        str(conda_prefix / "Library" / "usr" / "bin"),
        str(conda_prefix / "DLLs"),
        str(conda_prefix / "Scripts"),
    ]

    seen = set()
    merged = []
    for value in preferred + path_values:
        for part in str(value).split(os.pathsep):
            clean = part.strip()
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(clean)

    env["Path"] = os.pathsep.join(merged)
    return env


def main():
    parser = argparse.ArgumentParser(description="Launch a detached training process with explicit env vars.")
    parser.add_argument("--python", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--stdout", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--no-detach", action="store_true", help="Do not use Windows DETACHED_PROCESS")
    parser.add_argument("--plain-popen", action="store_true", help="Do not set Windows creationflags")
    parser.add_argument("--settle-seconds", type=float, default=0.0, help="Keep launcher alive briefly after Popen")
    args = parser.parse_args()

    stdout_path = Path(args.stdout)
    stderr_path = Path(args.stderr)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    env = normalize_windows_env(parse_env(args.env), args.python)
    creationflags = 0
    if os.name == "nt" and not args.plain_popen:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        if not args.no_detach:
            creationflags |= subprocess.DETACHED_PROCESS

    with stdout_path.open("ab", buffering=0) as stdout_f, stderr_path.open("ab", buffering=0) as stderr_f:
        proc = subprocess.Popen(
            [args.python, args.script],
            cwd=args.cwd,
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )
        if args.settle_seconds > 0:
            import time

            time.sleep(args.settle_seconds)
    print(f"pid={proc.pid}")
    print(f"stdout={stdout_path}")
    print(f"stderr={stderr_path}")


if __name__ == "__main__":
    main()
