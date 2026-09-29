import argparse
import copy
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Average two legacy NLCSNN checkpoints into one checkpoint.")
    parser.add_argument("--base", required=True, help="Base checkpoint path.")
    parser.add_argument("--other", required=True, help="Other checkpoint path.")
    parser.add_argument("--alpha", type=float, required=True, help="Weight for --other. Output = (1-alpha)*base + alpha*other.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    base_path = Path(args.base)
    other_path = Path(args.other)
    output_path = Path(args.output)

    base = torch.load(base_path, map_location="cpu", weights_only=False)
    other = torch.load(other_path, map_location="cpu", weights_only=False)
    if base.get("model_cfg") != other.get("model_cfg"):
        raise RuntimeError("model_cfg differs; refusing to average checkpoints.")
    if base.get("norm_cfg") != other.get("norm_cfg"):
        raise RuntimeError("norm_cfg differs; refusing to average checkpoints.")

    averaged = copy.deepcopy(base)
    state = {}
    for key, base_tensor in base["state_dict"].items():
        other_tensor = other["state_dict"].get(key)
        if other_tensor is None:
            raise RuntimeError(f"Missing key in other checkpoint: {key}")
        if tuple(base_tensor.shape) != tuple(other_tensor.shape):
            raise RuntimeError(f"Shape mismatch for {key}: {tuple(base_tensor.shape)} vs {tuple(other_tensor.shape)}")
        if torch.is_floating_point(base_tensor):
            state[key] = base_tensor * (1.0 - args.alpha) + other_tensor * args.alpha
        else:
            state[key] = base_tensor

    averaged["state_dict"] = state
    averaged["epoch"] = f"avg:{base.get('epoch')}:{other.get('epoch')}"
    averaged["score_nrmse_pct"] = float("nan")
    averaged["averaged_from"] = {
        "base": str(base_path),
        "other": str(other_path),
        "alpha_other": args.alpha,
    }
    averaged.pop("optimizer_state", None)
    averaged.pop("scheduler_state", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged, output_path)
    print(f"Saved averaged checkpoint: {output_path}")


if __name__ == "__main__":
    main()
