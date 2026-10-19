"""
Pre-train the initial grasp and place policies by behavior cloning.
"""
import argparse
import os
import threading
from pathlib import Path

import torch

from run_disco import train

RAW = "assets/processed_data/shelf_packing_one_object/data.pth"
DATA = "assets/processed_data/pretrain/shelf_policy.pth"


def build_dataset(raw=RAW, out=DATA):
    """Convert raw demos to the pretraining dataset, setting the gripper channel to 1 = grasp, 0 = place.

    Args:
        raw: Raw demo file with `input_pcd` and `goal_poses`.
        out: Output .pth path, written with `input_pcd` and `goal_pose`.
    """
    src = torch.load(raw, map_location="cpu", weights_only=False)
    pose = src["goal_poses"].float().clone()
    pose[:, 0, 7], pose[:, 1, 7] = 1.0, 0.0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"input_pcd": src["input_pcd"].float().contiguous(), "goal_pose": pose.contiguous()}, out)
    print(f"{out}: {pose.shape[0]} demonstrations")


def main():
    """Build the dataset if missing, then train the grasp and place policies in parallel."""
    ap = argparse.ArgumentParser(description=__doc__.strip())
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="models/pretrain")
    ap.add_argument("--iters", type=int, default=100_000)
    ap.add_argument("--gpus", default="0,1", help="grasp model on the first, place model on the last")
    ap.add_argument("--wandb", action="store_true", help="log training to wandb")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        build_dataset(out=args.data)
    channels = torch.load(args.data, map_location="cpu", weights_only=False)["input_pcd"].shape[-1]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]

    results = {}

    def fit(kind, gpu):
        """Train one P0 policy from scratch and store its checkpoint in `results`.

        Args:
            kind: "grasp" or "place".
            gpu: GPU index to train on.
        """
        results[kind] = train(kind, Path(args.data).resolve(), None, out, f"P0_{kind}", args.iters, gpu,
                              out / f"P0_{kind}.log", channels, args.wandb)

    threads = [threading.Thread(target=fit, args=(k, g)) for k, g in (("grasp", gpus[0]), ("place", gpus[-1]))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print({k: str(v) for k, v in results.items()})


if __name__ == "__main__":
    main()
