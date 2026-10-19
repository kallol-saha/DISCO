"""Sim2real dataset wrapper for the 2-pose (grasp + place) policy.

Reads the same `successful_grasp_place*.pth` files as
`ShelfPackingGraspPlaceDataset` but applies the sim2real augmentation
pipeline from `sim2real_augment.augment_sample` per __getitem__:

- per-point Gaussian noise,
- random holes (with at most 1 on the target object),
- independent SE(3) transforms for shelf vs object points,
- per-sample centering and max-radius scaling.

The returned tensor shapes match the legacy dataset so the trainer's
`prepare_batch` does not need to change. The model trained against this
dataset must use a fixed `[-1, 1]` workspace_normalizer (the dataloader
already normalizes positions into a unit ball).
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from .sim2real_augment import (
    DEFAULT_HOLE_N_RANGE,
    DEFAULT_HOLE_R_RANGE,
    DEFAULT_NOISE_SIGMA_MAX,
    DEFAULT_TRANS_RANGE,
    augment_sample,
    load_shelf_y_thresholds,
)


PLACEHOLDER_PROPRIO = torch.zeros(8, dtype=torch.float32)


class Sim2RealShelfPackingGraspPlaceDataset(Dataset):
    """Sim2real-augmented (grasp, place) pairs from MCTS rollouts."""

    train_copies = 10
    quat_format = "wxyz"

    def __init__(
        self,
        root,                                # path to successful_grasp_place*.pth
        env_root,                            # path to assets/environments/shelf_packing/<train|test>
        copies: Optional[int] = None,
        relative_action: bool = False,       # accepted for trainer parity, unused
        mem_limit: int = 8,                  # accepted for trainer parity, unused
        actions_only: bool = False,
        sigma_max: float = DEFAULT_NOISE_SIGMA_MAX,
        n_range=DEFAULT_HOLE_N_RANGE,
        r_range=DEFAULT_HOLE_R_RANGE,
        trans_range: float = DEFAULT_TRANS_RANGE,
        max_holes_on_target: Optional[int] = 1,
    ):
        super().__init__()
        self.copies = self.train_copies if copies is None else copies
        self._actions_only = actions_only

        self.data = torch.load(root, map_location="cpu")
        for required in ("input_pcd", "goal_pose", "env_index"):
            if required not in self.data:
                raise KeyError(
                    f"{root} missing key '{required}'. Got keys: {list(self.data.keys())}"
                )

        self.num_samples = self.data["input_pcd"].shape[0]
        if self.data["goal_pose"].shape[0] != self.num_samples:
            raise ValueError(
                f"goal_pose length {self.data['goal_pose'].shape[0]} != "
                f"input_pcd length {self.num_samples}"
            )
        if self.data["goal_pose"].ndim != 3 or self.data["goal_pose"].shape[1:] != (2, 8):
            raise ValueError(
                f"Expected goal_pose shape (N, 2, 8); got {tuple(self.data['goal_pose'].shape)}"
            )
        if self.data["input_pcd"].ndim != 3 or self.data["input_pcd"].shape[-1] not in (3, 4):
            raise ValueError(
                f"Expected input_pcd shape (N, P, 3|4); got {tuple(self.data['input_pcd'].shape)}"
            )
        self.pcd_input_channels = int(self.data["input_pcd"].shape[-1])

        # env_index → world-frame y threshold (computed before robot-base offset).
        self.shelf_y_thresholds = load_shelf_y_thresholds(Path(env_root))

        # Some `.pth` files reference env_indices for envs that no longer
        # exist on disk (env layouts were re-numbered / pruned over time).
        # Drop those rows here rather than failing — they're a small
        # fraction in practice and have no recoverable per-env shelf-y.
        ei = self.data["env_index"]
        keep_mask = torch.tensor(
            [int(i) in self.shelf_y_thresholds for i in ei.tolist()],
            dtype=torch.bool,
        )
        n_dropped = int((~keep_mask).sum())
        if n_dropped > 0:
            missing = sorted(
                {int(i) for i, k in zip(ei.tolist(), keep_mask.tolist()) if not k}
            )
            print(
                f"[Sim2RealShelfPackingGraspPlaceDataset] dropping {n_dropped} "
                f"samples whose env_index has no env yaml under {env_root}: "
                f"missing env_indices={missing}"
            )
            kept = keep_mask.nonzero(as_tuple=False).squeeze(-1)
            for k in ("input_pcd", "goal_pose", "env_index", "node_id"):
                if k in self.data:
                    self.data[k] = self.data[k][kept]
            if "reward" in self.data:
                self.data["reward"] = self.data["reward"][kept]
            self.num_samples = self.data["input_pcd"].shape[0]

        self._sigma_max = float(sigma_max)
        self._n_range = tuple(n_range)
        self._r_range = tuple(r_range)
        self._trans_range = float(trans_range)
        self._max_holes_on_target = max_holes_on_target

        self._proprio_placeholder = PLACEHOLDER_PROPRIO.clone()

        print(
            f"[Sim2RealShelfPackingGraspPlaceDataset] {self.num_samples} samples "
            f"from {root} (pcd channels={self.pcd_input_channels}, "
            f"σ_max={self._sigma_max}, holes={self._n_range}@{self._r_range}m, "
            f"trans=±{self._trans_range}m, max_holes_on_target={self._max_holes_on_target})"
        )

    def __len__(self):
        return self.copies * self.num_samples

    def __getitem__(self, idx):
        idx = idx % self.num_samples

        pcd_in = self.data["input_pcd"][idx]                 # (P, C)
        action_in = self.data["goal_pose"][idx]              # (2, 8)
        env_index = int(self.data["env_index"][idx].item())
        threshold = self.shelf_y_thresholds[env_index]

        pcd_out, action_out, _centroid, _scale = augment_sample(
            pcd_in,
            action_in,
            shelf_y_threshold=threshold,
            sigma_max=self._sigma_max,
            n_range=self._n_range,
            r_range=self._r_range,
            max_holes_on_target=self._max_holes_on_target if self.pcd_input_channels >= 4 else None,
            trans_range=self._trans_range,
        )

        if self._actions_only:
            return {"action": action_out.unsqueeze(0)}

        proprio = self._proprio_placeholder.view(1, 1, 8)

        return {
            "pcd": pcd_out.unsqueeze(0),         # (1, P, C)
            "action": action_out.unsqueeze(0),   # (1, 2, 8)
            "proprioception": proprio,           # (1, 1, 8) placeholder, ignored by 2-pose policy
        }
