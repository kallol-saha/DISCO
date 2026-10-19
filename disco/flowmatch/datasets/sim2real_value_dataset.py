"""Sim2real dataset wrapper for the value (Q-function) network.

Reads the same `q_value_data*.pth` files as `ShelfPackingValueDataset`
and applies the sim2real augmentation pipeline. The pcd is xyz-only
(3-channel), so the target-mask logic in `augment_sample` is skipped
automatically (no cap on holes-on-target).

Returns the same keys as `ShelfPackingValueDataset` so the value trainer
does not need to change.
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


class Sim2RealShelfPackingValueDataset(Dataset):
    """Pointwise (parent state, action, target) triples for the value
    network, with sim2real augmentation applied per __getitem__.
    """

    train_copies = 10

    def __init__(
        self,
        root,                                # path to q_value_data*.pth
        env_root,                            # path to assets/environments/shelf_packing/<train|test>
        copies: Optional[int] = None,
        relative_action: bool = False,       # accepted for trainer parity, unused
        mem_limit: int = 8,                  # accepted for trainer parity, unused
        actions_only: bool = False,
        sigma_max: float = DEFAULT_NOISE_SIGMA_MAX,
        n_range=DEFAULT_HOLE_N_RANGE,
        r_range=DEFAULT_HOLE_R_RANGE,
        trans_range: float = DEFAULT_TRANS_RANGE,
    ):
        super().__init__()
        self.copies = self.train_copies if copies is None else copies
        self._actions_only = actions_only

        blob = torch.load(root, map_location="cpu")
        for k in ("input_pcd", "goal_pose", "target", "n_leaves", "env_index"):
            if k not in blob:
                raise ValueError(
                    f"{root}: missing key '{k}'. Regenerate with `save_phase2_data.py`."
                )

        self.input_pcd = blob["input_pcd"]   # (N, P, 3) float32
        self.goal_pose = blob["goal_pose"]   # (N, 2, 8) float32
        self.target = blob["target"]         # (N,) float32
        self.n_leaves = blob["n_leaves"]     # (N,) int32
        self.env_index = blob["env_index"]   # (N,) int

        self.num_samples = self.input_pcd.shape[0]
        if self.num_samples == 0:
            raise ValueError(f"{root} contains zero samples.")

        if self.input_pcd.ndim != 3 or self.input_pcd.shape[-1] != 3:
            raise ValueError(
                f"Expected input_pcd shape (N, P, 3); got {tuple(self.input_pcd.shape)}"
            )
        if self.goal_pose.shape[-2:] != (2, 8):
            raise ValueError(
                f"Expected goal_pose shape (N, 2, 8); got {tuple(self.goal_pose.shape)}"
            )

        self.shelf_y_thresholds = load_shelf_y_thresholds(Path(env_root))

        # Drop rows whose env_index has no env yaml on disk (data files
        # may reference env_indices that were renumbered/pruned).
        keep_mask = torch.tensor(
            [int(i) in self.shelf_y_thresholds for i in self.env_index.tolist()],
            dtype=torch.bool,
        )
        n_dropped = int((~keep_mask).sum())
        if n_dropped > 0:
            missing = sorted(
                {int(i) for i, k in zip(self.env_index.tolist(), keep_mask.tolist()) if not k}
            )
            print(
                f"[Sim2RealShelfPackingValueDataset] dropping {n_dropped} "
                f"samples whose env_index has no env yaml under {env_root}: "
                f"missing env_indices={missing}"
            )
            kept = keep_mask.nonzero(as_tuple=False).squeeze(-1)
            self.input_pcd = self.input_pcd[kept]
            self.goal_pose = self.goal_pose[kept]
            self.target = self.target[kept]
            self.n_leaves = self.n_leaves[kept]
            self.env_index = self.env_index[kept]
            self.num_samples = self.input_pcd.shape[0]

        self._sigma_max = float(sigma_max)
        self._n_range = tuple(n_range)
        self._r_range = tuple(r_range)
        self._trans_range = float(trans_range)

        print(
            f"[Sim2RealShelfPackingValueDataset] {self.num_samples} samples "
            f"from {root} (target range "
            f"[{self.target.min().item():.3f}, {self.target.max().item():.3f}], "
            f"σ_max={self._sigma_max}, holes={self._n_range}@{self._r_range}m, "
            f"trans=±{self._trans_range}m)"
        )

    def __len__(self):
        return self.copies * self.num_samples

    def __getitem__(self, idx):
        idx = idx % self.num_samples

        pcd_in = self.input_pcd[idx]                          # (P, 3)
        action_in = self.goal_pose[idx]                       # (2, 8)
        env_index = int(self.env_index[idx].item())
        threshold = self.shelf_y_thresholds[env_index]

        pcd_out, action_out, _centroid, _scale = augment_sample(
            pcd_in,
            action_in,
            shelf_y_threshold=threshold,
            sigma_max=self._sigma_max,
            n_range=self._n_range,
            r_range=self._r_range,
            max_holes_on_target=None,   # no mask channel; cap is irrelevant
            trans_range=self._trans_range,
        )

        if self._actions_only:
            return {"action": action_out.unsqueeze(0)}

        target = self.target[idx].item()
        n_leaves = self.n_leaves[idx].item()

        return {
            "pcd": pcd_out.unsqueeze(0),                              # (1, P, 3)
            "actions": action_out.unsqueeze(0),                       # (1, 2, 8)
            "targets": torch.tensor([target], dtype=torch.float32),   # (1,)
            "n_leaves": torch.tensor([n_leaves], dtype=torch.float32),# (1,)
        }
