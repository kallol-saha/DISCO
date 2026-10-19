"""Dataset for the 2-pose (grasp + placement) variant of the packing policy.

Reads either the legacy 4-channel masked file produced by
`save_successful_grasp_place.py` / `save_phase2_data.py`, **or** the
3-channel target-only file produced by
`save_successful_grasp_place_target_only.py`:
    input_pcd:  FloatTensor (N, 4096, 4)   xyz + target_mask  — masked variant
                FloatTensor (N, 4096, 3)   xyz only           — target-only variant
                The masked variant marks which points belong to the
                grasp-target object via channel 4. The target-only variant
                achieves the same disambiguation by *removing* every other
                non-shelved object from the scene before sampling, so the
                target is the only object on the table side of the PCD.
    goal_pose:  FloatTensor (N, 2, 8)   [grasp, placement]; each token is
                xyz(3) + quat_wxyz(4) + gripper_state(1). Index 0 is always
                the grasp pose, index 1 is always the placement pose.
    reward, env_index, node_id: not used by training.

Output (per sample) — matches the dict shape that the shared trainer's
`prepare_batch` reads, so no trainer changes are required:
    pcd:            (1, 4096, C)   xyz in robot-base frame [+ optional mask]
    action:         (1, 2, 8)
    proprioception: (1, 1, 8)   zero placeholder. The 2-pose policy ignores
                                proprioception internally; this field exists
                                only because `BaseTrainTester.prepare_batch`
                                indexes `sample["proprioception"]`.
The model must be constructed with `pcd_input_channels=C` matching the
loaded data; the trainer wires this through.

This file is deliberately kept separate from `shelf_packing_dataset.py` so the
existing placement-only policy and its loader remain untouched.
"""

import torch
from torch.utils.data import Dataset


# +0.615 offset matches what the existing dataset and inference pipeline do
# to express positions in the robot base frame. See action_sampling.py:149,
# motionplanner.py:144, and shelf_packing_dataset.py:61.
ROBOT_BASE_X_OFFSET = 0.615

# Zero placeholder proprioception. The 2-pose (grasp+place) policy ignores
# this field — it exists only to satisfy the shared trainer's `prepare_batch`,
# which indexes sample["proprioception"]. We still build it on the dataset side
# (rather than mutating the trainer) so the trainer code path is unchanged.
PLACEHOLDER_PROPRIO = torch.zeros(8, dtype=torch.float32)


class ShelfPackingGraspPlaceDataset(Dataset):
    """Successful (grasp, place) pairs harvested from MCTS rollouts."""

    train_copies = 10
    quat_format = "wxyz"

    def __init__(
        self,
        root,            # path to successful_grasp_place.pth
        copies=None,     # repeat dataset to reduce loader restarts
        relative_action=False,
        mem_limit=8,     # accepted for parity with ShelfPackingDataset, unused
        actions_only=False,
    ):
        super().__init__()
        self.copies = self.train_copies if copies is None else copies
        self._relative_action = relative_action
        self._actions_only = actions_only

        self.data = torch.load(root, map_location="cpu")

        for required in ("input_pcd", "goal_pose"):
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
                "Expected input_pcd shape (N, P, 3) (target-only variant) "
                "or (N, P, 4) (masked variant). Got "
                f"{tuple(self.data['input_pcd'].shape)}. Regenerate the .pth "
                "file with save_successful_grasp_place.py / "
                "save_phase2_data.py / save_successful_grasp_place_target_only.py."
            )
        self.pcd_input_channels = int(self.data["input_pcd"].shape[-1])

        print(
            f"[ShelfPackingGraspPlaceDataset] loaded {self.num_samples} samples "
            f"from {root} (pcd channels={self.pcd_input_channels})"
        )

        self._proprio_placeholder = PLACEHOLDER_PROPRIO.clone()

    def __len__(self):
        return self.copies * self.num_samples

    def __getitem__(self, idx):
        idx = idx % self.num_samples

        # (1, 4096, C) where C is 4 (xyz + target_mask) or 3 (xyz only) →
        # shift xyz to robot base frame on the fly. Only channel 0 (x) is
        # offset; the optional mask channel must NOT be offset.
        # Use .clone() instead of copy.deepcopy(): deepcopy follows the
        # tensor view back to the (N, 4096, 3) parent storage and copies
        # all ~472 MB on every __getitem__ (~50 ms/call). .clone() copies
        # only the 49 KB view (~0.2 ms/call). See profile_grasp_place_dataloader.py.
        input_pcd = self.data["input_pcd"][idx].unsqueeze(0).clone()
        input_pcd[..., 0] = input_pcd[..., 0] + ROBOT_BASE_X_OFFSET

        # (1, 2, 8) — grasp at index 0, placement at index 1.
        goal_poses = self.data["goal_pose"][idx].unsqueeze(0).clone()
        goal_poses[..., 0] = goal_poses[..., 0] + ROBOT_BASE_X_OFFSET

        if self._actions_only:
            return {"action": goal_poses}

        proprio = self._proprio_placeholder.view(1, 1, 8)   # ignored by the model

        return {
            "pcd": input_pcd,            # (1, 4096, C) — xyz in robot-base frame [+ mask if C=4]
            "action": goal_poses,        # (1, 2, 8)
            "proprioception": proprio,   # (1, 1, 8) placeholder — model ignores it
        }
