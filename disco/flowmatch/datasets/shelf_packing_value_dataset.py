import torch
from torch.utils.data import Dataset


class ShelfPackingValueDataset(Dataset):
    """Pointwise dataset for the value network.

    Each saved row is one (parent state, action, target) triple. The PCD
    is 3-channel xyz only (no target-object mask — the mask experiment
    was reverted alongside the listwise loss on 2026-05-01).

    Expected `.pth` schema (flat dict, written by `save_phase2_data.py`):
        input_pcd:  FloatTensor (N, 4096, 3)   xyz
        goal_pose:  FloatTensor (N, 2, 8)
        target:     FloatTensor (N,)            n_succ / n_leaves
        n_leaves:   IntTensor   (N,)            log1p weight at training
        env_index:  IntTensor   (N,)            provenance
        parent_id:  IntTensor   (N,)            provenance
        node_id:    IntTensor   (N,)            provenance
    """

    train_copies = 10

    def __init__(
        self,
        root,
        copies=None,
        relative_action=False,  # unused; kept for trainer signature compat
        mem_limit=8,            # unused
        actions_only=False,
    ):
        super().__init__()
        self.copies = self.train_copies if copies is None else copies
        self._actions_only = actions_only

        blob = torch.load(root)
        for k in ("input_pcd", "goal_pose", "target", "n_leaves"):
            if k not in blob:
                raise ValueError(
                    f"{root}: missing key '{k}'. Regenerate with "
                    f"`save_phase2_data.py`."
                )

        self.input_pcd = blob["input_pcd"]   # (N, 4096, 4) float32
        self.goal_pose = blob["goal_pose"]   # (N, 2, 8) float32
        self.target = blob["target"]         # (N,) float32
        self.n_leaves = blob["n_leaves"]     # (N,) int32

        self.num_samples = self.input_pcd.shape[0]
        if self.num_samples == 0:
            raise ValueError(f"{root} contains zero samples.")

        assert self.input_pcd.shape[-2:] == (4096, 3)
        assert self.goal_pose.shape[-2:] == (2, 8)
        assert self.target.shape[0] == self.num_samples
        assert self.n_leaves.shape[0] == self.num_samples

        print(
            f"[ShelfPackingValueDataset] {self.num_samples} samples "
            f"(target range [{self.target.min().item():.3f}, "
            f"{self.target.max().item():.3f}], "
            f"n_leaves range [{int(self.n_leaves.min())}, "
            f"{int(self.n_leaves.max())}])"
        )

    def __getitem__(self, idx):
        idx = idx % self.num_samples

        pcd = self.input_pcd[idx].clone()         # (4096, 3)
        action = self.goal_pose[idx].clone()      # (2, 8)
        target = self.target[idx].item()
        n_leaves = self.n_leaves[idx].item()

        # Robot-frame +0.615 x-offset on PCD xyz and on action xyz, applied
        # exactly once each (mirrors mcts2._predict_q_hats / training conventions).
        pcd[..., 0] = pcd[..., 0] + 0.615
        action[..., 0] = action[..., 0] + 0.615

        if self._actions_only:
            return {"action": action.unsqueeze(0)}

        return {
            "pcd": pcd.unsqueeze(0),                                       # (1, 4096, 3)
            "actions": action.unsqueeze(0),                                # (1, 2, 8)
            "targets": torch.tensor([target], dtype=torch.float32),        # (1,)
            "n_leaves": torch.tensor([n_leaves], dtype=torch.float32),     # (1,)
        }

    def __len__(self):
        return self.copies * self.num_samples
