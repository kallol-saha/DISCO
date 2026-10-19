import json
import random

from .base import BaseDataset


PERACT_TASKS = [
    "place_cups", "close_jar", "insert_onto_square_peg",
    "light_bulb_in", "meat_off_grill", "open_drawer",
    "place_shape_in_shape_sorter", "place_wine_at_rack_location",
    "push_buttons", "put_groceries_in_cupboard",
    "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
    "slide_block_to_color_target", "stack_blocks", "stack_cups",
    "sweep_to_dustpan_of_size", "turn_tap"
]

PERACT_CUPBOARD_TASKS = [
    "put_groceries_in_cupboard",
]       # NOTE: Task-specific training for the cupboard task

PERACT2_TASKS = [
    'bimanual_push_box',
    'bimanual_lift_ball',
    'bimanual_dual_push_buttons',
    'bimanual_pick_plate',
    'bimanual_put_item_in_drawer',
    'bimanual_put_bottle_in_fridge',
    'bimanual_handover_item',
    'bimanual_pick_laptop',
    'bimanual_straighten_rope',
    'bimanual_sweep_to_dustpan',
    'bimanual_lift_tray',
    'bimanual_handover_item_easy',
    'bimanual_take_tray_out_of_oven'
]


class RLBenchDataset(BaseDataset):
    """RLBench dataset."""
    quat_format= 'xyzw'

    def __init__(
        self,
        root,
        instructions,
        copies=None,
        relative_action=False,
        mem_limit=8,
        actions_only=False,
        chunk_size=4
    ):
        super().__init__(
            root=root,
            instructions=instructions,
            copies=copies,
            relative_action=relative_action,
            mem_limit=mem_limit,
            actions_only=actions_only,
            chunk_size=chunk_size
        )

    def _get_task(self, idx):
        return [
            self.tasks[int(tid)]
            for tid in self.annos['task_id'][idx:idx + self.chunk_size]
        ]

    def _get_instr(self, idx):
        return [
            random.choice(self._instructions[self.tasks[int(t)]][str(int(v))])
            for t, v in zip(
                self.annos['task_id'][idx:idx + self.chunk_size],
                self.annos['variation'][idx:idx + self.chunk_size]
            )
        ]

    def _get_rgb2d(self, idx):
        if self.camera_inds2d is not None:
            return self._get_attr_by_idx(idx, 'rgb', False)[:, self.camera_inds2d]
        return None

    def _get_extrinsics(self, idx):
        return self._get_attr_by_idx(idx, 'extrinsics', True)

    def _get_intrinsics(self, idx):
        return self._get_attr_by_idx(idx, 'intrinsics', True)

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
            task_id: (N,) uint8
            variation: (N,) uint8
            extrinsics: (N, n_cam, 4, 4) float
            intrinsics: (N, n_cam, 3, 3) float
        }
        """
        # First detect which copy we fall into
        idx = idx % (len(self.annos['action']) // self.chunk_size)
        # and then which chunk
        idx = idx * self.chunk_size
        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            "task": self._get_task(idx),  # [str]
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_rgb(idx),  # tensor(n_cam3d, 3, H, W)
            "depth": self._get_depth(idx),  # tensor(n_cam3d, H, W)
            "rgb2d": self._get_rgb2d(idx),  # tensor(n_cam2d, 3, H, W)
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx),  # tensor(T, 8)
            "extrinsics": self._get_extrinsics(idx),  # tensor(n_cam3d, 4, 4)
            "intrinsics": self._get_intrinsics(idx)  # tensor(n_cam3d, 3, 3)
        }


class HiveformerDataset(RLBenchDataset):
    cameras = ("wrist", "front")
    camera_inds = None
    train_copies = 100
    camera_inds2d = None

    def _load_instructions(self, instruction_file):
        instr = json.load(open(instruction_file))
        self.tasks = list(instr.keys())
        return instr


class PeractDataset(RLBenchDataset):
    """RLBench dataset under Peract setup."""
    tasks = PERACT_CUPBOARD_TASKS
    cameras = ("left_shoulder", "right_shoulder", "wrist", "front")
    camera_inds = None
    train_copies = 10
    camera_inds2d = None

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
            task_id: (N,) uint8
            variation: (N,) uint8
            extrinsics: (N, n_cam, 4, 4) float
            intrinsics: (N, n_cam, 3, 3) float
        }
        """
        # First detect which copy we fall into
        idx = idx % (len(self.annos['action']) // self.chunk_size)
        # and then which chunk
        idx = idx * self.chunk_size
        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            "task": self._get_task(idx),  # [str]
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_rgb(idx),  # tensor(n_cam3d, 3, H, W)
            "pcd": self._get_attr_by_idx(idx, 'pcd', True),  # tensor(n_cam3d, H, W)
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx),  # tensor(T, 8)
        }


class PeractTwoCamDataset(PeractDataset):
    """RLBench dataset under Peract setup."""
    tasks = PERACT_CUPBOARD_TASKS
    cameras = ("wrist", "front")
    camera_inds = [2, 3]
    train_copies = 10
    camera_inds2d = None


class Peract2Dataset(RLBenchDataset):
    """RLBench dataset under Peract2 setup."""
    tasks = PERACT2_TASKS
    cameras = ("front", "wrist_left", "wrist_right")
    camera_inds = None
    train_copies = 10
    camera_inds2d = None


class Peract2SingleCamDataset(RLBenchDataset):
    """RLBench dataset under Peract2 setup."""
    tasks = PERACT2_TASKS
    cameras = ("front",)
    camera_inds = (0,)  # use only front camera
    train_copies = 10
    camera_inds2d = None
