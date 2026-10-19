"""
Train the grasp or place policy.
"""

import argparse
import os
from pathlib import Path
import sys

import torch
from torch import nn
import wandb

from disco.flowmatch.utils.common_utils import str2bool, str_none

# Train Tester:
from disco.flowmatch.train_tester import BaseTrainTester

# Dataset (2-pose grasp+place variant):
from disco.flowmatch.datasets.shelf_packing_grasp_place_dataset import (
    ShelfPackingGraspPlaceDataset,
    ROBOT_BASE_X_OFFSET,
)
from disco.flowmatch.datasets.sim2real_grasp_place_dataset import Sim2RealShelfPackingGraspPlaceDataset

# Model (2-pose grasp+place variant):
from disco.flowmatch.modeling.policy_grasp_place.denoise_actor_3d_packing import DenoiseActor


# Keys we want to keep in wandb plots — the loss-style metrics only.
# `compute_metrics` also produces accuracy and gripper metrics; those are
# noisy or meaningless for the 2-pose policy (gripper is hardcoded), so we
# filter them out before logging.
_WANDB_LOSS_SUBSTRINGS = ("traj_pos_l2", "traj_rot_l1")


def _filter_to_losses(metrics):
    return {k: v for k, v in metrics.items() if any(s in k for s in _WANDB_LOSS_SUBSTRINGS)}


class TrainTester(BaseTrainTester):
    """Three overrides relative to BaseTrainTester:
      * `evaluate_nsteps` filters wandb metrics down to the loss keys.
      * `get_workspace_normalizer` fits min/max from the training data
        instead of falling through to the hardcoded `scene_bounds`
        (the data-fit branch in BaseTrainTester is commented out).
      * `get_model` plumbs `--pcd_input_channels` through so the
        target-only data variant (3-channel xyz) can be trained without
        affecting the existing 4-channel masked variant. Also sanity-
        checks that the configured channel count matches the loaded
        training file.
    """

    def _gt_slice_for_mode(self, gt_action):
        """Slice the full (B, 2, 8) GT to match the model's output shape.

        grasp mode predicts pose index 0 only → (B, 1, 8).
        place mode predicts pose index 1 only → (B, 1, 8).
        joint mode predicts both             → (B, 2, 8) unchanged.
        """
        mode = self.args.mode
        if mode == "joint":
            return gt_action
        if mode == "grasp":
            return gt_action[:, 0:1, :]
        if mode == "place":
            return gt_action[:, 1:2, :]
        raise ValueError(f"Unknown mode {mode!r}")

    @torch.no_grad()
    def evaluate_nsteps(self, model, loader, step_id, val_iters, split='val'):
        from disco.flowmatch.utils.trainers.utils import compute_metrics
        import os
        import numpy as np

        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in enumerate(loader):
            if i == val_iters:
                break

            pred_action = self._model_forward(sample, training=False)
            gt_action = sample["action"].cuda(non_blocking=True)
            gt_action = self._gt_slice_for_mode(gt_action)

            losses, losses_B = compute_metrics(pred_action, gt_action)

            if (step_id + 1) % self.args.vis_freq == 0 and i == 0:
                vis_dir = os.path.join(self.args.log_dir, "vis")
                os.makedirs(vis_dir, exist_ok=True)
                idx = torch.randint(0, pred_action.shape[0], (1,)).item()
                filename = os.path.join(vis_dir, f"step_{(step_id + 1)}.npz")
                np.savez_compressed(
                    filename,
                    pcd=sample["pcd"][idx].cpu().numpy(),
                    pred=pred_action[idx].cpu().numpy(),
                    gt=gt_action[idx].cpu().numpy(),
                    step_id=(step_id + 1),
                )

            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

        values = {k: v.mean().item() for k, v in values.items()}
        print(f"Step {step_id}:")
        for key, value in values.items():
            print(f"{key}: {value:.03f}")

        return _filter_to_losses(values)

    def _mode_kwargs(self):
        """Translate `--mode {joint,grasp,place}` to DenoiseActor kwargs."""
        mode = self.args.mode
        if mode == "joint":
            return {"trajectory_length": 2, "grasp_condition_dim": None}
        if mode == "grasp":
            return {"trajectory_length": 1, "grasp_condition_dim": None}
        if mode == "place":
            return {"trajectory_length": 1, "grasp_condition_dim": 7}
        raise ValueError(f"--mode must be one of joint|grasp|place; got {mode!r}")

    def _pose_slice_for_mode(self):
        """Which pose index/indices does this mode predict?
        joint → [0,1], grasp → [0], place → [1]. Used by the workspace-
        normalizer fit so we only fit on the pose(s) this model predicts."""
        mode = self.args.mode
        if mode == "joint":
            return [0, 1]
        if mode == "grasp":
            return [0]
        if mode == "place":
            return [1]
        raise ValueError(f"--mode must be one of joint|grasp|place; got {mode!r}")

    def get_model(self):
        # Sanity-check that the data on disk has the channel count the
        # model is being asked to consume — mismatched values produce
        # silent shape errors deep in scene_pos_to_feat at first batch.
        data = torch.load(self.args.train_data_dir, map_location="cpu")
        data_channels = int(data["input_pcd"].shape[-1])
        if data_channels != int(self.args.pcd_input_channels):
            raise ValueError(
                f"--pcd_input_channels={self.args.pcd_input_channels} but "
                f"train_data_dir has {data_channels}-channel input_pcd. "
                "Pass --pcd_input_channels 3 for the target-only dataset, 4 "
                "for the masked dataset."
            )
        del data

        self.model = self.model_cls(
            embedding_dim=self.args.embedding_dim,
            num_attn_heads=self.args.num_attn_heads,
            num_shared_attn_layers=self.args.num_shared_attn_layers,
            relative=self.args.relative_action,
            rotation_format=self.args.rotation_format,
            denoise_timesteps=self.args.denoise_timesteps,
            denoise_model=self.args.denoise_model,
            lv2_batch_size=self.args.lv2_batch_size,
            pcd_input_channels=int(self.args.pcd_input_channels),
            **self._mode_kwargs(),
        )

        from disco.flowmatch.utils.common_utils import count_parameters
        count_parameters(self.model)
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.ndim > 1 and not param.is_contiguous():
                print(f"Fixing layout for: {name}")
                param.data = param.contiguous()
        return self.model

    def get_workspace_normalizer(self, ndims=3):
        """Per-pose min/max of positions in robot-base frame, sliced by mode.

        Returns a `(2, L, ndims)` parameter where L matches the model's
        `trajectory_length` (1 or 2). Per-pose fitting matches the diffusion
        noise scale to each pose's actual range — pickup-region and shelf-
        interior have very different per-axis spreads.

        The dataset's `__getitem__` adds ROBOT_BASE_X_OFFSET to the x
        channel before handing samples to the trainer; we mirror that
        shift here so the normalizer is computed in the same frame the
        model trains in.

        When `--sim2real` is set, the dataloader already centers and scales
        every sample into a unit ball, so the normalizer is forced to the
        identity range [-1, 1] (per pose, per ndim) and no data fit runs.
        """
        if getattr(self.args, "sim2real", False):
            pose_slice = self._pose_slice_for_mode()
            L = len(pose_slice)
            print(f"[grasp_place] sim2real on → workspace_normalizer = "
                  f"[-1, 1] for L={L} pose(s), ndims={ndims}")
            min_ = torch.full((L, ndims), -1.0)
            max_ = torch.full((L, ndims), 1.0)
            return nn.Parameter(
                torch.stack([min_, max_]).float(),    # (2, L, ndims)
                requires_grad=False,
            )

        print(f"[grasp_place] Fitting per-pose workspace_normalizer "
              f"(mode={self.args.mode}) from training data...")

        data = torch.load(self.args.train_data_dir, map_location="cpu")
        if "goal_pose" not in data:
            raise KeyError(
                f"{self.args.train_data_dir} has no 'goal_pose' field "
                f"(got {list(data.keys())})."
            )

        pose_slice = self._pose_slice_for_mode()
        # data["goal_pose"]: (N, 2, 8) → slice to (N, len(pose_slice), ndims)
        positions = data["goal_pose"][:, pose_slice, :ndims].clone().float()
        positions[..., 0] = positions[..., 0] + ROBOT_BASE_X_OFFSET

        buf = float(self.args.workspace_normalizer_buffer)
        # Min/max along the sample axis only — keeps per-pose dim.
        min_ = positions.min(dim=0).values - buf   # (L, ndims)
        max_ = positions.max(dim=0).values + buf   # (L, ndims)

        n, L, _ = positions.shape
        print(f"[grasp_place] data-fit normalizer (n={n} samples, L={L}, buffer={buf}):")
        full_names = {0: "grasp", 1: "place"}
        names = [full_names[i] for i in pose_slice]
        for i in range(L):
            print(
                f"  {names[i]:>5s}: "
                f"min={[round(v, 4) for v in min_[i].tolist()]}  "
                f"max={[round(v, 4) for v in max_[i].tolist()]}"
            )

        return nn.Parameter(
            torch.stack([min_, max_]).float(),     # (2, L, ndims)
            requires_grad=False,
        )


# Helper function to find run ID by name
def find_run_id(project_name, run_name):
    api = wandb.Api()
    try:
        runs = api.runs(project_name)
        for run in runs:
            if run.name == run_name:
                return run.id
    except (ValueError, wandb.errors.CommError):
        pass
    return None

def start_wandb_run(args):
    if not args.wandb:
        wandb.init(mode="disabled")
        return
    run_name = args.wandb_run_name
    run_id = find_run_id(args.wandb_project_name, run_name)
    if run_id:
        print(f"Resuming run: {run_name} (ID: {run_id})")
        wandb.init(project=args.wandb_project_name, id=run_id, resume="must")
    else:
        print(f"Creating new run: {run_name}")
        wandb.init(project=args.wandb_project_name, name=run_name)
    # Save args to wandb
    wandb.config.update(args, allow_val_change=True)

def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Tuples: (name, type, default)
    # Target-only variant (3-channel xyz, other non-shelved objects removed
    # from the PCD before sampling). Generated by
    # `save_successful_grasp_place_target_only.py --all_envs`.
    data_path = 'assets/processed_data/successful_grasp_place_target_only.pth'
    arguments = [
        # Dataset/loader arguments
        ('wandb', str2bool, False),
        ('wandb_project_name', str, "Value_Function_Planning"),
        # `__MODE__` is a sentinel that gets replaced post-parse with
        # the actual mode (grasp/place/joint), so the same command line
        # produces a sensibly named run regardless of --mode. Override
        # this to a literal string to opt out of the auto-naming.
        ('wandb_run_name', str, "grasp_place_cascaded___MODE___v3"),
        ('train_data_dir', Path, data_path),
        # num_workers=1: post-deepcopy fix the per-sample work is ~0.2 ms,
        # so multi-worker IPC adds more overhead than it saves. 1 worker is
        # enough to overlap data loading with the previous step's compute
        # via prefetch_factor=2.
        ('num_workers', int, 1),
        ('batch_size', int, 32),
        ('batch_size_val', int, 32),
        ('chunk_size', int, 1),
        ('memory_limit', float, 8),  # cache limit in GB
        # Logging arguments
        # ('base_log_dir', Path, Path(__file__).parent / "train_logs"),
        ('base_log_dir', Path, "models/flowmatch_actor/train_logs"),
        # Resume-from-last default. Points at this run's own `last.pth`
        # so re-running the same command auto-resumes; if the file
        # doesn't exist (first launch) the trainer prints a warning and
        # starts from scratch (load_checkpoint handles missing-file).
        # The `__MODE__` sentinel is replaced post-parse just like
        # `wandb_run_name`. Pass `--checkpoint null` to force from-scratch.
        ('checkpoint', str_none,
         "models/flowmatch_actor/"
         "train_logs/Value_Function_Planning/grasp_place_cascaded___MODE___v2/last.pth"),
        # val_freq 100 → 1000 (2026-05-04). Train-eval is expensive
        # (val_iters batches × denoise_timesteps inference passes); doing
        # it less often + with fewer batches recovers most of the wall
        # time. Per-step train loss is now logged separately by the
        # trainer for tighter optimizer feedback.
        ('val_freq', int, 1000),
        # val_iters 10 → 2 (2026-05-04): cap eval cost. The metric is
        # noisier per call, but with val_freq=1000 the chart updates
        # only ~7×/h anyway and the optimizer-side signal lives in
        # `train/step_loss`.
        ('val_iters', int, 2),
        ('vis_freq', int, 1000),            # NOTE: Should be a multiple of val_freq
        ('interm_ckpt_freq', int, 3000),       # NOTE: Should be a multiple of val_freq
        ('eval_only', str2bool, False),
        ('start_iter', int, None),
        ('lr', float, 1e-4),
        ('backbone_lr', float, 1e-4),
        ('lr_scheduler', str, "constant"),
        ('wd', float, 5e-3),
        ('train_iters', int, 600000),
        ('use_compile', str2bool, False),
        ('use_ema', str2bool, False),
        ('lv2_batch_size', int, 1),
        # Model arguments: general policy type
        ('model_type', str, 'denoise3d'),       # !!! TODO: Change here for using different models
        ('bimanual', str2bool, False),
        ('keypose_only', str2bool, True),
        ('pre_tokenize', str2bool, True),
        ('custom_img_size', int, None),
        ('workspace_normalizer_buffer', float, 0.04),
        # Model arguments: encoder
        ('backbone', str, "clip"),
        ('finetune_backbone', str2bool, False),
        ('finetune_text_encoder', str2bool, False),
        ('fps_subsampling_factor', int, 5),
        # Model arguments: encoder and head
        ('embedding_dim', int, 120),  # divisible by num_attn_heads
        ('num_attn_heads', int, 8),
        ('num_vis_instr_attn_layers', int, 3),
        ('num_history', int, 1),
        # Model arguments: head
        ('num_shared_attn_layers', int, 4),
        ('relative_action', str2bool, False),
        ('rotation_format', str, 'quat_wxyz'),
        # Bumped 10 -> 30 (2026-05-04). RF is step-count-agnostic at
        # inference, so 30 also applies to MCTS sampling without retraining.
        # Trade-off: 3x more transformer forwards per inference call.
        ('denoise_timesteps', int, 30),
        ('denoise_model', str, "rectified_flow"),
        # Width of the input fed to the head's `scene_pos_to_feat` linear.
        # 4 = legacy masked PCD (xyz + target_mask). 3 = target-only PCD
        # (xyz only; every other non-shelved object removed before
        # sampling). Must match the channel count of the loaded
        # `train_data_dir` .pth — `TrainTester.get_model` asserts this.
        # Default 3 matches the default `train_data_dir` (target-only).
        ('pcd_input_channels', int, 3),
        # Cascaded grasp→place mode (added 2026-05-04):
        #   joint  → legacy 2-pose model (predicts grasp+place jointly)
        #   grasp  → single-pose model that predicts only the grasp
        #   place  → single-pose model that predicts only the place,
        #            conditioned on the GT grasp (teacher forcing) at
        #            training time. At inference the place model takes
        #            the clean grasp from the trained grasp model.
        # Both single-pose modes use the same `goal_pose` field of the
        # dataset; the training-time slicing happens inside DenoiseActor.
        ('mode', str, 'grasp'),
        # Sim2real augmentation: when set, the dataset wraps the same .pth
        # in `Sim2RealShelfPackingGraspPlaceDataset`, which adds noise +
        # holes + per-part SE(3) and centers/scales each sample into a
        # unit ball. The trainer also forces the workspace_normalizer to
        # the identity [-1, 1] range. Requires `--env_root`.
        ('sim2real', str2bool, False),
        ('env_root', Path,
         "assets/environments/shelf_packing/train"),
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def suppress_output_on_non_main():
    if int(os.environ.get("RANK", 0)) != 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Arguments
    args = parse_arguments()
    print("Arguments:")
    print(args)
    print("-" * 100)

    # Resolve the __MODE__ sentinel in wandb_run_name + checkpoint so
    # the defaults pick up the right per-mode names automatically.
    # Users can opt out by passing literal strings without "__MODE__".
    if "__MODE__" in args.wandb_run_name:
        args.wandb_run_name = args.wandb_run_name.replace("__MODE__", args.mode)
    if args.checkpoint is not None and "__MODE__" in str(args.checkpoint):
        args.checkpoint = str(args.checkpoint).replace("__MODE__", args.mode)

    log_dir = args.base_log_dir / args.wandb_project_name / args.wandb_run_name
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count:", torch.cuda.device_count())
    # args.local_rank = int(os.environ["LOCAL_RANK"])
    suppress_output_on_non_main()

    # DDP initialization (NOTE: !!! Disabling Distributed Training for now)
    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    # Select dataset and model classes
    # dataset_class = fetch_dataset_class(args.dataset)
    # model_class = fetch_model_class(args.model_type)

    if args.sim2real:
        # Bind env_root so BaseTrainTester can keep its (root=, relative_action=,
        # mem_limit=) instantiation signature. Functools.partial would also
        # work; a closure is the same thing in fewer characters.
        env_root_for_dataset = args.env_root
        def dataset_cls(**kwargs):
            return Sim2RealShelfPackingGraspPlaceDataset(
                env_root=env_root_for_dataset,
                **kwargs,
            )
        print(f"[grasp_place] sim2real=True; using "
              f"Sim2RealShelfPackingGraspPlaceDataset with env_root={env_root_for_dataset}")
    else:
        dataset_cls = ShelfPackingGraspPlaceDataset

    train_tester = TrainTester(
        args=args,
        dataset_cls=dataset_cls,
        model_cls=DenoiseActor
    )

    start_wandb_run(args)
    train_tester.main()

    # Safe program termination
    if torch.distributed.is_initialized():
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
