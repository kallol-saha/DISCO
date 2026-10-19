"""
Train the value network.
"""

import argparse
import os
from pathlib import Path
import sys

import torch
import wandb

from disco.flowmatch.utils.common_utils import str2bool, str_none

# Train Tester:
from disco.flowmatch.train_tester_value import BaseTrainTester as TrainTester

# Dataset:
from disco.flowmatch.datasets.shelf_packing_value_dataset import ShelfPackingValueDataset
from disco.flowmatch.datasets.sim2real_value_dataset import Sim2RealShelfPackingValueDataset

# Model:
from disco.flowmatch.modeling.policy.value_network import ValueNetwork


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
    # Save args to wandb. allow_val_change=True so a resumed run tolerates
    # CLI/script default changes between attempts (e.g., bumped batch_size
    # after an OOM crash).
    wandb.config.update(args, allow_val_change=True)

def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Tuples: (name, type, default)
    data_path = 'assets/processed_data/q_value_data_grasp_place_phase2.pth'
    resume_ckpt = 'models/flowmatch_actor/train_logs/Value_Function_Planning/q_function_run_2/best.pth'
    arguments = [
        # Dataset/loader arguments
        ('wandb', str2bool, False),
        ('wandb_project_name', str, "Value_Function_Planning"),
        ('wandb_run_name', str, "q_function_pointwise_phase2_resume"),
        ('train_data_dir', Path, data_path),
        ('num_workers', int, 4),
        # Pointwise BCE — one (state, action) pair per row, no K dim,
        # so we can run the original batch size again (listwise had
        # forced this to 4 because of B × K_max effective batches).
        ('batch_size', int, 32),
        ('batch_size_val', int, 32),
        ('chunk_size', int, 1),
        ('memory_limit', float, 8),  # cache limit in GB
        # Logging arguments
        # ('base_log_dir', Path, Path(__file__).parent / "train_logs"),
        ('base_log_dir', Path, "models/flowmatch_actor/train_logs"),
        # Training and testing arguments — resume from the pre-listwise
        # pointwise BCE checkpoint by default. Pass `--checkpoint null`
        # (or a non-existent path) to train from scratch.
        ('checkpoint', str_none, resume_ckpt),
        ('val_freq', int, 20),
        ('vis_freq', int, 1000),            # NOTE: Should be a multiple of val_freq
        ('interm_ckpt_freq', int, 3000),       # NOTE: Should be a multiple of val_freq
        ('eval_only', str2bool, False),
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
        ('denoise_timesteps', int, 10),
        ('denoise_model', str, "rectified_flow"),
        # Sim2real augmentation: when set, the dataset wraps the same .pth
        # in `Sim2RealShelfPackingValueDataset`, which adds noise + holes +
        # per-part SE(3) and centers/scales each sample into a unit ball.
        # ValueNetwork does not consult workspace_normalizer internally, so
        # no normalizer override is required here. Requires `--env_root`.
        ('sim2real', str2bool, False),
        ('env_root', Path,
         "assets/environments/shelf_packing/train"),
        # Override checkpoint's stored iter. -1 = use ckpt value (default).
        ('start_iter', int, -1),
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
        env_root_for_dataset = args.env_root
        def dataset_cls(**kwargs):
            return Sim2RealShelfPackingValueDataset(
                env_root=env_root_for_dataset,
                **kwargs,
            )
        print(f"[value] sim2real=True; using Sim2RealShelfPackingValueDataset "
              f"with env_root={env_root_for_dataset}")
    else:
        dataset_cls = ShelfPackingValueDataset

    train_tester = TrainTester(
        args=args,
        dataset_cls=dataset_cls,
        model_cls=ValueNetwork
    )

    start_wandb_run(args)
    train_tester.main()

    # Safe program termination
    if torch.distributed.is_initialized():
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
