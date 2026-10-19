from copy import deepcopy
import os
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
import wandb

from disco.flowmatch.modeling.encoder.text import fetch_tokenizers
from disco.flowmatch.utils.common_utils import count_parameters
from disco.flowmatch.utils.depth2cloud import fetch_depth2cloud
from disco.flowmatch.utils.data_preprocessors import fetch_data_preprocessor
from disco.flowmatch.utils.ema import EMA
from disco.flowmatch.utils.schedulers import fetch_scheduler
from disco.flowmatch.utils.trainers.utils import compute_metrics, visualize_pred_gt


class BaseTrainTester:
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args, dataset_cls, model_cls):
        """Initialize."""
        self.args = args
        self.dataset_cls = dataset_cls
        self.model_cls = model_cls

        # self.preprocessor = fetch_data_preprocessor(self.args.dataset)(
        #     self.args.keypose_only,
        #     self.args.num_history,
        #     custom_imsize=self.args.custom_img_size,
        #     depth2cloud=fetch_depth2cloud(self.args.dataset)
        # )

        # TODO: This has to be replaced with wandb
        if torch.distributed.is_initialized() and dist.get_rank() == 0 and not self.args.eval_only:
            self.writer = SummaryWriter(log_dir=args.log_dir)

        self.get_loaders()
        self.get_model()

        # TODO: Can this be automated?
        self.scene_bounds = torch.tensor([-1.0, -0.8, -0.1, 
                             1.,  0.8,  1.0])        # [x_min, y_min, z_min, x_max, y_max, z_max]

    def get_loaders(self):

        """Initialize data loaders."""

        # Function to seed the random number generator for each worker
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Initialize the training dataset
        self.train_dataset = self.dataset_cls(
            root=self.args.train_data_dir,
            relative_action=self.args.relative_action,
            mem_limit=self.args.memory_limit,
        )

        # Initialize the random number generator for torch data loader
        g = torch.Generator()
        g.manual_seed(0)

        # Initialize the training data loader
        self.train_loader = DataLoader(
            self.train_dataset,                                              # The torch dataset object containing the data
            batch_size=self.args.batch_size,    # The training batch size
            shuffle=True,                                               # Whether to shuffle the data each epoch
            num_workers=self.args.num_workers,                          # Number of CPU subprocesses to use for data loading
            worker_init_fn=seed_worker,                                 # Function to seed the random number generator for each worker
            collate_fn=base_collate_fn,                                 # Per-sample value records → flat (B, ...) tensors via torch.cat.
            pin_memory=True,                                            # If True, the DataLoader will copy tensors into CUDA pinned memory before returning them
            drop_last=len(self.train_dataset) >= self.args.batch_size,  # keep a partial batch when the data is smaller than one batch
            generator=g,                                                # The random number generator to use for the data loading
            prefetch_factor=4,                                          # The number of batches to prefetch from the data loader
            persistent_workers=True                                     # Whether to keep the workers alive after the data loading is complete
        )
       
        return self.train_loader

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments (ValueNetwork specific)
        self.model = self.model_cls(
            embedding_dim=self.args.embedding_dim,
            num_attn_heads=self.args.num_attn_heads,
            num_shared_attn_layers=self.args.num_shared_attn_layers,
        )

        # Print basic modules' parameters
        count_parameters(self.model)

        # Useful for some models to ensure parameters are contiguous
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.ndim > 1 and not param.is_contiguous():
                print(f"Fixing layout for: {name}")
                param.data = param.contiguous()

        return self.model

    @torch.no_grad()
    def get_workspace_normalizer(self, ndims=3):
        """
        Computes workspace bounds (min/max) for position normalization by iterating through
        all training actions and finding the minimum and maximum values across the first `ndims`
        dimensions (typically 3 for x, y, z position coordinates).
        
        The bounds are extended by a buffer margin (workspace_normalizer_buffer) to ensure
        workspace coverage during training/inference. The returned normalizer can be used to
        normalize 3D positions to a standard range (typically [-1, 1]) for neural network training.
        
        Args:
            ndims: Number of dimensions to compute bounds for (default: 3 for x, y, z positions)
        
        Returns:
            nn.Parameter: A 2xndims tensor where:
                - First row contains minimum values for each dimension (with buffer subtracted)
                - Second row contains maximum values for each dimension (with buffer added)
                Shape: (2, ndims), requires_grad=False
        """
        # print("Computing workspace normalizer...")

        # # Initialize datasets with arguments
        # train_dataset = self.dataset_cls(
        #     root=self.args.train_data_dir,
        #     instructions=self.args.train_instructions,
        #     copies=1,
        #     relative_action=self.args.relative_action,
        #     mem_limit=0.1,
        #     actions_only=True,
        #     chunk_size=self.args.chunk_size
        # )

        # data_loader = DataLoader(
        #     train_dataset,
        #     batch_size=max(self.args.batch_size, 64) // self.args.chunk_size,
        #     collate_fn=actions_collate_fn,
        #     shuffle=False,
        #     num_workers=self.args.num_workers
        # )

        # # Loop and compute action min-max
        # min_, max_ = torch.ones(ndims) * 10000, -torch.ones(ndims) * 10000
        # for sample in tqdm(data_loader):
        #     action = sample["action"][..., :ndims].reshape([-1, ndims])
        #     min_ = torch.min(min_, action.min(0).values)
        #     max_ = torch.max(max_, action.max(0).values)

        # min_ = min_ - self.args.workspace_normalizer_buffer
        # max_ = max_ + self.args.workspace_normalizer_buffer

        # This is a (2, 3) shape tensor for min and max values for x, y, z positions of the workspace
        return nn.Parameter(
            torch.stack([
                self.scene_bounds[:3], 
                self.scene_bounds[3:]
            ]),
            requires_grad=False
        )

    def get_optimizer(self):
        """
        Initializes an AdamW optimizer with separate parameter groups for different types of parameters:
        
        - Group 0: Parameters that do not require gradients (e.g., frozen weights)
        - Group 1: Parameters that require gradients and are not part of the backbone (e.g., model weights)
        - Group 2: Parameters that require gradients and are part of the backbone (e.g., backbone weights)
        
        The optimizer is configured with specified learning rates and weight decay rates for each group.
        """
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.args.lr},
            {"params": [], "weight_decay": self.args.wd, "lr": self.args.lr}
        ]
        if self.args.finetune_backbone:
            optimizer_grouped_parameters.append({
                "params": [], "weight_decay": self.args.wd,
                "lr": self.args.backbone_lr
            })

        # Collect names of all norm parameters
        norm_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LocalResponseNorm,
            torch.nn.RMSNorm
        )
        norm_param_names = set()
        for module_name, module in self.model.named_modules():
            if isinstance(module, norm_types):
                for param_name, _ in module.named_parameters(recurse=False):
                    norm_param_names.add(f"{module_name}.{param_name}")

        # Now split parameters based on name
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name in norm_param_names or name.endswith(".bias"):
                optimizer_grouped_parameters[0]["params"].append(param)
            elif self.args.finetune_backbone and 'backbone' in name:
                optimizer_grouped_parameters[2]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.95)
        )
        return self.optimizer

    def main(self):
        """Run main training/testing pipeline."""

        # self.tokenizer = fetch_tokenizers(self.args.backbone)
        # ValueNetwork doesn't need workspace_normalizer

        # Initialize the optimizer
        self.optimizer = self.get_optimizer()
        lr_scheduler = fetch_scheduler(
            self.args.lr_scheduler, self.optimizer, self.args.train_iters
        )
        scaler = torch.GradScaler()

        # Move model to devices
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        # ValueNetwork doesn't have compute_loss method, forward handles both training and inference

        # Initialize EMA copy
        ema_model = deepcopy(self.model)
        self.ema = EMA()

        # Check for a checkpoint
        start_iter, best_loss = 0, None
        if self.args.checkpoint:
            start_iter, best_loss = self.load_checkpoint(self.model, ema_model, self.optimizer)
        if getattr(self.args, "start_iter", -1) >= 0:
            print(f"=> overriding start_iter {start_iter} -> {self.args.start_iter}")
            start_iter = self.args.start_iter

        # Eval only
        if self.args.eval_only:
            print("Test evaluation.......")
            self.model.eval()
            # Use train_loader for evaluation if val_loader doesn't exist
            eval_loader = getattr(self, 'val_loader', None) or self.train_loader
            self.evaluate_nsteps(
                ema_model if self.args.use_ema else self.model,
                eval_loader, step_id=-1,
                val_iters=-1
            )
            return ema_model if self.args.use_ema else self.model

        # Step the lr scheduler to the current step
        for _ in range(start_iter):
            lr_scheduler.step()

        # Step the sampler to the currect "epoch"
        samples_per_epoch = len(self.train_loader)
        epoch = start_iter // samples_per_epoch + 1

        # Training loop
        self.model.train()
        iter_loader = iter(self.train_loader)
        for step_id in trange(start_iter, self.args.train_iters):
            try:
                sample = next(iter_loader)
            except StopIteration:
                # when the iterator is exhausted, we need to reset it
                # and increment the epoch
                epoch += 1
                iter_loader = iter(self.train_loader)
                sample = next(iter_loader)

            self.train_one_step(scaler, lr_scheduler, sample)
            self.ema.step(self.model, ema_model, self.args.use_ema, step_id)

            if (step_id + 1) % self.args.val_freq == 0:

                # !!! NOTE This is where evaluation happens
                print("Train evaluation.......")

                self.model.eval()

                metrics = self.evaluate_nsteps(
                    ema_model if self.args.use_ema else self.model,
                    self.train_loader, step_id,
                    val_iters=10,
                    split='train'
                )

                new_loss = metrics['train-losses/mean/value_mse']

                wandb.log(metrics)

                # save model
                best_loss = self.save_checkpoint(
                    self.model, ema_model, self.optimizer, step_id,
                    new_loss, best_loss
                )
                self.model.train()

        return ema_model if self.args.use_ema else self.model

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        """Prepare a pointwise batch for ValueNetwork.

        Returns: (pcd, actions, targets, n_leaves).
        targets / n_leaves may be None in inference mode.
        """
        return (
            sample["pcd"],
            sample["actions"],
            sample.get("targets", None),
            sample.get("n_leaves", None),
        )

    def _model_forward(self, sample, training=True):
        """Pointwise forward pass with per-batch weighted-mean BCE."""
        pcd, actions, targets, n_leaves = self.prepare_batch(sample, augment=training)
        pcd = pcd.cuda(non_blocking=True).float()
        actions = actions.cuda(non_blocking=True).float()

        if training:
            if targets is None:
                raise ValueError("targets are required for training")
            targets = targets.cuda(non_blocking=True).float()
            if n_leaves is not None:
                n_leaves = n_leaves.cuda(non_blocking=True).float()
        else:
            targets = None
            n_leaves = None

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model(
                pcd=pcd,
                actions=actions,
                targets=targets,
                n_leaves=n_leaves,
            )
        return out  # scalar weighted-BCE loss if training, else (B,) sigmoid scores

    def train_one_step(self, scaler, lr_scheduler, sample):
        """Run a single training step."""
        self.optimizer.zero_grad()

        # Forward pass
        loss = self._model_forward(sample)

        # Backward pass
        scaler.scale(loss).backward()

        # Clip gradients
        scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

        # Update
        scaler.step(self.optimizer)
        scaler.update()

        # Step the lr scheduler
        lr_scheduler.step()

    @torch.inference_mode()
    def evaluate_nsteps(self, model, loader, step_id, val_iters, split='val'):
        """Pointwise eval: per-sample MSE/MAE on the predicted Q̂ vs target."""
        metrics_dict = {}
        model.eval()

        for i, sample in tqdm(enumerate(loader)):
            if i == val_iters:
                break

            pred_scores = self._model_forward(sample, training=False)  # (B,)
            gt_targets = sample.get("targets", None)
            if gt_targets is None:
                print("Warning: missing targets in sample, skipping metrics")
                continue

            gt_targets = gt_targets.cuda(non_blocking=True).float()
            diff = pred_scores.float() - gt_targets
            mse = diff.pow(2).mean()
            mae = diff.abs().mean()

            for key, val in [
                (f"{split}-losses/mean/value_mse", mse.item()),
                (f"{split}-losses/mean/value_mae", mae.item()),
            ]:
                metrics_dict.setdefault(key, []).append(val)

            if (step_id + 1) % self.args.vis_freq == 0 and i == 0:
                B = sample["pcd"].shape[0]
                idx = torch.randint(0, B, (1,)).item()

                vis_dir = os.path.join(self.args.log_dir, "vis")
                os.makedirs(vis_dir, exist_ok=True)

                pcd_np = sample["pcd"][idx].cpu().detach().float().numpy()
                pred_np = pred_scores[idx].cpu().detach().float().numpy()
                gt_np = gt_targets[idx].cpu().detach().float().numpy()

                filename = os.path.join(vis_dir, f"step_{(step_id + 1)}.npz")
                np.savez_compressed(
                    filename,
                    pcd=pcd_np,
                    pred_score=pred_np,
                    gt_target=gt_np,
                    step_id=(step_id + 1),
                )
                print(f"Saved visualization data to: {filename}")
                print(f"  PCD shape: {pcd_np.shape}")
                print(f"  Pred score: {float(pred_np):.4f}  GT target: {float(gt_np):.4f}")

        # Average metrics
        values = {k: np.mean(v) for k, v in metrics_dict.items()}
        
        # TODO: Uncomment for distributed training
        # if dist.get_rank() == 0:
        # if step_id > -1:
        #     for key, val in values.items():
        #         self.writer.add_scalar(key, val, step_id)

        # Also log to terminal
        print(f"Step {step_id}:")
        for key, value in values.items():
            print(f"{key}: {value:.06f}")

        return values

    def load_checkpoint(self, model, ema_model, optimizer):
        """Load from checkpoint.

        Two modes, distinguished by whether `--checkpoint` is inside the
        current run's `log_dir`:

        - **Same-run resume** (e.g. auto-restart after a crash, ckpt is
          `<log_dir>/best.pth` or `<log_dir>/last.pth`): preserves
          `iter`, `best_loss`, and optimizer state so training continues
          where it left off.
        - **Cross-run warm-start** (ckpt is in a *different* run dir):
          loads weights + EMA only; resets `iter=0`, `best_loss=None`,
          and skips optimizer state. The new run starts fresh in its
          own log_dir, just initialized from somebody else's weights.
        """
        print("=> trying checkpoint '{}'".format(self.args.checkpoint))
        if not os.path.exists(self.args.checkpoint):
            print('Warning: checkpoint was not found, starting from scratch')
            print('The main process will compute workspace bounds')
            return 0, None

        ckpt_path = os.path.realpath(self.args.checkpoint)
        log_dir = os.path.realpath(str(self.args.log_dir))
        same_run = os.path.dirname(ckpt_path) == log_dir
        mode = "same-run resume" if same_run else "cross-run warm-start"
        print(f"=> load mode: {mode}")

        model_dict = torch.load(
            self.args.checkpoint,
            map_location="cpu",
            # weights_only=False because old checkpoints stored
            # `best_loss` as a numpy scalar, which the post-2.6 safe
            # unpickler refuses. We trust our own checkpoints.
            weights_only=False
        )
        # Load weights flexibly
        msn, unxpct = model.load_state_dict(model_dict["weight"], strict=False)
        if msn:
            print(f"Missing keys (not found in checkpoint): {len(msn)}")
            print(msn)
        if unxpct:
            print(f"Unexpected keys (ignored): {len(unxpct)}")
            print(unxpct)
        if not msn and not unxpct:
            print("All keys matched successfully!")
        # EMA weights
        if model_dict.get("ema_weight") is not None:
            ema_model.load_state_dict(model_dict["ema_weight"], strict=True)

        if same_run:
            if 'optimizer' in model_dict and not self.args.eval_only:
                optimizer.load_state_dict(model_dict["optimizer"])
            start_iter = model_dict.get("iter", 0)
            best_loss = model_dict.get("best_loss", None)
        else:
            start_iter = 0
            best_loss = None

        print("=> loaded successfully '{}' (resume from step {})".format(
            self.args.checkpoint, start_iter
        ))
        del model_dict
        torch.cuda.empty_cache()
        return start_iter, best_loss

    def save_checkpoint(self, model, ema_model, optimizer,
                        step_id, new_loss, best_loss):
        """Save checkpoint if requested."""
        model_state = model.state_dict()
        ema_state = ema_model.state_dict() if self.args.use_ema else None

        # Best checkpoint
        if best_loss is None or new_loss <= best_loss:
            best_loss = new_loss
            torch.save({
                "weight": model_state,
                "ema_weight": ema_state,
                "iter": step_id + 1,
                "best_loss": best_loss
            }, self.args.log_dir / "best.pth")

        # Last checkpoint (always saved)
        torch.save({
            "weight": model_state,
            "ema_weight": ema_state,
            "optimizer": optimizer.state_dict(),
            "iter": step_id + 1,
            "best_loss": best_loss
        }, self.args.log_dir / "last.pth")

        # Save intermediate checkpoints
        if (step_id + 1) % self.args.interm_ckpt_freq == 0:
            torch.save({
                "weight": model_state,
                "ema_weight": ema_state,
                "iter": step_id + 1,
                "best_loss": best_loss
            }, self.args.log_dir / f"interm{step_id + 1}.pth")

        return best_loss


def base_collate_fn(batch):
    """Custom collate_fn, measured to be faster than default."""
    _dict = {}

    # Values for these come as lists
    list_keys = ["task", "instr"]
    for key in list_keys:
        if key not in batch[0].keys():
            continue
        _dict[key] = []
        for item in batch:
            _dict[key].extend(item[key])

    # Treat rest as tensors
    _dict.update({
        k_: (
            torch.cat([item[k_] for item in batch])
            if batch[0][k_] is not None else None
        )
        for k_ in batch[0].keys() if k_ not in list_keys
    })

    return _dict


def actions_collate_fn(batch):
    return {"action": torch.cat([item["action"] for item in batch])}


def relative_to_absolute(action, proprio):
    # action (B, T, 8), proprio (B, 1, 7)
    pos = proprio[..., :3] + action[..., :3].cumsum(1)

    orn = proprio[..., 3:6] + action[..., 3:6].cumsum(1)
    orn = (orn + torch.pi) % (2 * torch.pi) - torch.pi

    return torch.cat([pos, orn, action[..., 6:]], -1)
