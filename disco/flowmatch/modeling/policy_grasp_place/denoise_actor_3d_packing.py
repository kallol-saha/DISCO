import torch
from torch import nn
from torch.nn import functional as F
import einops

from ..noise_scheduler import fetch_schedulers
from ..utils.layers import AttentionModule
from ..utils.position_encodings import SinusoidalPosEmb, RotaryPositionEncoding3D
from ..utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)


class DenoiseActor(nn.Module):
    """Grasp+place packing policy. Three modes (set via __init__ kwargs):

      * **joint** (`trajectory_length=2, grasp_condition_dim=None`): legacy
        behavior — predicts grasp at index 0 and placement at index 1 in
        one shot. Both poses denoised in lockstep.
      * **grasp** (`trajectory_length=1, grasp_condition_dim=None`):
        single-pose model that predicts only the grasp.
      * **place** (`trajectory_length=1, grasp_condition_dim=7`):
        single-pose model that predicts the placement, conditioned on a
        clean grasp pose fed as `(B, 7) = pos+quat_wxyz` via the
        `grasp_cond` kwarg of `forward`/`compute_trajectory`. The clean
        grasp is converted to 9D (pos + 6D rot), embedded, and added as
        an extra context token in cross/self-attn alongside the scene
        tokens.

    The grasp+place pair was originally trained jointly; the cascaded
    grasp→place factorization (Option C in the design doc) trains a
    Grasp model on grasp-only targets and a Place model that conditions
    on the clean grasp at training time. Inference runs them sequentially:
    sample grasp → use as condition → sample place.

    Other notes:
      * Input is the point cloud only. Proprioception is ignored.
      * PCD has 3 (xyz) or 4 (xyz + target_mask) channels. The mask is
        fed only into the scene encoder; rotary PE always uses xyz only.
      * Gripper state is not predicted; it's hardcoded by mode (grasp=
        closed, place=open).
      * Per-pose loss weights apply only in joint mode (grasp 2× place);
        single-pose modes use weight 1.0.
    """

    def __init__(self,
                 # Encoder and decoder arguments
                 embedding_dim=60,
                 num_attn_heads=8,
                 nhist=1,         # accepted for parity with original ctor; unused
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_xyzw',
                 # Denoising arguments
                 denoise_timesteps=30,
                 denoise_model="ddpm",
                 # Training arguments
                 lv2_batch_size=1,
                 # Width of the input fed to `TransformerHead.scene_pos_to_feat`.
                 # 4 = xyz + target-object mask (the original masked variant).
                 # 3 = xyz only (the `target-only` data variant where every
                 # other non-shelved object has been removed from the scene
                 # before sampling, so the mask is implicit). Rotary PE still
                 # uses xyz only — this only affects scene_pos_to_feat.
                 pcd_input_channels=4,
                 # Cascaded-mode arguments (added 2026-05-04).
                 # trajectory_length=2 (default) → joint grasp+place mode.
                 # trajectory_length=1, grasp_condition_dim=None → grasp-only.
                 # trajectory_length=1, grasp_condition_dim=7 → place-only,
                 # conditioned on a clean grasp pose fed via `grasp_cond`.
                 trajectory_length=2,
                 grasp_condition_dim=None):
        super().__init__()
        # Arguments to be accessed by the main class
        self._rotation_format = rotation_format
        self._relative = relative
        self._lv2_batch_size = lv2_batch_size
        self._nhist = nhist
        self.pcd_input_channels = int(pcd_input_channels)
        self.trajectory_length = int(trajectory_length)
        self.grasp_condition_dim = (
            int(grasp_condition_dim) if grasp_condition_dim is not None else None
        )

        # Mode-specific gripper schedule and per-pose loss weights.
        # Joint: (closed at grasp idx, open at place idx); single-pose
        # modes carry one gripper value matching the predicted pose.
        # Per-pose loss weights collapse to 1.0 in single-pose modes.
        if self.trajectory_length == 2:
            assert self.grasp_condition_dim is None, \
                "joint mode (trajectory_length=2) cannot also use grasp_condition_dim"
            self._mode = "joint"
            gripper_schedule = (1.0, 0.0)
            pose_loss_weights = (2.0, 1.0)
        elif self.trajectory_length == 1 and self.grasp_condition_dim is None:
            self._mode = "grasp"
            gripper_schedule = (1.0,)
            pose_loss_weights = (1.0,)
        elif self.trajectory_length == 1 and self.grasp_condition_dim is not None:
            self._mode = "place"
            gripper_schedule = (0.0,)
            pose_loss_weights = (1.0,)
        else:
            raise ValueError(
                f"Unsupported (trajectory_length={trajectory_length}, "
                f"grasp_condition_dim={grasp_condition_dim}) combination."
            )

        # Per-pose loss weights as a non-learnable buffer so they move with
        # the module and are visible in `state_dict`.
        self.register_buffer(
            "_pose_loss_weights",
            torch.tensor(list(pose_loss_weights), dtype=torch.float32),
        )
        self.register_buffer(
            "_gripper_schedule",
            torch.tensor(list(gripper_schedule), dtype=torch.float32).view(1, -1, 1),
        )

        # Action decoder, runs at every denoising timestep
        self.traj_encoder = nn.Linear(
            6 if rotation_format == 'euler' else 9,  # XYZ + Euler or 6D
            embedding_dim
        )

        # Prediction head for denoising. The head is mode-aware: in
        # `place` mode it gets a `grasp_cond_dim` so it can build the
        # clean-grasp conditioning embedding.
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format == 'euler' else 6,
            pcd_feat_dim=self.pcd_input_channels,
            grasp_condition_dim=self.grasp_condition_dim,
        )

        # Noise/denoise schedulers and hyperparameters
        self.position_scheduler, self.rotation_scheduler = fetch_schedulers(
            denoise_model, denoise_timesteps
        )
        self.n_steps = denoise_timesteps

        # Per-pose workspace normalizer. Shape (2, L, ndims_norm):
        # axis 0 = (min, max), axis 1 = pose index (1 in single-pose
        # modes, 2 in joint mode), axis 2 = the dims being normalized
        # (3 for xyz; 6 if rotation is also normalized in euler mode).
        ndims_norm = 6 if rotation_format == 'euler' else 3
        self.workspace_normalizer = nn.Parameter(
            torch.stack([
                torch.zeros(self.trajectory_length, ndims_norm),
                torch.ones(self.trajectory_length, ndims_norm),
            ]),
            requires_grad=False,
        )
        self.nrm_dim = int(self.workspace_normalizer.size(-1))

    def encode_inputs(self, rgb3d, rgb2d, pcd, instruction, proprio):
        fixed_inputs = self.encoder(
            rgb3d, rgb2d, pcd, instruction,
            proprio.flatten(1, 2)
        )
        # Query trajectory (for relative trajectory prediction)
        query_trajectory = proprio[:, -1:]
        return (query_trajectory,) + fixed_inputs

    def downsample_pcd(self, pcd, height = 32, width = 32):
        """
        Downsample the point cloud to the given height and width.

        Args:
            - pcd: (B, N, 3)
            - height: int
            - width: int
        """
        # Point cloud
        num_cameras = pcd.shape[1]
        # Interpolate point cloud to get the corresponding locations
        pcd = F.interpolate(
            einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w"),
            (height, width),
            mode='bilinear'     # Bilinear interpolation for spatial up or down sampling.
        )

        # Merge different cameras
        pcd = einops.rearrange(
            pcd,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )
        return pcd

    def _encode_grasp_cond(self, grasp_cond):
        """Convert (B, 7) clean grasp = pos+quat_wxyz to (B, 9) = pos+6D rot.

        Mirrors `convert_rot` for a single (non-trajectory-length-axis) pose.
        Returns the 9D representation in absolute (unnormalized) coordinates,
        ready to be embedded by the head's `grasp_cond_emb`. The xyz portion
        is also returned so the head can use it for rotary PE.
        """
        if grasp_cond is None:
            return None, None
        # grasp_cond: (B, 7) = pos(3) + quat_wxyz(4)
        pos = grasp_cond[..., :3]                         # (B, 3)
        quat = normalise_quat(grasp_cond[..., 3:7])       # (B, 4) wxyz
        if self._rotation_format == 'quat_xyzw':
            quat = quat[..., (3, 0, 1, 2)]                # data is wxyz; convert if needed
        rot_mat = quaternion_to_matrix(quat)              # (B, 3, 3)
        rot_6d = get_ortho6d_from_rotation_matrix(rot_mat)  # (B, 6)
        cond_9d = torch.cat([pos, rot_6d], dim=-1)        # (B, 9)
        return cond_9d, pos

    def policy_forward_pass(self, trajectory, timestep, pcd, grasp_cond=None):
        """
        Forward pass through the denoising prediction head.

        Args:
            trajectory: (B, traj_len, 9) - noisy trajectory (pos + 6D rot)
            timestep: (B,) - denoising timestep
            pcd: (B, N, pcd_input_channels) - xyz [+ target_mask].
            grasp_cond: (B, 7) clean grasp pose — only used in `place`
                mode (when self.grasp_condition_dim is not None). Ignored
                otherwise.
        """
        trajectory_feats = self.traj_encoder(trajectory)

        # But use positions from unnormalized absolute trajectory
        traj_xyz = self.unnormalize_pos(trajectory)[..., :3]
        if self._relative:  # not used in this codebase
            traj_xyz = torch.cumsum(traj_xyz, dim=1)

        cond_feats, cond_xyz = self._encode_grasp_cond(grasp_cond)

        return self.prediction_head(
            trajectory_feats,
            traj_xyz,
            timestep,
            rgb3d_pos=pcd,
            grasp_cond=cond_feats,
            grasp_cond_xyz=cond_xyz,
        )

    def conditional_sample(self, trajectory, device, pcd, grasp_cond=None):
        """Iterative denoising to generate the noise-free trajectory."""
        self.position_scheduler.set_timesteps(self.n_steps, device=device)
        self.rotation_scheduler.set_timesteps(self.n_steps, device=device)

        timesteps = self.position_scheduler.timesteps
        for t_ind, t in enumerate(timesteps):
            out = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(device).long(),
                pcd,
                grasp_cond=grasp_cond,
            )
            out = out[-1]  # keep only the last layer's output
            pos = self.position_scheduler.step(
                out[..., :3], t_ind, trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_scheduler.step(
                out[..., 3:], t_ind, trajectory[..., 3:]
            ).prev_sample
            trajectory = torch.cat((pos, rot), -1)

        return trajectory

    def compute_trajectory(self, pcd, grasp_cond=None):
        """
        Generate trajectory from noise via iterative denoising.

        Args:
            pcd: (B, N, pcd_input_channels) - xyz [+ target_mask].
            grasp_cond: (B, 7) clean grasp pose — required in `place`
                mode, ignored in `joint`/`grasp` modes.

        Returns:
            trajectory: (B, L, 8). Position + quaternion (wxyz) + gripper.
                L is `self.trajectory_length` (1 or 2). The gripper channel
                is hardcoded by mode and is not predicted.
        """
        out_dim = 6 if self._rotation_format == 'euler' else 9
        trajectory = torch.randn(
            size=(pcd.shape[0], self.trajectory_length, out_dim),
            device=pcd.device,
        )
        trajectory = self.conditional_sample(
            trajectory, device=pcd.device, pcd=pcd, grasp_cond=grasp_cond
        )

        trajectory = self.unconvert_rot(trajectory)        # 9 -> pos + quat
        trajectory = self.unnormalize_pos(trajectory)

        # Append the hardcoded gripper schedule so the output tensor matches
        # the (B, L, 8) shape the rest of the pipeline expects.
        gripper = self._gripper_schedule.expand(trajectory.shape[0], -1, -1)
        return torch.cat([trajectory, gripper], dim=-1)

    def compute_loss(self, gt_trajectory, pcd, grasp_cond=None):
        """
        Args:
            gt_trajectory: (B, L, 8) from the dataset where L = trajectory_length.
                We strip the trailing gripper channel here because the model
                never predicts it — the gripper schedule is hardcoded at the
                output.
            pcd: (B, N, pcd_input_channels) — xyz in robot-base frame
                [+ target_mask].
            grasp_cond: (B, 7) clean grasp pose — required in `place` mode.
        """
        gt_trajectory = gt_trajectory[..., :7]             # drop gripper
        gt_trajectory = self.normalize_pos(gt_trajectory)
        gt_trajectory = self.convert_rot(gt_trajectory)    # -> pos(3) + 6D rot = 9

        weights = self._pose_loss_weights.view(1, -1, 1)   # (1, L, 1)

        total_loss = 0
        for _ in range(self._lv2_batch_size):
            noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)
            timesteps = self.position_scheduler.sample_noise_step(
                num_noise=len(noise), device=noise.device
            )
            pos_noisy = self.position_scheduler.add_noise(
                gt_trajectory[..., :3], noise[..., :3], timesteps
            )
            rot_noisy = self.rotation_scheduler.add_noise(
                gt_trajectory[..., 3:], noise[..., 3:], timesteps
            )
            noisy_trajectory = torch.cat((pos_noisy, rot_noisy), -1)

            pred = self.policy_forward_pass(
                noisy_trajectory, timesteps, pcd, grasp_cond=grasp_cond
            )

            denoise_target = self.position_scheduler.prepare_target(
                noise, gt_trajectory
            )

            for layer_pred in pred:
                pos_err = (layer_pred[..., :3] - denoise_target[..., :3]).abs()
                rot_err = (layer_pred[..., 3:] - denoise_target[..., 3:]).abs()
                pos_loss = (pos_err * weights).mean()
                rot_loss = (rot_err * weights).mean()
                total_loss = total_loss + 30 * pos_loss + 10 * rot_loss

        return total_loss / self._lv2_batch_size

    def normalize_pos(self, signal):
        _min = self.workspace_normalizer[0]
        _max = self.workspace_normalizer[1]
        diff = _max - _min

        out = signal.clone()
        out[..., :self.nrm_dim] = (
            (signal[..., :self.nrm_dim] - _min) / diff * 2.0
            - 1.0
        )
        return out

    def unnormalize_pos(self, signal):
        _min = self.workspace_normalizer[0]
        _max = self.workspace_normalizer[1]
        diff = _max - _min

        out = signal.clone()
        out[..., :self.nrm_dim] = (
            (signal[..., :self.nrm_dim] + 1.0) / 2.0 * diff
            + _min
        )
        return out

    def convert_rot(self, signal):
        # If Euler then no conversion
        if self._rotation_format == 'euler':
            return signal
        # Else assume quaternion
        rot = normalise_quat(signal[..., 3:7])
        res = signal[..., 7:] if signal.size(-1) > 7 else None
        # The following code expects wxyz quaternion format!
        if self._rotation_format == 'quat_xyzw':
            rot = rot[..., (3, 0, 1, 2)]
        # Convert to rotation matrix
        rot = quaternion_to_matrix(rot)
        # Convert to 6D
        if len(rot.shape) == 4:
            B, L, D1, D2 = rot.shape
            rot = rot.reshape(B * L, D1, D2)
            rot = get_ortho6d_from_rotation_matrix(rot)
            rot = rot.reshape(B, L, 6)
        else:
            rot = get_ortho6d_from_rotation_matrix(rot)
        # Concatenate pos, rot, other state info
        signal = torch.cat([signal[..., :3], rot], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        # If Euler then no conversion
        if self._rotation_format == 'euler':
            return signal
        # Else assume quaternion
        res = signal[..., 9:] if signal.size(-1) > 9 else None
        if len(signal.shape) == 3:
            B, L, _ = signal.shape
            rot = signal[..., 3:9].reshape(B * L, 6)
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
            quat = quat.reshape(B, L, 4)
        else:
            rot = signal[..., 3:9]
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
        # The above code handled wxyz quaternion format!
        if self._rotation_format == 'quat_xyzw':
            quat = quat[..., (1, 2, 3, 0)]
        signal = torch.cat([signal[..., :3], quat], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal

    def _slice_for_mode(self, gt_trajectory):
        """Map dataset's (B, 2, 8) tensor to (target, grasp_cond) per mode.

        The dataset always returns the full (grasp, place) pair as
        `(B, 2, 8)`. Each mode consumes this differently:
          * joint  → target = (B, 2, 8), grasp_cond = None
          * grasp  → target = (B, 1, 8) (the grasp slice), grasp_cond = None
          * place  → target = (B, 1, 8) (the place slice),
                     grasp_cond = (B, 7) (the grasp pos+quat)
        """
        if self._mode == "joint":
            return gt_trajectory, None
        if self._mode == "grasp":
            return gt_trajectory[:, 0:1, :], None
        # place mode: place slice as target, grasp as condition
        target = gt_trajectory[:, 1:2, :]
        cond = gt_trajectory[:, 0, :7]   # pos(3) + quat_wxyz(4)
        return target, cond

    def forward(
        self,
        gt_trajectory,
        pcd,
        proprioception=None,    # accepted for trainer compat; ignored
        run_inference=False,
        grasp_cond=None,        # explicit override for inference; if None,
                                # place-mode `compute_trajectory` requires
                                # grasp_cond passed by the caller (inference
                                # of a place model needs a clean grasp from
                                # somewhere — typically the grasp model's output)
    ):
        """
        Arguments:
            gt_trajectory: (B, 2, 8) at training — full (grasp, place) pair
                from the dataset. Mode dispatch slices this based on
                `self._mode`. At inference this is unused (sampled from noise).
            pcd: (B, N, pcd_input_channels) point cloud in robot-base frame.
            proprioception: ignored.
            grasp_cond: only used at inference time in `place` mode. At
                training, the conditioning is taken from the GT grasp pose
                in `gt_trajectory` (teacher forcing on the clean grasp).

        Returns:
            - loss scalar at training time, or
            - trajectory (B, L, 8) at inference (gripper channel hardcoded).
        """
        del proprioception  # explicitly unused

        if run_inference:
            # In place mode the caller MUST supply a clean grasp. At
            # production / MCTS time the planner passes one explicitly.
            # At train-time validation (evaluate_nsteps) the planner
            # path isn't available, so fall back to slicing the GT grasp
            # out of `gt_trajectory` — same teacher-forcing source the
            # training loss uses, which gives a meaningful val metric
            # for the place model in isolation.
            if self._mode == "place" and grasp_cond is None:
                if gt_trajectory is None or gt_trajectory.size(1) < 2:
                    raise ValueError(
                        "place mode at inference requires either "
                        "`grasp_cond=(B,7)` or a (B,2,8) gt_trajectory "
                        "to slice the grasp from."
                    )
                grasp_cond = gt_trajectory[:, 0, :7]
            return self.compute_trajectory(pcd, grasp_cond=grasp_cond)

        target, training_grasp_cond = self._slice_for_mode(gt_trajectory)
        return self.compute_loss(target, pcd, grasp_cond=training_grasp_cond)


class PointNetPPEncoder(nn.Module):
    """PointNet++-style local-aggregation block (no downsampling).

    Two EdgeConv layers with shared kNN graph in xyz space. Output keeps
    the input point count, so it's a drop-in replacement for the per-point
    MLP that was previously used for `scene_pos_to_feat`. EdgeConv:
        edge_feat = MLP([center_feat, neighbor_feat - center_feat])
        out_feat  = max_neighbors(edge_feat)

    The kNN graph is computed once per forward in xyz space and reused by
    both layers — neighborhoods don't change across layers because the
    kNN is in 3D coordinates, not in feature space.

    Memory: kNN computes (B, N, N) distances. We chunk along B (default
    chunk_size=4) so peak kNN memory is bounded by 4·N²·4 bytes.

    Input:  (B, N, in_dim)  — first 3 channels MUST be xyz
    Output: (B, N, out_dim)
    """

    def __init__(self, in_dim, out_dim, k=16, hidden_dim=None, knn_chunk=4):
        super().__init__()
        self.k = k
        self.knn_chunk = knn_chunk
        hidden_dim = hidden_dim if hidden_dim is not None else out_dim
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(2 * hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    @staticmethod
    def _gather_neighbors(feat, knn_idx):
        """feat: (B, N, D); knn_idx: (B, N, k); returns (B, N, k, D)."""
        B, N, D = feat.shape
        k = knn_idx.shape[-1]
        idx = knn_idx.reshape(B, N * k, 1).expand(-1, -1, D)
        return torch.gather(feat, 1, idx).reshape(B, N, k, D)

    def _knn(self, xyz):
        """xyz: (B, N, 3), returns (B, N, k) long indices."""
        B, N, _ = xyz.shape
        out = torch.empty(B, N, self.k, dtype=torch.long, device=xyz.device)
        # cdist needs FP32 for numerical stability; bf16 autocast disabled here
        with torch.amp.autocast("cuda", enabled=False):
            xyz_f = xyz.float()
            for s in range(0, B, self.knn_chunk):
                e = min(s + self.knn_chunk, B)
                d = torch.cdist(xyz_f[s:e], xyz_f[s:e])  # (b, N, N)
                d.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))  # exclude self
                out[s:e] = d.topk(self.k, dim=-1, largest=False).indices
        return out

    def _edge_layer(self, feat, knn_idx, mlp):
        # feat: (B, N, D); returns (B, N, mlp_out)
        center = feat.unsqueeze(2).expand(-1, -1, self.k, -1)        # (B,N,k,D)
        neighbor = self._gather_neighbors(feat, knn_idx)              # (B,N,k,D)
        edge = torch.cat([center, neighbor - center], dim=-1)         # (B,N,k,2D)
        return mlp(edge).max(dim=2).values                            # (B,N,F)

    def forward(self, points):
        # points: (B, N, in_dim) — first 3 channels are xyz
        xyz = points[..., :3]
        knn_idx = self._knn(xyz)
        h = self._edge_layer(points, knn_idx, self.edge_mlp1)
        h = self._edge_layer(h, knn_idx, self.edge_mlp2)
        return h


class TransformerHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 nhist=1,        # accepted for parity with original ctor; unused
                 rotary_pe=True,
                 rot_dim=6,
                 pcd_feat_dim=3,
                 grasp_condition_dim=None):
        super().__init__()
        del nhist  # proprio path removed

        self.pcd_feat_dim = pcd_feat_dim
        self.grasp_condition_dim = grasp_condition_dim

        # In `place` mode the head receives a (B, 9) clean-grasp embedding
        # (pos(3) + 6D rot, already prepared by DenoiseActor) plus its
        # xyz for rotary PE. The embedding is concatenated to the scene
        # tokens after the PointNet++ encoder, so it participates in
        # cross-attn (as a key/value) and self-attn (alongside scene
        # tokens). Only the trajectory tokens are read out by the position
        # and rotation heads, so adding this token doesn't change outputs.
        if self.grasp_condition_dim is not None:
            # Always 9 = pos(3) + 6D rot, regardless of grasp_condition_dim
            # (which is the *input* dim from the caller, currently fixed
            # at 7 = pos+quat — converted to 9 inside DenoiseActor before
            # reaching here).
            self.grasp_cond_emb = nn.Linear(9, embedding_dim)

        # Different embeddings
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        # Scene token features. PointNet++-style EdgeConv block — replaces
        # the legacy per-point MLP (`Linear → ReLU → Linear`) so the scene
        # encoder has access to local neighborhood structure (curvature,
        # surface orientation, occlusion edges). Output keeps the input
        # point count, so downstream cross/self-attn shapes are unchanged.
        # Accepts xyz alone (3-ch) or xyz + target_mask (4-ch); the kNN
        # is always computed in xyz space (channels 0:3).
        self.scene_pos_to_feat = PointNetPPEncoder(
            in_dim=pcd_feat_dim, out_dim=embedding_dim, k=16, knn_chunk=4
        )
        # 3D rotary positional encoding
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = AttentionModule(
            num_layers=1,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=False,
            use_adaln=False,
            is_self=False
        )

        # Estimate attends to context (no subsampling)
        self.cross_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=False
        )

        # Shared attention layers
        self.self_attn = AttentionModule(
            num_layers=num_shared_attn_layers,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=True
        )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        self.rotation_self_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=True
        )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rot_dim)
        )

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        self.position_self_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=True
        )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )

    def forward(self, traj_feats, trajectory, timesteps, rgb3d_pos,
                grasp_cond=None, grasp_cond_xyz=None):
        """
        Arguments:
            traj_feats: (B, trajectory_length, F)
            trajectory: (B, trajectory_length, 3+6+X)
            timesteps: (B,) or (B, 1)
            rgb3d_pos: (B, N, pcd_feat_dim). The first 3 channels are xyz
                (used for rotary positional encoding); any remaining
                channels are extra per-point features (e.g. target_mask)
                that are fed only into the scene encoder.
            grasp_cond: (B, 9) clean-grasp embedding input (pos + 6D rot)
                — only used when this head was constructed with
                grasp_condition_dim != None (place mode). Concatenated
                with scene tokens after the encoder.
            grasp_cond_xyz: (B, 3) absolute xyz of the clean grasp, used
                for rotary PE of the conditioning token.

        Returns:
            list of (B, trajectory_length, 3+6) — pos + 6D rot per token.
            Gripper channel is not predicted.
        """
        _, traj_len, _ = trajectory.shape

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_len, device=traj_feats.device)
        )[None, :]
        traj_feats = traj_feats + traj_time_pos

        traj_xyz = trajectory[..., :3]
        rgb3d_xyz = rgb3d_pos[..., :3]

        time_embs = self.encode_denoising_timestep(timesteps)

        # Scene encoder (PointNet++ EdgeConv block)
        rgb3d_feats = self.scene_pos_to_feat(rgb3d_pos)

        # Place-mode conditioning: append the clean-grasp token to the
        # scene context (one extra "scene point" with its own xyz for
        # rotary PE).
        if self.grasp_condition_dim is not None:
            assert grasp_cond is not None and grasp_cond_xyz is not None, \
                "place-mode TransformerHead requires grasp_cond and grasp_cond_xyz"
            cond_feat = self.grasp_cond_emb(grasp_cond).unsqueeze(1)   # (B, 1, F)
            cond_xyz = grasp_cond_xyz.unsqueeze(1)                      # (B, 1, 3)
            rgb3d_feats = torch.cat([rgb3d_feats, cond_feat], dim=1)
            rgb3d_xyz = torch.cat([rgb3d_xyz, cond_xyz], dim=1)

        rel_traj_pos, rel_scene_pos, rel_pos = self.get_positional_embeddings(
            traj_xyz, rgb3d_xyz
        )

        traj_feats = self.cross_attn(
            seq1=traj_feats,
            seq2=rgb3d_feats,
            seq1_pos=rel_traj_pos,
            seq2_pos=rel_scene_pos,
            ada_sgnl=time_embs
        )[-1]

        features = self.get_sa_feature_sequence(traj_feats, rgb3d_feats)
        features = self.self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=rel_pos,
            seq2_pos=rel_pos,
            ada_sgnl=time_embs
        )[-1]

        rotation = self.predict_rot(
            features, rel_pos, time_embs, traj_feats.shape[1]
        )
        position, _ = self.predict_pos(
            features, rel_pos, time_embs, traj_feats.shape[1]
        )

        return [torch.cat((position, rotation), -1)]

    def encode_denoising_timestep(self, timestep):
        """Sinusoidal denoising-step embedding. No proprioception."""
        if timestep.dim() > 1:
            timestep = timestep.squeeze(-1)
        return self.time_emb(timestep)

    def get_positional_embeddings(
        self,
        traj_xyz,
        rgb3d_xyz,
    ):
        # Rotary PE for trajectories and scene positions. Both inputs are
        # strictly xyz; any extra per-point features must be sliced off
        # by the caller before getting here.
        rel_traj_pos = self.relative_pe_layer(traj_xyz)
        rel_scene_pos = self.relative_pe_layer(rgb3d_xyz)
        rel_pos = torch.cat([rel_traj_pos, rel_scene_pos], dim=1)
        return rel_traj_pos, rel_scene_pos, rel_pos

    def get_sa_feature_sequence(
        self,
        traj_feats,
        rgb3d_feats
    ):
        return torch.cat([traj_feats, rgb3d_feats], 1)

    def predict_pos(self, features, pos, time_embs, traj_len):
        position_features = self.position_self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=pos,
            seq2_pos=pos,
            ada_sgnl=time_embs
        )[-1]
        position_features = position_features[:, :traj_len]
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, pos, time_embs, traj_len):
        rotation_features = self.rotation_self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=pos,
            seq2_pos=pos,
            ada_sgnl=time_embs
        )[-1]
        rotation_features = rotation_features[:, :traj_len]
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation
