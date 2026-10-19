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

    def __init__(self,
                 # Encoder and decoder arguments
                 embedding_dim=60,
                 num_attn_heads=8,
                 nhist=1,         # History length for Proprioception
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_xyzw',
                 # Denoising arguments
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 # Training arguments
                 lv2_batch_size=1):
        super().__init__()
        # Arguments to be accessed by the main class
        self._rotation_format = rotation_format
        self._relative = relative
        self._lv2_batch_size = lv2_batch_size
        self._nhist = nhist

        # Proprioception encoder (like encoder3d)
        # Learnable embedding for each history step
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
        # 3D rotary positional encoding for proprio positions
        self.proprio_relative_pe = RotaryPositionEncoding3D(embedding_dim)
        # Cross-attention from proprio to scene context
        self.gripper_context_head = AttentionModule(
            num_layers=3, d_model=embedding_dim, dim_fw=embedding_dim,
            n_heads=num_attn_heads, rotary_pe=True, use_adaln=False,
            pre_norm=False
        )
        # Scene feature encoder (for proprio context)
        # TODO: This is a new layer added because originally in 3DFA proprio features will attend to the rgb3d_feats from CLIP 
        self.scene_pos_to_feat = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Action decoder, runs at every denoising timestep
        self.traj_encoder = nn.Linear(
            6 if rotation_format == 'euler' else 9,  # XYZ + Euler or 6D
            embedding_dim
        )

        # Prediction head for denoising
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format == 'euler' else 6
        )

        # Noise/denoise schedulers and hyperparameters
        self.position_scheduler, self.rotation_scheduler = fetch_schedulers(
            denoise_model, denoise_timesteps
        )
        self.n_steps = denoise_timesteps

        # Normalization for the 3D space, will be loaded in the main process
        if rotation_format == 'euler':  # normalize pos+rot
            self.workspace_normalizer = nn.Parameter(
                torch.Tensor([[0., 0, 0, 0, 0, 0], [1., 1, 1, 1, 1, 1]]),
                requires_grad=False
            )
        else:
            self.workspace_normalizer = nn.Parameter(
                torch.Tensor([[0., 0., 0.], [1., 1., 1.]]),
                requires_grad=False
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

    def policy_forward_pass(self, trajectory, timestep, pcd, proprio_feats):
        """
        Forward pass through the denoising prediction head.

        Args:
            trajectory: (B, traj_len, 9) - noisy trajectory (pos + 6D rot)
            timestep: (B,) - denoising timestep
            pcd: (B, N, 3) - point cloud positions
            proprio_feats: (B, 1, embedding_dim) - encoded proprioception features
        """
        # Get features from normalized (relative) trajectory
        trajectory_feats = self.traj_encoder(trajectory)

        # But use positions from unnormalized absolute trajectory
        traj_xyz = self.unnormalize_pos(trajectory)[..., :3]
        query_trajectory = None
        if self._relative:  # relative to absolute  # This is False, ignore for now.
            traj_xyz = (
                query_trajectory[..., :3]
                + torch.cumsum(traj_xyz, dim=1)
            )

        return self.prediction_head(
            trajectory_feats,
            traj_xyz,
            timestep,
            rgb3d_pos=pcd,
            proprio_feats=proprio_feats
        )

    def conditional_sample(self, trajectory, device, pcd, proprio_feats):
        """
        Iterative denoising to generate trajectory.

        Args:
            trajectory: (B, traj_len, 9) - initial noisy trajectory
            device: torch device
            pcd: (B, N, 3) - point cloud positions
            proprio_feats: (B, 1, embedding_dim) - encoded proprioception features
        """
        # Set schedulers
        self.position_scheduler.set_timesteps(self.n_steps, device=device)
        self.rotation_scheduler.set_timesteps(self.n_steps, device=device)

        # Iterative denoising
        timesteps = self.position_scheduler.timesteps
        for t_ind, t in enumerate(timesteps):
            out = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(device).long(),
                pcd,
                proprio_feats
            )
            out = out[-1]  # keep only last layer's output
            pos = self.position_scheduler.step(
                out[..., :3],
                t_ind, trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_scheduler.step(
                out[..., 3:-1],
                t_ind, trajectory[..., 3:]
            ).prev_sample
            trajectory = torch.cat((pos, rot), -1)

        return torch.cat((trajectory, out[..., -1:]), -1)

    def compute_trajectory(self, pcd, proprio_feats):
        """
        Generate trajectory from noise via iterative denoising.

        Args:
            pcd: (B, N, 3) - point cloud positions
            proprio_feats: (B, 1, embedding_dim) - encoded proprioception features

        Returns:
            trajectory: (B, 1, 8) - predicted trajectory (pos + quat + gripper)     NOTE: gripper is 0 for closed and 1 for open
        """
        # Sample from learned model starting from noise
        out_dim = 6 if self._rotation_format == 'euler' else 9
        trajectory = torch.randn(
            size=(pcd.shape[0], 1, out_dim),  # Single action prediction
            device=pcd.device
        )
        trajectory = self.conditional_sample(
            trajectory,
            device=pcd.device,
            pcd=pcd,
            proprio_feats=proprio_feats
        )

        # Back to quaternion
        trajectory = self.unconvert_rot(trajectory)
        # unnormalize position
        trajectory = self.unnormalize_pos(trajectory)
        # Convert gripper status to probability
        trajectory[..., -1] = trajectory[..., -1].sigmoid()

        return trajectory

    def compute_loss(self, gt_trajectory, pcd, proprio_feats):
        """
        gt_trajectory: (B, trajectory_length, 3+4+1)        NOTE: Quaternion (wxyz) will be converted to 6D internally in the function.
        pcd: (B, 4096, 3) in world coordinates
        proprio_feats: (B, 1, embedding_dim) encoded proprioception features
        """
        # Process gt_trajectory
        gt_openess = gt_trajectory[..., -1:]
        gt_trajectory = gt_trajectory[..., :-1]
        # Normalize all pos
        gt_trajectory = self.normalize_pos(gt_trajectory)
        # Convert rotation parametrization
        gt_trajectory = self.convert_rot(gt_trajectory)

        # Loop lv2_batch_size times and sample different noises with same input
        # Trick to effectively increase the batch size without re-encoding
        # It speeds up training but may decrease performance a bit
        total_loss = 0
        for _ in range(self._lv2_batch_size):
            # Sample noise
            noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

            # Sample a random timestep
            timesteps = self.position_scheduler.sample_noise_step(
                num_noise=len(noise), device=noise.device
            )

            # Add noise to the clean trajectories
            pos = self.position_scheduler.add_noise(
                gt_trajectory[..., :3], noise[..., :3],
                timesteps
            )
            rot = self.rotation_scheduler.add_noise(
                gt_trajectory[..., 3:], noise[..., 3:],
                timesteps
            )
            noisy_trajectory = torch.cat((pos, rot), -1)

            # Predict the noise residual
            pred = self.policy_forward_pass(
                noisy_trajectory,
                timesteps, pcd, proprio_feats
            )

            # Compute loss
            for layer_pred in pred:
                pos = layer_pred[..., :3]
                rot = layer_pred[..., 3:-1]
                openess = layer_pred[..., -1:]
                denoise_target = self.position_scheduler.prepare_target(
                    noise, gt_trajectory
                )
                loss = (
                    30 * F.l1_loss(pos, denoise_target[..., :3], reduction='mean')
                    + 10 * F.l1_loss(rot, denoise_target[..., 3:], reduction='mean')
                    + F.binary_cross_entropy_with_logits(openess, gt_openess)
                )
                total_loss = total_loss + loss
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

    def encode_proprio(self, proprio, context_feats, context_pos):
        """
        Encode proprioception input (like encoder3d).

        Args:
            proprio: (B, nhist, 8) - gripper pose history (pos + quat + gripper)
            context_feats: (B, N, embedding_dim) - scene features
            context_pos: (B, N, 3) - scene positions

        Returns:
            proprio_feats: (B, nhist, embedding_dim) - encoded proprioception features
        """

        # Learnable embedding for proprioception (like encoder3d)
        proprio_feats = self.curr_gripper_embed.weight.unsqueeze(0).repeat(
            len(proprio), 1, 1
        )  # (B, nhist, embedding_dim)

        # Rotary positional encoding
        proprio_pos = self.proprio_relative_pe(proprio[..., :3])  # Use xyz positions
        context_pos_encoded = self.proprio_relative_pe(context_pos)

        # Attention to scene tokens
        proprio_feats = self.gripper_context_head(
            proprio_feats, context_feats,
            seq1_pos=proprio_pos, seq2_pos=context_pos_encoded
        )[-1]

        return proprio_feats

    def forward(
        self,
        gt_trajectory,
        pcd,
        proprioception,         # The current gripper pose for the placement policy (B, nhist, 8) or (B, 8)
        run_inference=False
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, 3+4+X)
            pcd: (B, N, 3) point cloud in world coordinates
            proprioception: (B, nhist, 8) or (B, 8) current gripper pose(s)

        Note:
            The input rotation is expressed either as:
                a) quaternion (4D), then the model converts it to 6D internally.
                b) Euler angles (3D).

        Returns:
            - loss: scalar, if run_inference is False
            - trajectory: (B, trajectory_length, 3+rot+1), at inference
        """
        # Compute scene features for proprio encoding
        context_feats = self.scene_pos_to_feat(pcd)  # (B, N, embedding_dim)

        # Encode proprioception with scene context (like encoder3d)
        proprio_feats = self.encode_proprio(proprioception, context_feats, pcd)

        # Inference, don't use gt_trajectory
        if run_inference:
            return self.compute_trajectory(pcd, proprio_feats)

        # Training, use gt_trajectory to compute loss
        return self.compute_loss(gt_trajectory, pcd, proprio_feats)


class TransformerHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 nhist=1,
                 rotary_pe=True,
                 rot_dim=6):
        super().__init__()

        # Different embeddings
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        # Scene token features from 3D positions only
        self.scene_pos_to_feat = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
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

        # 3. Openess
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, traj_feats, trajectory, timesteps, rgb3d_pos, proprio_feats):
        """
        Arguments:
            traj_feats: (B, trajectory_length, F)
            trajectory: (B, trajectory_length, 3+6+X)
            timesteps: (B, 1)
            rgb3d_pos: (B, N, 3)
            proprio_feats: (B, 1, F) - encoded proprioception features

        Returns:
            list of (B, trajectory_length, 3+6+X)
        """
        _, traj_len, _ = trajectory.shape

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_len, device=traj_feats.device)
        )[None, :]

        # Add time positional embedding to trajectory features
        traj_feats = traj_feats + traj_time_pos

        traj_xyz = trajectory[..., :3]

        # Denoising timestep embeddings (conditioned on proprioception like original 3DFA)
        time_embs = self.encode_denoising_timestep(timesteps, proprio_feats)

        # Positional embeddings (3D rotary PE from positions)
        rel_traj_pos, rel_scene_pos, rel_pos = self.get_positional_embeddings(
            traj_xyz, rgb3d_pos
        )

        # Build scene features from positions
        rgb3d_feats = self.scene_pos_to_feat(rgb3d_pos)

        # Cross attention from gripper to full context
        traj_feats = self.cross_attn(
            seq1=traj_feats,
            seq2=rgb3d_feats,
            seq1_pos=rel_traj_pos,
            seq2_pos=rel_scene_pos,
            ada_sgnl=time_embs
        )[-1]

        # Self attention among gripper and sampled context
        features = self.get_sa_feature_sequence(
            traj_feats, rgb3d_feats
        )
        features = self.self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=rel_pos,
            seq2_pos=rel_pos,
            ada_sgnl=time_embs
        )[-1]

        # Rotation head
        rotation = self.predict_rot(
            features, rel_pos, time_embs, traj_feats.shape[1]
        )

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, traj_feats.shape[1]
        )

        # Openess head from position head
        openess = self.openess_predictor(position_features)

        return [
            torch.cat((position, rotation, openess), -1)
        ]

    def encode_denoising_timestep(self, timestep, proprio_feats):
        """
        Compute denoising timestep features conditioned on proprioception (like original 3DFA).

        Args:
            - timestep: (B,) or (B, 1)
            - proprio_feats: (B, nhist, F)

        Returns:
            - time_feats: (B, F)
        """
        if timestep.dim() > 1:
            timestep = timestep.squeeze(-1)
        time_feats = self.time_emb(timestep)
        # Flatten proprio and project through curr_gripper_emb
        proprio_flat = proprio_feats.flatten(1)  # (B, nhist * F)
        curr_gripper_feats = self.curr_gripper_emb(proprio_flat)  # (B, F)
        return time_feats + curr_gripper_feats

    def get_positional_embeddings(
        self,
        traj_xyz,
        rgb3d_pos
    ):
        # Rotary PE for trajectories and scene positions
        rel_traj_pos = self.relative_pe_layer(traj_xyz)
        rel_scene_pos = self.relative_pe_layer(rgb3d_pos)
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
