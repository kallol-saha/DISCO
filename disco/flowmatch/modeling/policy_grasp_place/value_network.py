import torch
from torch import nn
from torch.nn import functional as F

from ..utils.layers import AttentionModule
from ..utils.position_encodings import RotaryPositionEncoding3D


class ValueNetwork(nn.Module):
    """Pointwise Q-function V(s, a_grasp, a_place).

    The action is a (grasp_pose, placement_pose) pair, each 8-D
    (xyz + quat + gripper). The scene PCD is 3-channel xyz only —
    no target-object mask. (The 4-channel masked-PCD experiment
    landed alongside the listwise loss refactor on 2026-04-30 and
    was reverted on 2026-05-01.)

    Loss: per-batch weighted-mean BCE on success-fraction targets,
    with sample weight = log1p(n_leaves).
    """

    ACTION_DIM = 8
    NUM_ACTION_TOKENS = 2  # grasp + placement

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 num_shared_attn_layers=4):
        super().__init__()

        self.action_encoder = nn.Sequential(
            nn.Linear(self.ACTION_DIM, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.action_type_embed = nn.Embedding(self.NUM_ACTION_TOKENS, embedding_dim)

        self.action_relative_pe = RotaryPositionEncoding3D(embedding_dim)
        self.action_context_head = AttentionModule(
            num_layers=3, d_model=embedding_dim, dim_fw=embedding_dim,
            n_heads=num_attn_heads, rotary_pe=True, use_adaln=False,
            pre_norm=False,
        )

        # 3-channel input: xyz only.
        self.scene_pos_to_feat = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.value_head = ValueTransformerHead(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            num_action_tokens=self.NUM_ACTION_TOKENS,
        )

    def encode_action(self, action, context_feats, context_pos):
        """
        Args:
            action: (B, 2, 8) — [grasp, placement], each pos(3)+quat(4)+gripper(1)
            context_feats: (B, N, F)
            context_pos: (B, N, 3)

        Returns:
            action_feats: (B, 2, F)
        """
        B = action.shape[0]

        action_feats = self.action_encoder(action)  # (B, 2, F)

        type_idx = torch.arange(
            self.NUM_ACTION_TOKENS, device=action.device
        ).unsqueeze(0).expand(B, -1)
        action_feats = action_feats + self.action_type_embed(type_idx)

        action_pos = self.action_relative_pe(action[..., :3])
        context_pos_encoded = self.action_relative_pe(context_pos)

        action_feats = self.action_context_head(
            action_feats, context_feats,
            seq1_pos=action_pos, seq2_pos=context_pos_encoded,
        )[-1]

        return action_feats

    def forward(self, pcd, actions, targets=None, n_leaves=None):
        """Pointwise scoring + per-batch weighted-mean BCE.

        Args:
            pcd:      (B, N, 3)   xyz only.
            actions:  (B, 2, 8)   grasp+place tokens.
            targets:  (B,) or None   success-fraction targets in [0, 1].
                                      If None, returns inference scores.
            n_leaves: (B,) or None   per-sample leaf count for log1p weighting.
                                      If None at training, samples are unweighted.

        Returns:
            training:   scalar weighted-mean BCE loss
            inference:  (B,) sigmoid scores
        """
        assert pcd.shape[-1] == 3, \
            f"pcd last dim must be 3 (xyz), got {tuple(pcd.shape)}"
        assert actions.shape[-2:] == (self.NUM_ACTION_TOKENS, self.ACTION_DIM), \
            f"actions must end with (2, 8), got {tuple(actions.shape)}"

        scene_feats = self.scene_pos_to_feat(pcd)              # (B, N, F)

        action_feats = self.encode_action(actions, scene_feats, pcd)
        logits = self.value_head(
            action_feats=action_feats, scene_feats=scene_feats,
        ).squeeze(-1)                                          # (B,)

        if targets is None:
            return torch.sigmoid(logits)

        targets = targets.to(logits.dtype)
        per_sample = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )                                                       # (B,)

        if n_leaves is None:
            return per_sample.mean()

        weights = torch.log1p(n_leaves.to(logits.dtype))        # (B,)
        return (weights * per_sample).sum() / weights.sum().clamp_min(1e-6)


class ValueTransformerHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 num_action_tokens=2):
        super().__init__()
        self.num_action_tokens = num_action_tokens

        self.value_query = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.action_proj = nn.Sequential(
            nn.Linear(embedding_dim * num_action_tokens, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.cross_attn = AttentionModule(
            num_layers=2, d_model=embedding_dim, dim_fw=embedding_dim,
            dropout=0.1, n_heads=num_attn_heads, pre_norm=False,
            rotary_pe=False, use_adaln=True, is_self=False,
        )

        self.self_attn = AttentionModule(
            num_layers=num_shared_attn_layers, d_model=embedding_dim,
            dim_fw=embedding_dim, dropout=0.1, n_heads=num_attn_heads,
            pre_norm=False, rotary_pe=False, use_adaln=True, is_self=True,
        )

        self.value_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, action_feats, scene_feats):
        """
        Args:
            action_feats: (B, num_action_tokens, F)
            scene_feats: (B, N, F)
        """
        B = action_feats.shape[0]

        value_query = self.value_query.expand(B, -1, -1)  # (B, 1, F)

        action_flat = action_feats.flatten(1)
        ada_sgnl = self.action_proj(action_flat)

        value_feats = self.cross_attn(
            seq1=value_query, seq2=scene_feats, ada_sgnl=ada_sgnl,
        )[-1]

        features = torch.cat([value_feats, action_feats, scene_feats], dim=1)
        features = self.self_attn(
            seq1=features, seq2=features, ada_sgnl=ada_sgnl,
        )[-1]

        value_feats = features[:, 0, :]
        return self.value_predictor(value_feats)
