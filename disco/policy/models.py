"""
Build, load, save and sample the grasp and place policies and the value network.
"""
import os

import numpy as np
import torch

ROBOT_BASE_X_OFFSET = 0.615
POLICY_KWARGS = dict(  # identical to disco/policy/shelf_sampler.py::_POLICY_KWARGS
    embedding_dim=120, num_attn_heads=8, nhist=1, num_shared_attn_layers=4,
    relative=False, rotation_format="quat_wxyz", denoise_timesteps=10,
    denoise_model="rectified_flow", lv2_batch_size=1,
)
CHUNK = int(os.environ.get("DISCO_CHUNK", "4"))  # inference batch chunk (4096-pt attention is heavy)


def _autocast():
    """Return a bf16 autocast context for inference (DISCO_FP32=1 disables it)."""
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=os.environ.get("DISCO_FP32") != "1")


def _set_steps(model):
    """Set the policy's denoising steps from DISCO_STEPS, if that env var is set.

    Args:
        model: Flow-matching policy.
    """
    if os.environ.get("DISCO_STEPS"):
        model.n_steps = int(os.environ["DISCO_STEPS"])


def _contiguous(model):
    """Make the model's multi-dim parameters contiguous in memory and return the model.

    Args:
        model: Network to fix in place.
    """
    for p in model.parameters():
        if p.ndim > 1 and not p.is_contiguous():
            p.data = p.data.contiguous()
    return model


def build_policy(kind, pcd_channels):
    """Build an untrained 3DFA grasp or place policy.

    Args:
        kind: "grasp" (predicts the grasp pose) or "place" (predicts the place pose given a grasp).
        pcd_channels: Point-cloud channels (3 = xyz, 4 = xyz + object mask).
    """
    from disco.flowmatch.modeling.policy_grasp_place.denoise_actor_3d_packing import DenoiseActor
    if kind not in ("grasp", "place"):
        raise ValueError(f"kind must be 'grasp' or 'place', got {kind!r}")
    return _contiguous(DenoiseActor(
        **POLICY_KWARGS, trajectory_length=1,
        grasp_condition_dim=7 if kind == "place" else None,
        pcd_input_channels=int(pcd_channels),
    ))


def build_value():
    """Build an untrained value network Q(o, a)."""
    from disco.flowmatch.modeling.policy_grasp_place.value_network import ValueNetwork
    return _contiguous(ValueNetwork(embedding_dim=120, num_attn_heads=8, num_shared_attn_layers=4))


def load_weights(model, path):
    """Load checkpoint weights into the model and return it.

    Missing keys raise an error; unexpected keys are ignored.

    Args:
        model: Network to fill.
        path: Checkpoint written by `save_weights`.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(blob["weight"], strict=False)
    if missing:
        raise RuntimeError(f"{path}: {len(missing)} missing keys, e.g. {missing[:5]}")
    if unexpected:
        print(f"[load_weights] {path}: ignoring {len(unexpected)} unexpected keys")
    return model


def save_weights(model, path, step, optimizer=None):
    """Atomically save the model weights (and optimizer state) to a checkpoint.

    Args:
        model: Network to save.
        path: Output checkpoint path.
        step: Training step stored with the weights.
        optimizer: Optional optimizer whose state is saved too.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    blob = {"weight": model.state_dict(), "iter": int(step)}
    if optimizer is not None:
        blob["optimizer"] = optimizer.state_dict()
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)  # atomic: an evaluator never reads a half-written checkpoint


@torch.no_grad()
def fit_policy_normalizer(model, goal_pose, kind, buffer=0.04):
    """Set the policy's workspace normalizer to the min/max robot-frame xyz of its target poses.

    Args:
        model: Grasp or place policy.
        goal_pose: (N, 2, 8) world-frame (grasp, place) poses from the dataset.
        kind: "grasp" or "place"; picks which of the two poses to fit.
        buffer: Margin added on each side, in meters.
    """
    i = 0 if kind == "grasp" else 1
    pos = goal_pose[:, i:i + 1, :3].float().clone()
    pos[..., 0] += ROBOT_BASE_X_OFFSET
    lo, hi = pos.min(0).values - buffer, pos.max(0).values + buffer
    model.workspace_normalizer.copy_(torch.stack([lo, hi]).to(model.workspace_normalizer))


def _device(model):
    """Return the device of the model's parameters.

    Args:
        model: Any torch module.
    """
    return next(model.parameters()).device


def _robot_pcd(pcd_world, device):
    """Shift a world-frame point cloud into the robot frame and return it as a float tensor.

    Args:
        pcd_world: (P, C) world-frame cloud (xyz, plus an optional mask channel).
        device: Target device.
    """
    t = torch.as_tensor(np.asarray(pcd_world), dtype=torch.float32, device=device).clone()
    t[..., 0] += ROBOT_BASE_X_OFFSET  # xyz only; a mask channel is untouched
    return t


@torch.no_grad()
def sample_grasps(model, pcd_world, n):
    """Sample stochastic grasp poses from the grasp policy.

    Args:
        model: Grasp policy.
        pcd_world: (P, C) world-frame scene cloud.
        n: Number of grasps to sample.

    Returns:
        (n, 7) world-frame grasp poses (xyz + wxyz quaternion) on the CPU.
    """
    _set_steps(model)
    dev = _device(model)
    pcd = _robot_pcd(pcd_world, dev).unsqueeze(0).expand(n, -1, -1).contiguous()
    with _autocast():
        outs = [model(None, pcd[s:s + CHUNK], run_inference=True).float() for s in range(0, n, CHUNK)]
    g = torch.cat(outs)[:, 0, :7].cpu()
    g[:, 0] -= ROBOT_BASE_X_OFFSET
    return g


@torch.no_grad()
def sample_places(model, pcd_world, grasps_world):
    """Sample one place pose per conditioning grasp from the place policy.

    Args:
        model: Grasp-conditioned place policy.
        pcd_world: (P, C) world-frame scene cloud.
        grasps_world: (n, 7) world-frame grasp poses.

    Returns:
        (n, 7) world-frame place poses on the CPU.
    """
    _set_steps(model)
    dev = _device(model)
    n = grasps_world.shape[0]
    pcd = _robot_pcd(pcd_world, dev).unsqueeze(0).expand(n, -1, -1).contiguous()
    cond = grasps_world.to(dev).float().clone()
    cond[:, 0] += ROBOT_BASE_X_OFFSET
    with _autocast():
        outs = [model(None, pcd[s:s + CHUNK], run_inference=True, grasp_cond=cond[s:s + CHUNK]).float()
                for s in range(0, n, CHUNK)]
    p = torch.cat(outs)[:, 0, :7].cpu()
    p[:, 0] -= ROBOT_BASE_X_OFFSET
    return p


@torch.no_grad()
def score(value_model, pcd_world, actions_world):
    """Score (grasp, place) actions in one scene with the value network.

    Args:
        value_model: Value network Q(o, a).
        pcd_world: (P, 3) world-frame scene cloud.
        actions_world: (M, 2, 8) world-frame (grasp, place) actions.

    Returns:
        (M,) numpy array of values in [0, 1].
    """
    m = actions_world.shape[0]
    if m == 0:
        return np.zeros(0, dtype=np.float32)
    dev = _device(value_model)
    pcd = _robot_pcd(pcd_world, dev)
    act = actions_world.to(dev).float().clone()
    act[..., 0] += ROBOT_BASE_X_OFFSET
    out = []
    for s in range(0, m, CHUNK):
        a = act[s:s + CHUNK]
        with _autocast():
            q = value_model(pcd=pcd.unsqueeze(0).expand(a.shape[0], -1, -1).contiguous(), actions=a)
        out.append(q.float().cpu())
    return torch.cat(out).numpy()
