"""Sim2real augmentation helpers for shelf-packing dataloaders.

Implements the per-sample pipeline described in
`docs/superpowers/specs/2026-05-10-sim2real-dataloader-design.md`:

1. Per-axis Gaussian noise on pcd xyz.
2. Random "holes": clusters of dropped points replaced with random
   bbox-sampled points.
3. Independent random SE(3) for shelf points and non-shelf (object)
   points, with matched transforms applied to action poses.
4. Per-sample centering + max-radius scaling on pcd and action xyz.

All operations are torch-only (no scipy at __getitem__ time).
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import yaml


# --- Shelf-y threshold from env yaml ----------------------------------------

def load_shelf_y_thresholds(env_root: Path, margin: float = 0.01) -> Dict[int, float]:
    """Build a dict env_index → world-frame y threshold above which points
    are considered to belong to the shelf.

    Reads every `env_<i>/environment.yaml` under `env_root`. Computes the
    shelf-front-face y in world frame as

        shelf_front_y = pose.y - sin(yaw) * depth / 2

    (Local +x is the back of the shelf; local -x is the front. The shelf
    URDF is rotated by a yaw quaternion around z, so local x maps to
    world (cos yaw, sin yaw, 0).) Returns the threshold

        threshold = shelf_front_y - margin

    so a point is "shelf" iff `pcd_y > threshold`. The margin absorbs the
    front-lip wall thickness.

    All envs in this repo have sin(yaw) > 0; the rule "y > threshold"
    therefore picks out the shelf interior (and the shelf back wall).
    """
    env_root = Path(env_root)
    thresholds: Dict[int, float] = {}
    for env_dir in sorted(env_root.glob("env_*")):
        try:
            idx = int(env_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        yaml_path = env_dir / "environment.yaml"
        if not yaml_path.exists():
            continue
        with yaml_path.open() as f:
            data = yaml.safe_load(f)
        shelf = data["shelf"]
        pose = shelf["pose"]
        depth = float(shelf["depth"])
        # pose = [x, y, z, qw, qx, qy, qz]
        qw, qx, qy, qz = float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])
        # yaw around z from a quaternion with negligible roll/pitch:
        # sin(yaw) = 2*(qw*qz + qx*qy); cos(yaw) = 1 - 2*(qy^2 + qz^2).
        sin_yaw = 2.0 * (qw * qz + qx * qy)
        front_y = float(pose[1]) - sin_yaw * (depth / 2.0)
        thresholds[idx] = front_y - margin
    if not thresholds:
        raise FileNotFoundError(
            f"No env_<i>/environment.yaml found under {env_root}. "
            "Sim2real dataloader needs per-env shelf-y thresholds."
        )
    return thresholds


# --- Random SO(3) and SE(3) -------------------------------------------------

def _random_quat(generator: torch.Generator) -> torch.Tensor:
    """Uniformly random unit quaternion (wxyz)."""
    v = torch.randn(4, generator=generator)
    return v / v.norm()


def _quat_to_rot(q_wxyz: torch.Tensor) -> torch.Tensor:
    """Quaternion (wxyz) → 3x3 rotation matrix."""
    w, x, y, z = q_wxyz.unbind(-1)
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]),
    ])
    return R


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 * q2 in wxyz convention. Both inputs (..., 4)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def random_se3(
    trans_range: float,
    generator: torch.Generator,
    yaw_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample T = (R, t) with t uniform in [-trans_range, trans_range]^3.

    If `yaw_only=True`, R is restricted to rotation about world z (yaw
    θ ~ U[-π, π], quat = (cos(θ/2), 0, 0, sin(θ/2))). Use this for the
    target object: it rests on the table, so roll/pitch is not a valid
    augmentation.

    If `yaw_only=False`, R is uniform on SO(3) via the 4-element-Gaussian
    normalization. Used for the shelf+background transform.

    Returns (R_3x3, t_3, q_wxyz). Returning both R and q saves repeated
    matrix→quaternion conversions in callers.
    """
    if yaw_only:
        theta = (torch.rand(1, generator=generator) * 2.0 - 1.0) * torch.pi
        half = theta * 0.5
        q = torch.stack([
            torch.cos(half).squeeze(0),
            torch.zeros((), dtype=theta.dtype),
            torch.zeros((), dtype=theta.dtype),
            torch.sin(half).squeeze(0),
        ])
    else:
        q = _random_quat(generator)
    R = _quat_to_rot(q)
    t = (torch.rand(3, generator=generator) * 2.0 - 1.0) * trans_range
    return R, t, q


def apply_se3_xyz(xyz: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply (R, t) to xyz of shape (..., 3): out = xyz @ R^T + t."""
    return xyz @ R.T + t


# --- Per-point Gaussian noise -----------------------------------------------

def add_gaussian_noise(
    pcd_xyz: torch.Tensor,
    sigma_max: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Add iid Gaussian noise N(0, sigma^2) per axis, per point. The
    standard deviation `sigma` is sampled once per call uniformly in
    [0, sigma_max]. Returns a new tensor.
    """
    sigma = torch.rand(1, generator=generator).item() * sigma_max
    noise = torch.randn(pcd_xyz.shape, generator=generator) * sigma
    return pcd_xyz + noise


# --- Random holes -----------------------------------------------------------

def add_holes(
    pcd_xyz: torch.Tensor,
    target_mask: Optional[torch.Tensor],
    n_range: Tuple[int, int],
    r_range: Tuple[float, float],
    max_holes_on_target: Optional[int],
    generator: torch.Generator,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Drop K clusters of points and refill with bbox-uniform random pts.

    Args:
        pcd_xyz: (P, 3) float tensor.
        target_mask: optional (P,) bool tensor marking target-object pts.
            If provided AND `max_holes_on_target` is not None, the number
            of holes whose center pt has `target_mask=True` is capped at
            `max_holes_on_target`. Replacement points always get
            `target_mask=False` regardless of which mask their original
            point had.
        n_range: (lo, hi) inclusive range for the number of holes K.
        r_range: (lo, hi) range for hole radius.
        max_holes_on_target: cap on holes centered on a target point.
            None disables the cap.
        generator: torch.Generator for reproducibility.

    Returns:
        (new_pcd_xyz, new_target_mask). `new_target_mask` is None iff
        the input mask was None.
    """
    P = pcd_xyz.shape[0]
    out_xyz = pcd_xyz.clone()
    out_mask = target_mask.clone() if target_mask is not None else None

    bbox_min = pcd_xyz.min(dim=0).values  # (3,)
    bbox_max = pcd_xyz.max(dim=0).values  # (3,)
    bbox_span = bbox_max - bbox_min        # (3,)

    n_lo, n_hi = n_range
    K = int(torch.randint(n_lo, n_hi + 1, (1,), generator=generator).item())

    holes_on_target = 0
    attempts_remaining_per_hole = 16  # bound rejection retries
    for _ in range(K):
        radius = (
            torch.rand(1, generator=generator).item() * (r_range[1] - r_range[0])
            + r_range[0]
        )

        # Pick a center point. If the cap is set, retry up to N times when
        # we'd exceed the cap.
        center_idx: Optional[int] = None
        for _retry in range(attempts_remaining_per_hole):
            cand = int(torch.randint(0, P, (1,), generator=generator).item())
            if (
                target_mask is not None
                and max_holes_on_target is not None
                and bool(target_mask[cand])
                and holes_on_target >= max_holes_on_target
            ):
                continue
            center_idx = cand
            if target_mask is not None and bool(target_mask[cand]):
                holes_on_target += 1
            break
        if center_idx is None:
            # All retries fell on target with cap exhausted; skip this hole.
            continue

        center_xyz = out_xyz[center_idx]
        within = (out_xyz - center_xyz).pow(2).sum(dim=-1).sqrt() < radius
        n_within = int(within.sum())
        if n_within == 0:
            continue

        replace = (
            torch.rand((n_within, 3), generator=generator) * bbox_span + bbox_min
        )
        out_xyz[within] = replace
        if out_mask is not None:
            out_mask[within] = False

    return out_xyz, out_mask


# --- Per-sample centering + max-radius scaling ------------------------------

def center_and_scale(
    pcd_xyz: torch.Tensor,
    *aux_xyz: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    """Subtract pcd centroid and divide by max radius from centroid.

    Args:
        pcd_xyz: (P, 3) float tensor.
        *aux_xyz: optional additional (..., 3) tensors (action xyz, etc.)
            that should receive the same affine.
        eps: floor on the scale to avoid divide-by-zero on degenerate pcds.

    Returns:
        (pcd_norm, aux_norm_tuple, centroid_3, scale_scalar).
    """
    centroid = pcd_xyz.mean(dim=0)              # (3,)
    centered = pcd_xyz - centroid
    radius = centered.norm(dim=-1).max()        # ()
    scale = torch.clamp(radius, min=eps)

    pcd_norm = centered / scale
    aux_norm = tuple((a - centroid) / scale for a in aux_xyz)
    return pcd_norm, aux_norm, centroid, scale


# --- Public façade for one sample -------------------------------------------

# Default knobs, matching the spec. Exposed as module constants so callers
# (datasets, tests) can import and override without re-declaring magic numbers.
DEFAULT_NOISE_SIGMA_MAX = 0.01     # 1 cm per axis, sampled per sample
DEFAULT_HOLE_N_RANGE = (2, 5)      # inclusive
DEFAULT_HOLE_R_RANGE = (0.02, 0.04)  # 2–4 cm radius
DEFAULT_TRANS_RANGE = 0.30         # ±30 cm per axis
DEFAULT_SHELF_MARGIN = 0.01        # 1 cm of slack below the geometric front
ROBOT_BASE_X_OFFSET = 0.615        # matches existing dataloaders


def augment_sample(
    pcd: torch.Tensor,                   # (P, C) where C ∈ {3, 4}
    action: torch.Tensor,                # (2, 8): grasp, place; xyz(3)+quat_wxyz(4)+gripper(1)
    shelf_y_threshold: float,            # in original world frame, BEFORE robot-base offset
    *,
    sigma_max: float = DEFAULT_NOISE_SIGMA_MAX,
    n_range: Tuple[int, int] = DEFAULT_HOLE_N_RANGE,
    r_range: Tuple[float, float] = DEFAULT_HOLE_R_RANGE,
    max_holes_on_target: Optional[int] = None,
    trans_range: float = DEFAULT_TRANS_RANGE,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full per-sample pipeline.

    Args:
        pcd: (P, C) tensor; C=3 (xyz only) or C=4 (xyz + target_mask).
        action: (2, 8) tensor; index 0 is grasp, index 1 is place.
        shelf_y_threshold: from `load_shelf_y_thresholds(...)[env_index]`.
            Compared against pcd y in the same world frame the .pth file
            stores (i.e. before the +0.615 robot-base x-offset).
        max_holes_on_target: cap; only meaningful when C=4. Use 1 for the
            grasp_place dataset, None for the value dataset.

    Returns:
        (pcd_out, action_out, centroid, scale).
        - pcd_out: (P, C) — same C as input. Position part centered+scaled.
        - action_out: (2, 8) — xyz centered+scaled, quats post-SE(3),
          gripper-state untouched.
        - centroid: (3,) for inversion at inference time.
        - scale: () for inversion at inference time.
    """
    if generator is None:
        generator = torch.default_generator

    pcd = pcd.clone()
    action = action.clone()

    pcd_xyz = pcd[..., :3]
    target_mask = pcd[..., 3] > 0.5 if pcd.shape[-1] >= 4 else None

    # 1. Identify shelf points (BEFORE any modification, in world frame).
    shelf_pt_mask = pcd_xyz[..., 1] > shelf_y_threshold

    # 2. Gaussian noise (BEFORE SE(3), per spec).
    pcd_xyz = add_gaussian_noise(pcd_xyz, sigma_max=sigma_max, generator=generator)

    # 3. Holes (also pre-SE(3); BBox is the noisy world-frame bbox).
    pcd_xyz, target_mask = add_holes(
        pcd_xyz,
        target_mask=target_mask,
        n_range=n_range,
        r_range=r_range,
        max_holes_on_target=max_holes_on_target,
        generator=generator,
    )

    # 4. Independent SE(3) transforms.
    R_obj, t_obj, q_obj = random_se3(
        trans_range=trans_range, generator=generator, yaw_only=True
    )
    R_shelf, t_shelf, q_shelf = random_se3(
        trans_range=trans_range, generator=generator, yaw_only=False
    )

    obj_pt_mask = ~shelf_pt_mask
    if obj_pt_mask.any():
        pcd_xyz[obj_pt_mask] = apply_se3_xyz(pcd_xyz[obj_pt_mask], R_obj, t_obj)
    if shelf_pt_mask.any():
        pcd_xyz[shelf_pt_mask] = apply_se3_xyz(pcd_xyz[shelf_pt_mask], R_shelf, t_shelf)

    # Action: grasp follows T_obj, place follows T_shelf.
    grasp_xyz = action[0, :3]
    place_xyz = action[1, :3]
    action[0, :3] = apply_se3_xyz(grasp_xyz, R_obj, t_obj)
    action[1, :3] = apply_se3_xyz(place_xyz, R_shelf, t_shelf)

    grasp_quat_wxyz = action[0, 3:7]
    place_quat_wxyz = action[1, 3:7]
    action[0, 3:7] = _quat_mul(q_obj, grasp_quat_wxyz)
    action[1, 3:7] = _quat_mul(q_shelf, place_quat_wxyz)

    # 5. +0.615 robot-base x-offset on every xyz.
    pcd_xyz[..., 0] = pcd_xyz[..., 0] + ROBOT_BASE_X_OFFSET
    action[0, 0] = action[0, 0] + ROBOT_BASE_X_OFFSET
    action[1, 0] = action[1, 0] + ROBOT_BASE_X_OFFSET

    # 6. Center + scale.
    pcd_xyz, (grasp_xyz_norm, place_xyz_norm), centroid, scale = center_and_scale(
        pcd_xyz,
        action[0, :3],
        action[1, :3],
    )
    action[0, :3] = grasp_xyz_norm
    action[1, :3] = place_xyz_norm

    # Reattach: pcd.x ← xyz; pcd.mask channel preserved untouched (target_mask
    # may have been mutated by add_holes — write back if present).
    pcd[..., :3] = pcd_xyz
    if target_mask is not None:
        pcd[..., 3] = target_mask.to(pcd.dtype)

    return pcd, action, centroid, scale
