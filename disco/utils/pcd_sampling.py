"""
Scene point cloud sampling for the policy and value inputs.
"""

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

_MASK_ACTORS_TO_IGNORE = ("table-workspace",)
_OBJID_ACTORS_TO_IGNORE = ("ground", "robot", "panda")


def sample_scene_with_target_mask(env, target_obj_idx, env_idx=0,
                                  num_points_per_actor=10000):
    """Sample surface points from each scene actor and mark the target object's points.

    Args:
        env: Shelf env with `actors` and `object_names`.
        target_obj_idx: Index of the target object in env.object_names.
        env_idx: Sim slot whose meshes are sampled.
        num_points_per_actor: Points per actor (equal per actor so small objects survive FPS).

    Returns:
        (M, 3) world-frame points and an (M,) mask that is 1.0 on target points.
    """
    target_name = env.object_names[target_obj_idx]

    pts_list, mask_list = [], []
    for actor in env.actors:
        if actor.name in _MASK_ACTORS_TO_IGNORE:
            continue
        mesh = actor.get_collision_meshes()[env_idx]
        pts, _ = trimesh.sample.sample_surface(mesh, num_points_per_actor)
        pts = np.asarray(pts, dtype=np.float32)
        is_target = float(actor.name == target_name)
        pts_list.append(pts)
        mask_list.append(np.full((pts.shape[0],), is_target, dtype=np.float32))

    points = np.concatenate(pts_list, axis=0)
    mask = np.concatenate(mask_list, axis=0)
    return points, mask


def downsample_pcd_with_mask(points, mask, num_points=4096):
    """Downsample points with farthest point sampling, carrying their mask values along.

    Args:
        points: (M, 3) points.
        mask: (M,) per-point mask values.
        num_points: Output size; pads with the last point if the input is smaller.

    Returns:
        (num_points, 4) array of xyz plus mask.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    mask = np.ascontiguousarray(mask, dtype=np.float32).reshape(-1)

    if points.shape[0] >= num_points:
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(points)
        down = o3d_pcd.farthest_point_down_sample(num_points)
        out_xyz = np.asarray(down.points, dtype=np.float32)

        tree = cKDTree(points)
        _, idx = tree.query(out_xyz, k=1)
        out_mask = mask[idx]
    else:
        # Pad-and-keep when there aren't enough points (rare — small mesh
        # corner case). Repeat the last point/mask to fill.
        pad_n = num_points - points.shape[0]
        pad_xyz = np.tile(points[-1:], (pad_n, 1))
        pad_mask = np.tile(mask[-1:], (pad_n,))
        out_xyz = np.concatenate([points, pad_xyz], axis=0)[:num_points]
        out_mask = np.concatenate([mask, pad_mask], axis=0)[:num_points]

    return np.concatenate([out_xyz, out_mask[:, None]], axis=-1)


def sample_scene_with_obj_ids(env, env_idx=0, num_points_per_actor=10000,
                              extra_ignore=()):
    """Sample surface points from each scene actor, tagged with the actor's object index.

    Args:
        env: Shelf env with `actors` and `object_names`.
        env_idx: Sim slot whose meshes are sampled.
        num_points_per_actor: Points sampled per actor.
        extra_ignore: Extra name substrings of actors to skip, e.g. ("table",).

    Returns:
        (M, 3) float32 points and (M,) int32 ids (index into env.object_names, -1 for shelf etc.).
    """
    name_to_idx = {n: i for i, n in enumerate(env.object_names)}
    ignore = _OBJID_ACTORS_TO_IGNORE + tuple(extra_ignore)

    pts_list, ids_list = [], []
    for actor in env.actors:
        if any(skip in actor.name for skip in ignore):
            continue
        mesh = actor.get_collision_meshes()[env_idx]
        pts, _ = trimesh.sample.sample_surface(mesh, num_points_per_actor)
        pts = np.asarray(pts, dtype=np.float32)
        oi = int(name_to_idx.get(actor.name, -1))
        pts_list.append(pts)
        ids_list.append(np.full((pts.shape[0],), oi, dtype=np.int32))

    if not pts_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    points = np.concatenate(pts_list, axis=0)
    obj_ids = np.concatenate(ids_list, axis=0)
    return points, obj_ids


def downsample_pcd_xyz(points, num_points=4096):
    """Downsample (M, 3) points to (num_points, 3) with farthest point sampling.

    Args:
        points: (M, 3) points.
        num_points: Output size; pads with the last point if the input is smaller.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)

    if points.shape[0] >= num_points:
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(points)
        down = o3d_pcd.farthest_point_down_sample(num_points)
        out_xyz = np.asarray(down.points, dtype=np.float32)
    else:
        pad_n = num_points - points.shape[0]
        pad_xyz = np.tile(points[-1:], (pad_n, 1))
        out_xyz = np.concatenate([points, pad_xyz], axis=0)[:num_points]

    return out_xyz


def shelved_object_indices(env, env_idx=0):
    """Return the set of object indices currently inside the shelf.

    Set the env to the state of interest before calling.

    Args:
        env: Shelf env.
        env_idx: Sim slot to check.
    """
    out = set()
    for i, name in enumerate(env.object_names):
        if env.check_object_inside_shelf(name, env_idx=env_idx):
            out.add(int(i))
    return out


def filter_pcd_target_only(scene_xyz, scene_obj_ids, target_obj_idx, shelved_idx_set):
    """Keep only points of the target, the shelved objects, and non-object actors.

    Args:
        scene_xyz: (M, 3) scene points.
        scene_obj_ids: (M,) object index per point (-1 for non-object actors).
        target_obj_idx: Index of the target object.
        shelved_idx_set: Set of indices of objects already in the shelf.

    Returns:
        (K, 3) filtered points.
    """
    keep = (scene_obj_ids == -1) | (scene_obj_ids == int(target_obj_idx))
    if shelved_idx_set:
        # In-place OR over each shelved index.
        for oi in shelved_idx_set:
            keep |= (scene_obj_ids == int(oi))
    return scene_xyz[keep]
