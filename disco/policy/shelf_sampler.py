"""
Shelf packing action sampler: grasp and place candidates from the policies.
"""
import json
import os
import copy
import time
import sapien
from sympy.core import I
import torch
import shutil
import numpy as np
from typing import Tuple
from omegaconf import OmegaConf

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.envs.utils import randomization
# from plan_scene_builder import TableSceneBuilder
from mani_skill.utils.building import articulations, actors
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.structs.pose import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from sapien.physx import PhysxRigidBodyComponent

from disco.envs.robot_controller import RobotController
from disco.motion.action_primitives import ActionPrimitives
from disco.envs.env_utils import ManiSkillEnvUtils


def _policy_autocast_dtype():
    """Return the policy autocast dtype from POLICY_DTYPE (None for fp32; default bf16)."""
    s = os.environ.get("POLICY_DTYPE", "bf16").lower()
    if s in ("fp32", "float32"):
        return None
    if s in ("fp16", "half", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16", ""):
        return torch.bfloat16
    raise ValueError(f"POLICY_DTYPE='{s}' not in {{fp32, fp16, bf16}}")

import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
from tqdm import tqdm
from disco.utils.common import (
    load_grasps, 
    to_numpy_pose, 
    to_sapien_pose, 
    to_torch_pose,
    flip_quat_about_z,
    flip_pose_quat_about_z,
    load_acronym_object_and_grasps,
    point_in_cuboid,
    generate_placement_quaternions
)

from disco.utils.visualization.plotting import (
    plot_pcd, 
    plot_pcd_with_highlighted_segment
)
from disco.utils.visualization.point_cloud_structures import make_gripper_visualization
from disco.utils.conversion_utils import (
    pose_to_transformation, 
    invert_transformation, 
    transform_pcd, 
    transformation_to_pose,
    move_pose_along_local_z,
    move_pose_along_local_x,
    move_pose_along_local_y,
    furthest_point_sample
)
from disco.flowmatch.utils.common_utils import count_parameters

from disco.envs.generation_utils import (
    create_shelf, 
    sample_object_pose_on_table, 
    sample_object_pose_on_table_multi_object,
    sample_object_pose_on_shelf, 
    sample_point_in_fixed_rectangle, 
    sample_point_in_fixed_rectangle_uniformly,
    compute_ray_box_intersection
)

POST_GRASP_LIFT = 0.15
# GRASP_DEPTH = 0.2
GRASP_DEPTH = 0.25
GRASP_RETRACTION_DISTANCE = 0.15
EE_LINK_CENTER_TO_GRIPPER_TIP = 0.13

CUROBO_SUCCESSES = 0

OPEN = 1
CLOSED = -1

_POLICY_KWARGS = dict(
    embedding_dim=120,
    num_attn_heads=8,
    nhist=1,
    num_shared_attn_layers=4,
    relative=False,
    rotation_format="quat_wxyz",
    denoise_timesteps=10,
    denoise_model="rectified_flow",
    lv2_batch_size=1,
)


def _build_policy(policy_type):
    """Build an untrained policy network of the given type.

    Args:
        policy_type: "placement" (single-pose place), "grasp_only" (cascaded
            grasp), or "place_conditional" (cascaded place given a grasp).
    """
    if policy_type == "placement":
        from disco.flowmatch.modeling.policy.denoise_actor_3d_packing import (
            DenoiseActor as PlacementActor,
        )
        return PlacementActor(**_POLICY_KWARGS)
    if policy_type in ("grasp_only", "place_conditional"):
        from disco.flowmatch.modeling.policy_grasp_place.denoise_actor_3d_packing import (
            DenoiseActor as GraspPlaceActor,
        )
        kwargs = dict(_POLICY_KWARGS)
        kwargs["trajectory_length"] = 1
        kwargs["grasp_condition_dim"] = 7 if policy_type == "place_conditional" else None
        # Cascaded grasp/place pair is trained on the target-only 3-channel
        # PCD (xyz, with non-target non-shelved objects removed; see
        # save_successful_grasp_place_target_only.py).
        kwargs["pcd_input_channels"] = 3
        return GraspPlaceActor(**kwargs)
    raise ValueError(
        f"Unknown policy_type '{policy_type}'. Expected one of "
        "'placement', 'grasp_only', 'place_conditional'."
    )


class ActionSampler:
    """
    Action sampling for robot manipulation using cuRobo motion planning.

    Owns the trained sampling policies (loaded via `load_policy`):
      * "placement"        – single-pose model used by `sample_placement_poses`.
                             Stored at `self.placement_model`.
      * "grasp_only"       – cascaded grasp model. Stored at
                             `self.grasp_only_model`.
      * "place_conditional" – cascaded place model (needs a clean grasp
                              as conditioning at inference). Stored at
                              `self.place_conditional_model`.

    `sample_grasp_place_actions` requires both cascaded models to be loaded.
    """

    def __init__(self, env):
        """Store the env and start with no policies loaded.

        Args:
            env: Shelf-packing env with stored grasps and motion primitives.
        """
        self.env = env

        self.placement_model = None
        self.grasp_only_model = None
        self.place_conditional_model = None
        self.policy_type = None
        self.policy_device = None

    # ------------------------ POLICY LOADING ------------------------ #

    def load_policy(self, policy_type, model_path, device="cuda:0"):
        """Load a trained policy checkpoint and store it on the sampler.

        Args:
            policy_type: "placement", "grasp_only", or "place_conditional".
            model_path: Path to the .pth checkpoint.
            device: Torch device string, e.g. "cuda:0".

        Returns:
            The loaded model in eval mode.
        """
        # Lazy import to avoid the action_sampling <-> shelf_packing_base
        # import cycle (shelf_packing_base imports ActionSampler at top level).
        from disco.envs.shelf_packing_base import load_checkpoint_for_eval

        device = torch.device(device)
        model = _build_policy(policy_type)

        count_parameters(model)
        for name, param in model.named_parameters():
            if param.requires_grad and param.ndim > 1 and not param.is_contiguous():
                print(f"Fixing layout for: {name}")
                param.data = param.contiguous()

        print(f"Loading {policy_type} policy from {model_path} onto {device}")
        model = load_checkpoint_for_eval(model_path, model)
        model = model.to(device).eval()

        if policy_type == "placement":
            self.placement_model = model
        elif policy_type == "grasp_only":
            self.grasp_only_model = model
        elif policy_type == "place_conditional":
            self.place_conditional_model = model
        else:
            raise ValueError(f"Unknown policy_type '{policy_type}'")

        self.policy_type = policy_type
        self.policy_device = device

        return model

    # ------------------------ GRASP SAMPLING ------------------------ #

    def get_object_grasp_poses(self, object_name, env_idx: int = 0, grasp_poses=None):
        """Transform an object's grasps from the object frame to the world frame.

        Args:
            object_name: Name of the object.
            env_idx: Sim slot to read the object pose from.
            grasp_poses: (N, 7) object-frame grasps (xyz, wxyz quat); defaults to the env's stored grasps.

        Returns:
            (N, 7) world-frame grasp poses on cuda:0.
        """

        if grasp_poses is None:
            grasp_poses = self.env.grasps[object_name]

        object_pose = self.env.get_object_pose(object_name)[env_idx].cpu().detach().numpy()

        # Using conversion utils:
        grasp_transformation = pose_to_transformation(grasp_poses, format='wxyz')
        object_transformation = pose_to_transformation(object_pose, format='wxyz')
        grasp_transformation = object_transformation @ grasp_transformation
        transformed_grasp_poses = transformation_to_pose(grasp_transformation, format='wxyz')

        transformed_grasp_poses = torch.tensor(transformed_grasp_poses, dtype=torch.float32, device="cuda:0")

        return transformed_grasp_poses

    def sample_grasp_poses(self):
        """Collect IK-feasible stored grasps for every object not yet on the shelf.

        Returns:
            (N, 9) tensor (xyz, wxyz quat, grasp cost, object index), shuffled and interleaved across objects.
        """

        grasp_poses = []
        grasp_costs = []
        grasp_object_ids = []

        for object_index in range(len(self.env.object_names)):
            # Get the transformed grasp pose tensors for the specified object:
            object_name = self.env.object_names[object_index]

            # Skip objects that are already in the shelf
            if self.env.check_object_inside_shelf(object_name, env_idx=0):
                continue

            obj_grasp_poses = self.get_object_grasp_poses(
                object_name,
                env_idx=0,
                grasp_poses=self.env.grasps[object_name]
            )
            grasp_poses.append(torch.tensor(obj_grasp_poses, device="cuda:0", dtype=torch.float32))
            grasp_costs.append(torch.tensor(self.env.grasp_costs[object_name], device="cuda:0", dtype=torch.float32))
            grasp_object_ids.append(torch.full((obj_grasp_poses.shape[0],), object_index, device="cuda:0", dtype=torch.int64))

        if len(grasp_poses) == 0:
            return torch.zeros((0, 7), device="cuda:0", dtype=torch.float32)

        grasp_poses = torch.cat(grasp_poses, dim=0)
        grasp_costs = torch.cat(grasp_costs, dim=0)
        grasp_object_ids = torch.cat(grasp_object_ids, dim=0)

        print(f"Doing IK for grasps")
        curobo_grasp_poses = grasp_poses.clone()
        curobo_grasp_poses[..., 0] = curobo_grasp_poses[..., 0] + 0.615
        _, success = self.env.action_primitives.inverse_kinematics(curobo_grasp_poses)

        feasible_indices = torch.where(success[:, 0])[0]

        if len(feasible_indices) == 0:
            return torch.zeros((0, 7), device="cuda:0", dtype=torch.float32)

        grasp_poses = grasp_poses[feasible_indices]
        grasp_costs = grasp_costs[feasible_indices].unsqueeze(-1)
        grasp_object_ids = grasp_object_ids[feasible_indices].unsqueeze(-1)

        # self.env.visualize_grasp_poses(grasp_poses)

        sampled_actions = torch.cat([grasp_poses, grasp_costs, grasp_object_ids], dim=-1)

        # Shuffle grasps per object, then interleave round-robin across objects
        # (so selection visits object 0, 1, 2, 0, 1, 2, ...; exhausted objects drop out).
        obj_ids_column = sampled_actions[:, -1].long()
        unique_obj_ids = torch.unique(obj_ids_column).tolist()
        per_object_rows = []
        for oid in unique_obj_ids:
            mask = obj_ids_column == oid
            rows = sampled_actions[mask]
            perm = torch.randperm(rows.shape[0], device=rows.device)
            per_object_rows.append(rows[perm])

        interleaved = []
        cursors = [0] * len(per_object_rows)
        while True:
            advanced = False
            for i, rows in enumerate(per_object_rows):
                if cursors[i] < rows.shape[0]:
                    interleaved.append(rows[cursors[i]])
                    cursors[i] += 1
                    advanced = True
            if not advanced:
                break
        sampled_actions = torch.stack(interleaved, dim=0)

        # (N, 9) tensor, with (x, y, z, qw, qx, qy, qz, grasp_cost, grasp_object_id)
        return sampled_actions

    def sample_grasp_poses_debug(self):
        """Return two hard-coded grasps in the `sample_grasp_poses` format, for debugging."""

        # (N, 9): (x, y, z, qw, qx, qy, qz, grasp_cost, grasp_object_id) — same as sample_grasp_poses return
        sampled_actions = torch.tensor(
            [
                [-0.2881, -0.0614, 0.1776, 0.0070, 0.9994, -0.0339, -0.0070, 0.0051, 1.0],
                [-0.1489, -0.3682, 0.2172, -0.0659, 0.3871, 0.9173, 0.0658, 1.1446, 0.0],
            ],
            device="cuda:0",
            dtype=torch.float32,
        )
        return sampled_actions
    
    # ------------------------ PLACEMENT SAMPLING ------------------------ #

    def sample_points_in_shelf(
        self,
        # Depth and width tolerance are based on the Franka Gripper dimensions. 
        # The gripper cylinder diameter is 0.12, the hand width is 0.2, and finger to hand center is 0.105 (here I use 0.08)
        depth_tolerance=0.08,
        width_tolerance=0.2,
        extension_distance=0.,
        height_tolerance=0.08,
        n = 100,
        ):
        """Sample random points inside the shelf volume, shrunk by the given margins.

        Args:
            depth_tolerance: Margin subtracted from the shelf depth (m).
            width_tolerance: Margin subtracted from the shelf width (m).
            extension_distance: Unused.
            height_tolerance: Margin kept from the shelf floor and ceiling (m).
            n: Number of points.

        Returns:
            (n, 3) numpy array of world-frame points.
        """

        center_points = sample_point_in_fixed_rectangle(
            fixed_rect_center=(self.env.shelf_position[0], self.env.shelf_position[1]),
            fixed_rect_dimensions=(self.env.shelf_depth - depth_tolerance, self.env.shelf_width - width_tolerance),
            fixed_rect_angle=self.env.shelf_rotation, # + np.pi/2,
            num_points=n
        )
        cz = self.env.shelf_distance_from_floor + np.random.uniform(height_tolerance, self.env.shelf_height - height_tolerance, size=n)
        cz = np.clip(
            cz, 
            self.env.shelf_distance_from_floor + height_tolerance, 
            self.env.shelf_distance_from_floor + self.env.shelf_height - height_tolerance
            )[:, np.newaxis]

        points = np.concatenate([center_points, cz], axis=-1)

        return points

    def sample_points_outside_shelf(
        self,
        extension_distance=0.2,
        height_tolerance=0.08,
        width_tolerance=0.1,
        n = 100,
        ):
        """Sample random points in a strip along the open front of the shelf.

        Args:
            extension_distance: Depth of the strip (m).
            height_tolerance: Margin kept from the shelf floor and ceiling (m).
            width_tolerance: Margin subtracted from the shelf width (m).
            n: Number of points.

        Returns:
            (n, 3) numpy array of world-frame points.
        """

        direction_vec = np.array([np.cos(self.env.shelf_rotation), np.sin(self.env.shelf_rotation)])
        new_shelf_center = self.env.shelf_position[:2] - (((extension_distance / 2) + self.env.shelf_depth) / 2) * direction_vec
        
        center_points = sample_point_in_fixed_rectangle(
            fixed_rect_center=(new_shelf_center[0], new_shelf_center[1]),
            fixed_rect_dimensions=(extension_distance, self.env.shelf_width - width_tolerance),
            fixed_rect_angle=self.env.shelf_rotation, # + np.pi/2,
            num_points=n
        )
        cz = self.env.shelf_distance_from_floor + np.random.uniform(height_tolerance, self.env.shelf_height - height_tolerance, size=n)
        cz = np.clip(
            cz, 
            self.env.shelf_distance_from_floor + height_tolerance, 
            self.env.shelf_distance_from_floor + self.env.shelf_height - height_tolerance
            )[:, np.newaxis]

        points = np.concatenate([center_points, cz], axis=-1)

        return points

    def sample_gripper_poses_in_shelf(
        self, 
        n = 1, 
        object_name = None,
        depth_tolerance=0.05,
        width_tolerance=0.1,
        height_tolerance=0.08
        ):
        """Sample gripper poses inside the shelf, facing into it with small random rotations.

        Args:
            n: Number of poses.
            object_name: Unused.
            depth_tolerance: Unused (default shelf margins apply).
            width_tolerance: Unused.
            height_tolerance: Unused.

        Returns:
            (n, 7) world-frame poses (xyz, wxyz quat) on cuda:0, backed off from the gripper tip to the EE link.
        """

        points = self.sample_points_in_shelf(n = n)
        points = torch.tensor(points, device="cuda:0", dtype=torch.float32)

        rotation_to_orient_with_shelf = R.from_euler('yz', [np.pi/2, self.env.shelf_rotation])
        base_quaternion = rotation_to_orient_with_shelf.as_quat().squeeze()[[3,0,1,2]]
        base_quaternion = flip_quat_about_z(base_quaternion)  # Otherwise the object is upside down.
        
        # Apply random rotations: X rotation (-25 to +25 deg), then Z rotation (-5 to +5 deg)
        quaternions = []
        for i in range(n):
            # Sample random angles
            x_angle_deg = np.random.uniform(-25, 25)
            z_angle_deg = np.random.uniform(-5, 5)
            
            # Convert base quaternion to Rotation object
            base_rot = R.from_quat(base_quaternion[[1, 2, 3, 0]])  # Convert from wxyz to xyzw
            
            # Apply X rotation, then Z rotation
            x_rot = R.from_euler('x', x_angle_deg, degrees=True)
            z_rot = R.from_euler('z', z_angle_deg, degrees=True)
            combined_rot = base_rot * x_rot * z_rot  # Apply X first, then Z
            
            # Convert back to wxyz format
            quat_xyzw = combined_rot.as_quat()  # Returns [x, y, z, w]
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # Convert to [w, x, y, z]
            quaternions.append(quat_wxyz)
        
        quaternions = np.array(quaternions)  # (n, 4)
        quaternions = torch.tensor(quaternions, device="cuda:0", dtype=torch.float32)

        shelf_poses = torch.zeros((n, 7), device="cuda:0", dtype=torch.float32)
        shelf_poses[:, :3] = points
        shelf_poses[:, 3:] = quaternions

        curobo_shelf_poses = move_pose_along_local_z(shelf_poses, -EE_LINK_CENTER_TO_GRIPPER_TIP)

        return curobo_shelf_poses

        # curobo_shelf_poses = torch.tensor(shelf_poses, device="cuda:0", dtype=torch.float32)
        # curobo_shelf_poses = shelf_poses.clone()
        # curobo_shelf_poses[..., 0] = curobo_shelf_poses[..., 0] + 0.615

        # if object_name is not None:
        #     self.action_primitives.attach_object_to_robot(object_name)

        # shelf_joints, ik_successes = self.action_primitives.inverse_kinematics(curobo_shelf_poses)

        # if object_name is not None:
        #     self.action_primitives.detach_object_from_robot(object_name)

        # shelf_poses = shelf_poses[torch.where(ik_successes[:, 0])[0]]

        # return shelf_poses, shelf_joints
    
    def sample_gripper_poses_outside_shelf(
        self, 
        n = 1, 
        object_name = None,
        extension_distance=0.12,
        height_tolerance=0.08,
        width_tolerance=0.1,
        ):
        """Sample gripper poses at the front of the shelf with random placement orientations.

        Args:
            n: Number of poses.
            object_name: Unused.
            extension_distance: Depth of the sampling strip (m).
            height_tolerance: Margin kept from the shelf floor and ceiling (m).
            width_tolerance: Margin subtracted from the shelf width (m).

        Returns:
            (n, 7) world-frame poses (xyz, wxyz quat) on cuda:0, backed off from the gripper tip to the EE link.
        """

        points = self.sample_points_outside_shelf(
            extension_distance=extension_distance,
            height_tolerance=height_tolerance,
            width_tolerance=width_tolerance,
            n = n
        )
        points = torch.tensor(points, device="cuda:0", dtype=torch.float32)

        # Get quaternions from generate_placement_quaternions
        quaternions = generate_placement_quaternions(num_quaternions=n, target_xy_angle=self.env.shelf_rotation)  # (n, 4) in wxyz format
        quaternions = torch.tensor(quaternions, device="cuda:0", dtype=torch.float32)

        shelf_poses = torch.zeros((n, 7), device="cuda:0", dtype=torch.float32)
        shelf_poses[:, :3] = points
        shelf_poses[:, 3:] = quaternions

        curobo_shelf_poses = move_pose_along_local_z(shelf_poses, -EE_LINK_CENTER_TO_GRIPPER_TIP)

        return curobo_shelf_poses    

    def sample_placement_poses(self, object_index: int, batch_size: int = 1):
        """Predict shelf placement poses for the held object with the placement policy.

        Slot 0 must hold the post-grasp state; outputs differ from the grasp only by a world-Z rotation.

        Args:
            object_index: Index of the held object in env.object_names.
            batch_size: Number of placements to sample.

        Returns:
            (N, 8) world-frame placements (xyz, wxyz quat, object index).
        """

        # TODO: Environment should already be reset to the desired state before calling this function.
        # TODO: So environment 0 should be the one from which everything is loaded.

        object_name = self.env.object_names[object_index]
        num_points = 4096   # TODO: This is hardcoded
        shelf_pcd = self.env.get_sampled_actor_point_cloud(self.env.shelf, num_points = num_points // 2)
        object_pcd = self.env.get_sampled_actor_point_cloud(self.env.objects[object_name], env_idx = 0, num_points = num_points // 2)

        input_pcd = np.concatenate([shelf_pcd, object_pcd], axis=0)
        input_pcd = torch.tensor(input_pcd, device="cuda:0", dtype=torch.float32)
        input_pcd = input_pcd.unsqueeze(0).repeat(batch_size, 1, 1)
        input_pcd[..., 0] = input_pcd[..., 0] + 0.615

        gripper_pose = self.env.action_primitives.fk(self.env.get_joint_state()[0].to("cuda:0"))
        current_gripper_quat_wxyz = gripper_pose[3:7].cpu().numpy()  # Save before overwrite
        gripper_pose = gripper_pose.unsqueeze(0).repeat(batch_size, 1, 1)
        zeros = torch.zeros((batch_size, 1, 1), device=gripper_pose.device, dtype=gripper_pose.dtype)   # For the gripper being closed
        gripper_pose = torch.cat([gripper_pose, zeros], dim=-1)

        max_batch_size = 4      # Reduce according to available CUDA memory
        all_placement_poses = []
        
        with torch.no_grad():
            for i in range(0, batch_size, max_batch_size):
                end_idx = min(i + max_batch_size, batch_size)
                chunk_pcd = input_pcd[i:end_idx]
                chunk_proprioception = gripper_pose[i:end_idx]
                
                placement_poses_chunk = self.placement_model(
                    gt_trajectory = None,
                    pcd = chunk_pcd,
                    proprioception = chunk_proprioception,
                    run_inference = True
                )
                
                placement_poses_chunk = placement_poses_chunk.squeeze(0)
                all_placement_poses.append(placement_poses_chunk)
        
        placement_poses = torch.cat(all_placement_poses, dim=0)
        placement_poses[..., 0] = placement_poses[..., 0] - 0.615
        placement_poses = placement_poses[:, 0, :7]     # Extract only the pose part, and squeeze the horizon dimension

        # Correct placement orientation: find the smallest local-Y rotation of the
        # placement pose such that the relative rotation to the grasp pose is purely
        # around world Z.
        #
        # Condition: R_corrected * R_grasp^-1 maps world Z to world Z.
        # With R_corrected = R_place * Ry(theta), this gives:
        #   Ry(theta) * g_z = p_z
        # where g_z = R_grasp^T @ [0,0,1] and p_z = R_place^T @ [0,0,1]
        # (world Z expressed in grasp and placement local frames).
        #
        # Solving the x,z components: theta = atan2(gz*px - gx*pz, gx*px + gz*pz)

        grasp_quat_xyzw = current_gripper_quat_wxyz[[1, 2, 3, 0]]
        R_grasp_mat = R.from_quat(grasp_quat_xyzw).as_matrix()  # (3, 3)

        place_quats_wxyz = placement_poses[:, 3:7].cpu().numpy()
        place_quats_xyzw = place_quats_wxyz[:, [1, 2, 3, 0]]
        R_place_scipy = R.from_quat(place_quats_xyzw)
        R_place_mats = R_place_scipy.as_matrix()  # (N, 3, 3)

        world_z = np.array([0.0, 0.0, 1.0])
        g_z = R_grasp_mat.T @ world_z  # (3,) world Z in grasp local frame
        p_z = np.einsum('nij,j->ni', R_place_mats.transpose(0, 2, 1), world_z)  # (N, 3) world Z in each placement local frame

        gx, gz = g_z[0], g_z[2]
        px, pz = p_z[:, 0], p_z[:, 2]

        theta = np.arctan2(gz * px - gx * pz, gx * px + gz * pz)

        R_y_corrections = R.from_euler('y', theta)  # (N,) local Y rotations
        R_corrected = R_place_scipy * R_y_corrections  # R_place * Ry(theta)

        # Fix backward-facing poses: if relative yaw to grasp > pi/2, flip by pi
        # around world Z (preserves the world-Z-only constraint)
        R_grasp_scipy = R.from_quat(grasp_quat_xyzw)
        R_rel = R_corrected * R_grasp_scipy.inv()
        rel_yaw = R_rel.as_euler('ZYX')[:, 0]
        flip_angles = np.where(np.abs(rel_yaw) > np.pi / 2, np.pi, 0.0)
        R_corrected = R.from_euler('Z', flip_angles) * R_corrected

        corrected_quats_xyzw = R_corrected.as_quat()  # (N, 4) xyzw
        corrected_quats_wxyz = corrected_quats_xyzw[:, [3, 0, 1, 2]]  # (N, 4) wxyz
        placement_poses[:, 3:7] = torch.tensor(corrected_quats_wxyz, device=placement_poses.device, dtype=placement_poses.dtype)

        # Add object_index at the end
        object_indices = torch.full((placement_poses.shape[0], 1), float(object_index), device=placement_poses.device, dtype=placement_poses.dtype)
        placement_poses = torch.cat([placement_poses, object_indices], dim=-1)

        # (N, 8) tensor, with (x, y, z, qw, qx, qy, qz, object_index)
        return placement_poses

    def execute_all_grasps(self, all_grasp_params, base_state, num_envs):
        """Run Pick on all candidate grasps in batches of num_envs sim slots.

        Args:
            all_grasp_params: (N, 9) world-frame grasps (xyz, wxyz quat, cost, object index).
            base_state: (state_dim,) sim state every batch starts from.
            num_envs: Number of parallel sim slots (32).

        Returns:
            Dict grasp index -> {state, trajectory, action} for successful picks,
            set of failed grasp indices, and failed-node result dicts for grasp-execution failures.
        """
        env = self.env
        Pick = env.action_primitives.Pick

        grasp_results = {}
        failed_grasp_indices = set()
        execution_failure_results = []

        num_grasps = all_grasp_params.shape[0]

        for batch_start in range(0, num_grasps, num_envs):
            batch_end = min(batch_start + num_envs, num_grasps)
            grasp_batch = all_grasp_params[batch_start:batch_end]
            batch_size = grasp_batch.shape[0]

            # Reset all envs to base state before each batch
            env.set_state(base_state.unsqueeze(0).repeat(num_envs, 1))
            if env.render_mode is not None:
                env.render()

            pick_result = Pick(grasp_batch)
            successful_states, pick_trajectories, exec_actions, _, pick_failures = pick_result

            # Identify which in-batch indices failed
            failed_batch_indices = set()
            for action, failure_type, state, traj in pick_failures:
                for j in range(batch_size):
                    if j not in failed_batch_indices and torch.equal(action, grasp_batch[j]):
                        failed_batch_indices.add(j)
                        grasp_idx = batch_start + j
                        failed_grasp_indices.add(grasp_idx)

                        if failure_type == "grasp_execution":
                            combined_action = torch.cat([
                                action,
                                torch.zeros(8, device=action.device, dtype=action.dtype),
                            ])
                            execution_failure_results.append({
                                'state': state,
                                'action': combined_action,
                                'trajectory': traj,
                                'reward': 0.0,
                                'is_failed': True,
                                'failure_type': failure_type,
                            })
                        break

            # Map successful results back to global grasp indices
            success_batch_indices = [j for j in range(batch_size) if j not in failed_batch_indices]
            for i, j in enumerate(success_batch_indices):
                grasp_idx = batch_start + j
                grasp_results[grasp_idx] = {
                    'state': successful_states[i],
                    'trajectory': pick_trajectories[i],
                    'action': exec_actions[i],
                }

        # Reset to base state
        env.set_state(base_state.unsqueeze(0).repeat(num_envs, 1))
        if env.render_mode is not None:
            env.render()

        return grasp_results, failed_grasp_indices, execution_failure_results

    def _run_placement_batch(self, place_params, post_grasp_state, pick_trajectory,
                              grasp_action, num_envs):
        """Run Place on placement candidates in batches, all starting from one post-grasp state.

        Args:
            place_params: (N, 8) world-frame placements (xyz, wxyz quat, object index).
            post_grasp_state: (state_dim,) sim state right after the grasp.
            pick_trajectory: Pick joint trajectory, prepended to each place trajectory.
            grasp_action: (9,) executed grasp params, prepended to each place action.
            num_envs: Number of parallel sim slots.

        Returns:
            (successes, failures) lists of result dicts; failures cover place-execution errors only.
        """
        env = self.env
        Place = env.action_primitives.Place
        successes, failures = [], []

        for batch_start in range(0, len(place_params), num_envs):
            batch_end = min(batch_start + num_envs, len(place_params))
            place_batch = place_params[batch_start:batch_end]

            env.set_state(post_grasp_state.unsqueeze(0).repeat(num_envs, 1))
            if env.render_mode is not None:
                env.render()

            p_states, p_trajs, p_actions, p_rewards, p_failures = Place(place_batch)

            grasp7 = grasp_action[:7].detach().cpu().float()

            for i in range(len(p_states)):
                combined_action = torch.cat([grasp_action, p_actions[i]])
                combined_traj = np.concatenate([pick_trajectory, p_trajs[i]], axis=0)
                place7 = p_actions[i][:7].detach().cpu().float()
                successes.append({
                    'state': p_states[i],
                    'action': combined_action,
                    'action_2x8': ActionSampler._make_action_2x8(grasp7, place7),
                    'trajectory': combined_traj,
                    'reward': p_rewards[i].item(),
                    'is_failed': False,
                    'failure_type': None,
                })

            for action, failure_type, state, traj in p_failures:
                if failure_type == "place_execution":
                    combined_action = torch.cat([grasp_action, action])
                    combined_traj = (
                        np.concatenate([pick_trajectory, traj], axis=0)
                        if traj is not None
                        else pick_trajectory.copy()
                    )
                    place7 = action[:7].detach().cpu().float()
                    failures.append({
                        'state': state,
                        'action': combined_action,
                        'action_2x8': ActionSampler._make_action_2x8(grasp7, place7),
                        'trajectory': combined_traj,
                        'reward': 0.0,
                        'is_failed': True,
                        'failure_type': failure_type,
                    })

        return successes, failures

    def _rotate_place_params_local_y(self, place_params, angle):
        """Return a copy of the placements rotated about each pose's local Y axis.

        Args:
            place_params: (N, 8) placements with wxyz quaternions in columns 3:7.
            angle: Rotation angle (rad).
        """
        quats_wxyz = place_params[:, 3:7].cpu().numpy()
        quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
        R_place = R.from_quat(quats_xyzw)
        R_rot = R.from_euler('y', angle)
        R_new = R_place * R_rot  # right-multiply = local-Y rotation

        new_quats_xyzw = R_new.as_quat()
        new_quats_wxyz = new_quats_xyzw[:, [3, 0, 1, 2]]

        new_params = place_params.clone()
        new_params[:, 3:7] = torch.tensor(
            new_quats_wxyz, dtype=place_params.dtype, device=place_params.device,
        )
        return new_params

    def execute_placements_for_grasp(self, grasp_info, base_state, num_envs, num_placements=16):
        """Sample and execute placements for one successful grasp.

        If none succeed, retries with the placements rotated ±90° about local Y.

        Args:
            grasp_info: {state, trajectory, action} entry from execute_all_grasps.
            base_state: (state_dim,) node state restored at the end.
            num_envs: Number of parallel sim slots.
            num_placements: Number of placements to sample.

        Returns:
            List of result dicts, successes first.
        """
        env = self.env
        post_grasp_state = grasp_info['state']
        pick_trajectory = grasp_info['trajectory']
        grasp_action = grasp_info['action']

        object_index = int(grasp_action[-1].item())

        # Reset to post-grasp state so sample_placement_poses sees the correct
        # point cloud (object in gripper), proprioception (lifted pose), and
        # gripper orientation (for placement orientation correction).
        env.set_state(post_grasp_state.unsqueeze(0).repeat(num_envs, 1))
        if env.render_mode is not None:
            env.render()

        place_params = self.sample_placement_poses(object_index, batch_size=num_placements)

        successes, failures = self._run_placement_batch(
            place_params, post_grasp_state, pick_trajectory, grasp_action, num_envs,
        )

        if len(successes) == 0:
            for angle in (np.pi / 2, -np.pi / 2):
                rotated_params = self._rotate_place_params_local_y(place_params, angle)
                retry_successes, retry_failures = self._run_placement_batch(
                    rotated_params, post_grasp_state, pick_trajectory, grasp_action, num_envs,
                )
                if len(retry_successes) > 0:
                    successes, failures = retry_successes, retry_failures
                    break

        env.set_state(base_state.unsqueeze(0).repeat(num_envs, 1))
        if env.render_mode is not None:
            env.render()

        return successes + failures

    # ------------------------ GRASP+PLACE POLICY SAMPLING ------------------------ #
    #
    # Used by MCTS expansion (disco/search/expansion.py). Samples (grasp,
    # place) candidates from the trained 2-pose policy, IK pre-filters, and
    # runs them in parallel. Reuses `execute_all_grasps` for Pick batching;
    # adds a sibling helper `_run_placement_batch_per_env_state` for Place
    # because the existing `_run_placement_batch` takes a single shared
    # post-grasp state, while here every env starts from its own (since
    # each candidate has its own grasp).

    # gripper-bit convention used during dataset gen
    # (save_successful_grasp_place.extract_grasp_place_tokens):
    #   grasp = 1.0 (closed after grasp), place = 0.0 (open after place)
    _GRIPPER_CLOSED = 1.0
    _GRIPPER_OPEN = 0.0
    _ROBOT_BASE_X_OFFSET = 0.615
    _PCD_NUM_POINTS = 4096

    @staticmethod
    def _make_action_2x8(grasp7, place7):
        """Build the (2, 8) grasp/place action token used by the value network.

        Args:
            grasp7: (7,) world-frame grasp pose (xyz, wxyz quat).
            place7: (7,) world-frame place pose (xyz, wxyz quat).

        Returns:
            (2, 8) world-frame tensor with gripper bit 1 (grasp row) and 0 (place row).
        """
        g = torch.zeros(8, dtype=torch.float32)
        g[:7] = grasp7
        g[7] = ActionSampler._GRIPPER_CLOSED
        p = torch.zeros(8, dtype=torch.float32)
        p[:7] = place7
        p[7] = ActionSampler._GRIPPER_OPEN
        return torch.stack([g, p], dim=0)

    @staticmethod
    def _make_action_17(grasp7, place7, object_index):
        """Build the (17,) action vector stored on tree nodes.

        Args:
            grasp7: (7,) grasp pose (xyz, wxyz quat).
            place7: (7,) place pose (xyz, wxyz quat).
            object_index: Index of the grasped object.

        Returns:
            (17,) tensor: grasp pose, cost 0, object index, place pose, open-gripper bit.
        """
        out = torch.zeros(17, dtype=torch.float32)
        out[:7] = grasp7
        out[7] = 0.0                                         # cost (unused)
        out[8] = float(object_index)                          # grasp obj_id
        out[9:16] = place7
        out[16] = ActionSampler._GRIPPER_OPEN
        return out

    def sample_grasp_place_actions(
        self, parent_state, num_grasps_per_object=2, num_places_per_grasp=2,
        total_budget=None, return_candidates_only=False,
    ):
        """Sample (grasp, place) candidates from the cascaded policies and execute them in sim.

        Every policy prediction becomes a result, including IK, pick, and place failures.

        Args:
            parent_state: (state_dim,) sim state of the node being expanded.
            num_grasps_per_object: Grasps K predicted per object not yet on the shelf.
            num_places_per_grasp: Places predicted per grasp when total_budget is None.
            total_budget: Optional total candidate count, split as K grasps times M places per object.
            return_candidates_only: Return the raw policy candidates, skipping IK and execution.

        Returns:
            (results, parent_pcd_xyz_robot): result dicts (state, action (17,), action_2x8,
            trajectory, reward, is_failed, failure_type) and the (4096, 3) robot-frame scene cloud.
        """
        env = self.env
        num_envs = env.num_envs

        if self.grasp_only_model is None or self.place_conditional_model is None:
            raise RuntimeError(
                "Cascaded grasp+place policy not loaded. Call:\n"
                "  load_policy('grasp_only', <grasp_ckpt>, <device>) and\n"
                "  load_policy('place_conditional', <place_ckpt>, <device>)."
            )

        # Stage timing for visibility — Phase 3 expansions are minutes long
        # and the user wants to see where the wall clock is going. `tic`
        # captures the wall time at the start of the current stage; `lap`
        # prints "elapsed since tic" + label and resets tic.
        _t_total = time.perf_counter()
        _tic = [_t_total]
        def _lap(label):
            """Reset the stage timer, printing the elapsed time when EXPANSION_VERBOSE=1.

            Args:
                label: Stage name to print.
            """
            # Stage timer; silenced by default to keep planner output clean.
            # Set EXPANSION_VERBOSE=1 to re-enable.
            now = time.perf_counter()
            if os.environ.get("EXPANSION_VERBOSE") == "1":
                print(f"    [stage] {label}: {now - _tic[0]:.2f}s")
            _tic[0] = now

        # ---- 1. set state, build per-object target-only 3-channel PCDs ----
        env.set_state(parent_state.unsqueeze(0).repeat(num_envs, 1))
        if env.render_mode is not None:
            env.render()

        # Lazy import keeps open3d/trimesh off the module-import path.
        # (These helpers used to live in the deleted root data-saving
        # scripts; extracted into disco.utils.pcd_sampling after cleanup
        # 17545c9 removed the originals.)
        from disco.utils.pcd_sampling import (
            sample_scene_with_target_mask, downsample_pcd_with_mask,
            sample_scene_with_obj_ids,
            downsample_pcd_xyz, shelved_object_indices, filter_pcd_target_only,
        )

        scene_xyz_with_ids, scene_obj_ids = sample_scene_with_obj_ids(
            env, env_idx=0, extra_ignore=("table",),
        )
        shelved_idx_set = shelved_object_indices(env, env_idx=0)
        per_object = []   # list of (object_index, name, pcd_xyz)
        for object_index, name in enumerate(env.object_names):
            if env.check_object_inside_shelf(name, env_idx=0):
                continue
            filtered_xyz = filter_pcd_target_only(
                scene_xyz_with_ids, scene_obj_ids,
                object_index, shelved_idx_set,
            )
            if filtered_xyz.shape[0] == 0:
                continue
            pcd = downsample_pcd_xyz(filtered_xyz, num_points=self._PCD_NUM_POINTS)
            per_object.append((object_index, name, pcd))

        if not per_object:
            empty_pcd = np.zeros((self._PCD_NUM_POINTS, 3), dtype=np.float32)
            print(f"    all objects shelved; nothing to expand "
                  f"({time.perf_counter() - _t_total:.2f}s)")
            return [], empty_pcd

        # Cache one xyz-only full-scene PCD (robot frame) for the value net.
        scene_xyz_full, _ = sample_scene_with_target_mask(env, 0, env_idx=0)
        parent_pcd_xyz = downsample_pcd_with_mask(
            scene_xyz_full,
            np.zeros(scene_xyz_full.shape[0], dtype=np.float32),
            num_points=self._PCD_NUM_POINTS,
        )[:, :3].copy()
        parent_pcd_xyz_robot = parent_pcd_xyz.copy()
        parent_pcd_xyz_robot[:, 0] += self._ROBOT_BASE_X_OFFSET

        _lap(f"PCD build  (n_objects={len(per_object)})")

        # ---- Budget allocation (option A) ---------------------------------
        # K is fixed at `num_grasps_per_object` for every kept object.
        # M_oi (placements per grasp for object oi) either equals
        # `num_places_per_grasp` (no budget) or is derived from `total_budget`.
        # `per_object_P` maps object_index -> M_oi for objects that survive
        # the allocation; objects with M_oi = 0 are dropped from per_object.
        K = num_grasps_per_object
        if total_budget is not None and len(per_object) > 0:
            N_eff = len(per_object)
            if K > total_budget:
                print(f"    budget {total_budget} < K {K}; skipping expansion "
                      f"({time.perf_counter() - _t_total:.2f}s)")
                return [], parent_pcd_xyz_robot
            if K * N_eff > total_budget:
                n_keep = total_budget // K
                keep_idx = sorted(np.random.choice(N_eff, size=n_keep, replace=False).tolist())
                per_object = [per_object[i] for i in keep_idx]
                per_object_P = {oi: 1 for (oi, _, _) in per_object}
            else:
                base_M = total_budget // (K * N_eff)
                rem = total_budget - base_M * K * N_eff
                n_extra = rem // K
                per_object_P_list = [base_M] * N_eff
                if n_extra > 0:
                    extras = np.random.choice(N_eff, size=n_extra, replace=False).tolist()
                    for idx in extras:
                        per_object_P_list[idx] += 1
                per_object_P = {
                    per_object[i][0]: per_object_P_list[i] for i in range(N_eff)
                }
            if os.environ.get("EXPANSION_VERBOSE") == "1":
                print(
                    f"    [budget] B={total_budget} K={K} N={N_eff} -> "
                    f"per-object M={[per_object_P[oi] for (oi, _, _) in per_object]} "
                    f"(total={K * sum(per_object_P.values())})"
                )
        else:
            per_object_P = {oi: num_places_per_grasp for (oi, _, _) in per_object}

        device = next(self.grasp_only_model.parameters()).device

        # ---- 2. Predict K = num_grasps_per_object grasps per object ----
        # `pcd_t_robot_by_oi` caches each object's PCD in robot frame for
        # reuse by the place model in step 4.
        flat = []
        pcd_t_robot_by_oi = {}
        with torch.no_grad():
            for object_index, name, pcd in per_object:
                pcd_t = torch.from_numpy(pcd).to(device=device, dtype=torch.float32)
                pcd_t[..., 0] += self._ROBOT_BASE_X_OFFSET
                pcd_t_robot_by_oi[object_index] = pcd_t
                pcd_batch = pcd_t.unsqueeze(0).expand(num_grasps_per_object, -1, -1).contiguous()

                chunk = max(1, int(os.environ.get("POLICY_CHUNK_SIZE", "1")))
                _dtype = _policy_autocast_dtype()
                grasp_traj_chunks = []
                for s in range(0, pcd_batch.shape[0], chunk):
                    e = min(s + chunk, pcd_batch.shape[0])
                    if _dtype is None:
                        out = self.grasp_only_model(
                            gt_trajectory=None, pcd=pcd_batch[s:e],
                            proprioception=None, run_inference=True,
                        )                                      # (chunk, 1, 8)
                    else:
                        with torch.autocast(device_type="cuda", dtype=_dtype):
                            out = self.grasp_only_model(
                                gt_trajectory=None, pcd=pcd_batch[s:e],
                                proprioception=None, run_inference=True,
                            )
                        out = out.float()
                    grasp_traj_chunks.append(out.detach())
                    del out
                grasp_traj = torch.cat(grasp_traj_chunks, dim=0).cpu()  # (K, 1, 8)
                grasp_traj[..., 0] -= self._ROBOT_BASE_X_OFFSET   # robot → world
                for i in range(grasp_traj.shape[0]):
                    flat.append({
                        "oi": object_index, "oname": name,
                        "grasp7": grasp_traj[i, 0, :7].clone(),   # WORLD frame
                    })

        _lap(f"grasp_only forward  (K={len(flat)})")

        # ---- 3. Use the predicted grasps directly (no ACRONYM snapping) ----
        for c in flat:
            q = c["grasp7"][3:7]
            c["grasp7"][3:7] = q / q.norm().clamp_min(1e-8)

        # ---- 4. Predict M_oi = per-object placements per grasp ----
        # Pure-inference contract: every (grasp, place) policy output
        # becomes a tree node, regardless of downstream IK / pick / place
        # outcomes. So predict places for **all** grasps (not just IK-
        # feasible ones), and store them as `c["place7_list"]` of length
        # M_oi for that object. Grouped by object so we batch one PCD
        # per object.
        flat_by_oi = {}
        for c in flat:
            flat_by_oi.setdefault(c["oi"], []).append(c)

        with torch.no_grad():
            for oi, group in flat_by_oi.items():
                pcd_t_robot = pcd_t_robot_by_oi[oi]
                K_oi = len(group)
                P_oi = per_object_P[oi]
                # Repeat-interleave grasps so the batch is
                # [g0, g0, ..., (P_oi times), g1, g1, ...] — each grasp
                # gets P_oi stochastic place samples.
                pcd_batch = pcd_t_robot.unsqueeze(0).expand(K_oi * P_oi, -1, -1).contiguous()
                grasp_cond_world = torch.stack(
                    [c["grasp7"] for c in group], dim=0,
                ).repeat_interleave(P_oi, dim=0).to(device=device, dtype=torch.float32)
                grasp_cond_robot = grasp_cond_world.clone()
                grasp_cond_robot[..., 0] += self._ROBOT_BASE_X_OFFSET

                chunk = max(1, int(os.environ.get("POLICY_CHUNK_SIZE", "1")))
                _dtype = _policy_autocast_dtype()
                place_traj_chunks = []
                for s in range(0, pcd_batch.shape[0], chunk):
                    e = min(s + chunk, pcd_batch.shape[0])
                    if _dtype is None:
                        out = self.place_conditional_model(
                            gt_trajectory=None, pcd=pcd_batch[s:e],
                            proprioception=None, run_inference=True,
                            grasp_cond=grasp_cond_robot[s:e],
                        )                                      # (chunk, 1, 8)
                    else:
                        with torch.autocast(device_type="cuda", dtype=_dtype):
                            out = self.place_conditional_model(
                                gt_trajectory=None, pcd=pcd_batch[s:e],
                                proprioception=None, run_inference=True,
                                grasp_cond=grasp_cond_robot[s:e],
                            )
                        out = out.float()
                    place_traj_chunks.append(out.detach())
                    del out
                place_traj = torch.cat(place_traj_chunks, dim=0).cpu()  # (K_oi*P_oi, 1, 8)
                place_traj[..., 0] -= self._ROBOT_BASE_X_OFFSET   # robot → world
                for j, c in enumerate(group):
                    c["place7_list"] = [
                        place_traj[j * P_oi + p, 0, :7].clone()
                        for p in range(P_oi)
                    ]

        _lap(f"place_conditional forward  (K_grasps={len(flat)} × P_oi varies)")

        # Early exit for the env-scaling benchmark (scripts/perf/): return
        # the raw policy candidates after stage 4, before any cuRobo /
        # execution work. The benchmark caches these once and replays
        # stages 5-7 itself at varying num_envs, so policy inference time
        # never contaminates the sim+cuRobo timing.
        if return_candidates_only:
            candidates = [
                {
                    "oi": c["oi"],
                    "oname": c["oname"],
                    "grasp7": c["grasp7"].clone(),
                    "place7_list": [p.clone() for p in c["place7_list"]],
                }
                for c in flat
            ]
            env.set_state(parent_state.unsqueeze(0).repeat(num_envs, 1))
            return candidates, parent_pcd_xyz_robot

        # ---- 5. cuRobo IK pre-filter on the predicted grasps ----
        # IK is per-grasp (the place doesn't affect feasibility of the
        # grasp pose itself). For IK-failed grasps, emit P "ik_infeasible"
        # results (one per place variant) so the per-object candidate
        # count stays at num_grasps_per_object × num_places_per_grasp.
        results = []
        grasps_world = torch.stack([c["grasp7"] for c in flat], dim=0).to("cuda:0")
        grasps_robot = grasps_world.clone()
        grasps_robot[..., 0] += self._ROBOT_BASE_X_OFFSET
        env.action_primitives.update_collision_world()
        _, ik_success = env.action_primitives.inverse_kinematics(grasps_robot)
        feasible_mask = ik_success[:, 0].cpu().bool()

        for i, c in enumerate(flat):
            if not bool(feasible_mask[i]):
                for place7 in c["place7_list"]:
                    results.append({
                        "state": parent_state.clone(),
                        "action": self._make_action_17(c["grasp7"], place7, c["oi"]),
                        "action_2x8": self._make_action_2x8(c["grasp7"], place7),
                        "trajectory": None,
                        "reward": 0.0,
                        "is_failed": True,
                        "failure_type": "ik_infeasible",
                    })

        feasible_flat = [c for i, c in enumerate(flat) if bool(feasible_mask[i])]

        _lap(f"IK pre-filter  ({int(feasible_mask.sum())}/{len(flat)} feasible)")

        if not feasible_flat:
            print(f"    no IK-feasible grasps "
                  f"({time.perf_counter() - _t_total:.2f}s)")
            return results, parent_pcd_xyz_robot

        # ---- 6. parallel Pick (reuse existing execute_all_grasps) ----
        # Pick runs once per **unique** grasp; the P place variants share
        # the same pick. Pick failures emit P "pick_failed" results per
        # failed grasp.
        grasp_params = torch.zeros((len(feasible_flat), 9), device="cuda:0", dtype=torch.float32)
        for j, c in enumerate(feasible_flat):
            grasp_params[j, :7] = c["grasp7"].to("cuda:0")
            grasp_params[j, 7] = 0.0
            grasp_params[j, 8] = float(c["oi"])

        grasp_results, failed_grasp_indices, exec_failures = self.execute_all_grasps(
            grasp_params, parent_state, num_envs,
        )

        # Pick failures → recorded. execute_all_grasps gives detailed failure
        # info only for "grasp_execution"; for motion-plan failures we know
        # the index but not the type, so default to "pick_failed".
        for cand_idx in failed_grasp_indices:
            c = feasible_flat[cand_idx]
            ftype, state, traj = "pick_failed", parent_state.clone(), None
            for ef in exec_failures:
                if torch.equal(ef["action"][:9].cpu(), grasp_params[cand_idx].cpu()):
                    ftype = ef["failure_type"]
                    state = ef["state"].clone()
                    traj = ef["trajectory"]
                    break
            for place7 in c["place7_list"]:
                results.append({
                    "state": state,
                    "action": self._make_action_17(c["grasp7"], place7, c["oi"]),
                    "action_2x8": self._make_action_2x8(c["grasp7"], place7),
                    "trajectory": traj,
                    "reward": 0.0,
                    "is_failed": True,
                    "failure_type": ftype,
                })

        _lap(f"pick exec  ({len(grasp_results)}/{len(feasible_flat)} succeeded, {len(failed_grasp_indices)} failed)")

        if not grasp_results:
            print(f"    no pick survivors "
                  f"({time.perf_counter() - _t_total:.2f}s)")
            return results, parent_pcd_xyz_robot

        # ---- 7. parallel Place per object (per-env post-grasp states) ----
        # For each successful grasp, run P place execs in parallel sharing
        # the same post-grasp state. The (grasp, place) pair is the unit
        # of result emission.
        by_object = {}
        for cand_idx, info in grasp_results.items():
            c = feasible_flat[cand_idx]
            by_object.setdefault(c["oi"], []).append((cand_idx, info, c))

        for oi in sorted(by_object.keys()):
            group = by_object[oi]
            G = len(group)
            P_oi = per_object_P[oi]
            place_params_t = torch.zeros((G * P_oi, 8), device="cuda:0", dtype=torch.float32)
            # Repeat-interleave post-grasp states / pick trajectories so
            # entry j*P_oi + p uses grasp j's state with place variant p.
            post_grasp_states_t = torch.stack(
                [info["state"] for _, info, _ in group], dim=0,
            ).repeat_interleave(P_oi, dim=0)
            pick_trajectories = [
                info["trajectory"] for _, info, _ in group for _p in range(P_oi)
            ]
            entry_grasps = [c for _, _, c in group for _p in range(P_oi)]
            entry_place7s = [
                place7
                for _, _, c in group
                for place7 in c["place7_list"]
            ]
            for idx in range(G * P_oi):
                place_params_t[idx, :7] = entry_place7s[idx].to("cuda:0")
                place_params_t[idx, 7] = float(oi)

            place_outcomes = self._run_placement_batch_per_env_state(
                place_params=place_params_t,
                post_grasp_states=post_grasp_states_t,
                base_state=parent_state,
                num_envs=num_envs,
            )

            for outcome, c, place7, pick_traj in zip(
                place_outcomes, entry_grasps, entry_place7s, pick_trajectories,
            ):
                # Combine pick+place trajectories the same way _run_placement_batch does.
                if outcome["is_failed"]:
                    combined_traj = pick_traj
                else:
                    if outcome.get("trajectory") is not None and pick_traj is not None:
                        combined_traj = np.concatenate(
                            [pick_traj, outcome["trajectory"]], axis=0,
                        )
                    else:
                        combined_traj = pick_traj

                results.append({
                    "state": outcome["state"],
                    "action": self._make_action_17(c["grasp7"], place7, c["oi"]),
                    "action_2x8": self._make_action_2x8(c["grasp7"], place7),
                    "trajectory": combined_traj,
                    "reward": outcome["reward"],
                    "is_failed": outcome["is_failed"],
                    "failure_type": outcome["failure_type"],
                })

        _lap(
            f"place exec  (G={sum(len(v) for v in by_object.values())}, "
            f"per-object P_oi varies)"
        )

        # Reset to parent state so subsequent tree code sees a clean slate.
        env.set_state(parent_state.unsqueeze(0).repeat(num_envs, 1))
        if env.render_mode is not None:
            env.render()

        print(f"    expansion took {time.perf_counter() - _t_total:.2f}s")

        return results, parent_pcd_xyz_robot

    def _run_placement_batch_per_env_state(
        self, place_params, post_grasp_states, base_state, num_envs,
    ):
        """Run Place on candidates in batches, each slot starting from its own post-grasp state.

        Args:
            place_params: (N, 8) world-frame placements (xyz, wxyz quat, object index) on cuda:0.
            post_grasp_states: (N, state_dim) post-grasp sim state per candidate.
            base_state: (state_dim,) state used to fill unused slots.
            num_envs: Number of parallel sim slots.

        Returns:
            N dicts in input order with is_failed, state, reward, failure_type, trajectory.
        """
        env = self.env
        Place = env.action_primitives.Place
        N = place_params.shape[0]
        if N == 0:
            return []

        results = [None] * N
        state_dim = post_grasp_states.shape[-1]

        for batch_start in range(0, N, num_envs):
            batch_end = min(batch_start + num_envs, N)
            B = batch_end - batch_start

            # envs 0..B-1 = per-candidate post-grasp states; pad rest with base_state.
            state_batch = torch.zeros(
                (num_envs, state_dim),
                device=base_state.device, dtype=base_state.dtype,
            )
            state_batch[:B] = post_grasp_states[batch_start:batch_end]
            if num_envs > B:
                state_batch[B:] = base_state.unsqueeze(0).repeat(num_envs - B, 1)
            env.set_state(state_batch)
            if env.render_mode is not None:
                env.render()

            place_batch = place_params[batch_start:batch_end]
            place_states, place_trajs, _, place_rewards, place_failures = Place(place_batch)

            failed_mask = [False] * B
            for f_action, f_ftype, f_state, f_traj in place_failures:
                for j in range(B):
                    if not failed_mask[j] and torch.equal(f_action.cpu(), place_batch[j].cpu()):
                        failed_mask[j] = True
                        results[batch_start + j] = {
                            "is_failed": True,
                            "state": f_state.clone(),
                            "reward": 0.0,
                            "failure_type": f_ftype,
                            "trajectory": f_traj,
                        }
                        break

            success_in_batch = [j for j in range(B) if not failed_mask[j]]
            for k, j in enumerate(success_in_batch):
                reward = float(place_rewards[k]) if k < len(place_rewards) else 0.0
                results[batch_start + j] = {
                    "is_failed": False,
                    "state": place_states[k].clone(),
                    "reward": reward,
                    "failure_type": None,
                    "trajectory": (
                        place_trajs[k].copy()
                        if k < len(place_trajs) and place_trajs[k] is not None
                        else None
                    ),
                }

        return results

    def sample_placement_poses_debug(self, object_index: int, batch_size: int = 1):
        """Return four hard-coded placements in the `sample_placement_poses` format, for debugging.

        Args:
            object_index: Index of the object; picks the purple-cup set or a default set.
            batch_size: Unused.
        """

        # TODO: Environment should already be reset to the desired state before calling this function.
        # TODO: So environment 0 should be the one from which everything is loaded.

        object_name = self.env.object_names[object_index]

        if object_name == '50651245d23db6bfd4c7c9add3ce42f3': # purple cup
            # (N, 8): (x, y, z, qw, qx, qy, qz, object_index) — same as sample_placement_poses return
            pose_7 = torch.tensor(
                [
                    [-0.2079, 0.1396, 0.4471, -0.4429, 0.5693, 0.4267, 0.5455],
                    [-0.2485, 0.1400, 0.5210, -0.3346, 0.6527, 0.4521, 0.5076],
                    [-0.1761, 0.1265, 0.5253, -0.4230, 0.5533, 0.6006, 0.3927],
                    [-0.2039, 0.1529, 0.5456, -0.3065, 0.6953, 0.5249, 0.3836],
                ],
                device="cuda:0",
                dtype=torch.float32,
            )
            object_indices = torch.full(
                (pose_7.shape[0], 1),
                float(object_index),
                device=pose_7.device,
                dtype=pose_7.dtype,
            )
            poses = torch.cat([pose_7, object_indices], dim=-1)
        else:
            # (N, 8): (x, y, z, qw, qx, qy, qz, object_index) — same as sample_placement_poses return
            pose_7 = torch.tensor(
                [
                    [-0.1732, 0.1340, 0.4323, -0.5088, 0.4794, 0.5423, 0.4659],
                    [-0.1876, 0.1525, 0.4667, -0.4122, 0.5463, 0.6079, 0.4027],
                    [-0.2049, 0.1731, 0.4401, -0.3550, 0.6249, 0.3863, 0.5781],
                    [-0.2113, 0.1734, 0.4187, 0.5142, -0.5029, -0.4942, -0.4883],
                ],
                device="cuda:0",
                dtype=torch.float32,
            )
            object_indices = torch.full(
                (pose_7.shape[0], 1),
                float(object_index),
                device=pose_7.device,
                dtype=pose_7.dtype,
            )
            poses = torch.cat([pose_7, object_indices], dim=-1)

        # (N, 8) tensor, with (x, y, z, qw, qx, qy, qz, object_index)
        return poses
    
    # ------------------------ PUSH SAMPLING ------------------------ #

    def sample_normals_from_actor(self, object_actor, env_idx: int = 0, n: int = 10000):
        """Sample surface points whose normals are near-horizontal and point outward.

        Args:
            object_actor: Actor whose collision mesh is sampled.
            env_idx: Sim slot whose mesh is used.
            n: Target number of points (stops after 10 rounds of 10000 samples).

        Returns:
            (M, 3) points and (M, 3) unit normals; M may differ from n.
        """
        
        object_mesh = object_actor.get_collision_meshes()[env_idx]
        
        # Compute object center for filtering interior-facing normals
        object_center = object_mesh.vertices.mean(axis=0)
        
        # Maximum vertical component for 30 degrees from horizontal: sin(30°) ≈ 0.5
        max_vertical_component = np.sin(np.deg2rad(30))
        
        # Sample points and filter - keep sampling until we have enough valid points
        valid_points = []
        valid_normals = []
        max_attempts = 10
        samples_per_attempt = 10000
        
        for attempt in range(max_attempts):
            # Sample points and face indices from mesh surface
            points, face_indices = trimesh.sample.sample_surface(object_mesh, samples_per_attempt)
            
            # Get normals for sampled points (from face normals)
            normals = object_mesh.face_normals[face_indices]
            
            # Normalize normals
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            
            # Filter 1: Normals within 30 degrees of horizontal (|normal_z| < sin(30°))
            horizontal_mask = np.abs(normals[:, 2]) < max_vertical_component
                        
            # Filter 2: Normals pointing outward (away from object center)
            # Vector from object center to surface point
            to_surface = points - object_center
            to_surface = to_surface / np.linalg.norm(to_surface, axis=1, keepdims=True)
            # Normal should point in roughly the same direction as to_surface (dot product > 0)
            outward_mask = np.sum(normals * to_surface, axis=1) > 0
            
            # Combine filters
            valid_mask = horizontal_mask & outward_mask
            
            if np.any(valid_mask):
                valid_points.append(points[valid_mask])
                valid_normals.append(normals[valid_mask])
            
            # Check if we have enough points
            total_valid = sum(len(p) for p in valid_points)
            if total_valid >= n:
                break
        
        # Concatenate all valid points
        if len(valid_points) == 0:
            print(f"Warning: No valid push points found")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        
        points = np.concatenate(valid_points, axis=0)
        normals = np.concatenate(valid_normals, axis=0)
        
        return points, normals
    
    def sample_push_actions(
        self,
        object_name: str,
        n: int = 1,
        min_push_distance: float = 0.08,
        max_push_distance: float = 0.15,
        max_offset_angle: float = np.pi / 12,  # 15 degrees max offset
        env_idx: int = 0
        ):
        """Sample IK-feasible push actions against an object's side surfaces.

        Args:
            object_name: Name of the object to push.
            n: Number of pushes to sample (half along gripper Y, half along gripper Z).
            min_push_distance: Minimum push length (m).
            max_push_distance: Maximum push length (m).
            max_offset_angle: Max random tilt of the push away from the surface normal (rad).
            env_idx: Sim slot whose object mesh is used.

        Returns:
            (M, 7) world-frame start poses, (M,) gripper push axes (1=Y, 2=Z), and (M,) signed push distances (m).
        """
        # Get object mesh (in world frame for current pose)
        object_actor = self.env.objects[object_name]
        # Sample points and normals from the object actor
        points, normals = self.sample_normals_from_actor(object_actor, env_idx=env_idx, n=n)
        
        if len(points) < n:
            print(f"Warning: Only {len(points)} valid push points found, requested {n}")
            # Re-writing n to the actual number of points sampled.
            n = len(points)

        if n == 0:
            # Return empty tensors
            return (
                torch.zeros((0, 7), device="cuda:0", dtype=torch.float32),
                torch.zeros((0,), device="cuda:0", dtype=torch.long),
                torch.zeros((0,), device="cuda:0", dtype=torch.float32)
            )
        
        # Randomly sample n points from valid points
        # Sample n/2 points, then create both Y and Z push for each point to get n total
        n_points = n // 2
        indices = np.random.choice(len(points), size=n_points, replace=False)
        sampled_points = points[indices]
        sampled_normals = normals[indices]
        
        # Compute push start positions: move outward along surface normal
        push_start_positions_half = sampled_points #+ standoff_distance * sampled_normals
        
        # The base push direction is INTO the object = -normal (in world frame)
        base_push_dir = -sampled_normals  # (n_points, 3)
        
        # Apply random angular offset to push direction
        push_world_dir_half = np.zeros_like(base_push_dir)
        for i in range(n_points):
            # Sample random offset angle (0 to max_offset_angle)
            offset_angle = np.random.uniform(0, max_offset_angle)
            
            # Sample random rotation axis perpendicular to base push direction
            # Find any perpendicular vector
            if np.abs(base_push_dir[i, 0]) < 0.9:
                perp = np.cross(base_push_dir[i], np.array([1, 0, 0]))
            else:
                perp = np.cross(base_push_dir[i], np.array([0, 1, 0]))
            perp = perp / np.linalg.norm(perp)
            
            # Random angle around the push direction to determine offset direction
            azimuth = np.random.uniform(0, 2 * np.pi)
            rotation_axis = R.from_rotvec(azimuth * base_push_dir[i]).apply(perp)
            
            # Apply the offset rotation
            offset_rotation = R.from_rotvec(offset_angle * rotation_axis)
            push_world_dir_half[i] = offset_rotation.apply(base_push_dir[i])
        
        # Re-filter to ensure offset directions are still within 30 degrees of horizontal
        max_vertical_component = np.sin(np.deg2rad(30))
        valid_offset_mask = np.abs(push_world_dir_half[:, 2]) < max_vertical_component
        if not np.all(valid_offset_mask):
            # For invalid ones, fall back to base direction
            push_world_dir_half[~valid_offset_mask] = base_push_dir[~valid_offset_mask]
        
        # Normalize push directions
        push_world_dir_half = push_world_dir_half / np.linalg.norm(push_world_dir_half, axis=1, keepdims=True)
        
        # Duplicate: each point gets both a Y push and a Z push
        # First n_points are Y pushes, next n_points are Z pushes
        push_start_positions = np.concatenate([push_start_positions_half, push_start_positions_half], axis=0)
        push_world_dir = np.concatenate([push_world_dir_half, push_world_dir_half], axis=0)
        push_axes = np.concatenate([np.ones(n_points, dtype=np.int64), 2 * np.ones(n_points, dtype=np.int64)])
        push_signs = np.ones(n_points * 2, dtype=np.int64)  # Will be updated for Y-axis pushes
        
        # Update n to actual count
        n = n_points * 2
        
        # Build gripper orientations based on push axis
        # The chosen axis should align with push_world_dir
        quaternions = np.zeros((n, 4))
        world_up = np.array([0, 0, 1])
        world_x = np.array([1, 0, 0])
        world_y = np.array([0, 1, 0])
        
        for i in range(n):
            push_dir = push_world_dir[i]
            axis = push_axes[i]
            
            if axis == 1:  # Push along gripper Y (+Y or -Y)
                # If push_dir is closer to world +Y, push along gripper -Y (gripper_y = -push_dir)
                # If push_dir is closer to world -Y, push along gripper +Y (gripper_y = push_dir)
                if np.dot(push_dir, world_y) > 0:
                    gripper_y = -push_dir  # Push along gripper -Y
                    push_signs[i] = -1
                else:
                    gripper_y = push_dir   # Push along gripper +Y
                    push_signs[i] = 1
                # Choose gripper Z perpendicular, prefer world up, 
                # NOTE: This makes sure that the z axis is always pointing forward w.r.t world frame
                gripper_z = np.cross(world_up, gripper_y)
                if np.linalg.norm(gripper_z) < 0.05:
                    # NOTE: This makes sure that if the push is too vertical, then the z-axis points dowan w.r.t world
                    gripper_z = np.cross(world_x, gripper_y)
                # Both the above gripper_z assignments makes sure that the manipulator is in a realistic position
                gripper_z = gripper_z / np.linalg.norm(gripper_z)
                gripper_x = np.cross(gripper_y, gripper_z)
                gripper_x = gripper_x / np.linalg.norm(gripper_x)
                
            else:  # axis == 2, Push along gripper Z (+Z)
                gripper_z = push_dir
                # Choose gripper Y perpendicular based on push direction relative to world Y
                if np.dot(push_dir, world_y) > 0:
                    # Push closer to +Y: use gripper_z cross world_up
                    gripper_y = np.cross(gripper_z, world_up)
                else:
                    # Push closer to -Y: use world_up cross gripper_z
                    gripper_y = np.cross(world_up, gripper_z)
                if np.linalg.norm(gripper_y) < 0.1:
                    gripper_y = np.cross(world_x, gripper_z)
                gripper_y = gripper_y / np.linalg.norm(gripper_y)
                # Compute gripper_x to complete right-handed coordinate system: x = y × z
                gripper_x = np.cross(gripper_y, gripper_z)
                gripper_x = gripper_x / np.linalg.norm(gripper_x)
            
            # Build rotation matrix
            rot_matrix = np.column_stack([gripper_x, gripper_y, gripper_z])
            r = R.from_matrix(rot_matrix)
            q = r.as_quat()  # xyzw
            quaternions[i] = [q[3], q[0], q[1], q[2]]  # convert to wxyz
        
        # Build push_start poses (n, 7): [x, y, z, qw, qx, qy, qz]
        push_starts_np = np.concatenate([push_start_positions, quaternions], axis=1)
        
        # Move back along the push axis to account for gripper tip offset
        # Apply offset along the axis we're pushing with
        for i in range(n):
            axis = push_axes[i]
            if axis == 1:   # Y pushes
                # TODO: Hardcoding these parameters
                push_starts_np[i:i+1] = move_pose_along_local_z(push_starts_np[i:i+1], -0.03, format='wxyz')
                push_starts_np[i:i+1] = move_pose_along_local_y(push_starts_np[i:i+1], -push_signs[i] * 0.13, format='wxyz')
            else:  # axis == 2, Z pushes
                push_starts_np[i:i+1] = move_pose_along_local_z(push_starts_np[i:i+1], -0.14, format='wxyz')
        
        # Convert to torch tensors
        push_starts = torch.tensor(push_starts_np, device="cuda:0", dtype=torch.float32)
        # push_directions is (n,) with axis_index
        push_directions = torch.tensor(
            push_axes, 
            device="cuda:0", dtype=torch.long
        )
        
        # Sample push distances and multiply by push_signs
        push_distances = torch.tensor(
            np.random.uniform(min_push_distance, max_push_distance, size=n) * push_signs,
            device="cuda:0", dtype=torch.float32
        )

        # For debug visualization:
        push_ends = push_starts.clone()
        for i in range(n):
            push_start_np = push_starts[i].cpu().numpy()
            push_distance = push_distances[i].item()
            if push_axes[i] == 1:  # Y-axis push
                push_end_np = move_pose_along_local_y(push_start_np.reshape(1, -1), push_distance, format='wxyz')
            else:  # axis == 2, Z-axis push
                push_end_np = move_pose_along_local_z(push_start_np.reshape(1, -1), push_distance, format='wxyz')
            push_ends[i] = to_torch_pose(push_end_np.squeeze())
        
        # Filter by IK feasibility
        curobo_push_starts = push_starts.clone()
        curobo_push_starts[..., 0] = curobo_push_starts[..., 0] + 0.615
        
        push_joints, ik_successes = self.env.action_primitives.inverse_kinematics(curobo_push_starts)
        valid_indices = torch.where(ik_successes[:, 0])[0]
        
        push_starts = push_starts[valid_indices]
        push_directions = push_directions[valid_indices]
        push_distances = push_distances[valid_indices]
        
        return push_starts, push_directions, push_distances

    
    #################### UNUSED FUNCTIONS ##################################