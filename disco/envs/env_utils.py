"""
Shared ManiSkill env helpers: point clouds, segmentation, object and gripper poses.
"""
import json
import os
import copy
import sapien
import torch
import numpy as np
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
from disco.motion.motionplanner import MotionPlanner

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
    load_acronym_object_and_grasps,
    point_in_cuboid
)

from disco.utils.visualization.plotting import plot_pcd, plot_pcd_with_highlighted_segment
from disco.utils.visualization.point_cloud_structures import make_gripper_visualization
from disco.utils.conversion_utils import (
    pose_to_transformation, 
    invert_transformation, 
    transform_pcd, 
    transformation_to_pose,
    move_pose_along_local_z,
    furthest_point_sample
)
from disco.flowmatch.modeling.policy.denoise_actor_3d_packing import DenoiseActor
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


def load_checkpoint_for_eval(checkpoint_path: str, model: torch.nn.Module) -> torch.nn.Module:
    """Load a checkpoint's "weight" state dict into a model, allowing missing or extra keys.

    Args:
        checkpoint_path: Path to the saved checkpoint file.
        model: Model to load the weights into.

    Returns:
        The same model with the loaded weights.
    """
    print(f"=> trying checkpoint '{checkpoint_path}'")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    model_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
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

    print(f"=> loaded successfully '{checkpoint_path}' (step {model_dict.get('iter', 0)})")
    del model_dict
    torch.cuda.empty_cache()

    return model


class ManiSkillEnvUtils:
    """
    Mixin class containing utility methods for ManiSkill environments.
    
    NOTE: This mixin requires the class to also inherit from BaseEnv.
    It accesses BaseEnv attributes like self.scene, self.device, etc.
    """
    
    def set_gripper_friction(self, friction_coefficient=2.0):
        """Set the static friction of the finger links' collision shapes.

        Args:
            friction_coefficient: Static friction coefficient to apply.
        """

        # Set high friction for the finger links
        self.finger_entities = [link._objs[0].entity for link in self.robot_links[-4:]]
        self.finger_components = [entity.find_component_by_type(PhysxRigidBodyComponent) for entity in self.finger_entities]

        for component in self.finger_components:
            if component is not None:
                for shape in component.collision_shapes:
                    shape.physical_material.set_static_friction(friction_coefficient)
    
    def stall(self, num_steps=None):
        """Keep rendering the scene without stepping the simulation.

        Args:
            num_steps: Number of render iterations; None loops forever.
        """

        print("Left click to stop")

        while True:
            if self.render_mode is not None:
                self.render()
            if num_steps is not None:
                num_steps -= 1
            if num_steps == 0:
                break
    
    def capture_video_frame(self):
        """Append the current render-camera image to self.video_frames, if a render camera exists."""
        if len(self.scene.human_render_cameras.items()) == 0:
            return
        render_img = self.scene.get_human_render_camera_images()['render_camera']
        render_img = render_img.cpu().detach().numpy()
        self.video_frames = np.concatenate([self.video_frames, render_img], axis=0)
    
    def capture_image(self):
        """Return the (H, W, 3) RGB render of env slot 0, or None if there is no render camera."""
        if len(self.scene.human_render_cameras.items()) == 0:
            return
        # render_img = self.scene.get_human_render_camera_images()['render_camera']
        render_img = self.render_rgb_array()
        render_img = render_img.cpu().detach().numpy()[0, ..., :3]
        return render_img
    
    def get_point_cloud(self, object_name=None):
        """Return env slot 0's cropped camera point cloud, without table or robot points.

        Args:
            object_name: If given, scene_mask marks only this object; otherwise all objects.

        Returns:
            (pcd, rgb, seg, scene_mask): (N, 3) xyz, (N, 3) uint8 colors, (N, 1) segment ids,
            and a bool mask of object points (computed before zero points are dropped).
        """

        pcd_obs = self.get_obs()['pointcloud']

        pcd = pcd_obs['xyzw'][0, :, :3].to(self.device)     # tensor(N, 3)
        rgb = pcd_obs['rgb'][0].to(self.device)         # In RGB uint8 format (0-255) --> tensor(N, 3)
        seg = pcd_obs['segmentation'][0].to(self.device)         # tensor(N, 1) int16

        mask = (pcd[:, 0] > self.pcd_bounds[0]) & \
               (pcd[:, 0] < self.pcd_bounds[3]) & \
               (pcd[:, 1] > self.pcd_bounds[1]) & \
               (pcd[:, 1] < self.pcd_bounds[4]) & \
               (pcd[:, 2] > self.pcd_bounds[2]) & \
               (pcd[:, 2] < self.pcd_bounds[5])

        pcd = pcd[mask]
        rgb = rgb[mask]
        seg = seg[mask]

        # 12, 14 --> right finger, left finger from from the robot's perspective
        # 16 --> Table
        # Objects start from 18
        if object_name is not None:
            # Filter to only the specified object and shelf
            object_id = self.object_ids[object_name]
            object_ids = [object_id]
        else:
            # Use all objects
            object_ids = [self.object_ids[object_name] for object_name in self.object_names]
        
        scene_ids = object_ids + [self.table_id]

        # Remove exactly half of the table points randomly (keep other half):
        table_mask = (seg == self.table_id)[..., 0]
        # true_idx = torch.where(table_mask)[0]
        # num_true = true_idx.numel()
        # # randomly select quarter of True indices to KEEP
        # perm = torch.randperm(num_true, device=table_mask.device)
        # to_keep = true_idx[perm[: num_true // 4]]

        # # build a keep mask: keep all non-table points, and only the selected half of table points
        # keep_mask = (~table_mask).clone()
        # keep_mask[to_keep] = True

        keep_mask = ~table_mask     # Keep all non-table points

        pcd = pcd[keep_mask]
        rgb = rgb[keep_mask]
        seg = seg[keep_mask]

        robot_mask = torch.zeros_like(seg, dtype=torch.bool, device=self.device)
        for id in range(16):
            robot_mask = robot_mask | (seg == id)

        # Invert robot mask, mask out robot points
        non_robot_mask = ~robot_mask[..., 0]
        pcd = pcd[non_robot_mask]
        rgb = rgb[non_robot_mask]
        seg = seg[non_robot_mask]
        
        # If object_name is specified, filter to only that object and shelf
        # if object_name is not None:
        #     object_shelf_mask = (seg == object_id) | (seg == shelf_id)
        #     pcd = pcd[object_shelf_mask.squeeze()]
        #     rgb = rgb[object_shelf_mask.squeeze()]
        #     seg = seg[object_shelf_mask.squeeze()]
        
        # Create scene mask: 1 for points belonging to objects/table, 0 otherwise (required for collision avoidance)
        scene_mask = torch.zeros_like(seg, dtype=torch.bool, device=self.device)
        for id in scene_ids:
            scene_mask = scene_mask | (seg == id)

        # Filter out zero points
        non_zero_mask = (pcd[:, 0] != 0) & (pcd[:, 1] != 0) & (pcd[:, 2] != 0)
        pcd = pcd[non_zero_mask]
        rgb = rgb[non_zero_mask]
        seg = seg[non_zero_mask]

        return pcd, rgb, seg, scene_mask
    
    def add_pointcloud_noise(self, points):
        """Add Gaussian jitter, random point dropout, and random outliers to a point cloud.

        Args:
            points: (N, 3) xyz tensor.

        Returns:
            (M, 3) noisy xyz tensor.
        """
        sigma = np.random.uniform(0.002, 0.006)
        noise = torch.randn_like(points) * sigma
        noisy_points = points + noise

        # Dropout
        drop_prob = np.random.uniform(0.01, 0.05)
        keep_mask = torch.rand(len(points)) > drop_prob
        noisy_points = noisy_points[keep_mask]

        # Outliers
        n_outliers = np.random.randint(50, 200)
        outliers = (torch.rand((n_outliers, 3), device=points.device) - 0.5) * 2.0
        noisy_points = torch.cat([noisy_points, outliers], dim=0)

        return noisy_points
    
    def get_observed_pcd(self, object_name):
        """Return a noisy 4096-point scene cloud, shifted into the robot base frame.

        Args:
            object_name: Forwarded to get_point_cloud; does not change the returned points.
        """

        scene_pcd, scene_rgb, scene_seg, scene_mask = self.get_point_cloud(object_name)
        noisy_pcd = self.add_pointcloud_noise(scene_pcd)
        noisy_pcd = furthest_point_sample(noisy_pcd, num_points=4096)
        noisy_pcd[:, 0] = noisy_pcd[:, 0] + 0.615    # TODO: This is a hack to move the point cloud to the robot base frame.

        return noisy_pcd
       
    def get_sampled_point_cloud(
        self, 
        num_points_per_actor: int = 10000,
        env_idx: int = 0,
        actors_to_ignore: list = ['table-workspace']
        ):
        """Sample surface points from each actor's collision mesh.

        Args:
            num_points_per_actor: Points sampled per actor.
            env_idx: Env slot to read the meshes from.
            actors_to_ignore: Names of actors to skip.

        Returns:
            (num_actors * num_points_per_actor, 3) numpy array.
        """

        sampled_points = []

        for actor in self.actors:
            if actor.name in actors_to_ignore:
                continue
            mesh = actor.get_collision_meshes()[env_idx]            # Initialize with env 0
            points, _ = trimesh.sample.sample_surface(mesh, num_points_per_actor)
            points = np.array(points)
            sampled_points.append(points)
            
        return np.concatenate(sampled_points, axis=0)

    def get_sampled_point_cloud2(
        self, 
        num_points: int = 4096,
        env_idx: int = 0,
        actors_to_ignore: list = ['table-workspace']
        ):
        """Sample a fixed total of surface points, split evenly across actors.

        Args:
            num_points: Total number of points.
            env_idx: Env slot to read the meshes from.
            actors_to_ignore: Names of actors to skip.

        Returns:
            (num_points, 3) numpy array.
        """

        # Count actors to sample from
        actors_to_sample = [actor for actor in self.actors if actor.name not in actors_to_ignore]
        num_actors = len(actors_to_sample)
        
        if num_actors == 0:
            return np.zeros((num_points, 3))
        
        # Divide points equally among actors
        points_per_actor = num_points // num_actors
        remainder = num_points % num_actors
        
        sampled_points = []
        for i, actor in enumerate(actors_to_sample):
            # Distribute remainder points to first 'remainder' actors
            num_points_this_actor = points_per_actor + (1 if i < remainder else 0)
            mesh = actor.get_collision_meshes()[env_idx]
            points, _ = trimesh.sample.sample_surface(mesh, num_points_this_actor)
            points = np.array(points)
            sampled_points.append(points)
            
        return np.concatenate(sampled_points, axis=0)

    def get_sampled_actor_point_cloud(self, actor, env_idx: int = 0, num_points: int = 2048):
        """Sample surface points from one actor's collision mesh.

        Args:
            actor: ManiSkill actor to sample.
            env_idx: Env slot to read the mesh from.
            num_points: Number of points.

        Returns:
            (num_points, 3) numpy array.
        """
        mesh = actor.get_collision_meshes()[env_idx]
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        points = np.array(points)
        return points

    def visualize_poses(self, poses):
        """Plot gripper markers at the given poses over a sampled scene point cloud.

        Args:
            poses: (n, 7) array or tensor of (x, y, z, qw, qx, qy, qz) poses.
        """
        pose_colors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.65, 0.0),
            (0.56, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 0.0),
        ]

        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().detach().numpy()

        num_poses = poses.shape[0]

        pose_pcds = []
        pose_rgbs = []

        for i in range(num_poses):
            pose = poses[i, :7]

            transform = pose_to_transformation(pose, format="wxyz")

            color = pose_colors[i % len(pose_colors)]

            gripper_pcd, gripper_colors = make_gripper_visualization(
                rotation=transform[:3, :3],
                translation=transform[:3, 3],
                length=0.05,
                density=50,
                color=color,
            )

            pose_pcds.append(gripper_pcd)
            pose_rgbs.append(gripper_colors)

        scene_pcd = self.get_sampled_point_cloud()
        scene_rgb = np.ones_like(scene_pcd) * 0.7

        full_pcd = np.vstack([scene_pcd] + pose_pcds)
        full_rgb = np.vstack([scene_rgb] + pose_rgbs)

        plot_pcd(full_pcd, full_rgb)

    def visualize_grasp_poses(self, grasp_poses):
        """Plot grasp poses over the scene (same as visualize_poses).

        Args:
            grasp_poses: (n, 7) array or tensor of (x, y, z, qw, qx, qy, qz) poses.
        """
        self.visualize_poses(grasp_poses)

    def get_object_pose(self, object_name=None):
        """Return an object's (num_envs, 7) pose tensor (x, y, z, qw, qx, qy, qz).

        Args:
            object_name: Object name, or the actor itself.
        """
        if isinstance(object_name, str):
            p = self.objects[object_name].pose.p
            q = self.objects[object_name].pose.q
            return torch.cat([p, q], dim=-1)
        else:
            p = object_name.pose.p
            q = object_name.pose.q
            return torch.cat([p, q], dim=-1)
    
    def set_object_pose(self, object_name, pose: torch.Tensor):
        """Teleport an object to the given pose in every env slot.

        Args:
            object_name: Object name, or the actor itself.
            pose: (7,) or (n, 7) pose; short inputs repeat the last row, long ones are truncated.
        """

        if not isinstance(pose, torch.Tensor):
            pose = torch.tensor(pose, dtype=torch.float32, device=self.device)
        
        if pose.ndim == 1:
            pose = pose.unsqueeze(0)
        
        if pose.shape[0] < self.num_envs:
            # Repeat the last element until num_envs
            last_element = pose[-1:]  # (1, 7)
            n_repeats = self.num_envs - pose.shape[0]
            repeated = last_element.repeat(n_repeats, 1)
            pose = torch.cat([pose, repeated], dim=0)
        elif pose.shape[0] > self.num_envs:
            # Truncate if longer
            pose = pose[:self.num_envs]

        pose = pose.to(self.device).to(torch.float32)

        # TODO: Should this function be parallelized, since it is only used in a single object context.
        if isinstance(object_name, str):
            self.objects[object_name].set_pose(Pose.create_from_pq(p=pose[..., :3], q=pose[..., 3:]))
        else:
            object_name.set_pose(Pose.create_from_pq(p=pose[..., :3], q=pose[..., 3:]))

        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
    
    @property
    def object_poses_tensor(self):
        """Return all object poses as a (num_envs, num_objects, 7) tensor."""
        object_poses = torch.zeros((self.num_envs, self.num_objects, 7), device=self.device, dtype=torch.float32)
        for i, object_name in enumerate(self.object_names):
            object_poses[:, i, :] = self.get_object_pose(object_name)
        return object_poses
    
    @property
    def object_centric_state(self):
        """Return each object's pose and box size as a (num_envs, num_objects, 10) tensor.

        Last dim is (x, y, z, qw, qx, qy, qz, dim_x, dim_y, dim_z); dims are 0 without self.bounds.
        """
        object_state = torch.zeros((self.num_envs, self.num_objects, 10), device=self.device, dtype=torch.float32)
        for i, object_name in enumerate(self.object_names):
            # Get pose (7 values)
            object_state[:, i, :7] = self.get_object_pose(object_name)
            
            # Get dimensions from bounds (3 values)
            if hasattr(self, 'bounds') and object_name in self.bounds:
                bounds = self.bounds[object_name]
                dim_x = float(bounds[0][1] - bounds[0][0])
                dim_y = float(bounds[1][1] - bounds[1][0])
                dim_z = float(bounds[2][1] - bounds[2][0])
                object_state[:, i, 7] = dim_x
                object_state[:, i, 8] = dim_y
                object_state[:, i, 9] = dim_z
        return object_state
    
    def set_object_poses_tensor(self, object_poses: torch.Tensor):
        """Teleport all objects to the given poses.

        Args:
            object_poses: (num_envs, num_objects, 7) tensor, or (num_objects, 7) applied to every slot.
        """
        if object_poses.ndim == 2:
            object_poses = object_poses.unsqueeze(0).repeat(self.num_envs, 1, 1)
        elif object_poses.ndim == 3 and object_poses.shape[0] != self.num_envs:
            object_poses = object_poses.repeat(self.num_envs, 1, 1)

        object_poses = object_poses.to(self.device).to(torch.float32)

        assert object_poses.shape == (self.num_envs, self.num_objects, 7)
        
        for i, object_name in enumerate(self.object_names):
            self.set_object_pose(object_name, object_poses[:, i, :])
    
    def control_robot(self, qpos: torch.Tensor):
        """Step the simulation once toward a target robot configuration.

        Args:
            qpos: (num_envs, 9) target arm joints plus two finger joints (0 to 0.04 m each).

        Returns:
            (obs, reward, done, info) from env.step.
        """
        # Scale 0 to 0.04 to -1 to 1
        gripper_state = (qpos[..., -1] - 0.02) / 0.02
        action = torch.cat([qpos[..., :-2], gripper_state.unsqueeze(-1)], dim=-1)
        # Convert to numpy for step() if needed
        obs, reward, done, info, _ = self.step(action)
        if self.render_mode is not None:
            self.render()
        return obs, reward, done, info
    
    def get_robot_qpos(self):
        """Return the (num_envs, 9) robot qpos: 7 arm joints plus 2 finger joints."""
        qpos = self.agent.robot.get_qpos()
        return qpos

    def get_joint_state(self):
        """Return the (num_envs, 7) arm joint positions."""
        qpos = self.get_robot_qpos()
        joint_state = qpos[..., :7]
        return joint_state
    
    def get_gripper_state(self):
        """Return the (num_envs, 2) finger joint positions."""
        qpos = self.get_robot_qpos()
        gripper_state = qpos[..., -2:]
        return gripper_state

    def get_gripper_pose(self):
        """Return the (num_envs, 7) world pose of the gripper hand link."""
        gripper_pose = self.agent.robot.get_links()[-5].pose.raw_pose
        return gripper_pose
    
    def set_robot_qpos(self, qpos):
        """Set the robot qpos directly.

        Args:
            qpos: (num_envs, 9) tensor of arm joints plus finger widths.
        """
        self.agent.robot.set_qpos(qpos)
    
    def set_joint_state(self, joint_state):
        """Set the arm joints in every env slot, keeping the current finger widths.

        Args:
            joint_state: (7,) or (n, 7) tensor; short inputs repeat the last row, long ones are truncated.
        """
        assert isinstance(joint_state, torch.Tensor)
        
        current_gripper_state = self.get_gripper_state()
        # Ensure shapes match for concatenation
        if joint_state.ndim == 1:
            joint_state = joint_state.unsqueeze(0)
        
        if joint_state.shape[0] < self.num_envs:
            # Repeat the last element until num_envs
            last_element = joint_state[-1:]  # (1, 7)
            n_repeats = self.num_envs - joint_state.shape[0]
            repeated = last_element.repeat(n_repeats, 1)
            joint_state = torch.cat([joint_state, repeated], dim=0)
        elif joint_state.shape[0] > self.num_envs:
            # Truncate if longer
            joint_state = joint_state[:self.num_envs]

        if current_gripper_state.ndim == 1:
            current_gripper_state = current_gripper_state.unsqueeze(0).repeat(self.num_envs, 1)
        elif current_gripper_state.ndim == 2 and current_gripper_state.shape[0] != self.num_envs:
            current_gripper_state = current_gripper_state.repeat(self.num_envs, 1)
        
        qpos = torch.cat([joint_state.to(current_gripper_state.device), current_gripper_state], dim=-1)
        self.set_robot_qpos(qpos)

        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
    
    def wait(self, steps: int = 50):
        """Step the simulation while holding the robot still.

        Args:
            steps: Number of sim steps.
        """

        current_robot_qpos = self.get_robot_qpos()

        i = 0
        while i < steps:
            _ = self.control_robot(current_robot_qpos)
            i += 1
    
    def wait_for_stability(self, max_steps: int = 200):
        """Step with the robot held still until objects stop moving.

        Args:
            max_steps: Step limit, in case objects never settle.
        """

        current_robot_qpos = self.get_robot_qpos()
        
        prev_object_poses = self.object_poses_tensor.clone()

        
        tolerance = 8e-3
        curr_error = torch.inf
        steps = 0

        while curr_error > tolerance and steps < max_steps:
            _ = self.control_robot(current_robot_qpos)

            curr_object_poses = self.object_poses_tensor.clone()

            error = torch.abs(curr_object_poses - prev_object_poses)
            curr_error = torch.max(error)

            prev_object_poses = curr_object_poses.clone()
            steps += 1
    
    def synchronize_object_poses(self, env_idx: int):
        """Copy one env slot's object poses to every slot.

        Args:
            env_idx: Env slot to copy from.
        """
        object_poses = self.object_poses_tensor
        object_poses_to_match = object_poses[env_idx]
        self.set_object_poses_tensor(object_poses_to_match)
    
    
    def wait_for_synchronized_stability(self):
        """Let objects settle, then copy the poses of the slot that moved least to every slot."""
        
        initial_object_poses = self.object_poses_tensor.clone()    
        self.wait_for_stability()
        final_object_poses = self.object_poses_tensor.clone()

        error = torch.abs(final_object_poses - initial_object_poses)
        error_per_env = error.amax(dim=(1, 2))
        env_with_min_error = torch.argmin(error_per_env)
        object_poses_to_match = final_object_poses[env_with_min_error]

        self.set_object_poses_tensor(object_poses_to_match)
    

        
    